"""Offline tests. Engine response expectations come from captured fixtures.

Only dataset API envelopes are synthetic, explicitly labelled SHAPE_ONLY.
No tests launch Docker, contact HTTP, import numpy, or touch a GPU.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cc.acc import common, compare, datasets, evaluate, gen, kld, mcq, server

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def chat():
    return fixture("server-chat-logprobs.granite.json")


class CapturedParsers(unittest.TestCase):
    def test_chat_logprobs(self):
        raw = chat()
        entries = raw["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        self.assertEqual(mcq.top_logprobs(raw), entries)
        prediction, scores = mcq.letter_argmax(entries, 4)
        self.assertEqual(prediction, raw["choices"][0]["message"]["content"])
        self.assertEqual(scores[prediction], entries[0]["logprob"])

    def test_completion_logprobs(self):
        raw = fixture("server-completion-nprobs.granite.json")
        entries = raw["completion_probabilities"][0]["top_logprobs"]
        self.assertEqual(mcq.top_logprobs(raw), entries)
        prediction, _ = mcq.letter_argmax(entries, 4)
        self.assertEqual(prediction, raw["content"].strip())
        self.assertEqual(common.response_metrics(raw)["prompt_tokens"], raw["tokens_evaluated"])

    def test_generation_fixture(self):
        raw = fixture("server-chat-gen.granite.json")
        text = raw["choices"][0]["message"]["content"]
        self.assertEqual(gen.extract_number(text), re.findall(r"\d+", text)[-1])
        metrics = common.response_metrics(raw)
        self.assertEqual(metrics["decode_tps"], raw["timings"]["predicted_per_second"])
        self.assertEqual(metrics["prompt_tokens"], raw["usage"]["prompt_tokens"])

    def test_props_fixture_non_thinking(self):
        props = fixture("server-props.granite.json")
        self.assertNotIn("enable_thinking", props["chat_template"])
        self.assertEqual(common.thinking_settings(props), {})
        self.assertIn("chat_template_caps", props)

    def test_kl_fixture_statistics(self):
        text = (FIXTURES / "perplexity-kld.granite-Q8_0.txt").read_text()
        parsed = kld.parse(text)
        labels = {"ppl": r"Mean PPL\(Q\)", "mean_kld": "Mean    KLD",
                  "kld_p99": "99.0%   KLD", "same_top_p": "Same top p", "rms_dp": "RMS Δp"}
        for key, label in labels.items():
            expected = float(re.search(label + r"\s*:\s*([\d.]+)", text).group(1))
            self.assertEqual(parsed[key], expected)
        header = re.search(r"computing over (\d+) chunks, n_ctx=(\d+)", text)
        self.assertEqual(parsed["n_chunks"], int(header[1]))
        self.assertEqual(parsed["n_ctx"], int(header[2]))
        self.assertEqual(parsed["probability_units"], "percent")

    def test_kl_incomplete_rejected(self):
        text = (FIXTURES / "perplexity-kld.granite-Q8_0.txt").read_text()
        with self.assertRaises(ValueError):
            kld.parse(text.split("====== KL divergence statistics")[0])


class LettersAndNumbers(unittest.TestCase):
    def setUp(self):
        self.entries = mcq.top_logprobs(chat())

    def test_leading_space_variants(self):
        entries = [{**entry, "token": " " + entry["token"]} for entry in self.entries]
        self.assertEqual(mcq.letter_argmax(entries, 4)[0], mcq.letter_argmax(self.entries, 4)[0])

    def test_variant_uses_max_not_sum(self):
        entries = self.entries + [{**self.entries[0], "token": " " + self.entries[0]["token"]}]
        self.assertEqual(mcq.letter_argmax(entries, 4), mcq.letter_argmax(self.entries, 4))

    def test_no_letter(self):
        entries = [e for e in self.entries if len(e["token"]) > 1]
        self.assertEqual(mcq.letter_argmax(entries, 4), (None, {}))

    def test_outside_choices_ignored(self):
        expected = max((e for e in self.entries if e["token"] == "A"), key=lambda e: e["logprob"])
        self.assertEqual(mcq.letter_argmax(self.entries, ["A"]), ("A", {"A": expected["logprob"]}))

    def test_prefix_and_newline_not_letters(self):
        entries = [{**self.entries[0], "token": token} for token in ("\nB", "B.", " B:", "BB")]
        self.assertEqual(mcq.letter_argmax(entries, 4), (None, {}))

    def test_bytes_fallback(self):
        entries = [{k: v for k, v in e.items() if k != "token"} for e in self.entries]
        self.assertEqual(mcq.letter_argmax(entries, 4), mcq.letter_argmax(self.entries, 4))

    def test_tie_breaking(self):
        score = self.entries[0]["logprob"]
        self.assertEqual(mcq.letter_argmax([dict(token=l, logprob=score) for l in ("B", "A")], 4)[0], "A")

    def test_nonfinite_ignored(self):
        self.assertEqual(mcq.letter_argmax([dict(token="A", logprob=-math.inf)], 4), (None, {}))

    def test_numeric_markers(self):
        self.assertEqual(gen.extract_number("12 steps. The answer is $1,234.50. Check 8"), "1234.50")
        self.assertEqual(gen.extract_number("work 9\n#### -2.50"), "-2.50")
        self.assertEqual(gen.extract_number("3 then .5"), "0.5")
        self.assertIsNone(gen.extract_number("The answer is unknown"))

    def test_numeric_exact_score(self):
        raw = fixture("server-chat-gen.granite.json")
        answer = gen.extract_number(raw["choices"][0]["message"]["content"])
        result = gen.score(dict(id="fixture", answer="#### " + answer + ".0"), raw)
        self.assertTrue(result["correct"])

    def test_payload_settings(self):
        props = fixture("server-props.granite.json")
        item = dict(question="fixture question", choices=["one", "two"], answer="A")
        payload = mcq.payload(item, props)
        self.assertEqual(payload["max_tokens"], 1)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_logprobs"], 20)
        self.assertEqual(payload["messages"][-1]["content"], mcq.format_prompt(item))
        props["chat_template"] += "{{ enable_thinking }}"
        self.assertEqual(mcq.payload(item, props)["chat_template_kwargs"], {"enable_thinking": False})

    def test_system_role_omitted_when_template_forbids_it(self):
        props = fixture("server-props.granite.json")
        props["chat_template_caps"]["supports_system_role"] = False
        item = dict(question="fixture question", choices=["one", "two"], answer="A")
        self.assertEqual([m["role"] for m in mcq.payload(item, props)["messages"]], ["user"])


class Statistics(unittest.TestCase):
    def test_wilson_empty(self):
        self.assertEqual(common.wilson(0, 0), (None, None))

    def test_wilson_edges(self):
        z = 1.959963984540054
        self.assertAlmostEqual(common.wilson(0, 10)[1], z*z / (10 + z*z))
        self.assertAlmostEqual(common.wilson(10, 10)[0], 10 / (10 + z*z))
        self.assertAlmostEqual(sum(common.wilson(5, 10)), 1)
        with self.assertRaises(ValueError):
            common.wilson(11, 10)

    def test_mcnemar_hand_tables(self):
        for b, c, expected in [(0, 0, 1), (1, 1, 1), (0, 3, .25), (1, 9, .021484375),
                               (9, 1, .021484375), (0, 1, 1)]:
            self.assertEqual(compare.mcnemar_exact(b, c), expected)

    def test_mcnemar_large_and_invalid(self):
        self.assertEqual(compare.mcnemar_exact(1000, 1000), 1)
        with self.assertRaises(ValueError):
            compare.mcnemar_exact(-1, 3)

    def test_pairing_uses_ids(self):
        base = {str(i): dict(correct=i == 0) for i in range(10)}
        candidate = {str(i): dict(correct=i != 0) for i in reversed(range(10))}
        result = compare.compare(base, candidate)
        self.assertEqual((result["b"], result["c"]), (1, 9))
        self.assertEqual(result["mcnemar_exact_p"], .021484375)
        self.assertAlmostEqual(result["delta_acc"], .8)
        self.assertIn("Wilson", compare.markdown(result))
        with self.assertRaises(ValueError):
            compare.compare(base, {"extra": dict(correct=True)})


class DatasetShapeOnly(unittest.TestCase):
    # SHAPE-ONLY: constructed from the documented rows API response envelope.
    def test_SHAPE_ONLY_api_envelope_and_mmlu(self):
        payload = dict(rows=[dict(row_idx=0, row=dict(question="toy?", options=["x", "y"],
                       answer="B", category="toy"), truncated_cells=[])], num_rows_total=1)
        rows, total = datasets.parse_page(payload)
        self.assertEqual(total, len(rows))
        row = datasets.normalize("mmlu-pro", *rows[0], "shape-only")
        self.assertEqual(row["answer"], "B")
        self.assertEqual(row["meta"]["category"], "toy")

    def test_SHAPE_ONLY_arc_numeric_labels(self):
        row = datasets.normalize("arc-challenge", 0, dict(question="toy?",
              choices=dict(label=["1", "2"], text=["x", "y"]), answerKey="2"), "shape-only")
        self.assertEqual(row["answer"], "B")

    def test_SHAPE_ONLY_gsm(self):
        row = datasets.normalize("gsm8k", 0, dict(question="toy?", answer="#### 1"), "shape-only")
        self.assertIsNone(row["choices"])
        self.assertEqual(row["answer"], "#### 1")

    def test_SHAPE_ONLY_gpqa_shuffle(self):
        raw = {"Question": "toy?", "Correct Answer": "right",
               **{f"Incorrect Answer {i}": f"wrong{i}" for i in range(1, 4)}}
        row = datasets.normalize("gpqa-diamond", 0, raw, "shape-only", 42)
        self.assertEqual(row, datasets.normalize("gpqa-diamond", 0, raw, "shape-only", 42))
        self.assertEqual(row["choices"][ord(row["answer"]) - 65], raw["Correct Answer"])

    def test_SHAPE_ONLY_fetch_pagination(self):
        def opener(url, timeout):
            offset = int(dict(__import__("urllib.parse", fromlist=["parse_qs"]).parse_qsl(url.split("?")[1]))["offset"])
            count = min(100, 101 - offset)
            return io.BytesIO(json.dumps(dict(rows=[dict(row_idx=i, row=dict(question="toy"))
                       for i in range(offset, offset + count)], num_rows_total=101)).encode())
        rows, pages = datasets.fetch_rows("shape-only", "default", "test", opener=opener)
        self.assertEqual(len(rows), 101)
        self.assertEqual([p["offset"] for p in pages], [0, 100])
        self.assertTrue(all(len(p["sha256"]) == 64 for p in pages))
        self.assertIn("length=100", datasets.rows_url("toy/a b", "default", "test"))
        with self.assertRaises(ValueError):
            datasets.rows_url("toy", "default", "test", length=101)

    def test_SHAPE_ONLY_stratified_subset(self):
        rows = [dict(id=str(i), meta=dict(category="small" if i < 2 else "large")) for i in range(10)]
        selected = datasets.subset(rows, 5, 42)
        self.assertEqual(selected, datasets.subset(rows, 5, 42))
        self.assertEqual(sum(r["meta"]["category"] == "small" for r in selected), 1)
        self.assertEqual(len({r["id"] for r in selected}), 5)
        with self.assertRaises(ValueError):
            datasets.subset(rows, 11, 42)


class Resume(unittest.TestCase):
    def test_skip_completed_ids_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            items = [dict(id="a"), dict(id="b")]
            self.assertEqual(common.pending_items(items, path), items)
            path.write_text(json.dumps(dict(id="a")) + "\n")
            self.assertEqual(common.pending_items(items, path), items[1:])
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                common.pending_items(items, path)

    def test_broken_line_fails_without_silent_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            path.write_text('{"id":')
            with self.assertRaisesRegex(ValueError, "repair interrupted line"):
                common.pending_items([], path)

    def test_evaluator_resume_does_not_request_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = root / "data.jsonl", root / "model.gguf"
            response = chat()
            answer = response["choices"][0]["message"]["content"]
            item = dict(id="fixture", source="captured", question="fixture?", choices=list("ABCD"), answer=answer)
            data.write_text(json.dumps(item) + "\n")
            model.write_bytes(b"test identity only")
            args = argparse.Namespace(image="fixture-image", model=str(model), outdir=str(root / "run"),
                data=str(data), env=[], engine_arg=[], dry_run=False, request_timeout=10)
            running = MagicMock()
            running.__enter__.return_value = running
            running.props = fixture("server-props.granite.json")
            running.url = "http://fixture.invalid"
            with patch.object(server, "from_args", return_value=running) as start, \
                 patch.object(common, "request_json", return_value=response) as request, redirect_stdout(io.StringIO()):
                for _ in range(2):
                    evaluate.run(args, "mcq", dict(max_tokens=1), mcq.payload, mcq.score, mcq.format_prompt)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(start.call_count, 1)
                args.engine_arg = ["changed"]
                with self.assertRaisesRegex(ValueError, "resume settings differ"):
                    evaluate.run(args, "mcq", dict(max_tokens=1), mcq.payload, mcq.score, mcq.format_prompt)
            result = json.loads((root / "run/summary.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["n"], 1)
            self.assertEqual(result["correct"], 1)

    def test_partial_failure_resume_preserves_items_without_final_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, model = root / "data.jsonl", root / "model.gguf"
            response = chat()
            answer = response["choices"][0]["message"]["content"]
            items = [dict(id=key, source="captured", question="fixture?", choices=list("ABCD"),
                          answer=answer) for key in ("first", "second")]
            data.write_text("".join(json.dumps(item) + "\n" for item in items))
            model.write_bytes(b"test identity only")
            args = argparse.Namespace(image="fixture-image", model=str(model), outdir=str(root / "run"),
                data=str(data), env=[], engine_arg=[], dry_run=False, request_timeout=10)
            running = MagicMock()
            running.__enter__.return_value = running
            running.props = fixture("server-props.granite.json")
            running.url = "http://fixture.invalid"
            with patch.object(server, "from_args", return_value=running), \
                 patch.object(common, "request_json", side_effect=[response, OSError("interrupted"), response]) as request, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    evaluate.run(args, "mcq", dict(max_tokens=1), mcq.payload, mcq.score, mcq.format_prompt)
                summary_path = root / "run/summary.json"
                self.assertFalse(json.loads(summary_path.read_text())["complete"])
                output = root / "run/items.jsonl"
                output.write_text(output.read_text().rstrip("\n"))
                evaluate.run(args, "mcq", dict(max_tokens=1), mcq.payload, mcq.score, mcq.format_prompt)
                self.assertEqual(request.call_count, 3)
                self.assertEqual([r["id"] for r in common.read_jsonl(output)], ["first", "second"])
                self.assertTrue(json.loads(summary_path.read_text())["complete"])


class DockerOffline(unittest.TestCase):
    def test_dry_run_never_reserves_socket_or_launches_process(self):
        args = argparse.Namespace(image="image", model="/tmp/model.gguf", outdir="/tmp/unused",
            engine_arg=[], env=[], health_timeout=600, dry_run=True)
        with patch.object(server, "free_port", side_effect=AssertionError("socket forbidden")), \
             patch.object(subprocess, "run", side_effect=AssertionError("process forbidden")):
            self.assertIn("--runtime nvidia", server.from_args(args).dry_run())

    def test_docker_prefix_is_one_shell_command(self):
        command = server.docker_command(["kill", "acc-test"], prefix="sg docker -c")
        self.assertEqual(command, ["sg", "docker", "-c", "docker kill acc-test"])

    def test_argv_and_env(self):
        with patch.dict("os.environ", {}, clear=True):
            running = server.Server("image", "/tmp/model dir/model.gguf", "/tmp/unused", ["-ngl", "99"],
                                    dict(EXPERT_FILL="1", VALUE="a b"), port=18080)
            cmd = running.command()
            self.assertEqual(cmd[cmd.index("--runtime") + 1], "nvidia")
            self.assertEqual(cmd[cmd.index("--network") + 1], "host")
            self.assertIn("NVIDIA_VISIBLE_DEVICES=all", cmd)
            self.assertIn("/tmp/model dir:/om:ro", cmd)
            self.assertIn("VALUE=a b", cmd)
            self.assertIn("127.0.0.1", cmd)
            self.assertIn("/om/model.gguf", cmd)
            self.assertIn("docker kill", running.dry_run())
            self.assertIn("docker rm -f", running.dry_run())
            self.assertNotEqual(running.name, server.Server("image", "/tmp/m", "/tmp/o", port=18080).name)

    def fake_run(self, argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="" if "ps" in argv else "0\n", stderr="")

    def test_kill_in_finally_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True), \
             patch.object(subprocess, "run", side_effect=self.fake_run) as run, \
             patch.object(subprocess, "Popen") as popen:
            running = server.OwnedContainer("image", "/tmp/model.gguf", tmp, "test", [])
            with self.assertRaisesRegex(RuntimeError, "item failed"):
                with running:
                    raise RuntimeError("item failed")
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(["docker", "kill", running.name], commands)
            self.assertIn(["docker", "rm", "-f", running.name], commands)
            self.assertEqual(popen.call_count, 2)
            watchdog = popen.call_args_list[0].args[0]
            self.assertEqual(watchdog[watchdog.index("--min-available-mib") + 1], "600")
            self.assertEqual(watchdog[watchdog.index("--max-swap-used-mib") + 1], "2560")
            self.assertEqual(shlex.split(watchdog[watchdog.index("--kill-cmd") + 1]), ["docker", "kill", running.name])
            for call in popen.call_args_list:
                self.assertIn("--out", call.args[0])
            running.close()
            self.assertEqual(sum(cmd == ["docker", "kill", running.name] for cmd in commands), 1)

    def test_launch_failure_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True):
            running = server.OwnedContainer("image", "/tmp/model.gguf", tmp, "test", [])
            def fail_launch(argv, **kwargs):
                if "run" in argv:
                    raise subprocess.CalledProcessError(1, argv)
                return self.fake_run(argv, **kwargs)
            with patch.object(subprocess, "run", side_effect=fail_launch) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    with running:
                        pass
                self.assertIn(["docker", "kill", running.name], [call.args[0] for call in run.call_args_list])

    def test_health_failure_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True), \
             patch.object(subprocess, "run", side_effect=self.fake_run) as run, \
             patch.object(subprocess, "Popen"), patch.object(server, "request_json", side_effect=OSError("offline")):
            running = server.Server("image", "/tmp/model.gguf", tmp, port=18080)
            with self.assertRaises(RuntimeError):
                with running:
                    pass
            self.assertIn(["docker", "kill", running.name], [call.args[0] for call in run.call_args_list])

    def test_sigterm_unwinds_context(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True), \
             patch.object(subprocess, "run", side_effect=self.fake_run) as run, patch.object(subprocess, "Popen"):
            running = server.OwnedContainer("image", "/tmp/model.gguf", tmp, "test", [])
            with self.assertRaises(SystemExit):
                with running:
                    running._signal(signal.SIGTERM, None)
            self.assertIn(["docker", "kill", running.name], [call.args[0] for call in run.call_args_list])

    def test_kld_argv_fixed_context_and_readonly_inputs(self):
        args = argparse.Namespace(image="image", model="/tmp/model.gguf", outdir="/tmp/out",
            text="/tmp/data/wiki.test.raw", logits="/tmp/logits/ref.bin", engine_arg=["-ngl", "99"],
            env=["EXPERT_FILL=1"], chunks=8, action="candidate")
        with patch.dict("os.environ", {}, clear=True):
            cmd = kld.build_runner(args).command()
            self.assertEqual(cmd[cmd.index("-c") + 1], "512")
            self.assertIn("--kl-divergence", cmd)
            self.assertIn("--kl-divergence-base", cmd)
            self.assertIn("/tmp/logits:/kl:ro", cmd)
            self.assertIn("/tmp/data:/d:ro", cmd)
            args.action = "reference"
            cmd = kld.build_runner(args).command()
            self.assertNotIn("--kl-divergence", cmd)
            self.assertIn("/tmp/logits:/kl:rw", cmd)


if __name__ == "__main__":
    unittest.main()
