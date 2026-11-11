"""Save reference logits, run candidate KL divergence, or parse captured output."""
import argparse
import json
from pathlib import Path
import re
import subprocess

try:
    from . import common, server
except ImportError:
    import common
    import server

FLOAT = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def parse(text):
    patterns = dict(ppl=r"Mean PPL\(Q\)\s*:\s*" + FLOAT,
                    mean_kld=r"Mean\s+KLD:\s*" + FLOAT,
                    kld_p99=r"99\.0%\s+KLD:\s*" + FLOAT,
                    same_top_p=r"Same top p:\s*" + FLOAT,
                    rms_dp=r"RMS Δp\s*:\s*" + FLOAT)
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"missing {key}: incomplete or unsupported KL output")
        result[key] = float(match.group(1))
    context = re.search(r"computing over\s+(\d+)\s+chunks,\s*n_ctx=(\d+)", text)
    rows = re.findall(r"^\s*(\d+)\s+" + FLOAT + r"\s*±", text, re.M)
    result["n_chunks"] = int(rows[-1][0]) if rows else None
    result["n_ctx"] = int(context.group(2)) if context else None
    result["planned_chunks"] = int(context.group(1)) if context else None
    result["probability_units"] = "percent"  # same_top_p and rms_dp, as printed
    return result


parse_kld = parse


def build_runner(args):
    text = Path(args.text).expanduser().resolve()
    logits = Path(args.logits).expanduser().resolve()
    argv = [*args.engine_arg, "-m", "/om/" + Path(args.model).name,
            "-f", "/d/" + text.name, "-c", "512", "--chunks", str(args.chunks),
            "-b", "512", "-ub", "1", "--kl-divergence-base", "/kl/" + logits.name]
    if args.action == "candidate":
        argv.append("--kl-divergence")
    return server.OwnedContainer(args.image, args.model, args.outdir, "llama-perplexity", argv,
        server.parse_env(args.env), mounts=[(text.parent, "/d", True),
                                          (logits.parent, "/kl", args.action == "candidate")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    parse_cmd = commands.add_parser("parse", help="Parse saved stdout + stderr")
    parse_cmd.add_argument("file", type=Path)
    for name in ("reference", "candidate"):
        cmd = commands.add_parser(name)
        server.add_arguments(cmd)
        cmd.add_argument("--logits", required=True, help="Host reference logits file (output/input)")
        cmd.add_argument("--text", default="~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw")
        cmd.add_argument("--chunks", type=int, default=8)
        cmd.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.action == "parse":
        print(json.dumps(parse(args.file.read_text()), indent=2))
        return
    if args.chunks < 1:
        parser.error("chunks must be positive")
    running = build_runner(args)
    if args.dry_run:
        print(running.dry_run())
        return
    out = Path(args.outdir)
    if out.exists():
        raise FileExistsError("KL runs require a fresh outdir")
    logits = Path(args.logits).expanduser().resolve()
    metadata = logits.with_suffix(logits.suffix + ".meta.json")
    corpus_sha = common.sha256(Path(args.text).expanduser())
    fixed = dict(text_sha256=corpus_sha, n_ctx=512, chunks=args.chunks, batch=512, ubatch=1)
    if args.action == "reference":
        if logits.exists() or metadata.exists():
            raise FileExistsError("refusing to overwrite reference logits")
        logits.parent.mkdir(parents=True, exist_ok=True)
    else:
        reference = json.loads(metadata.read_text())
        if reference["fixed"] != fixed or reference["logits_sha256"] != common.sha256(logits):
            raise ValueError("reference logits/corpus/context/chunks do not match")
    provenance = dict(fixed=fixed, model_sha256=common.sha256(Path(args.model).expanduser()),
                      image=args.image, env=args.env, engine_argv=args.engine_arg)
    with running:
        common.write_json(out / "provenance.json", provenance)
        completed = subprocess.run(server.docker_command(["wait", running.name]),
                                   capture_output=True, text=True, timeout=args.timeout, check=True)
        if completed.stdout.strip() != "0":
            raise RuntimeError(f"llama-perplexity exit {completed.stdout.strip()}; inspect logs")
    text = (out / "stdout.txt").read_text() + "\n" + (out / "stderr.txt").read_text()
    if args.action == "candidate":
        result = parse(text)
        if result["n_ctx"] != 512 or result["n_chunks"] != args.chunks:
            raise ValueError("KL output context/chunks do not match requested workload")
    else:
        if not logits.is_file() or not logits.stat().st_size:
            raise RuntimeError("reference did not produce logits")
        common.write_json(metadata, dict(**provenance, logits_sha256=common.sha256(logits),
                                        run_dir=str(out.resolve())))
        result = dict(reference_saved=str(logits), **fixed)
    common.write_json(out / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
