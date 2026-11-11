"""Common serial evaluator: one loaded server, one durable record per item."""
import json
import os
import time
import uuid

try:
    from . import common, server
except ImportError:
    import common
    import server


def run(args, mode, protocol, make_payload, score, format_prompt):
    if args.dry_run:
        print(server.from_args(args).dry_run())
        return
    items = common.read_jsonl(args.data)
    if not items:
        raise ValueError("empty dataset")
    common.indexed(items)
    for item in items:
        format_prompt(item)  # Validate before starting a costly model.
    out = common.prepare_run(args, mode, protocol)
    output = out / "items.jsonl"
    pending = common.pending_items(items, output)
    try:
        if pending:
            session = out / "sessions" / uuid.uuid4().hex
            with server.from_args(args, session) as running:
                template_path = out / "template.json"
                template = dict(chat_template=running.props.get("chat_template"),
                                chat_template_caps=running.props.get("chat_template_caps", {}),
                                thinking_settings=common.thinking_settings(running.props))
                if template_path.exists() and json.loads(template_path.read_text()) != template:
                    raise ValueError("server template changed since previous attempt")
                common.write_json(template_path, template)
                needs_newline = output.exists() and output.stat().st_size > 0
                if needs_newline:
                    with output.open("rb") as previous:
                        previous.seek(-1, os.SEEK_END)
                        needs_newline = previous.read(1) != b"\n"
                with output.open("a", encoding="utf-8") as stream:
                    if needs_newline:
                        stream.write("\n")
                    for item in pending:
                        request = make_payload(item, running.props, protocol["max_tokens"])
                        started = time.monotonic()
                        response = common.request_json(running.url + "/v1/chat/completions", request,
                                                       timeout=args.request_timeout)
                        elapsed_ms = (time.monotonic() - started) * 1000
                        row = dict(id=item["id"], answer=item["answer"], source=item["source"],
                                   meta=item.get("meta", {}), elapsed_ms=elapsed_ms,
                                   session=str(session.relative_to(out)), request=request, response=response,
                                   **score(item, response), **common.response_metrics(response))
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        print(f"{item['id']}: correct={row['correct']} ({elapsed_ms:.0f} ms)", flush=True)
    finally:
        rows = common.read_jsonl(output)
        result = common.summary(rows)
        result.update(items_file="items.jsonl", expected_n=len(items), complete=len(rows) == len(items),
                      mode=mode, protocol=protocol, dataset_sha256=common.sha256(args.data))
        common.write_json(out / "summary.json", result)
    print(json.dumps(result, indent=2))
