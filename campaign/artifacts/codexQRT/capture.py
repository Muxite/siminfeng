#!/usr/bin/env python3
"""Serial, CPU-only routing capture. All outputs remain in codexQRT."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/codexQRT'
MODEL = Path('/home/muk/models/campaign/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf')
records = []
for name, split in [('prose', 'TRAIN'), ('chat', 'EVAL'), ('code', 'EVAL')]:
    prompt = ROOT / f'artifacts/traces/in/{name}.txt'
    if not prompt.exists():
        continue
    trace = OUT / f'{name}.jsonl'
    command = ['nice', '-n', '10', str(OUT / 'build/bin/llama-routing-trace'),
               '-m', str(MODEL), '--load-mode', 'mmap', '--no-repack',
               '-ngl', '0', '-t', '8', '-tb', '8', '-b', '128', '-ub', '128',
               '-c', '4096', '-f', str(prompt), '-n', '64', '--trace-out', str(trace)]
    print(f'Start {name}', flush=True)
    start = time.monotonic()
    with (OUT / f'{name}.stdout').open('w') as stdout, (OUT / f'{name}.stderr').open('w') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    item = dict(name=name, split=split, command=command, exit_code=result.returncode,
                elapsed_seconds=time.monotonic()-start,
                prompt_bytes=prompt.stat().st_size,
                prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
                trace_bytes=trace.stat().st_size if trace.exists() else 0,
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None)
    records.append(item)
    (OUT / 'captures.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(item), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    inspect = subprocess.run(['python3', 'cc/model_inspect/inspect_model.py', str(MODEL),
                              '--trace', str(trace), '--out', str(OUT / f'{name}-inspect.json')],
                             cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    inspect.check_returncode()
