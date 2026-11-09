#!/usr/bin/env python3
"""Optional bounded CPU KL check using the existing BF16 reference."""
import json
import math
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/codexPAP'
BASE = ROOT / 'artifacts/kl/bf16.kl'
RESULT = OUT / 'cpu-sanity.json'
with BASE.open('rb') as f:
    magic, ctx, vocab, chunks = struct.unpack('<8sIii', f.read(20))
record = {'model': 'pap12_q3.gguf', 'started_utc': datetime.now(timezone.utc).isoformat(),
          'baseline_header': {'magic': magic.decode(), 'context': ctx, 'vocab': vocab, 'chunks': chunks}}
if (magic, ctx, vocab, chunks) != (b'_logits_', 512, 49155, 20):
    record.update(status='skipped', reason='incompatible BF16 KL header')
else:
    cmd = ['nice', '-n', '10', str(OUT / 'build/bin/llama-perplexity'),
           '-m', str(OUT / 'models/pap12_q3.gguf'),
           '-f', str(Path.home() / 'models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw'),
           '-c', '512', '--chunks', '20', '-t', '6', '-tb', '6', '-ngl', '0',
           '--kl-divergence-base', str(BASE), '--kl-divergence']
    record['command'] = cmd
    log = OUT / 'logs/cpu-sanity.log'
    with log.open('w') as f:
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=600, check=False)
            record['exit_code'] = proc.returncode
        except subprocess.TimeoutExpired:
            record['status'] = 'timeout'
    text = log.read_text()
    stats = re.search(r'Mean\s+KLD:\s+([0-9.eE+-]+)\s+±\s+([0-9.eE+-]+)', text)
    if stats and record.get('exit_code') == 0:
        record['mean_kld'], record['kld_uncertainty'] = map(float, stats.groups())
        assert math.isfinite(record['mean_kld'])
        record['status'] = 'passed'
        for key, pattern in [('ppl', r'Mean PPL\(Q\)\s*:\s*([0-9.eE+-]+)'),
                             ('ppl_base', r'Mean PPL\(base\)\s*:\s*([0-9.eE+-]+)'),
                             ('ppl_ratio', r'Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.eE+-]+)')]:
            m = re.search(pattern, text)
            if m:
                record[key] = float(m[1])
    elif 'status' not in record:
        record['status'] = 'failed'
        record['reason'] = 'No complete KL statistics or nonzero exit code; inspect log'
    record['log'] = str(log.relative_to(ROOT))
record['finished_utc'] = datetime.now(timezone.utc).isoformat()
RESULT.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
