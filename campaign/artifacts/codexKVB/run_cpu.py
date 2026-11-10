import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
phase = sys.argv[1]
model = str(Path.home() / 'models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf')
base = [str(root / 'build/bin/llama-completion'), '-m', model, '-p', 'hi', '-n', '4', '-c', '4096', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-t', '4', '--seed', '42']
cases = [('q8_0', [], {}), ('f16', ['-ctk', 'f16', '-ctv', 'f16'], {}), ('q4_0', ['-ctk', 'q4_0', '-ctv', 'q4_0'], {}), ('q8_0-no-rotation', [], {'LLAMA_ATTN_ROT_DISABLE': '1'})]
if phase == 'after':
    cases += [('q8_0-32k', ['-c', '32768'], {}), ('q8_0-no-fit', ['--fit', 'off'], {})]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
for name, extra, env in cases:
    cmd = ['nice', '-n', '10'] + base + extra
    prefix = root / f'{phase}-{name}'
    start = time.monotonic()
    with prefix.with_suffix('.stdout').open('w') as out, prefix.with_suffix('.stderr').open('w') as err:
        result = subprocess.run(cmd, stdout=out, stderr=err, env={**os.environ, **env}, timeout=180)
    record = {'command': cmd, 'env': env, 'returncode': result.returncode, 'seconds': time.monotonic() - start}
    prefix.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'case': name, **record}), flush=True)
