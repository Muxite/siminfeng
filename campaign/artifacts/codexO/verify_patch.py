"""Restore only the patched scratch file to the pinned blob; check and reapply."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parents[2]
tree = root / 'artifacts/codexO/tree'
patch = root / 'cc/moe_static_hot/static-hot.patch'
source = tree / 'src/llama-moe-cache.cpp'
env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tree.parent), GIT_CONFIG_NOSYSTEM="1")
base = '31288c9c66e59377f7788d920678a565bb89dd37'
original = subprocess.check_output(['git', '-C', str(root / 'ref/llama.cpp'), 'show', base + ':src/llama-moe-cache.cpp'])
source.write_bytes(original)
commands = []
for args in (['--check'], []):
    cmd = ['git', '-C', str(tree), 'apply', *args, str(patch)]
    subprocess.run(cmd, check=True, env=env)
    commands.append(cmd)
expected = source.read_bytes()
assert expected != original, "patch was silently skipped"
subprocess.run(['git', '-C', str(tree), 'apply', '--reverse', '--check', str(patch)], check=True, env=env)
result = dict(base=base, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
              commands=commands, apply_check='passed', apply='passed', reverse_check='passed',
              archive_git_discovery_isolated=True,
              source_sha256=hashlib.sha256(expected).hexdigest(),
              reference_status=subprocess.check_output(['git','-C',str(root/'ref/llama.cpp'),'status','--short'],text=True))
(root / 'artifacts/codexO/patch-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
