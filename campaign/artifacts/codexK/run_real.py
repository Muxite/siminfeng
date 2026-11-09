"""Offline assignment T-K reproduction; all outputs stay in authorized locations."""
import json
from pathlib import Path
import sys
import time
import yaml

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'cc/engine_handoff'))
import engine_handoff as eh

out = root / 'cc/engine_handoff/example-real'
start = time.monotonic()
measurements = {}
def timed(name, function):
    step = time.monotonic()
    result = function()
    measurements[name] = round(time.monotonic() - step, 3)
    print(name, measurements[name], flush=True)
    return result

archive = root / 'builds/results/b0-baseline-3/engine-image.docker.tar'
record = archive.parent / 'build-record.json'
config = Path('/home/muk/models/campaign/granite-3.1-1b-a400m-instruct/config.json')
gguf = Path('/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf')
facts = timed('facts_seconds', lambda: eh.collect_facts(record, archive, config, gguf))
(out / 'facts.json').write_text(json.dumps(facts, indent=2) + '\n')
(out / 'facts.md').write_text(eh.markdown(facts))
(out / 'workload.json').write_bytes((root / 'experiments/handoff-real/workload.json').read_bytes())
requirements = yaml.safe_load((eh.ROOT / 'example/requirements.granite.yaml').read_text())
requirements['inputs']['workload']['artifact'] = str(out / 'workload.json')
requirements_path = out / 'requirements.yaml'
requirements_path.write_text(yaml.safe_dump(requirements, sort_keys=False))
handoff = timed('generate_seconds', lambda: eh.generate(facts, out / 'workload.json',
    ['/app/llama-completion', '-m', 'MODEL', '-n', '24', '--temp', '0', '-s', '1', '-ngl', '99', '-no-cnv'],
    requirements, str(requirements_path)))
(out / 'handoff.yaml').write_text(yaml.safe_dump(handoff, sort_keys=False))
verdict = timed('validate_seconds', lambda: eh.validate(handoff, out))
(out / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
measurements['total_seconds'] = round(time.monotonic() - start, 3)
(out / 'timing.json').write_text(json.dumps(measurements, indent=2) + '\n')
print(json.dumps(verdict, indent=2), flush=True)
print(json.dumps(measurements), flush=True)
