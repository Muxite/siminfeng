#!/bin/bash
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1
while pgrep -f '[c]odex exec|[c]c1plus|[n]vcc ' >/dev/null; do sleep 20; done
echo "== CAL2 $(date +%T)"; rm -rf artifacts/codexCAL2/gpu
python3 -B cc/host_calibrate/calibrate.py plan --image sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e --model /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf --cpu-layers 0 12 --out artifacts/codexCAL2/gpu > artifacts/codexCAL2/command-plan.json
python3 -B cc/host_calibrate/run_plan.py artifacts/codexCAL2/gpu; echo "cal2 rc=$?"
cp artifacts/codexPL4/model-stats.json /dev/null; python3 -B -c "import json;from pathlib import Path;s=json.loads(Path('artifacts/codexPL4/model-stats.json').read_text())['mixE4K'];Path('artifacts/codexCAL2/model-stats.json').write_text(json.dumps(s,indent=2))"
python3 -B cc/host_calibrate/calibrate.py parse --capture artifacts/codexCAL2/gpu --stats artifacts/codexCAL2/model-stats.json --out artifacts/codexCAL2/host-calibration.json; echo "parse rc=$?"
echo "== night aa e1 kld $(date +%T)"; python3 -u artifacts/night/run.py aa e1 kld; echo "night rc=$?"
