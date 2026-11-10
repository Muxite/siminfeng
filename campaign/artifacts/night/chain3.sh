#!/bin/bash
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1
until [ -f /home/muk/models/campaign/qwen36-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.sha256 ]; do sleep 60; done
echo "== download verified $(date +%T)"; cat /home/muk/models/campaign/qwen36-mtp/*.sha256
echo "== e4 $(date +%T)"; python3 -u artifacts/night/run.py e4; echo "e4 rc=$?"
echo "chain3 done $(date +%T)"
