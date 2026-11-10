#!/bin/bash
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1
echo "== kld2 $(date +%T)"; python3 -u artifacts/night/run.py kld2; echo "kld2 rc=$?"
echo "== e2 (K30 + C3-8GB) $(date +%T)"; python3 -u -c "
import sys; sys.argv=['run.py']; sys.path.insert(0,'artifacts/night')
import runpy; m=runpy.run_path('artifacts/night/run.py', run_name='night')
import os; os.makedirs(m['OUT']+'/logs', exist_ok=True)
m['phase_e2'](('up1', ['-ngl','99','--n-cpu-moe','48','--moe-cache-mib','6093','--moe-cache-cpu-share','0.9']))"; echo "e2 rc=$?"
echo "== e3 $(date +%T)"; python3 -u artifacts/night/run.py e3; echo "e3 rc=$?"
echo "chain2 done $(date +%T)"
