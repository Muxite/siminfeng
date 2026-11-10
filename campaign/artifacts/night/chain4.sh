#!/bin/bash
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1
echo "== start $(date +%T)"; python3 -u artifacts/night/run2.py x2 x1 x3 x8 x4 x5 x6 x7; echo "run2 rc=$?"
echo "chain4 done $(date +%T)"
