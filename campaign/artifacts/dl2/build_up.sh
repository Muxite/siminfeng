#!/bin/bash
cd /home/muk/projects/siminfeng/campaign
U=3b4d6ab3f6f0d2b4e0f3b4b8a5c1f1b1d0d6e9a1
U=$(git -C ref/llama.cpp-up rev-parse 3b4d6ab3f)
python3 cc/build_engine/build_engine.py --source ref/llama.cpp-up --commit $U --tag up0 --out builds/results/up0 --linkage; echo "up0 rc=$?"
python3 cc/build_engine/build_engine.py --source ref/llama.cpp-up --commit $U --patch $PWD/cc/components/release-stack-2-up3b4d6ab.patch --extra-target test-backend-ops --tag up1 --out builds/results/up1 --linkage; echo "up1 rc=$?"
