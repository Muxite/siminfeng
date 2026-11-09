#!/bin/bash
set -u; cd ~/projects/siminfeng/campaign; export CAMPAIGN_ID=c1
until [ -f builds/results/kvfix-1/build-record.json ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('builds/results/kvfix-1/build-record.json'))['status'] in ('success','failed') else 1)" 2>/dev/null; do sleep 10; done
KST=$(python3 -c "import json;print(json.load(open('builds/results/kvfix-1/build-record.json'))['status'])"); echo "kvfix build: $KST"
(while true; do echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-3 /proc/loadavg) cc=$(pgrep -c -f '[c]c1plus|[n]vcc|[c]make --build') dl=$(pgrep -c -f '[h]f download')"; sleep 5; done > artifacts/x2/chain3-hostload.log 2>&1) & MON=$!
JP=$(bin/lab submit --image llamacpp-b0 --model $PWD/artifacts/codexPAP/models/pap12_q3.gguf --tests $(cat artifacts/x2/pap1-order.txt) --key PAP1-alt-1 --res timeout_s=600); echo "PAP $JP"
bin/run_op --agent claude-code --task X4 --desc "PAP1 timed job $JP" -- bin/lab run $JP > artifacts/x2/pap1-run.out 2>&1; echo "PAP rc=$?"
if [ "$KST" = success ]; then RR=$(python3 -c "import json;print(json.load(open('builds/results/kvfix-1/build-record.json'))['image']['run_ref'])"); bin/lab image add llamacpp-kvfix1 $RR >/dev/null
JK=$(bin/lab submit --image llamacpp-kvfix1 --model ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf --data ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw --tests $(cat artifacts/x2/kv1-order.txt) --key KV2-precision-32k-kvfix --res timeout_s=600); echo "KV2 $JK"
bin/run_op --agent claude-code --task X4 --desc "KV2 timed job $JK (kvfix image)" -- bin/lab run $JK > artifacts/x2/kv2-run.out 2>&1; echo "KV2 rc=$?"; fi
JL=$(bin/lab submit --image llamacpp-b0 --model ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf --data ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw --tests $(cat artifacts/x2/plan1-order.txt) --key PLAN1-validate-900 --res timeout_s=600); echo "PLAN $JL"
bin/run_op --agent claude-code --task X4 --desc "PLAN1 planner validation job $JL" -- bin/lab run $JL > artifacts/x2/plan1-run.out 2>&1; echo "PLAN rc=$?"
kill $MON; echo "${JP} ${JK:-none} ${JL}" > artifacts/x2/chain3.jobs; awk '{print $(NF-1),$NF}' artifacts/x2/chain3-hostload.log | sort | uniq -c
