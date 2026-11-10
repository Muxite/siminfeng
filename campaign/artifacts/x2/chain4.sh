#!/bin/bash
set -u; cd ~/projects/siminfeng/campaign; export CAMPAIGN_ID=c1
bin/lab image add llamacpp-release1 sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e >/dev/null
(while true; do echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-3 /proc/loadavg) cc=$(pgrep -c -f '[c]c1plus|[n]vcc|[c]make --build') dl=$(pgrep -c -f '[c]url -sS|[h]f download')"; sleep 5; done > artifacts/x2/chain4-hostload.log 2>&1) & MON=$!
JQ=$(bin/lab submit --image llamacpp-release1 --model ~/models/campaign/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf --tests $(cat artifacts/x2/qwen1-order.txt) --key QW1-frontier-1 --res timeout_s=900,memory=40g); echo "QW $JQ"
bin/run_op --agent claude-code --task X9 --desc "QW1 Qwen frontier job $JQ" -- bin/lab run $JQ > artifacts/x2/qw1-run.out 2>&1; echo "QW rc=$?"
JR=$(bin/lab submit --image llamacpp-release1 --model ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf --tests $(cat artifacts/x2/rel1-order.txt) --key REL1-speed-1 --res timeout_s=600); echo "REL $JR"
bin/run_op --agent claude-code --task X5 --desc "REL1 release-1 speed qualification $JR" -- bin/lab run $JR > artifacts/x2/rel1-run.out 2>&1; echo "REL rc=$?"
JP=$(bin/lab submit --image llamacpp-b0 --model $PWD/artifacts/codexPAP/models/pap12_q3.gguf --tests $(cat artifacts/x2/pap2-order.txt) --key PAP2-quiet-5r --res timeout_s=600); echo "PAP2 $JP"
bin/run_op --agent claude-code --task X4 --desc "PAP2 quiet-host rerun $JP" -- bin/lab run $JP > artifacts/x2/pap2-run.out 2>&1; echo "PAP2 rc=$?"
kill $MON; echo "$JQ $JR $JP" > artifacts/x2/chain4.jobs; awk '{print $(NF-1),$NF}' artifacts/x2/chain4-hostload.log | sort | uniq -c
