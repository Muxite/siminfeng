#!/usr/bin/env bash
# J-SPEC2: n-gram self-spec on over-memory Qwen3.6 (ABBA base/ngram-mod), after pipeline5 (waits on its PID, not a pgrep pattern).
cd "$(dirname "$0")/../.."

O=experiments/jetson/SPEC2-$(date +%Y%m%d-%H%M); IMG=campaign/llamacpp:jetson-up1108
for a in "base 1" "ngram-mod 1" "ngram-mod 2" "base 2"; do set -- $a
  bin/run_op --agent opus-jetson --task J-SPEC2 --desc "ngram self-spec $1 r$2" --timeout 5400 -- python3 cc/om_spec/om_spec.py $O/$1-r$2 $IMG $1 $2
done
echo SPEC2_DONE $O
