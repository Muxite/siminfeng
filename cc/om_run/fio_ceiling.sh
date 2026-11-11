#!/usr/bin/env bash
# NVMe read ceiling on the actual model file (read-only, O_DIRECT bypasses page cache). Waits for builds and sweeps.
cd "$(dirname "$0")/../.."
while pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null || pgrep -f "bash cc/om_run/sweep_baselines.sh" >/dev/null || pgrep -f "cc/om_run/sweep_" >/dev/null; do sleep 30; done
F=$HOME/models/campaign/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf; O=experiments/jetson/OM/fio; mkdir -p $O
for spec in "rand4k:randread:4k:1:1" "rand4k-qd32:randread:4k:32:1" "rand128k-qd8:randread:128k:8:1" "rand576k-qd1:randread:576k:1:1" "rand576k-qd8:randread:576k:8:1" "rand2m-qd4:randread:2m:4:1" "seq1m-qd8:read:1m:8:1" "rand576k-qd8-psync4:randread:576k:1:4"; do
  IFS=: read name rw bs qd jobs <<< "$spec"
  eng=libaio; [ "${name#*psync}" != "$name" ] && eng=psync
  bin/run_op --agent opus-jetson --task J-FIO --desc "fio $name on model file (O_DIRECT, read-only)" --timeout 120 -- fio --name=$name --filename=$F --readonly --direct=1 --rw=$rw --bs=$bs --iodepth=$qd --numjobs=$jobs --ioengine=$eng --runtime=20 --time_based --group_reporting --output-format=json --output=$O/$name.json >/dev/null 2>&1
  python3 -c "import json;j=json.load(open('$O/$name.json'))['jobs'][0]['read'];print('$name',round(j['bw']/1024,1),'MiB/s',round(j['iops']),'iops','lat_mean_us',round(j['lat_ns']['mean']/1000))"
done
echo FIO_DONE
