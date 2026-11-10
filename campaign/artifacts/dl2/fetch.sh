#!/bin/bash
# Pinned downloads: verify sha256 against HF LFS oid. No auth.
set -u
get(){ repo=$1; file=$2; dir=$3
  oid=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | python3 -c "import json,sys;print([f['lfs']['oid'] for f in json.load(sys.stdin) if f['path']=='$file'][0])")
  echo "$repo/$file expected $oid"
  for try in $(seq 60); do
    curl -sS -L --fail --speed-limit 100000 --speed-time 60 --retry 3 --retry-delay 5 -C - -o "$dir/$file" "https://huggingface.co/$repo/resolve/main/$file" && break
    echo "retry $try at $(stat -c%s "$dir/$file" 2>/dev/null) bytes"; sleep 5
  done
  got=$(sha256sum "$dir/$file" | cut -d' ' -f1); echo "$got  $file" > "$dir/$file.sha256"
  [ "$got" = "$oid" ] && echo "OK $file" || echo "SHA_MISMATCH $file $got"
}
get 12bitmisfit/Qwen3-30B-A3B_Pruned_REAP-15B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf /home/muk/models/campaign/qwen3-reap15b
get unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf /home/muk/models/campaign/qwen36-mtp
