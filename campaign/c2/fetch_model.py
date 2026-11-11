#!/usr/bin/env python3
"""fetch_model.py REPO FILE DEST_DIR — download one GGUF via huggingface_hub, verify size+sha256 against HF metadata, append manifest."""
import hashlib, json, os, sys, time
from huggingface_hub import hf_hub_download, HfApi
repo, fn, dest = sys.argv[1:4]
os.makedirs(dest, exist_ok=True)
info = {s.rfilename: s for s in HfApi().model_info(repo, files_metadata=True).siblings}[fn]
t0 = time.time(); p = hf_hub_download(repo, fn, local_dir=dest); dt = time.time() - t0
h = hashlib.sha256()
with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
sha = h.hexdigest(); size = os.path.getsize(p)
exp = info.lfs.sha256 if info.lfs else None
rec = dict(repo=repo, file=fn, path=p, size=size, expected_size=info.size, sha256=sha, expected_sha256=exp,
           ok=(size == info.size and (exp is None or exp == sha)), download_s=round(dt, 1), ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
open(os.path.expanduser('~/models/campaign/MANIFEST.jsonl'), 'a').write(json.dumps(rec) + '\n')
print(json.dumps(rec)); sys.exit(0 if rec['ok'] else 1)
