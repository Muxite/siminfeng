#!/usr/bin/env python3
"""gguf_manifest: content identity of a GGUF independent of metadata noise.
Usage: PYTHONPATH=<llama.cpp>/gguf-py python3 gguf_manifest.py MODEL.gguf > manifest.json
Output: file sha256/bytes, tensor_data_sha256 (over name|type|shape|bytes in file order), per-tensor sha256, metadata key/value digest
(values truncated for arrays). Compare two manifests: same tensor_data_sha256 => identical weights even if file sha differs."""
import sys, json, hashlib
from gguf import GGUFReader
p = sys.argv[1]; r = GGUFReader(p)
fh = hashlib.sha256()
with open(p, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""): fh.update(b)
tensors, agg = [], hashlib.sha256()
for t in r.tensors:
    d = bytes(t.data.tobytes()); h = hashlib.sha256(d).hexdigest()
    tensors.append(dict(name=t.name, type=t.tensor_type.name, shape=[int(x) for x in t.shape], bytes=len(d), sha256=h))
    agg.update(f"{t.name}|{t.tensor_type.name}|{list(map(int, t.shape))}|{h}\n".encode())
meta = {}
for k, fld in r.fields.items():
    if k.startswith("GGUF."): continue
    try:
        v = fld.contents()
        if isinstance(v, list): v = dict(len=len(v), sha256=hashlib.sha256(json.dumps(v, default=str).encode()).hexdigest())
    except Exception as e: v = f"<unreadable {e}>"
    meta[k] = v
print(json.dumps(dict(schema="gguf-manifest/0", file=p, file_sha256=fh.hexdigest(), n_tensors=len(tensors),
                      tensor_data_sha256=agg.hexdigest(), metadata=meta, tensors=tensors), indent=1, default=str))
