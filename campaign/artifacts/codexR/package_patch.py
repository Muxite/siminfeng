import hashlib
import json
from pathlib import Path
import subprocess

base=Path('ref/llama.cpp')
tree=Path('artifacts/codexR/tree')
files=['CMakeLists.txt','docs/ops.md','ggml/src/ggml-cpu/ggml-cpu.c',
       'src/llama-arch.cpp','src/llama-arch.h','src/llama-graph.cpp','src/llama-graph.h',
       'src/llama-model.h','src/models/granite-moe.cpp','src/models/granite.cpp',
       'tests/CMakeLists.txt','tests/test-hybrid-split.cpp']
patch=bytearray()
for name in files:
    old=base/name
    new=tree/name
    cmd=['git','diff','--no-ext-diff','--no-index','--',str(old) if old.exists() else '/dev/null',str(new)]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if proc.returncode != 1:
        raise RuntimeError((name,proc.returncode,proc.stderr.decode()))
    data=proc.stdout.replace(b'a/ref/llama.cpp/',b'a/').replace(b'b/artifacts/codexR/tree/',b'b/').replace(b'a/artifacts/codexR/tree/',b'a/')
    patch.extend(data)
output=Path('cc/hybrid_split/hybrid-split.patch')
output.write_bytes(patch)
report={'base_commit':subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip(),
        'patch':str(output),'patch_sha256':hashlib.sha256(patch).hexdigest(),
        'bytes':len(patch),'files':files}
Path('artifacts/codexR/patch.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
