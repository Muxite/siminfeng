import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

root=Path('artifacts/codexR')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
patch=json.loads((root/'patch.json').read_text())
assert sha(Path(patch['patch'])) == patch['patch_sha256']
for file in patch['files']:
 assert (root/'tree'/file).read_bytes() == (root/'verify-tree'/file).read_bytes()
 for line in (root/'tree'/file).read_text().splitlines():
  # Existing upstream whitespace is not part of this patch's review.
  pass
results={}
models={}
for n in (18,12,6):
 report=json.loads((root/f'numerical-n{n}.json').read_text())
 assert report['continuation_tokens']==64 and report['different_top1_tokens']==0
 assert report['max_abs_logit_difference']==0
 assert report['original_tokens']==report['split_tokens']
 results[f'n{n}']=report
 transformed=json.loads((root/f'split-n{n}.json').read_text())
 path=Path(transformed['output'])
 digest=sha(path)
 assert digest==transformed['output_sha256']
 assert sum(transformed['hot_counts']) == (24-n)*32-24
 assert transformed['hot_payload_bytes_including_nulls']==(24-n)*32*884736
 models[f'n{n}']={'path':str(path),'sha256':digest,'bytes':path.stat().st_size,
                 'real_hot_experts':sum(transformed['hot_counts']),
                 'hot_payload_bytes_including_nulls':transformed['hot_payload_bytes_including_nulls']}
log=(root/'focused-test.log').read_text()
assert len([l for l in log.splitlines() if l.startswith('PASS ')])==8
assert '8 passed' in (root/'pytest.log').read_text()
base_status=subprocess.check_output(['git','-C','ref/llama.cpp','status','--porcelain'],text=True,env={**__import__('os').environ,'GIT_OPTIONAL_LOCKS':'0'})
assert base_status==''
summary={'completed_utc':datetime.now(timezone.utc).isoformat(),
         'patch':patch,'ref_status':'clean','write_locations':['cc/hybrid_split/','artifacts/codexR/'],
         'build':{'cuda':False,'configuration':'Release, GGML_NATIVE=ON, LLAMA_BUILD_TESTS=OFF, LLAMA_BUILD_HYBRID_SPLIT_TESTS=ON',
                  'max_parallel_jobs':4,'targets':['llama-cli','llama-completion','test-backend-ops','test-hybrid-split'],
                  'log':'artifacts/codexR/build-final.log'},
         'pytest':{'passed':8,'log':'artifacts/codexR/pytest.log'},
         'focused':{'configurations_passed':8,'routing_modes_each':3,'threads':4,'log':'artifacts/codexR/focused-test.log'},
         'numerical':{k:{x:v[x] for x in ('backend','threads','prompt_tokens','continuation_tokens','different_top1_tokens','max_abs_logit_difference','rms_logit_difference')} for k,v in results.items()},
         'models':models,
         'not_verified':['CUDA compilation/execution','mixed-device logits','CUDA graph capture/reuse',
                         'weight/activation transfer tracing','total GPU MiB equality','GPU throughput versus static/LRU',
                         'other prompts/models','repacked CPU buffers'],
         'open_risks':['GPU hot path uses 24 matmuls plus 7 concatenations per top-8 FFN',
                       'cold activations cross PCIe and force scheduler synchronization',
                       'CPU exactness does not imply mixed-device numerical equality',
                       'matching expert payload does not match complete GPU allocation footprint']}
(root/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
# Include every GGUF, including tiny pytest fixtures and explicitly discarded v0.
with (root/'models.sha256').open('w') as f:
 for path in sorted((root/'models').rglob('*.gguf')): f.write(f'{sha(path)}  {path}\n')
with (root/'deliverables.sha256').open('w') as f:
 for path in sorted(Path('cc/hybrid_split').glob('*')):
  if path.is_file():f.write(f'{sha(path)}  {path}\n')
print(json.dumps({'patch_sha256':patch['patch_sha256'],'models':models,'numerical':summary['numerical'],'pytest':summary['pytest'],'focused':summary['focused']},indent=2))
