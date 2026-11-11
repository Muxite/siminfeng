#!/usr/bin/env python3
"""gguf_layout: header-only layout of a GGUF (no tensor hashing): per tensor name/type/shape/bytes/file offset,
plus MoE summary (expert tensors per layer, bytes per expert, resident vs expert bytes). Usage:
PYTHONPATH=<llama.cpp>/gguf-py python3 gguf_layout.py MODEL.gguf [--json OUT]"""
import sys, json, re, collections
from gguf import GGUFReader
p = sys.argv[1]; r = GGUFReader(p)
arch = r.fields['general.architecture'].contents()
kv = {k: r.fields[k].contents() for k in r.fields if k.startswith(arch + '.') and not isinstance(r.fields[k].contents(), list)}
rows = []; moe = collections.defaultdict(lambda: collections.defaultdict(int)); resident = 0
for t in r.tensors:
    b = int(t.n_bytes); rows.append(dict(name=t.name, type=t.tensor_type.name, shape=[int(x) for x in t.shape], bytes=b, offset=int(t.data_offset)))
    m = re.match(r'blk\.(\d+)\.(ffn_(gate|up|down|gate_up)_exps)\.weight', t.name)
    if m: moe[int(m[1])][m[2] + ':' + t.tensor_type.name] += b
    else: resident += b
n_exp = kv.get(f'{arch}.expert_count'); used = kv.get(f'{arch}.expert_used_count')
per_layer = {l: sum(d.values()) for l, d in moe.items()}
exp_total = sum(per_layer.values())
out = dict(file=p, arch=arch, n_tensors=len(rows), total_bytes=sum(x['bytes'] for x in rows), expert_bytes=exp_total, non_expert_bytes=resident,
           expert_count=n_exp, expert_used=used, layers_with_experts=len(moe),
           bytes_per_expert_by_layer={l: (v // n_exp if n_exp else None) for l, v in sorted(per_layer.items())},
           expert_types_by_layer={l: dict(d) for l, d in sorted(moe.items())}, kv=kv)
if n_exp and used and moe:
    bpe = exp_total / (n_exp * len(moe)); out['avg_bytes_per_expert'] = bpe
    out['routed_expert_bytes_per_token'] = bpe * used * len(moe)
if '--json' in sys.argv: json.dump(dict(out, tensors=rows), open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ('expert_types_by_layer', 'bytes_per_expert_by_layer', 'kv')}, indent=1, default=str))
print('types layer0:', dict(list(out['expert_types_by_layer'].items())[0][1]) if moe else None)
