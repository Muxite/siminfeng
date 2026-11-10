#!/usr/bin/env python3
"""Offline command planning and parsing; never starts Docker or inference."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/codexCAL'
PIN = '31288c9c66e59377f7788d920678a565bb89dd37'
# GGML block sizes and bytes, from pinned ggml-common.h.
BLOCK = {'Q2_K': (256, 84), 'Q3_K': (256, 110), 'Q4_K': (256, 144),
         'Q6_K': (256, 210), 'Q8_0': (32, 34)}
TYPES = ('q2_K', 'q3_K', 'q4_K', 'q6_K', 'q8_0')


def output_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(OUT):
        raise ValueError('output must stay under artifacts/codexCAL')
    return p


def read(path):
    return json.loads(Path(path).read_text())


def emit(path, obj):
    p = output_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def positive(x):
    x = float(x)
    if not math.isfinite(x) or x <= 0:
        raise ValueError('expected finite positive measurement')
    return x


def summary(values):
    xs = [positive(x) for x in values]
    if not xs:
        raise ValueError('no samples')
    return dict(median=statistics.median(xs), min=min(xs), max=max(xs), count=len(xs),
                stdev=statistics.stdev(xs) if len(xs)>1 else None,
                note='Observed spread, not a confidence interval; systematic model error unbounded.')


def parse_perf(text):
    """Pinned print_perf_console: ANSI, FLOPS for matmul; GiB/s labelled GB/s otherwise."""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    rows = []
    for line in text.splitlines():
        m = re.match(r'\s*(MUL_MAT(?:_ID)?)\((.*)\):\s+(\d+) runs -\s*([\d.]+) us/run - (.*)', line)
        if not m:
            continue
        op, params, runs, us, tail = m.groups()
        p = dict(re.findall(r'(\w+)=([^,\[\]]+)(?:,|$)', params))
        kind = p['type_a'].upper()
        block, size = BLOCK[kind]
        # Only the bounded contiguous decode shapes emitted by command_plan are accepted.
        if int(p['n']) != 1 or int(p['k']) % block:
            raise ValueError('not an aligned decode shape')
        if op == 'MUL_MAT':
            if (int(p['m']), int(p['k'])) != (4096, 14336) or 'bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0' not in params:
                raise ValueError('unexpected dense shape')
            used = 1
        else:
            if (int(p['m']), int(p['k']), int(p['n_mats']), int(p['n_used'])) != (768,2048,128,8) or p.get('b') != '0' or p.get('m_v') != '0' or p.get('ids_offset') != '0':
                raise ValueError('unexpected expert shape')
            used = int(p['n_used'])
        weight_bytes = int(p['m']) * (int(p['k'])//block) * size * used
        literal = re.search(r'([\d.]+) GB/s', tail)
        rows.append(dict(op=op, kind=kind, runs=int(positive(runs)), time_us=positive(us),
                         active_weight_bytes=weight_bytes,
                         weight_gbs=weight_bytes/(positive(us)*1000),
                         printed_gbs=float(literal[1]) if literal else None,
                         printed_decimal_gbs=float(literal[1])*2**30/1e9 if literal else None,
                         params=params, raw=line))
    if not rows:
        raise ValueError('no successful perf rows (empty filter, unsupported, timeout or malformed output)')
    return rows


def parse_bench(text, n_gen=64, coord=None):
    rows = json.loads(text)
    if len(rows) != 1:
        raise ValueError('expected one benchmark')
    row = rows[0]
    if row['n_prompt'] != 0 or row['n_gen'] != n_gen or (coord is not None and row['n_cpu_moe'] != coord):
        raise ValueError('benchmark workload mismatch')
    return [1e6/positive(x) for x in row['samples_ts']]


def parse_h2d(text):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    expected = {(2**i*1048576, r) for i in range(9) for r in range(3)}
    if {(r['bytes'],r['repeat']) for r in rows} != expected or len(rows) != 27:
        raise ValueError('incomplete H2D sweep')
    for r in rows:
        if r['direction'] != 'H2D' or r['pinned'] is not True:
            raise ValueError('wrong transfer')
        r['gbs'] = r['bytes']/(positive(r['ms'])*1e6)
    return rows


def identity(nvidia, lscpu):
    fields = dict(line.split(':',1) for line in lscpu.splitlines() if ':' in line)
    return dict(id=hashlib.sha256((nvidia+'\n'+lscpu).encode()).hexdigest(),
                cpu_model=fields.get('Model name','').strip() or None,
                nvidia_smi_text=nvidia, lscpu_text=lscpu,
                note='Raw identity text; unavailable GPU information remains unknown.')


def model_profile(model):
    """Header-only check: qualify the supplied small Granite allocation envelope."""
    sys.path.insert(0, str(ROOT/'cc/model_inspect'))
    from inspect_model import inspect
    stats = inspect(model)
    limits = dict(arch='granitemoe', n_layer=24, n_expert=32, n_expert_used=8,
                  embedding_dim=1024, expert_ffn_dim=512)
    if any(stats.get(k) != v for k,v in limits.items()):
        raise ValueError('allocation envelope only qualified for small Granite 1B/A400M; supply matching model')
    return stats


def command_plan(image, model, k0, k1, out):
    model = Path(model).resolve()
    out = output_path(out)
    if not image or not model.is_file() or model.stat().st_size > 1024**3:
        raise ValueError('supply release RUN_REF and an existing model <=1 GiB (remaining 1 GiB reserved for runtime)')
    profile = model_profile(model)
    if not 0 <= k0 < k1 <= profile['n_layer']:
        raise ValueError('need 0 <= K0 < K1 <= model layers')
    base = ['docker','run','--rm','--init','--network=none','--read-only',
            '--memory=2g','--memory-swap=2g','--pids-limit=128',
            '--tmpfs','/tmp:rw,size=64m','--stop-timeout=1']
    jobs = []
    def add(name, args, gpu=False, mounts=(), run_image=image):
        cmd = base + (['--gpus','device=0'] if gpu else []) + list(mounts)
        cmd += [run_image,'timeout','-k','1s','25s'] + args
        jobs.append(dict(name=name, argv=cmd))
    add('nvidia', ['nvidia-smi','-L'], True)
    add('lscpu', ['lscpu'])
    for backend in ('CUDA0','CPU'):
        for op in ('MUL_MAT_ID','MUL_MAT'):
            if op == 'MUL_MAT_ID':
                pattern = r'^type_a=(q4_K|q6_K|q8_0),type_b=f32,n_mats=128,n_used=8,b=0,m=768,n=1,k=2048,amax=1\.000000,m_v=0,ids_offset=0$'
            else:
                pattern = '^type_a=('+'|'.join(TYPES)+'),type_b=f32,m=4096,n=1,k=14336,bs=\\[1,1\\],nr=\\[1,1\\],per=\\[0,1,2,3\\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0$'
            add(f'{backend}-{op}', ['/app/test-backend-ops','perf','-b',backend,'-o',op,'-p',pattern], True)  # engine binary links libcuda even for -b CPU
    add('build-h2d', ['nvcc','-O2','/src/h2d.cu','-o','/out/h2d'], mounts=(
        '--mount',f'type=bind,src={ROOT}/cc/host_calibrate,dst=/src,readonly',
        '--mount',f'type=bind,src={out},dst=/out'), run_image='campaign/cuda-tools:c1')
    add('h2d', ['/probe/h2d'], True, ('--mount',f'type=bind,src={out},dst=/probe,readonly'))
    for k in (k0,k1):
        add(f'bench-{k}', ['/app/llama-bench','-m','/model/input.gguf','-ngl','999',
            '--n-cpu-moe',str(k),'-p','0','-n','64','-r','2','-t','8','-b','64','-ub','64','-o','json'],
            True, ('--mount',f'type=bind,src={model},dst=/model/input.gguf,readonly'))
    return jobs


def parse_capture(manifest, capture, stats):
    raw = {}
    for job in manifest['jobs']:
        name = job['name']
        meta = read(capture/f'{name}.status.json')
        if meta['returncode'] != 0:
            raise ValueError(f'{name} failed: {meta}')
        raw[name] = (capture/f'{name}.stdout').read_text()
    samples = {name:parse_perf(text) for name,text in raw.items() if '-MUL_MAT' in name}
    cpu = {}
    spreads = {}
    for kind in BLOCK:
        # Expert path when independently available; Q2/Q3 use dense decode screen.
        source = 'CPU-MUL_MAT_ID' if kind in ('Q4_K','Q6_K','Q8_0') else 'CPU-MUL_MAT'
        vals = [r['weight_gbs'] for r in samples[source] if r['kind']==kind]
        spreads[kind] = summary(vals)
        cpu[kind] = spreads[kind]['median']
    gpu = summary([r['weight_gbs'] for r in samples['CUDA0-MUL_MAT'] if r['kind']=='Q4_K'])
    h2d = parse_h2d(raw['h2d'])
    bw_h2d = summary([r['gbs'] for r in h2d if r['bytes']>=64*1048576])
    k0,k1 = manifest['cpu_layers']
    sys.path.insert(0,str(ROOT/'cc/moe_planner'))
    import decode_model as dm
    if not 0 <= k0 < k1 <= stats['n_layer'] or not stats.get('tensors'):
        raise ValueError('need actual tensor stats and valid layer coordinates')
    # Reject stale tensor headers rather than silently using planner's equal-layer fallback.
    tensors = [t for t in stats['tensors'] if t['role']=='expert_ffn']
    if not math.isclose(sum(t['bytes'] for t in tensors),stats['expert_bytes'],rel_tol=1e-6):
        raise ValueError('expert tensor byte totals do not match')
    a,b = [dm.traffic(stats,'static',k) for k in (k0,k1)]
    dc = sum((b['cpu_active_bytes'].get(q,0)-a['cpu_active_bytes'].get(q,0))/(positive(cpu[q])*1000)
             for q in set(a['cpu_active_bytes']) | set(b['cpu_active_bytes']))
    dg = (b['gpu_active_bytes']-a['gpu_active_bytes'])/(gpu['median']*1000)
    times = [parse_bench(raw[f'bench-{k}'],coord=k) for k in (k0,k1)]
    hs = [(v-u-dc-dg)/(k1-k0) for u,v in itertools.product(*times)]
    h = statistics.median(hs)
    if not math.isfinite(h) or h < 0:
        raise ValueError(f'negative/nonfinite residual handoff {h}; microbenchmark fit invalid; do not clamp')
    constants = dict(bw_gpu_gbs=gpu['median'],bw_cpu_gbs=cpu,bw_h2d_gbs=bw_h2d['median'],handoff_us=h,cache_miss_us=0.)
    return dict(schema='host-calibration/1', source_commit=PIN, image=manifest['image'],
                host=identity(raw['nvidia'],raw['lscpu']), constants=constants,
                raw_samples=dict(perf=samples,h2d=h2d,bench_us=times,handoff_us=hs),
                uncertainty=dict(gpu=gpu,cpu=spreads,h2d=bw_h2d,handoff=dict(min=min(hs),max=max(hs),count=len(hs),
                    note='Correlated pairwise repeat residuals; not independent samples or a CI.')),
                handoff_terms=dict(delta_cpu_us=dc,delta_gpu_us=dg,delta_layers=k1-k0),
                provenance=dict(manifest=manifest,raw_outputs=raw),
                caveats=['Microbenchmark effective bandwidth, not validated end-to-end fit.',
                    'Q2_K/Q3_K use dense GEMV, not expert MUL_MAT_ID (absent in pinned perf cases).',
                    'Q4_K dense GPU screen represents all GPU types; raw expert and other dense screens retained.',
                    'Raw pinned H2D excludes cache dispatch; cache_miss_us remains unmeasured zero.',
                    'Single shape/reruns may be cache-resident. Sample spread excludes systematic error.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode',required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--image',required=True)
    plan.add_argument('--model',required=True)
    plan.add_argument('--cpu-layers',nargs=2,type=int,required=True)
    plan.add_argument('--out',default=str(OUT/'gpu'))
    parse = sub.add_parser('parse')
    parse.add_argument('--capture',required=True)
    parse.add_argument('--stats',required=True,help='Normalized planner stats with tensors, expert_bytes, nonexpert_bytes')
    parse.add_argument('--out',default=str(OUT/'host-calibration.json'))
    args = p.parse_args()
    try:
        if args.mode == 'plan':
            out = output_path(args.out)
            jobs = command_plan(args.image,args.model,*args.cpu_layers,out)
            manifest = dict(schema='host-calibration-plan/1',image=args.image,model=str(Path(args.model).resolve()),cpu_layers=args.cpu_layers,jobs=jobs)
            emit(out/'manifest.json',manifest)
            emit(out/'commands.json',[j['argv'] for j in jobs])
            print(json.dumps([j['argv'] for j in jobs],indent=2))
        else:
            capture = Path(args.capture)
            emit(args.out,parse_capture(read(capture/'manifest.json'),capture,read(args.stats)))
    except (ValueError,KeyError,OSError) as exc:
        p.error(str(exc))

if __name__ == '__main__':
    main()
