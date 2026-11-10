#!/usr/bin/env python3
"""Indicative shared-host CPU quantization screen, pinned CPU-only llama-bench."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'artifacts/codexPL4'


def load_snapshot():
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                loadavg=Path('/proc/loadavg').read_text().strip(),
                cpu_pressure=Path('/proc/pressure/cpu').read_text().strip(),
                cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                cpu_stat=Path('/proc/stat').read_text().splitlines()[0])


def checked_out(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError('screen output must stay under artifacts/codexPL4')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', default=str(OUT/'tree/build/bin/llama-bench'))
    parser.add_argument('--out', default=str(OUT/'cpu-screen'))
    parser.add_argument('--models', nargs='+', help='GGUF paths; defaults to available Granite expert-type variants')
    parser.add_argument('--timeout', type=float, default=300.)
    args = parser.parse_args(argv)
    out = checked_out(args.out); out.mkdir(parents=True, exist_ok=True)
    bench = Path(args.bench).resolve()
    cache = OUT/'tree/build/CMakeCache.txt'
    if not bench.is_relative_to(OUT/'tree') or not cache.is_file() or 'GGML_CUDA:BOOL=OFF' not in cache.read_text():
        parser.error('requires the assignment scratch CPU-only build')
    sys.path.insert(0,str(ROOT/'cc/model_inspect'))
    from inspect_model import inspect
    defaults = list((Path.home()/'models/campaign/gguf').glob('*mix*.gguf'))
    q8 = Path.home()/'models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf'
    defaults += [q8] if q8.is_file() else []
    defaults += sorted((ROOT/'artifacts/codexPAP/models').glob('pap24_*.gguf'))
    results = []
    for index,path in enumerate(map(Path,args.models or sorted(defaults))):
        stats = inspect(path)
        types = sorted({t['ggml_type'] for t in stats['tensors'] if t['role']=='expert_ffn'})
        if stats['arch'] != 'granitemoe' or len(types)!=1:
            raise ValueError(f'need Granite with one expert quant type: {path}')
        kind = types[0]
        prefix = out/f'{index:02d}-{path.stem}'
        cmd = ['nice','-n','10',str(bench),'-m',str(path),'-ngl','0','-p','0','-n','32','-r','3','-t','8','-o','json']
        before = load_snapshot()
        with prefix.with_suffix('.stdout').open('w') as stdout, prefix.with_suffix('.stderr').open('w') as stderr:
            process = subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8'})
            samples = []
            start = time.monotonic()
            while process.poll() is None:
                samples.append(load_snapshot())
                if time.monotonic()-start > args.timeout:
                    process.kill(); process.wait(); break
                time.sleep(1)
        after = load_snapshot()
        result = dict(model_path=str(path.resolve()), model_size_bytes=path.stat().st_size,
                      type=kind, command=cmd, returncode=process.returncode,
                      load_before=before, load_after=after, load_samples=samples,
                      stdout=str(prefix.with_suffix('.stdout').relative_to(ROOT)),
                      stderr=str(prefix.with_suffix('.stderr').relative_to(ROOT)))
        if process.returncode == 0:
            raw = json.loads(prefix.with_suffix('.stdout').read_text())
            if len(raw)!=1 or raw[0]['n_gpu_layers']!=0 or raw[0]['n_gen']!=32:
                raise ValueError('unexpected benchmark output')
            benchrow = raw[0]
            expert_active = stats['totals']['real_expert_bytes']*stats['n_expert_used']/stats['n_expert']
            active = expert_active+stats['totals']['non_expert_bytes_including_shared']
            result.update(decode_tps=benchrow['avg_ts'], samples_tps=benchrow['samples_ts'],
                          active_bytes_per_token=active, expert_active_bytes_per_token=expert_active,
                          effective_gbs=active*benchrow['avg_ts']/1e9,
                          expert_only_gbs=expert_active*benchrow['avg_ts']/1e9,
                          bench=benchrow)
        results.append(result)
        print(json.dumps({k:result.get(k) for k in ('type','model_path','returncode','decode_tps','effective_gbs')}),flush=True)
    import statistics
    constants = {kind:statistics.median(r['effective_gbs'] for r in results if r['type']==kind and r['returncode']==0)
                 for kind in sorted({r['type'] for r in results if r['returncode']==0})}
    report = dict(schema='cpu-type-screen/1', indicative=True, results=results, effective_gbs=constants,
                  benchmark_sha256=hashlib.sha256(bench.read_bytes()).hexdigest(),
                  reference_commit='31288c9c6',
                  caveat='Whole-model CPU decode, including nonexperts; effective bytes/s is an end-to-end proxy, not isolated expert GEMV bandwidth. Shared sandbox host and load affect timings. Does not replace fitted mixed GPU/CPU constants.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    cal_path = HERE/'calibration.json'; cal = json.loads(cal_path.read_text())
    physical = cal.setdefault('physical_decode',{})
    physical['cpu_type_screen'] = dict(report=str((out/'report.json').relative_to(ROOT)), indicative=True,
                                       effective_gbs=constants, caveat=report['caveat'])
    physical['screen_bw_cpu_gbs'] = constants
    physical.setdefault('constants',{})['screen_bw_cpu_gbs'] = constants
    cal_path.write_text(json.dumps(cal,indent=2)+'\n')
    return 0 if results and all(r['returncode']==0 for r in results) else 1


if __name__ == '__main__': raise SystemExit(main())
