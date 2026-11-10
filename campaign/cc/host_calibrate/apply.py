#!/usr/bin/env python3
"""Produce an independent planner calibration object; never edits bundled calibration."""
import argparse
import copy
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from calibrate import ROOT, emit, read


def apply(host, base):
    if host['schema'] != 'host-calibration/1':
        raise ValueError('expected host-calibration/1')
    c = host['constants']
    for k in ('bw_gpu_gbs','bw_h2d_gbs','handoff_us'):
        if not math.isfinite(c[k]) or c[k] < 0 or (k != 'handoff_us' and c[k] == 0):
            raise ValueError(f'invalid {k}')
    if 'Q4_K' not in c['bw_cpu_gbs'] or any(not math.isfinite(v) or v<=0 for v in c['bw_cpu_gbs'].values()):
        raise ValueError('invalid CPU bandwidths')
    result = copy.deepcopy(base)
    physical = result['physical_decode']
    physical['constants'] = copy.deepcopy(c)
    physical.pop('screen_bw_cpu_gbs',None)
    physical['host_calibration'] = host
    # Old fit residuals do not describe the new host constants.
    physical['evaluation'] = {'summary':{}}
    physical['leave_one_model_out'] = {}
    physical['report'] = 'host-calibration/1: microbenchmark; end-to-end residuals unavailable'
    physical['host_id'] = host['host']['id']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host_calibration')
    p.add_argument('--base',default=str(ROOT/'cc/moe_planner/calibration.json'))
    p.add_argument('--out',default=str(ROOT/'artifacts/codexCAL/planner-calibration.json'))
    a = p.parse_args()
    try:
        emit(a.out,apply(read(a.host_calibration),read(a.base)))
    except (ValueError,KeyError,OSError) as e:
        p.error(str(e))

if __name__ == '__main__': main()
