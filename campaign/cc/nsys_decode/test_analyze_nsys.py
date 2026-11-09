import importlib.util
from pathlib import Path
import sqlite3

import pytest


spec = importlib.util.spec_from_file_location('analyze_nsys', Path(__file__).with_name('analyze_nsys.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def trace(tmp_path):
    path = tmp_path / 'tiny.sqlite'
    c = sqlite3.connect(path)
    c.executescript('''
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000'),
          (2,'cudaStreamSynchronize_v3020'),(3,'cudaMemcpyAsync_v3020'),
          (4,'visible_kernel');
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'HtoD'),(2,'DtoH');
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
          start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(
          start INTEGER,end INTEGER,graphExecId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
          start INTEGER,end INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
          start INTEGER,end INTEGER,nameId INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(9000,10000,2,16,0);
    ''')
    # Four identical 100us tokens. Graph 10..50 overlaps a kernel 20..30
    # and H2D 40..60: compute union=40us, transfer-only=11us (incl logits).
    # Idle=49us; sync covers 60..80=20us and memcpy API 98..99=1us.
    for i in range(4):
        a = 10000 + i * 100000
        corr = 10 + i * 3
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(?,?,7)', (a+10000,a+50000))
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,4)', (a+20000,a+30000))
        c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?)',
                      [(a+40000,a+60000,1,8,corr), (a+99000,a+100000,2,16,corr+1)])
        c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)',
                      [(a+5000,a+6000,1,corr+2), (a+45000,a+80000,2,corr+2),
                       (a+39000,a+41000,3,corr), (a+98000,a+99000,3,corr+1)])
    c.commit()
    c.close()
    return path


def test_decode_accounting_and_overlap(trace):
    result = mod.analyze(trace, warmup=1, marker_bytes=16)
    assert result['token_count'] == 3
    assert result['marker_count'] == 5
    values = {k:v['mean'] for k,v in result['metrics'].items()}
    for key, expected in dict(wall_us=100, compute_envelope_us=40,
                              transfer_only_us=11, gpu_busy_us=51, gpu_idle_us=49,
                              idle_sync_copy_api_us=21, idle_other_host_us=28,
                              graph_launches=1, explicit_kernel_records=1,
                              idle_gap_count=2, d2h_bytes=16,h2d_bytes=8).items():
        assert values[key] == pytest.approx(expected)
    assert sum(values[k] for k in ('compute_envelope_us','transfer_only_us',
                                  'idle_sync_copy_api_us','idle_other_host_us')) == 100
    assert result['steady_top_explicit_kernels'][0]['total_us'] == 30
    assert {x['bytes']:x['matched_api_calls_per_token'] for x in result['copies']} == {8:1,16:1}
    # Reopen without writes; no sidecars or source modification required.
    assert mod.analyze(trace, warmup=0, marker_bytes=16)['token_count'] == 4


def test_graph_only_export_and_missing_markers(trace):
    c = sqlite3.connect(trace)
    c.execute('DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL')
    c.commit()
    c.close()
    result = mod.analyze(trace,warmup=1,marker_bytes=16)
    assert result['steady_top_explicit_kernels'] == []
    assert result['metrics']['compute_envelope_us']['mean'] == 40
    with pytest.raises(ValueError,match='insufficient'):
        mod.analyze(trace,warmup=1,marker_bytes=1234)


def test_interval_clipping_and_cross_boundary_api(trace):
    c = sqlite3.connect(trace)
    # A sync straddles the first retained boundary: count only at its start,
    # but clip its time on both sides. It must not be lost by start-only slicing.
    c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(109000,112000,2,999)')
    c.commit()
    c.close()
    result = mod.analyze(trace,warmup=1,marker_bytes=16)
    first = result['tokens'][0]
    assert first['idle_sync_copy_api_us'] == 23
    assert first['idle_other_host_us'] == 26
    sync = next(x for x in result['apis'] if 'Synchronize' in x['name'])
    assert sync['calls_per_token'] == 1
    assert sync['us_per_token'] == pytest.approx(35 + 2/3)
