import json, os, subprocess, sys
from pathlib import Path
import shutil
import pytest
import resource_observe as ro

FX = os.path.join(os.path.dirname(__file__), "fixtures")
def fx(n): return open(os.path.join(FX, n)).read()
def names(ms): return {m["name"]: m for m in ms}

def test_real_fixture():
    for line in fx("tegrastats-idle-orin-nano-r36.5.txt").splitlines():
        m, u, e = ro.parse_tegrastats(line)
        d = names(m)
        assert not e
        assert d["ram_used"]["value"] == 1539 and d["ram_total"]["value"] == 7607
        assert d["ram_used"]["scope"] == "system-shared-dram" and d["ram_used"]["unit"] == "MiB"
        assert d["ram_lfb"]["value"] == 52 and d["swap_total"]["value"] == 3804 and d["swap_cached"]["value"] == 0
        assert len([k for k in d if k.endswith("_load") and k.startswith("cpu")]) == 6
        assert d["cpu4_freq"]["unit"] == "MHz"
        assert d["gpu_util"]["value"] == 0 and d["temp_tj"]["unit"] == "celsius"
        assert d["power_VDD_IN_instant"]["value"] > 4000 and d["power_VDD_IN_avg"]["unit"] == "mW"
        assert "power_VDD_CPU_GPU_CV_avg" in d
        assert any(x["name"] == "gpu_freq" for x in u) and any(x["name"] == "emc_freq" for x in u)
        assert "gpu_freq" not in d and "emc_freq" not in d  # no false values

def test_gr3d_and_emc_with_freq():
    m, u, e = ro.parse_tegrastats("RAM 1/2MB SWAP 0/1MB CPU [1%@100,off] GR3D_FREQ 50%@[612] EMC_FREQ 7%@2133")
    d = names(m)
    assert d["gpu_freq"]["value"] == 612 and d["emc_freq"]["value"] == 2133 and d["emc_util"]["value"] == 7
    assert any(x["name"] == "cpu1_load" and "offline" in x["reason"] for x in u)

def test_empty_and_garbage():
    m, u, e = ro.parse_tegrastats("")
    assert not m and e and "line_sha256_16" in e[0]
    m, u, e = ro.parse_tegrastats("complete garbage \x00")
    assert not m and e and len(e[0]["line_sha256_16"]) == 16
    assert ro.parse_tegrastats(None)[2]

def test_truncated_and_missing_ram():
    full = fx("tegrastats-idle-orin-nano-r36.5.txt").splitlines()[0]
    m, u, e = ro.parse_tegrastats(full[:60])  # cut inside SWAP
    assert "ram_used" in names(m)
    m, u, e = ro.parse_tegrastats(full.replace("RAM 1539/7607MB (lfb 13x4MB) ", ""))
    assert "ram_used" not in names(m) and e and any(x["name"] == "ram_used" for x in u)
    assert "gpu_util" in names(m)  # rest still parsed

def test_unknown_rails_and_extra_fields():
    m, u, e = ro.parse_tegrastats("RAM 1/2MB SWAP 0/1MB CPU [1%@1] GR3D_FREQ 0% NEWFIELD 5 foo@12.5C VDD_WEIRD_X9 10mW/20mW")
    d = names(m)
    assert not e and d["power_VDD_WEIRD_X9_avg"]["value"] == 20 and d["temp_foo"]["value"] == 12.5

def test_proc_fixtures(tmp_path):
    (tmp_path / "meminfo").write_text(fx("proc-meminfo-orin-nano.txt"))
    (tmp_path / "loadavg").write_text(fx("proc-loadavg-orin-nano.txt"))
    m, u = ro.proc_sample(str(tmp_path))
    d = names(m)
    assert d["mem_total"]["value"] == pytest.approx(7789756 / 1024, rel=1e-6) and d["mem_total"]["unit"] == "MiB"
    assert d["load1"]["value"] == 0.56
    assert any(x["name"] == "nvmap" for x in u)

def test_proc_missing_fields(tmp_path):
    (tmp_path / "meminfo").write_text("MemTotal: 1024 kB\nNvMapMemUsed: 2048 kB\n")
    m, u = ro.proc_sample(str(tmp_path))
    d = names(m)
    assert d["mem_total"]["value"] == 1.0 and d["meminfo_NvMapMemUsed"]["value"] == 2.0
    un = {x["name"] for x in u}
    assert {"mem_available", "swap_total", "loadavg"} <= un and "mem_available" not in d
    m, u = ro.proc_sample(str(tmp_path / "nope"))
    assert not m and u

def test_pid_and_cgroup(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text("VmHWM:\t 2048 kB\nVmRSS:\t 1024 kB\n")
    cg = tmp_path / "cg"; cg.mkdir()
    (cg / "memory.current").write_text("1048576\n"); (cg / "memory.max").write_text("max\n")
    m, u = ro.proc_sample(str(tmp_path), pid=42, cgroup=str(cg))
    d = names(m)
    assert d["pid_rss"]["value"] == 1.0 and d["pid_rss"]["scope"] == "process" and d["pid_hwm"]["value"] == 2.0
    assert d["cgroup_current"]["scope"] == "cgroup" and "accounting_note" in d["cgroup_current"]
    un = {x["name"] for x in u}
    assert "cgroup_peak" in un and "cgroup_max" in un and "cgroup_max" not in d
    m, u = ro.proc_sample(str(tmp_path), pid=999)
    assert any(x["name"] == "pid_rss" for x in u) and "pid_rss" not in names(m)

def test_no_double_count_scopes(tmp_path):
    (tmp_path / "meminfo").write_text(fx("proc-meminfo-orin-nano.txt"))
    (tmp_path / "7").mkdir(); (tmp_path / "7" / "status").write_text("VmRSS: 1024 kB\n")
    cg = tmp_path / "cg"; cg.mkdir(); (cg / "memory.current").write_text("4096")
    tm, _, _ = ro.parse_tegrastats(fx("tegrastats-idle-orin-nano-r36.5.txt").splitlines()[0])
    pm, _ = ro.proc_sample(str(tmp_path), pid=7, cgroup=str(cg))
    scopes = {x["scope"] for x in tm + pm if "ram" in x["name"] or "mem" in x["name"] or "pid" in x["name"] or "cgroup" in x["name"]}
    assert {"system-shared-dram", "process", "cgroup"} <= scopes
    # no metric aggregates across scopes
    assert not [x for x in tm + pm if x["name"] in ("mem_sum", "total_used", "ram_plus_rss")]
    assert names(tm)["ram_used"]["scope"] != names(pm)["pid_rss"]["scope"]

def test_nvml_csv_with_na():
    m, u = ro.parse_nvml_csv("1024, 8192, 37, 1500, 55, [N/A]\n", "123, python, 512\n")
    d = names(m)
    assert d["gpu_mem_used"]["value"] == 1024 and d["gpu_mem_used"]["scope"] == "device-discrete"
    assert "gpu_power" not in d and any(x["name"] == "gpu_power" for x in u)
    assert d["gpu_proc_mem_pid123"]["value"] == 512

def test_nvml_all_na_is_unavailable_on_orin():
    class R: returncode = 0; stdout = "[N/A], [N/A], [N/A], [N/A], [N/A], [N/A]\n"; stderr = ""
    m, u = ro.nvml_sample(lambda a: R())
    assert not m and any(x["name"] == "gpu_mem" for x in u)

def test_sysfs_fixture_roots(tmp_path):
    dv = tmp_path / "dv"; dv.mkdir()
    for f, v in (("cur_freq", "306000000"), ("min_freq", "306000000"), ("max_freq", "1020000000")): (dv / f).write_text(v)
    tz = tmp_path / "th" / "thermal_zone0"; tz.mkdir(parents=True)
    (tz / "type").write_text("gpu-thermal\n"); (tz / "temp").write_text("48500\n")
    m, u = ro.sysfs_sample(str(dv), str(tmp_path / "th"), nvpmodel_mode="MAXN_SUPER")
    d = names(m)
    assert d["gpu_freq_max"]["value"] == 1020 and d["thermal_thermal_zone0_gpu-thermal"]["value"] == 48.5
    assert d["nvpmodel_mode"]["value"] == "MAXN_SUPER" and not u
    m, u = ro.sysfs_sample(str(tmp_path / "none"), str(tmp_path / "none"), query_nvpmodel=False)
    assert not m and len(u) >= 3

def _write(path, recs):
    with open(path, "w") as f:
        for i, t in enumerate(recs):
            f.write(json.dumps(ro.record("proc", i, [ro.M("x", float(i), "MiB", "system", "t")], mono=t)) + "\n")

def test_summarize_gap_detection(tmp_path):
    p = tmp_path / "a.jsonl"
    _write(p, [0, 0.5, 1.0, 2.5, 3.0])
    s = ro.summarize(str(p), interval_ms=500)
    c = s["coverage"]["proc"]
    assert c["received"] == 5 and c["expected"] == 7 and len(c["gaps_gt_2x"]) == 1 and c["gaps_gt_2x"][0]["gap_s"] == 1.5
    x = s["metrics"]["proc/x"]
    assert (x["min"], x["max"], x["high_water"]) == (0.0, 4.0, 4.0) and x["mean"] == 2.0

def test_summarize_no_gaps(tmp_path):
    p = tmp_path / "b.jsonl"; _write(p, [0, 0.5, 1.0])
    assert ro.summarize(str(p), 500)["coverage"]["proc"]["gaps_gt_2x"] == []

def test_cli_sample_fake_and_summarize(tmp_path):
    out = tmp_path / "o.jsonl"
    here = os.path.dirname(__file__)
    r = subprocess.run([sys.executable, os.path.join(here, "resource_observe.py"), "sample", "--backends", "proc",
                        "--interval-ms", "100", "--duration-s", "0.3", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) >= 3 and all(x["schema"] == ro.SCHEMA and x["backend"] == "proc" for x in recs)
    assert recs[0]["seq"] == 0 and {"ts", "mono_s", "host_id", "metrics", "unavailable", "parse_errors"} <= set(recs[0])


def test_campaign_host_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ro.socket, "gethostname", lambda: "fallback-host")
    assert ro.record("proc", 0, [])['host_id'] == "fallback-host"
    path = tmp_path / ".config" / "siminfeng" / "host_id"
    path.parent.mkdir(parents=True)
    path.write_text("jetson-arm\n")
    assert ro.record("proc", 0, [])['host_id'] == "jetson-arm"
    assert ro.record("proc", 0, [], host_id="explicit")['host_id'] == "explicit"
    path.write_text("\n")
    assert ro.record("proc", 0, [])['host_id'] == "fallback-host"


def test_discovered_gpu_frequency_hz(tmp_path):
    captured = Path(__file__).parent / "tests/fixtures/gpu-devfreq"
    gpu = tmp_path / 'different-address.gpu'
    gpu.mkdir()
    for name in ("cur_freq", "min_freq", "max_freq"):
        shutil.copyfile(captured / name, gpu / name)
    assert ro.discover_gpu_devfreq(str(tmp_path)) == str(gpu)
    metrics, unavailable = ro.frequency_sample(devfreq_root=str(tmp_path))
    assert names(metrics)['gpu_freq']['value'] == int((captured / 'cur_freq').read_text())
    assert names(metrics)['gpu_freq']['unit'] == 'Hz'
    assert not unavailable
    (gpu / 'cur_freq').unlink()
    metrics, unavailable = ro.frequency_sample(devfreq_root=str(tmp_path))
    assert not metrics and unavailable[0]['name'] == 'gpu_freq'


def test_tegrastats_sampling_adds_discovered_frequency(monkeypatch):
    real = ro.frequency_sample()
    if 'gpu_freq' not in names(real[0]):
        pytest.skip('no readable GPU sysfs frequency on this host')
    m, u, e = ro.observed_tegrastats(fx('tegrastats-idle-orin-nano-r36.5.txt').splitlines()[0])
    assert names(m)['gpu_freq']['unit'] == 'Hz'
    assert not any(x['name'] == 'gpu_freq' for x in u)
