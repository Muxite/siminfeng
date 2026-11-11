#!/usr/bin/env python3
"""PREDICTION (not a measurement): expected decode t/s vs GPU memory for an expert-granular hit/miss cache, from
 (a) cache hit rates replayed from real Granite routing traces (experiments/routing/*.json; cold-start LRU/LFU, decode phase)
 (b) per-expert CPU-miss cost and per-slot GPU bytes CALIBRATED from the measured static --n-cpu-moe sweep (job-36b7630119, engine b0, mixE4K).
Assumptions (all unverified at expert granularity): miss cost linear in missed experts, no PCIe transfer (misses computed on CPU), CPU and GPU parts serialized (upper bound on time) or overlapped (lower bound)."""
import json, sys
tg0, tg24 = 704.0, 128.0                       # measured: all experts on GPU / all 24 layers' experts on CPU
gpu0, gpu24 = 1218.0, 576.0                    # measured GPU MiB
t0 = 1000 / tg0                                 # ms per token, all on GPU
extra_all = 1000 / tg24 - t0                    # ms extra per token with ALL experts on CPU
miss_cost = extra_all / (24 * 8)                # ms per missed expert activation
mib_per_slot = (gpu0 - gpu24) / (24 * 32)       # MiB per resident expert (per layer-slot)
static = {6: (1056, 336), 12: (894, 212), 18: (734, 161), 24: (576, 128)}
print(f"calibration: miss cost {miss_cost*1000:.1f} us/expert, {mib_per_slot:.3f} MiB/expert-slot, base token {t0:.2f} ms")
print("static --n-cpu-moe (measured): " + ", ".join(f"n{n}: {m} MiB {t} t/s" for n, (m, t) in static.items()))
rows = []
for wl in ("prose", "code", "chat"):
    d = json.load(open(f"/home/muk/projects/siminfeng/campaign/experiments/routing/{wl}.json"))["phases"]["decode"]["layers"]
    for pol in ("LRU", "LFU"):
        for cap in (4, 8, 12, 16, 24):
            h = sum(l["cache"][pol][str(cap)]["hits"] for l in d.values()); r = sum(l["cache"][pol][str(cap)]["requests"] for l in d.values())
            hit = h / r; gib = gpu24 + 24 * cap * mib_per_slot
            miss_ms = 24 * 8 * (1 - hit) * miss_cost
            ser = 1000 / (t0 + miss_ms)                           # CPU misses serialized after GPU work
            ovl = 1000 / max(t0, miss_ms) if miss_ms > 0 else tg0   # perfect CPU/GPU overlap (upper bound)
            rows.append((wl, pol, cap, hit, gib, ser, ovl))
print("\nPREDICTION (decode, cold-start hit rates; NOT measured at expert granularity)")
print("%-6s %-4s %5s %8s %9s %12s %12s   | static layer-offload at similar MiB (measured)" % ("trace", "pol", "slots", "hit", "GPU MiB", "tg serial", "tg overlap"))
def near(m):
    return min(static.items(), key=lambda kv: abs(kv[1][0] - m))
for wl, pol, cap, hit, gib, ser, ovl in rows:
    n, (sm, st) = near(gib)
    print("%-6s %-4s %5d %7.1f%% %9.0f %10.0f %12.0f   | n%d: %d MiB %d t/s" % (wl, pol, cap, hit * 100, gib, ser, ovl, n, sm, st))
json.dump([dict(trace=a, policy=b, slots=c, hit=d_, gpu_mib=e, tg_serial=f, tg_overlap=g) for a, b, c, d_, e, f, g in rows], open("/home/muk/projects/siminfeng/campaign/experiments/routing/prediction.json", "w"), indent=1)
