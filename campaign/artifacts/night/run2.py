#!/usr/bin/env python3
"""D047 overnight-2 runner (X1-X8). Reuses D043 run.py helpers; appends to artifacts/night/results.jsonl."""
import os, re, sys, runpy
m = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), run_name="night")
completion, server_run, ppl, record, QWEN, Q36, OUT, ENG = (m[k] for k in ("completion", "server_run", "ppl", "record", "QWEN", "Q36", "OUT", "ENG"))
os.makedirs(f"{OUT}/logs", exist_ok=True); os.makedirs(f"{OUT}/kl", exist_ok=True)
SP = m["spec_prompts"]()
CHAT36 = open(f"{m['TIN']}/chat36.txt").read()
P36 = {"chat": CHAT36, "rewrite": SP["rewrite"], "repeat": SP["repeat"]}
G = lambda K: ["-ngl", "99", "--n-cpu-moe", str(K)]
MTP3 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
def C(mib, share, ncpu=48): return ["-ngl", "99", "--n-cpu-moe", str(ncpu), "--moe-cache-mib", str(mib), "--moe-cache-cpu-share", str(share)]
def abba(arms, fn):
    for rnd in (1, 2):
        for a in (arms if rnd == 1 else arms[::-1]): fn(a, rnd)

def x1():
    arms = [("K28-none", G(28)), ("K30-mtp3", G(30) + MTP3)]
    abba(arms, lambda a, r: server_run("x1", a[0], "up1", Q36, a[1], P36, rnd=r))

def x2():
    ppl("x2", "ref-K18", "up1", Q36, G(18), kld_save="q36-k18.kl")
    for ub in (1, 4):
        ppl("x2", f"K18-ub{ub}", "up1", Q36, G(18), kld_base="q36-k18.kl", ub=ub)

def x3():
    base = completion("x3cal", "base-K41", "up1", Q36, G(41), "chat36.txt", n=16)["peak_mib"]
    for K in (30, 36):
        sp = completion("x3cal", f"S-K{K}", "up1", Q36, G(K), "chat36.txt", n=16)["peak_mib"]
        cmib = max(256, int(sp - base - 300))
        arms = [("S", G(K)), ("C2", C(cmib, 0, 41)), ("C3", C(cmib, 0.9, 41))]
        def run(a, r):
            for pr in ("chat36.txt", "code.txt"):
                completion("x3", a[0], "up1", Q36, a[1] + ["--ignore-eos"], pr, rnd=r, extra={"K": K, "cache_mib": cmib, "base_mib": base, "static_cal_mib": sp})
        abba(arms, run)

def x4():
    for mib in (6093, 8137):
        arms = [(f"share{s}", C(mib, s)) for s in (0, 0.3, 0.6, 0.9)]
        def run(a, r):
            for pr in ("chat.txt", "code.txt"):
                completion("x4", a[0], "up1", QWEN, a[1], pr, rnd=r, extra={"cache_mib": mib})
        abba(arms, run)

def x5():
    arms = [(f"{pl}-t{t}", eng, args + ["-t", str(t)]) for pl, eng, args in (("S30", "rel1", G(30)), ("C3", "up1", C(6093, 0.9))) for t in (4, 6, 8, 12, 16)]
    abba(arms, lambda a, r: completion("x5", a[0], a[1], QWEN, a[2], "chat.txt", rnd=r))

def x6():
    arms = [(f"{pl}-ub{ub}", eng, args + ["-b", "2048", "-ub", str(ub)]) for pl, eng, args in (("S30", "rel1", G(30)), ("C3", "up1", C(6093, 0.9))) for ub in (256, 512, 1024, 2048)]
    abba(arms, lambda a, r: completion("x6", a[0], a[1], QWEN, a[2], "code.txt", rnd=r))

def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)

def x7():
    path = os.path.join(*QWEN)
    conds = (("warm-mmap", False, []), ("cold-mmap", True, []), ("warm-nommap", False, ["--no-mmap"]), ("cold-nommap", True, ["--no-mmap"]))
    if os.environ.get("X7_A1"):
        conds = (("warm-lmnone", False, ["-lm", "none"]), ("cold-lmnone", True, ["-lm", "none"]), ("warm-mlock", False, ["-lm", "mmap+mlock"]), ("cold-mlock", True, ["-lm", "mmap+mlock"]))
    for cond, cold, extra in conds:
        for i in range(3):
            if cold: evict(path)
            elif i == 0: completion("x7warmup", cond, "rel1", QWEN, G(30) + extra, "chat.txt", n=1)
            rec = completion("x7", cond, "rel1", QWEN, G(30) + extra, "chat.txt", n=1, rnd=i)
            err = open(f"{OUT}/logs/x7-{cond}-chat-r{i}.err").read()
            lt = re.findall(r"load time =\s+([\d.]+) ms", err)
            record(dict(exp="x7load", arm=cond, round=i, wall_s=rec["wall_s"], load_ms=float(lt[-1]) if lt else None, cold=cold, file_gb=round(os.path.getsize(path) / 1e9, 2)))

def x8():
    recs = [r for r in map(__import__("json").loads, open(m["RES"])) if r.get("exp") == "x3" and r.get("K") == 30 and r.get("arm") == "C3"]
    cmib = recs[0]["cache_mib"] if recs else 4500
    arms = [("C3-none", C(cmib, 0.9, 41)), ("C3-mtp3", C(cmib, 0.9, 41) + MTP3), ("K30-mtp3", G(30) + MTP3)]
    exp = "x8a1" if os.environ.get("X8_A1") else "x8"
    def run(a, r):
        mem = m["sh"](["free", "-m"]).stdout.split("\n")[1:3]
        server_run(exp, a[0], "up1", Q36, a[1], P36, rnd=r, extra={"cache_mib": cmib, "free_m": mem})
    abba(arms, run)

if __name__ == "__main__":
    print("engines:", ENG, flush=True)
    for ph in sys.argv[1:]:
        print(f"== {ph}", flush=True)
        try: globals()[ph]()
        except Exception as e:
            import traceback; traceback.print_exc(); record(dict(exp=ph, error=repr(e)[:300]))
