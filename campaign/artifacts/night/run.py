#!/usr/bin/env python3
"""D043 overnight runner. Phases: aa, e1, e2, e3, e4, kld. Appends JSONL to artifacts/night/results.jsonl.
Every arm records: quiet-host status, curl-running flag, peak VRAM, decode t/s, prompt ms, text sha, cache stats."""
import json, os, re, subprocess, sys, time, hashlib, threading, urllib.request, signal

ROOT = "/home/muk/projects/siminfeng/campaign"
OUT = f"{ROOT}/artifacts/night"
RES = f"{OUT}/results.jsonl"
TIN = f"{ROOT}/artifacts/traces/in"
ENG = {}  # filled from build records
ENG["rel1"] = "sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e"
for tag in ("up0", "up1"):
    p = f"{ROOT}/builds/results/{tag}/build-record.json"
    if os.path.exists(p):
        r = json.load(open(p))
        ref = (r.get("image") or {}).get("run_ref")
        if r.get("status") in ("success", "passed") and ref:
            ENG[tag] = ref
QWEN = ("/home/muk/models/campaign/qwen3-30b-a3b", "Qwen3-30B-A3B-Q4_K_M.gguf")
REAP = ("/home/muk/models/campaign/qwen3-reap15b", "Qwen3-30B-A3B-Q4_K_M.gguf")
Q36 = ("/home/muk/models/campaign/qwen36-mtp", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
BUSY = r"[c]c1plus|[n]vcc |[c]make --build|[c]odex exec"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def busy():
    r = sh(["pgrep", "-af", BUSY])
    return [l for l in r.stdout.splitlines() if "pgrep" not in l and str(os.getpid()) not in l.split()[0]]


def curl_pids():
    r = sh(["pgrep", "-f", r"[c]url -sS -L --fail"])
    return [int(x) for x in r.stdout.split()]


def wait_quiet(limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if not busy():
            return True
        time.sleep(15)
    return False


class Peak:
    def __init__(self):
        self.peak = 0; self.stop = False
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        while not self.stop:
            try:
                m = int(sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).stdout.strip().splitlines()[0])
                self.peak = max(self.peak, m)
            except Exception:
                pass
            time.sleep(0.5)

    def done(self):
        self.stop = True; self.t.join(); return self.peak


def record(rec):
    with open(RES, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def docker_base(name, engine, model, entry, extra_mounts=()):
    return ["docker", "run", "--rm", "--name", name, "--label", "campaign=c1", "--gpus", "all",
            "-v", f"{model[0]}:/m:ro", "-v", f"{TIN}:/tin:ro", *extra_mounts, "--entrypoint", entry, ENG[engine]]


def completion(exp, arm, engine, model, args, prompt, n=128, rnd=0, extra=None):
    quiet = wait_quiet()
    name = f"night-{os.getpid()}"
    cmd = docker_base(name, engine, model, "/app/llama-completion") + [
        "-m", f"/m/{model[1]}", "-f", f"/tin/{prompt}", "-n", str(n), "-c", "4096", "--temp", "0", "-s", "1",
        "-fa", "on", "-no-cnv", "--no-warmup", "-lv", "4"] + args
    pk = Peak(); t0 = time.time()
    r = sh(cmd, timeout=900)
    peak = pk.done()
    err = r.stderr
    tps = re.findall(r"eval time =\s+[\d.]+ ms /\s+\d+ runs\s+\(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)", err)
    pp = re.findall(r"prompt eval time =\s+([\d.]+) ms", err)
    cache = re.findall(r"llama_moe_cache: (.*)", err)
    tag = f"{exp}-{arm}-{prompt.split('.')[0]}-r{rnd}"
    open(f"{OUT}/logs/{tag}.err", "w").write(err)
    rec = dict(exp=exp, arm=arm, engine=engine, model=model[1], prompt=prompt, round=rnd, args=args, rc=r.returncode,
               tps=float(tps[-1]) if tps else None, prompt_ms=float(pp[-1]) if pp else None, peak_mib=peak,
               text_sha=hashlib.sha256(r.stdout.encode()).hexdigest()[:8], cache=cache[-3:], quiet=quiet,
               curl_running=bool(curl_pids()), wall_s=round(time.time() - t0, 1), t=time.strftime("%H:%M:%S"))
    if extra: rec.update(extra)
    record(rec)
    return rec


# ---------- phases ----------
K_B = {4: 42, 6: 36, 8: 30, 10: 24}
BASE_MIB = 1800  # all-CPU-experts footprint measured in D039 (K=48: 1798 MiB)


def phase_aa():
    pids = curl_pids()
    for i in range(3):
        completion("aa", "with_dl", "rel1", QWEN, ["-ngl", "99", "--n-cpu-moe", "30"], "chat.txt", rnd=i)
    for p in pids: os.kill(p, signal.SIGSTOP)
    try:
        for i in range(3):
            completion("aa", "dl_paused", "rel1", QWEN, ["-ngl", "99", "--n-cpu-moe", "30"], "chat.txt", rnd=i)
    finally:
        for p in pids: os.kill(p, signal.SIGCONT)


def e1_arms(B, cache_mib):
    arms = [("S1", "rel1", ["-ngl", "99", "--n-cpu-moe", str(K_B[B])]),
            ("C1", "rel1", ["-ngl", "99", "--n-cpu-moe", "48", "--moe-cache-mib", str(cache_mib)])]
    if "up1" in ENG:
        arms += [("S2", "up1", ["-ngl", "99", "--n-cpu-moe", str(K_B[B])]),
                 ("C2", "up1", ["-ngl", "99", "--n-cpu-moe", "48", "--moe-cache-mib", str(cache_mib), "--moe-cache-cpu-share", "0"]),
                 ("C3", "up1", ["-ngl", "99", "--n-cpu-moe", "48", "--moe-cache-mib", str(cache_mib), "--moe-cache-cpu-share", "0.9"])]
    return arms


def phase_e1(budgets=(4, 6, 8, 10)):
    for B in budgets:
        # calibrate the static peak first, then size the cache to static peak - base - margin
        s = completion("e1cal", "S1", "rel1", QWEN, ["-ngl", "99", "--n-cpu-moe", str(K_B[B])], "chat.txt", n=16, extra={"B": B})
        cache_mib = max(256, int(s["peak_mib"] - BASE_MIB - 300))
        arms = e1_arms(B, cache_mib)
        for rnd in (1, 2):
            order = arms if rnd == 1 else list(reversed(arms))
            for arm, eng, args in order:
                for prompt in ("chat.txt", "code.txt"):
                    completion("e1", arm, eng, QWEN, args, prompt, rnd=rnd, extra={"B": B, "cache_mib": cache_mib})


def server_run(exp, arm, engine, model, sargs, prompts, n=256, rnd=0, extra=None):
    quiet = wait_quiet()
    name = f"night-srv-{os.getpid()}"
    sh(["docker", "rm", "-f", name])
    cmd = ["docker", "run", "-d", "--name", name, "--label", "campaign=c1", "--gpus", "all", "-p", "127.0.0.1:18081:8080",
           "-v", f"{model[0]}:/m:ro", "--entrypoint", "/app/llama-server", ENG[engine], "-m", f"/m/{model[1]}",
           "-fa", "on", "-c", "8192", "--host", "0.0.0.0", "--port", "8080", "-np", "1", "-lv", "4"] + sargs
    pk = Peak()
    r = sh(cmd)
    ok = False
    for _ in range(400):
        try:
            if urllib.request.urlopen("http://127.0.0.1:18081/health", timeout=2).status == 200:
                ok = True; break
        except Exception:
            pass
        time.sleep(1)
    out = []
    if ok:
        for pn, p in prompts.items():
            body = json.dumps({"prompt": p, "n_predict": n, "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
            try:
                j = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18081/completion", body,
                                                                            {"Content-Type": "application/json"}), timeout=900))
            except Exception as e:
                record(dict(exp=exp, arm=arm, engine=engine, prompt=pn, round=rnd, error=str(e)[:300])); continue
            t = j.get("timings", {})
            rec = dict(exp=exp, arm=arm, engine=engine, model=model[1], prompt=pn, round=rnd, sargs=sargs,
                       tps=t.get("predicted_per_second"), prompt_ms=t.get("prompt_ms"), draft_n=t.get("draft_n", 0),
                       draft_acc=t.get("draft_n_accepted", 0), n=t.get("predicted_n"),
                       text_sha=hashlib.sha256(j["content"].encode()).hexdigest()[:8], quiet=quiet,
                       curl_running=bool(curl_pids()), t=time.strftime("%H:%M:%S"))
            if extra: rec.update(extra)
            out.append(rec)
    logs = sh(["docker", "logs", name]).stdout + sh(["docker", "logs", name]).stderr
    open(f"{OUT}/logs/{exp}-{arm}-r{rnd}-server.log", "w").write(logs)
    sh(["docker", "rm", "-f", name])
    peak = pk.done()
    for rec in out:
        rec["peak_mib"] = peak; record(rec)
    if not ok:
        record(dict(exp=exp, arm=arm, engine=engine, round=rnd, error="server not ready", rc=r.returncode, stderr=r.stderr[-300:]))
    return out


def spec_prompts():
    code = open(f"{TIN}/code.txt").read()[:3000]
    return {"rewrite": "/no_think Rewrite the following code exactly, but rename the variable `i` to `idx` everywhere. Output only code.\n\n" + code + "\n\nRewritten code:\n",
            "repeat": "/no_think Repeat this list exactly once more:\n" + "\n".join(f"item {i}: value={i*7%13}" for i in range(60)) + "\n\nRepeated list:\n",
            "chat": open(f"{TIN}/chat.txt").read()}


def phase_e2(best_cache_args=None):
    P = spec_prompts()
    places = [("K30", "rel1", ["-ngl", "99", "--n-cpu-moe", "30"])]
    if best_cache_args:
        places.append(("cache", best_cache_args[0], best_cache_args[1]))
    arms = []
    for pl, eng, a in places:
        arms += [(f"{pl}-none", eng, a), (f"{pl}-ngram", eng, a + ["--spec-type", "ngram-mod"])]
    for rnd in (1, 2):
        for arm, eng, a in (arms if rnd == 1 else list(reversed(arms))):
            server_run("e2", arm, eng, QWEN, a, P, rnd=rnd)


def phase_e3():
    for rnd in (1, 2):
        for arm, model, a in [("reap-K0", REAP, ["-ngl", "99"]), ("reap-K12", REAP, ["-ngl", "99", "--n-cpu-moe", "12"]),
                              ("full-K30", QWEN, ["-ngl", "99", "--n-cpu-moe", "30"])][:: 1 if rnd == 1 else -1]:
            for prompt in ("chat.txt", "code.txt"):
                completion("e3", arm, "rel1", model, a, prompt, rnd=rnd)


def phase_e4(engine="up1"):
    P = spec_prompts()
    for K, B in ((18, 12), (30, 8)):
        arms = [(f"K{K}-none", ["-ngl", "99", "--n-cpu-moe", str(K)]),
                (f"K{K}-mtp2", ["-ngl", "99", "--n-cpu-moe", str(K), "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
                (f"K{K}-mtp3", ["-ngl", "99", "--n-cpu-moe", str(K), "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])]
        for rnd in (1, 2):
            for arm, a in (arms if rnd == 1 else list(reversed(arms))):
                server_run("e4", arm, engine, Q36, a, P, rnd=rnd, extra={"B": B})


def ppl(exp, arm, engine, model, args, kld_base=None, kld_save=None, chunks=8):
    quiet = wait_quiet()
    mounts = ("-v", "/home/muk/models/campaign/data/wikitext/wikitext-2-raw:/w:ro", "-v", f"{OUT}/kl:/kl")
    cmd = docker_base(f"night-ppl-{os.getpid()}", engine, model, "/app/llama-perplexity", mounts) + [
        "-m", f"/m/{model[1]}", "-f", "/w/wiki.test.raw", "--chunks", str(chunks), "-c", "512", "-fa", "on", "-lv", "4"] + args
    if kld_save: cmd += ["--kl-divergence-base", f"/kl/{kld_save}"]
    if kld_base: cmd += ["--kl-divergence-base", f"/kl/{kld_base}", "--kl-divergence"]
    pk = Peak(); r = sh(cmd, timeout=3600); peak = pk.done()
    txt = r.stdout + r.stderr
    open(f"{OUT}/logs/{exp}-{arm}-ppl.log", "w").write(txt)
    g = lambda pat: (re.findall(pat, txt) or [None])[-1]
    rec = dict(exp=exp, arm=arm, engine=engine, model=model[1], args=args, rc=r.returncode, peak_mib=peak, quiet=quiet,
               ppl=g(r"Final estimate: PPL = ([\d.]+)"), mean_kld=g(r"Mean\s+KLD:\s+([\d.]+)"),
               same_top=g(r"Same top p:\s+([\d.]+)"), t=time.strftime("%H:%M:%S"))
    record(rec)
    return rec


def phase_kld():
    os.makedirs(f"{OUT}/kl", exist_ok=True)
    ppl("kld", "ref-S1-K18", "rel1", QWEN, ["-ngl", "99", "--n-cpu-moe", "18"], kld_save="qwen-k18.kl")
    for arm, eng, a in [("S1-K42", "rel1", ["-ngl", "99", "--n-cpu-moe", "42"]),
                        ("C1-6000", "rel1", ["-ngl", "99", "--n-cpu-moe", "48", "--moe-cache-mib", "6000"])] + \
                       ([("C3-6000", "up1", ["-ngl", "99", "--n-cpu-moe", "48", "--moe-cache-mib", "6000", "--moe-cache-cpu-share", "0.9"])] if "up1" in ENG else []):
        ppl("kld", arm, eng, QWEN, a, kld_base="qwen-k18.kl")


if __name__ == "__main__":
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    print("engines:", ENG, flush=True)
    for ph in sys.argv[1:]:
        globals()[f"phase_{ph}"]()
