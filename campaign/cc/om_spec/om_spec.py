#!/usr/bin/env python3
"""om_spec.py - guarded over-memory n-gram self-speculation probe on jetson-arm (adapted from x86 artifacts/spec1/probe2.py).
Starts llama-server in a campaign-labelled container (Qwen3.6-35B over-memory, -lm mmap -ngl 99 -cmoe), runs the same
watchdog as om_run.sh, sends 3 prompts (rewrite/repeat copy-heavy, chat), records timings + draft acceptance + text hash + IO.
usage: om_spec.py OUTDIR IMAGE CFG [ROUND]   CFG in base|ngram-mod"""
import json, subprocess, sys, time, urllib.request, hashlib, os, signal
OUT, IMG, CFG = sys.argv[1:4]; RND = sys.argv[4] if len(sys.argv) > 4 else "1"
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.makedirs(OUT, exist_ok=True)
MD = os.path.expanduser("~/models/campaign/qwen36-35b-a3b")
code = open(R + "/artifacts/traces/in/code.txt").read(); chat = open(R + "/artifacts/traces/in/chat.txt").read()
PROMPTS = {"rewrite": "/no_think Rewrite the following code exactly, but rename the variable `i` to `idx` everywhere. Output only code.\n\n" + code[:1500] + "\n\nRewritten code:\n",
           "repeat": "/no_think Repeat this list exactly once more:\n" + "\n".join(f"item {i}: value={i*7%13}" for i in range(40)) + "\n\nRepeated list:\n",
           "chat": chat}
EXTRA = {"base": [], "ngram-mod": ["--spec-type", "ngram-mod"]}[CFG]
NAME = f"omspec-{int(time.time())}"
def dk(cmd): return subprocess.run(["sg", "docker", "-c", "docker " + cmd], capture_output=True, text=True)
def devstat():
    for l in open("/sys/class/block/nvme0n1/stat"): f = l.split(); return int(f[0]), int(f[2])
args = " ".join(["-m /om/model.gguf -ngl 99 -cmoe -lm mmap -c 4096 --host 0.0.0.0 --port 8080 -np 1"] + EXTRA)
r = dk(f"run -d --name {NAME} --runtime nvidia --label campaign=c1 --label om=1 -p 127.0.0.1:18080:8080 -v {MD}:/om:ro --entrypoint /app/llama-server {IMG} {args}")
if r.returncode: print("START_FAIL", r.stderr); sys.exit(2)
wd = subprocess.Popen(["python3", R + "/cc/resource_observe/resource_observe.py", "watchdog", "--min-available-mib", "600", "--max-swap-used-mib", "2560",
                       "--max-swap-rate-mibs", "1000", "--interval-ms", "250", "--kill-cmd", f"sg docker -c 'docker kill {NAME}'", "--out", OUT + "/watchdog.jsonl"])
sm = subprocess.Popen(["python3", R + "/cc/resource_observe/resource_observe.py", "sample", "--backends", "io,proc,tegrastats", "--path", MD,
                       "--interval-ms", "1000", "--duration-s", "5400", "--out", OUT + "/telemetry.jsonl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
recs = []; status = "ok"
try:
    t0 = time.time()
    while True:
        try:
            if urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2).status == 200: break
        except Exception: pass
        if time.time() - t0 > 900 or NAME not in dk("ps -q --filter name=" + NAME + " --format '{{.Names}}'").stdout: raise SystemExit("server not ready")
        time.sleep(2)
    load_s = round(time.time() - t0, 1)
    for pn, p in PROMPTS.items():
        a = devstat(); ts = time.time()
        body = json.dumps({"prompt": p, "n_predict": 96, "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
        resp = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18080/completion", body, {"Content-Type": "application/json"}), timeout=1800))
        b = devstat(); t = resp.get("timings", {})
        rec = dict(round=RND, cfg=CFG, prompt=pn, load_s=load_s, wall_s=round(time.time() - ts, 1), tps=t.get("predicted_per_second"), pp_tps=t.get("prompt_per_second"),
                   prompt_n=t.get("prompt_n"), n=t.get("predicted_n"), draft_n=t.get("draft_n"), draft_acc=t.get("draft_n_accepted"),
                   nvme_gib=round((b[1] - a[1]) * 512 / 2**30, 2), avg_kib_io=round((b[1] - a[1]) * 0.5 / max(b[0] - a[0], 1)),
                   text_sha=hashlib.sha256(resp["content"].encode()).hexdigest()[:12], head=resp["content"][:60])
        print(json.dumps(rec), flush=True); recs.append(rec)
except BaseException as e:
    status = "failed: " + repr(e)[:300]
finally:
    lg = dk("logs " + NAME); open(OUT + "/server.log", "w").write(lg.stdout + lg.stderr)
    dk("rm -f " + NAME); wd.terminate(); sm.terminate()
    trips = sum(1 for l in open(OUT + "/watchdog.jsonl") if "watchdog_trip" in l) if os.path.exists(OUT + "/watchdog.jsonl") else 0
    json.dump(dict(image=IMG, cfg=CFG, round=RND, status="watchdog_killed" if trips else status, records=recs), open(OUT + "/result.json", "w"), indent=1)
    print("STATUS", status, "trips", trips)
