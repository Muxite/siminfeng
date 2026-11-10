#!/usr/bin/env python3
# Indicative probe: n-gram self-speculation on Qwen3-30B-A3B via llama-server. Records timings + draft acceptance.
import json, subprocess, sys, time, urllib.request, hashlib
R="sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e"
D="/home/muk/projects/siminfeng/campaign/artifacts/spec1"
code=open("/home/muk/projects/siminfeng/campaign/artifacts/traces/in/code.txt").read()
chat=open("/home/muk/projects/siminfeng/campaign/artifacts/traces/in/chat.txt").read()
PROMPTS={"rewrite": "/no_think Rewrite the following code exactly, but rename the variable `i` to `idx` everywhere. Output only code.\n\n"+code[:3000]+"\n\nRewritten code:\n",
         "chat": chat, "repeat": "/no_think Repeat this list exactly once more:\n"+"\n".join(f"item {i}: value={i*7%13}" for i in range(60))+"\n\nRepeated list:\n"}
CFGS={"base":[], "ngram-mod":["--spec-type","ngram-mod"], "ngram-map-k":["--spec-type","ngram-map-k"]}
def server(name,extra):
    subprocess.run(["docker","rm","-f","spec1"],capture_output=True)
    cmd=["docker","run","-d","--name","spec1","--label","campaign=c1","--gpus","all","-p","127.0.0.1:18080:8080",
         "-v","/home/muk/models/campaign/qwen3-30b-a3b:/m:ro","--entrypoint","/app/llama-server",R,
         "-m","/m/Qwen3-30B-A3B-Q4_K_M.gguf","-ngl","99","--n-cpu-moe","30","-fa","on","-c","8192","--host","0.0.0.0","--port","8080","-np","1"]+extra
    subprocess.run(cmd,check=True,capture_output=True)
    for _ in range(240):
        try:
            if urllib.request.urlopen("http://127.0.0.1:18080/health",timeout=2).status==200: return
        except Exception: pass
        time.sleep(1)
    raise SystemExit("server not ready "+name)
def ask(p):
    body=json.dumps({"prompt":p,"n_predict":256,"temperature":0,"seed":1,"cache_prompt":False}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18080/completion",body,{"Content-Type":"application/json"}),timeout=600))
    return r
out=[]
for rnd in (1,):
  for name,extra in (CFGS.items() if rnd==1 else reversed(list(CFGS.items()))):
    server(name,extra)
    for pn,p in PROMPTS.items():
        r=ask(p); t=r.get("timings",{})
        rec={"round":rnd,"cfg":name,"prompt":pn,"tps":t.get("predicted_per_second"),"pp_ms":t.get("prompt_ms"),
             "draft_n":t.get("draft_n"),"draft_acc":t.get("draft_n_accepted"),"n":t.get("predicted_n"),
             "draft_n2":(r.get("timings") or {}).get("draft_n",0),"head":r["content"][:40],"text_sha":hashlib.sha256(r["content"].encode()).hexdigest()[:8]}
        print(json.dumps(rec),flush=True); out.append(rec)
    subprocess.run(["docker","logs","spec1"],stdout=open(f"{D}/{name}-r{rnd}b.log","w"),stderr=subprocess.STDOUT)
subprocess.run(["docker","rm","-f","spec1"],capture_output=True)
json.dump(out,open(f"{D}/results2.json","w"),indent=1)
