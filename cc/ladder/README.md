# Model ladder on Jetson Orin Nano 8GB

Stdlib Python; run from the campaign root. This module prepares one bounded,
serial on-device step per GPU job. Implementation validation is offline only:
**no new fit, speed, MMLU or quant-fidelity evidence has been collected.**
`models.json` copies the candidate sizes/scores and cited URLs from
`research/model-candidates-20261111.md`. Published thinking-mode scores are
context, not substitutes for the local non-thinking MCQ protocol.

## Lead: freeze the datasets (network)

```sh
bash cc/ladder/fetch_sets.sh
```

This calls `cc.acc.datasets.fetch_rows`, `normalize` and seeded `subset`;
`datasets.py`'s top-level CLI uses proportional stratification, so this small
adapter instead samples **10 per category** for `mmlu_pro_140.jsonl` and **5 per
category** for `mmlu_pro_70.jsonl`, seed 1. Both use the same fetched test split;
the 70 IDs are nested in the 140. Output is
`~/models/campaign/data/acc/`, with dataset URLs, page hashes, selected source
row offsets, category counts and file SHA256 in `.meta.json` sidecars. Existing
files are refused before fetching. `--outdir PATH` relocates both sets. This
script is never called automatically by inference or tests.

## Exact queued commands for the three downloaded models

Run fit before speed/acc/KL for each model; subsequent steps automatically use
its first successful config. Wait for each queue job to finish before advancing.
The image defaults to `campaign/llamacpp:jetson-up1108-om11`. Submit on the Jetson
with the normal Docker access environment (`DOCKER_CMD_PREFIX='sg docker -c'`
if required). GPU/build exclusion is provided by jobq; label checks also refuse
an already running `om=1` container. Ladder jobs additionally hold an output-root
lock. Do not overlap unqueued GPU or CPU-heavy work.

```sh
bin/jobq submit --class gpu --name ladder-fit-qwen36 --timeout 7500 -- python3 cc/ladder/ladder.py fit qwen36-35b-a3b
bin/jobq submit --class gpu --name ladder-fit-qwen9 --timeout 7500 -- python3 cc/ladder/ladder.py fit qwen35-9b
bin/jobq submit --class gpu --name ladder-fit-gemma4 --timeout 7500 -- python3 cc/ladder/ladder.py fit gemma4-e4b

bin/jobq submit --class gpu --name ladder-speed-qwen36 --timeout 7500 -- python3 cc/ladder/ladder.py speed qwen36-35b-a3b
bin/jobq submit --class gpu --name ladder-speed-qwen9 --timeout 7500 -- python3 cc/ladder/ladder.py speed qwen35-9b
bin/jobq submit --class gpu --name ladder-speed-gemma4 --timeout 7500 -- python3 cc/ladder/ladder.py speed gemma4-e4b

bin/jobq submit --class gpu --name ladder-acc-qwen36 --timeout 15000 -- python3 cc/ladder/ladder.py acc qwen36-35b-a3b --set mmlu_pro_70
bin/jobq submit --class gpu --name ladder-acc-qwen9 --timeout 15000 -- python3 cc/ladder/ladder.py acc qwen35-9b --set mmlu_pro_140
bin/jobq submit --class gpu --name ladder-acc-gemma4 --timeout 15000 -- python3 cc/ladder/ladder.py acc gemma4-e4b --set mmlu_pro_140

bin/jobq submit --class gpu --name ladder-kl-aa-qwen36 --timeout 15000 -- python3 cc/ladder/ladder.py kld qwen36-35b-a3b --ref qwen36-35b-a3b --chunks 2 --max-seconds 14400
bin/jobq submit --class gpu --name ladder-kl-aa-qwen9 --timeout 7500 -- python3 cc/ladder/ladder.py kld qwen35-9b --ref qwen35-9b --chunks 8
bin/jobq submit --class gpu --name ladder-kl-aa-gemma4 --timeout 7500 -- python3 cc/ladder/ladder.py kld gemma4-e4b --ref gemma4-e4b --chunks 8

python3 cc/ladder/ladder.py report
```

The three initial KL commands are **A/A stored-logit floor checks**, not quant
loss measurements: these models have different tokenizers and cannot reference
one another. The corpus must already exist at
`~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw`.

Jobq returns IDs. Use `bin/jobq show ID`, `bin/jobq wait ID --timeout 600`, and
`bin/jobq list`. A worker must be active; `bin/jobq worker --once` executes one
queued job. No step invokes jobq or starts another job by itself.

For a direct comparison with the over-memory model, also measure both resident
models on the **identical 70-item file**; a 140-item score and 70-item score do
not enter the same frontier:

```sh
bin/jobq submit --class gpu --name ladder-acc70-qwen9 --timeout 15000 -- python3 cc/ladder/ladder.py acc qwen35-9b --set mmlu_pro_70
bin/jobq submit --class gpu --name ladder-acc70-gemma4 --timeout 15000 -- python3 cc/ladder/ladder.py acc gemma4-e4b --set mmlu_pro_70
```

Wave-2 spec IDs: `qwen36-35b-a3b-iq3s`, `qwen35-4b-q8`,
`qwen35-4b-q4`, `qwen36-35b-a3b-iq2m`, `gemma4-12b`.
Missing model paths fail cleanly before Docker. Once both 4B quants have passed
fit, a practical cross-quant comparison is:

```sh
bin/jobq submit --class gpu --name ladder-kl-qwen4 --timeout 7500 -- python3 cc/ladder/ladder.py kld qwen35-4b-q4 --ref qwen35-4b-q8
```

Q8 is a near-lossless *proxy*, not BF16. The Qwen3.6 lower quants can use
`--ref qwen36-35b-a3b`; that Q4 reference measures *incremental* loss, not BF16
loss. Both spec tokenizer groups/architectures and actual GGUF tokenizer
metadata digests must match. The reader uses stdlib and reads metadata only.

## Approximate wall time and bounds

These are planning assumptions, not newly measured timings. Every step has
`--max-seconds` (fit/speed/KL default 7200, acc 14400). This bounds workload plus
preflight, with up to roughly two minutes of owned-container cleanup afterward;
jobq timeouts leave cleanup headroom. `OM_MAX_S` is set to cover the step's
remaining budget and controls telemetry duration, **not** the workload deadline.

| Step | Resident models | Over-memory Qwen3.6 Q4 | Assumptions |
|---|---|---|---|
| fit | 1–5 min/config | 2–5 min/config, two candidates | startup/load + short prefill + 32 generated tokens; failures may be slower |
| speed | 5–20 min | 30–60 min | six cold-process loads, three short prompts twice; 256 tokens/run, roughly 1.4 tok/s for Q4 vs 5–15 for resident models |
| acc | 15–60 min for 140 | 80–140 min for 70 | resident item 5–20 s; MoE prefill ~5.6 tok/s, item ~60–100 s, plus startup/hash overhead |
| KL | 15–60 min for 8 chunks, two arms | 30–90 min for 2 chunks, two arms | context 512, ubatch 1; two corpus passes plus hashes/base IO; highly workload dependent |
| report | seconds | seconds | reads saved summaries, no inference |

The MoE MCQ estimate is why its default is N=70; resident defaults use N=140.
`--set` chooses either frozen set; `--data PATH` supports a custom frozen JSONL
with its hash retained. MCQ request timeout is the harness default 600 s;
outer deadline can interrupt and resume the item loop. Increase
`--max-seconds` and the jobq timeout together if actual costs are higher.

## Configs, prompts and measurement

Resident candidates, in order: full GPU + flash attention at `-c 4096`; same
with q8 KV; full GPU explicit mmap; partial GPU 24/12/0 layers with mmap + flash
attention. Layer counts are explicit candidates, not automatically tuned values.
Over-memory candidates use `-ngl 99 -lm mmap -cmoe -c 4096`, first with J-OM11's
frd16 + pregate K=8/order=after/inflight=4/lookahead=1, then without expert env.
All env names/values and effective/activation regexes are copied from J-OM11's
pg8 arm; the older silent K fallback is not used. `fit --all` sweeps past the
first success. `--config ID` restricts fit or explicitly selects a downstream
placement; ordinarily let the measured fit select it.

Fit uses llama-completion, a chat prompt and `-n 32 --ignore-eos`. It accepts
status ok, no OOM/watchdog trip, all required expert signatures, positive precise
ms/token, and at least 31 eval runs (llama.cpp accounts the first generated
token in prompt evaluation). Each trial saves minimum available MiB, max swap
MiB, load seconds and precise ms/token through `run_summary.py`. A successful
probe of an over-memory model means it *runs under the watchdog*, not that its
weights fit in DRAM. That regime remains labelled over-memory in the report.

Speed uses the model's built-in Jinja chat template with one conversation turn.
Fixed user prompts live in `prompts/{chat,code,reason}.txt`; order is
chat/code/reason/reason/code/chat (ABC-CBA). Each run uses `-n 256 --temp 0 -s 1`,
`--ignore-eos`, `--reasoning off`, and template kwargs `enable_thinking=false`.
These flags and the kwargs are saved in command/result artifacts. The actual
binary's help is checked before execution; unsupported flags fail preflight.
Six successful runs are required. Metrics are medians of precise
`eval_ms_per_token`, prompt tok/s, **decode-window** NVMe GiB/token, J/token and
GPU utilization; decode tok/s is 1000 / median ms/token, not a rounded engine
rate. JSON records available sample counts per metric. Missing telemetry stays
null. No independent warmup is excluded; inspect pair spread and clocks.

Accuracy delegates to `cc/acc/mcq.py`, which owns `cc/acc/server.py` and records
raw responses, per-item evidence, dataset/model hashes, Wilson 95% intervals,
no-letter counts and the actual template kwargs. The server is additionally
started with `--reasoning off`. Context, placement, KV and expert env match
speed; `-np 1` prevents context sharing between server slots. A watchdog trip,
OOM, cleanup error or effective-config mismatch leaves the run unscored. For
one-token MCQ, decode-only expert filling may not execute; effective settings
are checked, without claiming pregate work from its environment alone.

KL delegates to `cc/acc/kld.py`, whose fixed **context 512**, batch 512, ubatch 1
replace the placement config's context 4096. The summary explicitly records
this override; other placement/env settings stay identical. Chunks are capped
using the larger declared vocabulary upper bound, fp16 logits, 64 bytes/token
and 64 MiB overhead, so the planned base is below **decimal 3 GB**. Actual GGUF
vocabularies are checked against the bound, and actual base size is checked
before candidate use. Reference bases and metadata live under
`~/models/campaign/kl/`, named by reference stat/config/image/corpus/protocol
identity. A saved reference is reused only with matching provenance and hash.
The harness validates corpus/chunks/context against its sidecar. Inspect an
interrupted base and remove its incomplete files before retrying; no existing
base is overwritten. `--text`, `--chunks` and `--kl-dir` are configurable.

## Results and resuming

```
experiments/jetson/LADDER/<model_id>/<step>-<UTC timestamp>/
  step.json                    # identity and settings
  summary.json                 # complete step envelope for report
  <config>/                    # fit: om_run logs/result/watchdog/telemetry/summary
  001-chat/ ... 006-chat/       # speed: the same artifacts per run
  acc/                         # accuracy harness, items + per-attempt sessions
  reference/ and candidate/    # KL owned-container evidence
```

Only the relevant subdirectories exist for each step. Repeating an identical
command reuses its step directory: completed steps are no-ops, fit skips recorded
trials, speed skips successful slots and retries failed slots in fresh owned
attempt directories, and acc resumes fsynced item IDs. Model size/mtime, image,
placement, prompts and dataset/corpus identity guard resume; the acc/KL harnesses
also retain full-file hashes. `--new-run` requests an independent replication.
A changed immutable image or model stat cannot silently select an old fit.
A failed final fit is complete evidence (exit 1); use `--new-run` to reprobe.

SIGTERM/deadline cleanup stops the child process group, then removes only saved
container IDs or generated acc container names owned by this step. SIGKILL or
power loss still requires operator recovery using saved IDs. Interrupted trials
retain their artifacts. All inference is serial; no background model is left
running intentionally.

`report` writes `REPORT.md` and `ladder.json` at the result root (override with
`--out PATH`). The Markdown table has a Pareto column, local Wilson intervals,
tiers >=2/>=5/>=10, published MMLU-Pro, KL reference, energy and GPU utilization;
JSON retains full published scores/URLs, all measured accuracy sets and fit
telemetry. Text scatter and frontier/tier winners use only complete local speed
and accuracy pairs. Frontiers are computed independently per dataset hash and
MCQ protocol; different subset sizes never compete. If multiple accuracy sets
exist, the table displays the largest, and the frontier list covers every set.
Newer compatible complete steps supersede earlier ones. Different images or
model identities remain separate. Dominance is descriptive, not a statistical
claim: overlapping Wilson CIs and unpinned clocks limit interpretation. Phase
IO/energy timing is approximate; see `cc/abrun/README.md` and
`c2/MEASUREMENT-GAPS.md`.

## Offline review and tests

```sh
python3 cc/ladder/ladder.py fit qwen35-9b --dry-run
python3 cc/ladder/ladder.py fit qwen36-35b-a3b --dry-run
python3 cc/ladder/ladder.py speed gemma4-e4b --dry-run
python3 cc/ladder/ladder.py acc qwen35-9b --set mmlu_pro_140 --dry-run
python3 cc/ladder/ladder.py kld qwen35-4b-q4 --ref qwen35-4b-q8 --dry-run
python3 cc/ladder/ladder.py report --fixtures --dry-run
python3 -m pytest cc/ladder cc/acc cc/abrun -q
```

Dry-runs do not invoke Docker, inspect busy containers, fetch datasets, read
model/corpus bytes, preflight images, or write outputs. Downstream dry-runs
without a fit preview the first candidate and label that assumption. A dry
report with no ladder records automatically previews the captured fixtures;
`--fixtures` explicitly selects them. Normal reports never substitute fixture
results for measurements. The fixture README names original files and limits:
OM10c legacy summaries and Granite protocol examples are never MMLU scores or
frontier candidates.
