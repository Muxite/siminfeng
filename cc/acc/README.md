# acc: small on-device accuracy experiments

Python 3 standard library only. No numpy, datasets package, pip, or model
conversion is needed. Run from the campaign root. This harness has offline
fixture/mocked-process validation; **no live model accuracy or cost has been
measured yet**. The captured Granite files are protocol examples, not benchmark
scores. Costs per item and total startup time must be measured by the lead.

## Prepare a frozen dataset

These commands require Internet access and are for the lead to run. They use
only `urllib` and the HF datasets-server `/rows` endpoint, in pages of at most
100. The entire split is fetched before seeded category selection,
so small subsets still require downloading all rows. No dataset execution or
HF authentication is used.

```sh
python3 cc/acc/datasets.py mmlu-pro --per-category 100 --seed 42
python3 cc/acc/datasets.py mmlu-pro --proportional -n 100 --seed 42
python3 cc/acc/datasets.py arc-challenge -n 100 --seed 42
python3 cc/acc/datasets.py gsm8k -n 100 --seed 42
python3 cc/acc/datasets.py gpqa-diamond -n 50 --seed 42
```

MMLU-Pro defaults to exactly N items per category (`-n N` or `--per-category N`).
Selection fails if any category has fewer than N rows. Other datasets default to
proportional allocation of `-n N` total items; `--proportional` explicitly selects
this mode. Both modes use `--seed` and record mode, seed and category counts.
Fetching retries HTTP 429/5xx and URL errors up to eight attempts with exponential
backoff and jitter, honoring numeric or HTTP-date Retry-After. `--page-delay`
sets the delay between successful pages (default 0.5 seconds).

Default output is `~/models/campaign/data/acc/<name>-<mode>-<n>-s<seed>.jsonl` and
`<name>-<mode>-<n>-s<seed>.meta.json`. Files are never overwritten. Each line has
`{id, source, question, choices, answer, meta}`; source is the exact page URL
that supplied the row. MCQ choices are ordered strings
with an uppercase letter answer; GSM8K choices are null and its answer retains
the original reasoning and `####` target. IDs use the source row offset.
Metadata records the seed, source URL, config/split, requested and actual page
offsets, selected row offsets, category allocation, every response hash, and
the output SHA256. In proportional mode, largest-remainder allocation preserves category proportions;
categories smaller than the sample resolution may receive no items.

Sources: `TIGER-Lab/MMLU-Pro/default/test`,
`allenai/ai2_arc/ARC-Challenge/test`, `openai/gsm8k/main/test`.
GPQA probes an **unverified** public `fingertap/GPQA-Diamond` mirror, test then
train. If both are unavailable/gated it prints `SKIP GPQA-Diamond` and exits 0
without writing a dataset. Other network failures remain errors. A lead can
select another public Diamond mirror with `--gpqa-dataset OWNER/REPO --config
default --split train`. Mirrors must expose the original `Question`, `Correct
Answer`, `Incorrect Answer 1/2/3` columns; verify Diamond membership and source
provenance before publishing. Choice order is deterministically shuffled per
row and seed; the correct answer is not always placed first. GPQA availability
was not checked online during implementation. API shape tests are explicitly
labelled `SHAPE_ONLY`; they are not captured dataset evidence.

## Queue every on-device inference run

GPU and build classes exclude each other. Submit the entire evaluator as a
single GPU job; it loads one server and evaluates items serially. Do not run
an evaluator directly alongside another job. All examples below are exact
shell command lines; adjust image/model paths to the intended immutable build
and GGUF. Inspect that image's supported flags before an expensive run, using
the lab preflight workflow inside a queued GPU job as appropriate.

```sh
IMG=campaign/llamacpp:jetson-up1108
MODEL="$HOME/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf"
DATA="$HOME/models/campaign/data/acc"
OUT=experiments/jetson/ACC1-example
export OM_MAX_S=86400
```

MCQ, one generated token per item (same CLI for MMLU-Pro, ARC and GPQA):

```sh
bin/jobq submit --class gpu --name acc-mmlu-base --timeout 14400 -- python3 cc/acc/mcq.py --image "$IMG" --model "$MODEL" --data "$DATA/mmlu-pro-per-category-100-s42.jsonl" --outdir "$OUT/mmlu-base" --engine-arg=-ngl --engine-arg=99 --engine-arg=-np --engine-arg=1 --engine-arg=-c --engine-arg=4096
bin/jobq submit --class gpu --name acc-arc-base --timeout 14400 -- python3 cc/acc/mcq.py --image "$IMG" --model "$MODEL" --data "$DATA/arc-challenge-proportional-100-s42.jsonl" --outdir "$OUT/arc-base" --engine-arg=-ngl --engine-arg=99 --engine-arg=-np --engine-arg=1 --engine-arg=-c --engine-arg=4096
bin/jobq submit --class gpu --name acc-gpqa-base --timeout 14400 -- python3 cc/acc/mcq.py --image "$IMG" --model "$MODEL" --data "$DATA/gpqa-diamond-proportional-50-s42.jsonl" --outdir "$OUT/gpqa-base" --engine-arg=-ngl --engine-arg=99 --engine-arg=-np --engine-arg=1 --engine-arg=-c --engine-arg=4096
```

GSM8K, greedy generation with a configurable cap (default 384):

```sh
bin/jobq submit --class gpu --name acc-gsm-base --timeout 86400 -- python3 cc/acc/gen.py --image "$IMG" --model "$MODEL" --data "$DATA/gsm8k-proportional-100-s42.jsonl" --outdir "$OUT/gsm-base" --max-tokens 384 --engine-arg=-ngl --engine-arg=99 --engine-arg=-np --engine-arg=1 --engine-arg=-c --engine-arg=4096
```

Pass engine switches as repeated literal `--engine-arg=VALUE` arguments and
container environment as repeated `--env NAME=VALUE`; expert_fill/pregate
variables are passed without interpretation. Set `DOCKER_CMD_PREFIX='sg docker
-c'` in the submitting environment when required; the whole Docker invocation
is passed as one quoted argument. A candidate uses its own image, model,
flags/env and fresh output directory. Reuse the **identical dataset file** for
both arms. For over-memory workloads, pass the image-supported mmap/offload
settings described in `c2/METHODS.md`, rather than relying on auto placement.

Observe job IDs returned by submit, then run a worker if needed:

```sh
bin/jobq worker --once
bin/jobq list
```

`bin/jobq show ID`, `bin/jobq wait ID --timeout 600`, and `bin/jobq cancel ID`
operate on the returned ID. Timeout/cancel SIGTERM unwinds Python cleanup.
The server standalone CLI is also queueable (use the URL printed after health):

```sh
bin/jobq submit --class gpu --name acc-server --timeout 2400 -- python3 cc/acc/server.py --image "$IMG" --model "$MODEL" --outdir "$OUT/server" --duration 1800 --engine-arg=-ngl --engine-arg=99
```

Inspect commands without Docker, HTTP, model reads, or output writes:

```sh
python3 cc/acc/server.py --image "$IMG" --model "$MODEL" --outdir "$OUT/dry" --dry-run
python3 cc/acc/mcq.py --image "$IMG" --model "$MODEL" --data "$DATA/mmlu-pro-per-category-100-s42.jsonl" --outdir "$OUT/dry" --dry-run
```

Dry-run uses display-only port 18080 and does not reserve a socket.
The runner chooses a free local port, uses host networking, binds llama-server
to `127.0.0.1`, enables the model's own Jinja chat template, and mounts the model
parent read-only at `/om`. NVIDIA runtime, visible devices and compute/utility
capabilities are explicit. Every attempt owns a unique container. Context exit,
startup/request errors, SIGTERM, SIGINT and atexit kill and remove only that
container. SIGKILL/power loss cannot execute Python cleanup; the saved command
contains the owned name for recovery. Free-port selection has the usual small
bind-to-start race; restart a failed attempt rather than reuse another server.

Watchdog thresholds match current om_run: `OM_MIN_AVAIL_MIB=600`,
`OM_MAX_SWAP_MIB=2560`, `OM_MAX_SWAP_RATE=1000`, interval 250 ms. Telemetry uses
io/proc/tegrastats at 500 ms and `OM_MAX_S=1800` seconds. For long datasets set
`OM_MAX_S` before submitting to cover the full job; it limits sampling, not the
workload. Use jobq `--timeout` for the workload deadline. Health timeout defaults
to 600 seconds (`--health-timeout`); request timeouts default to 600 seconds for
MCQ and 1800 for generation (`--request-timeout`). Missing telemetry remains
missing evidence; inspect coverage and watchdog events before scoring timings.

## Scoring, evidence and resuming

MCQ prompt is fixed and saved in every request:

```text
Question: {question}
A. {first choice}
B. {second choice}
...
Answer with the letter only.
Answer:
```

A fixed system message (`You are a helpful assistant.`) avoids template-injected
dates where the model supports system messages; it is omitted if template caps
explicitly forbid the system role. Chat goes to
`/v1/chat/completions`: temperature 0, max_tokens 1, logprobs true, top_logprobs
20. Prediction is the highest individual logprob among valid choice-letter
tokens, accepting `A` and ` A` variants (maximum, not summed probability).
Tokens containing punctuation/newlines or non-choice letters are ignored.
Ties use choice order. If none appear in returned top-k, prediction is null,
`no_letter=true`, and the item counts incorrect. This is a truncated-logprob
estimator, not full sequence likelihood. Inspect `no_letter` before interpreting
an apparent change. The raw requested and returned top-k are preserved.

If `/props`'s template mentions `enable_thinking`, requests set
`chat_template_kwargs: {enable_thinking: false}`. `template.json` and saved
requests record the choice; other templates record empty kwargs and may still
reason internally. Template presence is a capability heuristic, not proof of
model behavior. The first token is scored regardless of finish_reason.

GSM8K uses `Question: {question}\nSolve the problem. Finish with: The answer is
<number>.`, temperature 0, and the same thinking policy. Extraction prefers the
last explicit `The answer is NUMBER` / `#### NUMBER`; otherwise the last number.
Commas, signs, decimals, dollars and exponents are handled. `Decimal` equality
compares numeric values exactly (72 = 72.0), without floating tolerance.
No-number output is incorrect. Truncated output is still extracted/scored;
`finish_reason` and full responses permit auditing cap effects.

Outputs: `items.jsonl`, `summary.json`, `run.json`, `template.json`, and
`sessions/<unique-attempt>/{cmd.txt,invocation.json,container.id,props.json,
stdout.txt,stderr.txt,result.json,watchdog.jsonl,telemetry.jsonl}`. Each item
records its ID/target, prediction, correctness, request, raw response, timings,
prompt token usage, and session. Summary reports n/correct/accuracy, Wilson 95%
CI, no_letter, mean wall ms/item, total prompt tokens, and mean per-item decode
tokens/s from server timings when available. `complete` distinguishes partial
runs. Mean wall time excludes startup, hashing and telemetry teardown; session
wall time includes the loaded-server period. Server-reported decode rate may
be zero or unstable for one-token MCQ; do not use it as a speed benchmark.

Resubmit the exact same evaluator command to resume. Existing IDs are skipped;
new records flush and fsync after every item. Matching settings are enforced by
dataset SHA256, model SHA256/size/mtime, image string, argv/env and protocol;
template consistency is checked across attempts. Pin an immutable image;
container image IDs are saved in session results. Model hashing reads the GGUF
once per invocation, which is additional I/O and precedes measured items.
Malformed final lines fail explicitly: inspect and repair only the incomplete
line before resuming. Duplicate or foreign IDs fail. Use one writer per outdir.

## Paired accuracy comparison

```sh
python3 cc/acc/compare.py "$OUT/mmlu-base/summary.json" "$OUT/mmlu-candidate/summary.json"
python3 cc/acc/compare.py "$OUT/gsm-base/summary.json" "$OUT/gsm-candidate/summary.json" --json
```

The candidate directory is created by a second queued evaluator run with the
candidate flags/image/GGUF. Comparison requires complete runs with identical
ID sets, targets, dataset hash and scoring protocol. It recomputes counts from
items and reports candidate minus reference accuracy (percentage points in
Markdown), per-arm Wilson CIs, and two-sided exact paired McNemar p. Here b is
reference-correct/candidate-wrong and c is reference-wrong/candidate-correct.
The exact test is `min(1, 2 * P[Binomial(b+c, 0.5) <= min(b,c)])`; no discordant
pairs gives p=1. Marginal Wilson intervals are not a CI for the paired delta.

## Reference logits and candidate KL

Use the same architecture/tokenizer and corpus with both models; save one
reference per model/protocol. The wrapper fixes context 512, batch 512 and
microbatch 1 to exercise decode kernels, and defaults to 8 chunks. The corpus
defaults to `~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw`.
Reference logits can be large. Use a fresh run directory for each KL run.

```sh
LOGITS="$HOME/models/campaign/kl/acc1-granite-c512-ch8.bin"
bin/jobq submit --class gpu --name acc-kl-reference --timeout 14400 -- python3 cc/acc/kld.py reference --image "$IMG" --model "$MODEL" --outdir "$OUT/kl-reference" --logits "$LOGITS" --chunks 8 --engine-arg=-ngl --engine-arg=99
bin/jobq worker --once
bin/jobq submit --class gpu --name acc-kl-candidate --timeout 14400 -- python3 cc/acc/kld.py candidate --image "$IMG" --model "$MODEL" --outdir "$OUT/kl-candidate" --logits "$LOGITS" --chunks 8 --engine-arg=-ngl --engine-arg=99
bin/jobq worker --once
python3 cc/acc/kld.py parse cc/acc/fixtures/perplexity-kld.granite-Q8_0.txt
```

Replace the candidate's image/GGUF/flags for the actual experiment. Candidate
logits are mounted read-only; reference output is a separate writable mount.
`--kl-divergence-base FILE` saves reference logits, while candidate combines it
with `--kl-divergence`. Corpus and logits SHA256 plus context/chunks/batch must
match the reference sidecar. Save model hashes, image/build identity and an
A/A run to characterize the stored-logit floor before setting a quality budget.
KL summary includes ppl, mean_kld, kld_p99, same_top_p, rms_dp, n_chunks, n_ctx;
same_top_p and rms_dp retain printed **percent** units. Parsing uses actual
logged context: the captured fixture is c128, not the wrapper's c512 default.

## Interpretation and validation

This is a zero-shot, letter-logprob MCQ / greedy numeric harness. Published
MMLU-Pro, GPQA, ARC and GSM8K results may use few-shot prompts, CoT, different
splits, answer extraction, chat templates, caps and full datasets. Record those
differences and use the same protocol for engine/quant A/B. A seeded subset with
Wilson intervals supports a local paired estimate; it does not reproduce a
published leaderboard number automatically. Fix subsets, model/tokenizer,
context, template and extraction before comparing. Report n, errors/no_letter,
CI, and protocol with every score; small subsets have low detection power.

At 1–10 tok/s, the cap-only generation budget for 384 tokens is 38.4–384 seconds
per item, excluding prefill/startup. This is arithmetic, **not measured cost**.
MCQ emits one token but still pays prompt prefill; measure mean_ms_per_item
before choosing n. Actual per-dataset item costs: **pending live measurements**.
Quality evidence does not establish a speed gain; follow `c2/METHODS.md` for
A/A, counterbalanced timing, activation evidence and contamination checks.

```sh
python3 -m pytest cc/acc -q
python3 cc/acc/mcq.py --help
python3 cc/acc/gen.py --help
python3 cc/acc/server.py --help
python3 cc/acc/datasets.py --help
```

The tests read all five captured files; Docker/subprocess and HTTP calls are
mocked. No engine run, network fetch or GPU access is part of validation.

## Orphan containers

Every accuracy container has `acc=1` and, when queued, `jobq_job=$JOBQ_JOB_ID`.
Before starting a container the harness inspects existing accuracy containers
and asks `bin/jobq show` about their owners. Running or exited containers are
removed if their job is absent or no longer running. Containers without a job
label are removed only after six hours. Job lookup failures other than unknown
jobs abort reaping. Preview removals with read-only Docker discovery:

```sh
python3 cc/acc/server.py reap --dry-run
bin/jobq reap-containers
bin/jobq reap-containers --yes
```

The queue command considers both `acc=1` and `om=1` and retains containers
without a job label. Both dry-run commands print only proposed Docker removal
commands. `DOCKER_CMD_PREFIX` applies to discovery and removals alike.
