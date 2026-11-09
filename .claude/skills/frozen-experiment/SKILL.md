---
name: frozen-experiment
description: Use when running a timed or quality comparison on the GPU lab that must be pre-registered - a "protocol frozen BEFORE results" experiment (D026, D027, D029, D030, D032, D035). Trigger on "run the experiment", "A/B decode comparison", "lab submit", "alternating order", "host-load monitor", "aggregate job results", "accept or reject candidate". Covers protocol note, tests JSON, alternating order, submit/run, monitor, aggregation, verdict.
---

# Frozen-protocol experiment (campaign/lab)

Principle: write the acceptance rule down and commit it BEFORE any timed result exists; then the numbers decide. Run from `campaign/`.
Scripts in `.claude/skills/frozen-experiment/scripts/`: `make_order.py`, `agg.py`, `loadmon.sh`. One GPU job at a time; no compiles,
downloads or CPU inference during a timing window (a download alone gave 10-15% spreads in D035; compiles distort CPU-expert speeds, D015).

## 1. Protocol note (decisions/D0xx, written first)
Model on D026/D027/D032. State: candidate + patch sha256; image(s) and run_ref; models (+sha); the PRIMARY comparison (same binary, switch OFF vs ON
removes build variance, D026); tests file + order file; control arms (e.g. `staticflags` = same flags without the candidate, D027); acceptance
numbers (e.g. decode >= +2% / +5% beyond the observed ~3% spread, GPU MiB within +3%, KLD <= threshold, text/PPL identity); what
"rejected/inconclusive" means; the claim category it will get (see `decision-note`). Amendments are dated and made BEFORE timed results (D027 amendment 1).
Commit the note + tests + order file. (D026 was frozen in commit 474be3b.)

## 2. Tests JSON (`lab/tests/<exp>.json`)
One new file per experiment; copy entries from `match1.json` (decode: `llama-completion -f /tin/chat.txt -n 160 -c 4096 --temp 0 -s 1 -ngl 99 -no-cnv --no-warmup`
with `docker_args` `-v <campaign>/artifacts/traces/in:/tin:ro`), `c1q8.json` (`llama-bench -p 0 -n 128 -r 10 -o json`, parser `bench_json`, `env` switch), `pap1.json`/`mem2.json`
(KLD: `llama-perplexity -c 512 --chunks 20 --kl-divergence-base /out/bf16.kl --kl-divergence`, `needs_data: true`, mount `artifacts/kl:/out:ro`), `kv1.json` (PPL).
- Test names are global: `load_tests()` merges ALL `lab/tests/*.json` with `dict.update` - a duplicate name silently overwrites. Prefix names (`pap_`, `c1_`).
- `model_file` per test (basename in the model dir) lets one job compare models; `env` carries engine switches; absolute paths for ro mounts.
- Same prompt/seed/-c for every arm; declare cold/warm cache (`cold_cache`) if memory is measured.

## 3. Alternating order
```bash
S=.claude/skills/frozen-experiment/scripts
python3 $S/make_order.py --group a_n24,a_n18,a_n12 --rounds 3 --once kl_a,kl_b --out artifacts/x2/<exp>-order.txt
```
Per group, round i forward if even else reversed (pair, 2 rounds = ABBA); `--repeat N` repeats all groups (c1: `--group q8_tg_r0,q8_tg_r1 --group ... --rounds 2 --repeat 2`);
`--once` = deterministic quality tests run once. Verified to regenerate `pap1`, `c1`, `match1`, `hsplit1` order files byte-for-byte. 3 rounds minimum; keep the
same job for all arms.

## 4. Submit, monitor, run
```bash
export CAMPAIGN_ID=c1
# monitor FIRST; validate it (below), then:
nohup $S/loadmon.sh artifacts/x2/<exp>-hostload.log >/dev/null 2>&1 & MON=$!
JOB=$(bin/lab submit --image llamacpp-b0 --model $PWD/<model>.gguf [--data ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw] \
      --tests $(cat artifacts/x2/<exp>-order.txt) --key <EXP>-alt-1 --res timeout_s=600)
bin/run_op --agent claude-code --task X4 --desc "<exp> timed job $JOB" -- bin/lab run $JOB > artifacts/x2/<exp>-run.out 2>&1
kill $MON; $S/loadmon.sh --summary artifacts/x2/<exp>-hostload.log
```
- `--data` is REQUIRED when any test has `needs_data` (submit fails otherwise). `--key` makes submit idempotent; job.json embeds the test definitions
  and per-input hashes (D025/T-Q), so edits to `lab/tests/` after submit do not change the job. A rerun is a new attempt (`att-2`), never a relabel.
- Image names: `bin/lab image add NAME <build-record image.run_ref>` (see `engine-candidate`); `llamacpp-b0` is the baseline.
- Chains of several jobs: copy `artifacts/x2/chain3.sh` (wait for a build record, submit/run sequentially, one monitor around all).
- Monitor validation: baseline must read `cc=0`; positive control from a SEPARATE command: `cp /bin/sleep /tmp/cc1plus; /tmp/cc1plus 8 &` must raise `cc` to 1.
  The `[c]c1plus` bracket pattern stops pgrep matching the monitor itself (the first monitor reported a phantom compiler). Note a
  command line that merely contains the word `cc1plus` (your own shell wrapper) also matches. A window with compilers/downloads active is labelled UNTIMED / PROVISIONAL.

## 5. Aggregate
```bash
python3 $S/agg.py $JOB                                  # per-test n, gpu MiB, host MiB, medians dec/pre/ppl/kld/top1/tg/pp, [min-max]
python3 $S/agg.py $JOB --strip _r0,_r1 --compare c1_q8_tg_r0=c1_q8_tg_r1   # % delta of medians
python3 $S/agg.py $JOB --json > artifacts/x2/<exp>-agg.json
.claude/skills/engine-candidate/scripts/text_fp.py $JOB --filter gen         # text identity per test
```
Refuse to score unless the header shows `N/N` clean (`tests_completed_cleanly == tests_expected`). Check `BAD` counts, the spread vs the effect size,
disjoint ranges for small effects, and the host-load summary. Tested on job-ad883810b3 (D022), job-34c09bec98 (D029: tg +2.07%), job-3d2fbbd061 (D035).

## 6. Verdict
Apply the frozen rule literally; report accepted / rejected / inconclusive with numbers and ranges, the unmet criterion, and any provisional flag.
Write `decisions/D0xx-...md` with `decision-note`. Register nothing as a claim outside the frozen scope.

## Done when
- [ ] protocol note + tests JSON + order file committed before the job ran
- [ ] one job, 100% clean, monitor log saved and summarized (validated monitor)
- [ ] agg output saved under `artifacts/x2/`; identity checks (text sha / PPL) reported
- [ ] verdict note written with category and limits; attempt ids and job id cited
