---
name: decision-note
description: Use when recording a campaign result or protocol as a decisions/D0xx note, or deciding what category of claim a result supports. Trigger on "write the decision", "D0xx note", "freeze the protocol", "classify this result", "full-engine gain or trade-off", "claim category", "accepted/rejected verdict".
---

# Decision notes (campaign/decisions/D0xx-*.md)

One short note per decision: either a PROTOCOL frozen before results, or a RESULT with a verdict. Style of D026/D029/D030/D033/D035/D036: dense, numeric, cited.
Next number = highest `D0xx` + 1 (currently D036 is last; `decisions/gaps.md` is the running friction log, not numbered).

```bash
.claude/skills/decision-note/scripts/new_note.py <slug> 'Headline with the verdict' --category 'memory-fit'            # result skeleton
.claude/skills/decision-note/scripts/new_note.py <slug> 'Candidate X protocol' --protocol --category 'full-engine code gain'   # frozen protocol
```
Run from `campaign/`; it prints the new path and never overwrites. Fill every field; delete none (write "not tested" instead).

## Result template (fields in order)
Headline (`# D0xx — <what> <ACCEPTED|REJECTED|...>: <one-line number>`), Claim category, Evidence ids, Conditions, Protocol (frozen rule), Numbers, Verdict vs the rule,
Interpretation, Limits, Next. Evidence ids = job ids (`job-34c09bec98`), `builds/results/<name>/build-record.json`, op ids / task in `evidence/*.jsonl`, `artifacts/...` paths.
Conditions must let someone reproduce the run: image run_ref + config digest, commit, patch sha256, model (+sha), flags/context, host-load monitor result, rounds, clean N/N.
Numbers: medians with [min-max] and n; deltas in %; GPU MiB (nvidia-smi delta) and unit named (MiB vs MB). Quote the frozen threshold next to the measured value.

## Claim categories (pick exactly one primary; say what it is NOT)
| Category | Use when | Required evidence | Not allowed to claim |
| --- | --- | --- | --- |
| full-engine code gain | engine source change, computation-preserving, speeds up/saves in the real engine (D029 q8_reuse +2.06%) | text sha + PPL identical, test-backend-ops pass, same-binary OFF/ON, engagement | memory reduction unless measured; other GPUs/models |
| runtime-placement trade-off | existing flags/offload mix moves cost between GPU MiB and speed (D022 static vs moe-cache) | equal-footprint comparison, GPU MiB per arm | "engine improvement" |
| representation trade-off | quantization/KV precision/model recipe changes numerics (D033 KV q8_0, D035 PAP/imatrix) | KLD/top-1 or PPL vs bf16/f16 with threshold, bytes in stated unit, controls at same K | computation-preserving; task-level quality from wikitext only |
| memory-fit | what fits in a budget / footprint observation for an existing option (D028 flash-attn, planner D034) | nvidia-smi delta, ctx declared, prediction vs measurement | speed claims without timing protocol |
| layer-component result | an engine layer or component isolated/swapped (D036 binary .so swap, cc/components if present) | identity of components (sha/ABI), A/B on the swapped piece | that other layers/ABI versions are interchangeable |
| infrastructure validation | lab/Build CC/shim/monitor behaviour (D021, D023, D025) | real records (not fixtures), what is NOT demonstrated | performance or memory-guarantee claims |
| failed-inconclusive | rejected candidate, never engaged, noisy window, unmet gate (D024, D030) | the unmet criterion and the numbers; hypotheses for why | any win; keep the failure documented |

## Rules
1. Protocol note is committed before the first timed result; changes are dated amendments before results (D027). Verdicts quote the frozen rule literally.
2. Separate measured from predicted: kept distinct, label "predicted", "provisional" (noisy host, D035), "untimed" (D024), "exploratory" (open correctness issue, D027 amendment).
3. Always add Limits: hardware (RTX 5070 Ti), model (Granite 3.1 1B-A400M), workload (batch-1 chat decode, wikitext), spread vs effect, what is not demonstrated.
4. Rejections and failures get notes too, with the next discriminating experiment (D030 -> D031 nsys).
5. Methodology lessons and factory rules go in the note's last lines ("Factory rules: ...", "Method lesson: ..."); friction with tools/providers goes in `gaps.md`.
6. Do not claim from a single-sample or contaminated run; cite which job/attempt (a rerun is a new attempt, never a relabel).
7. Related skills: `frozen-experiment` (produces the numbers), `engine-candidate`, `model-variant`, `codex-assignment`.

## Done when
- [ ] number unused; file in `decisions/`; headline carries the verdict
- [ ] category chosen from the table with a "not claimed" statement
- [ ] every number traceable to a job id / build record / artifact path; units named
- [ ] limits and next step present; committed (protocol notes committed before results)
