# F12 — Evolve a bare engine: per-feature component CCs, A/B methodology CC, evolve task

Repo: `/home/muk/projects/jiuwenswarm` (`ai4r/iterative-cc-factory`, `.venv`). Read: `jiuwenswarm/factory/{AGENT.md,manager.py,lab_runner.py,contract.py,controller.py,api.py}`, `jiuwenswarm/factory/ccs/*`, siminfeng `cc/abrun/README.md`, `cc/abrun/abrun.py`, `c2/skills/engine-ab-experiment/SKILL.md`, `c2/skills/engine-build/`, `c2/METHODS.md`, patches `cc/mmap_advice/patches/0[1-9]-*.patch` (a cumulative series on llama.cpp `cf34d3113`), `experiments/jetson/OM/J-OM1*/REPORT.md` (prior results; do NOT copy numbers into capsules). Factory home `~/.jiuwenswarm-factory` READ ONLY; tests use temp homes and fake job tools. No Docker/GPU.

Goal: the factory (driven by the Codex-backed JiuWenSwarm agent via the existing codex_subscription adaptor) starts from a **bare** llama.cpp and grows it: integrate a component, build, measure against the current best with a stated methodology, keep or revert, repeat. Everything visible in story/monitor/telemetry.

## 1. Component capsules (material-only, `cc: '2'`, no new top-level fields)
Under `jiuwenswarm/factory/ccs/components/<name>/` with `cc.yaml`, `component.yaml`, patch files:
| name | patches | enable env | engagement (stderr regex) |
|---|---|---|---|
| mmap_advice | 01 | `LLAMA_MMAP_ADVICE=random` | from patch log lines |
| router_prefetch | 02 (depends mmap_advice) | per patch | per patch |
| expert_fill | 03,04,05 (depends 01,02) | `LLAMA_EXPERT_FILL=1 …` | `expert_fill: active` |
| expert_pregate | 06,07,08 (depends expert_fill) | `LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8 …` | `expert_pregate: active`, `expert_config: effective` |
| thread_names | 09 (depends expert_fill) | none | `xfill-*` threads (sampler) |
Read each patch to fill env names/ranges/log lines accurately. component.yaml: `base` (cf34d3113), ordered patch files + sha256, `depends_on`, `enable_env`, `engagement_checks`, `hypothesis` (mechanism, one sentence, no numbers), `measure: perf|none`, `correctness_obligation: identical greedy stdout`, `rollback`. Verify in a test that applying the series in dependency order onto a cf34d3113 checkout succeeds with `git apply --check` (use `~/projects/siminfeng/ref/llama.cpp` or the owned checkout read-only via `git worktree`-free `git apply --check --directory`; skip the test with a reason if no checkout is available). Keep the existing `engine_component` (patch 09 on om11 source) for the recorded demo; mark it legacy in its README.

## 2. Methodology capsule `ccs/methodology_ab/` (material)
`AB.md` adapted from the engine-ab-experiment skill for the factory: hypothesis first, one variable per experiment, env-only arms on one image where possible else image-vs-image, ABBA counterbalancing, A/A + drift gate (from gate.py), output identity required, keep/revert rule, how to read the verdict. Short (≤ 120 lines).

## 3. Lab A/B capability (lab_client + Manager)
`lab_client` gains `ab_submit{base_image, candidate_image, base_env, candidate_env, rounds, idem_key}`, `ab_status`, `ab_result`. Manager admits both images (same admission rules), writes an abrun spec from the contract workload (argv/env/model mount) with `order: ABBA`, `metric: eval_ms_per_token` (check abrun supports it; else eval_tps), candidate `expect_switch` + regexes from the component's engagement checks (agent passes component id; Manager reads component.yaml — agent cannot supply regexes), submits ONE durable `gpu` jobq job running abrun (no nested submit), ingests `verdict.json`, `REPORT.md`, per-run summaries as artifacts and an `ab_result` observation `{gate, ratio, aa_gap, output_class, per_arm_ms}`. Correctness: output class must be identical AND base-arm stdout sha must equal the contract reference; otherwise `candidate_failed`.

## 4. Contract `qwen36-bare-greedy-v1` (unfrozen)
Same workload as qwen36-greedy-v1 minus patch-specific env; reference image `campaign/llamacpp:jetson-up1108` (bare cf34d311). Leave `reference: null`; operator freezes via the existing freeze-reference path. Engagement checks come from component.yaml at evaluation time (candidate-only), not the contract.

## 5. Evolve task + controller
`tasks/evolve-qwen36.json`: objective "Starting from bare llama.cpp cf34d3113 in the owned checkout branch `evolve`, build the bare engine, then grow it one component at a time (choose order from component dependencies and hypotheses). For each: integrate (commit), build, A/B vs the current best image with methodology_ab, keep if gate improved and output identical, otherwise revert the commit and record why. End with the best image, its component stack and evidence." Budget 40 turns, 8 h. Allowed decision action types extended (in AGENT.md + controller validation): `integrate`, `build`, `measure`, `keep_component{component, image, ratio}`, `revert_component{component, reason}`, plus existing terminal ones. Story + monitor render keep/revert lines and the engine stack from these decisions (update story.py/monitor.py if F11 landed; else story.py only).
Controller init for this task creates branch `evolve` at `cf34d3113` in the owned checkout if missing (never touches other branches).

## Acceptance
- G1 `.venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider` all pass (new: component yaml validity + material paths, patch-series apply check, ab_submit admission + spec generation with fake jobq, ab_result ingest from a fixture verdict, keep/revert decision validation, story renders keep/revert).
- G2 `jiuwenswarm factory capsules --plain` lists the 5 components + methodology_ab.
- G3 No commits/pushes/builds/lab runs; no writes to `~/.jiuwenswarm-factory`; do not change `Store.query` behaviour.
Final message: files, per-component env/engagement actually found in patches, open questions.
