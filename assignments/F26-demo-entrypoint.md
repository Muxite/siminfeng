# F26 — One automated, Claude-drivable 10-minute demo entrypoint (extend existing story/replay/monitor; no new framework)

Worktree: `/home/muk/projects/jiuwenswarm-demo` (branch `ai4r/demo`). Python: `PYTHONPATH=/home/muk/projects/jiuwenswarm-demo /home/muk/projects/jiuwenswarm/.venv/bin/python`. Read-only access to `~/.jiuwenswarm-factory` (use the store read-only/snapshot APIs) and `/home/muk/projects/siminfeng`. No Docker/GPU, no compiles, no commits. Be quick; this is presentation plumbing.

Terminology: this is a **software factory that uses CCs** (not a "CC factory").

## Story (≈10 min; durations are defaults, editable in one JSON)
1. Two opening slides — local copies already in `demo/assets/slides/{engine,factory}.html` (single `<section>` each in a 1920×1080 inline-styled format; wrap them in a local page scaled to fit; fonts fall back offline).
2. Baseline + contract: plain llama.cpp `cf34d311` image `factory/llamacpp:bare-cf34d311` (`sha256:a9a0c71c…`), its generated output (frozen reference stdout from contract `jiuwenswarm/factory/contracts/qwen36-bare-greedy-v1.json` and the Run 4 bare evaluation), and the task contract (Run 4 task record from the store; show also `tasks/expert-streaming-qwen36.json` as the current brief).
3. JiuwenSwarm invoking CCs (Run 4 `factory-bd173c082bf34ae3b8471938c19b8068`): capsule material reads and `cc_call` receipts from telemetry/story (`jiuwenswarm.factory.telemetry`, `story.py`), short and labelled RECORDED.
4. Source change + build receipt: Run 4 integration commit diffs for expert_fill / expert_pregate (artifacts in the store) — show a concise diff excerpt (hook site in `src/llama-context.cpp`, `src/llama-expert-fill.h`) — and the build receipt (source commit → image id).
5. Expert activity: `/home/muk/projects/siminfeng/experiments/jetson/OM/J-XS4-activity/activity.html` (copy into the pack), labelled "separate unscored capture run (J-XS4), image factory/llamacpp:xs-trace"; caption: selected vs predicted experts; 92% of predicted vs 55% of unpredicted expert bytes ready before demand; 79% predicted.
6. One evidence-driven plan revision: Run 5 `factory-53db494559954fc6a690a9a014afbf21` decision 13 / graph rev 13 (`CONFIG_MISSING` on the router-only image → integrate 03–08 switched off for strict reporting, router measure moved after the new build), from the store + `taskgraph` replay (before/after graph). Label "Run 5 excerpt; Run 5 was held for later and kept no component".
7. Result: J-OM13 (`/home/muk/projects/siminfeng/experiments/jetson/OM/J-OM13/REPORT.md`): plain 841.5 vs factory engine 711.4 ms/token (1.18×), 12-run ABBA, identical output; Run 4 single lab runs 903.95 → 785.22 (label single-run); show generated output text.
8. Handoff: capsule `engine.expert_streaming 2.0.0` at tag `cc-baseline-2026-11-12` (commit 4fe167f9), `RSI-START-HERE.md`.
Backup views (not in the timed flow): source map, ontology, design-change, responsibility map, Run 4 full story.

## Build
- `jiuwenswarm/factory/demo.py` + CLI `jiuwenswarm factory demo {preflight,build,serve,next,prev,pause,resume,goto N,status,live}`:
  - `build` assembles a self-contained pack dir (default `~/.jiuwenswarm-factory/demo-pack`): `index.html` + section JSON + copied assets, all content extracted from REAL evidence (store records, contract file, J-OM13 report, activity HTML), each section stamped with its provenance (task id, record ids, file paths, image ids) and a label LIVE / RECORDED / RECONSTRUCTED / SEPARATE CAPTURE. No network, no external fonts required.
  - `serve` starts a local HTTP server (reuse monitor.py's server pattern) on 127.0.0.1:8790 and opens the browser (`xdg-open`, ignore failure, print URL). The page shows one section at a time with a short explanation, a progress bar with the elapsed vs 10:00 budget, auto-advance per section duration, and keyboard controls (→ ← space).
  - `next/prev/pause/resume/goto/status` talk to the server (`/api/control`) so Claude can drive it from the terminal; the page polls `/api/state`.
  - `live` runs the REAL read-only JiuwenSwarm invocations for the current section (e.g. `jiuwenswarm factory capsules --plain`, `factory plan <run5> --rev 13`, `factory story <run4>`) with a timeout and, on failure, shows the recorded output captured at build time, labelled RECORDED FALLBACK.
  - `preflight` checks: pack built, assets present, server port free, store readable, task ids exist, image ids present locally (`sg docker -c "docker image inspect"` read-only, optional), real entrypoint command available; prints PASS/FAIL lines.
  - The page always shows the real end-to-end command (`jiuwenswarm factory init … && jiuwenswarm factory run … --grant …`) as "not executed during the talk".
- Update `docs/architecture/factory/DEMO-10MIN.md`: one launch command, controls, fallback, timings.
- Tests: one small test that `build` produces a pack from a temp store fixture and that the control API transitions state. Run that + `tests/factory/test_views.py`.
Final message: commands, section list with sources, anything faked or missing (should be none; mark gaps).
