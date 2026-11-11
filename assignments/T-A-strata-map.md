# Assignment T-A (parent: campaign c1 / orchestrator, task stage1)
Question: how is Strata composed — routing vs residency vs execution?
Input: /home/muk/projects/Strata @ ffa80d73133ea5c548aefe57889a16e8085a938f (READ ONLY; do not edit, build, or run GPU code).
Read AGENTS.md first. Cover: entry points; model identification/compat checks; weight loading; routing (src/core/layer.cpp); expert identities/slots/residency (include/strata/core/expert_cache.hpp); execution adapter (expert_source.hpp); CPU/GPU coordination (hit_hook.hpp); prefill vs decode; KV/session state + reset; memory budgeting; build paths; telemetry/profiling hooks; strata_control/journal.py and calibration_jobs.py.
Deliverable: /home/muk/projects/siminfeng/campaign/artifacts/codexA/strata-map.md. Every statement cites file:line. Separate: routing decisions / residency decisions / actual execution. Mark model-specific coupling (DeepSeek/Qwen assumptions vs Granite) and reusable-vs-adapter-vs-inseparable boundaries.
Write only under campaign/artifacts/codexA/. No network. Time limit: 40 min. Final message: list of files written + open questions.
