# Expert-streaming development artifacts (2026-11-12)

Preserved copies of work done in private llama.cpp clones under ~/projects/xs-dev (not on any remote).
- patches/10,11,16: readiness, in-flight join, decode-phase stats (stacked on cf34d3113 + 01-08; also packaged as factory components in jiuwenswarm engine.expert_streaming).
- patches/12-15: F16 candidates (selection trace, populate backend, expert pin, temporal predictor), each on 01-08, CPU-tested only.
- patches/12-selection-trace-on-16.patch: trace rebased onto 01-08+10+11+16 (image factory/llamacpp:xs-trace, used by J-XS4-activity).
- opt/: offline trace simulator + F16 check notes. placement/: F19 CPU/GPU placement decision record and operator scripts (GPU runs not executed).
