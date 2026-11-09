# D007 — GPU admission lock removed (user decision)
- User: "remove the gpu lock, lets not use it anymore."
- Change: removed flock admission from bin/lab and bin/lab_run; deleted /tmp/campaign-gpu.lock. Attempts record admission="none".
- Consequence: concurrent attempts/jobs may share the GPU simultaneously. Any performance or profiling measurement is only valid if the operator ensures nothing else runs (or verifies it via gpu_pre/gpu_post memory + process list). LAB-FEASIBILITY.md items 10 (serialisation) and P1 (lock vs exclusivity) are now historical: demonstrated before removal.
- Superseded earlier statement: plan "one GPU admission mechanism" no longer applies; exclusivity for scored runs becomes a procedural check, to be evidenced per run.
