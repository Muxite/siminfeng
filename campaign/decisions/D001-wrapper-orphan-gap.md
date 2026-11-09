# D001 — run_op leaves orphan child when wrapper is SIGKILLed
- Question: does the capture wrapper keep intent/outcome truthful under abnormal wrapper death?
- Evidence: evidence/orchestrator.jsonl op-b2f5ebfb4653 (intent, no outcome); `sleep 30` pid 64002 survived (ppid reparented), killed by me by PID after observation.
- Finding: record is honest (UNFINISHED/UNKNOWN) but the child is NOT contained. A lab runner must own children via cgroup/container, not process group alone.
- Chosen next action: keep run_op as is for host-side ops; run all engine work in labelled containers (`--label campaign=<id>`) so ownership + cleanup is by label/cgroup. Add `reconcile` listing intents lacking outcomes.
- Uncertainty: pgid recorded only in outcome, so a killed wrapper leaves no pgid in the intent. Cheap fix: write pgid in a second "started" record.
- Would change if: Docker-based ops make host-side wrapping rare.
- Capability signal (observed): "reconcile unfinished ops against live owned processes" — see capabilities/.
