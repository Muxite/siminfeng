# J-SPEC1 — window-union replay: does speculative decoding help a disk-bound over-memory decoder?
Owner: Codex (bounded, CPU-only; no GPU, no docker, no builds, no network). Branch: work in tree, do not commit.
## Inputs
- Routing traces (Qwen3.6-35B-A3B, 40 layers x 256 experts, top-8): experiments/jetson/OM/trace-20261110-0504/{chat,code,math,wiki}/routing.jsonl (format: cc/routing_trace/README.md).
- Simulator: cc/expert_replay/expert_replay.py (+ tests). Per-expert bytes 1,900,544 (589,824+589,824+720,896).
- x86 context: `git show origin/x86/c2-resume:campaign/research/x86-c2-digest.md` is NOT there; read campaign/research/x86-c2-digest.md (ours) for T-SPECM/MTP numbers.
## Task
Add a `specwin` subcommand to expert_replay.py: for decode tokens only, group consecutive tokens into verify windows of size W (W=1,2,3,4,6,8), take the union of experts per layer per window, and replay that union stream through the existing cache policies (LRU, Belady) at capacities 650/1300/2600 slots (expert-slot = 1,900,544 B). Assume an acceptance model: a window of W drafted tokens yields E[accepted]+1 tokens where per-token acceptance a in {0.5,0.7,0.9} (expected tokens = (1-a^(W))/(1-a) ... use the standard geometric formula, state it). Report per workload: distinct experts/layer/window, miss bytes per window, miss bytes per emitted token, and predicted speedup vs W=1 under the disk model decode_tps = 570 MiB/s / miss_bytes_per_token (also report a compute-floor-capped version with floor 6 t/s per verify step for reference, labeled hypothetical).
Note: union of actually-realized consecutive tokens is an optimistic proxy for verify windows (rejected drafts route differently) — state this as a limitation; also provide a pessimistic variant where rejected positions add an independent random top-8 per layer.
## Tests
pytest cc/expert_replay -q must pass; add tests for union/window logic on a tiny synthetic trace.
## Output
Write experiments/jetson/SPEC1/results.json and SPEC1.md (tables), and a <=25-line summary to stdout. Stop if traces lack decode-token marking (report how you'd infer it instead).
