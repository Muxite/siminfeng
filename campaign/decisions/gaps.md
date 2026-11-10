- GAP G-001: Claude Code project hooks (.claude/settings.json) installed mid-session; no live claude_hook records observed (only selftest). Native Claude record = session transcript under ~/.claude/projects until restart; verify after restart.

## c2-resume: delegation friction (2026-11-09)
- T-KVB (Codex) was refused by the provider ("flagged for possible cybersecurity risk") after 130 s — the brief used crash/SIGSEGV/gdb/backtrace wording for an ordinary engine assertion bug. Rephrased as T-KVB2 (assertion in graph construction, correctness bug). Methodology: phrase engine-debug briefs in correctness terms; capture provider refusals as failed attempts (op-9a911bb7e729), not silent drops.
- T-PL4's CPU per-type screen runs concurrently with T-QRT's CPU Qwen tracing (both Codex, 2026-11-10) -> PL4 CPU constants indicative only; orchestrator reruns cc/moe_planner/cpu_type_screen.py on a quiet host before adoption.
Host change 2026-11-10T03:44:37+00:00: kernel.perf_event_paranoid 4->1 (runtime only, reverts on reboot; restore: sudo sysctl -w kernel.perf_event_paranoid=4). Authorized by user 2026-11-10.
