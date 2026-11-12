# F25 — Revise CC-v2 in JiuwenSwarm from real factory use (small, clean, keep e2e working)

Worktree: `/home/muk/projects/jiuwenswarm-ccv2` (branch `ai4r/ccv2-revise`). Python: `PYTHONPATH=/home/muk/projects/jiuwenswarm-ccv2 /home/muk/projects/jiuwenswarm/.venv/bin/python`. No Docker/GPU, no compiles, no commits, never touch `/home/muk/projects/jiuwenswarm` or `~/.jiuwenswarm-factory`. Be quick and conservative: the operator wants CCs to work very cleanly for a small demo, not a redesign.

Read: `jiuwenswarm/ccv2/{ccv2.py,agent_mcp.py,source_build.py,applications.py,schemas/}`, `tests/ccv2/`, the factory's use of it (`jiuwenswarm/factory/{controller.py,capsules.py,api.py}`, `jiuwenswarm/factory/ccs/*/cc.yaml`, `ccs/*/run.py`). Reference design (read-only git repo `~/tools`): `AI4Research-Notebook/Designs/CC Protocols/{README.md,USAGE.md,FORMAT.md,IMPLEMENTATION.md,reference/ccv2.py}` on `main`, and branch `origin/ccv2-runner-fixes` (commit ea057c1: "Reference runner: detach helper stdin, never pull on finite image calls" — `git -C ~/tools show ea057c1`).

1. Compare: write `docs/architecture/factory/CCV2-REVISION.md` (≤80 lines): what the jiuwenswarm ccv2 has vs the reference (main + runner fixes), and the friction actually hit in factory runs:
   - sealed base interpreter lacked the venv's packages, worked around with `FACTORY_PYTHON` (`controller.configure_agent`);
   - components nested inside a capsule needed a second `--ccs` root;
   - long-running work must be durable jobs (calls end within the 60–300 s MCP timeout);
   - material `kind` added (608656eb);
   - 9 pre-existing failing `tests/ccv2` tests (`test_source_build` ×8, `test_products::test_exited_launcher_does_not_confirm_children_stopped`).
2. Port from the reference only what applies: the runner fixes (detach stdin; no pull on finite image calls) and, if not already present, the ease-of-use defaults (semantic checks opt-in, advisory failures keep results, explicit required checks gate acceptance) — only where our code has the same mechanism; no new subsystems.
3. Fix from experience, minimal:
   a. Native binding interpreter: let a host configure the interpreter used for `{python}` (e.g. `Host(python=...)` / env `CCV2_PYTHON`, default unchanged), and make the factory use it instead of the `FACTORY_PYTHON` indirection (keep `FACTORY_PYTHON` working for old capsules: run.py files may keep reading it, but the host should pass the venv python). Update the factory capsules' run.py only if needed.
   b. Discovery of nested capsules: a capsule root may contain `<capsule>/components/*/cc.yaml`; discover them from one `--ccs` root (bounded depth 2, deterministic order), and remove the factory's second `--ccs` argument.
   c. The 9 failing tests: diagnose; fix if they are environment assumptions (e.g. missing git identity, Windows paths, timing) by making the test self-contained, else mark `skip` with a precise reason. Do not weaken real checks.
4. Tests: `tests/ccv2` (if it hangs >5 min inside your sandbox, stop and say so — the operator runs it) and `tests/factory` must pass. Report counts.
Final message: changes, before/after test counts, what was deliberately not ported.
