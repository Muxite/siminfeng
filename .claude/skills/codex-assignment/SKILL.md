---
name: codex-assignment
description: Use when delegating bounded code or analysis work to Codex in the siminfeng campaign (writing/launching an assignment brief, waiting for its outcome, reviewing and committing its deliverables). Trigger on "delegate to codex", "write an assignment", "T-<name> task", "review Codex output". Opus orchestrates; Codex does bounded work; Codex has no Docker/GPU.
---

# Delegating to Codex (campaign/assignments + bin/run_op)

Standing rule (memory `delegate-to-codex`): Opus orchestrates, picks hypotheses, reviews, runs GPU/Docker steps; bounded
code/analysis goes to Codex; synthesis/mechanical checks to Sonnet/Haiku. Run everything from `/home/muk/projects/siminfeng/campaign`.

## 1. Write the brief
Copy `.claude/skills/codex-assignment/assignment-template.md` to `campaign/assignments/T-<ID>-<slug>.md` (real examples:
`T-MI-model-inspect.md`, `T-KVB2-kv-quant-assert.md`, `T-COMP-component-versioning.md`). A brief MUST have:
- Pinned base (llama.cpp commit `31288c9c6`, `ref/llama.cpp` is read-only; Codex works on a `git archive` scratch tree under `artifacts/codex<ID>/`).
- One goal, concrete deliverable paths, CLI/schema, invariants (default path unchanged, never load full tensors, ...).
- **Allowed writes** (`cc/<name>/**`, `artifacts/codex<ID>/**`), time limit, `nice`, `-j4`, "no Docker, no GPU, no network".
- HOST NOTE: if a GPU timing job is running say "pure Python, no compiles" (compiles contaminate timing, D015/D032).
- Tests on REAL records (name the real files) plus synthetic edge cases. Final-message contents (files, test result, numbers, risks).
- Wording: describe engine defects in correctness terms (assertion, capacity, wrong result). T-KVB was REFUSED by the provider after 130 s
  ("flagged for possible cybersecurity risk") because it said crash/SIGSEGV/gdb/backtrace; T-KVB2 (same bug, reworded) succeeded
  (`decisions/gaps.md`, ops op-9a911bb7e729 vs op-ad9b8c4302f2). `launch_codex.sh` lints for these words (override: `FORCE=1`).

## 2. Launch (background, recorded)
```bash
.claude/skills/codex-assignment/scripts/launch_codex.sh ID assignments/T-ID-slug.md [--timeout 3000] [--parent X-c2] [--dry-run]
```
Expands to the pattern in `evidence/codex-*.jsonl` intent records:
`nohup nice -n 15 bin/run_op --agent codex-ID --parent X-c2 --task T-ID --desc "codex exec ..." --timeout N -- codex exec --json -s workspace-write -C <campaign> -o artifacts/codexID/last-message.txt "Read and execute the assignment in <abs path> exactly. Stay within its write locations."` with `CAMPAIGN_ID=c1`.
Use `--dry-run` first to see the command. Several Codex jobs in parallel are fine if none compiles during a timing window.

## 3. Wait
Do not poll stdout. `python3 .claude/skills/codex-assignment/scripts/wait_outcome.py codex-ID` blocks until the `outcome` record exists
in `evidence/codex-ID.jsonl` (exit 0 ok / 1 failed,timeout / 2 not finished with `--no-wait`). Reports status, elapsed, survivors, paths
to stdout/stderr (`artifacts/op-*/`) and `last-message.txt`. A `failed` outcome within ~2 min and no deliverables = suspected provider
refusal: read the stdout tail, add a line to `decisions/gaps.md`, reword, relaunch under a NEW task id (T-<ID>2). A failed attempt is evidence, never delete it.

## 4. Review checklist (before accepting or building on anything)
1. `git status` / `git diff --stat`: only the allowed write locations changed? (revert/flag anything else).
2. Re-run the claimed tests yourself (pytest line from last-message; use `PYTHONDONTWRITEBYTECODE=1`, `nice`).
3. **Re-verify against REAL records, not Codex's fixtures.** T-N passed 76 mocked tests and failed the first real build (novmm-1); T-N2/T-N3
   again (D023, D025, MANAGER-DESIGN c2 §6). Run the tool on: real build-record.json, real lab job dirs, real GGUFs, real buildx metadata.
4. Check numbers in the last message against files (sizes, hashes, MiB vs MB), and that "claims allowed" sections do not overreach.
5. For patches: `git -C ref/llama.cpp apply --check $PWD/<patch>` (absolute path), then sha256 it and record the hash.
6. GPU/Docker steps Codex could not do are listed as "orchestrator must run"; do them (see `engine-candidate`, `frozen-experiment`).
7. Record gaps found (to `decisions/gaps.md` or the decision note, see `decision-note`).

## 5. Commit
Stage only reviewed paths (`cc/<name>/`, `assignments/`, `artifacts/codexID/` small logs/JSON; never model files), `git status` first
(the tree usually holds many untracked `artifacts/codex*`). Commit message: `T-ID <what> (reviewed: <how>)` plus the attribution lines.
Commit protocols BEFORE results (see `frozen-experiment`).

## Done when
- [ ] assignment file committed; `evidence/codex-ID.jsonl` has intent+started+outcome
- [ ] outcome ok (or failed attempt logged in gaps.md and relaunched)
- [ ] write-scope check clean; tests re-run by you; tool re-run on a real record
- [ ] patch hashes / key numbers copied into the decision note or CC README
