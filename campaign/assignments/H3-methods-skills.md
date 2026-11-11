# Assignment H3 — methods catalog + repo skills (formalization; host jetson-arm; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Read: CLOSEOUT.md (all; sections 5 and 6 are the raw material), c2/PLAN.md, decisions/D012, D023, D025-D033 (skim), bin/README-jobq.md, cc/abrun/README.md, cc/om_run/om_run.sh, cc/build_engine (README or --help), cc/kernel_screen (README), cc/resource_observe/README.md, cc/expert_replay, cc/routing_trace (READMEs), assignments/J-OM8-expert-pread-fill.md and assignments/H1-jobq.md (the assignment format with acceptance criteria).
Goal: turn tribal knowledge into (a) a catalog someone new can use to run inference-engine experiments correctly on this lab, (b) Claude Code skills that load the right procedure on demand.

## Deliverables
1. `c2/METHODS.md` (<= 400 lines), sections:
   - Lab map: hosts (jetson-arm facts that change methodology: UMA, devfreq unpinned, NVMe ceiling, zram), what runs where, artefact locations.
   - Tool catalog table: tool | path | purpose | inputs/outputs | qualified-by (test count / live validation evidence path) | known limits. Include every tool in bin/ and cc/ (mark retired ones e.g. pipeline2-6, build_queue.sh as superseded by jobq/abrun; mark untested ones e.g. q8_reuse, gpu_budget as UNTESTED).
   - Methods (each: when to use, procedure steps, acceptance rule, failure modes seen + D-note citation): counterbalanced A/B (ABBA/ABCCBA), A/A gap + drift gate (cc/abrun/gate.py rule verbatim), switch-active verification, flag preflight, decode-path quality gate (ub1 KL vs saved base logits; batch-512 ppl is blind to decode kernels), output-class (identical/differs/lossy), over-memory fair baseline (-lm mmap, -cmoe/-ncmoe, pin -c), contamination rule (one gpu/build job; UNSCORED labels), profiler use (nsys inventory as hypothesis source; ncu blocked), kernel screen (useful for kernel-local changes, NOT graph-fusion), routing trace + expert replay (simulation class), first-order IO model (D031 formula), claim classes (D012).
   - Profilers/tracers/telemetry: resource_observe backends + metric names, what each answers, sampling caveats (peak rate artefact D031).
   - Delegation protocol: assignment template (copy the structure: Read-first, Why, Deliverables, Acceptance commands, Constraints, Final message), check-in by a cheap agent running acceptance verbatim + spot checks, worklog fields (c2/worklog.jsonl).
   - Anti-patterns list (CLOSEOUT 6 "Approaches to avoid" + ops gotchas: pgrep self-match, sudo, docker group, LD_LIBRARY_PATH, --runtime nvidia).
2. Skills in `c2/skills/<name>/SKILL.md` (Claude Code skill format: YAML frontmatter with `name` and `description` (description says WHEN to use it, <= 2 sentences), then concise procedure <= 80 lines each, referencing repo paths, not duplicating METHODS.md prose):
   - `engine-ab-experiment`: write an abrun spec, preflight, submit via jobq, read verdict.json/REPORT.md, classify the claim.
   - `engine-build`: build_engine with a patch via jobq; ccache facts; identity = commit+patch sha+toolchain, not binary hash; gitignore D024.
   - `over-memory-run`: om_run usage, watchdog limits, fair baseline, IO metrics to read.
   - `codex-delegation`: write an assignment, launch with c2/launch_codex.sh (through jobq cpu class when GPU jobs may be queued), Haiku check-in prompt template, worklog row.
   - `kernel-profile`: nsys decode inventory + kernel_screen; what each can/cannot predict.
3. `c2/skills/README.md`: how to install (copy to /home/muk/projects/siminfeng/.claude/skills/), one line per skill.

## Acceptance (checker runs these)
- A1 `wc -l c2/METHODS.md` <= 400 and `grep -c '^## ' c2/METHODS.md` >= 6.
- A2 for each of the 5 skills: file exists, starts with `---`, has `name:` and `description:` lines in the frontmatter, <= 80 lines.
- A3 Every path mentioned in backticks in METHODS.md that starts with `bin/`, `cc/`, `decisions/`, `experiments/`, `c2/` exists (checker extracts with grep -o and tests each with `ls -d`; globs allowed). Report missing ones; >0 missing = FAIL.
- A4 METHODS.md quotes the gate rule `max(0.03, 2*aa_gap, 2*drift)` (or the exact expression used in cc/abrun/gate.py) and cites D012.
Constraints: documentation only; do not edit code, evidence, or decisions; no docker/GPU; do not commit. Time 45 min.
Final message <= 8 lines.
