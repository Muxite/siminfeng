# Lab Manager design (derived from campaign operations; living document)
Scope: deterministic coordination only (no reasoning agent). The other team owns the lab agent / sandbox provisioning; our reference adapter is a TEMPORARY Docker-container stand-in (`bin/lab`), never claimed to be their lab.

## Responsibility map
| Actor | Owns | Evidence |
|---|---|---|
| Builder (coding sandbox) | source edits, patches, build CC invocations | builds/results/*/build-record.json |
| Build CC | source+patch+bases -> runtime image + archive + record (incl. failures) | W4 attempt 1 failure record |
| Lab Manager (controller) | immutable job spec, attempts, admission, ownership token, state log, export, registration, reconcile | lab/jobs/*, LAB-FEASIBILITY.md |
| Lab agent (other team) | accept/reject/needs-info, choose+provision sandbox | NOT exercised |
| Test sandbox | runs ONE test container per request; disposable; no docker socket; RO model/data | job CR1-crash |
| Evidence store | outside sandbox: stdout/stderr streamed to files, state.jsonl, export w/ MANIFEST | lab/export/* |

## Demonstrated behaviors (op/job refs)
idempotent submit + conflicting-spec rejection (job-3d376c81fb); attempts/retry-as-new (job-9252e2d8bf att-2; job-3739814377 att-2); cancel requested != confirmed (job-93d03ebe43); runner death -> orphan container found by label+token and killed (job-e91db9b081); death during export -> resume, no rescoring (D006 fix, job-5bbc09b093); late conflicting / foreign-token result rejected (cases 4a/4b); duplicate delivery during active attempt rejected rc 8 and replacement blocked until owner gone/reconciled (job-3739814377); controlled sandbox kill -> interrupted_sandbox_terminated with coverage, never pass, controller responsive (job-3739814377 att-1).

## Open / not demonstrated
Admission/queueing (user removed lock D007: single-job-at-a-time is procedural; no capacity model), leases/heartbeats (deliberately not used: liveness = pid+start-time and labelled containers), runner timeouts, disk/capture budgets, retention/GC, remote receiver acknowledgement states (received/accepted/rejected/needs-information are specified in engine_handoff but not yet exchanged with a real lab agent), host-loss failure domain (protected domain = sandbox/process only; evidence on same host).

## Additions from the memory-objective phase (D012-D015)
Demonstrated (job/op refs): image admission by archive: wrong sha256 -> rc 10, wrong expected config id after load -> rc 12, correct -> admitted (op "image-load b0 archive", builds/results/b0-baseline-3); Build CC emitted a patched engine in 18 s via ccache cache mount (builds/results/rt1) vs 7-11 min clean (b0-baseline-2: 11-min build lost to a link error; recorded as failed attempt with stage + full log); per-test memory sampling (GPU peak via nvidia-smi, container cgroup memory.peak and io.stat) with own-file cold cache and container memory limits (job-6e6b2d39b8); test definitions as data (lab/tests/*.json) with per-test env, model file, docker mounts, memory limit.
New requirements found by failure:
- R-A **Interface contracts between CCs must be pinned, not described.** The handoff CC (T-E) was given a flat build-record schema; the Build CC (T-D) emitted a nested one; admission refused a genuinely good delivery (experiments/handoff-real/verdict.json). Needed: a versioned build-record schema owned by one place, plus a normalizer test across both. -> Manager/CC-registry concern, not agent reasoning.
- R-B **Observed vs declared must be enforced by the admission tool**: values from the build record are claims; image config id, platform and executable hashes must be re-observed from the archive (T-K).
- R-C **Concurrent work contaminates measurements** even without a GPU lock: build/compile jobs (load avg up to 18/16 threads) distorted CPU-expert speeds (D015). Admission must also cover CPU/IO load or record load average and mark runs contaminated. (GPU lock was removed by the user, D007; this is the observed cost.)
- R-D **Memory is a first-class measurement**: GPU peak + container peak + physical reads, with cold/warm declared per test; a result without a declared cache state is not comparable.
- R-E Delegated agents (Codex x11, Opus x1, Sonnet x1, gpt-6-astra advice x1) required written assignments, write-location scoping and independent re-verification: Codex G's and H's analysis was high quality; Codex-written code passed its own tests but the integration (R-A) failed only when two outputs met.
