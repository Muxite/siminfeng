# Lab core feasibility (campaign c1) — what the prototype `bin/lab` demonstrated
Scope: image = container (immutable id), tests = named definitions run inside it, job = immutable submission, attempt = one execution. No UI. Evidence: lab/jobs/*/att-*/state.jsonl, result.json; ops in evidence/orchestrator.jsonl.

## Demonstrated (ran against real containers + GPU)
| # | Guarantee | Evidence |
|---|---|---|
| 1 | Images bound to immutable id; same name cannot silently point at a different image | `lab image add` check |
| 2 | Idempotent submission by key; same key + different spec rejected; job.json write-protected | job-3d376c81fb dup/conflict |
| 3 | Happy path, 3 tests (ppl2, gen_smoke, bench_pg) in one admitted attempt; ppl2 = 7.5325 matches E001 | job-3d376c81fb/att-1 |
| 4 | Five result sections kept separate (execution / quality / performance / evidence / cleanup) | result.json |
| 5 | Runner SIGKILL mid-test: orphan container found by label+token, killed, attempt `interrupted`, partial stdout kept, NOT scored | job-e91db9b081 |
| 6 | Cancellation: request != confirmed; confirmed only after owned containers gone; live runner left alone by reconcile | job-93d03ebe43 state: cancel_requested -> cancel_confirmed |
| 7 | Death during export: staged copy retained; register rejected until durable export; reconcile resumes export without rescoring | job-5bbc09b093 |
| 8 | Registration idempotent (identical = no-op), conflicting late result rejected (rc 6), foreign/stale token rejected (rc 5), no-durable-export rejected (rc 7) | cases 4a/4b |
| 9 | Retry = new attempt (att-2) | job-9252e2d8bf |
| 10 | (HISTORICAL — lock removed in D007) GPU admission by flock serialised concurrent attempts (waited 3.5 s) | job C6-conc |

## Problems found (these drive the interface)
- P1 **Lock != exclusivity.** flock vanished with the dead runner while its container kept using the GPU (case 5). Admission must be tied to the *container's* lifetime / cgroup, or reconcile must precede re-admission.
- P2 **Recovery bug (D006):** first reconcile overwrote a finished result. Recovery must be phase-aware and results write-once. This was found only because the test killed the runner *in each phase*; "runner died" as one test would have missed it.
- P3 **No per-job single-active-attempt policy.** Two invocations of one job = two attempts (case 6). Needs an explicit policy (reject / queue / coalesce).
- P4 **Measurement confound is invisible to the lab:** SM clock at launch was 2955 MHz vs 1155 MHz across attempts; the lab records it but cannot control it. Perf tests need clock/warmup policy declared in the test, not guessed.
- P5 **First-rep warmup** in bench_pg samples (14.9k vs ~18k t/s): raw samples must be kept; averaging hides it.
- P6 **Root-owned outputs:** container ran as root, so converted GGUF is root-owned on the host. Cleanup/export needs a uid policy.
- P7 **Test definitions are code in the lab, not data.** Each new test (parser + assessor) required editing `bin/lab`. A test needs a declarative contract: command, input mounts, parser, assessor, whether it needs exclusive GPU.
- P8 **Wrapper blind spot (D001)** repeated at a larger scale: process-group tracking insufficient; label+token ownership worked.
- P9 Host-visible only: ownership by docker label can be forged by anyone who can run docker; this is "ownership by convention + token", not enforcement.

## Not demonstrated (proposals only)
Heartbeat/lease handling, timeouts for the *runner* (only per-test timeout), disk/capture budgets, multi-GPU, queue fairness, retention/GC of exports, remote execution, authentication.
- NOTE (D007): GPU lock removed by user; P1 is now moot, replaced by 'no admission control: scored runs need a per-run exclusivity check'.
