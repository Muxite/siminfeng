# Six-step build-to-result rehearsal (5–8 minutes)

From the campaign checkout:

```bash
DEMO_DRY=1 bash cc/demo/demo.sh       # print commands; no wrapper, Docker, or writes
DEMO_PAUSE=1 bash cc/demo/demo.sh     # live rehearsal; Enter between numbered steps
bash cc/demo/demo.sh                # live, without pauses
```

Budget about one minute per step, with up to two minutes for interruption and
export. The talk track supplies the pacing; automatic mode does not add theatrical
delays. Model hashing, cold file reads, and GPU load can extend the live timing.
`DEMO_MODEL` can override the default existing mixE4K path, but the directory must
contain `granite-3.1-1b-a400m-instruct-mixE4K.gguf`, as the pinned tests require.

Run live on the prepared x86 host with Python 3, Bash, GNU date, Docker access,
the already-loaded `llamacpp-novmm4` image, the existing model and
`artifacts/traces/in/chat.txt`, and an otherwise idle GPU. No image build, image
load, model download, or new test capability is part of this script. Reserve the
GPU operationally: the lab has no GPU admission lock. Do not run multiple demos
concurrently.

## 1 — Identity, approximately 1 minute

Say: “This is the successful novmm-4 build record from the chain that already ran.
The config digest identifies the image configuration/content. The local Docker
run reference is a separate field: this containerd store reports an OCI index
identity there. The archive SHA identifies transport bytes. These must not be
substituted for one another.”

Point to status and the linkage summary: dynamic libcudart, cudaMalloc/cudaFree
imports, and no cuMemCreate import in this no-VMM build. These are recorded
build/linkage observations, not a fresh archive verification or proof of runtime
correctness. Reference: `decisions/D025-build-to-result-chain.md` and
`cc/build_engine/README.md`.

## 2 — Fresh immutable submission, approximately 1 minute

Say: “We create a new timestamped key and job rather than replaying a historic
result. The existing lab submit command records model/input hashes and embeds
the exact definitions of m1_static_n12 and m1_static_n0. The runner uses those
embedded definitions. mixE4K is the real existing GGUF model.”

Show the new job ID. The demo copies the existing image registration and current
test files into an isolated lab store; it does not alter tests, register a new
image, or manufacture input hashes. Retrying this job later means a new attempt,
not changing this submission.

## 3 — Compatibility receipts, approximately 1 minute

Say: “This offline envelope joins the build identity to the submitted inputs,
invocation, receiver, and explicit requester policy. The x86 profile accepts it.
The same proposal is rejected by the Jetson template; read the actual platform,
host ID, CUDA architecture mismatches and missing declarations.”

The policy is a fresh demo declaration with campaign c1, experiment DEMO,
attempt att-1, the actual build-record locator, and the actual export destination.
It is not copied with the stale locators from the example policy. The host
profiles are under `cc/engine_handoff/hosts/`, not top-level `hosts/`.
Acceptance does not reserve capacity, enforce the whole-job wall/capture budgets,
or run a remote receiver. Host RAM cgroups do not cap discrete VRAM.

## 4 — Kill the test sandbox, approximately 1–2 minutes

Say: “The controller runs outside the test container. We start the existing lab
runner in the background, wait for its test-container journal entry, and confirm
that exact container is running. Then the existing chaos command kills only
containers owned by this attempt's token. The host and controller stay alive.”

The exact CLI is `bin/lab chaos JOB att-1`. Its implemented action is
`kill-sandbox`; adding `kill-sandbox` as another positional argument is rejected
by argparse. The script clears inherited `LAB_FAULT`; neither runner-death nor
export-death injection is used here. It never sends a process signal or a
host-wide kill command.

Say: “Now wait for the controller to finish cleanup and export. The journal
records the interruption. Every exported file must exist and match MANIFEST.json
by a fresh SHA256 check, displayed in sha256sum-check style. The result envelope
keeps interrupted execution, incomplete coverage, durable sequence and confirmed
stop separate from test assessment.”

The literal current outcome is `interrupted_sandbox_terminated`, not the shorter
`interrupted` used by reconciled runner-death records. The script checks this
exact value, incomplete/unscored evidence, manifest presence, and stop confirmation.
It shows the entire envelope, including `evidence_completeness`,
`last_durable_sequence`, `stop_confirmed`, and artifact references.

The lab exports before appending the live journal's terminal event. Verify the
export journal against its own manifest; do not compare it to the later live
journal. “Durable sequence” here is local journal line count/timestamp, not a
remote acknowledgement or proof of host-loss survival.

This is the sandbox-loss variant. `LAB-FEASIBILITY.md` also documents the earlier
runner-SIGKILL variant: reconcile found token-owned orphan containers, stopped
them, preserved partial evidence, and allowed a new attempt. This script leaves
the runner alive, so its normal cleanup/export reaches terminal and no reconcile
is needed. Do not run the global `bin/lab reconcile` against unrelated jobs to
make this scene look more dramatic.

## 5 — A new attempt, approximately 1 minute

Say: “Before retry, we hash every file in att-1's live directory and exported
copy. Running the same job again creates att-2. It executes both pinned tests,
exports evidence, and yields a completed result with two clean completions and
confirmed stop. We then compare the entire att-1 file sets and hashes.”

Point to the separate attempt identity and owner-token hash in the result.
Att-2 does not fill att-1's evidence gaps or rewrite its assessment. A clean
completion is exit zero with explicit non-partial evidence; read test assessment
separately. The script checks execution/coverage, not a performance threshold.
The existing lab spec does not persist campaign/experiment IDs, so those fields
can be null in result envelopes even though wrapper operations and proposal
policy carry c1/DEMO. The display does not fill those gaps with invented values.

## 6 — Five statements, approximately 1 minute

Read the script's five-line summary: build identity; new pinned job; x86/Jetson
receipts; interrupted att-1 with verified evidence; completed att-2 with unchanged
att-1. End by pointing to the printed record directory so reviewers can inspect
the bytes and receipts independently.

## What is real, what is temporary, and claims to avoid

Real: the existing Build CC record and registered image; the existing model,
prompt and test definitions; live local container execution when run on the
prepared host; token-selected sandbox kill; exported stdout/stderr and journals;
fresh manifest verification; separate attempt result envelopes and hash comparison.

Temporary adapter: **our `bin/lab` is the thin reference adapter, not the other
team's lab agent**. `envelopes.py` is the existing offline projection and
compatibility checker. `inspect.py` only displays/asserts records, copies existing
configuration, declares policy, polls the journal, and hashes evidence. It adds
no executor, assessor, scheduler, transport, or recovery capability.

Do not claim remote execution, Jetson execution/acceptance, integration with the
other team's agent, authentication, hard GPU isolation, a hard VRAM cap, enforced
whole-job/capture budgets, GPU reservation/exclusivity, host-loss durability,
bitwise reproducibility, correctness/accuracy, or performance superiority. Do not
score partial evidence, equate admission with execution, or call the local
manifest an independent authenticity/provenance proof.

## Recording, write locations, and validation limits

Every external command executed by the demo goes through
`bin/run_op --agent demo --task DEMO`, with `CAMPAIGN_ID=c1`. Shell builtins perform
printing, variables, branching, backgrounding and `wait`. The wrapper buffers
command output until completion; the background runner's output can appear
between foreground messages. Every command is shell-quoted and printed before
execution. Dry mode prints the planned commands using explicit job/timestamp
placeholders and never invokes `run_op`, Docker, sleeps, or pause prompts.

All runtime writes are below `artifacts/codexDEMO1/`: `CAMPAIGN_ROOT` redirects
wrapper logs to `evidence/demo.jsonl` and captures to `artifacts/op-*`; each unique
session has its own `lab/{images,tests,jobs,export}` and policy/receipt/snapshot
files. The supplied `evidence/.gitkeep` preserves the parent directory required
by the existing wrapper. Source checkout, shared lab store, model and prompt
remain read-only. Implementation files are confined to `cc/demo/`.

Author validation is limited to Bash syntax, shellcheck if installed, dry runs,
and offline record/CLI inspection. No Docker or `bin/lab run` was executed while
preparing this assignment. A live rehearsal is still required to establish
current host readiness and timing. The start-event/Docker-start race is handled
with a bounded running-state probe; if the short test finishes before the kill,
the exact interruption assertion fails rather than claiming a successful fault
injection. Each test has the existing 300-second timeout; the journal wait is
bounded at 180 seconds. The runner is never killed on a wait/probe failure.
