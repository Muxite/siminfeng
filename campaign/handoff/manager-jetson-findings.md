# What the Jetson consumer taught the Lab Manager design (for the x86 Manager owner)
Source: host jetson-arm (Orin Nano 8GB, linux/arm64, shared DRAM), branch jetson/c1, 2026-11-10. Read against MANAGER-DESIGN.md and LAB-FEASIBILITY.md.
Class key: [MP] Manager policy | [LA] lab-agent/provider implementation (other team) | [CC-op] reusable CC operation | [COMP] component | [AR] agent reasoning.
Status key: DEMO = ran on Jetson with job evidence; REVIEW = code/unit tests/review only; NOT = not done.
Caveat: the reference adapter is bin/lab (temporary Docker stand-in). Everything below describes it, not the other team's lab. No real lab agent was exercised.

## 1. Host capability inventory and admission decisions
- Event: lab/hosts/{x86-main,jetson-arm}.json (host_id, platform, gpu.{request,sampler,memory_domain,arch}, resources_max, capabilities, l4t_major). Admission decision recorded per job in admission.json with host_profile_sha256.
  - job-ecf12fd41a: accepted (reasons empty). Real run, completed.
  - job-0ec3b98dbb: rejected: platform mismatch linux/amd64 (x2), cuda_arch mismatch, missing requirement runtime and l4t_major. `bin/lab run` on it exits 10 (op-f8f22d97344c).
  - job-82c4a257d3: needs_information "model path missing". job-c7046d34b6: needs_information "unknown image record nope".
- Design must: make the three-way decision (accepted / rejected = cannot ever fit this host / needs_information = fixable by the submitter) a first-class, durable, per-job record taken BEFORE any attempt, naming the host profile hash it was computed against. `run` must refuse non-accepted jobs. Reasons are machine-readable strings today; they should be codes.
- Class: decision vocabulary, record shape and "no attempt without accepted record" = [MP]. Deciding what a host can run and provisioning = [LA] (their agent owns accept/reject; our rules are a stand-in). Profile content is host-owned data.
- Status: DEMO for the four outcomes above. The admission was a pure function of (image record, spec, host profile, local file existence); no remote negotiation.
- Declared vs locally verified images: `lab image add --declared --platform P` stores image_id=null, verified_locally=false, provenance=declared; admission then yields needs_information "image not verified on this host". Requirements are claims from the build record (target_requirements: platform, cuda_arch, runtime, l4t_major); the arch/platform checks run against the host profile, not against the image. DEMO of the record shape in admissions above; REVIEW for the declared-image path (no real job exercised it by name in the jobs I read). Class: [MP] (declared never equals admitted) + [CC-op] image-load (archive re-observation, already x86 R-B).
- Admission hashing: job.json carries admission_sha256; `run` re-reads admission.json, compares hash (mismatch -> rc 11), then RE-EVALUATES admission against the current image record and host profile and requires accepted. DEMO of the hash recorded: job-4433960e61 queued state carries admission_sha256. REVIEW only for the mismatch path (rc 11) and for re-evaluation; I did not see a job that tripped rc 11.
- Legacy records: jobs without admission.json run as "legacy-unrecorded" (x86 only, and only if the image record has no platform); J-T4-era admission without hash is "recorded-unhashed (pre-J-T4b)" but still re-checked, so an old rejected job stays refused (D023 note). Observed: job-ecf12fd41a, 5c142dd5df, 43ef050ce9 have no admission_sha256 in their queued state (pre-hash era). Design must: version the admission record and never let "missing" read as "accepted". [MP]. Status: REVIEW (code paths), partially DEMO (the unhashed jobs exist; I did not re-run them).

## 2. Platform-specific GPU request (D022)
- Event: D022. Plain `--runtime nvidia` on Jetson: exit 1 "CUDA driver version is insufficient for CUDA runtime version". Adding NVIDIA_VISIBLE_DEVICES=all and NVIDIA_DRIVER_CAPABILITIES=all passed the CUDA smoke (Orin CC 8.7, 0/1024 mismatches). Generic arm64 images do not set the env; l4t bases do.
- Design must: the GPU request is a host-profile field (`gpu.request`: gpus-all vs nvidia-runtime-env) expanded by the launcher; the job spec must not hardcode `--gpus`. A CUDA init failure must be classified as "GPU not exposed to container / request missing" candidate, not "driver incompatible".
- Class: request selection = [MP]/host profile; making the launcher apply it = [LA] (their provisioner must do the equivalent); error classification = [COMP] (result classifier). Status: DEMO of the underlying behaviour (D022) and of the request in real jobs (bin/lab gpu_request emits runtime + env; note it uses capabilities=compute,utility, whereas D022's passing smoke used all). The classifier fix is NOT done: nothing in bin/lab re-labels that CUDA error.

## 3. Memory domain and engine defaults derived from free memory (D023)
- Event: job-4433960e61 att-1 gen_smoke (no -c) chose n_ctx=45568; shared-DRAM high-water 6568 MiB vs baseline 1634 MiB on a 7607 MiB device (earlier same test 5647/6288 MiB in job-ecf12fd41a / 43ef050ce9 att-2). Cause is source-supported: llama.cpp fit uses device free memory, which on Tegra is system-available DRAM.
- Design must: host profile declares gpu.memory_domain (shared-dram vs discrete-vram). Any engine default derived from "free device memory" is a host-memory policy on shared-dram; test definitions must pin it (done for Jetson: lab/tests/jetson.json gen_smoke_c2k). Admission should refuse or flag a test that leaves such a default unpinned on a shared-dram host.
- Class: declaring domain, requiring pinned test params = [MP]; the pin itself = test definition data (our side); an allocator cap (cudaMemGetInfo interposer, x86 cc/gpu_budget) = [COMP], untested on Jetson. Status: DEMO of the symptom; fix applied to Jetson tests only. The admission check that flags unpinned defaults is NOT done (memory_domain is recorded in the profile but bin/lab admission does not read it). x86 gen_smoke result also depends on free VRAM; D023 leaves that decision to x86.

## 4. Telemetry scope
- Event: result.json mem for Jetson tests: gpu_peak_mib/gpu_delta_mib = null, shared_dram_peak_mib + scope "system-shared-dram" + gpu_memory_reason "shared DRAM is not GPU memory"; gpu_post = {gpu:null, compute_apps:null, unavailable:[reason]} (all Jetson jobs).
- Design must: telemetry records carry scope; a missing metric is null plus a reason, never 0; "GPU memory" and "shared DRAM" must not share a field; baseline absence is reported.
- Defects found by the J5 runs, fixed later (compare job-ecf12fd41a/5c142dd5df/43ef050ce9 with job-4433960e61):
  - Early jobs report exclusive_before_after=false although compute_apps is unknown (null). That is a false statement. Job-4433960e61 reports null. Rule: exclusivity unknown is null, not false. [MP] + [COMP].
  - Early jobs have shared_dram_baseline_mib=null with "no tegrastats sample available" (sampler had not produced a line yet); job-4433960e61 has baseline 1634 after a retry-until-sample baseline. Rule: baseline is required or the result is marked baseline-missing. [COMP].
  - Note even now, "exclusivity" for any host is only "no compute apps at end"; during-run interference is not observed (bin/lab says so). On Jetson it cannot be observed at all through this sampler.
- Class: scope vocabulary and null-with-reason rules = [MP]; backends (tegrastats/nvml/cgroup) = [COMP] cc/resource_observe (schema resource-observe/0 proposed as common contract); a provider must supply an equivalent = [LA]. Status: DEMO (both fix and defect); the CC's backends are unit-tested; I did not verify its tests here.
- Sampler lifecycle: tegrastats is a child of the runner; SIGKILL cannot run handlers, so reconcile kills it by recorded pid+start-time (owner.json tegrastats_pid). REVIEW only; owner.json of job-43ef050ce9 att-1 shows no tegrastats fields (runner killed before sampler recording or before that code), so the orphan-sampler cleanup was not demonstrated by that job.

## 5. Crash semantics observed
- Sandbox kill (job-5c142dd5df att-1, op-a9edfb1d8dbd chaos): container exit 137 -> state sandbox_terminated -> execution "interrupted_sandbox_terminated", ppl null, coverage tests_expected 2 / started 1 / completed_cleanly 0, "incomplete data is never scored", partial stdout/stderr hashed in evidence.files, cleanup confirmed. Export reported hash-valid (checkpoint-1; I did not re-hash it). att-2 (retry as new attempt) completed with ppl 10.0094. Class: state names and never-score-incomplete = [MP]; killing the sandbox and reporting it = [LA]. DEMO.
- Runner SIGKILL (job-43ef050ce9 att-1, op-8255cfa1c2cc status signaled): last durable state "running"; immediate rerun refused rc 8 "still active ... no duplicate execution" (op-1a37540ab664); `lab reconcile` then wrote reconciler=true exporting/terminal, execution "interrupted", killed [] (container had already exited, nothing to kill), partial files kept, unscored; rerun created att-2 which completed. Class: single-active-attempt policy, reconcile-before-retry, results write-once = [MP] (resolves x86 P3); who runs reconcile and when = [LA]/[MP] interface (nothing automatic: a human/agent invoked it). DEMO.
- Not exercised on Jetson: death during export, late conflicting/foreign-token registration, cancel. x86 demonstrated these; Jetson lab code is shared, not independently tested. Reconcile with the sampler alive was not tested.

## 6. Evidence durability limits
- Export dir and job state live on the same host as the runner. Verified failure domain: test-sandbox loss only (5c142dd5df). Jetson host loss, disk loss and SD/NVMe failure lose evidence; git is the only cross-host link (evidence/jetson-arm/*.jsonl, handoff docs). There is no live endpoint, no receipt, no acknowledgement. D024: evidence/results can reach git by accident (bulk build outputs nearly pushed, caught at push). Class: durability tiers, "received/accepted" acknowledgement states, retention = [MP] design + [LA] implementation (remote store). Status: NOT done; documented limits only.
- Manager should state per result a durability level (sandbox-survivable / host-local / replicated) rather than imply one. Evidence records must not be assumed present on the x86 host; the x86 view of a Jetson job today is whatever was committed.

## 7. Image identity under the Docker containerd store
- Event: image id in jobs = sha256:0b6c97b1... (job.json image_id). checkpoint-1 and bin/lab archive_ids: with the containerd snapshotter `.Id` is the manifest/index digest, not the config digest; RepoDigests holds a non-pullable local name@digest. bin/lab imageload therefore compares the loaded id to the archive's top-level index digest, and separately the archive's config digest.
- Design must: name digests by role: local_image_digest (what docker reports as Id on this host), config_digest (read from the archive), registry_digest (only after a push, otherwise absent). Never use RepoDigests as evidence of pullability. Immutable binding image name -> id remains (lab image add refuses rebinding).
- Class: envelope field names = [MP] (x86 owns the Build CC envelope, D016 follow-up); archive re-observation = [CC-op]. Status: DEMO that imageload works with the index-digest rule on Jetson (image used by all jobs); the field renaming is a request, NOT done. Jetson image archive sha256 323c787c... for jetson-c1 (note: job image_id above is the local id of the loaded image).

## 8. Build cache keying friction
- Event: J-T1b ccache fix (commit 5459da3 message; I did not re-read the fix) and the cost gap: first Jetson native build 52 min; patched rebuilds are cheap only with ccache. x86 saw 18 s vs 7-11 min (MANAGER-DESIGN D012-D015).
- Observed friction (from commit/checkpoint text, not re-measured here): cache keys depend on paths/host toolchain, so a different host or builder cannot reuse x86 caches, and an arm64 JetPack builder is host-native. Design must: treat build cache as per-(host, toolchain, target) state declared in the Build CC record, never a correctness input; record cache hit/miss. Class: [CC-op]/[COMP]. Status: PARTIAL (works on Jetson per commit 5459da3; cross-host cache sharing NOT done and probably not wanted).

## 9. Per-host single GPU slot
- Event: one integrated GPU, one 7.6 GB shared DRAM pool, no compute-app enumeration. Concurrency was never tested; D007 removed the lock and bin/lab prints "admission: none (gpu lock removed by user decision)" in every attempt. D023 shows a single test already consumes ~6.5 of 7.6 GB, so two concurrent jobs would OOM or contaminate each other (x86 R-C).
- Design must: host profile declares gpu_slots (=1 here) and a host memory budget; the Manager serialises attempts per host (queue or reject) and the rc-8 duplicate guard is only per job. Today the exclusivity is procedural. Class: slot count and queueing = [MP]; enforcement on the host = [LA]. Status: NOT done; the single-active-attempt guard is per-job only.

## Minimum Manager changes requested from x86
- Add an admission record (accepted / rejected / needs_information, coded reasons, host_profile_sha256, record version) as a precondition for any attempt; absent record never means accepted; keep re-evaluation at run time.
- Host profile schema: platform, gpu.request, gpu.sampler, gpu.memory_domain, arch, resources_max, gpu_slots; submit gets host_id; GPU request is expanded from the profile.
- Telemetry contract (resource-observe/0 or successor): scope field, null + reason for unavailable, required baseline, exclusivity as true/false/null (unknown never false).
- Result classifier: CUDA init failures on Jetson-style hosts must be reported as "GPU not exposed" candidates; never as driver mismatch.
- Rule that engine defaults depending on free memory are pinned by the test on shared-dram hosts; decide whether x86 gen_smoke pins -c (D023).
- Image identity fields by role (local_image_digest, config_digest, registry_digest); stop treating RepoDigests as pullable.
- Per-result durability level and a defined remote receipt (received / accepted / rejected / needs_information) before claiming cross-host evidence; keep the single-active-attempt + reconcile-before-retry rule from job-43ef050ce9.
- Per-host slot/queue policy (1 slot on Jetson) instead of the removed global lock.

## Explicitly not needed / avoid
- No host-specific branches in the Manager (if jetson ...); everything via profile fields.
- Do not pool or compare raw Jetson and x86 numbers (different weights file hashes bb010fb6 vs af76ade0, different domain); keep results per host.
- No heartbeats/leases for Jetson reasons; pid+start-time plus label/token ownership sufficed in job-43ef050ce9.
- Do not make agent reasoning part of admission; the observed decisions were deterministic functions of record + profile.
- Do not assume Jetson GPU memory numbers exist; do not synthesise zeros for them.
