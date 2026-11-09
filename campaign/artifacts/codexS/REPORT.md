# T-S completion

Implemented entirely within `cc/engine_handoff/` and `artifacts/codexS/`.
No Docker or network used. Existing `engine_handoff.py` remains unchanged.

Files delivered:
- `cc/engine_handoff/envelopes.py`: proposal/admit/result Python API and JSON CLI.
- `cc/engine_handoff/hosts/x86-main.json` and `hosts/jetson-arm.TEMPLATE.json`.
- `cc/engine_handoff/test_envelopes.py`: 24 offline checks.
- `cc/engine_handoff/fixtures/envelopes/`: real build identity excerpt, both jobs,
  three attempts, owner/state/result files and export manifests; supplemental
  current definitions and explicitly labelled example requester policy.
- `cc/engine_handoff/ENVELOPES.md`: API, field meanings, invocation mapping,
  fixture provenance, admission/evidence limitations, and receiver questions.
- `artifacts/codexS/{proposal-real,admission-jetson,result-interrupted,result-retry,result-completed}.json`:
  CLI examples from the specified real records.

Verification: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q cc/engine_handoff
--basetemp=cc/engine_handoff/.envelopes-test-tmp
-o cache_dir=cc/engine_handoff/.pytest_cache` -> **90 passed in 0.66s**.
This includes 24 new envelope tests and all 66 existing handoff checks.
Copied job/attempt/export fixture bytes were independently compared with their
sources; every copied file matched. All retained build excerpt fields matched.
`git diff --check -- cc/engine_handoff` passed. Temporary test outputs removed.

Proposal fields: format; campaign/request/experiment/attempt IDs; target
host/platform/capabilities/platform_compat; release role/source commit+tree/
patch SHA/CMake defines/transport kind+SHA+bytes/config ID/index digest/executable
hashes/build-record reference; input hashes; ordered test argv/environment/cache
state and role mounts; wall/capture/memory/concurrency policy; external-controller
and durable-output evidence contract; unresolved and computed dispatchable.

Result fields: format; IDs; execution_outcome; test_assessment; claim_category;
evidence_completeness (expected/started/clean/missing, manifest presence and journal
errors); last_durable_sequence (line count/last timestamp); resource_generation
(attempt/owner-token SHA); stop_confirmed; manifest/artifact path+SHA references.
No aggregate success boolean.

Observed fixture limitations: both legacy jobs lack spec.inputs; the supplied
build and historical job also name different image identities. These gaps stay
unresolved, so the real proposal is not dispatchable. AMD64/sm_120 is rejected
for the Jetson ARM64/sm_87 template. With an otherwise complete proposal, missing
transport SHA yields needs-information. Historical interrupted coverage is
expected=10, started=9, established clean=0; retry coverage is 10/10/10; the other
completed attempt is 27/27/27. Exit-zero clean completion and assessor outcome
remain separate. Evidence manifests identify exported bytes without claiming a
fresh artifact rehash or host-loss durability.

Open questions for the lab-agent team:
1. Which JetPack/L4T, runtime/driver, shared-DRAM accounting and memory guarantees
   belong in the Jetson profile?
2. How will role inputs be staged and archive/config/executable identities be
   independently checked before execution?
3. What acknowledgement, owner-generation binding and confirmed-stop evidence
   govern retries and stale results?
4. Which receiver services enforce wall/capture budgets and GPU concurrency, and
   where does durable evidence survive receiver host loss?
