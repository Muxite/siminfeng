# T-VER execution report

Writes are confined to `cc/verdict/` and `artifacts/codexVER/`.
Implementation: `cc/verdict/verdict.py` (stdlib CLI, criteria/1 → verdict/1).
Criteria, documentation, authoring utility, and 26 regression/corruption tests
are alongside the implementation. Individual verdict JSON files and test log
are in this directory.

Read D026, D029, D033, D035, D040 plus D028 to reconcile KV acceptance wording;
read `bin/lab`, original parent-project agg.py/make_order.py/text_fp.py, and real
job results/logs for job-34c09bec98, job-a945fa8656, job-3d2fbbd061,
job-d2e342e743, job-85805b616c. The scripts were not in campaign/.claude; they
were located in ../.claude/skills/{frozen-experiment,engine-candidate}/scripts.
Lab state is campaign/lab as configured by bin/lab; LAB was unset.

Real-record reproduction:

- D029 q8_reuse PASS: +2.067782%, both model groups' text/PPL identity,
  secondary base/OFF/ON text identity, CUDA 2445/2445, monitored compiler count 0.
- D029 separate mixE4K track FAIL: +1.782166% < 2%.
- D033 reported PPL gate: q8_0 PASS (+0.009286%), q4_0 FAIL (+7.936592%).
- Full D028 check: q8_0 FAIL, q4_0 FAIL. q8_0 stdout differs from f16;
  KLD is pending/unrecorded. A full frozen-protocol q8_0 acceptance cannot be
  reproduced from these records. Scope is explicit in the separate JSON files.

No job records were fabricated. Missing KLD remains missing. D035/D040 monitors
lack a sample after the last test; full execution coverage fails closed. D029's
available host counter measures compiler activity, not downloads/CPU inference;
external backend/text sidecars are explicitly distinguished from lab evidence.

Validation command:

```sh
python3 -B -m unittest discover -s cc/verdict -p 'test_*.py' -v
```

26 tests pass. Fault injection operates on disposable copies of the real D029
job, removing/corrupting data rather than inventing measurements. Tests cover
real outcomes and values, equivalence to original aggregation, CLI exit/JSON,
PAP quality/rerun behavior, incomplete/missing/unclean evidence, unknown metrics
and operators, exact counts/order, nonfinite metrics, zero baseline, missing
KLD/counters/sidecars, backend selection, and monitor gaps.
