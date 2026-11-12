# om12 = factory-built image (2026-11-12)

The campaign build of om12 (combined 01..09) failed twice (nvcc not on PATH; then non-empty --out). Instead, the
CC factory (Stellven/jiuwenswarm `ai4r/iterative-cc-factory`, task `factory-d4b71bcce337406b96f627d469bec092`)
integrated patch 09 on top of `cf34d311 + combined-01..08` (owned checkout commit `690e4713a`, tree `43f95b17`)
and built it with the same `cc/build_engine/build_engine.py --target jetson-orin` recipe.

- image: `factory/llamacpp:d4b71bcc-thread-names-1` = `sha256:f02f8a93453e8b97ee852054ed2a4dfd8fa876cb89db1adbb04b8dd564836ba0`
- retagged as `campaign/llamacpp:jetson-up1108-om12` for the J-OM12 spec (same image Id; labels say patch_sha256 null because
  the change is a commit, not a --patch)
- build record + archive sha256: factory evidence bundle `artifacts/factory/evidence/factory-d4b71bcc.../builds/`
- correctness: greedy output byte-identical to om11 under contract qwen36-greedy-v1; xfill-* threads observed
