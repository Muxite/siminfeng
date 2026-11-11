---
name: engine-build
description: Use when building a pinned inference engine or applying a candidate patch through Build CC on this lab.
---

Work from /home/muk/projects/siminfeng/campaign.
Read cc/build_engine/README.md and c2/METHODS.md build identity/ops guidance.
Read decisions/D024-bulk-build-outputs-nearly-pushed.md and
decisions/D026-ccache-verified-binaries-not-reproducible.md.

1. Verify clean source HEAD matches the requested commit with git rev-parse/status.
   Keep ref trees read-only. Supply a separate patch; Build CC snapshots source
   and applies/checks the patch there. Choose a fresh output per attempt.
2. Use jetson-orin here (arm64/sm_87), not default x86-5070ti. Confirm required
   toolchain and digest-pinned runtime base from existing build evidence.
   BUILD_JOBS defaults to 3. Older sessions may need
   `DOCKER_CMD_PREFIX='sg docker -c'`; set it before submitting so jobq captures it.
3. Submit as build class, optionally --after a prerequisite job:
   ```sh
   bin/jobq submit --class build --name BUILD_ID --timeout 7200 -- \
     python3 cc/build_engine/build_engine.py --target jetson-orin \
     --source ref/llama.cpp --commit COMMIT --patch PATCH \
     --tag campaign/llamacpp:UNIQUE_TAG --out FRESH_OUTPUT
   ```
   Arrange a worker, then use jobq wait/show. GPU timing must not overlap builds.
   Omit --patch for baseline; add --extra-target for additional shipped tools.
   test-backend-ops requires --with-tests as well as --extra-target.
4. Inspect build-record.json status/failure_stage, logs, source/tree, patch SHA,
   toolchain/flags, base digest, target architecture, archive and ccache delta.
   Failure may leave an image/partial evidence; retry with a new output directory.
5. ccache keys use staged-source CCACHE_BASEDIR, NOHASHDIR=1 and
   COMPILERCHECK=content. D026 measured 348/360 hits, 141 s versus 3154 s cold.
   New source/CUDA changes may miss; counters are shared and can be distorted.
   Retained build directories refer to deleted staging and are not resumable.
6. Identity = commit + patch SHA + toolchain + base digest + build record/archive
   SHA. Binary hashes differ across equivalent-input attempts; do not use them
   as equivalence/deduplication. Keep local image/config/registry digests distinct.
7. Verify runtime and quality separately before claiming an engine result.
   D024: keep build/staging/archive outputs gitignored; never stage a directory
   a running worker writes. Inspect intended file diffs/staged stats before any
   separately authorized commit; do not alter evidence to make checks pass.
