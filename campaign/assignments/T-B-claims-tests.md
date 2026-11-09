# Assignment T-B (parent: campaign c1 / orchestrator, task stage1)
Question: which source claims are testable on a 16 GB RTX 5070 Ti with Granite or small fixtures, and which tests protect which behaviors?
Input: /home/muk/projects/Strata @ ffa80d73133ea5c548aefe57889a16e8085a938f (READ ONLY; do not build or run GPU code).
Deliverables under /home/muk/projects/siminfeng/campaign/artifacts/codexB/:
 1. claims.md — 8-12 candidate claims (docs/AI_REFERENCE.md, docs/DETAILS.md, code comments citing measurements). For each: ID, exact file:line, what it claims, conditions, applicability to Granite/this machine, a falsifiable local question, class (reproduction / related mechanism / new hypothesis). Treat historical-measurement comments as upstream claims, not facts.
 2. tests.md — inventory of tests under tests/ and tools/test_*: what each protects, whether it needs GPU, how to run it. Highlight tests/counterexamples for stale activation reuse, wrong expert identity, double-applied routing weights, incompatible geometry, unsafe async state change (cite them or say none found).
Write only under campaign/artifacts/codexB/. No network. Time limit: 40 min. Final message: files written + open questions.
