# D006 — BUG found by recovery test: reconcile overwrote a completed result
- Case: runner SIGKILLed during export (LAB_FAULT=die_in_export), job-9252e2d8bf att-1. Tests had finished; result.json existed.
- Observed: first reconcile implementation classified any dead non-terminal attempt as "interrupted" and wrote a stub result.json over the valid one; the staged copy was then rebuilt from the overwritten dir. Valid result lost (that attempt's original quality data is unrecoverable; ppl2 is deterministic so re-run is cheap, but in general not).
- Root cause: reconcile keyed on "last state", not on "which artifacts exist"; export() deletes/recreates staging.
- Fix: if result.json exists, reconcile only resumes export; result files are write-once; never rescore. Re-verified below (job R3b).
- Lab-Manager requirement (demonstrated by failure): phase-specific recovery; write-once result artifacts; staging must not be rebuilt from mutated source; recovery must be tested per phase, not just "runner died".
