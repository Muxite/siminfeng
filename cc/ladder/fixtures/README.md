These are byte-for-byte copies, not fabricated benchmark observations:

* `om/*/summary.json`: `experiments/jetson/OM/J-OM10c/*/summary.json`.
  Legacy Qwen3.6 Q4, 64 generated tokens, raw prompt, om10b-era image.
  Some newer metrics were not yet saved and remain null in fixture replay.
* The three Granite files: identically named files in `cc/acc/fixtures/`.
  Captured HTTP protocol and c128 KL examples; no dataset accuracy result.

`ladder.py report --fixtures --dry-run` shows these separately labelled legacy
configurations and a separate Granite KL row. It does not infer MMLU accuracy
from one response, assign Granite KL to Qwen, or enter fixtures in the frontier.
Tests may score the captured response against fixture targets to exercise the
aggregation interface; these are explicitly protocol tests, never benchmarks.
