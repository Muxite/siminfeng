# Generic expert-cache replay

Python 3.10 standard library; no GPU, network, or third-party packages.
This is a logical simulation. Transfer counts and latency **predictions** are
assumptions, not measured GPU residency, performance, or speedup.

From the repository root:

```sh
python3 cc/expert_replay/expert_replay.py profile --trace trace.jsonl \
  --layers 24 --experts 32 --split 4:1 --weighting raw --out profile.json
python3 cc/expert_replay/expert_replay.py simulate --trace trace.jsonl \
  --layers 24 --experts 32 --split 4:1 --capacity-experts 128 \
  --policy static-profile:profile.json --warmup 2 \
  --expert-bytes 1048576 --bw-gbs 20 --compute-ms 5
python3 cc/expert_replay/expert_replay.py sweep --trace trace.jsonl \
  --layers 24 --experts 32 --capacities 0,32,64,128 --policy belady --out sweep.csv
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s cc/expert_replay -v
```

## Input and diagnostics

The primary input is the JSONL format described in `../routing_trace/README.md`:
`{"seq":0,"pos":0,"layer":0,"experts":[0,1],"weights":[0.5,0.5],"phase":"prefill"}`.
Unlike the original model-specific replay, any nonempty distinct top-k is valid.
Weights, when present, must be aligned, finite and nonnegative; they do not
weight cache accesses. The simple form is also supported:
`{"tok":0,"layer":0,"experts":[0,1],"phase":"decode"}`.
`request` is an optional integer alias for `seq`; absent request IDs mean one
request. IDs and positions must be nonnegative integers. Phases are required.

Provide both dimensions with flags or a metadata line:
`{"type":"metadata","layers":24,"experts":32}`. No model dimensions are
hardcoded or inferred from observed maxima. Conflicting metadata/flags and
out-of-range layer/expert IDs fail with exit code 2.

Malformed records (including missing fields, duplicates, invalid weights and
invalid JSON) are skipped, counted and reported with line numbers in JSON and
on stderr. An invalid final line lacking a newline explicitly increments
`possible_truncated_tail`. A valid final JSON object needs no newline. The
parser reports tokens with incomplete layer coverage separately; it cannot
detect an entirely missing token or a file truncated exactly at a record
boundary. Capture success and expected token counts remain external checks.
No valid records, inconsistent token phases, missing dimensions, invalid
configuration and I/O failures return nonzero. A successful replay with
reported errors is a partial-data result, not proof of a complete capture.

## Replay conventions

Records are ordered by `(seq, position, layer)` and expert accesses follow array
rank order, sequentially, including within a token. This deliberately differs
from routing_trace's atomic token-entry scoring. All policies start cold and
keep state across phases and requests. `--warmup N` replays the first N test
tokens without scoring them. It does not warm from the training partition.
Tokens include all their available layers; rates aggregate expert requests,
not averages of layer rates. Missing rates are JSON null. Per-layer metrics
use tokens observed at that layer; overall metrics count each token once.

`--capacity-experts C` is a global number of `(layer,expert)` slots.
`--per-layer-capacity` instead assigns **C slots to each layer** (up to L*C
total), rather than dividing C among layers. Zero capacity is supported.

* `lru`: evict least recently accessed resident.
* `lfu`: resident-only access counts, reset on eviction; LRU breaks ties.
* `lfu-decay:H`: same counts with continuous exponential decay, half-life H
  global expert lookups. Counts are evaluated lazily at eviction, avoiding
  growing exponentials. LRU breaks ties. This differs from periodic halving.
* `static-profile:PATH`: pin the top C ranked pairs globally or C per layer.
  Selected pairs enter on first demand; other misses bypass. No free preload
  hits or profile updates. Profiles must match dimensions.
* `belady`: **reads-the-future upper bound**, not deployable online. Evict the
  resident or bypass the incoming pair according to furthest next use. Allowing
  bypass lets the oracle bound pinned static caching as well as admitting
  policies under the same cold-start and capacity conventions.

Belady sees the replay partition, including warmup, not training records.
All traces are held in memory. LFU/Belady victim selection scans resident slots;
this is a correctness-oriented reference simulator, not a runtime cache.

## Training and profiles

`--split train:test` gives nonnegative training and positive test weights;
`4:1` selects the first floor(4/5 * token_count) whole tokens for training.
Token order is the normalized `(seq,pos)` order, so sequence IDs should reflect
the desired request order. Simulate/sweep only replay the test suffix. Profile
uses only the training prefix when split is supplied, otherwise all records.
No splitting inside a token, test-based ranking, or training-state warming.

Profiles rank all L*E pairs by score, then `(layer,expert)` for ties, including
zero-score pairs. `raw` counts expert accesses. `per-request` divides each
request's counts by its total expert accesses so every request has unit mass;
provide distinct `seq`/`request` IDs for distinct requests. The weighting and
scores are stored explicitly. Strata's make_profile uses raw counts and can
preserve an existing base ranking; make_profile_mix normalizes per input file
and applies file weights. This implementation has no base ranking and
normalizes by request, not file.

Profiles retain SHA-256 fingerprints of normalized training records. Split
replay rejects profiles without provenance or with any test-record overlap,
including warmup records. This catches accidental reuse of a full-trace
profile; it is not an authentication mechanism or proof of independence
between semantically equivalent traces with changed IDs. Without `--split`,
the caller owns train/test separation.

## Bytes and prediction

A miss moves `--expert-bytes B` bytes (default 0). Optional
`--per-layer-bytes B0,B1,...` overrides it with exactly L nonnegative integer
sizes. No cache writeback, prefetch, overlap, compression or fixed overhead is
modeled. JSON reports overall, per-layer and decode-only hit rate, misses per
token and bytes per token. Decode predictions use decode-only moved bytes:

`prediction seconds/token = compute_ms/1000 + decode_bytes_per_token/(bw_gbs*1e9)`

`--bw-gbs` is decimal GB/s (default 1); `--compute-ms` defaults to 0.
The prediction is null if no scored decode tokens exist. Sweep writes one CSV
row per capacity with overall metrics, decode hit rate, decode prediction and
malformed/truncation counts. Full line diagnostics remain on stderr.

## Credit and verification

Ideas credited to **Strata (MIT license)**: `ref/Strata/tools/bench_eviction.py`,
`replay_cache_policy.py`, `make_profile.py`, and `make_profile_mix.py`.
The references informed logical pair-key replay, policy comparisons, offline
oracle bounds and frequency-profile weighting. This is an independent
implementation; no reference files were copied wholesale or modified.

Tests include hand-computed counts for every policy, decay behavior, exhaustive
oracle comparisons on all 729 length-six traces over three experts and four
capacities, train/test and warmup isolation, profile leakage rejection,
primary/simple formats, malformed/truncated tails, byte predictions, weighting,
CLI JSON/CSV, dimensions and invalid configuration.

## Speculative verify windows

`python3 cc/expert_replay/expert_replay.py specwin --trace routing.jsonl --out results.json`

Repeat `--trace` for multiple workloads (the parent directory names label them).
This decode-only sweep covers W=1,2,3,4,6,8, acceptance=.5,.7,.9,
LRU/Belady and global capacities 650/1300/2600 expert slots. W counts total
verify positions; expected output is `(1-a**W)/(1-a)`, including one guaranteed
position. Partial windows use actual length. Requires explicit decode phases
and complete per-token layer coverage, and refuses malformed records.
The pessimistic variant adds seeded independent random top-k routes after a
sampled prefix rejection, retaining realized routes. `--seed` defaults to
20261110. It is one conservative realization, not a measured draft trace.
See `experiments/jetson/SPEC1/SPEC1.md` for assumptions and full tables.
