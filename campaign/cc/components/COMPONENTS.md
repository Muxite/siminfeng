# Engine component versioning (T-COMP / E2)

The catalog is [components.json](components.json), validated by
[components.schema.json](components.schema.json) (JSON Schema 2020-12).
The engine source pin is `31288c9c66e59377f7788d920678a565bb89dd37`.
Source, extracted libraries, decisions and factory inputs were read only.
All work and CPU-only checks for this assignment write under `cc/components/`
and `artifacts/codexCOMP/`. No Docker, GPU, network or reference-tree edits were used.

A component is a separately identified change at one or more engine layers.
Its identity is the tuple of id, base commit, kind, content hashes, configuration,
prerequisites and qualification scope. An accepted status means acceptance within
that recorded scope; it does not qualify every combination of accepted components.
Rejected candidates remain in the catalog for experiments and must not be selected
automatically.

| Field | Meaning |
| --- | --- |
| `id` | Stable logical name, unique within this catalog. A new content hash/configuration identifies a new revision of the logical component. |
| `layers` | Every touched engine layer; `unmapped` explicitly preserves files outside the supplied mapping. |
| `kind` | `source-patch`, `binary`, `artifact` (including a production recipe), or `runtime-flag`. |
| `base_commit` | Full pinned engine commit against which the component was qualified. For infrastructure it records the engine context, not the shim's source commit. |
| `identities` | SHA256 of each patch, source, recipe, binary or provenance file. Hashes describe these files, not reconstructed provenance. Runtime flags have no file identity; their exact ordered argv/env/build configuration is their content. |
| `outputs` | Expected output names, ABI versions and observed hashes/paths when available. A null hash means not observed here, never wildcard compatibility. Empty outputs are valid for runtime flags. Artifact outputs use null ABI. |
| `ggml_abi_version` | Engine context `0.27.0`. Actual libllama and common outputs are `0.7.0`; the shim has an unversioned ABI. |
| `supported_models`, `supported_arch` | Measured qualification envelope, not a claim that other models/architectures cannot work. |
| `prerequisites` | Component references plus source, build, runtime, artifact, quality and measurement requirements. Component references must resolve in this catalog. |
| `config` | Ordered argv tokens, environment and build settings. Descriptive prerequisites may contain additional conditions. |
| `tests`, `evidence` | Qualification observations and evidence paths/decision IDs. Existing GPU measurements come from decisions; the new checks are CPU-only. |
| `status`, `scope` | Acceptance/rejection/provisional state and limits of the claim. |

The imatrix entry versions the calibration/quantization recipe and the observed
imatrix + mix4_imat GGUF, with source/calibration/evaluation hashes in the referenced
T-PAP provenance manifest. D035 accepts imatrix within its measured scope. Its PAP
lower-bit placement experiments have separate outcomes; this entry does not promote
provisional PAP speed results. The shim is infrastructure with a cooperative
intercepted-allocation cap. `GGML_CUDA_NO_VMM=ON` is a build requirement, and actual
interception must be verified; the cap does not include all physical GPU memory.

## Patch layers and composability

| Patch | Files / hunks | Layers and outputs |
| --- | --- | --- |
| q8_reuse | 3 / 10 | `backend-cuda` → libggml-cuda.so |
| kv_quant_fix | 1 / 1 | `ggml-core` (scheduler) → libggml-base.so |
| hybrid_split | 12 / 20 | `backend-cpu` → libggml-cpu.so; `libllama` → libllama.so; `build`; `unmapped` docs/ops.md and tests/test-hybrid-split.cpp |
| static_hot | 1 / 9 | `libllama` → libllama.so |

[Detailed file/hunk mapping](../../artifacts/codexCOMP/layer-map.json) includes
all four patches. Mapping follows the assignment table: CUDA/CPU subdirectories,
ggml backend/core/allocator files, src, common, tools and CMake. Documentation and
test source outside that table are reported as `unmapped`, not silently discarded.
All pairs in the requested stacks share neither files nor engine layers.
The static-hot patch is mapped but was not included in the requested application stacks.

`patch_stack.py` reads a pinned `git archive` into scratch, builds a private Git
index, and tests **every permutation** using `git apply --index`. Each order starts
from the same archived baseline, with no three-way merge or fuzz fallback. Git may
accept exact-context offsets. Pairwise hunk overlap is advisory, based on original
line intervals including insertion points; actual apply results are authoritative.
The report retains the failing patch and Git stderr for conflicts. All orders must
apply and have the same Git tree hash. Tree hashes include names, content, modes
and symlinks, and exclude Git metadata/timestamps. Scratch trees are removed after
inspection; JSON reports and release patch remain.

| Stack | Orders | Resulting Git tree hash |
| --- | --- | --- |
| q8_reuse + kv_quant_fix | 2/2 clean, identical | `954f7c74045ee731531b20ae68fd0a7701b48c4c` |
| q8_reuse + kv_quant_fix + hybrid_split | 6/6 clean, identical | `c15a1e300f4ce6d2ef62676ec4efbf686de03998` |

The archive baseline hash is `158a7ade930818efdb06c5561920a8d018d404ba`,
also the source commit's actual Git tree hash. See
[release report](../../artifacts/codexCOMP/release-stack.json) and
[multi-layer report](../../artifacts/codexCOMP/multi-layer-stack.json).
Order independence is a source-composition result, not runtime correctness,
performance acceptance or approval of hybrid_split.

The combined [release-stack-1.patch](release-stack-1.patch) is the staged diff of
the successful two-patch stack against the baseline. SHA256:

```text
2e340d159bc51d9918bfdf41327f989478daa52cd2dab3e876a8fe722d663031
```

A separate application of this combined patch reproduced the release tree hash;
see [release replay](../../artifacts/codexCOMP/release-replay.json).

## Binary experiment and ABI policy

`abi_diff.py` reuses T-N3's `cc/build_engine/host_linkage.py:inspect_elf` for dynamic
section validation and dependencies, extending that reader locally for SONAME,
defined global/weak/GNU-unique exports, visibility, ELF class/machine, and GNU
`.gnu.version` / `.gnu.version_d` export versions. ELF32/64 and both byte orders
are supported. Undefined, local, hidden and internal symbols are excluded. Function
code size/address changes do not change ABI identity; object/TLS size changes do.

The comparison direction is **before → after**. Added symbols are allowed.
Removed or changed existing symbol contracts, changed SONAME, DT_NEEDED,
architecture, library inventory or symlink aliases cause conservative rejection.
Unavailable/malformed ELF inspection is never compatible. Files are grouped by
resolved in-directory target; symlink aliases are recorded without counting the
same binary three times. Directory inputs must contain complete symlink targets.

Results for `artifacts/x7-comp/base` → `reuse`: 13 physical shared libraries
compared, 12 byte-identical, and only `libggml-cuda.so.0.27.0` differs.
SONAME (`libggml-cuda.so.0`), dependencies, architecture and all `ggml_*` exports
are unchanged. However, the full exported surface **does change**:

- Removed: two weak, unversioned `std::_Hashtable<..., unique_ptr<ggml_cuda_graph>, ...>` destructors (`D1` and `D2`).
- Added: the weak, unversioned corresponding `clear()` and a weak `std::_Hashtable<unsigned long, pair<const unsigned long, void*>, ...>::_M_rehash(...)`.

The [ABI report](../../artifacts/codexCOMP/abi-diff.json) contains the full mangled
names, versions, per-library hashes, SONAMEs, dependencies and export sets.
An independent [readelf check](../../artifacts/codexCOMP/readelf-crosscheck.txt)
confirms all four differences. This contradicts the assignment's expectation of
no exported-symbol change. The strict checker returns **incompatible** (CLI exit 1).
It does not filter away weak C++ exports to obtain the expected answer.

| CUDA file | SHA256 |
| --- | --- |
| base | `14a7d2d8f0c2cda1c1d1f172bdb5211ce4339259a883ae519b24eb3c29987634` |
| reuse | `f4011d9adf0926764924c2e73ed631c5182700e5be12ac603cee07cb81fd5735` |

D036 still provides empirical evidence for this exact engine/backend swap.
The new observation limits a general ABI guarantee: an arbitrary consumer could
reference a removed weak export. Matching symbol surfaces alone cannot prove C/C++
struct layout, semantics, toolchain ABI, CUDA behavior, dependency availability or
compatibility across ggml releases. No such claim is made here. Sectionless ELF
and exports with unresolved version definitions are explicitly unavailable.

## Reproduction and remaining gaps

Run from the campaign root. Python standard library and Git suffice for the two
tools; tests additionally use pytest and jsonschema. Tests reuse T-N3 synthetic
ELF fixtures, including a small compiler fixture when a C compiler is available.

```sh
nice -n 10 python3 -B cc/components/patch_stack.py \
  cc/q8_reuse/q8-activation-reuse.patch cc/kv_quant_fix/kv-quant-fix.patch \
  --out artifacts/codexCOMP/release-stack.json \
  --release cc/components/release-stack-1.patch
nice -n 10 python3 -B cc/components/patch_stack.py \
  cc/q8_reuse/q8-activation-reuse.patch cc/kv_quant_fix/kv-quant-fix.patch \
  cc/hybrid_split/hybrid-split.patch --out artifacts/codexCOMP/multi-layer-stack.json
nice -n 10 python3 -B cc/components/abi_diff.py \
  artifacts/x7-comp/base artifacts/x7-comp/reuse \
  --out artifacts/codexCOMP/abi-diff.json
# Exit 1 above is the observed conservative ABI rejection.
PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -B -m pytest \
  cc/components/test_components.py -q -p no:cacheprovider \
  --basetemp=artifacts/codexCOMP/pytest-tmp
```

The tools write only caller-specified output/scratch locations; defaults place
scratch under artifacts/codexCOMP. Keep output arguments within the assignment
roots. Git archive staging is private and the source repository is never mutated.
Permutation cost is factorial; intended stacks are small. Patch inventory accepts
ordinary unquoted unified file headers (including plain unified patches and Git
patches), and records hunks; it is not a general Git binary-patch parser.

The combined release patch has not been built or run on a GPU here. The KV fix's
rebuilt libggml-base hash is unobserved in this assignment and remains null.
No cross-version ABI guarantee or new performance/quality result is claimed.
Runtime flag and artifact qualification remains limited to the recorded models,
architecture and datasets. See [factory mapping note](FACTORY-MAPPING.md) for
catalog reconciliation; no factory file was edited.
