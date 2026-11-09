# Read-only reconciliation with T-FAC

Read `cc/factory/components.json` as a parallel-worker input. Snapshot SHA256:
`d669757702db4b6cbb36934d28c20cc92ad9f474ee742ecd3cd70f3b96084f75`.
Its schema is `factory-components/0`; engine base commit matches this catalog.
The mapping below records differences for the factory owner. No factory file
was edited, and this document does not change factory selection behavior.

| Factory id / kind | Component id / kind | Mapping and evidence |
| --- | --- | --- |
| q8_reuse / engine-patch | q8_reuse / source-patch | Same patch and accepted D029 scope. New strict exported-surface check rejects general binary substitution because weak C++ exports change; D036's measured swap remains valid evidence. |
| kv_quant_fix / engine-patch | kv_quant_fix / source-patch | Same patch. Factory snapshot says pending and CUDA qualification pending; D033 now records acceptance and 15/15 clean tests. Our status is accepted within D033 scope. |
| pap / model-transform | imatrix-quantization / artifact | Related recipe, not an exact alias. D035 accepts imatrix as a default; PAP12-Q3 speed remains provisional, other lower-bit variants fail quality. Do not promote the entire PAP family because imatrix is accepted. |
| hybrid_split / engine-patch | hybrid_split / source-patch | Both rejected, D030/D031. Clean source composition does not change rejection. |
| moe_static_hot / engine-patch | static_hot / source-patch | Logical rename; same rejected patch and failed engagement in D024. |
| gpu_budget / infrastructure | gpu_budget / binary | Same infrastructure intent, D021/D023. Versioned binary/source/recipe identities added. Cooperative interception cap; requires compile-time NO_VMM and linkage/engagement checks. |
| flash-attn / runtime-recommendation | flash-attn / runtime-flag | Same `-fa on`, D028. D033 supplies the later quantized-KV evidence missing from the snapshot. FA vs non-FA quality equivalence remains unmeasured. |
| absent | kv-q8 / runtime-flag | New explicit runtime component: `-fa on -ctk q8_0 -ctv q8_0`; depends on kv_quant_fix and flash-attn, D033. |

Factory statuses and human summaries cannot be silently substituted for component
qualification. A future importer should translate kinds and IDs, preserve rejected
statuses and scope, resolve prerequisites, and verify content hashes. It should
keep imatrix acceptance separate from placement-aware lower-bit model acceptance.
All status differences above are grounded in decisions rather than stale README
handoff notes. Re-read the factory input before implementing an importer because
T-FAC can continue changing it independently.
