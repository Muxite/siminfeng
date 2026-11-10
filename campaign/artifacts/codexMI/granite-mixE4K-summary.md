granitemoe: 24 layers, 32 experts, top-8
embedding=1024 FFN=512 expert-FFN=512 heads=16/8 head_dim=64 context=131072
| Weight role | MiB |
| --- | ---: |
| router | 3.00 |
| expert_ffn | 648.00 |
| norm | 0.19 |
| attn_qkv | 51.00 |
| attn_out | 25.50 |
| embed | 51.00 |
| TOTAL | 778.69 |
Expert share: 83.22%; active weight MiB/token: 292.69

| Weight placement floor | GPU MiB |
| --- | ---: |
| all_experts_on_host (f=0) | 130.69 |
| all_experts_on_gpu (f=1) | 778.69 |
| hot_fraction (f=0.25) | 292.69 |
| hot_fraction (f=0.5) | 454.69 |
| hot_fraction (f=0.75) | 616.69 |

| KV cache type | Bytes/token | MiB at model context |
| --- | ---: | ---: |
| f16 | 49152 | 6,144.00 |
| bf16 | 49152 | 6,144.00 |
| q8_0 | 26112 | 3,264.00 |
| q5_1 | 18432 | 2,304.00 |
| q5_0 | 16896 | 2,112.00 |
| q4_1 | 15360 | 1,920.00 |
| q4_0 | 13824 | 1,728.00 |

Floors exclude KV, runtime, compute buffers and transfer staging.
