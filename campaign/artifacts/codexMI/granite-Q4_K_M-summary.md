granitemoe: 24 layers, 32 experts, top-8
embedding=1024 FFN=512 expert-FFN=512 heads=16/8 head_dim=64 context=131072
| Weight role | MiB |
| --- | ---: |
| router | 3.00 |
| expert_ffn | 697.50 |
| norm | 0.19 |
| attn_qkv | 28.55 |
| attn_out | 13.50 |
| embed | 39.38 |
| TOTAL | 782.12 |
Expert share: 89.18%; active weight MiB/token: 258.99

| Weight placement floor | GPU MiB |
| --- | ---: |
| all_experts_on_host (f=0) | 84.62 |
| all_experts_on_gpu (f=1) | 782.12 |
| hot_fraction (f=0.25) | 258.99 |
| hot_fraction (f=0.5) | 433.37 |
| hot_fraction (f=0.75) | 607.74 |

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
