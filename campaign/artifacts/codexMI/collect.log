| Granite variant | Weights MiB | Experts MiB | Expert share | Active MiB/token | Host-expert GPU weight floor MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 2,547.20 | 2,304.00 | 90.45% | 819.20 | 243.20 |
| Q8_0 | 1,354.69 | 1,224.00 | 90.35% | 436.69 | 130.69 |
| Q4_K_M | 782.12 | 697.50 | 89.18% | 258.99 | 84.62 |
| mixE4K | 778.69 | 648.00 | 83.22% | 292.69 | 130.69 |
| split-n12 | 798.95 | 668.25 | 83.64% | 297.77 | 130.70 |

24 layers, 32 experts, top-8; embedding 1024; expert FFN 512; attention heads 16/8; head dimension 64; max context 131072.

MiB = 2^20 bytes. All values are weights only. Physical split storage and its proportional active estimate include 20.25 MiB of null experts; JSON additionally reports an active estimate excluding nulls.

Real validations: Granite variants and hybrid split. Qwen3 shared experts and legacy/current DeepSeek2 MLA are synthetic fixtures. Other named architectures have explicit pattern checks but no real GGUF in this assignment.
