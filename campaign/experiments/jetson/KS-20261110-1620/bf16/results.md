# CC kernel screen

Ratios are A/arm time; >1 means that arm is faster. Spread = 100*(max/min - 1).
A2 duplicates A for an interleaved A/A control when using default arms.

| Case | A median us / spread % | A2 median us / spread % | B median us / spread % | A/A2 | A/B |
|---|---|---|---|---|---|
| MUL_MAT(type_a=bf16,type_b=f32,m=1024,n=1,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 45.85 / 2.07 | 45.67 / 0.44 | 45.75 / 0.55 | 1.0039 | 1.0022 |
| MUL_MAT(type_a=bf16,type_b=f32,m=1024,n=8,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 10.56 / 2.52 | 10.35 / 3.34 | 10.44 / 4.91 | 1.0203 | 1.0115 |
| MUL_MAT(type_a=bf16,type_b=f32,m=512,n=1,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 27.99 / 4.51 | 27.82 / 0.69 | 27.47 / 1.71 | 1.0061 | 1.0189 |
| MUL_MAT(type_a=bf16,type_b=f32,m=512,n=8,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 11.10 / 5.45 | 11.28 / 0.80 | 10.83 / 3.23 | 0.9840 | 1.0249 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 106.32 / 19.43 | 104.48 / 0.66 | 105.21 / 12.42 | 1.0176 | 1.0106 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 388.99 / 15.71 | 372.19 / 2.49 | 353.15 / 2.74 | 1.0451 | 1.1015 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 109.87 / 20.42 | 110.77 / 2.35 | 109.68 / 5.77 | 0.9919 | 1.0017 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 367.53 / 18.47 | 364.13 / 10.65 | 384.64 / 5.60 | 1.0093 | 0.9555 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 99.36 / 0.99 | 97.98 / 5.54 | 96.83 / 2.09 | 1.0141 | 1.0261 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 335.68 / 9.75 | 316.11 / 3.70 | 321.64 / 3.60 | 1.0619 | 1.0437 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 101.41 / 4.41 | 101.90 / 4.03 | 102.48 / 4.35 | 0.9952 | 0.9896 |
| MUL_MAT_ID(type_a=bf16,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 327.39 / 5.89 | 331.18 / 3.18 | 324.54 / 8.28 | 0.9886 | 1.0088 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.52 / 1.42 | 3.54 / 2.90 | 3.55 / 2.85 | 0.9944 | 0.9915 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.18 / 0.63 | 3.15 / 5.67 | 3.17 / 6.69 | 1.0095 | 1.0032 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.25 / 5.08 | 3.17 / 4.42 | 3.24 / 9.94 | 1.0252 | 1.0031 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 2.88 / 6.09 | 2.87 / 5.04 | 2.92 / 5.84 | 1.0035 | 0.9863 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 9.60 / 8.59 | 9.51 / 3.15 | 9.52 / 4.94 | 1.0095 | 1.0084 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 8.63 / 7.46 | 8.82 / 0.23 | 8.58 / 3.86 | 0.9785 | 1.0058 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 6.00 / 2.92 | 6.03 / 1.17 | 5.99 / 1.69 | 0.9950 | 1.0017 |
| MUL_MAT_VEC_FUSION(type=bf16,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 5.49 / 0.36 | 5.28 / 6.07 | 5.51 / 7.21 | 1.0398 | 0.9964 |

Status: success

