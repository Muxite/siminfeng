# CC kernel screen

Ratios are A/arm time; >1 means that arm is faster. Spread = 100*(max/min - 1).
A2 duplicates A for an interleaved A/A control when using default arms.

| Case | A median us / spread % | A2 median us / spread % | B median us / spread % | A/A2 | A/B |
|---|---|---|---|---|---|
| MUL_MAT(type_a=q8_0,type_b=f32,m=1024,n=1,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 14.40 / 0.77 | 14.54 / 0.41 | 14.40 / 1.12 | 0.9904 | 1.0000 |
| MUL_MAT(type_a=q8_0,type_b=f32,m=1024,n=8,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 35.82 / 0.81 | 35.46 / 2.49 | 35.76 / 1.07 | 1.0102 | 1.0017 |
| MUL_MAT(type_a=q8_0,type_b=f32,m=512,n=1,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 19.04 / 10.00 | 19.04 / 8.70 | 19.06 / 0.79 | 1.0000 | 0.9990 |
| MUL_MAT(type_a=q8_0,type_b=f32,m=512,n=8,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 27.84 / 0.43 | 28.03 / 1.51 | 27.84 / 0.61 | 0.9932 | 1.0000 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 78.47 / 8.17 | 78.00 / 0.92 | 77.72 / 0.80 | 1.0060 | 1.0097 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 458.42 / 7.78 | 454.52 / 1.38 | 451.52 / 1.52 | 1.0086 | 1.0153 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 124.02 / 7.11 | 123.91 / 0.81 | 124.24 / 1.63 | 1.0009 | 0.9982 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 397.14 / 3.46 | 386.27 / 4.91 | 385.13 / 1.06 | 1.0281 | 1.0312 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 74.87 / 7.12 | 74.76 / 0.21 | 74.80 / 0.04 | 1.0015 | 1.0009 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 431.24 / 0.50 | 432.91 / 1.16 | 431.93 / 0.96 | 0.9961 | 0.9984 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 120.79 / 0.59 | 120.82 / 0.15 | 120.77 / 0.04 | 0.9998 | 1.0002 |
| MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 367.10 / 5.60 | 361.94 / 4.15 | 361.17 / 2.23 | 1.0143 | 1.0164 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.55 / 1.41 | 3.54 / 3.14 | 3.50 / 0.00 | 1.0028 | 1.0143 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.18 / 2.20 | 3.18 / 6.93 | 3.12 / 3.62 | 1.0000 | 1.0192 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.22 / 1.57 | 3.26 / 2.51 | 3.22 / 0.31 | 0.9877 | 1.0000 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 2.93 / 8.01 | 2.96 / 0.68 | 2.95 / 9.52 | 0.9899 | 0.9932 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 9.43 / 5.97 | 9.52 / 1.16 | 9.70 / 2.32 | 0.9905 | 0.9722 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 8.61 / 5.62 | 8.18 / 5.52 | 8.58 / 5.63 | 1.0526 | 1.0035 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 5.96 / 1.52 | 5.95 / 0.67 | 5.93 / 2.55 | 1.0017 | 1.0051 |
| MUL_MAT_VEC_FUSION(type=q8_0,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 5.58 / 7.06 | 5.44 / 6.06 | 5.48 / 0.37 | 1.0257 | 1.0182 |

Status: success

