# CC kernel screen

Ratios are A/arm time; >1 means that arm is faster. Spread = 100*(max/min - 1).
A2 duplicates A for an interleaved A/A control when using default arms.

| Case | A median us / spread % | A2 median us / spread % | B median us / spread % | A/A2 | A/B |
|---|---|---|---|---|---|
| MUL_MAT(type_a=q4_K,type_b=f32,m=1024,n=1,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 15.89 / 1.85 | 15.74 / 2.11 | 15.89 / 1.89 | 1.0095 | 1.0000 |
| MUL_MAT(type_a=q4_K,type_b=f32,m=1024,n=8,k=512,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 15.34 / 7.28 | 15.29 / 5.60 | 14.50 / 1.53 | 1.0033 | 1.0579 |
| MUL_MAT(type_a=q4_K,type_b=f32,m=512,n=1,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 12.85 / 9.83 | 13.60 / 10.14 | 12.93 / 2.94 | 0.9449 | 0.9938 |
| MUL_MAT(type_a=q4_K,type_b=f32,m=512,n=8,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 25.39 / 4.22 | 26.07 / 4.99 | 25.48 / 7.28 | 0.9739 | 0.9965 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 89.59 / 2.21 | 88.41 / 2.60 | 89.20 / 0.69 | 1.0133 | 1.0044 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=0,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 337.14 / 5.14 | 340.58 / 0.44 | 336.05 / 5.40 | 0.9899 | 1.0032 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 66.09 / 2.90 | 68.15 / 1.54 | 66.73 / 2.77 | 0.9698 | 0.9904 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 290.15 / 5.19 | 281.44 / 4.78 | 282.59 / 2.93 | 1.0309 | 1.0268 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=1,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 89.15 / 3.55 | 82.20 / 15.15 | 88.00 / 2.52 | 1.0845 | 1.0131 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=1,m=1024,n=8,k=512,amax=1.000000,m_v=0,ids_offset=0) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 310.26 / 9.04 | 321.33 / 8.46 | 316.34 / 3.71 | 0.9655 | 0.9808 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=1,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 65.06 / 7.63 | 55.07 / 16.15 | 66.97 / 4.17 | 1.1814 | 0.9715 |
| MUL_MAT_ID(type_a=q4_K,type_b=f32,n_mats=32,n_used=8,b=1,m=512,n=8,k=1024,amax=1.000000,m_v=0,ids_offset=0) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 271.57 / 8.73 | 275.03 / 4.51 | 273.09 / 3.05 | 0.9874 | 0.9944 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.72 / 5.18 | 3.74 / 1.08 | 3.84 / 2.86 | 0.9947 | 0.9688 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=1,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.64 / 10.85 | 3.78 / 1.32 | 3.66 / 6.23 | 0.9630 | 0.9945 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.40 / 2.40 | 3.31 / 3.93 | 3.42 / 4.18 | 1.0272 | 0.9942 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=1,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=1,n_used=8] | 3.34 / 7.67 | 3.37 / 5.52 | 3.27 / 6.33 | 0.9911 | 1.0214 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 9.82 / 0.31 | 9.82 / 1.65 | 9.81 / 1.97 | 1.0000 | 1.0010 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=8,n=1024,k=512,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=down,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 8.91 / 10.35 | 8.96 / 8.96 | 9.74 / 1.44 | 0.9944 | 0.9148 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=0,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 5.97 / 0.34 | 6.02 / 2.69 | 6.14 / 1.14 | 0.9917 | 0.9723 |
| MUL_MAT_VEC_FUSION(type=q4_K,glu_op=2,m=8,n=512,k=1024,use_id=1,n_mats=32,n_used=8,b=1,with_bias=0,with_gate=1,with_lane_scale=0,batch_dims=[1,1]) [direction=up,n_embd=1024,n_expert=32,n_ff=512,n_tokens=8,n_used=8] | 5.94 / 4.94 | 5.97 / 1.69 | 5.96 / 3.37 | 0.9950 | 0.9966 |

Status: success

