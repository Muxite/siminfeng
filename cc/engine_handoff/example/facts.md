# Engine handoff facts

Values describe bytes or claims; no execution was performed.

- `archive.artifact` [observed]: "/home/muk/projects/siminfeng/campaign/cc/engine_handoff/fixtures/engine.tar"
- `archive.sha256` [observed]: "c55b7e4e07f28dc721ca49317d149c6b1c273dd3b658c83f294f82d617e5c400"
- `archive.bytes` [observed]: 20480
- `build_record.artifact` [observed]: "/home/muk/projects/siminfeng/campaign/cc/engine_handoff/fixtures/build-record.json"
- `build_record.sha256` [observed]: "26390063c1d8c4945a1eacd168b980fa60fa2028950d32254db304ed06d2c727"
- `build_record.claims.image_config_id` [declared]: "sha256:f3428dc085e0800c14b72be27aa43110a21c4abfe380501ee9ffe29a465fa572"
- `build_record.claims.archive_sha256` [declared]: "c55b7e4e07f28dc721ca49317d149c6b1c273dd3b658c83f294f82d617e5c400"
- `build_record.claims.archive_bytes` [declared]: 20480
- `build_record.claims.executable_sha256` [declared]: {"/engine/run": "6e148a66f68c95f6ddb721ce30c72fb7c91b6ce3d57ec9533c66aa5320721ad6"}
- `build_record.claims.platform` [declared]: "linux/amd64"
- `build_record.claims.base_image_digests` [declared]: []
- `build_record.claims.source_commit` [declared]: "fixture-not-a-real-build"
- `build_record.claims.patch_sha256` [declared]: null
- `build_record.claims.status` [declared]: "fixture-only"
- `model_config.artifact` [observed]: "/home/muk/models/campaign/granite-3.1-1b-a400m-instruct/config.json"
- `model_config.sha256` [observed]: "f1b64813debfabeb9cc20f1a13ed8248da6774fe54e532d19912d4b3f2ab335d"
- `model_config.values.architectures` [observed]: ["GraniteMoeForCausalLM"]
- `model_config.values.attention_bias` [observed]: false
- `model_config.values.attention_dropout` [observed]: 0.0
- `model_config.values.attention_multiplier` [observed]: 0.015625
- `model_config.values.bos_token_id` [observed]: 0
- `model_config.values.embedding_multiplier` [observed]: 12.0
- `model_config.values.eos_token_id` [observed]: 0
- `model_config.values.hidden_act` [observed]: "silu"
- `model_config.values.hidden_size` [observed]: 1024
- `model_config.values.initializer_range` [observed]: 0.02
- `model_config.values.intermediate_size` [observed]: 512
- `model_config.values.logits_scaling` [observed]: 6.0
- `model_config.values.max_position_embeddings` [observed]: 131072
- `model_config.values.model_type` [observed]: "granitemoe"
- `model_config.values.num_attention_heads` [observed]: 16
- `model_config.values.num_experts_per_tok` [observed]: 8
- `model_config.values.num_hidden_layers` [observed]: 24
- `model_config.values.num_key_value_heads` [observed]: 8
- `model_config.values.num_local_experts` [observed]: 32
- `model_config.values.output_router_logits` [observed]: false
- `model_config.values.pad_token_id` [observed]: 0
- `model_config.values.residual_multiplier` [observed]: 0.22
- `model_config.values.rms_norm_eps` [observed]: 1e-06
- `model_config.values.rope_scaling` [observed]: null
- `model_config.values.rope_theta` [observed]: 1500000.0
- `model_config.values.router_aux_loss_coef` [observed]: 0.0
- `model_config.values.tie_word_embeddings` [observed]: true
- `model_config.values.torch_dtype` [observed]: "bfloat16"
- `model_config.values.transformers_version` [observed]: "4.47.0"
- `model_config.values.use_cache` [observed]: true
- `model_config.values.vocab_size` [observed]: 49155
- `gguf.arch` [observed]: "granitemoe"
- `gguf.file_type` [observed]: 32
- `gguf.expert_count` [observed]: 32
- `gguf.expert_used_count` [observed]: 8
- `gguf.block_count` [observed]: 24
- `gguf.context_length` [observed]: 131072

## Unresolved

- build_record.image_config_id: declared (not independently verified)
- build_record.executable_sha256: declared (not independently verified)
- build_record.platform: declared (not independently verified)
- build_record.base_image_digests: declared (not independently verified)
- build_record.source_commit: declared (not independently verified)
- build_record.patch_sha256: declared (not independently verified)
- build_record.status: declared (not independently verified)
