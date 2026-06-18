# DeepSeek V4 Reduced Pretraining on Megatron-LM

This recipe reuses the Megatron training portion of
`references/amd/miles-deepseek-v4-pr1300/scripts/amd/run_deepseek_v4.py` and
turns it into a pretraining-only entrypoint:

- no rollout workers
- no GRPO or advantage estimator
- no SGLang serving path
- deterministic ROCm/Megatron defaults from the Miles PR1300 launcher
- the requested 12-layer reduced DeepSeek V4 Flash shape

The model shape is:

```text
layers=12
hidden_size=4096
vocab_size=129280
num_attention_heads=64
q_lora_rank=1024
kv_lora_rank=512
qk_nope_head_dim=448
qk_rope_head_dim=64
v_head_dim=512
num_experts=256
moe_ffn_hidden_size=2048
moe_router_topk=6
seq_length=4096
dsv4_compress_ratios=[0,0,4,128,4,128,4,128,4,128,4,128]
```

The launcher defaults to `--mock-data` and `NullTokenizer` so the first run is a
fixed-shape systems test. Set `DATA_PATH=/path/to/indexed/dataset_prefix` to use
a real Megatron indexed dataset.

Example:

```bash
cd /Users/sonle5/.codex/worktrees/f232/rocm-libraries
NUM_GPUS_PER_NODE=8 \
TP=8 EP=8 MBS=1 GBS=8 TRAIN_ITERS=20 \
references/amd/dsv4-pretrain-megatron/run_pretrain_dsv4_megatron.sh
```

The real DSV4 Megatron block comes from the Miles plugin spec used by PR1300:

```text
miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec
```

By default the launcher adds `references/nvidia/miles-deepseek-v4-pr1045` to
`PYTHONPATH` if it exists, because this repository already contains a fuller
Miles snapshot with `miles_plugins`. Override with `MILES_PLUGIN_PATH=/path/to/miles`
or set `MEGATRON_DSV4_SPEC=""` for parser/config dry runs that do not import the
plugin.
