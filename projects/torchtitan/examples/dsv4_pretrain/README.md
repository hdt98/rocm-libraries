# DeepSeek V4 Reduced 12-Layer Pretraining

This example wires the requested reduced DeepSeek V4 Flash shape into
TorchTitan through the existing DeepSeek V3 MLA/MoE model path plus the
`sqrtsoftplus` router added for DSV4 parity.

Run with:

```bash
cd /Users/sonle5/.codex/worktrees/f232/rocm-libraries
NUM_GPUS_PER_NODE=8 TP=8 EP=8 STEPS=20 \
  projects/torchtitan/examples/dsv4_pretrain/run_dsv4_reduced_12l.sh
```

Useful local validation:

```bash
DRY_RUN=1 projects/torchtitan/examples/dsv4_pretrain/run_dsv4_reduced_12l.sh
```

The active TorchTitan config is:

```text
MODULE=graph_trainer.deepseek_v3
CONFIG=graph_trainer_deepseek_v4_reduced_12l
```

The model config captures the requested DSV4 reduced dimensions. The current
TorchTitan path does not implement the PR1300 Miles DSV4 hyper-connection or
compressed-attention plugin; use the Megatron recipe for exact DSV4 plugin
coverage and this TorchTitan recipe for the working TorchTitan MLA/MoE training
baseline.
