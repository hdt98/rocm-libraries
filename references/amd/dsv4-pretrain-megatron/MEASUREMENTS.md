# DSV4 Reduced 12-Layer Measurements

This table is the measurement contract for the PyTorch oracle, the two
Megatron paths, and the TorchTitan path. Fill it with JSON emitted by
`tools/compare_logprobs.py` and render it with `tools/render_comparison_table.py`.

| Implementation | Recipe | Max logprob diff | Mean logprob diff | TFLOP/s/GPU | Status | Notes |
|---|---|---:|---:|---:|---|---|
| megatron_miles_pr1300_async_rl_train | `references/amd/miles-deepseek-v4-pr1300/scripts/amd/run_deepseek_v4.py` | TBD | TBD | TBD | not_run | Run on ROCm cluster and compare exported train-side logprobs against oracle. |
| megatron_pretrain_dsv4_reduced_12l | `references/amd/dsv4-pretrain-megatron/run_pretrain_dsv4_megatron.sh` | TBD | TBD | TBD | not_run | Run pretraining recipe with fixed batch and compare exported logprobs against oracle. |
| torchtitan_graph_trainer_dsv4_reduced_12l | `projects/torchtitan/examples/dsv4_pretrain/run_dsv4_reduced_12l.sh` | TBD | TBD | TBD | not_run | Run TorchTitan graph-trainer recipe and compare exported logprobs against oracle. |

Local validation in this worktree covered syntax and dry-run command assembly
only. Full correctness and TFLOP/s/GPU values require a ROCm training node with
the PR1300 Miles/Megatron runtime.
