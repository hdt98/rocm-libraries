# Target Sources

The active target contract is:

- `torchtitan_deepseek_v4_flash_12layer_ceiling_probe_bf16_4x8xmi350_canary.yaml`: real DeepSeek-V4-Flash first-12-layer BF16 ceiling probe for direct TorchTitan `torchrun`. It uses full dimensions, `S=4096`, `GBS=128`, `TP=PP=CP=1`, `FSDP=4/8`, `EP=4/8`, and the first-12-layer recipe mix `(1,1,4,128,4,128,4,128,4,128,4,128)`. The local TorchTitan code represents the two dense recipe entries as `0`, so its parsed ratio list must be `[0,0,4,128,4,128,4,128,4,128,4,128]`.

Retained baseline target:

- `torchtitan_deepseek_v4_flash_4layer_bf16_4x8xmi350_canary.yaml`: real DeepSeek-V4-Flash 4-layer BF16 canary target for direct TorchTitan `torchrun`. Its current best 4xMI350 MORI+AITER run clears the thread's token-throughput gate under 4-layer accounting, but it is now baseline evidence rather than the active promotion target.

The `torchtitan_deepseek_v3_16b_*` files in this directory are systems-proxy evidence only. They are useful for FSDP/EP/checkpoint plumbing, but they must not be promoted as the DeepSeek-V4 target or used as the source config for final DSv4 reporting.

Keep target artifacts compact and source-reviewable. Large checkpoints stay outside the repo.
