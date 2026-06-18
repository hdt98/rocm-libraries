# Miles DeepSeek V4 ROCm Reference

This directory snapshots the AMD DeepSeek V4 recipe files from:

- Repository: `https://github.com/radixark/miles`
- Pull request: `radixark/miles#1300`
- PR head: `0dc10df6488aab5a08d883d7eebb1565303158fd`
- PR base merge-base: `74198b451f8bb63889e416b8686844630bd0fc92`

The copied PR range is:

```text
0dc10df6 Tune AMD DeepSeek V4 runtime defaults
71364662 Add AMD DeepSeek V4 launcher
53f0737d Patch miles runtime
b0ce0be3 Add DeepSeek V4 FP8 Docker runtime patches
f909d28c Add ROCm DeepSeek V4 Dockerfile
```

Files copied verbatim from the PR:

```text
docker/Dockerfile.rocm_MI350-5_DSV4
docker/amd_patch/deepseek_v4/megatron_dsv4_compat.patch
docker/amd_patch/deepseek_v4/megatron_grad_clip.patch
docker/amd_patch/deepseek_v4/miles.patch
docker/amd_patch/deepseek_v4/sglang_dsv4_compressor_fp32.patch
docker/amd_patch/deepseek_v4/sglang_post_process_weights.patch
docker/amd_patch/deepseek_v4/tile_kernels_lazy_import.patch
scripts/amd/run_deepseek_v4.py
```

The launcher is an async RL recipe: it prepares DeepSeek V4 checkpoints, configures
ROCm/Megatron/SGLang defaults, and runs Miles `train_async.py` with separate actor
and rollout resources. Pretraining-only recipes in this repository reuse the
ROCm and Megatron training portions without the rollout or GRPO path.
