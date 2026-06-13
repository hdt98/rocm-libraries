# Ascend CANN DeepSeek-V4 Image Source Snapshot

Lightweight source-only snapshot extracted from the DeepSeek-V4 TorchTitan CANN recipe image. This is for implementation inspection only; the full image tarball and compiled kernels are not vendored here.

## Image

- Recipe: `sources/references/ascend-cann-recipes-train/llm_pretrain/deepseekv4/README.md`
- Image tarball URL: `https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/deepseek_train/dsv4_train_torchtitan_cann9.0.0_v3.0.tar.gz`
- Local tarball inspected: `/Users/sonle5/Documents/Codex/2026-06-01/files-mentioned-by-the-user-docs/work/dsv4_train_torchtitan_cann9.0.0_v3.0.tar.gz`
- Tarball SHA256: `184e6bb1e5b479191cdcf182dd246ec3edf164489892661056a76576dc268193`
- Image tag from Docker save manifest: `dsv4_train_torchtitan:cann9.0.0_v3.0`
- Architecture: `arm64`
- Image config: `65f542a2d955e5308763122ab392e2fb7771f52244b5223c678cadab6909b46d.json`
- Extracted layer: `2f5258b056ee7765d95db2c7c2c8157d8c7929f7e58d867931ff67629e80c289/layer.tar`

## Contents

This snapshot keeps the text/source files needed to inspect the DSv4 custom sparse attention path:

- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_lightning_indexer_grad_kl_loss/`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic/`
- `custom_transformer/op_proto/inc/`
- `mindspeed_ops/` sparse shared-KV and sparse LightningIndexer wrapper sources

Compiled artifacts such as `.o`, `.so`, `.a`, and `.pyc` were intentionally excluded. Snapshot size after trimming is about `556K`.

## Key Read

`SparseAttnSharedkvGrad` confirms the five-stage SASG backward structure in source:

- `cube1`: recomputes scores, `s = q * k^T`.
- `cube2`: computes output-gradient scores, `dp = dy * v^T`.
- Vector side computes softmax from `lse`, then softmax backward, sink gradients, and stores probability/score-gradient workspaces.
- `cube5`: computes `dv = p^T * dy`.
- `cube4`: computes `dk = ds^T * q`.
- `cube3`: computes `dq = ds * k`.
- `ScatterAdd`: reads `mm4ResWorkspaceGm` and `mm5ResWorkspaceGm`, scales `dk`, adds the value-side contribution, and atomically scatters into the original/compressed KV gradient workspace.

The useful AMD lesson is not "Ascend avoids scatter-add." It does not. The source shows `SetAtomicAdd<float>()` in the final scatter path. The difference is that the scatter happens inside one fused operator after Cube-core MFMA-like workspaces and Vector-core softmax/grad work, with only tile-local selected-K workspaces and ping-pong synchronization exposed inside the kernel.

For the MI350 sparse-MLA backward, this supports the current direction:

- Keep the query-major selected-key schedule and MFMA-shaped contribution math.
- Do not materialize large selected K/V or dense route-weight tensors at the PyTorch boundary.
- Fuse target-probability, KL derivative, LI/compressor gradients, and sparse-MLA dQ/dKV around the same tile schedule.
- Replace the current global many-to-one dKV atomic storm with tile-local grouped/segmented accumulation followed by a much smaller final scatter-add.

`SparseLightningIndexerGradKLLoss` also uses a fused gather/vector/cube/scatter structure. Its base source initializes gather workspaces, BMM workspaces, softmax max/sum inputs, `dWeight`, `dKeyIndex`, and `dQueryIndex`, then drives separate Vector and Cube services. This confirms that Ascend keeps LI backward integrated with its own workspaces rather than materializing target probabilities and derivatives as independent framework tensors.

## High-Value Files

- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/sasg_basic.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_op.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_modules/cube1.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_modules/cube2.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_modules/cube3.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_modules/cube4.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/cube_modules/cube5.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv_grad/arch22/basic_modules/vec_op.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_lightning_indexer_grad_kl_loss/sparse_lightning_indexer_grad_kl_loss_base.h`
- `custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_lightning_indexer_grad_kl_loss/sparse_lightning_indexer_grad_kl_loss_service_cube.h`

