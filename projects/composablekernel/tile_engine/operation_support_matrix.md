# CK Tile Engine Operation Support Matrix

This matrix shows all CK Tile operations with per-data-type, per-layout, and per-GPU support status. It uses a three-state convention: ✅ = supported by both CK Tile and tile engine, ❌ = supported by CK Tile but not yet in the tile engine, blank = not supported by CK Tile itself.

| Op | CK Tile Kernel | fp16 | fp8 | bf16 | bf8 | int8 | fp4 | fp6 | rcr | rrr | ccr | crr | 90a | 942 | 950 | 1201 |
| :--: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Activation | softmax | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Activation | topk_softmax<br>example: `09_topk_softmax/` | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Attention | fmha<br>example: `01_fmha/` | ❌ | ❌ | ❌ | ❌ | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Attention | sparse_attn<br>example: `50_sparse_attn/` | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Conv | grouped_conv<br>example: `20_grouped_convolution/` | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | batched_transpose<br>example: `35_batched_transpose/` | ❌ | ❌ | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | image_to_column<br>example: `04_img2col/` | ❌ | | | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | permute<br>example: `06_permute/` | ❌ | ❌ | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Elementwise | elementwise<br>example: `21_elementwise/` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_multi_d [1]<br>engine: `gemm/gemm_multi_d/`<br>example: `19_gemm_multi_d/` | ✅ | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| GEMM | gemm_preshuffle [2]<br>engine: `gemm/gemm_preshuffle/` | ✅ | ✅ | ✅ | ✅ | | | | ✅ | | | | ✅ | ✅ | ✅ | ❌ |
| GEMM | gemm_universal [3]<br>engine: `gemm/gemm_universal/`<br>example: `03_gemm/` | ✅ | ✅ | ✅ | ✅ | ❌ | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GEMM | streamk_gemm [4]<br>engine: `gemm_streamk/`<br>example: `40_streamk_gemm/` | ✅ | ✅ | ✅ | ✅ | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GEMM | batched_contraction<br>example: `41_batched_contraction/` | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | batched_gemm<br>example: `16_batched_gemm/` | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | block_scale_gemm<br>example: `38_block_scale_gemm/` | ❌ | ❌ | ❌ | ❌ | | ❌ | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | flatmm<br>example: `18_flatmm/` | ❌ | ❌ | ❌ | ❌ | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_multi_abd<br>example: `22_gemm_multi_abd/` | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_quant | | | | | | ❌ | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm<br>example: `17_grouped_gemm/` | ❌ | ❌ | ❌ | ❌ | | | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm_quant | | | | | | | | | | | | | | | |
| MoE | fused_moe<br>example: `15_fused_moe/` | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | add_rmsnorm2d_rdquant<br>example: `11_add_rmsnorm2d_rdquant/` | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | layernorm2d<br>example: `02_layernorm2d/` | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | norm_reduce | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | rmsnorm2d<br>example: `10_rmsnorm2d/` | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Pooling | pooling<br>example: `36_pooling/` | ❌ | | | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Quant | smoothquant<br>example: `12_smoothquant/` | ❌ | ❌ | ❌ | ❌ | ❌ | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Reduce | multi_reduce2d [5]<br>engine: `reduce/`<br>example: `05_reduce/` | ✅ | | ❌ | | | | | | | | | ❌ | ✅ | ✅ | ❌ |
| Reduce | reduce2d<br>example: `05_reduce/` | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |

**Notes:**

- [1] **gemm_multi_d:** Builder only accepts `fp16`. The 4-character layout codes (e.g., `rcrr`) include a D tensor layout (always `r`). CMake does not include gfx1201 in `DESIRED_TARGETS`; gfx1201 has fp16 entries in the shared `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS` but the gemm_multi_d builder does not target it.
- [2] **gemm_preshuffle:** Only layout `rcr` is supported due to preshuffle data format requirements. No dedicated example directory exists. CMake `DESIRED_TARGETS` includes gfx90a, gfx942, and gfx950. The `GEMM_PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS` dict has entries for fp16, bf16, fp8, and bf8 on gfx90a/gfx942/gfx950 but not gfx1201. Int8 entries exist in the shared preshuffle dict for gfx942 only, but the builder does not accept `int8` in its `--datatype` choices.
- [3] **gemm_universal:** CMake defaults to `fp8;fp16` but the builder also accepts `bf16` and `bf8` (override with `-DGEMM_UNIVERSAL_DATATYPE`). The `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS` dict has `int8_int8_int32` entries for gfx942, but the builder does not accept `int8` — CK Tile's universal GEMM kernel is type-generic so int8 is marked ❌. gfx1201 only has fp16 tile configurations.
- [4] **streamk_gemm:** CMake defaults to `fp8;fp16` and layout `rcr`, but the builder accepts all four layouts and also `bf16`/`bf8` (override with `-DGEMM_STREAMK_DATATYPE` and `-DGEMM_STREAMK_LAYOUT`). `DESIRED_TARGETS` is `gfx90a;gfx942` — gfx950 support is pending (TODO in CMakeLists.txt). The builder also lists `fp32` and `fp64` in its `--datatype` choices, but no tile configurations or CK Tile example implementations exist for those types. The streamk validation dict has `int8_int8_int32` for gfx942 but the builder does not accept `int8`.
- [5] **multi_reduce2d:** The tile engine only supports `fp16` (per `reduce_parameter.py` TYPE_MAP). CK Tile also implements bf16 (referenced in example code), but the tile engine has no bf16 configurations. `DESIRED_TARGETS` is `gfx942;gfx950` — gfx90a is not included.

**Legend:**
- **CK Tile Kernel column:** First line is the kernel name. Lines prefixed with "engine:" show the tile engine directory under `ops/`. Lines prefixed with "example:" show the CK Tile example directory under `example/ck_tile/`.
- **Green cell** (✅): CK Tile implementation exists **and** the tile engine supports it.
- **Red cell** (❌): CK Tile implementation exists **but** the tile engine does **not** support it.
- **Grey cell** (blank): No CK Tile implementation exists for this combination.

**Layout Codes:**

Each layout code is a 3-character string where each character specifies row-major (`r`) or column-major (`c`) for tensors A, B, and C respectively. For example, `rcr` means A=row-major, B=column-major, C=row-major. For `gemm_multi_d`, a 4-character code is used where the 4th character is the D tensor layout (always `r`). Layout columns only apply to GEMM-category operations; all other operations leave these columns blank.

**Data Type Mapping:**

| Config Label | A (source) | B (source) | Acc | C (output) |
| --- | --- | --- | --- | --- |
| fp16 | fp16 | fp16 | fp32 | fp16 |
| bf16 | bf16 | bf16 | fp32 | bf16 |
| int8 | int8 | int8 | int32 | int32 |
| fp8 | fp8 | fp8 | fp32 | fp16 |
| bf8 | bf8 | bf8 | fp32 | fp16 |
| fp6 | fp6 | fp6 | fp32 | fp32 |
| fp4 | fp16/bf16 | fp4 | fp32 | fp16/bf16 |
