# Example 03 — `rocm_ck` Vector Add (Full Tuning Surface)

Demonstrates CK Tile's **Signature + Algorithm** pattern for distributing
pre-compiled GPU kernels with full control over the ElementWise tuning surface.

## Key Concepts

### Signature / Algorithm Separation

Following CK Tile's builder pattern, kernel configuration is split into two
orthogonal concerns:

- **Signature** (`ElementwiseSignature`): *What* the kernel computes — data types.
- **Algorithm** (`ElementwiseAlgorithm`): *How* the kernel executes — tile geometry,
  warp count, vector width, padding.
- **Config** (`ElementwiseConfig`): User-facing API combining Signature + Algorithm.

This separation lets the same operation (vector add) be compiled with many
different tuning configurations, each producing a distinct `.hsaco` binary.

### Tuning Surface

The algorithm exposes four independent knobs matching CK Tile's `ElementWiseShape`:

| Parameter | Field | Description |
|-----------|-------|-------------|
| BlockTile | `block_tile` | Elements processed per thread block |
| BlockWarps | `block_warps` | Warps per thread block (must be power of 2) |
| WarpTile | `warp_tile` | Controls vector load width (`kVectorM`) |
| Pad | `pad` | Enable padding for unaligned problem sizes |

Derived quantities (validated at compile time by `make_kernel`):
- `kVectorM = min(128 / type_bits, warp_tile / 64)` — vector elements per thread
- `kRepeatM = block_tile / (block_warps × kVectorM × 64)` — iterations per warp
- `thread_block_size = 64 × block_warps` — actual threads launched

### Variant Registry

`rocm_vector_add_registry.hpp` provides programmatic variant selection:
- `ALL_VARIANTS[]` — constexpr table of all compiled kernel variants
- `findVariant(DataType, problem_size)` — selects the best variant: largest
  `block_tile` that divides `problem_size` cleanly, or padded fallback

## Compiled Variants

| Variant | Type | BlockTile | BlockWarps | WarpTile | Threads |
|---------|------|-----------|------------|----------|---------|
| `fp32_b256` | FP32 | 256 | 1 | 256 | 64 |
| `fp32_b512` | FP32 | 512 | 1 | 512 | 64 |
| `fp32_b1024` | FP32 | 1024 | 1 | 1024 | 64 |
| `fp16_b512` | FP16 | 512 | 1 | 512 | 64 |
| `fp16_b1024` | FP16 | 1024 | 1 | 1024 | 64 |
| `bf16_b512` | BF16 | 512 | 1 | 512 | 64 |
| `fp32_b256_sa` | FP32 | 256 | 1 | 256 | 64 |
| `fp32_b2048_w8` | FP32 | 2048 | 8 | 64 | 512 |
| `fp16_b1024_w2` | FP16 | 1024 | 2 | 512 | 128 |

The `_sa` suffix indicates explicit Signature+Algorithm API (functionally
identical to `fp32_b256`). The `_w8`/`_w2` suffixes indicate multi-warp variants.

## File Roles

| File | Purpose |
|------|---------|
| `rocm_vector_add_api.hpp` | Shared ABI, Config/Signature/Algorithm types, `make_kernel` validation |
| `rocm_vector_add_dev.hpp` | Device interface — maps `VectorAddKernel` to CK Tile types |
| `rocm_vector_add_registry.hpp` | `VariantDescriptor` table and `findVariant` selection (host-only) |
| `vector_add_*.hip` | Variant instantiations (~15 lines each) |
| `pack.py` | Archive packer with variant metadata |
| `main.cpp` | Host loader — variant selection demo + verify-all mode |
| `CMakeLists.txt` | Build system (variant × arch nested loop) |

## Build

```bash
cd experimental/kpack/examples/03_rocm_ck_vector_add
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"
ninja -C build
```

## Run

```bash
./build/kpack_rocm_ck_vector_add build/kernels.kpack
```

Expected output:
```
Opened build/kernels.kpack — architectures: gfx90a, gfx942

Variant selection for N=4096:
  FP32 -> vector_add_fp32_b2048_w8 (tile=2048, warps=8)
  FP16 -> vector_add_fp16_b1024 (tile=1024, warps=1)
  BF16 -> vector_add_bf16_b512 (tile=512, warps=1)

Running all 9 variants:
  vector_add_fp32_b256: tile=256, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp32_b256 (grid=16, block=64): PASSED
  ...
  vector_add_fp32_b2048_w8: tile=2048, warps=8, threads=512, N=4096 (aligned)
  vector_add_fp32_b2048_w8 (grid=2, block=512): PASSED
  vector_add_fp16_b1024_w2: tile=1024, warps=2, threads=128, N=4096 (aligned)
  vector_add_fp16_b1024_w2 (grid=4, block=128): PASSED
```

## Design Notes

**C++20 required.** Struct NTTPs and `consteval` validation are C++20 features.

**`thread_block_size = 64 × block_warps`, not `block_tile`.** The number of
GPU threads equals the warp count times the wavefront size (64 on AMD CDNA).
Each warp iterates `kRepeatM` times to cover its share of the block tile.

**Archive metadata.** `pack.py` writes a `variant_metadata` section in the
kpack TOC containing each variant's tuning parameters (compute_type,
block_tile, block_warps, warp_tile, pad). This is ignored by the kpack reader
but available for tooling that inspects archives.
