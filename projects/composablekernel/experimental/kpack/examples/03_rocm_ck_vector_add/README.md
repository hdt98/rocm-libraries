# Example 03 — `rocm_ck` Vector Add (Full Tuning Surface)

Demonstrates CK Tile's **Signature + Algorithm** pattern for distributing
pre-compiled GPU kernels with full control over the ElementWise tuning surface.
Supports mixed input/output types and scaled addition (`c = alpha * a + beta * b`).

## Key Concepts

### Signature / Algorithm Separation

Following CK Tile's builder pattern, kernel configuration is split into two
orthogonal concerns:

- **Signature** (`ElementwiseSignature`): *What* the kernel computes — data types,
  specified via an optional dtype hierarchy. Users set only what differs:
  `{.dtype = FP32}` for homogeneous, `{.in_dtype = FP16, .out_dtype = FP32}` for
  mixed. `resolve_types()` flattens the hierarchy into concrete types at compile time.
- **Algorithm** (`ElementwiseAlgorithm`): *How* the kernel executes — tile geometry,
  warp count, vector width, padding.
- **Config** (`ElementwiseConfig`): User-facing API combining Signature + Algorithm.

This separation lets the same operation (vector add) be compiled with many
different tuning configurations, each producing a distinct `.hsaco` binary.

### Optional dtype Hierarchy

The signature uses an optional hierarchy so users specify the minimum:

```
dtype                    (kernel-level default)
├── in_dtype             (input default, overrides dtype for inputs)
│   ├── a_dtype          (overrides in_dtype)
│   └── b_dtype          (overrides in_dtype)
└── out_dtype            (overrides dtype for output)
```

Resolution chains:
- `a_dtype   = a_dtype   ?? in_dtype ?? dtype ?? error`
- `b_dtype   = b_dtype   ?? in_dtype ?? dtype ?? error`
- `out_dtype = out_dtype ?? dtype    ?? error`

Note: `in_dtype` does NOT cascade to `out_dtype` — they are separate branches.

```cpp
// Same-type: all FP32
.signature = {.dtype = DataType::FP32}

// Widening: fp16 inputs, fp32 output
.signature = {.in_dtype = DataType::FP16, .out_dtype = DataType::FP32}

// Narrowing: fp32 inputs, fp16 output
.signature = {.in_dtype = DataType::FP32, .out_dtype = DataType::FP16}

// Asymmetric inputs (future operations like GEMM — vector add requires a == b)
.signature = {.a_dtype = DataType::FP16, .b_dtype = DataType::BF16, .out_dtype = DataType::FP32}
```

For mixed types, `kVectorM` is constrained by the wider type (fewer elements
per 128-bit register), validated at compile time by `make_kernel`.

### Scaled Addition

The kernel always computes `c = alpha * a + beta * b`. Scalar parameters
`alpha` and `beta` are part of the `VectorAddArgs` struct, matching the BLAS
convention where scalar parameters are always present. For plain addition,
pass `alpha = 1.0f` and `beta = 1.0f`.

### Custom CK Tile Kernel

The device kernel (`runVectorAdd`) uses CK Tile tile primitives directly
(`load_tile`, `sweep_tile_span`, `store_tile`) rather than the stock
`ElementWiseKernel`. This enables passing scalar parameters and provides
a more instructive example of building custom kernels from CK Tile building
blocks.

### Tuning Surface

The algorithm exposes four independent knobs matching CK Tile's `ElementWiseShape`:

| Parameter | Field | Description |
|-----------|-------|-------------|
| BlockTile | `block_tile` | Elements processed per thread block |
| BlockWarps | `block_warps` | Warps per thread block (must be power of 2) |
| WarpTile | `warp_tile` | Controls vector load width (`kVectorM`) |
| Pad | `pad` | Enable padding for unaligned problem sizes |

Derived quantities (validated at compile time by `make_kernel`):
- `kVectorM = min(128 / max_type_bits, warp_tile / 64)` — vector elements per thread
- `kRepeatM = block_tile / (block_warps × kVectorM × 64)` — iterations per warp
- `thread_block_size = 64 × block_warps` — actual threads launched

### Variant Registry

`rocm_vector_add_registry.hpp` provides programmatic variant selection:
- `ALL_VARIANTS[]` — constexpr table of all compiled kernel variants
- `findVariant(in_dtype, out_dtype, problem_size)` — selects the best variant:
  largest `block_tile` that divides `problem_size` cleanly, or padded fallback

## Compiled Variants

| Variant | In | Out | BlockTile | BlockWarps | WarpTile | Threads |
|---------|----|-----|-----------|------------|----------|---------|
| `fp32_b256` | FP32 | FP32 | 256 | 1 | 256 | 64 |
| `fp32_b512` | FP32 | FP32 | 512 | 1 | 512 | 64 |
| `fp32_b1024` | FP32 | FP32 | 1024 | 1 | 1024 | 64 |
| `fp16_b512` | FP16 | FP16 | 512 | 1 | 512 | 64 |
| `fp16_b1024` | FP16 | FP16 | 1024 | 1 | 1024 | 64 |
| `bf16_b512` | BF16 | BF16 | 512 | 1 | 512 | 64 |
| `fp32_b256_sa` | FP32 | FP32 | 256 | 1 | 256 | 64 |
| `fp32_b2048_w8` | FP32 | FP32 | 2048 | 8 | 64 | 512 |
| `fp16_b1024_w2` | FP16 | FP16 | 1024 | 2 | 512 | 128 |
| `fp16_fp32_b1024` | FP16 | FP32 | 1024 | 1 | 1024 | 64 |
| `fp32_fp16_b1024` | FP32 | FP16 | 1024 | 1 | 1024 | 64 |

The `_sa` suffix indicates a standalone allocation variant (functionally
identical to `fp32_b256`). The `_w8`/`_w2` suffixes indicate multi-warp variants.

## File Roles

| File | Purpose |
|------|---------|
| `rocm_vector_add_api.hpp` | Shared ABI, Config/Signature/Algorithm types, `resolve_types`, `make_kernel` validation |
| `rocm_vector_add_dev.hpp` | Device kernel — uses CK Tile tile primitives for `c = alpha*a + beta*b` |
| `rocm_vector_add_registry.hpp` | `VariantDescriptor` table and `findVariant` selection (host-only) |
| `vector_add_*.hip` | Variant instantiations (~15 lines each) |
| `pack.py` | Archive packer with variant metadata |
| `main.cpp` | Host loader — variant selection demo, verify-all, scaled-add test |
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

Running all 11 variants (alpha=1, beta=1):
  vector_add_fp32_b256: tile=256, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp32_b256 (grid=16, block=64): PASSED
  ...
  vector_add_fp16_fp32_b1024: tile=1024, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp16_fp32_b1024 (grid=4, block=64): PASSED
  vector_add_fp32_fp16_b1024: tile=1024, warps=1, threads=64, N=4096 (aligned)
  vector_add_fp32_fp16_b1024 (grid=4, block=64): PASSED

Scaled-add test (alpha=2, beta=0.5):
  vector_add_fp32_b2048_w8: tile=2048, warps=8, threads=512, N=4096 (aligned)
  vector_add_fp32_b2048_w8 (grid=2, block=512): PASSED
```

## Design Notes

**C++20 required.** Struct NTTPs and `consteval` validation are C++20 features.

**`thread_block_size = 64 × block_warps`, not `block_tile`.** The number of
GPU threads equals the warp count times the wavefront size (64 on AMD CDNA).
Each warp iterates `kRepeatM` times to cover its share of the block tile.

**Archive metadata.** `pack.py` writes a `variant_metadata` section in the
kpack TOC containing each variant's tuning parameters (in_dtype, out_dtype,
block_tile, block_warps, warp_tile, pad). This is ignored by the kpack reader
but available for tooling that inspects archives.

**Per-operation signatures.** Each operation defines its own signature struct
with optional DataType fields organized in a hierarchy. This is intentionally
not a universal struct — GEMM, convolution, and FMHA will each define their
own signatures with different type hierarchies.