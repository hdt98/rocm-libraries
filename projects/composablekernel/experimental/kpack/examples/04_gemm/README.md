# Example 04 — GEMM (Multi-Type, Parameterized Tile Geometry, Fused Epilogue)

Demonstrates kpack distribution of a **GEMM kernel** with multiple data type
variants, parameterized tile geometry, and fused epilogues. Extends example 03's
`{.signature, .algorithm}` pattern to GEMM's multi-type domain.

## Key Concepts

### GemmSignature — the "WHAT" of a GEMM

Describes data types and memory layouts for a GEMM. Uses a two-level optional
dtype hierarchy (simpler than elementwise's three levels):

```
dtype                    (kernel-level default)
├── a_dtype              (A override)
├── b_dtype              (B override)
├── c_dtype              (C output override)
└── acc_dtype            (accumulator, defaults to FP32)
```

**Why no `in_dtype`?** GEMM's A and B are asymmetric (M×K vs K×N) — a shared
"input default" suggests false symmetry. Two levels (dtype → per-operand) is
cleaner.

**Why `acc_dtype` defaults to FP32?** Every practical GEMM accumulates in fp32,
even with fp16/bf16 inputs (MFMA instructions). This is a sensible default,
not inherited from `dtype`.

### GemmAlgorithm — the "HOW" of a GEMM

Describes tile geometry through three `Dim3{m, n, k}` fields:

```cpp
struct Dim3 { int m, n, k; };

struct GemmAlgorithm {
    Dim3 block_tile;   // Elements per thread block {M, N, K}
    Dim3 block_warps;  // Warp distribution {M, N, K}
    Dim3 warp_tile;    // Elements per warp per MFMA step {M, N, K}
};
```

Independent of data types — paired with `GemmSignature` in `GemmConfig`:

```cpp
make_kernel(GemmConfig{
    .signature = {.dtype = DataType::FP16},
    .algorithm = {.block_tile  = {128, 128, 32},
                  .block_warps = {2, 2, 1},
                  .warp_tile   = {32, 32, 16}}})
```

### Consteval Validation

`make_kernel()` performs compile-time validation:

- **Warp tile validity**: Checks against CK Tile's `WarpGemmDispatcher` table
  via `is_valid_warp_gemm()`. For example, FP32 supports 32×32×{4,8} but not
  32×32×16, while FP16 supports 32×32×{8,16}.
- **Tile divisibility**: `block_tile.m` must be divisible by
  `block_warps.m × warp_tile.m` (and similarly for N and K).
- **CShuffleEpilogue constraint**: `block_warps.k` must be 1.
- **Thread block size**: Derived as `block_warps.m × block_warps.n × 64`.

Invalid configurations produce compile errors — no runtime surprises.

### GemmKernel — Structural NTTP

`make_kernel()` produces a `GemmKernel` struct with all types, layouts, and
tile geometry resolved. All members are structural types (enums, ints,
aggregates), so `GemmKernel` works as a C++20 non-type template parameter.

### CK Tile Type Wiring (gemm_dev.hpp)

`CkTypeMap` and `CkLayoutMap` map our schema enums to CK Tile's C++ types and
layout tags. `runGemm<K>` wires the 7-type CK Tile GEMM stack (shape, traits,
problem, pipeline, partitioner, epilogue, kernel) from a `GemmKernel` NTTP.
Tile geometry flows from `K.block_tile`, `K.block_warps`, and `K.warp_tile`.

Each `.hip` variant file is ~20 lines:

```cpp
static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmConfig{
        .signature = {.dtype = rocm_ck::DataType::FP16},
        .algorithm = {.block_tile  = {128, 128, 32},
                      .block_warps = {2, 2, 1},
                      .warp_tile   = {16, 16, 16}}});

extern "C" __global__ void gemm_fp16(rocm_ck::GemmArgs args) {
    rocm_ck::runGemm<kernel>(args);
}
```

### Mixed-Precision Epilogue

All variants accumulate in fp32. For fp16 and bf16, the CShuffleEpilogue
handles the fp32 → fp16/bf16 conversion when storing to global memory. This is
CK Tile's standard mixed-precision path.

### Fused Epilogue (EpilogueOp)

The signature includes an `epilogue_op` field that controls post-GEMM fusion:

```cpp
enum class EpilogueOp { None, Add, Multiply };
```

- **None** — plain GEMM: `E = A * B` (no D tensors, uses `GemmArgs`)
- **Add** — fused addition: `E = A * B + D0 [+ D1]` (uses `GemmArgs1D`)
- **Multiply** — fused scaling: `E = A * B * D0 [* D1]`

D tensors are specified in the signature alongside the epilogue operation:

```cpp
make_kernel(GemmConfig{
    .signature = {.dtype       = DataType::FP16,
                  .epilogue_op = EpilogueOp::Add,
                  .d0_dtype    = DataType::FP16},
    .algorithm = {.block_tile  = {128, 128, 32},
                  .block_warps = {2, 2, 1},
                  .warp_tile   = {16, 16, 16}}})
```

The schema validates consistency at compile time: `None` rejects D tensors,
`Add`/`Multiply` require at least `d0_dtype`. D tensors are M x N (same shape
as output). Broadcast via stride tricks (1xN bias) is a future extension.

## Compiled Variants

| Variant | A | B | E | Acc | Epilogue | D0 | BlockTile | WarpTile | Threads |
|---------|---|---|---|-----|----------|----|-----------|----------|---------|
| `gemm_fp32` | FP32 | FP32 | FP32 | FP32 | None | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16` | FP16 | FP16 | FP16 | FP32 | None | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_bf16` | BF16 | BF16 | BF16 | FP32 | None | — | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16_w32` | FP16 | FP16 | FP16 | FP32 | None | — | 128×128×32 | 32×32×16 | 256 |
| `gemm_fp16_add` | FP16 | FP16 | FP16 | FP32 | Add | FP16 | 128×128×32 | 16×16×16 | 256 |

All variants use 128×128×32 block tile with 2×2×1 warp layout (4 warps =
256 threads). `gemm_fp16_w32` demonstrates a wider 32×32 warp tile.
`gemm_fp16_add` demonstrates a fused epilogue with one D tensor.

## File Roles

| File | Purpose |
|------|---------|
| `gemm_api.hpp` | `GemmSignature`, `GemmAlgorithm`, `GemmConfig`, `GemmKernel`, `EpilogueOp`, `is_valid_warp_gemm`, `make_kernel`, static_asserts |
| `gemm_dev.hpp` | `CkTypeMap`, `CkLayoutMap`, `CkEpilogueOpMap`, `EpilogueTypes`, `runGemm<K>` — maps schema to CK Tile types |
| `gemm_args.hpp` | `GemmArgs` (plain GEMM) and `GemmArgs1D` (1 D tensor) ABI structs |
| `gemm_fp32.hip` | fp32 variant (16×16×16 warp tile) |
| `gemm_fp16.hip` | fp16 variant (16×16×16 warp tile) |
| `gemm_bf16.hip` | bf16 variant (16×16×16 warp tile) |
| `gemm_fp16_w32.hip` | fp16 variant (32×32×16 warp tile) |
| `gemm_fp16_add.hip` | fp16 + D0 addition (fused epilogue) |
| `pack.py` | Archive packer with per-variant dtype, epilogue, and tile metadata |
| `main.cpp` | Host loader — multi-variant loop with D tensor support, CPU reference verification |
| `CMakeLists.txt` | Build system (variant × arch nested loop) |

## Build

```bash
cd experimental/kpack/examples/04_gemm
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"
ninja -C build
```

Requires C++20 for struct NTTPs and `consteval` validation.

## Run

```bash
./build/kpack_gemm build/gemm.kpack
```

Expected output:
```
Opened build/gemm.kpack — architectures: gfx90a, gfx942
Detected GPU: gfx90a
gemm_fp32: M=512, N=512, K=256, grid=16, block=256
gemm_fp32: PASSED
gemm_fp16: M=512, N=512, K=256, grid=16, block=256
gemm_fp16: PASSED
gemm_bf16: M=512, N=512, K=256, grid=16, block=256
gemm_bf16: PASSED
gemm_fp16_w32: M=512, N=512, K=256, grid=16, block=256
gemm_fp16_w32: PASSED
gemm_fp16_add: M=512, N=512, K=256, grid=16, block=256
gemm_fp16_add: PASSED
```

## Design Notes

**Two-level hierarchy, not three.** Elementwise operations have symmetric
inputs, so `in_dtype` makes sense as a shared default. GEMM's A and B are
fundamentally asymmetric (M×K vs K×N), so the three-level hierarchy would
suggest false symmetry. Two levels (dtype → per-operand) is the right
abstraction.

**Layouts are non-optional.** Unlike data types (where defaults reduce
boilerplate), layouts have universally sensible BLAS defaults: A=Row, B=Col,
C=Row. Making them optional would add complexity without reducing verbosity.

**Signature and Algorithm are independent.** The same tile geometry works
across fp32, fp16, and bf16 (with appropriate warp tiles). Separating "what
types" from "how to tile" lets each vary independently — different types can
share tile configs, and the same type can use different tile configs
(`gemm_fp16` vs `gemm_fp16_w32`).

**Consteval catches mistakes at compile time.** Invalid warp tile / dtype
combinations (e.g., fp32 with 32×32×16 warp tile) and bad tile divisibility
produce compile errors, not runtime crashes. The `is_valid_warp_gemm()`
lookup table mirrors CK Tile's WarpGemmDispatcher specializations.

**Test values kept small.** Input values are `i % 8` (max 7). Worst-case
accumulation: 256 × (7 × 7) = 12544, within fp16's exact integer range.
This ensures the fp32 CPU reference matches GPU output for all four variants.
