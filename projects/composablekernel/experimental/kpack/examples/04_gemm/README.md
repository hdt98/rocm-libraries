# Example 04 — GEMM (Multi-Type)

Demonstrates kpack distribution of a **GEMM kernel** with multiple data type
variants (fp32, fp16, bf16). Extends example 03's optional dtype hierarchy
from elementwise operations to GEMM's asymmetric, multi-type domain.

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

Resolution chains:
- `a_dtype   = a_dtype   ?? dtype ?? error`
- `b_dtype   = b_dtype   ?? dtype ?? error`
- `c_dtype   = c_dtype   ?? dtype ?? error`
- `acc_dtype = acc_dtype  ?? FP32`

### Concise Variant Specification

```cpp
// Homogeneous fp32
make_kernel(GemmSignature{.dtype = DataType::FP32})

// Homogeneous fp16 (accumulates in fp32 automatically)
make_kernel(GemmSignature{.dtype = DataType::FP16})

// Widening: fp16 inputs, fp32 output
make_kernel(GemmSignature{.dtype = DataType::FP16, .c_dtype = DataType::FP32})

// Per-operand override
make_kernel(GemmSignature{.dtype = DataType::FP32, .a_dtype = DataType::FP16})
```

### GemmKernel — Structural NTTP

`resolve_types()` flattens the optional hierarchy, and `make_kernel()` produces
a `GemmKernel` struct with all types and layouts resolved. All members are enum
classes (structural types), so `GemmKernel` works as a C++20 non-type template
parameter — each variant instantiates the CK Tile type stack at compile time.

### CK Tile Type Wiring (gemm_dev.hpp)

`CkTypeMap` and `CkLayoutMap` map our schema enums to CK Tile's C++ types and
layout tags. `runGemm<K>` wires the 7-type CK Tile GEMM stack (shape, traits,
problem, pipeline, partitioner, epilogue, kernel) from a `GemmKernel` NTTP.

Each `.hip` variant file is ~15 lines:

```cpp
static constexpr rocm_ck::GemmKernel kernel =
    rocm_ck::make_kernel(rocm_ck::GemmSignature{.dtype = rocm_ck::DataType::FP16});

extern "C" __global__ void gemm_fp16(rocm_ck::GemmArgs args) {
    rocm_ck::runGemm<kernel>(args);
}
```

### Mixed-Precision Epilogue

All three variants accumulate in fp32. For fp16 and bf16, the CShuffleEpilogue
handles the fp32 → fp16/bf16 conversion when storing to global memory. This is
CK Tile's standard mixed-precision path.

## Compiled Variants

| Variant | A | B | C | Acc | BlockTile | WarpTile | Threads |
|---------|---|---|---|-----|-----------|----------|---------|
| `gemm_fp32` | FP32 | FP32 | FP32 | FP32 | 128×128×32 | 16×16×16 | 256 |
| `gemm_fp16` | FP16 | FP16 | FP16 | FP32 | 128×128×32 | 16×16×16 | 256 |
| `gemm_bf16` | BF16 | BF16 | BF16 | FP32 | 128×128×32 | 16×16×16 | 256 |

All variants use the same tile geometry (128×128×32 block tile, 2×2×1 warp
layout, 16×16×16 warp tile). This configuration is valid for fp32, fp16, and
bf16 per CK Tile's WarpGemmDispatcher.

## File Roles

| File | Purpose |
|------|---------|
| `gemm_api.hpp` | `GemmSignature`, `Layout`, `GemmKernel`, `resolve_types`, `make_kernel`, static_asserts |
| `gemm_dev.hpp` | `CkTypeMap`, `CkLayoutMap`, `runGemm<K>` — maps schema to CK Tile types |
| `gemm_args.hpp` | `GemmArgs` ABI struct, tile constants (`M_TILE`, `N_TILE`, `BLOCK_SIZE`) |
| `gemm_fp32.hip` | fp32 variant instantiation |
| `gemm_fp16.hip` | fp16 variant instantiation |
| `gemm_bf16.hip` | bf16 variant instantiation |
| `pack.py` | Archive packer with per-variant dtype and tile metadata |
| `main.cpp` | Host loader — multi-variant loop with typed buffers, CPU reference verification |
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

**Tile geometry is hardcoded.** This example focuses on the *type* dimension
of the tuning surface. A future GemmConfig will combine
`{.signature, .algorithm}` to expose tile geometry as an independent knob,
following example 03's pattern.

**Test values kept small.** Input values are `i % 8` (max 7). Worst-case
accumulation: 256 × (7 × 7) = 12544, within fp16's exact integer range.
This ensures the fp32 CPU reference matches GPU output for all three types.
