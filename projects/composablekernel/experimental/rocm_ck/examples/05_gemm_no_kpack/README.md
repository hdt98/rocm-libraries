# Example 05 — GEMM without kpack

Demonstrates rocm_ck as a conventional single-file HIP program. The kernel is
compiled into the binary and launched with `<<<>>>` — no kpack archive, no
runtime module loading, no Python packing step.

## When to Use This Pattern

**Use this (compile-in)** when you have a fixed kernel configuration and want
the simplest possible build. One file, one compilation, one binary.

**Use kpack (example 04)** when you need multiple kernel variants, multi-arch
dispatch, or runtime kernel selection. kpack adds a packaging step but enables
shipping many kernels in one archive.

## Key Concepts

### Single-File HIP Compilation

The HIP compiler runs two passes over the `.hip` file:

1. **Host pass** — compiles `main()`, sees the kernel as a launch stub
2. **Device pass** — compiles the kernel body with CK Tile headers

`gemm_dev.hpp` contains CK Tile GPU intrinsics that only compile in the device
pass. The `#ifdef __HIP_DEVICE_COMPILE__` guard excludes it from the host pass.

### Spec-Driven Host Code

The kernel spec (`kSpec`) is `constexpr` — visible to both passes. Host code
uses it at compile time for:

- **Tensor dtypes**: `kSpec.lhs().dtype` determines TypedBuffer precision
- **Tensor slots**: `kSpec.lhs().args_slot` indexes into `Args::tensors[]`
- **Layouts and strides**: `kSpec.rhs().layout` resolves to Col-major (BLAS default)
- **Grid dimensions**: `kSpec.block_tile` determines tiles per dimension
- **Block size**: `kSpec.workgroup_size` is derived from wave count

### Default Layouts

`GemmOp` implies BLAS convention: A=Row, B=Col, C=Row. Both the GPU kernel
and CPU reference must use matching strides from the spec via `layoutStrides()`.

## Files

| File                 | Purpose                            |
|----------------------|------------------------------------|
| `gemm_no_kpack.hip`  | Single-file kernel + host program  |
| `CMakeLists.txt`     | Minimal build (no kpack dependency) |

## Build

```bash
cd experimental/rocm_ck/examples/05_gemm_no_kpack
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a"
ninja -C build -j16
```

Requires C++20 for `consteval` and structural NTTPs.

## Run

```bash
build/gemm_no_kpack
```

No arguments — the kernel is compiled into the binary.

Expected output:

```text
gemm_fp16_add_relu: M=512, N=512, K=256, grid=16x1x1, block=256
gemm_fp16_add_relu: PASSED
```
