# Insights: GEMM without kpack — rocm_ck as a Conventional HIP Program

This example answers a design question: can rocm_ck work without kpack? The
answer is yes — the schema layer (Signature, GemmSpec, Args) has no kpack
dependency. This example proves it with a single-file HIP program.

## 1. <<<>>> Works with 1552-Byte Args

The `Args` struct is 1552 bytes — 38% of the 4096-byte HSA kernarg budget.
We verified that `<<<>>>` passes it correctly through kernarg with a
byte-level round-trip test (echo kernel that copies Args to device memory
and back). By-value and by-pointer round-trips were identical.

`<<<>>>` compiles down to `hipLaunchKernelGGL`, which uses the same kernarg
mechanism as `hipModuleLaunchKernel`. The HIP runtime adds ~56 bytes of
implicit arguments after user parameters, well within budget.

> **Lesson**: Don't assume large structs need special handling. Verify first.

## 2. gemm_dev.hpp Works in Dual-Pass Compilation

The `gemm_dev.hpp` header has a `#error` guard requiring `--cuda-device-only`.
This is overly conservative — it works fine in normal HIP dual-pass compilation
when guarded with `#ifdef __HIP_DEVICE_COMPILE__`. The CK Tile headers compile
in the device pass; the host pass sees an empty kernel body and generates the
launch stub.

The `#error` exists because all prior examples used separate device-only
compilation (the kpack model). This example proves the guard is sufficient.

## 3. B Defaults to Column-Major

The `GemmOp` implies BLAS convention: A=Row, B=Col, C=Row. This caught us
during development — the CPU reference initially hardcoded row-major access
for all tensors (`h_b[k * N + n]`), producing wrong results.

The fix: use `layoutStrides(kSpec.rhs().layout, K, N)` for the CPU reference,
matching what the GPU kernel sees. Example 04 already does this via its
`cpu_gemm()` function with explicit stride parameters.

> **Lesson**: Always derive strides from the spec. The default B layout
> is Col-major, which is non-obvious if you're coming from a row-major
> mental model. The spec is the source of truth.

## 4. What kpack Adds (and What It Costs)

| Aspect | This example (compile-in) | Example 04 (kpack) |
|--------|---------------------------|---------------------|
| Build steps | 1 (HIP compile) | 3 (device compile, spec extract, pack) |
| Variants | 1 hardcoded | 24+ in one archive |
| Multi-arch | Set `GPU_TARGETS` | Per-variant arch filtering |
| Runtime selection | None | Iterate archive TOC |
| Dependencies | hip::host, rocm_ck | + rocm_kpack, Python |
| Binary contains | Kernel embedded | Host only; kernels in .kpack |

kpack is a distribution mechanism. rocm_ck's type system is independent of it.

## 5. Compilation Boundary

```text
     <hip/hip_runtime.h>            (HIP runtime — both passes)
     <rocm_ck/gemm_spec.hpp>        (schema types — both passes)
     <rocm_ck/args.hpp>             (Args ABI — both passes)
              |
     <rocm_ck/gemm_dev.hpp>         (CK Tile bridge — device pass only)
              |
     gemm_no_kpack.hip
        __global__ kernel           (device pass: run<kSpec>, host pass: empty stub)
        main()                      (host pass only)
```

The key insight: `gemm_spec.hpp` and `args.hpp` are dual-pass safe. Only
`gemm_dev.hpp` (which pulls in CK Tile) requires the device pass guard.
