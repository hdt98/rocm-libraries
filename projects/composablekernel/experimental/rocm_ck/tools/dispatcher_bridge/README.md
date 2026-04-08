# Dispatcher Bridge PoC

Load a self-describing kpack archive and register its GEMM kernels with the CK dispatcher's `KernelInstance` interface. Proves that kpack kernels can plug into the dispatcher's registry, selection, and execution pipeline without modifying the dispatcher.

## What This Demonstrates

1. **Projection** — `GemmSpec` (rocm\_ck's consteval schema) maps 1:1 to `KernelKey` (the dispatcher's kernel configuration). The mapping covers dtypes, layouts, pipelines, schedulers, epilogues, and tile geometry.

2. **KpackKernelInstance** — A new dispatcher backend that implements `KernelInstance` by loading device code from a kpack archive at runtime via `hipModuleLoadData`. Translates the dispatcher's `(a_ptr, b_ptr, c_ptr, Problem)` calling convention to rocm\_ck's slot-based `Args` struct using the `GemmSpec` physical tensor table.

3. **Registration** — `registerKpackKernels()` reads `variant_specs` from the kpack TOC, filters by GPU and supported features, projects each `GemmSpec` to a `KernelKey`, and registers `KpackKernelInstance`s in the dispatcher's `Registry`.

4. **Validation** — `KpackKernelInstance::validate()` downloads device buffers, computes a CPU reference GEMM with layout-aware strides, and compares against the kernel output.

## Architecture

```
gemm.kpack
    ├── variant_specs (msgpack)     — GemmSpec per variant
    └── hsaco blobs                 — device code (loaded lazily)

Registration (eager):

    readVariantSpecs(kpack) → VariantInfo[]
                │
                ↓
    projectToDispatcher(GemmSpec, arch) → KernelKey
                │
                ↓
    KpackKernelInstance(key, spec, archive)
                │
                ↓
    Registry.register_kernel()

Dispatch:

    Dispatcher.select_kernel() → KpackKernelInstance
                │
                ↓
    run()
      ├─ first call: KpackKernel::load() → hipModuleLoadData
      │              (lazily loads hsaco blob from archive)
      ├─ Args translation via GemmSpec tensor table
      └─ hipModuleLaunchKernel
```

## Files

| File | Role |
|------|------|
| `include/rocm_ck/projection.hpp` | `GemmSpec` → `KernelKey` projection (reusable header, no HIP dep) |
| `kpack_kernel_instance.hpp` | `KpackKernelInstance` — dispatcher backend for kpack kernels |
| `register_kpack.hpp` | Load kpack, iterate variants, project, register |
| `main.cpp` | Demo: register, list, dispatch, verify |
| `CMakeLists.txt` | Standalone build |

## Build

Requires example 04's `gemm.kpack` archive (built separately):

```bash
# Build example 04 first (if not already built)
cd experimental/rocm_ck/examples/04_gemm
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"
ninja -C build -j16

# Build the bridge
cd experimental/rocm_ck/tools/dispatcher_bridge
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C build -j16
```

## Run

```bash
build/dispatcher_bridge ../../examples/04_gemm/build/gemm.kpack
```

Expected output (on gfx90a):

```
Registered 15 kernels from kpack

Registered kernels:
  gemm_fp32                       fp32_rcr_compv1_intrawave_cshuffle_128x128x32_...
  gemm_fp16                       fp16_rcr_compv1_intrawave_cshuffle_128x128x32_...
  gemm_bf16                       bf16_rcr_compv1_intrawave_cshuffle_128x128x32_...
  ...

Running gemm_fp16: M=512, N=512, K=256, block=256
gemm_fp16: PASSED
validate(): PASSED
```

## Current Limitations

- **D tensors** — Not wired through `run()`. Kernels with bias/scale epilogues register but `run()` rejects non-null `d_ptrs`. All existing dispatcher backends have the same limitation.
- **Preshuffle** — Skipped during registration (requires host-side B matrix rearrangement).
- **StreamK** — Skipped (grid calculation differs from tile-based partitioners).
- **Quantized (INT4 BQuant)** — Skipped (requires special INT4 packing and scale tensors).
- **Batched GEMM** — Not supported (dispatcher `Problem` has no batch dimension).
