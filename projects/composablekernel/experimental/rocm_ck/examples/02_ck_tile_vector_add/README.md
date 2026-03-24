# CK Tile Vector Add — kpack Example

This example demonstrates how to compile a [CK Tile](https://github.com/ROCm/composable_kernel) kernel into a standalone `.hsaco` code object and load it at runtime via kpack. It builds on the `01_hello_world` example by replacing the hand-written HIP kernel with CK Tile's `ElementWiseKernel`.

## The Problem

CK Tile kernels are C++ template instantiations. The `__global__` entry point is created through compile-time template specialization with mangled C++ symbols, and host and device code share the same translation unit. There is no `extern "C"` symbol, no standalone device compilation, and no way to swap in a different `.hsaco` at runtime.

kpack requires the opposite: an `extern "C"` entry point compiled to a standalone `.hsaco`, loaded via `hipModuleLoadData`, launched via `hipModuleLaunchKernel` with a flat argument buffer.

## The Bridge Pattern

The solution is a thin `extern "C" __global__` wrapper in `ck_tile_add.hip` that:

1. Takes flat arguments (pointers, ints) — compatible with `hipModuleLaunchKernel`
2. Constructs CK Tile types on-device (tuples, etc.)
3. Delegates to `ElementWiseKernel::operator()`

This cleanly separates compilation: the device kernel is compiled with `--cuda-device-only` using CK Tile headers, while the host binary only needs HIP runtime and kpack — no CK Tile headers at all.

## Files

| File | Role |
|------|------|
| `ck_tile_add.hip` | Device kernel: `extern "C"` bridge around `ElementWiseKernel<Add>` |
| `pack.py` | Packer: compiles device code per-architecture, creates `.kpack` archive |
| `main.cpp` | Host loader: opens archive, detects GPU, loads kernel, runs and verifies |
| `CMakeLists.txt` | Standalone build with CK Tile include paths for device compilation |

## Build and Run

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"

ninja -C build

./build/kpack_ck_tile_vector_add build/kernels.kpack
```

Expected output:

```
Opened build/kernels.kpack — architectures: gfx90a, gfx942
Detected GPU: gfx90a
Loaded kernel: 1274848 bytes
Running ck_tile_vector_add (N=1024, grid=4, block=256)...
Verification PASSED!
```

## How It Works

**Build time** (CMake + pack.py):

```
ck_tile_add.hip  --[hipcc --cuda-device-only]--> gfx90a.hsaco, gfx942.hsaco
                 --[pack.py]------------------>  kernels.kpack
```

**Run time** (main.cpp):

```
kernels.kpack --[kpack_open]-----------> archive handle
              --[kpack_get_kernel]-----> .hsaco bytes for detected GPU
              --[hipModuleLoadData]----> HIP module
              --[hipModuleGetFunction]-> kernel function handle
              --[hipModuleLaunchKernel]> run and verify
```

## Design Choices

- **Hard-coded for float32 Add.** The point is demonstrating the bridge pattern, not building a general-purpose elementwise launcher.
- **Block size = 256.** Matches `BlockTile = sequence<256>` in the device kernel. Grid size = ceil(N / 256).
- **CK Tile is referenced by include path, not linked.** The device kernel only needs headers. The host binary only needs kpack + HIP runtime.