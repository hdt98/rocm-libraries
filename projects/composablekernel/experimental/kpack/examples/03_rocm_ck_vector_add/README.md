# Example 03 — `rocm_ck` Vector Add (Multi-Variant)

Builds on [Example 02](../02_ck_tile_vector_add/) by introducing a `rocm_ck`
abstraction layer that hides CK Tile behind a clean, config-driven interface
and compiles multiple kernel variants into one kpack archive.

## What This Teaches

- **Shared ABI**: A plain C++ struct (`VectorAddArgs`) defines the kernel
  argument layout, shared between host and device with no CK Tile dependency
  on the host side.
- **`consteval` validation**: `make_vector_add_kernel()` validates configuration
  at compile time. Invalid block sizes are compile-time errors, not runtime
  surprises.
- **Parameterized variants**: Three block sizes (256, 512, 1024) are compiled
  from the same device template, each as a separate `.hip` → `.hsaco` file.
- **Multi-binary archives**: `pack.py` packages all variants × architectures
  into a single `.kpack` file. The host selects variants by name at runtime.

## File Roles

| File | Purpose |
|------|---------|
| `rocm_vector_add_api.hpp` | Shared ABI + config (no CK Tile dependency) |
| `rocm_vector_add_dev.hpp` | Device interface — maps config to CK Tile types |
| `vector_add_block{256,512,1024}.hip` | Variant instantiations (~6 lines each) |
| `pack.py` | Multi-binary archive packer |
| `main.cpp` | Host loader — runs and verifies all variants |
| `CMakeLists.txt` | Build system (variant × arch nested loop) |

## Build

```bash
cd experimental/kpack/examples/03_rocm_ck_vector_add
cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx942"
ninja -C build
```

## Run

```bash
./build/kpack_rocm_ck_vector_add build/kernels.kpack
```

Expected output:
```
Opened build/kernels.kpack — architectures: gfx942
Detected GPU: gfx942
  vector_add_block256 (grid=16, block=256): PASSED
  vector_add_block512 (grid=8, block=512): PASSED
  vector_add_block1024 (grid=4, block=1024): PASSED
```

## Design Decisions

**C++20 required.** Struct NTTPs (`template <vector_add_config Config>`) and
`consteval` are C++20 features. The device compiler (clang 17+) supports both.

**`block_size` = thread block size for launch.** CK Tile's actual thread count
is `warp_size × BlockWarps` (= 64 for this example). Launching with more
threads than needed wastes occupancy but is harmless — extra threads do no
work. This keeps the host-side launch logic simple: grid = N / block_size.

**BlockWarps always 1.** `block_size` is the only configuration knob. The
number of warps per block stays at 1, so varying `block_size` changes how many
elements each warp processes (via `kRepeatM`), not the parallelism structure.

**NUM_ELEMENTS = 4096.** Divisible by 256, 512, and 1024. Avoids partial-block
edge cases that would obscure the multi-variant pattern being demonstrated.
