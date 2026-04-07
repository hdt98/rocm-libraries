# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

Composable Kernel (CK) is AMD's high-performance kernel library for machine learning workloads. It contains three tile programming frameworks:

1. **CK (Original)** - Templated operators with manual kernel selection
2. **CK_Tile** - Next-gen tile-based API, AMD-specific, optimized for MFMA
3. **MINT** - Portable tile programming model (ROCm + CUDA)
4. **Unified Tile** - Bridge layer unifying CK_Tile and MINT behind a single API

---

## Build Environment

All builds happen inside the Docker container `awsm_wilbur`:

```bash
# Start and enter container
docker start awsm_wilbur
docker exec -it awsm_wilbur /bin/bash

# Path mapping
# Host:   /home/khuagarw1/repos/rocm-libraries/projects/composablekernel
# Docker: /root/workspace
```

### Build Commands

```bash
# Inside Docker
cd /root/workspace
mkdir -p build && cd build
../script/cmake-ck-dev.sh ../ gfx942 -G Ninja

# Build specific targets
ninja <target_name>

# Run
./bin/<binary_name>
```

### Build From Host (via Docker exec)

```bash
# Reconfigure cmake (after adding new CMakeLists.txt)
docker exec awsm_wilbur bash -c "cd /root/workspace/build && cmake .."

# Build
docker exec awsm_wilbur bash -c "cd /root/workspace/build && ninja <target>"

# Run
docker exec awsm_wilbur bash -c "cd /root/workspace/build && ./bin/<binary>"
```

### Key CMake Flags

- `GPU_TARGETS` - Target architectures (e.g., `"gfx942"`) - **required**
- `BUILD_DEV=ON` - Development mode (verbose, better errors)
- `DTYPES` - Filter data types: `"fp64;fp32;fp16;fp8;bf16;int8"`

### Testing

```bash
ctest -j$(nproc)          # Run all tests
ctest -R gemm             # Run tests matching pattern
ctest --output-on-failure # Show failures in detail
```

---

## Project Structure

```
include/
├── ck/                   # Original CK library (templated operators)
├── ck_tile/              # Next-gen tile-based API (AMD-specific)
│   ├── core/             # Building blocks (containers, tensor abstractions)
│   │   └── tensor/       # tensor_descriptor, tensor_view, tile_window,
│   │                     # buffer_view, load_tile, store_tile
│   ├── ops/              # Operations (GEMM, FMHA, convolution, etc.)
│   │   └── gemm/         # GEMM kernel, pipeline, block gemm, policy
│   └── host/             # Kernel launching, DeviceMem, HIP utilities
├── mint/                 # Portable tile programming model (ROCm + CUDA)
│   ├── core/             # Arrays, tuples, memory views (global, shared, vgpr)
│   ├── poly/             # Morphers (index transformations: split, merge, etc.)
│   ├── tensor/           # tensor_descriptor, tensor_view, distributed_window
│   └── tile/             # Load/store ops, MFMA intrinsics (generic/, rocm/)
└── unified_tile/         # Bridge layer between CK_Tile and MINT
    ├── config.hpp        # Backend selection, UNIFIED_TILE_HOST_DEVICE macro
    ├── core/             # backend_traits, type_aliases, type_concepts
    ├── tensor/           # descriptor, view, address_space
    ├── distribution/     # (Task 02 - pending)
    ├── ops/              # (Tasks 04-08 - pending)
    └── gemm/             # (Task 09 - pending)

example/ck_tile/
├── 03_gemm/              # CK_Tile GEMM examples
├── 99_mint/              # MINT basic examples (copy kernel, naive gemm)
├── mint_example/         # MINT advanced example (gemm_xdl_v1)
└── unified_tile/         # Unified tile examples
    ├── 00_descriptor/    # Descriptor creation (both backends)
    └── 01_view/          # Tensor view creation (both backends)

test/unified_tile/        # Unified tile tests (Google Test, both backends)
```

---

## GEMM Pipeline: How It Works

### CK_Tile GEMM Pipeline

**Entry:** `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`
**Pipeline:** `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`

**Step 1: DRAM to VGPR tile windows**
1. Create `tensor_descriptor` using `make_naive_tensor_descriptor`
2. Create `tensor_view` using `make_tensor_view`
3. Apply padding using `pad_tensor_view`
4. Create logical `tile_window` using `make_tile_window`
5. Align thread mapping at pipeline level via `MakeADramTileDistribution`
   (defined in `gemm_universal_pipeline_ag_bg_cr_policy.hpp`)

**Step 2: VGPR to LDS tile windows**
- Descriptor and view created via `GetABLdsTensorViews` (base class)
- Distribution defined via `MakeABlockDistributionEncode`
  (in `block_universal_gemm_as_bs_cr.hpp`)
- Windows created in `GetAWindows` (base class)

**Step 3: LDS to VGPR tile windows (for MFMA)**
- Another tile_window for reading from LDS into registers for MFMA

**Step 4: Pipeline execution (double-buffered)**
```
1. Load first block tile from DRAM, move window to next position
2. Store to LDS via LocalPrefill
3. Load second block tile from DRAM, move window
4. Load from LDS to registers via LocalPrefetch
5. do {
     Store to LDS via LocalPrefill
     Load next DRAM tile, move window
     Block GEMM (MFMA operation)
     Load from LDS via LocalPrefetch
   } while (more K iterations)
```

**Key CK_Tile files:**

| File | Purpose |
|------|---------|
| `kernel/universal_gemm_kernel.hpp` | Kernel entry, descriptor/view/window creation |
| `pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp` | Main pipeline loop |
| `pipeline/gemm_pipeline_ag_bg_cr_base.hpp` | GlobalPrefetch, LocalPrefill, LocalPrefetch, GetWindows |
| `pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp` | MakeADramTileDistribution, vector sizes |
| `block/block_universal_gemm_as_bs_cr.hpp` | Block GEMM, MakeABlockDistributionEncode, MFMA dispatch |

### MINT GEMM Pipeline

**Entry:** `example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp`

**Step 1: DRAM to VGPR distributed windows**
1. Create `tensor_descriptor` using `make_aliased_naive_packed_tensor_descriptor`
2. Create `tensor_view` using `make_tensor_view(desc, make_global_memory_view(ptr, size))`
3. Create `distributed_window` using `make_distributed_window` with:
   - `distributed_tensor_descriptor` (from morphers: split, merge)
   - Element layout descriptor
   - Partition info (`this_thread_block{}`)

**Step 2: VGPR to SMEM distributed windows**
- Same pattern: descriptor → view (with `make_shared_memory_view`) → distributed_window
- Uses same block-level distribution and element layout

**Step 3: SMEM to VGPR distributed windows (for MFMA)**
- Uses warp-level distribution (`make_warp_matmul_a_m_k_distribution`)
- Partition: `thread_in_this_warp{}`

**Step 4: Pipeline execution**
```
1. Load A/B block tiles from GMEM using masked_load_no_shuffle_vectorized
2. Store to SMEM using masked_store_no_shuffle_vectorized
3. do {
     Load next GMEM tiles
     block_sync_lds()
     Load warp tiles from LDS using masked_load_no_shuffle_vectorized
     matmul_xdl (MFMA operation)
     block_sync_lds()
     Store to SMEM
   } while (more K iterations)
```

**Key MINT files:**

| File | Purpose |
|------|---------|
| `include/mint/tensor/tensor_descriptor_helper.h` | `make_aliased_naive_packed_tensor_descriptor` |
| `include/mint/tensor/tensor_view.h` | `make_tensor_view` |
| `include/mint/tensor/distributed_window.h` | `make_distributed_window`, `move_window` |
| `include/mint/tile/generic/load_no_shuffle_vectorized.h` | `masked_load_no_shuffle_vectorized` |
| `include/mint/tile/generic/store_no_shuffle_vectorized.h` | `masked_store_no_shuffle_vectorized` |
| `include/mint/tile/rocm/warp/matmul_xdl.h` | `matmul_xdl` (MFMA intrinsics) |
| `include/mint/poly/fundamental_morpher.h` | split, merge, pass_through, rotate2d |

---

## CK_Tile vs MINT API Mapping

| Operation | CK_Tile | MINT |
|-----------|---------|------|
| Descriptor | `make_naive_tensor_descriptor(lengths, strides)` | `make_aliased_naive_packed_tensor_descriptor(aliases, offset, lengths)` |
| Tensor view | `make_tensor_view<addr_space>(ptr, desc)` | `make_tensor_view(desc, make_global_memory_view(ptr, size))` |
| Padding | `pad_tensor_view(view, tile_lens, do_pads)` | N/A (masking at load/store) |
| Tile window | `make_tile_window(view, lens, origin, dstr)` | `make_distributed_window(view, origin, dstr, elem_layout, partition)` |
| Distribution | `tile_distribution_encoding` + `sequence<...>` | `distributed_tensor_descriptor` + polymorphers |
| Load | `load_tile(window)` | `masked_load_no_shuffle_vectorized<vec_dims, vec_lens, freeze>(win, mask)` |
| Store | `store_tile(window, tensor)` | `masked_store_no_shuffle_vectorized<vec_dims, vec_lens, freeze>(win, mask, tensor)` |
| Move window | `move_tile_window(window, step)` | `move_window(window, delta)` |
| MFMA | `WarpGemm{}(c, a, b)` | `matmul_xdl<SwapAB, M, N>(c, a, b)` |
| Sync | `block_sync_lds()` | `block_sync_lds()` |
| Num dims | `desc.get_num_of_dimension()` | `Descriptor::top_ndim()` |
| Get lengths | `desc.get_lengths()` | `desc.top_lengths()` |
| View size | `view.get_tensor_descriptor().get_element_space_size()` | `view.memory_view().size()` |
| Host/device | `CK_TILE_HOST_DEVICE` | `MINT_HOST_DEVICE` |
| Index type | `ck_tile::index_t` (int32_t) | `mint::index_t` (int64_t) |
| Constant | `ck_tile::number<N>` | `mint::constant<N>` |
| Multi-index | `ck_tile::make_tuple(m, n)` | `mint::nd_index<2>{m, n}` |

---

## Unified Tile Bridge Layer

### Purpose

Write GEMM kernels once, compile against either backend via a flag:
- `-DUNIFIED_TILE_BACKEND_CK_TILE`
- `-DUNIFIED_TILE_BACKEND_MINT`

Zero runtime overhead - all dispatch is `if constexpr` or template specialization.

### Current Status

| Task | Description | Status |
|------|-------------|--------|
| 00 | Descriptor fix & test | COMPLETE |
| 01 | Tensor View | COMPLETE |
| 02 | Distribution Config | COMPLETE |
| 03 | Tile Window | COMPLETE |
| 04 | Load Operations | Pending |
| 05 | Store Operations | Pending |
| 06 | Window Movement | Pending |
| 07 | Warp-Level MFMA | Pending |
| 08 | Sync & Distributed Tensor | Pending |
| 09 | Full GEMM Pipeline | Pending |

**Plan file:** `~/.claude/plans/rosy-growing-wilkinson.md`
**Session summary:** `include/unified_tile/SESSION_SUMMARY.md`

### Unified Tile API (implemented so far)

```cpp
namespace unified_tile::descriptor {
    auto make_descriptor(Dims... dims);             // packed descriptor
    auto make_descriptor_with_strides(lens, strides);
    auto get_num_dimensions(desc);
    auto get_lengths(desc);
}

namespace unified_tile::view {
    auto make_tensor_view<AddrSpace>(ptr, desc);    // global, shared, vgpr
    auto pad_view(view, tile_lengths, do_pads);     // CK_TILE pads, MINT no-op
    auto get_view_size(view);
}

namespace unified_tile::distribution {
    // Shared config: block_copy_2d_config<BS, Dim0, Dim1, VecSize>
    //   - kThreadsOuter, kThreadsInner, kElemsOuter, kElemsPerThread
    //   - static_asserts for divisibility

    // Generic 2D distribution
    auto make_block_copy_2d_distribution<BS, Dim0, Dim1, Vec>();

    // Named A/B with proper MINT aliases (M/K, K/N)
    auto make_block_copy_a_distribution<BS, M, K, Vec>();
    auto make_block_copy_b_distribution<BS, K, N, Vec>();

    // Query
    auto get_elements_per_thread(dstr);  // CK_TILE: ys_to_d, MINT: element_size()
    auto get_num_tile_dims(dstr);        // CK_TILE: NDimX, MINT: top_ndim()
}

namespace unified_tile::window {
    auto make_tile_window(view, lengths, origin, distribution);
    void move_window(window, step);
}
```

### Design Pattern

All unified_tile code follows traits-based dispatch:

```cpp
template <backend_type Backend>
struct some_traits;

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
template <>
struct some_traits<backend_type::ck_tile> {
    static constexpr auto do_something(...) {
        return ck_tile::backend_function(...);
    }
};
#endif

#ifdef UNIFIED_TILE_BACKEND_MINT
template <>
struct some_traits<backend_type::mint> { ... };
#endif
```

Public APIs are split into `#ifdef` blocks (not a single template) to avoid
ADL ambiguity when passing `ck_tile::tuple` arguments.

### Build Targets (unified_tile)

```bash
# Examples (each has _ck and _mint variants)
ninja example_unified_tile_00_descriptor_ck   # or _mint
ninja example_unified_tile_01_view_ck         # or _mint
ninja example_unified_tile_02_distribution_ck # or _mint
ninja example_unified_tile_03_window_ck       # or _mint

# Tests (Google Test, each has _ck and _mint variants)
ninja test_unified_tile_descriptor_ck    # or _mint
ninja test_unified_tile_view_ck          # or _mint
ninja test_unified_tile_distribution_ck  # or _mint
ninja test_unified_tile_window_ck        # or _mint
```

**MINT targets require:** `-DMINT_BACKEND_ROCM -fgpu-rdc` (compile) + `-fgpu-rdc` (link)

### Key Learnings

1. **ADL trap:** Don't name unified functions same as `ck_tile::` functions (e.g., use
   `make_descriptor` not `make_naive_tensor_descriptor_packed`) - causes ambiguity via ADL
   when passing `ck_tile::tuple` args.
2. **Use `UNIFIED_TILE_HOST_DEVICE`** not `CK_TILE_HOST_DEVICE` in unified code.
3. **Use `ck_tile::DeviceMem`** and `HIP_CHECK_ERROR` for host-side memory management.
4. **`pad_tensor_view` does not increase buffer size** - it adds right-pad transforms
   that return 0 for OOB reads.
5. **MINT `alias_t` (static_string) has no `operator+`** - cannot concatenate aliases at
   compile time. Pass all sub-dimension aliases explicitly as template parameters.
6. **MINT `constant<dstr>` requires constexpr** - function parameters are not constexpr.
   Default-construct `Distribution{}` since all info is in NTTPs (no data members).
7. **CK_TILE origin uses `multi_index<N>`** (= `array<index_t, N>`), not `tuple`.
   `ck_tile::multi_index<2>{m, k}` for origins, `ck_tile::make_tuple(number<M>{}, number<K>{})` for lengths.

---

## MINT Critical Bug Fix (Required)

**Problem:** `include/mint/core/nd_array.h` uses `std::copy` in `MINT_HOST_DEVICE` code.

**Fix:** Replace `std::copy` with manual loops in two constructors:

**Lines 18-26:**
```cpp
MINT_HOST_DEVICE constexpr nd_array(initializer_list<sub_nd_array_type> init)
    : data_{} {
  if constexpr (kN > 0) {
    index_t idx = 0;
    for (auto it = init.begin(); it != init.end(); ++it, ++idx) {
      data_[idx] = *it;
    }
  }
}
```

**Lines 91-96:**
```cpp
MINT_HOST_DEVICE constexpr nd_array(initializer_list<T> init) : data_{} {
  index_t idx = 0;
  for (auto it = init.begin(); it != init.end(); ++it, ++idx) {
    data_[idx] = *it;
  }
}
```

---

## Code Style

- clang-format v18.1.3 with `.clang-format` config (100-char line limit, 4-space indent)
- Header-only library (all code in `.hpp` / `.h` headers)
- C++20 with concepts
- `SPDX-License-Identifier: MIT` header on all files
- Namespace: `ck_tile`, `mint`, `unified_tile`

## Hardware

| Architecture | GPU | Notes |
|--------------|-----|-------|
| gfx942 | MI300X | Primary development target |
| gfx950 | MI350X | Newer generation |
| gfx90a | MI200 | CDNA2 |
| gfx908 | MI100 | CDNA1 |

**Instruction sets:** MFMA (Matrix Fused Multiply-Add), XDL, DPP, WMMA

---

**Last Updated:** 2026-04-07
**Tested GPU:** gfx942 (MI300X)
**ROCm Version:** 7.2.0
**Branch:** `users/khuagarw/ck/mint`
