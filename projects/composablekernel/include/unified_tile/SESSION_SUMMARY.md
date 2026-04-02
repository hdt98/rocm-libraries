# Unified Tile Project - Session Summary

**Last Updated:** 2026-04-02
**Current Phase:** Tasks 00-01 COMPLETE, Task 02 (Distribution Config) next
**Plan File:** `~/.claude/plans/rosy-growing-wilkinson.md`

---

## Project Goal

Create a zero-overhead unified interface layer between CK_Tile and MINT backends for GPU GEMM operations. Users write a GEMM kernel once using `unified_tile` APIs and compile it against either backend via a compile flag (`-DUNIFIED_TILE_BACKEND_CK_TILE` or `-DUNIFIED_TILE_BACKEND_MINT`).

---

## 10-Task Roadmap

| Task | Description | Status |
|------|-------------|--------|
| 00 | Descriptor fix & test | COMPLETE |
| 01 | Tensor View (`view.hpp`, `address_space.hpp`) | COMPLETE |
| 02 | Distribution Config (`distribution_config.hpp`, block copy, warp gemm) | Next |
| 03 | Tile Window (`window.hpp`, `partition.hpp`) | Pending |
| 04 | Load Operations (`load.hpp`, `load_policy.hpp`) | Pending |
| 05 | Store Operations (`store.hpp`, `store_policy.hpp`) | Pending |
| 06 | Window Movement (`move_window.hpp`) | Pending |
| 07 | Warp-Level MFMA (`gemm.hpp`, `gemm_config.hpp`) | Pending |
| 08 | Sync & Distributed Tensor (`sync.hpp`, `distributed_tensor.hpp`) | Pending |
| 09 | Full GEMM Pipeline (`gemm_pipeline.hpp`, `gemm_shape.hpp`) | Pending |

**Dependency:** 00 -> 01 -> 02 (parallel with 01) -> 03 -> {04,05,06,07,08} (parallel) -> 09

---

## Completed Work

### Task 00: Descriptor Testing & Stabilization COMPLETE

**Bugs fixed in existing code:**

1. **`config.hpp`** - Added `UNIFIED_TILE_HOST_DEVICE` / `UNIFIED_TILE_DEVICE` / `UNIFIED_TILE_HOST` macros
   - Maps to `CK_TILE_HOST_DEVICE` or `MINT_HOST_DEVICE` based on active backend
   - All unified_tile code must use these instead of backend-specific annotations

2. **`type_aliases.hpp`** - Two fixes:
   - `make_multi_index` was using `ck_tile::number<values>{}...` (compile-time only); replaced with runtime-compatible overloads
   - Replaced all `CK_TILE_HOST_DEVICE` with `UNIFIED_TILE_HOST_DEVICE`

3. **`type_concepts.hpp`** - Replaced `CK_TILE_HOST_DEVICE` with `UNIFIED_TILE_HOST_DEVICE`

4. **`descriptor.hpp`** - Major refactor:
   - Split public API into `#ifdef UNIFIED_TILE_BACKEND_CK_TILE` / `#else` blocks (no more cross-backend `ck_tile::tuple` dependency)
   - **Renamed functions** to avoid ADL ambiguity (see Key Learnings below):
     - `make_descriptor(dims...)` - variadic convenience
     - `make_strided_descriptor(lengths, strides)` - explicit strides
     - `make_packed_descriptor(lengths)` - packed/row-major
     - `make_descriptor_with_strides(lengths, strides)` - convenience wrapper
   - Added unified query functions with backend dispatch:
     - `get_num_dimensions(desc)` - CK_TILE: `desc.get_num_of_dimension()`, MINT: `Descriptor::top_ndim()`
     - `get_lengths(desc)` - CK_TILE: `desc.get_lengths()`, MINT: `desc.top_lengths()`

**Files created:**

| File | Purpose |
|------|---------|
| `example/ck_tile/unified_tile/00_descriptor/main.cpp` | Example demonstrating descriptor creation |
| `example/ck_tile/unified_tile/00_descriptor/README.md` | Documentation |
| `example/ck_tile/unified_tile/00_descriptor/CMakeLists.txt` | Build config (both backends) |
| `test/unified_tile/test_descriptor.cpp` | Google Test with 3 test cases |
| `test/unified_tile/CMakeLists.txt` | Test build config (both backends) |

**Files modified:**

| File | Change |
|------|--------|
| `example/ck_tile/CMakeLists.txt` | Added `add_subdirectory(unified_tile)` |
| `example/ck_tile/unified_tile/CMakeLists.txt` | Created, includes `00_descriptor` |
| `test/CMakeLists.txt` | Added `add_subdirectory(unified_tile)` |

**Test results (gfx942):**
- `test_unified_tile_descriptor_ck`: 3/3 PASSED
- `test_unified_tile_descriptor_mint`: 3/3 PASSED
- `example_unified_tile_00_descriptor_ck`: All tests passed
- `example_unified_tile_00_descriptor_mint`: All tests passed

### Task 01: Tensor View Abstraction COMPLETE

**Files created:**

| File | Purpose |
|------|---------|
| `include/unified_tile/tensor/address_space.hpp` | Unified `address_space` enum (global, shared, vgpr) |
| `include/unified_tile/tensor/view.hpp` | `make_tensor_view<AddrSpace>()`, `pad_view()`, `get_view_size()` |
| `example/ck_tile/unified_tile/01_view/main.cpp` | Example demonstrating view creation |
| `example/ck_tile/unified_tile/01_view/README.md` | Documentation |
| `example/ck_tile/unified_tile/01_view/CMakeLists.txt` | Build config (both backends) |
| `test/unified_tile/test_view.cpp` | Google Test with 3 test cases |

**API provided:**

| Unified | CK_TILE | MINT |
|---------|---------|------|
| `make_tensor_view<global>(ptr, desc)` | `ck_tile::make_tensor_view<global>(ptr, desc)` | `mint::make_tensor_view(desc, make_global_memory_view(ptr, size))` |
| `make_tensor_view<shared>(ptr, desc)` | `ck_tile::make_tensor_view<lds>(ptr, desc)` | `mint::make_tensor_view(desc, make_shared_memory_view(ptr, size))` |
| `pad_view(view, tile_lens, do_pads)` | `ck_tile::pad_tensor_view(...)` | returns view unchanged (masking at load/store) |
| `get_view_size(view)` | `view.get_tensor_descriptor().get_element_space_size()` | `view.memory_view().size()` |

**Key learnings:**
- CK_TILE's `make_tensor_view` internally calls `make_buffer_view`, so `buffer_view.hpp` must be included alongside `tensor_view.hpp`
- `pad_tensor_view` does NOT increase `get_element_space_size()` - it adds right-pad transforms that return 0 for OOB reads, but the underlying buffer size stays the same

**Test results (gfx942):**
- `test_unified_tile_view_ck`: 3/3 PASSED (GlobalViewSize, StridedViewSize, PaddedViewSize)
- `test_unified_tile_view_mint`: 3/3 PASSED (GlobalViewSize, StridedViewSize, PaddedViewSize)

---

## Key Design Decisions

### 1. Traits-Based Architecture (CRITICAL)

**Pattern for ALL files:**

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
struct some_traits<backend_type::mint> {
    static constexpr auto do_something(...) {
        return mint::backend_function(...);
    }
};
#endif

// Public API routes through traits (avoids ADL issues)
auto unified_api(...) {
    return some_traits<active_backend>::do_something(...);
}
```

### 2. Separate Public APIs Per Backend (CRITICAL)

**The public API is split into `#ifdef` blocks**, not a single template dispatching on `Backend`. This avoids:
- ADL ambiguity when passing `ck_tile::tuple` to functions named similarly to `ck_tile::` functions
- Compilation errors from MINT code seeing `ck_tile::tuple` types when only MINT is active

```cpp
#ifdef UNIFIED_TILE_BACKEND_CK_TILE
template <typename... Dims>
auto make_descriptor(Dims... dims) {
    return tensor_descriptor_traits<backend_type::ck_tile>::make_naive_packed(
        ck_tile::make_tuple(dims...));
}
#else
template <typename... Dims>
auto make_descriptor(Dims... dims) {
    return tensor_descriptor_traits<backend_type::mint>::make_naive_packed(
        mint::nd_index<sizeof...(Dims)>{static_cast<mint::index_t>(dims)...});
}
#endif
```

### 3. MINT Dimension Aliases

**MINT uses NUMERIC aliases, not strings:**

```cpp
// Dimensions: {1, 2, 3, ..., N}
// Offset: {0}
constexpr auto dim_seq = mint::sequence<index_t, 1, 2, 3>{};
constexpr auto offset_alias = mint::integral_constant<index_t, 0>{};
```

### 4. Host-Side Memory Management

**Use `ck_tile::DeviceMem` and `HIP_CHECK_ERROR`** (not raw `hipMalloc`/`hipFree`):

```cpp
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"

ck_tile::DeviceMem buf(size_in_bytes);
buf.SetZero();
kernel<<<grid, block>>>(reinterpret_cast<T*>(buf.GetDeviceBuffer()));
HIP_CHECK_ERROR(hipDeviceSynchronize());
buf.FromDevice(host_ptr);
```

---

## Key Learnings

### ADL (Argument-Dependent Lookup) Trap

When a function in `unified_tile::descriptor` namespace is named `make_naive_tensor_descriptor` and receives `ck_tile::tuple` arguments, the compiler also finds `ck_tile::make_naive_tensor_descriptor` via ADL, causing ambiguity.

**Solution:** Name unified functions differently:
- `make_descriptor()` instead of `make_naive_tensor_descriptor_packed()`
- `make_strided_descriptor()` instead of `make_naive_tensor_descriptor()`
- Route through `tensor_descriptor_traits<Backend>::make_naive()` internally

### MINT vs CK_TILE API Differences

| Operation | CK_TILE | MINT |
|-----------|---------|------|
| Num dimensions | `desc.get_num_of_dimension()` | `Descriptor::top_ndim()` (static) |
| Get lengths | `desc.get_lengths()` | `desc.top_lengths()` |
| Host/device annotation | `CK_TILE_HOST_DEVICE` | `MINT_HOST_DEVICE` |
| Index type | `ck_tile::index_t` (int32_t) | `mint::index_t` (int64_t) |
| Compile-time constant | `ck_tile::number<N>` | `mint::constant<N>` |
| Tuple | `ck_tile::tuple<Ts...>` | `mint::tuple<Ts...>` |
| Multi-index | `ck_tile::make_tuple(m, n)` | `mint::nd_index<2>{m, n}` |

---

## Build & Run (Inside Docker)

```bash
# All builds happen inside Docker container awsm_wilbur
# Host: /home/khuagarw1/repos/rocm-libraries/projects/composablekernel
# Docker: /root/workspace

# Reconfigure cmake (needed after adding new CMakeLists.txt)
docker exec awsm_wilbur bash -c "cd /root/workspace/build && cmake .."

# Build specific targets
docker exec awsm_wilbur bash -c "cd /root/workspace/build && ninja <target_name>"

# Run
docker exec awsm_wilbur bash -c "cd /root/workspace/build && ./bin/<binary_name>"
```

**Target naming:**
- Examples: `example_unified_tile_00_descriptor_ck`, `example_unified_tile_00_descriptor_mint`
- Tests: `test_unified_tile_descriptor_ck`, `test_unified_tile_descriptor_mint`

**MINT-specific compile flags:** `-DUNIFIED_TILE_BACKEND_MINT -DMINT_BACKEND_ROCM -fgpu-rdc`
**MINT-specific link flags:** `-fgpu-rdc`

---

## Current Public API (descriptor.hpp)

```cpp
namespace unified_tile::descriptor {

// ---- Convenience (variadic) ----
auto make_descriptor(Dims... dims);
    // CK_TILE: make_naive_tensor_descriptor_packed(make_tuple(dims...))
    // MINT: make_aliased_naive_packed_tensor_descriptor(aliases, offset, {dims...})

// ---- With strides ----
auto make_descriptor_with_strides(lengths, strides);
    // CK_TILE: make_naive_tensor_descriptor(lengths, strides)
    // MINT: make_aliased_naive_tensor_descriptor(aliases, offset, lengths, strides)

// ---- Explicit (backend-typed args) ----
auto make_strided_descriptor(lengths, strides, vec_len, vec_stride);
auto make_packed_descriptor(lengths, vec_len);

// ---- Query ----
auto get_num_dimensions(desc);  // int
auto get_lengths(desc);         // tuple or nd_index
}
```

---

## Reference Files

| What | Path |
|------|------|
| Plan | `~/.claude/plans/rosy-growing-wilkinson.md` |
| CLAUDE.md | `CLAUDE.md` (repo root) |
| CK_TILE descriptor | `include/ck_tile/core/tensor/tensor_descriptor.hpp` |
| MINT descriptor | `include/mint/tensor/tensor_descriptor_helper.h` |
| CK_TILE tensor view | `include/ck_tile/core/tensor/tensor_view.hpp` |
| MINT tensor view | `include/mint/tensor/tensor_view.h` |
| CK_TILE tile window | `include/ck_tile/core/tensor/tile_window.hpp` |
| MINT distributed window | `include/mint/tensor/distributed_window.h` |
| CK_TILE load/store | `include/ck_tile/core/tensor/load_tile.hpp`, `store_tile.hpp` |
| MINT load/store | `include/mint/tile/generic/load_no_shuffle_vectorized.h`, `store_no_shuffle_vectorized.h` |
| CK_TILE GEMM pipeline | `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp` |
| MINT GEMM example | `example/ck_tile/mint_example/gemm_xdl_v1/rocm_gemm_xdl_v1.cpp` |
| CK_TILE MFMA | `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp` |
| MINT MFMA | `include/mint/tile/rocm/warp/matmul_xdl.h` |

---

## Common Mistakes to Avoid

1. **Using same function names as backend** - Causes ADL ambiguity. Use distinct names.
2. **Using `CK_TILE_HOST_DEVICE` in unified code** - Use `UNIFIED_TILE_HOST_DEVICE`.
3. **Single template function for both backends** - Split into `#ifdef` blocks.
4. **Using string aliases for MINT** - Use numeric aliases `{1, 2, 3, ...}`.
5. **Raw hipMalloc in examples** - Use `ck_tile::DeviceMem`.
6. **Calling backend-specific methods in user code** - Use unified query functions.
7. **Forgetting `-fgpu-rdc` for MINT** - Required for both compile and link.
8. **Building outside Docker** - Build dir was created inside Docker (`/root/workspace/build`).
