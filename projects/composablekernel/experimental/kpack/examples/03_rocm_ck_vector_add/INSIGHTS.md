# Insights: Prototyping the rocm_ck API

This example is the design prototype for the production rocm_ck API. It validates config validation, variant management, type dispatch, and host/device separation patterns that will scale to GEMM, convolution, and FMHA.

## 1. Config -> consteval -> Kernel: The Three-Layer Pattern

User-facing `ElementwiseConfig` (uses `std::optional`, not valid as an NTTP) is transformed by `consteval make_kernel()` into `VectorAddKernel` (a structural type, valid as an NTTP). This separation is fundamental:

- **Config** — flexible, user-facing. Uses `std::optional` fields so users specify only what differs from defaults.
- **consteval** — validates and transforms. Throws at compile time if the config is invalid. Invalid configs never produce binaries.
- **Kernel** — structural, concrete. Every field has a definite value. Used as a C++20 NTTP to instantiate device templates.

```cpp
// User writes:
constexpr auto kernel = make_kernel(ElementwiseConfig{
    .signature = {.dtype = DataType::FP32},
    .algorithm = {.block_tile = 1024, .block_warps = 1, .warp_tile = 1024, .pad = true}});

// make_kernel returns a VectorAddKernel with all fields resolved and validated.
// This value is used directly as a template parameter: runVectorAdd<kernel>(args)
```

> **Production implication**: Every rocm_ck operation will follow this pattern. The Config struct is the public API; the Kernel struct is the internal interface. `consteval` is the bridge that enforces invariants.

## 2. Optional Dtype Hierarchy

`ElementwiseSignature` uses 5 optional fields with clear resolution chains:

```
dtype                    (kernel-level default)
+-- in_dtype             (overrides dtype for inputs)
|   +-- a_dtype          (overrides in_dtype)
|   +-- b_dtype          (overrides in_dtype)
+-- out_dtype            (overrides dtype for output -- NOT from in_dtype)
```

`in_dtype` does NOT cascade to `out_dtype`. They are separate branches rooted at `dtype`. This prevents ambiguity: setting `in_dtype = FP16` with `dtype = FP32` gives FP16 inputs and FP32 output, not FP16 everywhere.

The `resolve_types` consteval function flattens the hierarchy. 10 `static_assert` tests verify all resolution paths — homogeneous, widening, narrowing, and per-operand overrides.

> **Production implication**: Each operation defines its own signature struct with the `DataType` fields it needs. GEMM will have `{a_dtype, b_dtype, c_dtype, compute_dtype}`; FMHA will have its own set. One-size-fits-all signatures would be either too restrictive or too confusing.

## 3. consteval for Immediate Feedback

Throwing in a `consteval` function produces a compile error at the call site with the throw message as the diagnostic. `make_kernel` validates 7 constraints:

- Input types must match (`a_dtype == b_dtype` for vector add)
- Tile dimensions must be positive
- `block_warps` must be a power of 2
- `kVectorM` must be >= 1 (validates type/tile compatibility)
- `block_tile` must be divisible by `(block_warps * kVectorM * warp_size)`
- `kRepeatM` must be >= 1

Invalid configs fail at compile time with readable messages like `"block_tile must be divisible by (block_warps * kVectorM * warp_size)"`. This is dramatically better than runtime validation — broken kernel configs never produce `.hsaco` files.

## 4. Signature / Algorithm Separation

"What" (data types) vs "How" (tile geometry) are orthogonal concerns:

- **Signature**: `{dtype, in_dtype, a_dtype, b_dtype, out_dtype}` — specifies the types
- **Algorithm**: `{block_tile, block_warps, warp_tile, pad}` — specifies the tile geometry

The same signature can be paired with different algorithms to explore the tuning surface. The same algorithm can run on FP32, FP16, or BF16 (subject to `kVectorM` constraints validated by `make_kernel`).

> **Production implication**: Tuning tools generate Algorithm variants; the Signature comes from the user's problem specification. This composition is how rocm_ck will expose its tuning surface.

## 5. _api / _dev Interface Model

Two header files, strict dependency boundary:

| Header | CK Tile dependency | Used by |
|--------|-------------------|---------|
| `rocm_vector_add_api.hpp` | None | Host, device, tests |
| `rocm_vector_add_dev.hpp` | Yes (CK Tile) | Device only |

The API header contains the config types, `make_kernel`, `VectorAddArgs`, `VectorAddKernel`, and all compile-time validation. It is testable with `constexpr`/`consteval` without any GPU — the 10 `static_assert` tests in the source run on any C++20 compiler.

The device header contains the CK Tile kernel implementation, `CkTypeMap`, and `runVectorAdd<K>`. It is compiled only with `--cuda-device-only` and never included in host code.

> **Production implication**: This is the model for all rocm_ck operation headers. The `_api` header is the public interface; the `_dev` header is an implementation detail. Consumers include only `_api`.

## 6. VectorAddArgs ABI Validation

The args struct uses `void*` pointers (not typed), has explicit padding awareness, and 6 `static_assert` checks:

```cpp
struct VectorAddArgs {
    index_t n;          // offset 0
    float alpha;        // offset 4
    float beta;         // offset 8
    // 4 bytes implicit padding
    const void* a;      // offset 16
    const void* b;      // offset 24
    void* c;            // offset 32
};

static_assert(std::is_trivially_copyable_v<VectorAddArgs>);
static_assert(sizeof(VectorAddArgs) == 40);
static_assert(alignof(VectorAddArgs) == 8);
static_assert(offsetof(VectorAddArgs, a) == 16);
// ... 3 more offset checks
```

This is the binary contract between host and device across separate compilation boundaries. Pointers are `void*` because the actual type depends on the variant (FP32, FP16, BF16) — type dispatch happens inside the device kernel via `CkTypeMap`.

> **Production implication**: Every kernel's args struct needs this level of ABI validation. A single-byte offset mismatch between host and device produces silent data corruption.

## 7. One .hip File Per Variant

Each variant file is ~15 lines:

```cpp
#include "rocm_vector_add_dev.hpp"
static constexpr auto kernel = rocm_ck::make_kernel(
    rocm_ck::ElementwiseConfig{
        .signature = {.dtype = rocm_ck::DataType::FP32},
        .algorithm = {.block_tile = 1024, ...}});

extern "C" __global__ void vector_add_fp32_b1024(rocm_ck::VectorAddArgs args) {
    rocm_ck::runVectorAdd<kernel>(args);
}
```

Separate files enable parallel compilation and clear traceability — each `.hip` produces one `.hsaco` per architecture. CMake loops over `(variant x arch)` to produce `N x M` code objects.

> **Production implication**: Variant generation can be automated. The `.hip` file is mechanical — operation + config + symbol name. A code generator producing these files from a variant table is a natural next step.

## 8. CkTypeMap for Enum -> Type Dispatch

Maps the runtime `DataType` enum to CK Tile types on the device side:

```cpp
template <DataType> struct CkTypeMap;  // primary template undefined

template <> struct CkTypeMap<DataType::FP32> { using type = float; };
template <> struct CkTypeMap<DataType::FP16> { using type = ck_tile::half_t; };
template <> struct CkTypeMap<DataType::BF16> { using type = ck_tile::bf16_t; };
```

The primary template is intentionally undefined — only valid specializations compile. An unsupported `DataType` enum value produces a compile error, not a runtime failure.

This is the device-side bridge between the enum-based API (shared with the host) and CK Tile's template system (device-only).

## 9. Custom CK Tile Kernel Using Primitives

`runVectorAdd` uses `load_tile`, `sweep_tile_span`, `store_tile` directly instead of the stock `ElementWiseKernel`:

```cpp
auto a_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(args.a)));
auto b_tile = ck_tile::load_tile(make_input_window(static_cast<const X*>(args.b)));

auto y_tile = ck_tile::make_static_distributed_tensor<Y>(a_tile.get_tile_distribution());

ck_tile::sweep_tile_span(spans[ck_tile::number<0>{}], [&](auto idx) {
    const float a_val = ck_tile::type_convert<float>(a_tile(make_tuple(idx)));
    const float b_val = ck_tile::type_convert<float>(b_tile(make_tuple(idx)));
    y_tile(make_tuple(idx)) = ck_tile::type_convert<Y>(alpha * a_val + beta * b_val);
});

ck_tile::store_tile(y_window, y_tile);
```

The stock `ElementWiseKernel` default-constructs its functor — no path to pass runtime `alpha`/`beta`. Building from primitives enables scalar parameters and mixed-type compute (convert inputs to float, compute, convert output).

> **Production implication**: Production kernels will be built from CK Tile building blocks, not from stock kernels. The primitives (`load_tile`, `sweep_tile_span`, `store_tile`, tile distributions) are the actual API surface of CK Tile for rocm_ck purposes.

## 10. kVectorM: Wider Type Governs Mixed-Type Vector Loads

For mixed-type operations, the wider type constrains vector width:

```
kVectorM = min(128 / max_type_bits, warp_tile / warp_size)
```

FP16+FP32 mixed: `max_type_bits = 32`, so `kVectorM_a = 4` (vs 8 for homogeneous FP16). The constraint comes from the 128-bit vector register width — wider elements mean fewer elements per vector load/store.

This is validated at `consteval` time. A tile configuration that works for FP32 may not work for FP8 (which allows 16 elements per vector but requires larger minimum block tiles).

## 11. thread_block_size = warp_size * block_warps, NOT block_tile

Common confusion caught during development. Block tile determines elements per block; threads are determined by warp count:

```
thread_block_size = warp_size * block_warps    (threads launched per block)
block_tile        = thread_block_size * kRepeatM * kVectorM  (elements processed per block)
```

A `block_tile = 1024` with `block_warps = 1` still launches only 64 threads — each warp iterates `kRepeatM` times, processing `kVectorM` elements per iteration.

## 12. Variant Registry and Selection

`ALL_VARIANTS[]` is a constexpr table of `{name, kernel}` pairs. `findVariant()` selects at runtime:

1. Filter by matching `in_dtype` and `out_dtype`
2. Among matches, prefer the largest `block_tile` that divides the problem size evenly (aligned)
3. Fall back to the largest padded variant if no aligned variant exists

This is a simple but effective model. Production will need more dimensions (architecture-specific variants, memory layout preferences), but the basic pattern — constexpr table + runtime selection function — scales naturally.

## 13. Archive Metadata Mirrors Source

`pack.py` embeds tuning parameters in the kpack TOC's `variant_metadata` section: block tile, block warps, warp tile, pad flag, input/output dtypes. This enables tooling to inspect archives without source code — a profiler can read the metadata to understand what kernel configuration produced a particular performance result.

Basic readers ignore the metadata; advanced tools consume it. The kpack format supports this naturally because MessagePack maps are extensible.

## 14. Shared Utility Headers as the Beginning of rocm_ck

`include/rocm_ck/` contains `datatype_utils.hpp`, `gpu_arch.hpp`, `hip_check.hpp` — independent of CK Tile, reusable across operations. These follow CK's own convention (`include/ck/`, `include/ck_tile/`) and are the seed of the rocm_ck public include library.

These headers reduce boilerplate and establish common patterns (error checking, type conversion, architecture detection) that all rocm_ck operations will share.