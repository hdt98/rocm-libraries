// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared argument struct and compile-time configuration for rocm_ck vector add.
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files) to share the kernel ABI and
// compile-time-validated configuration.

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/resolve.hpp>
#include <rocm_ck/tensor_desc.hpp>
#include <rocm_ck/types.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace rocm_ck {

/// Kernel arguments passed by value through hipModuleLaunchKernel.
/// Layout must match exactly between host and device.
///
/// Always computes c = alpha * a + beta * b. For plain addition, pass
/// alpha = 1.0f and beta = 1.0f. This matches the BLAS convention where
/// scalar parameters are always present.
struct VectorAddArgs
{
    index_t n;
    float alpha;
    float beta;
    // 4 bytes implicit padding (pointers need 8-byte alignment)
    const void* a;
    const void* b;
    void* c;
};

static_assert(std::is_trivially_copyable_v<VectorAddArgs>,
              "VectorAddArgs must be trivially copyable for kernarg passing");
static_assert(sizeof(VectorAddArgs) == 40, "unexpected VectorAddArgs size");
static_assert(alignof(VectorAddArgs) == 8, "unexpected VectorAddArgs alignment");
static_assert(offsetof(VectorAddArgs, n) == 0, "unexpected offset for n");
static_assert(offsetof(VectorAddArgs, alpha) == 4, "unexpected offset for alpha");
static_assert(offsetof(VectorAddArgs, beta) == 8, "unexpected offset for beta");
static_assert(offsetof(VectorAddArgs, a) == 16, "unexpected offset for a");
static_assert(offsetof(VectorAddArgs, b) == 24, "unexpected offset for b");
static_assert(offsetof(VectorAddArgs, c) == 32, "unexpected offset for c");

/// Signature: describes WHAT the kernel computes (types).
///
/// Uses an optional dtype hierarchy for concise specification:
///
///     dtype                    (kernel-level default)
///     ├── in_dtype             (input default, overrides dtype for inputs)
///     │   ├── a_dtype          (overrides in_dtype)
///     │   └── b_dtype          (overrides in_dtype)
///     └── out_dtype            (overrides dtype for output)
///
/// Specify only what differs — use {.dtype = FP32} for homogeneous kernels,
/// {.in_dtype = FP16, .out_dtype = FP32} for mixed types, or override
/// individual fields for asymmetric inputs. Call resolve_tensors() to flatten
/// the hierarchy into concrete TensorDesc entries.
struct ElementwiseSignature
{
    std::optional<DataType> dtype;     // kernel-level default
    std::optional<DataType> in_dtype;  // input default (overrides dtype)
    std::optional<DataType> a_dtype;   // input A (overrides in_dtype)
    std::optional<DataType> b_dtype;   // input B (overrides in_dtype)
    std::optional<DataType> out_dtype; // output (overrides dtype; NOT from in_dtype)

    // Operation: binary add with named tensor slots
    struct Add
    {
        std::string_view lhs = "A";
        std::string_view rhs = "B";
        std::string_view out = "out";
    };
    Add op{};
};

/// Algorithm: describes HOW the kernel executes (tile geometry, pipeline).
/// In CK Tile's builder pattern, this is the "how" — independent of
/// the data types being computed.
struct ElementwiseAlgorithm
{
    int block_tile;  // Elements processed per thread block (BlockTile)
    int block_warps; // Number of warps per thread block (BlockWarps)
    int warp_tile;   // Warp tile size for vector width calculation (WarpTile)
    bool pad;        // Enable padding for unaligned problem sizes
};

/// Compile-time configuration for an elementwise kernel variant.
/// This is the user-facing API — contains a Signature (what) and Algorithm (how).
struct ElementwiseConfig
{
    ElementwiseSignature signature;
    ElementwiseAlgorithm algorithm;
};

/// Validated kernel descriptor used as NTTP and host launch info.
/// All members are structural types (no std::optional, no pointers, etc.).
struct VectorAddKernel
{
    int block_tile;        // Elements per thread block (for grid calculation)
    int thread_block_size; // Threads per block (= warp_size * block_warps)
    DataType in_dtype;     // Input storage type (a, b)
    DataType out_dtype;    // Output storage type (c)
    int block_warps;       // Warps per block
    int warp_tile;         // Warp tile size
    bool pad;              // Padding enabled
};

/// Resolve the optional dtype hierarchy into concrete TensorDesc entries.
///
/// Resolution chains:
///   a_dtype   = a_dtype   ?? in_dtype ?? dtype ?? error
///   b_dtype   = b_dtype   ?? in_dtype ?? dtype ?? error
///   out_dtype = out_dtype ?? dtype    ?? error
///
/// Note: in_dtype does NOT cascade to out_dtype — they are separate branches.
consteval std::array<TensorDesc, 3> resolve_tensors(ElementwiseSignature sig)
{
    DataType a = sig.a_dtype    ? *sig.a_dtype
                 : sig.in_dtype ? *sig.in_dtype
                 : sig.dtype    ? *sig.dtype
                                : throw "a_dtype unresolvable: set a_dtype, in_dtype, or dtype";

    DataType b = sig.b_dtype    ? *sig.b_dtype
                 : sig.in_dtype ? *sig.in_dtype
                 : sig.dtype    ? *sig.dtype
                                : throw "b_dtype unresolvable: set b_dtype, in_dtype, or dtype";

    DataType out = sig.out_dtype ? *sig.out_dtype
                   : sig.dtype   ? *sig.dtype
                                 : throw "out_dtype unresolvable: set out_dtype or dtype";

    return {TensorDesc{"A", a, 1, Layout::Contiguous},
            TensorDesc{"B", b, 1, Layout::Contiguous},
            TensorDesc{"out", out, 1, Layout::Contiguous}};
}

// --- resolve_tensors compile-time tests ---
// clang-format off

// Homogeneous: dtype sets everything
static_assert(resolve_tensors({.dtype = DataType::FP32})[0].dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP32})[1].dtype == DataType::FP32);
static_assert(resolve_tensors({.dtype = DataType::FP32})[2].dtype == DataType::FP32);

// Mixed-type: dtype or in_dtype for inputs, out_dtype for output
static_assert(resolve_tensors({.dtype = DataType::FP16, .out_dtype = DataType::FP32})[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP16, .out_dtype = DataType::FP32})[2].dtype == DataType::FP32);
static_assert(resolve_tensors({.in_dtype = DataType::FP16, .out_dtype = DataType::FP32})[0].dtype == DataType::FP16);

// Override chain: in_dtype overrides dtype for inputs, NOT for output
static_assert(resolve_tensors({.dtype = DataType::FP32, .in_dtype = DataType::FP16})[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP32, .in_dtype = DataType::FP16})[2].dtype == DataType::FP32);

// Per-operand overrides: a_dtype/b_dtype override in_dtype
static_assert(resolve_tensors({.a_dtype = DataType::FP16, .b_dtype = DataType::FP16, .out_dtype = DataType::FP32})[0].dtype == DataType::FP16);
static_assert(resolve_tensors({.dtype = DataType::FP32, .a_dtype = DataType::FP16, .b_dtype = DataType::FP16})[2].dtype == DataType::FP32);

// TensorDesc metadata: name, rank, layout
static_assert(resolve_tensors({.dtype = DataType::FP32})[0].name == "A");
static_assert(resolve_tensors({.dtype = DataType::FP32})[1].name == "B");
static_assert(resolve_tensors({.dtype = DataType::FP32})[2].name == "out");
static_assert(resolve_tensors({.dtype = DataType::FP32})[0].rank == 1);
static_assert(resolve_tensors({.dtype = DataType::FP32})[2].rank == 1);
static_assert(resolve_tensors({.dtype = DataType::FP32})[0].layout == Layout::Contiguous);
static_assert(resolve_tensors({.dtype = DataType::FP32})[1].layout == Layout::Contiguous);
static_assert(resolve_tensors({.dtype = DataType::FP32})[2].layout == Layout::Contiguous);

// Operation slots match tensor names
static_assert(ElementwiseSignature{}.op.lhs == "A");
static_assert(ElementwiseSignature{}.op.rhs == "B");
static_assert(ElementwiseSignature{}.op.out == "out");

// Error cases (uncommenting would produce consteval compile errors):
// resolve_tensors({})                                     — nothing resolvable
// resolve_tensors({.a_dtype = DataType::FP16})            — b_dtype, out_dtype unknown
// resolve_tensors({.in_dtype = DataType::FP16})           — out_dtype unknown
// resolve_tensors({.out_dtype = DataType::FP32})          — inputs unknown
// clang-format on

/// Validate an elementwise config and produce a structural kernel descriptor.
///
/// Resolves the signature's optional dtype hierarchy, validates that vector add
/// inputs match (a_dtype == b_dtype), then checks CK Tile ElementWiseShape
/// compatibility:
///   kVectorM = min(128 / max_type_bits, warp_tile / warp_size) — must be >= 1
///   kRepeatM = block_tile / (block_warps * kVectorM * warp_size) — must be >= 1, integer
///   block_warps must be power of 2 (required by CK Tile reduce_on_sequence)
///   thread_block_size = warp_size * block_warps (NOT block_tile)
///
/// For mixed types, kVectorM is constrained by the wider type (fewer elements
/// per 128-bit register).
///
/// Invalid configs produce a compile-time error (consteval).
consteval VectorAddKernel make_kernel(ElementwiseConfig cfg)
{
    std::array<TensorDesc, 3> tensors = resolve_tensors(cfg.signature);
    ElementwiseAlgorithm algo         = cfg.algorithm;

    if(tensors[0].dtype != tensors[1].dtype)
        throw "vector add requires a_dtype == b_dtype";

    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_warps <= 0)
        throw "block_warps must be positive";
    if((algo.block_warps & (algo.block_warps - 1)) != 0)
        throw "block_warps must be a power of 2 (required by CK Tile reduce_on_sequence)";
    if(algo.warp_tile <= 0)
        throw "warp_tile must be positive";
    if(algo.warp_tile < warp_size)
        throw "warp_tile must be >= warp_size (64)";

    int in_bits    = data_type_bits(tensors[0].dtype);
    int out_bits   = data_type_bits(tensors[2].dtype);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;             // max vector from wider type
    int kVectorM_b = algo.warp_tile / warp_size; // max vector from warp tile
    int kVectorM   = kVectorM_a < kVectorM_b ? kVectorM_a : kVectorM_b;

    if(kVectorM < 1)
        throw "computed kVectorM must be >= 1 (warp_tile too small for type)";

    int elements_per_iter = algo.block_warps * kVectorM * warp_size;
    if(algo.block_tile % elements_per_iter != 0)
        throw "block_tile must be divisible by (block_warps * kVectorM * warp_size)";

    int kRepeatM = algo.block_tile / elements_per_iter;
    if(kRepeatM < 1)
        throw "computed kRepeatM must be >= 1 (block_tile too small for given warps)";

    return {algo.block_tile,
            warp_size * algo.block_warps,
            tensors[0].dtype,
            tensors[2].dtype,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

/// Check if problem size N is aligned to a variant's block_tile (no padding needed).
constexpr bool isAligned(VectorAddKernel k, int n) { return n > 0 && n % k.block_tile == 0; }

// ============================================================================
// make_kernel overload for operator-centric Signature
// ============================================================================

/// Resolve and validate a vector add using the operator-centric Signature.
///
/// Pattern-matches: first op must be AddOp. For standalone AddOp
/// (no preceding GemmOp), tensors get rank=1, Layout::Contiguous
/// as elementwise defaults.
consteval VectorAddKernel make_kernel(Signature sig, ElementwiseAlgorithm algo)
{
    ResolvedSignature resolved = resolve(sig);

    // First op must be AddOp
    if(!std::holds_alternative<AddOp>(sig.ops[0]))
        throw "vector add make_kernel requires AddOp as first operator";
    const AddOp& add = std::get<AddOp>(sig.ops[0]);

    // Remaining ops must be empty
    for(int i = 1; i < kMaxOps; ++i)
        if(!std::holds_alternative<std::monostate>(sig.ops[i]))
            throw "vector add make_kernel only supports a single AddOp";

    TensorDesc a_td   = resolved.tensor(add.lhs);
    TensorDesc b_td   = resolved.tensor(add.rhs);
    TensorDesc out_td = resolved.tensor(add.out);

    if(a_td.dtype != b_td.dtype)
        throw "vector add requires matching input dtypes";

    if(algo.block_tile <= 0)
        throw "block_tile must be positive";
    if(algo.block_warps <= 0)
        throw "block_warps must be positive";
    if((algo.block_warps & (algo.block_warps - 1)) != 0)
        throw "block_warps must be a power of 2 (required by CK Tile reduce_on_sequence)";
    if(algo.warp_tile <= 0)
        throw "warp_tile must be positive";
    if(algo.warp_tile < warp_size)
        throw "warp_tile must be >= warp_size (64)";

    int in_bits    = data_type_bits(a_td.dtype);
    int out_bits   = data_type_bits(out_td.dtype);
    int max_bits   = in_bits > out_bits ? in_bits : out_bits;
    int kVectorM_a = 128 / max_bits;
    int kVectorM_b = algo.warp_tile / warp_size;
    int kVectorM   = kVectorM_a < kVectorM_b ? kVectorM_a : kVectorM_b;

    if(kVectorM < 1)
        throw "computed kVectorM must be >= 1 (warp_tile too small for type)";

    int elements_per_iter = algo.block_warps * kVectorM * warp_size;
    if(algo.block_tile % elements_per_iter != 0)
        throw "block_tile must be divisible by (block_warps * kVectorM * warp_size)";

    int kRepeatM = algo.block_tile / elements_per_iter;
    if(kRepeatM < 1)
        throw "computed kRepeatM must be >= 1 (block_tile too small for given warps)";

    return {algo.block_tile,
            warp_size * algo.block_warps,
            a_td.dtype,
            out_td.dtype,
            algo.block_warps,
            algo.warp_tile,
            algo.pad};
}

// --- Signature-based make_kernel equivalence tests ---
// clang-format off

// --- New Signature produces same kernel as old ElementwiseConfig ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP32,
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).in_dtype == DataType::FP32);

static_assert(make_kernel(
    Signature{.dtype = DataType::FP32,
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).out_dtype == DataType::FP32);

// --- Equivalence: matches old path ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP32,
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).block_tile ==
    make_kernel(ElementwiseConfig{
        .signature = {.dtype = DataType::FP32},
        .algorithm = {1024, 1, 1024, true}}).block_tile);

// --- Mixed types via new Signature ---
static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).in_dtype == DataType::FP16);

static_assert(make_kernel(
    Signature{.dtype = DataType::FP16,
              .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
              .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    ElementwiseAlgorithm{1024, 1, 1024, true}).out_dtype == DataType::FP32);
// clang-format on

} // namespace rocm_ck