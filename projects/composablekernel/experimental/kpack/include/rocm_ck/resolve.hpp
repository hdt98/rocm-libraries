// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Signature resolution: resolves a Signature into concrete tensor descriptors.
//
// resolve() walks the operator graph, collects tensor slots with operator-implied
// defaults, propagates rank/layout through connected tensors, merges explicit
// Tensor entries, and applies the dtype cascade. All at compile time (consteval).
//
// This header has NO CK Tile dependency.

#pragma once

#include <rocm_ck/signature.hpp>
#include <rocm_ck/tensor_desc.hpp>

#include <array>

namespace rocm_ck {

/// Resolved signature: all tensors have concrete dtype.
/// Rank and layout are concrete for tensors with operator-implied defaults
/// or explicit Tensor entries; may remain 0/Auto for tensors where the
/// operation type does not specify them (e.g., standalone AddOp).
struct ResolvedSignature
{
    int num_tensors                             = 0;
    std::array<TensorDesc, kMaxTensors> tensors = {};

    /// Find a resolved tensor by name. Compile-time error if not found.
    consteval TensorDesc tensor(std::string_view name) const
    {
        for(int i = 0; i < num_tensors; ++i)
            if(tensors[i].name == name)
                return tensors[i];
        throw "tensor not found in resolved signature";
    }
};

/// Resolve a Signature into concrete tensor descriptors.
///
/// Phases:
///   1. Register tensor slots from operators (with op-implied rank/layout)
///   2. Propagate rank/layout through connected tensors (forward + backward)
///   3. Merge explicit Tensor entries from sig.tensors (overrides propagation)
///   4. Apply dtype cascade: explicit tensor -> sig.dtype -> error
///   5. SSA check: each tensor name produced by at most one operator
///
/// GemmOp slots get operator-implied defaults:
///   lhs -> rank 2, Row;  rhs -> rank 2, Col;  out -> rank 2, Row
///
/// Binary ops (AddOp, MulOp) and unary ops propagate rank/layout from
/// the first connected slot that has known values.
consteval ResolvedSignature resolve(Signature sig)
{
    // --- Intermediate tracking ---
    struct Info
    {
        std::string_view name;
        bool dtype_set = false;
        DataType dtype = DataType::FP32;
        int rank       = 0;
        Layout layout  = Layout::Auto;
    };

    Info infos[kMaxTensors] = {};
    int num                 = 0;

    // Find tensor by name, return index or -1.
    auto find = [&](std::string_view name) -> int {
        for(int i = 0; i < num; ++i)
            if(infos[i].name == name)
                return i;
        return -1;
    };

    // Find tensor by name; add if not present.
    auto find_or_add = [&](std::string_view name) -> int {
        if(name.empty())
            throw "operator slot has empty tensor name";
        int idx = find(name);
        if(idx >= 0)
            return idx;
        if(num >= kMaxTensors)
            throw "too many unique tensors (max kMaxTensors)";
        infos[num].name = name;
        return num++;
    };

    // Set rank/layout only if currently unknown.
    auto set_if_unknown = [&](int idx, int rank, Layout layout) {
        if(infos[idx].rank == 0 && rank != 0)
            infos[idx].rank = rank;
        if(infos[idx].layout == Layout::Auto && layout != Layout::Auto)
            infos[idx].layout = layout;
    };

    // ================================================================
    // Phase 1: Register tensor slots from operators
    // ================================================================
    std::string_view output_names[kMaxOps] = {};
    int num_outputs                        = 0;

    for(int i = 0; i < kMaxOps; ++i)
    {
        const Op& op = sig.ops[i];
        if(std::holds_alternative<std::monostate>(op))
            continue;

        std::string_view out_name;

        if(std::holds_alternative<GemmOp>(op))
        {
            const GemmOp& g = std::get<GemmOp>(op);
            set_if_unknown(find_or_add(g.lhs), 2, Layout::Row);
            set_if_unknown(find_or_add(g.rhs), 2, Layout::Col);
            set_if_unknown(find_or_add(g.out), 2, Layout::Row);
            out_name = g.out;
        }
        else if(std::holds_alternative<AddOp>(op))
        {
            const AddOp& a = std::get<AddOp>(op);
            find_or_add(a.lhs);
            find_or_add(a.rhs);
            find_or_add(a.out);
            out_name = a.out;
        }
        else if(std::holds_alternative<MulOp>(op))
        {
            const MulOp& m = std::get<MulOp>(op);
            find_or_add(m.lhs);
            find_or_add(m.rhs);
            find_or_add(m.out);
            out_name = m.out;
        }
        else if(std::holds_alternative<ReluOp>(op))
        {
            const ReluOp& r = std::get<ReluOp>(op);
            find_or_add(r.in);
            find_or_add(r.out);
            out_name = r.out;
        }
        else if(std::holds_alternative<FastGeluOp>(op))
        {
            const FastGeluOp& f = std::get<FastGeluOp>(op);
            find_or_add(f.in);
            find_or_add(f.out);
            out_name = f.out;
        }
        else if(std::holds_alternative<GeluOp>(op))
        {
            const GeluOp& g = std::get<GeluOp>(op);
            find_or_add(g.in);
            find_or_add(g.out);
            out_name = g.out;
        }
        else if(std::holds_alternative<SiluOp>(op))
        {
            const SiluOp& s = std::get<SiluOp>(op);
            find_or_add(s.in);
            find_or_add(s.out);
            out_name = s.out;
        }
        else if(std::holds_alternative<SigmoidOp>(op))
        {
            const SigmoidOp& s = std::get<SigmoidOp>(op);
            find_or_add(s.in);
            find_or_add(s.out);
            out_name = s.out;
        }
        else if(std::holds_alternative<SoftmaxOp>(op))
        {
            const SoftmaxOp& s = std::get<SoftmaxOp>(op);
            find_or_add(s.in);
            find_or_add(s.out);
            out_name = s.out;
        }
        else if(std::holds_alternative<ScaleOp>(op))
        {
            const ScaleOp& s = std::get<ScaleOp>(op);
            find_or_add(s.in);
            find_or_add(s.out);
            out_name = s.out;

            // Validate scalar reference
            if(s.scale.empty())
                throw "ScaleOp.scale must name a Scalar parameter";
            bool found_scalar = false;
            for(int si = 0; si < kMaxScalars; ++si)
            {
                if(sig.scalars[si].name == s.scale)
                {
                    found_scalar = true;
                    break;
                }
            }
            if(!found_scalar)
                throw "ScaleOp.scale references undeclared Scalar";
        }

        // SSA uniqueness: each output name may appear at most once
        if(!out_name.empty())
        {
            for(int j = 0; j < num_outputs; ++j)
                if(output_names[j] == out_name)
                    throw "SSA violation: tensor produced by multiple operators";
            output_names[num_outputs++] = out_name;
        }
    }

    // ================================================================
    // Phase 2: Propagate rank/layout through connected tensors
    // ================================================================

    // Propagate between binary op slots: first known -> all others.
    auto propagate_binary =
        [&](std::string_view lhs_name, std::string_view rhs_name, std::string_view out_name) {
            int li = find(lhs_name);
            int ri = find(rhs_name);
            int oi = find(out_name);
            if(li < 0 || ri < 0 || oi < 0)
                return;

            int src = -1;
            if(infos[li].rank != 0)
                src = li;
            else if(infos[ri].rank != 0)
                src = ri;
            else if(infos[oi].rank != 0)
                src = oi;

            if(src >= 0)
            {
                set_if_unknown(li, infos[src].rank, infos[src].layout);
                set_if_unknown(ri, infos[src].rank, infos[src].layout);
                set_if_unknown(oi, infos[src].rank, infos[src].layout);
            }
        };

    // Propagate between unary op slots: known -> unknown.
    auto propagate_unary = [&](std::string_view in_name, std::string_view out_name) {
        int ii = find(in_name);
        int oi = find(out_name);
        if(ii < 0 || oi < 0)
            return;

        if(infos[ii].rank != 0)
            set_if_unknown(oi, infos[ii].rank, infos[ii].layout);
        else if(infos[oi].rank != 0)
            set_if_unknown(ii, infos[oi].rank, infos[oi].layout);
    };

    auto propagate_op = [&](const Op& op) {
        if(std::holds_alternative<AddOp>(op))
        {
            const AddOp& a = std::get<AddOp>(op);
            propagate_binary(a.lhs, a.rhs, a.out);
        }
        else if(std::holds_alternative<MulOp>(op))
        {
            const MulOp& m = std::get<MulOp>(op);
            propagate_binary(m.lhs, m.rhs, m.out);
        }
        else if(std::holds_alternative<ReluOp>(op))
        {
            const ReluOp& r = std::get<ReluOp>(op);
            propagate_unary(r.in, r.out);
        }
        else if(std::holds_alternative<FastGeluOp>(op))
        {
            const FastGeluOp& f = std::get<FastGeluOp>(op);
            propagate_unary(f.in, f.out);
        }
        else if(std::holds_alternative<GeluOp>(op))
        {
            const GeluOp& g = std::get<GeluOp>(op);
            propagate_unary(g.in, g.out);
        }
        else if(std::holds_alternative<SiluOp>(op))
        {
            const SiluOp& s = std::get<SiluOp>(op);
            propagate_unary(s.in, s.out);
        }
        else if(std::holds_alternative<SigmoidOp>(op))
        {
            const SigmoidOp& s = std::get<SigmoidOp>(op);
            propagate_unary(s.in, s.out);
        }
        else if(std::holds_alternative<SoftmaxOp>(op))
        {
            const SoftmaxOp& s = std::get<SoftmaxOp>(op);
            propagate_unary(s.in, s.out);
        }
        else if(std::holds_alternative<ScaleOp>(op))
        {
            const ScaleOp& s = std::get<ScaleOp>(op);
            propagate_unary(s.in, s.out);
        }
        // GemmOp: already set in phase 1, no propagation needed
    };

    // Forward pass: propagate downstream effects
    for(int i = 0; i < kMaxOps; ++i)
        propagate_op(sig.ops[i]);

    // Backward pass: propagate upstream effects
    for(int i = kMaxOps - 1; i >= 0; --i)
        propagate_op(sig.ops[i]);

    // ================================================================
    // Phase 3: Merge explicit Tensor entries (override propagation)
    // ================================================================
    for(int i = 0; i < kMaxTensors; ++i)
    {
        if(sig.tensors[i].name.empty())
        {
            // Catch entries with metadata but no name (likely a mistake)
            if(sig.tensors[i].dtype.has_value() || sig.tensors[i].rank != 0 ||
               sig.tensors[i].layout != Layout::Auto)
                throw "Tensor entry has metadata but no name";
            continue;
        }
        int idx = find_or_add(sig.tensors[i].name);
        if(sig.tensors[i].dtype.has_value())
        {
            infos[idx].dtype_set = true;
            infos[idx].dtype     = *sig.tensors[i].dtype;
        }
        if(sig.tensors[i].rank != 0)
            infos[idx].rank = sig.tensors[i].rank;
        if(sig.tensors[i].layout != Layout::Auto)
            infos[idx].layout = sig.tensors[i].layout;
    }

    // ================================================================
    // Phase 4: dtype cascade
    // ================================================================
    for(int i = 0; i < num; ++i)
    {
        if(!infos[i].dtype_set)
        {
            if(sig.dtype.has_value())
                infos[i].dtype = *sig.dtype;
            else
                throw "tensor dtype unresolvable: set tensor dtype or signature dtype";
        }
    }

    // ================================================================
    // Phase 5: Build result
    // ================================================================
    ResolvedSignature result{};
    result.num_tensors = num;
    for(int i = 0; i < num; ++i)
    {
        result.tensors[i] =
            TensorDesc{infos[i].name, infos[i].dtype, infos[i].rank, infos[i].layout};
    }

    return result;
}

// ============================================================================
// resolve compile-time tests
// ============================================================================
// clang-format off

// --- Simple GemmOp: resolves to 3 tensors with operator defaults ---
static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).num_tensors == 3);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").dtype == DataType::FP16);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").rank == 2);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").layout == Layout::Row);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("B").layout == Layout::Col);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("C").layout == Layout::Row);

// --- GemmOp with custom tensor names ---
static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "X", .rhs = "Y", .out = "Z"}}}).tensor("X").rank == 2);

// --- dtype cascade: sig.dtype applies to all tensors ---
static_assert(resolve(Signature{
    .dtype = DataType::BF16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").dtype == DataType::BF16);

static_assert(resolve(Signature{
    .dtype = DataType::BF16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("C").dtype == DataType::BF16);

// --- Explicit tensor dtype overrides sig.dtype ---
static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("C").dtype == DataType::FP32);

static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .tensors = {Tensor{.name = "C", .dtype = DataType::FP32}},
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").dtype == DataType::FP16);

// --- Explicit tensor rank/layout overrides operator defaults ---
static_assert(resolve(Signature{
    .dtype = DataType::FP16,
    .tensors = {Tensor{.name = "A", .rank = 3}},
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").rank == 3);

// --- GEMM + Add + Relu: propagation test ---
constexpr ResolvedSignature gemm_add_relu_resolved = resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
            AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
            ReluOp{.in = "D", .out = "E"}}});

static_assert(gemm_add_relu_resolved.num_tensors == 6); // A, B, C, bias, D, E
static_assert(gemm_add_relu_resolved.tensor("C").rank == 2);
static_assert(gemm_add_relu_resolved.tensor("bias").rank == 2);        // propagated from C via AddOp
static_assert(gemm_add_relu_resolved.tensor("bias").layout == Layout::Row);
static_assert(gemm_add_relu_resolved.tensor("D").rank == 2);           // propagated
static_assert(gemm_add_relu_resolved.tensor("D").layout == Layout::Row);
static_assert(gemm_add_relu_resolved.tensor("E").rank == 2);           // propagated via ReluOp
static_assert(gemm_add_relu_resolved.tensor("E").layout == Layout::Row);

// --- Standalone AddOp: rank/layout unresolved (no op implies them) ---
static_assert(resolve(Signature{
    .dtype = DataType::FP32,
    .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}).num_tensors == 3);

static_assert(resolve(Signature{
    .dtype = DataType::FP32,
    .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").rank == 0);

static_assert(resolve(Signature{
    .dtype = DataType::FP32,
    .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}}).tensor("A").layout == Layout::Auto);

// --- FMHA pattern: two GemmOps + SoftmaxOp ---
constexpr ResolvedSignature fmha_resolved = resolve(Signature{
    .dtype = DataType::FP16,
    .ops = {GemmOp{.lhs = "Q", .rhs = "K", .out = "S"},
            SoftmaxOp{.in = "S", .out = "P"},
            GemmOp{.lhs = "P", .rhs = "V", .out = "O"}}});

static_assert(fmha_resolved.num_tensors == 6); // Q, K, S, P, V, O
static_assert(fmha_resolved.tensor("Q").rank == 2);
static_assert(fmha_resolved.tensor("S").rank == 2);
static_assert(fmha_resolved.tensor("P").rank == 2); // propagated via SoftmaxOp from S
static_assert(fmha_resolved.tensor("O").rank == 2);

// Error cases (uncommenting any would produce consteval compile errors):
//
// No dtype:
// resolve(Signature{.ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}})
//
// SSA violation (two ops output "C"):
// resolve(Signature{.dtype = DataType::FP16,
//         .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
//                 AddOp{.lhs = "X", .rhs = "Y", .out = "C"}}})

// clang-format on

} // namespace rocm_ck
