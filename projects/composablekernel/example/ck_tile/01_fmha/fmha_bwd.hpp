// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "mask.hpp"
#include "bias.hpp"

#include <type_traits>
#include <utility>
#include <variant>
#include <iostream>
#include <functional>

struct FmhaBwdFp32
{
};

struct FmhaBwdFp16
{
};

struct FmhaBwdBf16
{
};

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp32>
{
    using QDataType             = float;
    using KDataType             = float;
    using VDataType             = float;
    using GemmDataType          = float;
    using BiasDataType          = float;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = float;
    using OGradDataType         = float;
    using QGradDataType         = float;
    using KGradDataType         = float;
    using VGradDataType         = float;
    using BiasGradDataType      = float;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using GemmDataType          = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::half_t;
    using OGradDataType         = ck_tile::half_t;
    using QGradDataType         = ck_tile::half_t;
    using KGradDataType         = ck_tile::half_t;
    using VGradDataType         = ck_tile::half_t;
    using BiasGradDataType      = ck_tile::half_t;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using GemmDataType          = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::bf16_t;
    using OGradDataType         = ck_tile::bf16_t;
    using QGradDataType         = ck_tile::bf16_t;
    using KGradDataType         = ck_tile::bf16_t;
    using VGradDataType         = ck_tile::bf16_t;
    using BiasGradDataType      = ck_tile::bf16_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* dq_acc_ptr;
    const void*
        sink_ptr; // sink scores [batch, nhead] in log-space (LSEDataType); nullptr disables sink
    void* d_sink_ptr; // sink gradient output [nhead] (LSEDataType); nullptr disables sink gradient

    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //     lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //     sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //     exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t max_seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_do;
    ck_tile::index_t stride_dq_acc;
    ck_tile::index_t stride_dq;
    ck_tile::index_t stride_dk;
    ck_tile::index_t stride_dv;
    ck_tile::index_t stride_dbias;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::long_index_t nhead_stride_dq_acc;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_dbias;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_lsed;
    ck_tile::long_index_t batch_stride_dq_acc;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;
    ck_tile::index_t batch_stride_dbias;
    ck_tile::index_t split_stride_dq_acc;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

    // Legacy 58-arg constructor for backward compatibility (old API without sink fields)
    fmha_bwd_args(const void* q_ptr_,
                  const void* k_ptr_,
                  const void* v_ptr_,
                  const void* bias_ptr_,
                  const void* o_ptr_,
                  const void* lse_ptr_,
                  const void* do_ptr_,
                  void* d_ptr_,
                  void* rand_val_ptr_,
                  void* dq_ptr_,
                  void* dk_ptr_,
                  void* dv_ptr_,
                  void* dbias_ptr_,
                  void* dq_acc_ptr_,
                  const void* seqstart_q_ptr_,
                  const void* seqstart_k_ptr_,
                  const void* seqlen_q_ptr_,
                  const void* seqlen_k_ptr_,
                  const void* cu_seqlen_q_ptr_,
                  const void* cu_seqlen_k_ptr_,
                  ck_tile::index_t seqlen_q_,
                  ck_tile::index_t seqlen_k_,
                  ck_tile::index_t batch_,
                  ck_tile::index_t max_seqlen_q_,
                  ck_tile::index_t max_seqlen_k_,
                  ck_tile::index_t hdim_q_,
                  ck_tile::index_t hdim_v_,
                  ck_tile::index_t nhead_q_,
                  ck_tile::index_t nhead_k_,
                  float scale_,
                  ck_tile::index_t stride_q_,
                  ck_tile::index_t stride_k_,
                  ck_tile::index_t stride_v_,
                  ck_tile::index_t stride_bias_,
                  ck_tile::index_t stride_o_,
                  ck_tile::index_t stride_randval_,
                  ck_tile::index_t stride_do_,
                  ck_tile::index_t stride_dq_acc_,
                  ck_tile::index_t stride_dq_,
                  ck_tile::index_t stride_dk_,
                  ck_tile::index_t stride_dv_,
                  ck_tile::index_t stride_dbias_,
                  ck_tile::index_t nhead_stride_q_,
                  ck_tile::index_t nhead_stride_k_,
                  ck_tile::index_t nhead_stride_v_,
                  ck_tile::index_t nhead_stride_bias_,
                  ck_tile::index_t nhead_stride_o_,
                  ck_tile::index_t nhead_stride_randval_,
                  ck_tile::index_t nhead_stride_do_,
                  ck_tile::index_t nhead_stride_lsed_,
                  ck_tile::long_index_t nhead_stride_dq_acc_,
                  ck_tile::index_t nhead_stride_dq_,
                  ck_tile::index_t nhead_stride_dk_,
                  ck_tile::index_t nhead_stride_dv_,
                  ck_tile::index_t nhead_stride_dbias_,
                  ck_tile::index_t batch_stride_q_,
                  ck_tile::index_t batch_stride_k_,
                  ck_tile::index_t batch_stride_v_,
                  ck_tile::index_t batch_stride_bias_,
                  ck_tile::index_t batch_stride_o_,
                  ck_tile::index_t batch_stride_randval_,
                  ck_tile::index_t batch_stride_do_,
                  ck_tile::index_t batch_stride_lsed_,
                  ck_tile::long_index_t batch_stride_dq_acc_,
                  ck_tile::index_t batch_stride_dq_,
                  ck_tile::index_t batch_stride_dk_,
                  ck_tile::index_t batch_stride_dv_,
                  ck_tile::index_t batch_stride_dbias_,
                  ck_tile::index_t split_stride_dq_acc_,
                  ck_tile::index_t window_size_left_,
                  ck_tile::index_t window_size_right_,
                  ck_tile::index_t mask_type_,
                  float p_drop_,
                  float p_undrop_,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset_)
        : q_ptr(q_ptr_),
          k_ptr(k_ptr_),
          v_ptr(v_ptr_),
          bias_ptr(bias_ptr_),
          o_ptr(o_ptr_),
          lse_ptr(lse_ptr_),
          do_ptr(do_ptr_),
          d_ptr(d_ptr_),
          rand_val_ptr(rand_val_ptr_),
          dq_ptr(dq_ptr_),
          dk_ptr(dk_ptr_),
          dv_ptr(dv_ptr_),
          dbias_ptr(dbias_ptr_),
          dq_acc_ptr(dq_acc_ptr_),
          sink_ptr(nullptr),
          d_sink_ptr(nullptr),
          seqstart_q_ptr(seqstart_q_ptr_),
          seqstart_k_ptr(seqstart_k_ptr_),
          seqlen_q_ptr(seqlen_q_ptr_),
          seqlen_k_ptr(seqlen_k_ptr_),
          cu_seqlen_q_ptr(cu_seqlen_q_ptr_),
          cu_seqlen_k_ptr(cu_seqlen_k_ptr_),
          seqlen_q(seqlen_q_),
          seqlen_k(seqlen_k_),
          batch(batch_),
          max_seqlen_q(max_seqlen_q_),
          max_seqlen_k(max_seqlen_k_),
          hdim_q(hdim_q_),
          hdim_v(hdim_v_),
          nhead_q(nhead_q_),
          nhead_k(nhead_k_),
          scale(scale_),
          stride_q(stride_q_),
          stride_k(stride_k_),
          stride_v(stride_v_),
          stride_bias(stride_bias_),
          stride_o(stride_o_),
          stride_randval(stride_randval_),
          stride_do(stride_do_),
          stride_dq_acc(stride_dq_acc_),
          stride_dq(stride_dq_),
          stride_dk(stride_dk_),
          stride_dv(stride_dv_),
          stride_dbias(stride_dbias_),
          nhead_stride_q(nhead_stride_q_),
          nhead_stride_k(nhead_stride_k_),
          nhead_stride_v(nhead_stride_v_),
          nhead_stride_bias(nhead_stride_bias_),
          nhead_stride_o(nhead_stride_o_),
          nhead_stride_randval(nhead_stride_randval_),
          nhead_stride_do(nhead_stride_do_),
          nhead_stride_lsed(nhead_stride_lsed_),
          nhead_stride_dq_acc(nhead_stride_dq_acc_),
          nhead_stride_dq(nhead_stride_dq_),
          nhead_stride_dk(nhead_stride_dk_),
          nhead_stride_dv(nhead_stride_dv_),
          nhead_stride_dbias(nhead_stride_dbias_),
          batch_stride_q(batch_stride_q_),
          batch_stride_k(batch_stride_k_),
          batch_stride_v(batch_stride_v_),
          batch_stride_bias(batch_stride_bias_),
          batch_stride_o(batch_stride_o_),
          batch_stride_randval(batch_stride_randval_),
          batch_stride_do(batch_stride_do_),
          batch_stride_lsed(batch_stride_lsed_),
          batch_stride_dq_acc(batch_stride_dq_acc_),
          batch_stride_dq(batch_stride_dq_),
          batch_stride_dk(batch_stride_dk_),
          batch_stride_dv(batch_stride_dv_),
          batch_stride_dbias(batch_stride_dbias_),
          split_stride_dq_acc(split_stride_dq_acc_),
          window_size_left(window_size_left_),
          window_size_right(window_size_right_),
          mask_type(mask_type_),
          p_drop(p_drop_),
          p_undrop(p_undrop_),
          drop_seed_offset(drop_seed_offset_)
    {
    }

    // Full 60-arg constructor (new API with sink fields)
    fmha_bwd_args(const void* q_ptr_,
                  const void* k_ptr_,
                  const void* v_ptr_,
                  const void* bias_ptr_,
                  const void* o_ptr_,
                  const void* lse_ptr_,
                  const void* do_ptr_,
                  void* d_ptr_,
                  void* rand_val_ptr_,
                  void* dq_ptr_,
                  void* dk_ptr_,
                  void* dv_ptr_,
                  void* dbias_ptr_,
                  void* dq_acc_ptr_,
                  const void* sink_ptr_,
                  void* d_sink_ptr_,
                  const void* seqstart_q_ptr_,
                  const void* seqstart_k_ptr_,
                  const void* seqlen_q_ptr_,
                  const void* seqlen_k_ptr_,
                  const void* cu_seqlen_q_ptr_,
                  const void* cu_seqlen_k_ptr_,
                  ck_tile::index_t seqlen_q_,
                  ck_tile::index_t seqlen_k_,
                  ck_tile::index_t batch_,
                  ck_tile::index_t max_seqlen_q_,
                  ck_tile::index_t max_seqlen_k_,
                  ck_tile::index_t hdim_q_,
                  ck_tile::index_t hdim_v_,
                  ck_tile::index_t nhead_q_,
                  ck_tile::index_t nhead_k_,
                  float scale_,
                  ck_tile::index_t stride_q_,
                  ck_tile::index_t stride_k_,
                  ck_tile::index_t stride_v_,
                  ck_tile::index_t stride_bias_,
                  ck_tile::index_t stride_o_,
                  ck_tile::index_t stride_randval_,
                  ck_tile::index_t stride_do_,
                  ck_tile::index_t stride_dq_acc_,
                  ck_tile::index_t stride_dq_,
                  ck_tile::index_t stride_dk_,
                  ck_tile::index_t stride_dv_,
                  ck_tile::index_t stride_dbias_,
                  ck_tile::index_t nhead_stride_q_,
                  ck_tile::index_t nhead_stride_k_,
                  ck_tile::index_t nhead_stride_v_,
                  ck_tile::index_t nhead_stride_bias_,
                  ck_tile::index_t nhead_stride_o_,
                  ck_tile::index_t nhead_stride_randval_,
                  ck_tile::index_t nhead_stride_do_,
                  ck_tile::index_t nhead_stride_lsed_,
                  ck_tile::long_index_t nhead_stride_dq_acc_,
                  ck_tile::index_t nhead_stride_dq_,
                  ck_tile::index_t nhead_stride_dk_,
                  ck_tile::index_t nhead_stride_dv_,
                  ck_tile::index_t nhead_stride_dbias_,
                  ck_tile::index_t batch_stride_q_,
                  ck_tile::index_t batch_stride_k_,
                  ck_tile::index_t batch_stride_v_,
                  ck_tile::index_t batch_stride_bias_,
                  ck_tile::index_t batch_stride_o_,
                  ck_tile::index_t batch_stride_randval_,
                  ck_tile::index_t batch_stride_do_,
                  ck_tile::index_t batch_stride_lsed_,
                  ck_tile::long_index_t batch_stride_dq_acc_,
                  ck_tile::index_t batch_stride_dq_,
                  ck_tile::index_t batch_stride_dk_,
                  ck_tile::index_t batch_stride_dv_,
                  ck_tile::index_t batch_stride_dbias_,
                  ck_tile::index_t split_stride_dq_acc_,
                  ck_tile::index_t window_size_left_,
                  ck_tile::index_t window_size_right_,
                  ck_tile::index_t mask_type_,
                  float p_drop_,
                  float p_undrop_,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset_)
        : q_ptr(q_ptr_),
          k_ptr(k_ptr_),
          v_ptr(v_ptr_),
          bias_ptr(bias_ptr_),
          o_ptr(o_ptr_),
          lse_ptr(lse_ptr_),
          do_ptr(do_ptr_),
          d_ptr(d_ptr_),
          rand_val_ptr(rand_val_ptr_),
          dq_ptr(dq_ptr_),
          dk_ptr(dk_ptr_),
          dv_ptr(dv_ptr_),
          dbias_ptr(dbias_ptr_),
          dq_acc_ptr(dq_acc_ptr_),
          sink_ptr(sink_ptr_),
          d_sink_ptr(d_sink_ptr_),
          seqstart_q_ptr(seqstart_q_ptr_),
          seqstart_k_ptr(seqstart_k_ptr_),
          seqlen_q_ptr(seqlen_q_ptr_),
          seqlen_k_ptr(seqlen_k_ptr_),
          cu_seqlen_q_ptr(cu_seqlen_q_ptr_),
          cu_seqlen_k_ptr(cu_seqlen_k_ptr_),
          seqlen_q(seqlen_q_),
          seqlen_k(seqlen_k_),
          batch(batch_),
          max_seqlen_q(max_seqlen_q_),
          max_seqlen_k(max_seqlen_k_),
          hdim_q(hdim_q_),
          hdim_v(hdim_v_),
          nhead_q(nhead_q_),
          nhead_k(nhead_k_),
          scale(scale_),
          stride_q(stride_q_),
          stride_k(stride_k_),
          stride_v(stride_v_),
          stride_bias(stride_bias_),
          stride_o(stride_o_),
          stride_randval(stride_randval_),
          stride_do(stride_do_),
          stride_dq_acc(stride_dq_acc_),
          stride_dq(stride_dq_),
          stride_dk(stride_dk_),
          stride_dv(stride_dv_),
          stride_dbias(stride_dbias_),
          nhead_stride_q(nhead_stride_q_),
          nhead_stride_k(nhead_stride_k_),
          nhead_stride_v(nhead_stride_v_),
          nhead_stride_bias(nhead_stride_bias_),
          nhead_stride_o(nhead_stride_o_),
          nhead_stride_randval(nhead_stride_randval_),
          nhead_stride_do(nhead_stride_do_),
          nhead_stride_lsed(nhead_stride_lsed_),
          nhead_stride_dq_acc(nhead_stride_dq_acc_),
          nhead_stride_dq(nhead_stride_dq_),
          nhead_stride_dk(nhead_stride_dk_),
          nhead_stride_dv(nhead_stride_dv_),
          nhead_stride_dbias(nhead_stride_dbias_),
          batch_stride_q(batch_stride_q_),
          batch_stride_k(batch_stride_k_),
          batch_stride_v(batch_stride_v_),
          batch_stride_bias(batch_stride_bias_),
          batch_stride_o(batch_stride_o_),
          batch_stride_randval(batch_stride_randval_),
          batch_stride_do(batch_stride_do_),
          batch_stride_lsed(batch_stride_lsed_),
          batch_stride_dq_acc(batch_stride_dq_acc_),
          batch_stride_dq(batch_stride_dq_),
          batch_stride_dk(batch_stride_dk_),
          batch_stride_dv(batch_stride_dv_),
          batch_stride_dbias(batch_stride_dbias_),
          split_stride_dq_acc(split_stride_dq_acc_),
          window_size_left(window_size_left_),
          window_size_right(window_size_right_),
          mask_type(mask_type_),
          p_drop(p_drop_),
          p_undrop(p_undrop_),
          drop_seed_offset(drop_seed_offset_)
    {
    }

    // Default constructor
    fmha_bwd_args() = default;
};

template <typename FmhaBwdDQDKDVKernel>
auto fmha_bwd_dq_dk_dv_create_kargs_and_grids(fmha_bwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        constexpr bool dq_uss_acc  = FmhaBwdDQDKDVKernel::kMaxSeqLenQ == 0;
        const auto dq_ptr          = dq_uss_acc ? args.dq_acc_ptr : args.dq_ptr;
        const auto stride_dq       = dq_uss_acc ? args.stride_dq_acc : args.stride_dq;
        const auto nhead_stride_dq = dq_uss_acc ? args.nhead_stride_dq_acc : args.nhead_stride_dq;
        const auto batch_stride_dq = dq_uss_acc ? args.batch_stride_dq_acc : args.batch_stride_dq;

        // create group mode kernel arguments
        if constexpr(FmhaBwdDQDKDVKernel::kIsGroupMode)
        {
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      dq_ptr,
                                                      args.seqstart_q_ptr,
                                                      args.seqstart_k_ptr,
                                                      args.seqlen_q_ptr,
                                                      args.seqlen_k_ptr,
                                                      args.cu_seqlen_q_ptr,
                                                      args.cu_seqlen_k_ptr,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.split_stride_dq_acc,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      dq_ptr,
                                                      args.seqlen_q,
                                                      args.seqlen_k,
                                                      args.batch,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      stride_dq,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      nhead_stride_dq,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.batch_stride_q,
                                                      args.batch_stride_k,
                                                      args.batch_stride_v,
                                                      args.batch_stride_bias,
                                                      args.batch_stride_randval,
                                                      args.batch_stride_do,
                                                      args.batch_stride_lsed,
                                                      batch_stride_dq,
                                                      args.batch_stride_dk,
                                                      args.batch_stride_dv,
                                                      args.batch_stride_dbias,
                                                      args.split_stride_dq_acc,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
    }();

    dim3 grids = FmhaBwdDQDKDVKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_k);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqstart_q_ptr,
                                                     args.seqlen_q_ptr,
                                                     args.cu_seqlen_q_ptr,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.lse_ptr,
                                                     args.sink_ptr,
                                                     args.d_sink_ptr,
                                                     args.p_undrop,
                                                     args.seqlen_q,
                                                     args.hdim_v,
                                                     args.nhead_q,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed,
                                                     args.batch_stride_do,
                                                     args.batch_stride_o,
                                                     args.batch_stride_lsed);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdConvertQGradKernel>
auto fmha_bwd_convert_dq_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdConvertQGradKernel::kIsGroupMode)
        {
            return FmhaBwdConvertQGradKernel::MakeKargs(args.dq_acc_ptr,
                                                        args.dq_ptr,
                                                        args.seqstart_q_ptr,
                                                        args.seqstart_k_ptr,
                                                        args.seqlen_q_ptr,
                                                        args.seqlen_k_ptr,
                                                        args.cu_seqlen_q_ptr,
                                                        args.cu_seqlen_k_ptr,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.stride_dq_acc,
                                                        args.nhead_stride_dq,
                                                        args.nhead_stride_dq_acc,
                                                        args.split_stride_dq_acc);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdConvertQGradKernel::MakeKargs(args.dq_acc_ptr,
                                                        args.dq_ptr,
                                                        args.seqlen_q,
                                                        args.seqlen_k,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.stride_dq_acc,
                                                        args.nhead_stride_dq,
                                                        args.nhead_stride_dq_acc,
                                                        args.batch_stride_dq,
                                                        args.batch_stride_dq_acc,
                                                        args.split_stride_dq_acc,
                                                        args.batch,
                                                        args.nhead_q);
        }
    }();

    dim3 grids = FmhaBwdConvertQGradKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename FmhaDropout_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          ck_tile::index_t kPadD_,
          ck_tile::index_t kPadDv_,
          bool kIsDeterministic_,
          bool kUseTrLoad_,
          ck_tile::index_t MaxSeqLenQ_,
          ck_tile::index_t kN0>
struct fmha_bwd_dq_dk_dv_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dq_dk_dv_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dq_dk_dv_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dq_dk_dv_get_name_();
template <typename Traits_, typename Arch = void>
int fmha_bwd_dq_dk_dv_maxq_();
struct fmha_bwd_traits;
template <typename Traits_, typename Arch = void>
int fmha_bwd_dq_dk_dv_dq_acc_splits_(const fmha_bwd_traits& t);
template <typename Traits_, typename Arch = void>
bool fmha_bwd_dq_dk_dv_needs_zero_dq_acc_();

template <ck_tile::index_t HDim_, typename DataType_, bool kIsGroupMode_, bool kPadS_, bool kPadDv_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kPadS            = kPadS_;
    static constexpr bool kPadDv           = kPadDv_;
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_dot_do_o_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_dot_do_o_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_dot_do_o_get_name_();

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kPadS_,
          bool kPadD_,
          bool kIsDeterministic_,
          ck_tile::index_t kN0>
struct fmha_bwd_convert_dq_traits_
{
};

template <typename Traits_, typename Arch = void>
float fmha_bwd_convert_dq_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
void fmha_bwd_convert_dq_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_, typename Arch = void>
std::string fmha_bwd_convert_dq_get_name_();

// Traits that are used to dispatch different kernel implementations for fmha backward
struct fmha_bwd_traits
{
    int seqlen_q;
    int seqlen_k;
    int batch;
    int max_seqlen_q;
    int max_seqlen_k;
    int hdim_q;
    int hdim_v;
    int nhead_q;
    int nhead_k;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;
    // TODO: padding check is inside this api

    // Legacy constructor for backward compatibility (old 10-field API)
    fmha_bwd_traits(int hdim_q_,
                    int hdim_v_,
                    std::string data_type_,
                    bool is_group_mode_,
                    mask_enum mask_type_,
                    bias_enum bias_type_,
                    bool has_dbias_,
                    bool has_dropout_,
                    bool is_store_randval_,
                    bool is_deterministic_)
        : seqlen_q(0),
          seqlen_k(0),
          batch(0),
          max_seqlen_q(0),
          max_seqlen_k(0),
          hdim_q(hdim_q_),
          hdim_v(hdim_v_),
          nhead_q(0),
          nhead_k(0),
          data_type(std::move(data_type_)),
          is_group_mode(is_group_mode_),
          mask_type(mask_type_),
          bias_type(bias_type_),
          has_dbias(has_dbias_),
          has_dropout(has_dropout_),
          is_store_randval(is_store_randval_),
          is_deterministic(is_deterministic_)
    {
    }

    // Full constructor for new 17-field API (replaces aggregate initialization)
    fmha_bwd_traits(int seqlen_q_,
                    int seqlen_k_,
                    int batch_,
                    int max_seqlen_q_,
                    int max_seqlen_k_,
                    int hdim_q_,
                    int hdim_v_,
                    int nhead_q_,
                    int nhead_k_,
                    std::string data_type_,
                    bool is_group_mode_,
                    mask_enum mask_type_,
                    bias_enum bias_type_,
                    bool has_dbias_,
                    bool has_dropout_,
                    bool is_store_randval_,
                    bool is_deterministic_)
        : seqlen_q(seqlen_q_),
          seqlen_k(seqlen_k_),
          batch(batch_),
          max_seqlen_q(max_seqlen_q_),
          max_seqlen_k(max_seqlen_k_),
          hdim_q(hdim_q_),
          hdim_v(hdim_v_),
          nhead_q(nhead_q_),
          nhead_k(nhead_k_),
          data_type(std::move(data_type_)),
          is_group_mode(is_group_mode_),
          mask_type(mask_type_),
          bias_type(bias_type_),
          has_dbias(has_dbias_),
          has_dropout(has_dropout_),
          is_store_randval(is_store_randval_),
          is_deterministic(is_deterministic_)
    {
    }

    fmha_bwd_traits() = default;
};

template <typename T0 /*dot_do_o_trait*/,
          typename T1 /*dq_dk_dv_trait*/,
          typename T2 /*convert_dq_trait*/,
          typename Arch>
float fmha_bwd_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if constexpr(!std::is_same_v<T2, void>)
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_convert_dq_get_name_<T2, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) {
                fmha_bwd_convert_dq_oneshot_<T2, Arch>(s_, a);
            });
    }
    else
    {
        if(s.log_level_ > 0)
            std::cout << ", " << fmha_bwd_dot_do_o_get_name_<T0, Arch>() << "@"
                      << fmha_bwd_dq_dk_dv_get_name_<T1, Arch>() << std::flush;
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dot_do_o_oneshot_<T0, Arch>(s_, a); },
            [=](const ck_tile::stream_config& s_) { fmha_bwd_dq_dk_dv_oneshot_<T1, Arch>(s_, a); });
    }
}

template <int Version = 2>
float fmha_bwd(const fmha_bwd_traits&, fmha_bwd_args, const ck_tile::stream_config&);

struct fmha_bwd_launcher
{
    std::function<float(fmha_bwd_args, const ck_tile::stream_config&)> run{};
    ck_tile::index_t dq_acc_splits{0};
    bool needs_zero_dq_acc{true};

    fmha_bwd_launcher(const fmha_bwd_traits&);

    template <typename... Args>
    float operator()(Args&&... args) const
    {
        return run(std::forward<Args>(args)...);
    }
};
