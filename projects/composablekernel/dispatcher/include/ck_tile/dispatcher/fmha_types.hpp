// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// FMHA type definitions for the dispatcher.
//
// Strategy: if the upstream example headers are available, include fmha_fwd.hpp
// as the single source of truth for forward types. Backward types are always
// provided here (fmha_bwd.hpp cannot be co-included with fmha_fwd.hpp due to
// a FmhaMasks redefinition in the upstream code).
//
// When building standalone (without the example tree), all types are provided
// as fallback definitions identical to the upstream.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

// --- Detect example headers ---
#if __has_include("example/ck_tile/01_fmha/fmha_fwd.hpp")
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#define CK_TILE_FMHA_TYPES_FROM_EXAMPLE 1
#endif

// =========================================================================
// Fallback definitions: only compiled when example headers are NOT available
// =========================================================================
#ifndef CK_TILE_FMHA_TYPES_FROM_EXAMPLE

enum class mask_enum
{
    no_mask = 0,
    mask_top_left,
    mask_bottom_right,
    window_generic,
};

enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

enum class quant_scale_enum
{
    no_scale      = 0,
    pertensor     = 1,
    blockscale    = 2,
    kv_blockscale = 3,
};

enum class rope_enum
{
    none         = 0,
    interleaved  = 1,
    half_rotated = 2,
};

struct fmha_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    const void* seqstart_q_ptr  = nullptr;
    const void* seqstart_k_ptr  = nullptr;
    const void* seqlen_q_ptr    = nullptr;
    const void* seqlen_k_ptr    = nullptr;
    const void* cu_seqlen_q_ptr = nullptr;
    const void* cu_seqlen_k_ptr = nullptr;
    const void* block_scale_seqstart_q_ptr;
    const void* block_scale_seqstart_k_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_q_descale;
    ck_tile::index_t nhead_stride_k_descale;
    ck_tile::index_t nhead_stride_v_descale;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_q_descale;
    ck_tile::index_t batch_stride_k_descale;
    ck_tile::index_t batch_stride_v_descale;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

    ck_tile::index_t block_scale_size_q;
    ck_tile::index_t block_scale_size_kv;
};

struct fmha_fwd_pagedkv_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* lse_ptr;
    void* o_ptr;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table;
    ck_tile::index_t page_block_size;
    bool is_gappy;

    const void* cache_batch_idx;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;
};

struct fmha_fwd_splitkv_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    void* lse_acc_ptr;
    void* o_acc_ptr;
    void* lse_ptr;
    void* o_ptr;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table;
    ck_tile::index_t page_block_size;
    bool is_gappy;

    const void* cache_batch_idx;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    ck_tile::index_t num_splits;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias;
    ck_tile::index_t stride_o_acc;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_lse_acc;
    ck_tile::index_t nhead_stride_o_acc;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_lse_acc;
    ck_tile::index_t batch_stride_o_acc;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t split_stride_lse_acc;
    ck_tile::index_t split_stride_o_acc;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;
};

struct fmha_fwd_appendkv_args
{
    void* q_ptr;
    void* k_ptr;
    const void* knew_ptr;
    void* v_ptr;
    const void* vnew_ptr;

    const void* seqlen_k_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_knew;
    ck_tile::index_t batch;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    const void* rotary_cos_ptr;
    const void* rotary_sin_ptr;
    ck_tile::index_t rotary_dim;
    bool has_mask;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table;
    ck_tile::index_t page_block_size;

    const void* cache_batch_idx;
    const void* sink_ptr;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_knew;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_vnew;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_knew;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_vnew;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_knew;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_vnew;
};

struct fmha_batch_prefill_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    const void* seqstart_q_ptr;
    const void* sink_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    int32_t num_total_pages;
    ck_tile::index_t page_block_size;
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout;
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table;
    void* kv_indptr;
    void* kv_page_indices;
    void* kv_last_page_lens;
    void* seqlen_k_ptr;
    ck_tile::index_t batch_stride_block_table;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t mask_type;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

    ck_tile::index_t nblock_stride_kv_block_descale = 0;
    ck_tile::index_t nhead_stride_kv_block_descale  = 0;
};

struct fmha_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_lse;
    bool has_dropout;
    quant_scale_enum qscale_type;
    bool skip_min_seqlen_q = false;
    bool has_sink          = false;
};

struct fmha_fwd_pagedkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_lse             = false;
    bool use_pagedkv         = true;
    bool do_fp8_static_quant = false;
    bool skip_min_seqlen_q   = false;
    bool has_sink            = false;
};

struct fmha_fwd_splitkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_lse;
    bool do_fp8_static_quant = false;
    bool has_sink            = false;
};

struct fmha_fwd_appendkv_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_v_rowmajor;
    rope_enum rope_type;
};

struct fmha_batch_prefill_traits : public fmha_fwd_traits
{
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout =
        ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table =
        ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    int page_size = 1;
};

#endif // CK_TILE_FMHA_TYPES_FROM_EXAMPLE

// =========================================================================
// Backward types: always provided here.
// fmha_bwd.hpp is NOT included via __has_include because it redefines
// FmhaMasks (also in fmha_fwd.hpp). These definitions are identical to
// the upstream and are harmless when fmha_bwd.hpp is not in the TU.
// In bwd kernel TUs (which include fmha_bwd.hpp directly), these types
// would conflict -- but bwd kernel TUs never include fmha_types.hpp.
// =========================================================================

struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
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

    const void* seqstart_q_ptr  = nullptr;
    const void* seqstart_k_ptr  = nullptr;
    const void* seqlen_q_ptr    = nullptr;
    const void* seqlen_k_ptr    = nullptr;
    const void* cu_seqlen_q_ptr = nullptr;
    const void* cu_seqlen_k_ptr = nullptr;

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
    ck_tile::index_t stride_bias;
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
};

struct fmha_bwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type;
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;
};
