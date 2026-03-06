// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

/**
 * @brief Architecture-specific LDS bank configuration
 *
 * @tparam NumBanks_ Number of LDS banks for the target operation
 *
 * gfx950 has different bank configurations for reads (64 banks) and writes (32 banks).
 * Other architectures typically use 32 banks for both.
 */
template <index_t NumBanks_>
struct lds_bank_config
{
    static constexpr index_t NumBanks     = NumBanks_;
    static constexpr index_t BytesPerBank = 4; // 4 bytes per bank word

    /**
     * @brief Compute the bank index for a given byte offset
     *
     * @param byte_offset Byte offset into LDS
     * @return Bank index (0 to NumBanks-1)
     */
    CK_TILE_HOST_DEVICE static constexpr index_t compute_bank(index_t byte_offset)
    {
        return (byte_offset / BytesPerBank) % NumBanks;
    }
};

// Predefined bank configurations for gfx950
using lds_bank_config_gfx950_read  = lds_bank_config<64>;
using lds_bank_config_gfx950_write = lds_bank_config<32>;

// Default configuration for other architectures
using lds_bank_config_default = lds_bank_config<32>;

/**
 * @brief Result structure for LDS bank conflict analysis
 *
 * Contains statistics about bank conflicts for a given access pattern.
 *
 * @tparam NumThreads Number of threads in the analysis
 * @tparam NumBanks Number of LDS banks
 */
template <index_t NumThreads, index_t NumBanks>
struct lds_bank_conflict_result
{
    index_t total_conflicts;                    // Sum of (threads_per_bank - 1) across all banks
    index_t max_threads_per_bank;               // Worst-case conflict (max threads accessing same bank)
    array<index_t, NumBanks> threads_per_bank;  // Histogram: threads per bank

    /**
     * @brief Check if any bank conflicts exist
     */
    CK_TILE_HOST_DEVICE constexpr bool has_conflicts() const { return total_conflicts > 0; }

    /**
     * @brief Get the conflict ratio (0.0 = no conflicts, higher = worse)
     *
     * A ratio of 0 means each thread accesses a unique bank.
     * A ratio of (NumThreads - NumBanks) / NumBanks would be the worst case
     * when all threads access the same bank.
     */
    CK_TILE_HOST_DEVICE constexpr float conflict_ratio() const
    {
        return static_cast<float>(total_conflicts) / static_cast<float>(NumThreads);
    }
};

namespace detail {

/**
 * @brief Extract thread coordinates from distribution encoding
 *
 * This function decomposes a thread ID into coordinates based on the
 * distribution encoding's P-to-RH mappings.
 *
 * @tparam DstrEncode Distribution encoding type
 * @tparam ThreadId Compile-time thread ID
 * @return Multi-index with (M, N) coordinates for the first element (Y=0)
 */
template <typename DstrEncode, index_t ThreadId>
CK_TILE_HOST_DEVICE constexpr auto get_thread_coordinates_impl()
{
    constexpr index_t NDimX = DstrEncode::NDimX;
    constexpr index_t NDimP = DstrEncode::NDimP;

    // Initialize coordinates to zero
    array<index_t, NDimX> coords{0};

    // Extract the RH dimension lengths from the encoding
    constexpr auto rhs_lengthss = DstrEncode::detail::rhs_lengthss_;

    // Process each P dimension to extract thread contribution to coordinates
    // ThreadId encodes multiple P dimensions; we need to extract each one's contribution

    // For standard CK-tile distributions:
    // - P0 typically encodes warp_id and parts of lane_id
    // - The encoding maps P indices to RH (R + H) dimensions

    // Decompose ThreadId based on P dimension structure
    index_t remaining_tid = ThreadId;

    // Calculate total P dimension size from encoding
    constexpr auto compute_p_lengths = []() {
        array<index_t, NDimP> p_lengths{1};
        static_for<0, NDimP, 1>{}([&](auto idim_p) {
            constexpr index_t ndim_low = DstrEncode::ps_to_rhss_major_[idim_p].size();
            index_t p_len              = 1;
            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                constexpr index_t rh_major = DstrEncode::ps_to_rhss_major_[idim_p][idim_low];
                constexpr index_t rh_minor = DstrEncode::ps_to_rhss_minor_[idim_p][idim_low];
                p_len *= rhs_lengthss[rh_major][rh_minor];
            });
            p_lengths[idim_p] = p_len;
        });
        return p_lengths;
    };
    constexpr auto p_lengths = compute_p_lengths();

    // Extract P indices from ThreadId (P0 is fastest varying in standard encoding)
    array<index_t, NDimP> p_indices{0};
    index_t tid_work = remaining_tid;

    // Decompose ThreadId into P indices (assuming P0 varies fastest)
    static_for<0, NDimP, 1>{}([&](auto idim_p) {
        p_indices[idim_p] = tid_work % p_lengths[idim_p];
        tid_work /= p_lengths[idim_p];
    });

    // Now map P indices back to X (spatial) coordinates
    // Each P dimension contributes to H dimensions which map to X coordinates
    static_for<0, NDimP, 1>{}([&](auto idim_p) {
        constexpr index_t ndim_low = DstrEncode::ps_to_rhss_major_[idim_p].size();
        index_t p_idx              = p_indices[idim_p];

        // Decompose p_idx along the RH dimensions it spans
        static_for<ndim_low - 1, -1, -1>{}([&](auto idim_low) {
            constexpr index_t rh_major = DstrEncode::ps_to_rhss_major_[idim_p][idim_low];
            constexpr index_t rh_minor = DstrEncode::ps_to_rhss_minor_[idim_p][idim_low];
            constexpr index_t rh_len   = rhs_lengthss[rh_major][rh_minor];

            if constexpr(rh_major > 0)
            {
                // This is an H dimension (X coordinate contribution)
                constexpr index_t x_dim = rh_major - 1; // H dimensions are 1-indexed

                // Calculate the stride for this H dimension within the X coordinate
                constexpr auto compute_h_stride = [](index_t dim_x, index_t h_minor) {
                    index_t stride = 1;
                    for(index_t h = 0; h < h_minor; ++h)
                    {
                        stride *= DstrEncode::hs_lengthss_[number<dim_x>{}][h];
                    }
                    return stride;
                };

                index_t h_idx   = p_idx % rh_len;
                index_t stride  = compute_h_stride(x_dim, rh_minor);
                coords[x_dim]   = coords[x_dim] + h_idx * stride;
            }

            p_idx /= rh_len;
        });
    });

    return coords;
}

} // namespace detail

/**
 * @brief Compute LDS bank conflicts for a simple linear access pattern
 *
 * This simplified version analyzes conflicts when threads access elements
 * with a known stride pattern, without requiring full distribution/descriptor
 * infrastructure.
 *
 * @tparam NumThreads Number of threads accessing LDS simultaneously
 * @tparam NumBanks Number of LDS banks
 * @tparam ByteStride Byte stride between successive thread accesses
 * @tparam ByteOffset Starting byte offset for thread 0
 * @return Bank conflict analysis result
 */
template <index_t NumThreads,
          index_t NumBanks,
          index_t ByteStride,
          index_t ByteOffset = 0>
CK_TILE_HOST_DEVICE constexpr auto compute_lds_bank_conflicts_linear()
{
    using Config = lds_bank_config<NumBanks>;

    lds_bank_conflict_result<NumThreads, NumBanks> result{};
    result.total_conflicts      = 0;
    result.max_threads_per_bank = 0;

    // Initialize histogram to zero
    static_for<0, NumBanks, 1>{}([&](auto bank) { result.threads_per_bank[bank] = 0; });

    // Count threads per bank
    static_for<0, NumThreads, 1>{}([&](auto tid) {
        constexpr index_t byte_off = ByteOffset + tid * ByteStride;
        constexpr index_t bank     = Config::compute_bank(byte_off);
        result.threads_per_bank[bank]++;
    });

    // Compute statistics
    static_for<0, NumBanks, 1>{}([&](auto bank) {
        index_t count = result.threads_per_bank[bank];
        if(count > 1)
        {
            result.total_conflicts += (count - 1);
        }
        result.max_threads_per_bank = max(result.max_threads_per_bank, count);
    });

    return result;
}

/**
 * @brief Compute LDS bank conflicts for an array of per-thread byte offsets
 *
 * This version accepts an array of offsets, one per thread, allowing analysis
 * of arbitrary access patterns.
 *
 * @tparam NumThreads Number of threads
 * @tparam NumBanks Number of LDS banks
 * @param byte_offsets Array of byte offsets, one per thread
 * @return Bank conflict analysis result
 */
template <index_t NumThreads, index_t NumBanks>
CK_TILE_HOST_DEVICE constexpr auto
compute_lds_bank_conflicts_from_offsets(const array<index_t, NumThreads>& byte_offsets)
{
    using Config = lds_bank_config<NumBanks>;

    lds_bank_conflict_result<NumThreads, NumBanks> result{};
    result.total_conflicts      = 0;
    result.max_threads_per_bank = 0;

    // Initialize histogram to zero
    for(index_t bank = 0; bank < NumBanks; ++bank)
    {
        result.threads_per_bank[bank] = 0;
    }

    // Count threads per bank
    for(index_t tid = 0; tid < NumThreads; ++tid)
    {
        index_t bank = Config::compute_bank(byte_offsets[tid]);
        result.threads_per_bank[bank]++;
    }

    // Compute statistics
    for(index_t bank = 0; bank < NumBanks; ++bank)
    {
        index_t count = result.threads_per_bank[bank];
        if(count > 1)
        {
            result.total_conflicts += (count - 1);
        }
        result.max_threads_per_bank = max(result.max_threads_per_bank, count);
    }

    return result;
}

/**
 * @brief Analyze bank conflicts for a warp accessing LDS with XOR swizzling
 *
 * This function models the XOR swizzling pattern used in CShuffle epilogue
 * to verify that the swizzle eliminates bank conflicts.
 *
 * @tparam WarpSize Number of threads in warp (64 for gfx950)
 * @tparam NumBanks Number of LDS banks
 * @tparam VectorLen Number of elements per thread access (e.g., 8 for FP16x8)
 * @tparam RowStride Stride between rows in elements
 * @tparam XorMask XOR mask applied: new_col = col ^ (row & XorMask)
 * @tparam DataSize Bytes per element
 * @return Bank conflict analysis result
 */
template <index_t WarpSize,
          index_t NumBanks,
          index_t VectorLen,
          index_t RowStride,
          index_t XorMask,
          index_t DataSize>
CK_TILE_HOST_DEVICE constexpr auto compute_lds_bank_conflicts_xor_swizzle()
{
    using Config = lds_bank_config<NumBanks>;

    lds_bank_conflict_result<WarpSize, NumBanks> result{};
    result.total_conflicts      = 0;
    result.max_threads_per_bank = 0;

    // Initialize histogram
    static_for<0, NumBanks, 1>{}([&](auto bank) { result.threads_per_bank[bank] = 0; });

    // Analyze access pattern with XOR swizzle
    // Each thread accesses VectorLen consecutive elements
    // We check the first element of each thread's vector
    static_for<0, WarpSize, 1>{}([&](auto tid) {
        // Compute logical (row, col) from thread ID
        // Assuming standard MFMA-like layout where tid maps to (m, n)
        constexpr index_t row = tid / (RowStride / VectorLen);
        constexpr index_t col = (tid % (RowStride / VectorLen)) * VectorLen;

        // Apply XOR swizzle: new_col = col ^ (row & XorMask)
        constexpr index_t swizzled_col = col ^ (row & XorMask);

        // Compute byte offset
        constexpr index_t byte_offset = (row * RowStride + swizzled_col) * DataSize;

        // Determine bank
        constexpr index_t bank = Config::compute_bank(byte_offset);
        result.threads_per_bank[bank]++;
    });

    // Compute statistics
    static_for<0, NumBanks, 1>{}([&](auto bank) {
        index_t count = result.threads_per_bank[bank];
        if(count > 1)
        {
            result.total_conflicts += (count - 1);
        }
        result.max_threads_per_bank = max(result.max_threads_per_bank, count);
    });

    return result;
}

/**
 * @brief Print bank conflict analysis result (host-only debug utility)
 */
template <index_t NumThreads, index_t NumBanks>
CK_TILE_HOST void print(const lds_bank_conflict_result<NumThreads, NumBanks>& result)
{
    printf("lds_bank_conflict_result{\n");
    printf("  NumThreads: %d, NumBanks: %d\n",
           static_cast<int>(NumThreads),
           static_cast<int>(NumBanks));
    printf("  total_conflicts: %d\n", static_cast<int>(result.total_conflicts));
    printf("  max_threads_per_bank: %d\n", static_cast<int>(result.max_threads_per_bank));
    printf("  has_conflicts: %s\n", result.has_conflicts() ? "true" : "false");
    printf("  conflict_ratio: %.3f\n", result.conflict_ratio());

    // Print non-zero bank counts
    printf("  active_banks: [");
    bool first = true;
    for(index_t b = 0; b < NumBanks; ++b)
    {
        if(result.threads_per_bank[b] > 0)
        {
            if(!first)
                printf(", ");
            printf("b%d:%d", static_cast<int>(b), static_cast<int>(result.threads_per_bank[b]));
            first = false;
        }
    }
    printf("]\n}\n");
}

} // namespace ck_tile
