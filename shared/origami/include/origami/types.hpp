// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <compare>
#include <tuple>
#include <string>
#include <ostream>
#include <cmath>

namespace origami {

enum class grid_selection_t : std::uint8_t {
    number_of_cus        = 0,
    min_resources        = 1,
    energy_aware         = 2,
    reduction_cost_aware = 3,
    data_parallel        = 4,
    analytical           = 5,
    k_split_aware        = 6
};

enum class reduction_t {
    // BasicReduction,
    Tree,
    Parallel,
    // AtomicReduction,
    Count,
    None = Count
};

inline reduction_t int_to_reduction_t(int rt)
{
    return (reduction_t)rt;
}

// Forward declaration for data_type_t
enum class data_type_t : int;

/**
 * @brief A compact 3-D dimension triple (M, N, K).
 *
 * Provides convenient accessors for common GEMM tiling parameters
 * and helpers like mnk() for volume.
 */
struct dim3_t {
    /// M dimension (rows).
    std::size_t m;

    /// N dimension (columns).
    std::size_t n;

    /// K dimension (reduction).
    std::size_t k;

    constexpr bool operator==(const dim3_t& o) const noexcept {
        return m == o.m && n == o.n && k == o.k;
    }
    
    constexpr bool operator!=(const dim3_t& o) const noexcept {
        return !(*this == o);
    }

    /// @return Product m*n.
    constexpr std::size_t mn()  const noexcept { return m * n; }
    
    /// @return Product m*k.
    constexpr std::size_t mk()  const noexcept { return m * k; }
    
    /// @return Product n*k.
    constexpr std::size_t nk()  const noexcept { return n * k; }

    /// @return Product m*n*k.
    constexpr std::size_t mnk() const noexcept { return m * n * k; }
};


/**
 * @brief Full kernel configuration (tile shape + execution parameters).
 *
 * Holds the geometric tile sizes along with occupancy,
 * work-group mapping (WGM), and cache-control hints.
 */
struct config_t {
    /// Macro tile and matrix-instruction shape.
    dim3_t mt;
    dim3_t mi;
    
    /// Occupancy (number of waves resident per CU).
    std::size_t occupancy;

    /// Reorder workgroup id for L2 reuse.
    int workgroup_mapping{};

    /// Whether operand A is accessed with cache-flags.
    int cache_hints_a{};

    /// Whether operand B is accessed with cache-flags.
    int cache_hints_b{};

    /// Workspace size parameters.
    std::size_t workspace_size{};
    std::size_t workspace_size_per_elem_c{};

    /// Reduction strategy.
    reduction_t reduction_strategy{};

    /// Predicted latency.
    mutable double latency{};


};

/**
 * @brief Struct to define the problem.
 * 
 */
struct problem_t {
    /// Size of the problem: M, N, K.
    dim3_t size;

    /// Batch size.
    std::size_t batch;

    /// Transpose types (TT, TN, NT, TT.)
    bool transpose_a;
    bool transpose_b;

    /// Data types: A, B, C, D.
    data_type_t a_dtype;
    data_type_t b_dtype;
    data_type_t c_dtype;
    data_type_t d_dtype;

    /// Compute type.
    data_type_t mi_dtype;

    /// MX block size.
    std::size_t mx_block_size;
};

} // namespace origami