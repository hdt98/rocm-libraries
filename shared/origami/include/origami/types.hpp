// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>

namespace origami {

/**
 * @brief Grid selection algorithms for StreamK.
 *
 * Different algorithms to select the grid size for kernel execution.
 */
enum class grid_selection_t : std::uint32_t {
  number_of_cus        = 0,  ///< Use number of compute units
  min_resources        = 1,  ///< Use minimum required resources
  energy_aware         = 2,  ///< Energy-aware selection
  reduction_cost_aware = 3,  ///< Reduction cost-aware selection
  data_parallel        = 4,  ///< Data parallel approach
  analytical           = 5,  ///< Analytical model-based selection
  k_split_aware        = 6,  ///< K-split aware selection
  count,                     ///< Count of Grid selection algos
  none = 0xFFFFFFFFu         ///< Explicitly invalid
};

/**
 * @brief Reduction strategy types for StreamK.
 *
 * Different algorithms for reduction operations in StreamK.
 */
enum class reduction_t : std::uint32_t {
  spinlock = 0,       ///< Spinlock-based reduction
  tree     = 1,       ///< Tree-based reduction
  parallel = 2,       ///< Parallel reduction
  atomic   = 3,       ///< Atomic Add-based reduction
  count,              ///< Count of reduction types
  none = 0xFFFFFFFFu  ///< Explicitly invalid / no reduction
};

/**
 * @brief Convert integer to reduction_t enum.
 *
 * @param rt Integer value to convert
 * @return reduction_t Corresponding reduction type
 */
inline reduction_t int_to_reduction_t(int rt) { return static_cast<reduction_t>(rt); }

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

  constexpr bool operator!=(const dim3_t& o) const noexcept { return !(*this == o); }

  /// @return Product m*n.
  constexpr std::size_t mn() const noexcept { return m * n; }

  /// @return Product m*k.
  constexpr std::size_t mk() const noexcept { return m * k; }

  /// @return Product n*k.
  constexpr std::size_t nk() const noexcept { return n * k; }

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
  mutable dim3_t mi;

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
 * @brief Struct to define the GEMM problem characteristics.
 *
 * Contains all the parameters needed to describe a GEMM operation,
 * including matrix dimensions, data types, and operation flags.
 */
struct problem_t {
  /// Size of the problem: M, N, K.
  dim3_t size;

  /// Batch size.
  std::size_t batch;

  /// Transpose types (TT, TN, NT, TT.)
  bool a_transpose;
  bool b_transpose;

  /// Data types: A, B, C, D.
  data_type_t a_dtype;
  data_type_t b_dtype;
  data_type_t c_dtype;
  data_type_t d_dtype;

  /// Compute type.
  data_type_t mi_dtype;

  /// MX block size.
  std::size_t a_mx_block_size;
  std::size_t b_mx_block_size;
};

}  // namespace origami