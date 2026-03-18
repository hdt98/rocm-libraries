// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

/// @brief GF(2) matrix operating on dimension indices
///
/// A square matrix over GF(2) where each element indicates whether to XOR
/// the corresponding input dimension into the output dimension.
///
/// For matrix M and input index I, output O is computed as:
///   O[i] = XOR_{j where M[i][j]=true} I[j]
///
/// This operates directly on CK's multi_index types without bit packing.
///
/// @tparam NDim Number of dimensions (matrix is NDim x NDim)
template <index_t NDim_>
struct gf2_matrix
{
    static constexpr index_t NDim = NDim_;

    // matrix[i][j] = true means: out[i] ^= in[j]
    array<array<bool, NDim>, NDim> data;

    /// @brief Default constructor - zero matrix
    CK_TILE_HOST_DEVICE constexpr gf2_matrix() : data{} {}

    /// @brief Construct from 2D array
    CK_TILE_HOST_DEVICE constexpr gf2_matrix(const array<array<bool, NDim>, NDim>& m) : data{m} {}

    /// @brief Get element at (row, col)
    CK_TILE_HOST_DEVICE constexpr bool get(index_t row, index_t col) const
    {
        return data[row][col];
    }

    /// @brief Set element at (row, col)
    CK_TILE_HOST_DEVICE constexpr void set(index_t row, index_t col, bool value)
    {
        data[row][col] = value;
    }

    /// @brief Apply transform to multi_index
    template <typename IdxOut, typename IdxIn>
    CK_TILE_HOST_DEVICE constexpr void apply(IdxOut& out, const IdxIn& in) const
    {
        static_for<0, NDim, 1>{}([&](auto i) {
            index_t val = 0;
            static_for<0, NDim, 1>{}([&](auto j) {
                if(data[i.value][j.value])
                {
                    val ^= in[j];
                }
            });
            out(i) = val;
        });
    }

    /// @brief Apply transform, returning new multi_index
    template <typename IdxIn>
    CK_TILE_HOST_DEVICE constexpr auto apply(const IdxIn& in) const
    {
        multi_index<NDim> out;
        apply(out, in);
        return out;
    }

    /// @brief Check equality
    CK_TILE_HOST_DEVICE constexpr bool operator==(const gf2_matrix& other) const
    {
        for(index_t i = 0; i < NDim; ++i)
        {
            for(index_t j = 0; j < NDim; ++j)
            {
                if(data[i][j] != other.data[i][j])
                    return false;
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE constexpr bool operator!=(const gf2_matrix& other) const
    {
        return !(*this == other);
    }
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Create identity matrix
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr auto make_gf2_identity()
{
    gf2_matrix<NDim> result;
    for(index_t i = 0; i < NDim; ++i)
    {
        result.data[i][i] = true;
    }
    return result;
}

/// @brief Create XOR swizzle for 2D coordinates (row, col)
///
/// Implements: row' = row, col' = col ^ row
/// This matches CK's xor_t behavior when row < col_width.
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle_2d()
{
    gf2_matrix<2> result;
    // row' = row
    result.data[0][0] = true;
    result.data[0][1] = false;
    // col' = row ^ col
    result.data[1][0] = true;
    result.data[1][1] = true;
    return result;
}

/// @brief Create XOR swizzle that XORs dimension `src` into dimension `dst`
///
/// out[dst] = in[dst] ^ in[src]
/// All other dimensions pass through unchanged.
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle(index_t src, index_t dst)
{
    auto result = make_gf2_identity<NDim>();
    result.data[dst][src] = true;
    return result;
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// @brief Compose two GF(2) matrices: C = A @ B
///
/// C[i][k] = XOR_{j} (A[i][j] AND B[j][k])
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr auto gf2_compose(const gf2_matrix<NDim>& a,
                                                const gf2_matrix<NDim>& b)
{
    gf2_matrix<NDim> result;
    for(index_t i = 0; i < NDim; ++i)
    {
        for(index_t k = 0; k < NDim; ++k)
        {
            bool val = false;
            for(index_t j = 0; j < NDim; ++j)
            {
                val ^= (a.data[i][j] && b.data[j][k]);
            }
            result.data[i][k] = val;
        }
    }
    return result;
}

/// @brief Compute inverse using Gaussian elimination over GF(2)
///
/// @param m Input matrix
/// @param success Set to true if matrix is invertible
/// @return Inverse matrix (undefined if not invertible)
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<NDim>& m, bool& success)
{
    // Augmented matrix [M | I]
    array<array<bool, 2 * NDim>, NDim> aug{};

    // Initialize: left half = M, right half = I
    for(index_t i = 0; i < NDim; ++i)
    {
        for(index_t j = 0; j < NDim; ++j)
        {
            aug[i][j] = m.data[i][j];
        }
        aug[i][NDim + i] = true; // Identity on right
    }

    // Gauss-Jordan elimination
    for(index_t col = 0; col < NDim; ++col)
    {
        // Find pivot
        index_t pivot = col;
        while(pivot < NDim && !aug[pivot][col])
        {
            ++pivot;
        }

        if(pivot == NDim)
        {
            success = false;
            return gf2_matrix<NDim>{};
        }

        // Swap rows
        if(pivot != col)
        {
            for(index_t k = 0; k < 2 * NDim; ++k)
            {
                bool tmp       = aug[col][k];
                aug[col][k]    = aug[pivot][k];
                aug[pivot][k]  = tmp;
            }
        }

        // Eliminate column in all other rows
        for(index_t row = 0; row < NDim; ++row)
        {
            if(row != col && aug[row][col])
            {
                for(index_t k = 0; k < 2 * NDim; ++k)
                {
                    aug[row][k] = aug[row][k] != aug[col][k]; // XOR
                }
            }
        }
    }

    // Extract right half
    gf2_matrix<NDim> inv;
    for(index_t i = 0; i < NDim; ++i)
    {
        for(index_t j = 0; j < NDim; ++j)
        {
            inv.data[i][j] = aug[i][NDim + j];
        }
    }

    success = true;
    return inv;
}

/// @brief Compute inverse (simpler interface)
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<NDim>& m)
{
    bool success;
    return gf2_inverse(m, success);
}

/// @brief Check if matrix is invertible
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_invertible(const gf2_matrix<NDim>& m)
{
    bool success;
    gf2_inverse(m, success);
    return success;
}

/// @brief Check if matrix is self-inverse (M @ M = I)
template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_self_inverse(const gf2_matrix<NDim>& m)
{
    return gf2_compose(m, m) == make_gf2_identity<NDim>();
}

} // namespace ck_tile
