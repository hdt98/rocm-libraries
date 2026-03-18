// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

// =============================================================================
// Part 1: Dimension-Level GF(2) Matrix
// =============================================================================
// Simple XOR of entire dimension values. Sufficient for basic swizzle patterns.

/// @brief GF(2) matrix operating on dimension indices
///
/// Each element indicates whether to XOR the corresponding input dimension
/// into the output dimension: out[i] = XOR_{j where M[i][j]=true} in[j]
///
/// @tparam NDim Number of dimensions (matrix is NDim x NDim)
template <index_t NDim_>
struct gf2_matrix
{
    static constexpr index_t NDim = NDim_;

    // matrix[i][j] = true means: out[i] ^= in[j]
    array<array<bool, NDim>, NDim> data;

    CK_TILE_HOST_DEVICE constexpr gf2_matrix() : data{} {}

    CK_TILE_HOST_DEVICE constexpr gf2_matrix(const array<array<bool, NDim>, NDim>& m) : data{m} {}

    CK_TILE_HOST_DEVICE constexpr bool get(index_t row, index_t col) const
    {
        return data[row][col];
    }

    CK_TILE_HOST_DEVICE constexpr void set(index_t row, index_t col, bool value)
    {
        data[row][col] = value;
    }

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

    template <typename IdxIn>
    CK_TILE_HOST_DEVICE constexpr auto apply(const IdxIn& in) const
    {
        multi_index<NDim> out;
        apply(out, in);
        return out;
    }

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

// Dimension-level factory functions

template <index_t NDim>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_gf2_identity()
{
    gf2_matrix<NDim> result;
    for(index_t i = 0; i < NDim; ++i)
    {
        result.data[i][i] = true;
    }
    return result;
}

/// @brief Create XOR swizzle for 2D coordinates: row' = row, col' = col ^ row
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle_2d()
{
    gf2_matrix<2> result;
    result.data[0][0] = true;  // row' = row
    result.data[1][0] = true;  // col' = row ^ col
    result.data[1][1] = true;
    return result;
}

template <index_t NDim>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle(index_t src, index_t dst)
{
    auto result = make_gf2_identity<NDim>();
    result.data[dst][src] = true;
    return result;
}

// Dimension-level matrix operations

template <index_t NDim>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_compose(const gf2_matrix<NDim>& a,
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

template <index_t NDim>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<NDim>& m, bool& success)
{
    array<array<bool, 2 * NDim>, NDim> aug{};

    for(index_t i = 0; i < NDim; ++i)
    {
        for(index_t j = 0; j < NDim; ++j)
        {
            aug[i][j] = m.data[i][j];
        }
        aug[i][NDim + i] = true;
    }

    for(index_t col = 0; col < NDim; ++col)
    {
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

        if(pivot != col)
        {
            for(index_t k = 0; k < 2 * NDim; ++k)
            {
                bool tmp      = aug[col][k];
                aug[col][k]   = aug[pivot][k];
                aug[pivot][k] = tmp;
            }
        }

        for(index_t row = 0; row < NDim; ++row)
        {
            if(row != col && aug[row][col])
            {
                for(index_t k = 0; k < 2 * NDim; ++k)
                {
                    aug[row][k] = aug[row][k] != aug[col][k];
                }
            }
        }
    }

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

template <index_t NDim>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<NDim>& m)
{
    bool success;
    return gf2_inverse(m, success);
}

template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_invertible(const gf2_matrix<NDim>& m)
{
    bool success;
    gf2_inverse(m, success);
    return success;
}

template <index_t NDim>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_self_inverse(const gf2_matrix<NDim>& m)
{
    return gf2_compose(m, m) == make_gf2_identity<NDim>();
}

// =============================================================================
// Part 2: Bit-Level GF(2) Matrix
// =============================================================================
// Full bit-level control for advanced swizzle patterns.

/// @brief GF(2) matrix operating on individual bits
///
/// Each row is a bitmask indicating which input bits to XOR together
/// to produce the corresponding output bit.
///
/// @tparam Bits Number of bits (matrix is Bits x Bits for square transforms)
template <index_t Bits_>
struct gf2_bit_matrix
{
    static_assert(Bits_ > 0 && Bits_ <= 64, "Bits must be in range [1, 64]");

    static constexpr index_t Bits = Bits_;

    // rows[i] is a bitmask: output bit i = XOR of input bits where mask is set
    array<uint64_t, Bits> rows;

    CK_TILE_HOST_DEVICE constexpr gf2_bit_matrix() : rows{} {}

    CK_TILE_HOST_DEVICE constexpr gf2_bit_matrix(const array<uint64_t, Bits>& r) : rows{r} {}

    CK_TILE_HOST_DEVICE constexpr bool get(index_t row, index_t col) const
    {
        return (rows[row] >> col) & 1;
    }

    CK_TILE_HOST_DEVICE constexpr void set(index_t row, index_t col, bool value)
    {
        if(value)
            rows[row] |= (1ULL << col);
        else
            rows[row] &= ~(1ULL << col);
    }

    /// @brief Apply transform to bit vector
    CK_TILE_HOST_DEVICE constexpr uint64_t apply(uint64_t input) const
    {
        constexpr uint64_t mask = (Bits == 64) ? ~0ULL : ((1ULL << Bits) - 1);
        input &= mask;

        uint64_t output = 0;
        for(index_t i = 0; i < Bits; ++i)
        {
            // output bit i = popcount(input & rows[i]) mod 2
            uint64_t masked = input & rows[i];
            // XOR-folding: compute parity in O(log n) without warp divergence
            masked ^= masked >> 32;
            masked ^= masked >> 16;
            masked ^= masked >> 8;
            masked ^= masked >> 4;
            masked ^= masked >> 2;
            masked ^= masked >> 1;
            output |= (masked & 1) << i;
        }
        return output;
    }

    CK_TILE_HOST_DEVICE constexpr bool operator==(const gf2_bit_matrix& other) const
    {
        for(index_t i = 0; i < Bits; ++i)
        {
            if(rows[i] != other.rows[i])
                return false;
        }
        return true;
    }

    CK_TILE_HOST_DEVICE constexpr bool operator!=(const gf2_bit_matrix& other) const
    {
        return !(*this == other);
    }
};

// Bit-level factory functions

template <index_t Bits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_gf2_bit_identity()
{
    gf2_bit_matrix<Bits> result;
    for(index_t i = 0; i < Bits; ++i)
    {
        result.rows[i] = 1ULL << i;
    }
    return result;
}

/// @brief Create bit-level XOR swizzle matching dimension-level behavior
///
/// For RowBits row and ColBits col, creates block-diagonal structure where
/// each dimension's bits are XORed uniformly (col' = col ^ row).
///
/// @tparam RowBits Number of bits for row dimension
/// @tparam ColBits Number of bits for col dimension
template <index_t RowBits, index_t ColBits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle_bits()
{
    constexpr index_t TotalBits = RowBits + ColBits;
    gf2_bit_matrix<TotalBits> result;

    // Row bits pass through (bits 0 to RowBits-1)
    for(index_t i = 0; i < RowBits; ++i)
    {
        result.rows[i] = 1ULL << i;
    }

    // Col bits XOR with corresponding row bits
    // col'[k] = col[k] ^ row[k] for k < min(RowBits, ColBits)
    for(index_t k = 0; k < ColBits; ++k)
    {
        index_t out_bit = RowBits + k;
        index_t col_bit = RowBits + k;

        // col'[k] = col[k]
        result.rows[out_bit] = 1ULL << col_bit;

        // XOR with row[k] if k < RowBits
        if(k < RowBits)
        {
            result.rows[out_bit] |= 1ULL << k;
        }
    }

    return result;
}

/// @brief Create custom bit-level swizzle for bank conflict avoidance
///
/// XORs specified row bits into specified col bits.
///
/// @param row_bits Which row bits to use as XOR sources
/// @param col_bits Which col bits to XOR into
/// @tparam RowBits Total row bits
/// @tparam ColBits Total col bits
/// @tparam NumXor Number of bit pairs to XOR
template <index_t RowBits, index_t ColBits, index_t NumXor>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_custom_swizzle_bits(
    const array<index_t, NumXor>& row_bits,
    const array<index_t, NumXor>& col_bits)
{
    constexpr index_t TotalBits = RowBits + ColBits;
    auto result = make_gf2_bit_identity<TotalBits>();

    // Add XOR connections: col_bits[i] ^= row_bits[i]
    for(index_t i = 0; i < NumXor; ++i)
    {
        index_t src_bit = row_bits[i];                    // Row bit index
        index_t dst_bit = RowBits + col_bits[i];          // Col bit index (offset by RowBits)
        result.rows[dst_bit] |= 1ULL << src_bit;
    }

    return result;
}

// Bit-level matrix operations

template <index_t Bits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_bit_compose(const gf2_bit_matrix<Bits>& a,
                                                                  const gf2_bit_matrix<Bits>& b)
{
    gf2_bit_matrix<Bits> result;

    for(index_t i = 0; i < Bits; ++i)
    {
        uint64_t row = 0;
        for(index_t j = 0; j < Bits; ++j)
        {
            // C[i,j] = XOR_{k} (A[i,k] AND B[k,j])
            bool bit = false;
            for(index_t k = 0; k < Bits; ++k)
            {
                bool a_ik = (a.rows[i] >> k) & 1;
                bool b_kj = (b.rows[k] >> j) & 1;
                bit ^= (a_ik && b_kj);
            }
            row |= static_cast<uint64_t>(bit) << j;
        }
        result.rows[i] = row;
    }
    return result;
}

template <index_t Bits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_bit_inverse(const gf2_bit_matrix<Bits>& m, bool& success)
{
    // Augmented matrix [M | I] needs 2*Bits columns in 64-bit storage
    static_assert(Bits <= 32, "gf2_bit_inverse requires Bits <= 32 (augmented matrix needs 2*Bits columns)");

    // Augmented matrix [M | I] stored as 2*Bits columns
    array<uint64_t, Bits> aug;
    for(index_t i = 0; i < Bits; ++i)
    {
        aug[i] = m.rows[i] | (1ULL << (Bits + i));
    }

    for(index_t col = 0; col < Bits; ++col)
    {
        index_t pivot = col;
        while(pivot < Bits && !((aug[pivot] >> col) & 1))
        {
            ++pivot;
        }

        if(pivot == Bits)
        {
            success = false;
            return gf2_bit_matrix<Bits>{};
        }

        if(pivot != col)
        {
            uint64_t tmp = aug[col];
            aug[col]     = aug[pivot];
            aug[pivot]   = tmp;
        }

        for(index_t row = 0; row < Bits; ++row)
        {
            if(row != col && ((aug[row] >> col) & 1))
            {
                aug[row] ^= aug[col];
            }
        }
    }

    gf2_bit_matrix<Bits> inv;
    constexpr uint64_t right_mask = (Bits == 64) ? ~0ULL : ((1ULL << Bits) - 1);
    for(index_t i = 0; i < Bits; ++i)
    {
        inv.rows[i] = (aug[i] >> Bits) & right_mask;
    }

    success = true;
    return inv;
}

template <index_t Bits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto gf2_bit_inverse(const gf2_bit_matrix<Bits>& m)
{
    bool success;
    return gf2_bit_inverse(m, success);
}

template <index_t Bits>
CK_TILE_HOST_DEVICE constexpr bool gf2_bit_is_invertible(const gf2_bit_matrix<Bits>& m)
{
    bool success;
    gf2_bit_inverse(m, success);
    return success;
}

template <index_t Bits>
CK_TILE_HOST_DEVICE constexpr bool gf2_bit_is_self_inverse(const gf2_bit_matrix<Bits>& m)
{
    return gf2_bit_compose(m, m) == make_gf2_bit_identity<Bits>();
}

// =============================================================================
// Part 3: Coordinate Packing Utilities
// =============================================================================
// Bridge between multi_index and bit vectors for the bit-level transform.

/// @brief Pack/unpack coordinates to/from bit vectors
template <index_t... DimBits>
struct coordinate_packer
{
    static constexpr index_t NDims     = sizeof...(DimBits);
    static constexpr index_t TotalBits = (DimBits + ...);

    static_assert(TotalBits <= 64, "Total bits must not exceed 64");

    static constexpr array<index_t, NDims> widths{DimBits...};

    static constexpr auto offsets = []() constexpr
    {
        array<index_t, NDims> offs{};
        index_t offset = 0;
        for(index_t i = 0; i < NDims; ++i)
        {
            offs[i] = offset;
            offset += widths[i];
        }
        return offs;
    }();

    template <typename Idx>
    CK_TILE_HOST_DEVICE static constexpr uint64_t pack(const Idx& idx)
    {
        uint64_t result = 0;
        for(index_t i = 0; i < NDims; ++i)
        {
            uint64_t mask = (widths[i] == 64) ? ~0ULL : ((1ULL << widths[i]) - 1);
            result |= (static_cast<uint64_t>(idx[i]) & mask) << offsets[i];
        }
        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr auto unpack(uint64_t bits)
    {
        multi_index<NDims> result;
        for(index_t i = 0; i < NDims; ++i)
        {
            uint64_t mask = (widths[i] == 64) ? ~0ULL : ((1ULL << widths[i]) - 1);
            result(i)     = static_cast<index_t>((bits >> offsets[i]) & mask);
        }
        return result;
    }
};

// =============================================================================
// Part 4: Bit-Level Transform with Multi-Index Interface
// =============================================================================
// Combines bit-level matrix with coordinate packing for CK integration.

/// @brief Bit-level GF(2) transform with multi_index interface
///
/// Wraps a gf2_bit_matrix and provides apply() that works with multi_index.
///
/// @tparam DimBits... Number of bits for each dimension
template <index_t... DimBits>
struct gf2_bit_transform
{
    static constexpr index_t NDims     = sizeof...(DimBits);
    static constexpr index_t TotalBits = (DimBits + ...);

    using Packer = coordinate_packer<DimBits...>;
    using Matrix = gf2_bit_matrix<TotalBits>;

    Matrix matrix;

    CK_TILE_HOST_DEVICE constexpr gf2_bit_transform() : matrix{} {}

    CK_TILE_HOST_DEVICE constexpr gf2_bit_transform(const Matrix& m) : matrix{m} {}

    template <typename IdxOut, typename IdxIn>
    CK_TILE_HOST_DEVICE constexpr void apply(IdxOut& out, const IdxIn& in) const
    {
        uint64_t bits_in  = Packer::pack(in);
        uint64_t bits_out = matrix.apply(bits_in);
        out               = Packer::unpack(bits_out);
    }

    template <typename IdxIn>
    CK_TILE_HOST_DEVICE constexpr auto apply(const IdxIn& in) const
    {
        multi_index<NDims> out;
        apply(out, in);
        return out;
    }

    CK_TILE_HOST_DEVICE constexpr bool operator==(const gf2_bit_transform& other) const
    {
        return matrix == other.matrix;
    }
};

/// @brief Create bit-level transform matching dimension-level XOR swizzle
template <index_t RowBits, index_t ColBits>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_bit_xor_swizzle()
{
    return gf2_bit_transform<RowBits, ColBits>{make_xor_swizzle_bits<RowBits, ColBits>()};
}

/// @brief Create custom bit-level transform for bank conflict avoidance
///
/// Example: To XOR row bits 5,6 into col bits 3,4:
///   make_custom_bit_swizzle<7, 7>({5, 6}, {3, 4})
template <index_t RowBits, index_t ColBits, index_t NumXor>
[[nodiscard]] CK_TILE_HOST_DEVICE constexpr auto make_custom_bit_swizzle(
    const array<index_t, NumXor>& row_bits,
    const array<index_t, NumXor>& col_bits)
{
    return gf2_bit_transform<RowBits, ColBits>{
        make_custom_swizzle_bits<RowBits, ColBits, NumXor>(row_bits, col_bits)};
}

} // namespace ck_tile
