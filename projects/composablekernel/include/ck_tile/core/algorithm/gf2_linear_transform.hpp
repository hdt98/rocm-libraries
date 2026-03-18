// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {

/// @brief Count number of set bits (population count)
/// @param x Input value
/// @return Number of 1 bits in x
CK_TILE_HOST_DEVICE constexpr index_t popcount(uint64_t x)
{
#if CK_TILE_USE_AMD_GCN_INLINE_ASM
    if(!__builtin_is_constant_evaluated())
    {
        // Use hardware popcount on device
        return __builtin_popcountll(x);
    }
#endif
    // Compile-time or host fallback
    index_t count = 0;
    while(x)
    {
        count += (x & 1);
        x >>= 1;
    }
    return count;
}

/// @brief GF(2) matrix for linear transforms over the binary field
///
/// A matrix over GF(2) where:
/// - Elements are 0 or 1
/// - Addition is XOR
/// - Multiplication is AND
///
/// Used for expressing swizzle patterns, permutations, and XOR-based
/// coordinate transforms in a unified framework.
///
/// @tparam InBits Number of input bits (columns)
/// @tparam OutBits Number of output bits (rows)
template <index_t InBits_, index_t OutBits_>
struct gf2_matrix
{
    static_assert(InBits_ > 0 && InBits_ <= 64, "InBits must be in range [1, 64]");
    static_assert(OutBits_ > 0 && OutBits_ <= 64, "OutBits must be in range [1, 64]");

    static constexpr index_t InBits  = InBits_;
    static constexpr index_t OutBits = OutBits_;

    // Each row is stored as a bitmask indicating which input bits to XOR
    array<uint64_t, OutBits> rows;

    /// @brief Default constructor - zero matrix
    CK_TILE_HOST_DEVICE constexpr gf2_matrix() : rows{} {}

    /// @brief Construct from initializer list of row values
    CK_TILE_HOST_DEVICE constexpr gf2_matrix(const array<uint64_t, OutBits>& rows_) : rows{rows_} {}

    /// @brief Apply the transform: output = M @ input (over GF(2))
    ///
    /// Each output bit is the XOR of input bits where the corresponding
    /// matrix row has 1s.
    ///
    /// @param input Input bit vector (only lower InBits are used)
    /// @return Output bit vector (only lower OutBits are meaningful)
    CK_TILE_HOST_DEVICE constexpr uint64_t apply(uint64_t input) const
    {
        // Mask input to valid bits
        constexpr uint64_t in_mask = (InBits == 64) ? ~0ULL : ((1ULL << InBits) - 1);
        input &= in_mask;

        uint64_t output = 0;
        for(index_t i = 0; i < OutBits; ++i)
        {
            // output bit i = popcount(input AND row[i]) mod 2
            uint64_t masked = input & rows[i];
            output |= static_cast<uint64_t>(popcount(masked) & 1) << i;
        }
        return output;
    }

    /// @brief Get element at (row, col)
    CK_TILE_HOST_DEVICE constexpr bool get(index_t row, index_t col) const
    {
        return (rows[row] >> col) & 1;
    }

    /// @brief Set element at (row, col)
    CK_TILE_HOST_DEVICE constexpr void set(index_t row, index_t col, bool value)
    {
        if(value)
        {
            rows[row] |= (1ULL << col);
        }
        else
        {
            rows[row] &= ~(1ULL << col);
        }
    }

    /// @brief Check equality
    CK_TILE_HOST_DEVICE constexpr bool operator==(const gf2_matrix& other) const
    {
        for(index_t i = 0; i < OutBits; ++i)
        {
            if(rows[i] != other.rows[i])
                return false;
        }
        return true;
    }

    CK_TILE_HOST_DEVICE constexpr bool operator!=(const gf2_matrix& other) const
    {
        return !(*this == other);
    }
};

/// @brief Create identity matrix
template <index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_gf2_identity()
{
    gf2_matrix<N, N> result;
    for(index_t i = 0; i < N; ++i)
    {
        result.rows[i] = 1ULL << i;
    }
    return result;
}

/// @brief Compose two GF(2) matrices: C = A @ B
///
/// If A is OutA x MidBits and B is MidBits x InB,
/// then C is OutA x InB.
///
/// C[i,j] = XOR_{k} (A[i,k] AND B[k,j])
template <index_t InB, index_t MidBits, index_t OutA>
CK_TILE_HOST_DEVICE constexpr auto gf2_compose(const gf2_matrix<MidBits, OutA>& a,
                                                const gf2_matrix<InB, MidBits>& b)
{
    gf2_matrix<InB, OutA> result;

    for(index_t i = 0; i < OutA; ++i)
    {
        uint64_t row = 0;
        for(index_t j = 0; j < InB; ++j)
        {
            // C[i,j] = XOR_{k} (A[i,k] AND B[k,j])
            index_t bit = 0;
            for(index_t k = 0; k < MidBits; ++k)
            {
                bool a_ik = (a.rows[i] >> k) & 1;
                bool b_kj = (b.rows[k] >> j) & 1;
                bit ^= (a_ik & b_kj);
            }
            row |= static_cast<uint64_t>(bit) << j;
        }
        result.rows[i] = row;
    }
    return result;
}

/// @brief Compute inverse of a square GF(2) matrix using Gaussian elimination
///
/// @param m Input matrix (must be square and invertible)
/// @param success Output parameter set to true if matrix is invertible
/// @return Inverse matrix (undefined if not invertible)
template <index_t N>
CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<N, N>& m, bool& success)
{
    // Augmented matrix [M | I]
    // We store 2N bits per row: left N bits are M, right N bits are I
    array<uint64_t, N> aug;
    for(index_t i = 0; i < N; ++i)
    {
        // Left half: original matrix row
        // Right half: identity row (1 at position i)
        aug[i] = m.rows[i] | (1ULL << (N + i));
    }

    // Forward elimination with full row reduction (Gauss-Jordan)
    for(index_t col = 0; col < N; ++col)
    {
        // Find pivot (row with 1 in this column)
        index_t pivot = col;
        while(pivot < N && !((aug[pivot] >> col) & 1))
        {
            ++pivot;
        }

        if(pivot == N)
        {
            // No pivot found - matrix is singular
            success = false;
            return gf2_matrix<N, N>{};
        }

        // Swap rows if needed
        if(pivot != col)
        {
            uint64_t tmp  = aug[col];
            aug[col]      = aug[pivot];
            aug[pivot]    = tmp;
        }

        // Eliminate this column in all other rows
        for(index_t row = 0; row < N; ++row)
        {
            if(row != col && ((aug[row] >> col) & 1))
            {
                aug[row] ^= aug[col]; // XOR = subtract in GF(2)
            }
        }
    }

    // Extract right half (the inverse)
    gf2_matrix<N, N> inv;
    constexpr uint64_t right_mask = (N == 64) ? ~0ULL : ((1ULL << N) - 1);
    for(index_t i = 0; i < N; ++i)
    {
        inv.rows[i] = (aug[i] >> N) & right_mask;
    }

    success = true;
    return inv;
}

/// @brief Compute inverse of a square GF(2) matrix
///
/// Simpler interface that doesn't return success flag.
/// Use gf2_inverse(m, success) if you need to check invertibility.
template <index_t N>
CK_TILE_HOST_DEVICE constexpr auto gf2_inverse(const gf2_matrix<N, N>& m)
{
    bool success;
    return gf2_inverse(m, success);
}

/// @brief Check if a square GF(2) matrix is invertible
template <index_t N>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_invertible(const gf2_matrix<N, N>& m)
{
    bool success;
    gf2_inverse(m, success);
    return success;
}

/// @brief Check if a square GF(2) matrix is self-inverse (M @ M = I)
template <index_t N>
CK_TILE_HOST_DEVICE constexpr bool gf2_is_self_inverse(const gf2_matrix<N, N>& m)
{
    return gf2_compose(m, m) == make_gf2_identity<N>();
}

// =============================================================================
// Swizzle Pattern Builders
// =============================================================================

/// @brief Create XOR swizzle matrix matching CK's xor_t transform
///
/// Implements: row' = row, col' = col ^ (row % col_width)
/// When col_width = 2^ColBits, the modulo becomes a bit mask.
///
/// The resulting matrix is:
///   [I_row    0   ]
///   [I_row  I_col ]  (XOR of row bits into col bits)
///
/// @tparam RowBits Number of bits for row coordinate
/// @tparam ColBits Number of bits for column coordinate
template <index_t RowBits, index_t ColBits>
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle()
{
    static_assert(RowBits >= ColBits, "RowBits must be >= ColBits for xor_t compatibility");

    constexpr index_t TotalBits = RowBits + ColBits;
    gf2_matrix<TotalBits, TotalBits> result;

    // Input layout:  [row bits (RowBits) | col bits (ColBits)]
    // Output layout: [row' bits (RowBits) | col' bits (ColBits)]

    // Row bits pass through unchanged
    for(index_t i = 0; i < RowBits; ++i)
    {
        result.rows[i] = 1ULL << i;
    }

    // Col bits: col'[j] = col[j] XOR row[j] (for j < ColBits)
    for(index_t j = 0; j < ColBits; ++j)
    {
        // col'[j] = row[j] XOR col[j]
        // row[j] is at bit position j
        // col[j] is at bit position RowBits + j
        result.rows[RowBits + j] = (1ULL << j) | (1ULL << (RowBits + j));
    }

    return result;
}

/// @brief Create a general XOR swizzle with configurable XOR bits
///
/// Allows specifying which row bits XOR into which column bits.
/// The XorMask indicates which of the low bits of row should XOR into col.
///
/// @tparam RowBits Number of bits for row coordinate
/// @tparam ColBits Number of bits for column coordinate
/// @tparam XorBits Number of row bits to XOR into column (must be <= min(RowBits, ColBits))
template <index_t RowBits, index_t ColBits, index_t XorBits>
CK_TILE_HOST_DEVICE constexpr auto make_xor_swizzle_n()
{
    static_assert(XorBits <= RowBits && XorBits <= ColBits,
                  "XorBits must be <= min(RowBits, ColBits)");

    constexpr index_t TotalBits = RowBits + ColBits;
    gf2_matrix<TotalBits, TotalBits> result;

    // Row bits pass through unchanged
    for(index_t i = 0; i < RowBits; ++i)
    {
        result.rows[i] = 1ULL << i;
    }

    // Col bits: only first XorBits get XORed with row
    for(index_t j = 0; j < ColBits; ++j)
    {
        if(j < XorBits)
        {
            // col'[j] = row[j] XOR col[j]
            result.rows[RowBits + j] = (1ULL << j) | (1ULL << (RowBits + j));
        }
        else
        {
            // col'[j] = col[j] (pass through)
            result.rows[RowBits + j] = 1ULL << (RowBits + j);
        }
    }

    return result;
}

/// @brief Create a bit permutation matrix
///
/// @tparam N Number of bits
/// @param perm Permutation array where perm[i] is the source bit for output bit i
template <index_t N>
CK_TILE_HOST_DEVICE constexpr auto make_gf2_permutation(const array<index_t, N>& perm)
{
    gf2_matrix<N, N> result;
    for(index_t i = 0; i < N; ++i)
    {
        result.rows[i] = 1ULL << perm[i];
    }
    return result;
}

// =============================================================================
// Coordinate Packing/Unpacking Utilities
// =============================================================================

/// @brief Pack multiple coordinates into a single bit vector
///
/// @tparam Bits Sequence of bit widths for each coordinate
template <index_t... Bits>
struct coordinate_packer
{
    static constexpr index_t NumDims   = sizeof...(Bits);
    static constexpr index_t TotalBits = (Bits + ...);

    static_assert(TotalBits <= 64, "Total bits must not exceed 64");

    // Bit offsets for each dimension
    static constexpr auto offsets = []() constexpr
    {
        array<index_t, NumDims> offs{};
        array<index_t, NumDims> widths{Bits...};
        index_t offset = 0;
        for(index_t i = 0; i < NumDims; ++i)
        {
            offs[i] = offset;
            offset += widths[i];
        }
        return offs;
    }();

    static constexpr auto widths = array<index_t, NumDims>{Bits...};

    /// @brief Pack coordinates into a single uint64_t
    template <typename... Coords>
    CK_TILE_HOST_DEVICE static constexpr uint64_t pack(Coords... coords)
    {
        static_assert(sizeof...(Coords) == NumDims, "Wrong number of coordinates");
        array<index_t, NumDims> c{static_cast<index_t>(coords)...};
        uint64_t result = 0;
        for(index_t i = 0; i < NumDims; ++i)
        {
            uint64_t mask = (widths[i] == 64) ? ~0ULL : ((1ULL << widths[i]) - 1);
            result |= (static_cast<uint64_t>(c[i]) & mask) << offsets[i];
        }
        return result;
    }

    /// @brief Unpack a single uint64_t into coordinates
    CK_TILE_HOST_DEVICE static constexpr auto unpack(uint64_t bits)
    {
        array<index_t, NumDims> result;
        for(index_t i = 0; i < NumDims; ++i)
        {
            uint64_t mask = (widths[i] == 64) ? ~0ULL : ((1ULL << widths[i]) - 1);
            result[i]     = static_cast<index_t>((bits >> offsets[i]) & mask);
        }
        return result;
    }
};

} // namespace ck_tile
