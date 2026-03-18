// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/algorithm/gf2_linear_transform.hpp"

using namespace ck_tile;

// =============================================================================
// Basic Matrix Operations
// =============================================================================

TEST(GF2Matrix, DefaultConstruction)
{
    gf2_matrix<4, 4> m;
    for(index_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(m.rows[i], 0ULL);
    }
}

TEST(GF2Matrix, GetSet)
{
    gf2_matrix<4, 4> m;

    // Set some bits
    m.set(0, 0, true);
    m.set(1, 2, true);
    m.set(3, 3, true);

    EXPECT_TRUE(m.get(0, 0));
    EXPECT_FALSE(m.get(0, 1));
    EXPECT_TRUE(m.get(1, 2));
    EXPECT_FALSE(m.get(2, 2));
    EXPECT_TRUE(m.get(3, 3));

    // Clear a bit
    m.set(0, 0, false);
    EXPECT_FALSE(m.get(0, 0));
}

TEST(GF2Matrix, Equality)
{
    gf2_matrix<3, 3> a;
    gf2_matrix<3, 3> b;

    a.rows[0] = 0b001;
    a.rows[1] = 0b010;
    a.rows[2] = 0b100;

    b.rows[0] = 0b001;
    b.rows[1] = 0b010;
    b.rows[2] = 0b100;

    EXPECT_EQ(a, b);

    b.rows[1] = 0b011;
    EXPECT_NE(a, b);
}

// =============================================================================
// Identity Matrix
// =============================================================================

TEST(GF2Matrix, Identity2x2)
{
    auto id = make_gf2_identity<2>();
    EXPECT_EQ(id.rows[0], 0b01ULL);
    EXPECT_EQ(id.rows[1], 0b10ULL);
}

TEST(GF2Matrix, Identity4x4)
{
    auto id = make_gf2_identity<4>();
    EXPECT_EQ(id.rows[0], 0b0001ULL);
    EXPECT_EQ(id.rows[1], 0b0010ULL);
    EXPECT_EQ(id.rows[2], 0b0100ULL);
    EXPECT_EQ(id.rows[3], 0b1000ULL);
}

TEST(GF2Matrix, IdentityApply)
{
    auto id = make_gf2_identity<8>();

    // Identity should pass through all values unchanged
    for(uint64_t i = 0; i < 256; ++i)
    {
        EXPECT_EQ(id.apply(i), i);
    }
}

// =============================================================================
// Apply Transform
// =============================================================================

TEST(GF2Matrix, ApplySimple)
{
    // Matrix that XORs bit 0 into bit 1: [1 0; 1 1]
    // out[0] = in[0]
    // out[1] = in[0] ^ in[1]
    gf2_matrix<2, 2> m;
    m.rows[0] = 0b01; // out[0] = in[0]
    m.rows[1] = 0b11; // out[1] = in[0] ^ in[1]

    // Input: 00 -> Output: 00
    EXPECT_EQ(m.apply(0b00), 0b00ULL);

    // Input: 01 -> Output: 01 (in[0]=1, in[1]=0 -> out[0]=1, out[1]=1^0=1)
    EXPECT_EQ(m.apply(0b01), 0b11ULL);

    // Input: 10 -> Output: 10 (in[0]=0, in[1]=1 -> out[0]=0, out[1]=0^1=1)
    EXPECT_EQ(m.apply(0b10), 0b10ULL);

    // Input: 11 -> Output: 01 (in[0]=1, in[1]=1 -> out[0]=1, out[1]=1^1=0)
    EXPECT_EQ(m.apply(0b11), 0b01ULL);
}

TEST(GF2Matrix, ApplyMasksInput)
{
    auto id = make_gf2_identity<4>();

    // Input with extra bits should be masked
    EXPECT_EQ(id.apply(0xFF), 0x0FULL);
    EXPECT_EQ(id.apply(0x1234), 0x04ULL);
}

// =============================================================================
// Matrix Composition
// =============================================================================

TEST(GF2Matrix, ComposeIdentityLeft)
{
    auto id = make_gf2_identity<4>();

    gf2_matrix<4, 4> m;
    m.rows[0] = 0b0001;
    m.rows[1] = 0b0011;
    m.rows[2] = 0b0101;
    m.rows[3] = 0b1001;

    auto result = gf2_compose(id, m);
    EXPECT_EQ(result, m);
}

TEST(GF2Matrix, ComposeIdentityRight)
{
    auto id = make_gf2_identity<4>();

    gf2_matrix<4, 4> m;
    m.rows[0] = 0b0001;
    m.rows[1] = 0b0011;
    m.rows[2] = 0b0101;
    m.rows[3] = 0b1001;

    auto result = gf2_compose(m, id);
    EXPECT_EQ(result, m);
}

TEST(GF2Matrix, ComposeSelfInverse)
{
    // XOR swizzle is self-inverse
    auto swizzle = make_xor_swizzle<3, 3>();
    auto composed = gf2_compose(swizzle, swizzle);
    auto id = make_gf2_identity<6>();

    EXPECT_EQ(composed, id);
}

TEST(GF2Matrix, ComposeChain)
{
    // Verify (A @ B) @ C == A @ (B @ C) by checking apply results
    gf2_matrix<4, 4> a, b, c;

    a.rows[0] = 0b0001;
    a.rows[1] = 0b0011;
    a.rows[2] = 0b0100;
    a.rows[3] = 0b1000;

    b.rows[0] = 0b0010;
    b.rows[1] = 0b0001;
    b.rows[2] = 0b1100;
    b.rows[3] = 0b0100;

    c.rows[0] = 0b1001;
    c.rows[1] = 0b0110;
    c.rows[2] = 0b0011;
    c.rows[3] = 0b1100;

    auto ab = gf2_compose(a, b);
    auto bc = gf2_compose(b, c);
    auto ab_c = gf2_compose(ab, c);
    auto a_bc = gf2_compose(a, bc);

    // Check associativity via apply
    for(uint64_t i = 0; i < 16; ++i)
    {
        EXPECT_EQ(ab_c.apply(i), a_bc.apply(i));
    }
}

// =============================================================================
// Matrix Inversion
// =============================================================================

TEST(GF2Matrix, InverseIdentity)
{
    auto id = make_gf2_identity<4>();
    bool success;
    auto inv = gf2_inverse(id, success);

    EXPECT_TRUE(success);
    EXPECT_EQ(inv, id);
}

TEST(GF2Matrix, InverseSimple2x2)
{
    // [1 1; 0 1] - upper triangular
    // Inverse should be [1 1; 0 1] (self-inverse in this case)
    gf2_matrix<2, 2> m;
    m.rows[0] = 0b01; // [1, 0]
    m.rows[1] = 0b11; // [1, 1]

    bool success;
    auto inv = gf2_inverse(m, success);

    EXPECT_TRUE(success);

    // Verify M @ M^-1 = I
    auto product = gf2_compose(m, inv);
    auto id = make_gf2_identity<2>();
    EXPECT_EQ(product, id);
}

TEST(GF2Matrix, InverseXorSwizzle)
{
    // XOR swizzle is self-inverse
    auto swizzle = make_xor_swizzle<4, 4>();
    bool success;
    auto inv = gf2_inverse(swizzle, success);

    EXPECT_TRUE(success);
    EXPECT_EQ(inv, swizzle); // Self-inverse
}

TEST(GF2Matrix, InverseSingular)
{
    // Singular matrix (row 0 and row 1 are identical)
    gf2_matrix<2, 2> m;
    m.rows[0] = 0b11;
    m.rows[1] = 0b11;

    bool success;
    gf2_inverse(m, success);

    EXPECT_FALSE(success);
}

TEST(GF2Matrix, InverseZeroRow)
{
    // Matrix with zero row is singular
    gf2_matrix<3, 3> m;
    m.rows[0] = 0b001;
    m.rows[1] = 0b000; // Zero row
    m.rows[2] = 0b100;

    bool success;
    gf2_inverse(m, success);

    EXPECT_FALSE(success);
}

TEST(GF2Matrix, InverseRoundTrip)
{
    // Create a random-ish invertible matrix
    gf2_matrix<4, 4> m;
    m.rows[0] = 0b0001;
    m.rows[1] = 0b0011;
    m.rows[2] = 0b0111;
    m.rows[3] = 0b1111;

    bool success;
    auto inv = gf2_inverse(m, success);
    EXPECT_TRUE(success);

    // Verify M @ M^-1 = I
    auto product = gf2_compose(m, inv);
    auto id = make_gf2_identity<4>();
    EXPECT_EQ(product, id);

    // Verify M^-1 @ M = I
    auto product2 = gf2_compose(inv, m);
    EXPECT_EQ(product2, id);

    // Verify (M^-1)^-1 = M
    auto inv_inv = gf2_inverse(inv, success);
    EXPECT_TRUE(success);
    EXPECT_EQ(inv_inv, m);
}

TEST(GF2Matrix, IsInvertible)
{
    auto id = make_gf2_identity<4>();
    EXPECT_TRUE(gf2_is_invertible(id));

    auto swizzle = make_xor_swizzle<3, 3>();
    EXPECT_TRUE(gf2_is_invertible(swizzle));

    // Singular matrix
    gf2_matrix<2, 2> singular;
    singular.rows[0] = 0b11;
    singular.rows[1] = 0b11;
    EXPECT_FALSE(gf2_is_invertible(singular));
}

TEST(GF2Matrix, IsSelfInverse)
{
    // Identity is self-inverse
    auto id = make_gf2_identity<4>();
    EXPECT_TRUE(gf2_is_self_inverse(id));

    // XOR swizzle is self-inverse
    auto swizzle = make_xor_swizzle<3, 3>();
    EXPECT_TRUE(gf2_is_self_inverse(swizzle));

    // General matrix may not be self-inverse
    gf2_matrix<2, 2> m;
    m.rows[0] = 0b10; // Swap bits
    m.rows[1] = 0b01;
    EXPECT_TRUE(gf2_is_self_inverse(m)); // Permutation is self-inverse if it's an involution

    // Non-self-inverse example
    gf2_matrix<3, 3> shift;
    shift.rows[0] = 0b010; // Cyclic shift
    shift.rows[1] = 0b100;
    shift.rows[2] = 0b001;
    EXPECT_FALSE(gf2_is_self_inverse(shift));
}

// =============================================================================
// XOR Swizzle
// =============================================================================

TEST(GF2Swizzle, XorSwizzle3x3Structure)
{
    // 3 row bits, 3 col bits -> 6x6 matrix
    auto swizzle = make_xor_swizzle<3, 3>();

    // Verify structure:
    // Row bits (0-2) pass through
    EXPECT_EQ(swizzle.rows[0], 0b000001ULL); // row[0] = in[0]
    EXPECT_EQ(swizzle.rows[1], 0b000010ULL); // row[1] = in[1]
    EXPECT_EQ(swizzle.rows[2], 0b000100ULL); // row[2] = in[2]

    // Col bits (3-5) XOR with row bits
    EXPECT_EQ(swizzle.rows[3], 0b001001ULL); // col[0] = in[0] ^ in[3]
    EXPECT_EQ(swizzle.rows[4], 0b010010ULL); // col[1] = in[1] ^ in[4]
    EXPECT_EQ(swizzle.rows[5], 0b100100ULL); // col[2] = in[2] ^ in[5]
}

TEST(GF2Swizzle, XorSwizzleMatchesCKBehavior)
{
    // Test that our GF(2) XOR swizzle matches CK's xor_t behavior:
    // row' = row
    // col' = col ^ (row % col_width)

    constexpr index_t RowBits = 3;
    constexpr index_t ColBits = 3;
    auto swizzle = make_xor_swizzle<RowBits, ColBits>();

    using Packer = coordinate_packer<RowBits, ColBits>;

    for(index_t row = 0; row < 8; ++row)
    {
        for(index_t col = 0; col < 8; ++col)
        {
            uint64_t input = Packer::pack(row, col);
            uint64_t output = swizzle.apply(input);
            auto result = Packer::unpack(output);

            index_t expected_row = row;
            index_t expected_col = col ^ (row % 8);

            EXPECT_EQ(result[0], expected_row) << "row=" << row << " col=" << col;
            EXPECT_EQ(result[1], expected_col) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2Swizzle, XorSwizzleSelfInverse)
{
    // Verify XOR swizzle is self-inverse for various sizes
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<2, 2>()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<3, 3>()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<4, 4>()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<6, 4>()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<7, 7>()));
}

TEST(GF2Swizzle, XorSwizzleNPartialXor)
{
    // Test partial XOR: only first 2 bits XOR
    auto swizzle = make_xor_swizzle_n<3, 3, 2>();

    // Row bits pass through
    EXPECT_EQ(swizzle.rows[0], 0b000001ULL);
    EXPECT_EQ(swizzle.rows[1], 0b000010ULL);
    EXPECT_EQ(swizzle.rows[2], 0b000100ULL);

    // Only first 2 col bits XOR
    EXPECT_EQ(swizzle.rows[3], 0b001001ULL); // col[0] = in[0] ^ in[3]
    EXPECT_EQ(swizzle.rows[4], 0b010010ULL); // col[1] = in[1] ^ in[4]
    EXPECT_EQ(swizzle.rows[5], 0b100000ULL); // col[2] = in[5] only (no XOR)
}

TEST(GF2Swizzle, XorSwizzleNZeroXor)
{
    // Zero XOR bits = identity
    auto swizzle = make_xor_swizzle_n<3, 3, 0>();
    auto id = make_gf2_identity<6>();
    EXPECT_EQ(swizzle, id);
}

// =============================================================================
// Permutation
// =============================================================================

TEST(GF2Permutation, Identity)
{
    array<index_t, 4> perm{0, 1, 2, 3};
    auto m = make_gf2_permutation<4>(perm);
    auto id = make_gf2_identity<4>();
    EXPECT_EQ(m, id);
}

TEST(GF2Permutation, Swap)
{
    // Swap bits 0 and 1
    array<index_t, 4> perm{1, 0, 2, 3};
    auto m = make_gf2_permutation<4>(perm);

    EXPECT_EQ(m.apply(0b0001), 0b0010ULL); // bit 0 -> bit 1
    EXPECT_EQ(m.apply(0b0010), 0b0001ULL); // bit 1 -> bit 0
    EXPECT_EQ(m.apply(0b0011), 0b0011ULL); // bits 0,1 -> bits 1,0 = same
    EXPECT_EQ(m.apply(0b1100), 0b1100ULL); // bits 2,3 unchanged
}

TEST(GF2Permutation, Reverse)
{
    // Reverse all bits
    array<index_t, 4> perm{3, 2, 1, 0};
    auto m = make_gf2_permutation<4>(perm);

    EXPECT_EQ(m.apply(0b0001), 0b1000ULL);
    EXPECT_EQ(m.apply(0b0010), 0b0100ULL);
    EXPECT_EQ(m.apply(0b1001), 0b1001ULL); // Palindrome
    EXPECT_EQ(m.apply(0b1010), 0b0101ULL);
}

TEST(GF2Permutation, SelfInverse)
{
    // Swap is self-inverse
    array<index_t, 4> swap{1, 0, 3, 2};
    auto m = make_gf2_permutation<4>(swap);
    EXPECT_TRUE(gf2_is_self_inverse(m));

    // Reverse of even length is self-inverse
    array<index_t, 4> rev{3, 2, 1, 0};
    auto m2 = make_gf2_permutation<4>(rev);
    EXPECT_TRUE(gf2_is_self_inverse(m2));
}

// =============================================================================
// Coordinate Packer
// =============================================================================

TEST(CoordinatePacker, PackUnpack2D)
{
    using Packer = coordinate_packer<3, 4>; // 3 bits for dim 0, 4 bits for dim 1

    for(index_t a = 0; a < 8; ++a)
    {
        for(index_t b = 0; b < 16; ++b)
        {
            auto packed = Packer::pack(a, b);
            auto unpacked = Packer::unpack(packed);

            EXPECT_EQ(unpacked[0], a);
            EXPECT_EQ(unpacked[1], b);
        }
    }
}

TEST(CoordinatePacker, PackUnpack3D)
{
    using Packer = coordinate_packer<2, 3, 4>; // 2, 3, 4 bits

    for(index_t a = 0; a < 4; ++a)
    {
        for(index_t b = 0; b < 8; ++b)
        {
            for(index_t c = 0; c < 16; ++c)
            {
                auto packed = Packer::pack(a, b, c);
                auto unpacked = Packer::unpack(packed);

                EXPECT_EQ(unpacked[0], a);
                EXPECT_EQ(unpacked[1], b);
                EXPECT_EQ(unpacked[2], c);
            }
        }
    }
}

TEST(CoordinatePacker, BitLayout)
{
    using Packer = coordinate_packer<3, 4>; // 3 bits, then 4 bits

    // dim 0 should be in bits 0-2
    // dim 1 should be in bits 3-6
    EXPECT_EQ(Packer::pack(0b111, 0), 0b0000111ULL);
    EXPECT_EQ(Packer::pack(0, 0b1111), 0b1111000ULL);
    EXPECT_EQ(Packer::pack(0b111, 0b1111), 0b1111111ULL);
}

// =============================================================================
// Integration: End-to-End Swizzle
// =============================================================================

TEST(GF2Integration, SwizzleRoundTrip)
{
    // Simulate LDS swizzle: write with swizzle, read with same swizzle
    constexpr index_t RowBits = 4;
    constexpr index_t ColBits = 4;
    auto swizzle = make_xor_swizzle<RowBits, ColBits>();

    using Packer = coordinate_packer<RowBits, ColBits>;

    for(index_t row = 0; row < 16; ++row)
    {
        for(index_t col = 0; col < 16; ++col)
        {
            // Original coordinate
            uint64_t original = Packer::pack(row, col);

            // Swizzle (write)
            uint64_t swizzled = swizzle.apply(original);

            // Swizzle again (read) - should return to original
            uint64_t restored = swizzle.apply(swizzled);

            EXPECT_EQ(restored, original) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2Integration, LargeMatrix)
{
    // Test with realistic LDS dimensions: 64 rows (6 bits) x 128 cols (7 bits)
    constexpr index_t RowBits = 6;
    constexpr index_t ColBits = 6; // XOR with 6 bits of col

    auto swizzle = make_xor_swizzle<RowBits, ColBits>();

    // Verify it's invertible
    EXPECT_TRUE(gf2_is_invertible(swizzle));

    // Verify self-inverse
    EXPECT_TRUE(gf2_is_self_inverse(swizzle));

    // Spot check a few values
    using Packer = coordinate_packer<RowBits, ColBits>;

    auto check = [&](index_t row, index_t col) {
        uint64_t input = Packer::pack(row, col);
        uint64_t output = swizzle.apply(input);
        auto result = Packer::unpack(output);

        EXPECT_EQ(result[0], row);
        EXPECT_EQ(result[1], col ^ (row % 64));
    };

    check(0, 0);
    check(1, 0);
    check(63, 63);
    check(32, 16);
    check(15, 47);
}

// =============================================================================
// Compile-Time Evaluation
// =============================================================================

TEST(GF2CompileTime, ConstexprIdentity)
{
    constexpr auto id = make_gf2_identity<4>();
    static_assert(id.rows[0] == 0b0001);
    static_assert(id.rows[1] == 0b0010);
    static_assert(id.rows[2] == 0b0100);
    static_assert(id.rows[3] == 0b1000);
}

TEST(GF2CompileTime, ConstexprApply)
{
    constexpr auto id = make_gf2_identity<4>();
    static_assert(id.apply(0b1010) == 0b1010);
}

TEST(GF2CompileTime, ConstexprXorSwizzle)
{
    constexpr auto swizzle = make_xor_swizzle<3, 3>();
    static_assert(swizzle.rows[0] == 0b000001);
    static_assert(swizzle.rows[3] == 0b001001);
}

TEST(GF2CompileTime, ConstexprSelfInverse)
{
    constexpr auto swizzle = make_xor_swizzle<3, 3>();
    constexpr auto composed = gf2_compose(swizzle, swizzle);
    constexpr auto id = make_gf2_identity<6>();
    static_assert(composed == id);
}

TEST(GF2CompileTime, ConstexprInverse)
{
    constexpr auto id = make_gf2_identity<4>();
    constexpr bool success = gf2_is_invertible(id);
    static_assert(success);
}

// =============================================================================
// Popcount
// =============================================================================

TEST(Popcount, BasicValues)
{
    EXPECT_EQ(popcount(0ULL), 0);
    EXPECT_EQ(popcount(1ULL), 1);
    EXPECT_EQ(popcount(0b1010ULL), 2);
    EXPECT_EQ(popcount(0b1111ULL), 4);
    EXPECT_EQ(popcount(0xFFFFFFFFFFFFFFFFULL), 64);
    EXPECT_EQ(popcount(0xAAAAAAAAAAAAAAAAULL), 32);
}

TEST(Popcount, Constexpr)
{
    static_assert(popcount(0ULL) == 0);
    static_assert(popcount(0b1111ULL) == 4);
    static_assert(popcount(0xFFULL) == 8);
}
