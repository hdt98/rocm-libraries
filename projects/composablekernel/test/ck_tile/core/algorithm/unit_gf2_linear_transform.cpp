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
    gf2_matrix<3> m;
    for(index_t i = 0; i < 3; ++i)
    {
        for(index_t j = 0; j < 3; ++j)
        {
            EXPECT_FALSE(m.get(i, j));
        }
    }
}

TEST(GF2Matrix, GetSet)
{
    gf2_matrix<3> m;

    m.set(0, 0, true);
    m.set(1, 2, true);
    m.set(2, 1, true);

    EXPECT_TRUE(m.get(0, 0));
    EXPECT_FALSE(m.get(0, 1));
    EXPECT_TRUE(m.get(1, 2));
    EXPECT_FALSE(m.get(2, 2));
    EXPECT_TRUE(m.get(2, 1));

    m.set(0, 0, false);
    EXPECT_FALSE(m.get(0, 0));
}

TEST(GF2Matrix, Equality)
{
    auto a = make_gf2_identity<3>();
    auto b = make_gf2_identity<3>();

    EXPECT_EQ(a, b);

    b.set(0, 1, true);
    EXPECT_NE(a, b);
}

// =============================================================================
// Identity Matrix
// =============================================================================

TEST(GF2Matrix, Identity2)
{
    auto id = make_gf2_identity<2>();

    EXPECT_TRUE(id.get(0, 0));
    EXPECT_FALSE(id.get(0, 1));
    EXPECT_FALSE(id.get(1, 0));
    EXPECT_TRUE(id.get(1, 1));
}

TEST(GF2Matrix, Identity4)
{
    auto id = make_gf2_identity<4>();

    for(index_t i = 0; i < 4; ++i)
    {
        for(index_t j = 0; j < 4; ++j)
        {
            EXPECT_EQ(id.get(i, j), i == j);
        }
    }
}

TEST(GF2Matrix, IdentityApply)
{
    auto id = make_gf2_identity<3>();

    multi_index<3> in{5, 10, 15};
    auto out = id.apply(in);

    EXPECT_EQ(out[0], 5);
    EXPECT_EQ(out[1], 10);
    EXPECT_EQ(out[2], 15);
}

// =============================================================================
// Apply Transform
// =============================================================================

TEST(GF2Matrix, ApplyXorSwizzle2D)
{
    auto swizzle = make_xor_swizzle_2d();

    // row' = row, col' = row ^ col
    {
        multi_index<2> in{0, 0};
        auto out = swizzle.apply(in);
        EXPECT_EQ(out[0], 0); // row' = 0
        EXPECT_EQ(out[1], 0); // col' = 0 ^ 0 = 0
    }
    {
        multi_index<2> in{3, 5};
        auto out = swizzle.apply(in);
        EXPECT_EQ(out[0], 3);     // row' = 3
        EXPECT_EQ(out[1], 3 ^ 5); // col' = 3 ^ 5 = 6
    }
    {
        multi_index<2> in{7, 7};
        auto out = swizzle.apply(in);
        EXPECT_EQ(out[0], 7);     // row' = 7
        EXPECT_EQ(out[1], 7 ^ 7); // col' = 0
    }
}

TEST(GF2Matrix, ApplyGenericXor)
{
    // XOR dimension 0 into dimension 2 in a 3D space
    auto swizzle = make_xor_swizzle<3>(0, 2);

    multi_index<3> in{5, 10, 15};
    auto out = swizzle.apply(in);

    EXPECT_EQ(out[0], 5);       // unchanged
    EXPECT_EQ(out[1], 10);      // unchanged
    EXPECT_EQ(out[2], 5 ^ 15);  // XOR'd with dim 0
}

TEST(GF2Matrix, ApplyMultipleXor)
{
    // out[0] = in[0] ^ in[1]
    // out[1] = in[1]
    gf2_matrix<2> m;
    m.set(0, 0, true);
    m.set(0, 1, true);
    m.set(1, 1, true);

    multi_index<2> in{3, 5};
    auto out = m.apply(in);

    EXPECT_EQ(out[0], 3 ^ 5); // = 6
    EXPECT_EQ(out[1], 5);
}

TEST(GF2Matrix, ApplyThreeWayXor)
{
    // out[0] = in[0] ^ in[1] ^ in[2]
    gf2_matrix<3> m;
    m.set(0, 0, true);
    m.set(0, 1, true);
    m.set(0, 2, true);
    m.set(1, 1, true);
    m.set(2, 2, true);

    multi_index<3> in{1, 2, 4};
    auto out = m.apply(in);

    EXPECT_EQ(out[0], 1 ^ 2 ^ 4); // = 7
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 4);
}

// =============================================================================
// Matrix Composition
// =============================================================================

TEST(GF2Matrix, ComposeIdentityLeft)
{
    auto id = make_gf2_identity<3>();
    auto swizzle = make_xor_swizzle<3>(0, 1);

    auto result = gf2_compose(id, swizzle);
    EXPECT_EQ(result, swizzle);
}

TEST(GF2Matrix, ComposeIdentityRight)
{
    auto id = make_gf2_identity<3>();
    auto swizzle = make_xor_swizzle<3>(0, 1);

    auto result = gf2_compose(swizzle, id);
    EXPECT_EQ(result, swizzle);
}

TEST(GF2Matrix, ComposeSelfInverse)
{
    auto swizzle = make_xor_swizzle_2d();
    auto composed = gf2_compose(swizzle, swizzle);
    auto id = make_gf2_identity<2>();

    EXPECT_EQ(composed, id);
}

TEST(GF2Matrix, ComposeAssociativity)
{
    // Create three different transforms
    auto a = make_xor_swizzle<3>(0, 1); // XOR dim 0 into dim 1
    auto b = make_xor_swizzle<3>(1, 2); // XOR dim 1 into dim 2
    auto c = make_xor_swizzle<3>(0, 2); // XOR dim 0 into dim 2

    auto ab = gf2_compose(a, b);
    auto bc = gf2_compose(b, c);
    auto ab_c = gf2_compose(ab, c);
    auto a_bc = gf2_compose(a, bc);

    // Verify via apply
    multi_index<3> in{3, 5, 7};
    EXPECT_EQ(ab_c.apply(in), a_bc.apply(in));
}

TEST(GF2Matrix, ComposeChainApply)
{
    // Composing A @ B should give same result as applying B then A
    auto a = make_xor_swizzle<3>(0, 1);
    auto b = make_xor_swizzle<3>(1, 2);

    auto composed = gf2_compose(a, b);

    multi_index<3> in{3, 5, 7};

    // Apply b then a
    auto temp = b.apply(in);
    auto sequential = a.apply(temp);

    // Apply composed
    auto direct = composed.apply(in);

    EXPECT_EQ(direct, sequential);
}

// =============================================================================
// Matrix Inversion
// =============================================================================

TEST(GF2Matrix, InverseIdentity)
{
    auto id = make_gf2_identity<3>();
    bool success;
    auto inv = gf2_inverse(id, success);

    EXPECT_TRUE(success);
    EXPECT_EQ(inv, id);
}

TEST(GF2Matrix, InverseXorSwizzle)
{
    auto swizzle = make_xor_swizzle_2d();
    bool success;
    auto inv = gf2_inverse(swizzle, success);

    EXPECT_TRUE(success);
    EXPECT_EQ(inv, swizzle); // Self-inverse
}

TEST(GF2Matrix, InverseVerifyCompose)
{
    auto m = make_xor_swizzle<3>(0, 2);
    bool success;
    auto inv = gf2_inverse(m, success);

    EXPECT_TRUE(success);

    // M @ M^-1 = I
    auto product = gf2_compose(m, inv);
    auto id = make_gf2_identity<3>();
    EXPECT_EQ(product, id);

    // M^-1 @ M = I
    auto product2 = gf2_compose(inv, m);
    EXPECT_EQ(product2, id);
}

TEST(GF2Matrix, InverseSingular)
{
    // Matrix with duplicate rows is singular
    gf2_matrix<2> m;
    m.set(0, 0, true);
    m.set(0, 1, true);
    m.set(1, 0, true);
    m.set(1, 1, true); // Same as row 0

    bool success;
    gf2_inverse(m, success);

    EXPECT_FALSE(success);
}

TEST(GF2Matrix, InverseZeroRow)
{
    gf2_matrix<3> m = make_gf2_identity<3>();
    // Make row 1 all zeros
    m.set(1, 1, false);

    bool success;
    gf2_inverse(m, success);

    EXPECT_FALSE(success);
}

TEST(GF2Matrix, InverseRoundTrip)
{
    // More complex invertible matrix
    gf2_matrix<3> m;
    m.set(0, 0, true);
    m.set(0, 2, true);  // out[0] = in[0] ^ in[2]
    m.set(1, 1, true);
    m.set(1, 2, true);  // out[1] = in[1] ^ in[2]
    m.set(2, 2, true);  // out[2] = in[2]

    bool success;
    auto inv = gf2_inverse(m, success);
    EXPECT_TRUE(success);

    // (M^-1)^-1 = M
    auto inv_inv = gf2_inverse(inv, success);
    EXPECT_TRUE(success);
    EXPECT_EQ(inv_inv, m);

    // Verify by applying
    multi_index<3> in{3, 5, 7};
    auto transformed = m.apply(in);
    auto restored = inv.apply(transformed);
    EXPECT_EQ(restored, in);
}

TEST(GF2Matrix, IsInvertible)
{
    EXPECT_TRUE(gf2_is_invertible(make_gf2_identity<4>()));
    EXPECT_TRUE(gf2_is_invertible(make_xor_swizzle_2d()));
    EXPECT_TRUE(gf2_is_invertible(make_xor_swizzle<3>(0, 1)));

    // Singular
    gf2_matrix<2> singular;
    singular.set(0, 0, true);
    // Row 1 is all zeros
    EXPECT_FALSE(gf2_is_invertible(singular));
}

TEST(GF2Matrix, IsSelfInverse)
{
    EXPECT_TRUE(gf2_is_self_inverse(make_gf2_identity<4>()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle_2d()));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<3>(0, 1)));
    EXPECT_TRUE(gf2_is_self_inverse(make_xor_swizzle<4>(2, 3)));

    // Cyclic permutation is NOT self-inverse
    gf2_matrix<3> cyclic;
    cyclic.set(0, 2, true); // out[0] = in[2]
    cyclic.set(1, 0, true); // out[1] = in[0]
    cyclic.set(2, 1, true); // out[2] = in[1]
    EXPECT_FALSE(gf2_is_self_inverse(cyclic));
}

// =============================================================================
// XOR Swizzle Specific Tests
// =============================================================================

TEST(GF2Swizzle, MatchesCKBehavior)
{
    // Verify XOR swizzle matches CK's xor_t: col' = col ^ row
    auto swizzle = make_xor_swizzle_2d();

    for(index_t row = 0; row < 16; ++row)
    {
        for(index_t col = 0; col < 16; ++col)
        {
            multi_index<2> in{row, col};
            auto out = swizzle.apply(in);

            EXPECT_EQ(out[0], row) << "row=" << row << " col=" << col;
            EXPECT_EQ(out[1], row ^ col) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2Swizzle, SelfInverseProperty)
{
    // Applying swizzle twice should return to original
    auto swizzle = make_xor_swizzle_2d();

    for(index_t row = 0; row < 16; ++row)
    {
        for(index_t col = 0; col < 16; ++col)
        {
            multi_index<2> original{row, col};
            auto swizzled = swizzle.apply(original);
            auto restored = swizzle.apply(swizzled);

            EXPECT_EQ(restored, original) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2Swizzle, BankConflictAvoidance)
{
    // Demonstrate how swizzle spreads bank access
    // Assume 32 banks, 4-byte words, col is the fast dimension

    auto swizzle = make_xor_swizzle_2d();

    // Without swizzle: threads at same column hit same bank
    // With swizzle: different rows access different banks

    auto get_bank = [](index_t col) { return col % 32; };

    // Check rows 0 and 1, column 0
    {
        multi_index<2> row0{0, 0};
        multi_index<2> row1{1, 0};

        // Without swizzle: both hit bank 0
        EXPECT_EQ(get_bank(row0[1]), get_bank(row1[1]));

        // With swizzle: different banks
        auto swiz0 = swizzle.apply(row0);
        auto swiz1 = swizzle.apply(row1);
        EXPECT_NE(get_bank(swiz0[1]), get_bank(swiz1[1]));
    }
}

// =============================================================================
// Compile-Time Evaluation
// =============================================================================

TEST(GF2CompileTime, ConstexprIdentity)
{
    constexpr auto id = make_gf2_identity<3>();
    static_assert(id.get(0, 0) == true);
    static_assert(id.get(0, 1) == false);
    static_assert(id.get(1, 1) == true);
    static_assert(id.get(2, 2) == true);
}

TEST(GF2CompileTime, ConstexprXorSwizzle)
{
    constexpr auto swizzle = make_xor_swizzle_2d();
    static_assert(swizzle.get(0, 0) == true);
    static_assert(swizzle.get(0, 1) == false);
    static_assert(swizzle.get(1, 0) == true);
    static_assert(swizzle.get(1, 1) == true);
}

TEST(GF2CompileTime, ConstexprCompose)
{
    constexpr auto swizzle = make_xor_swizzle_2d();
    constexpr auto composed = gf2_compose(swizzle, swizzle);
    constexpr auto id = make_gf2_identity<2>();
    static_assert(composed == id);
}

TEST(GF2CompileTime, ConstexprSelfInverse)
{
    constexpr auto swizzle = make_xor_swizzle_2d();
    static_assert(gf2_is_self_inverse(swizzle));
}

TEST(GF2CompileTime, ConstexprApply)
{
    constexpr auto swizzle = make_xor_swizzle_2d();
    constexpr multi_index<2> in{3, 5};
    constexpr auto out = swizzle.apply(in);
    static_assert(out[0] == 3);
    static_assert(out[1] == (3 ^ 5));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(GF2EdgeCases, Dimension1)
{
    auto id = make_gf2_identity<1>();
    EXPECT_TRUE(id.get(0, 0));

    multi_index<1> in{42};
    auto out = id.apply(in);
    EXPECT_EQ(out[0], 42);
}

TEST(GF2EdgeCases, LargeDimension)
{
    constexpr index_t N = 8;
    auto id = make_gf2_identity<N>();

    multi_index<N> in{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = id.apply(in);

    for(index_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(out[i], in[i]);
    }
}

TEST(GF2EdgeCases, ZeroInput)
{
    auto swizzle = make_xor_swizzle_2d();

    multi_index<2> in{0, 0};
    auto out = swizzle.apply(in);

    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
}

TEST(GF2EdgeCases, LargeValues)
{
    auto swizzle = make_xor_swizzle_2d();

    multi_index<2> in{1000, 2000};
    auto out = swizzle.apply(in);

    EXPECT_EQ(out[0], 1000);
    EXPECT_EQ(out[1], 1000 ^ 2000);
}
