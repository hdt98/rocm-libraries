// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck_tile/core/algorithm/gf2_linear_transform.hpp"

using namespace ck_tile;

// =============================================================================
// Part 1: Dimension-Level Tests
// =============================================================================

TEST(GF2DimMatrix, DefaultConstruction)
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

TEST(GF2DimMatrix, Identity)
{
    auto id = make_gf2_identity<3>();

    multi_index<3> in{5, 10, 15};
    auto out = id.apply(in);

    EXPECT_EQ(out[0], 5);
    EXPECT_EQ(out[1], 10);
    EXPECT_EQ(out[2], 15);
}

TEST(GF2DimMatrix, XorSwizzle2D)
{
    auto swizzle = make_xor_swizzle_2d();

    multi_index<2> in{3, 5};
    auto out = swizzle.apply(in);

    EXPECT_EQ(out[0], 3);     // row' = row
    EXPECT_EQ(out[1], 3 ^ 5); // col' = row ^ col = 6
}

TEST(GF2DimMatrix, SelfInverse)
{
    auto swizzle = make_xor_swizzle_2d();
    EXPECT_TRUE(gf2_is_self_inverse(swizzle));

    // Apply twice should return to original
    multi_index<2> original{7, 13};
    auto swizzled = swizzle.apply(original);
    auto restored = swizzle.apply(swizzled);
    EXPECT_EQ(restored, original);
}

TEST(GF2DimMatrix, Compose)
{
    auto a = make_xor_swizzle<3>(0, 1);
    auto b = make_xor_swizzle<3>(1, 2);
    auto composed = gf2_compose(a, b);

    multi_index<3> in{3, 5, 7};
    auto via_compose = composed.apply(in);

    auto temp = b.apply(in);
    auto via_chain = a.apply(temp);

    EXPECT_EQ(via_compose, via_chain);
}

TEST(GF2DimMatrix, Inverse)
{
    auto m = make_xor_swizzle<3>(0, 2);
    bool success;
    auto inv = gf2_inverse(m, success);

    EXPECT_TRUE(success);

    auto product = gf2_compose(m, inv);
    EXPECT_EQ(product, make_gf2_identity<3>());
}

// =============================================================================
// Part 2: Bit-Level Tests
// =============================================================================

TEST(GF2BitMatrix, DefaultConstruction)
{
    gf2_bit_matrix<8> m;
    for(index_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(m.rows[i], 0ULL);
    }
}

TEST(GF2BitMatrix, Identity)
{
    auto id = make_gf2_bit_identity<8>();

    for(uint64_t i = 0; i < 256; ++i)
    {
        EXPECT_EQ(id.apply(i), i);
    }
}

TEST(GF2BitMatrix, GetSet)
{
    gf2_bit_matrix<8> m;

    m.set(3, 5, true);
    EXPECT_TRUE(m.get(3, 5));
    EXPECT_FALSE(m.get(3, 4));
    EXPECT_FALSE(m.get(2, 5));

    m.set(3, 5, false);
    EXPECT_FALSE(m.get(3, 5));
}

TEST(GF2BitMatrix, SimpleXor)
{
    // out[0] = in[0] ^ in[1]
    // out[1] = in[1]
    gf2_bit_matrix<2> m;
    m.rows[0] = 0b11; // XOR bits 0 and 1
    m.rows[1] = 0b10; // Just bit 1

    EXPECT_EQ(m.apply(0b00), 0b00ULL);
    EXPECT_EQ(m.apply(0b01), 0b01ULL); // in[0]=1,in[1]=0 -> out[0]=1^0=1, out[1]=0
    EXPECT_EQ(m.apply(0b10), 0b11ULL); // in[0]=0,in[1]=1 -> out[0]=0^1=1, out[1]=1
    EXPECT_EQ(m.apply(0b11), 0b10ULL); // in[0]=1,in[1]=1 -> out[0]=1^1=0, out[1]=1
}

TEST(GF2BitMatrix, XorSwizzleBits)
{
    // 3-bit row, 3-bit col
    auto swizzle = make_xor_swizzle_bits<3, 3>();

    // Input: [row=3, col=5] packed as 0b101_011 = row in bits 0-2, col in bits 3-5
    // row = 011 = 3, col = 101 = 5
    uint64_t input = 3 | (5 << 3);

    // Expected: row' = row = 3, col' = col ^ row = 5 ^ 3 = 6
    uint64_t expected = 3 | (6 << 3);

    EXPECT_EQ(swizzle.apply(input), expected);
}

TEST(GF2BitMatrix, SelfInverse)
{
    auto swizzle = make_xor_swizzle_bits<4, 4>();
    EXPECT_TRUE(gf2_bit_is_self_inverse(swizzle));
}

TEST(GF2BitMatrix, Compose)
{
    auto id = make_gf2_bit_identity<8>();
    auto swizzle = make_xor_swizzle_bits<4, 4>();

    auto left = gf2_bit_compose(id, swizzle);
    auto right = gf2_bit_compose(swizzle, id);

    EXPECT_EQ(left, swizzle);
    EXPECT_EQ(right, swizzle);
}

TEST(GF2BitMatrix, Inverse)
{
    auto swizzle = make_xor_swizzle_bits<3, 3>();
    bool success;
    auto inv = gf2_bit_inverse(swizzle, success);

    EXPECT_TRUE(success);
    EXPECT_EQ(inv, swizzle); // Self-inverse

    auto product = gf2_bit_compose(swizzle, inv);
    EXPECT_EQ(product, make_gf2_bit_identity<6>());
}

// =============================================================================
// Part 3: Bit-Level Transform with Multi-Index Interface
// =============================================================================

TEST(GF2BitTransform, XorSwizzle)
{
    auto transform = make_bit_xor_swizzle<6, 7>(); // 64 rows, 128 cols

    for(index_t row = 0; row < 64; ++row)
    {
        for(index_t col = 0; col < 128; ++col)
        {
            multi_index<2> in{row, col};
            auto out = transform.apply(in);

            EXPECT_EQ(out[0], row);
            // col' = col ^ row, but only lower 6 bits of row (since ColBits > RowBits after bit 5)
            index_t expected_col = col ^ row;
            EXPECT_EQ(out[1], expected_col) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2BitTransform, SelfInverse)
{
    auto transform = make_bit_xor_swizzle<6, 7>();

    for(index_t row = 0; row < 64; row += 7)
    {
        for(index_t col = 0; col < 128; col += 11)
        {
            multi_index<2> original{row, col};
            auto swizzled = transform.apply(original);
            auto restored = transform.apply(swizzled);

            EXPECT_EQ(restored, original) << "row=" << row << " col=" << col;
        }
    }
}

// =============================================================================
// Part 4: Custom Swizzle for Bank Conflict Avoidance
// =============================================================================

TEST(GF2CustomSwizzle, HighBitSwizzle)
{
    // For 4x1 wave layout problem: rows 0, 32, 64, 96 need different banks
    // These rows differ in bits 5 and 6
    // XOR row bits 5,6 into col bits 3,4 to spread bank access

    constexpr index_t RowBits = 7; // rows 0-127
    constexpr index_t ColBits = 7; // cols 0-127

    array<index_t, 2> row_bits{5, 6};
    array<index_t, 2> col_bits{3, 4};

    auto swizzle = make_custom_bit_swizzle<RowBits, ColBits, 2>(row_bits, col_bits);

    // Check that rows 0, 32, 64, 96 at same col get different swizzled cols
    index_t col = 0;

    multi_index<2> r0{0, col};
    multi_index<2> r32{32, col};
    multi_index<2> r64{64, col};
    multi_index<2> r96{96, col};

    auto s0 = swizzle.apply(r0);
    auto s32 = swizzle.apply(r32);
    auto s64 = swizzle.apply(r64);
    auto s96 = swizzle.apply(r96);

    // Row 0:  bits 5,6 = 0,0 -> col XOR 0 = col
    // Row 32: bits 5,6 = 1,0 -> col XOR (1<<3) = col ^ 8
    // Row 64: bits 5,6 = 0,1 -> col XOR (1<<4) = col ^ 16
    // Row 96: bits 5,6 = 1,1 -> col XOR (3<<3) = col ^ 24

    EXPECT_EQ(s0[1], col);
    EXPECT_EQ(s32[1], col ^ 8);
    EXPECT_EQ(s64[1], col ^ 16);
    EXPECT_EQ(s96[1], col ^ 24);

    // All different!
    EXPECT_NE(s0[1], s32[1]);
    EXPECT_NE(s0[1], s64[1]);
    EXPECT_NE(s0[1], s96[1]);
    EXPECT_NE(s32[1], s64[1]);
    EXPECT_NE(s32[1], s96[1]);
    EXPECT_NE(s64[1], s96[1]);
}

TEST(GF2CustomSwizzle, SelfInverse)
{
    array<index_t, 2> row_bits{5, 6};
    array<index_t, 2> col_bits{3, 4};

    auto swizzle = make_custom_bit_swizzle<7, 7, 2>(row_bits, col_bits);

    // Custom swizzles are also self-inverse
    for(index_t row = 0; row < 128; row += 13)
    {
        for(index_t col = 0; col < 128; col += 17)
        {
            multi_index<2> original{row, col};
            auto swizzled = swizzle.apply(original);
            auto restored = swizzle.apply(swizzled);

            EXPECT_EQ(restored, original) << "row=" << row << " col=" << col;
        }
    }
}

TEST(GF2CustomSwizzle, BankMapping)
{
    // Verify that the swizzle spreads bank access for 32-bank LDS
    // Bank = (byte_address / 4) % 32 = (col * 2 / 4) % 32 = (col / 2) % 32 for fp16

    array<index_t, 2> row_bits{5, 6};
    array<index_t, 2> col_bits{3, 4};

    auto swizzle = make_custom_bit_swizzle<7, 7, 2>(row_bits, col_bits);

    auto get_bank = [](index_t col) { return (col / 2) % 32; };

    // Without swizzle: rows 0, 32, 64, 96 at col 0 all hit bank 0
    EXPECT_EQ(get_bank(0), get_bank(0)); // trivially same

    // With swizzle: they should hit different banks
    multi_index<2> r0{0, 0};
    multi_index<2> r32{32, 0};
    multi_index<2> r64{64, 0};
    multi_index<2> r96{96, 0};

    auto s0 = swizzle.apply(r0);
    auto s32 = swizzle.apply(r32);
    auto s64 = swizzle.apply(r64);
    auto s96 = swizzle.apply(r96);

    index_t bank0 = get_bank(s0[1]);
    index_t bank32 = get_bank(s32[1]);
    index_t bank64 = get_bank(s64[1]);
    index_t bank96 = get_bank(s96[1]);

    // All different banks
    EXPECT_NE(bank0, bank32);
    EXPECT_NE(bank0, bank64);
    EXPECT_NE(bank0, bank96);
    EXPECT_NE(bank32, bank64);
    EXPECT_NE(bank32, bank96);
    EXPECT_NE(bank64, bank96);
}

// =============================================================================
// Part 5: Equivalence Tests
// =============================================================================

TEST(GF2Equivalence, BitLevelMatchesDimensionLevel)
{
    // Bit-level with block-diagonal structure should match dimension-level
    auto dim_swizzle = make_xor_swizzle_2d();
    auto bit_swizzle = make_bit_xor_swizzle<6, 6>(); // Same bits per dim

    for(index_t row = 0; row < 64; ++row)
    {
        for(index_t col = 0; col < 64; ++col)
        {
            multi_index<2> in{row, col};

            auto dim_out = dim_swizzle.apply(in);
            auto bit_out = bit_swizzle.apply(in);

            EXPECT_EQ(dim_out, bit_out) << "row=" << row << " col=" << col;
        }
    }
}

// =============================================================================
// Part 6: Coordinate Packer Tests
// =============================================================================

TEST(CoordinatePacker, PackUnpack)
{
    using Packer = coordinate_packer<6, 7>; // 6-bit row, 7-bit col

    for(index_t row = 0; row < 64; ++row)
    {
        for(index_t col = 0; col < 128; ++col)
        {
            multi_index<2> in{row, col};
            uint64_t packed = Packer::pack(in);
            auto unpacked = Packer::unpack(packed);

            EXPECT_EQ(unpacked, in);
        }
    }
}

TEST(CoordinatePacker, BitLayout)
{
    using Packer = coordinate_packer<3, 4>;

    // dim 0 in bits 0-2, dim 1 in bits 3-6
    multi_index<2> in{0b111, 0b1111};
    uint64_t packed = Packer::pack(in);

    EXPECT_EQ(packed, 0b1111111ULL);
    EXPECT_EQ(packed & 0b111, 0b111ULL);        // row
    EXPECT_EQ((packed >> 3) & 0b1111, 0b1111ULL); // col
}

// =============================================================================
// Part 7: Edge Case Tests
// =============================================================================

TEST(GF2BitMatrix, SingularMatrixInversionFails)
{
    gf2_bit_matrix<4> singular;
    singular.rows[0] = 0b0011;
    singular.rows[1] = 0b0011;  // Duplicate row = singular matrix
    singular.rows[2] = 0b0100;
    singular.rows[3] = 0b1000;

    bool success;
    gf2_bit_inverse(singular, success);
    EXPECT_FALSE(success);
}

TEST(GF2DimMatrix, SingularMatrixInversionFails)
{
    gf2_matrix<2> singular;
    singular.set(0, 0, true);
    singular.set(0, 1, true);
    singular.set(1, 0, true);
    singular.set(1, 1, true);  // Rows are identical = singular

    bool success;
    gf2_inverse(singular, success);
    EXPECT_FALSE(success);
}

TEST(GF2DimMatrix, Dimension1)
{
    auto id = make_gf2_identity<1>();
    EXPECT_TRUE(id.get(0, 0));

    multi_index<1> in{42};
    auto out = id.apply(in);
    EXPECT_EQ(out[0], 42);

    EXPECT_TRUE(gf2_is_invertible(id));
    EXPECT_TRUE(gf2_is_self_inverse(id));
}

TEST(GF2BitMatrix, Bits1)
{
    auto id = make_gf2_bit_identity<1>();
    EXPECT_EQ(id.apply(0), 0ULL);
    EXPECT_EQ(id.apply(1), 1ULL);
}

TEST(GF2DimMatrix, ZeroMatrixNotInvertible)
{
    gf2_matrix<2> zero;  // Default = all false
    EXPECT_FALSE(gf2_is_invertible(zero));
}

// =============================================================================
// Part 8: Compile-Time Tests
// =============================================================================

TEST(GF2CompileTime, DimLevel)
{
    constexpr auto id = make_gf2_identity<3>();
    static_assert(id.get(0, 0) == true);
    static_assert(id.get(0, 1) == false);

    constexpr auto swizzle = make_xor_swizzle_2d();
    static_assert(gf2_is_self_inverse(swizzle));
}

TEST(GF2CompileTime, BitLevel)
{
    constexpr auto id = make_gf2_bit_identity<8>();
    static_assert(id.rows[0] == 0b00000001);
    static_assert(id.rows[7] == 0b10000000);

    constexpr auto swizzle = make_xor_swizzle_bits<3, 3>();
    static_assert(gf2_bit_is_self_inverse(swizzle));
}

TEST(GF2CompileTime, CustomSwizzle)
{
    constexpr array<index_t, 2> row_bits{5, 6};
    constexpr array<index_t, 2> col_bits{3, 4};
    constexpr auto swizzle = make_custom_swizzle_bits<7, 7, 2>(row_bits, col_bits);

    // Verify it's invertible
    static_assert(gf2_bit_is_invertible(swizzle));
}
