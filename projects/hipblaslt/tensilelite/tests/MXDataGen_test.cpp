// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <mxDataGenerator/mxDataGen.hpp>

#include <cstdint>
#include <vector>

/**
 * @brief Returns true if a 4-bit FP4 E2M1 nibble represents zero.
 *
 * FP4 E2M1 values are packed two-per-byte (low nibble first).
 * Both 0x0 (+0) and 0x8 (-0) decode to zero.
 */
static bool isZeroNibble(uint8_t nibble)
{
    // FP4 E2M1: 0x0 = +0.0, 0x8 = -0.0
    return (nibble == 0x0) || (nibble == 0x8);
}

/**
 * @brief Count elements that decode to zero in a packed FP4 buffer.
 */
static size_t countZerosFP4(const uint8_t* packedData, size_t numPackedBytes)
{
    size_t zeros = 0;
    for(size_t i = 0; i < numPackedBytes; ++i)
    {
        uint8_t lo = packedData[i] & 0x0F;
        uint8_t hi = (packedData[i] >> 4) & 0x0F;
        if(isZeroNibble(lo))
            ++zeros;
        if(isZeroNibble(hi))
            ++zeros;
    }
    return zeros;
}

class MXDataGenFP4Test : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool>>
{
};

/**
 * @brief Verify that generateMXInput produces FP4 data with an acceptable zero frequency.
 *
 * FP4 E2M1 has 16 nibble values, 2 of which are zero (0x0 = +0, 0x8 = -0), giving a
 * naive baseline of 2/16 = 12.5%. MX block scaling slightly elevates this: the block
 * maximum is guaranteed non-zero, pushing small elements toward zero. Empirically the
 * zero frequency converges to ~12.89% for large matrices with bounded [-1, 1] input.
 */
TEST_P(MXDataGenFP4Test, ZeroFrequencyWithinBounds)
{
    auto [rows, cols, mxBlock, isTranspose] = GetParam();

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataBuffer(numPacked, 0);
    std::vector<uint8_t> scaleBuffer(numScales, 0);

    std::vector<size_t> emptySwizzle;
    std::vector<size_t> emptyTile;

    DGen::generateMXInput(DGen::DataFormat::Fp4,
                    DGen::ScaleType::E8M0,
                    dataBuffer.data(),
                    scaleBuffer.data(),
                    rows,
                    cols,
                    rows, // stride = rows (column-major)
                    isTranspose,
                    emptySwizzle,
                    emptyTile,
                    mxBlock,
                    1,
                    true,
                    "Bounded",
                    -1.0f,
                    1.0f);

    size_t zeros       = countZerosFP4(dataBuffer.data(), numPacked);
    double zeroPercent = 100.0 * static_cast<double>(zeros) / static_cast<double>(numElements);

    EXPECT_LT(zeroPercent, 13.0)
        << "Zero frequency " << zeroPercent << "% exceeds 13% upper bound for "
        << rows << "x" << cols << " FP4 matrix (transpose=" << isTranspose << ")";

    // Ensure non-trivial data was actually generated (not all zeros)
    EXPECT_GT(numElements - zeros, 0u)
        << "All elements are zero for " << rows << "x" << cols << " FP4 matrix";
}

INSTANTIATE_TEST_SUITE_P(
    FP4ZeroFrequency,
    MXDataGenFP4Test,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose
        std::make_tuple(128u,  128u,  32, true),
        std::make_tuple(256u,  256u,  32, true),
        std::make_tuple(2048u, 1026u, 32, true),
        std::make_tuple(2048u, 514u,  32, false)
    )
);

/**
 * @brief Regression guard: generateMXInput must be deterministic (fixed seed).
 *
 * Any post-generation overwrite of the MXSA/MXSB buffers (e.g., the general
 * tensor-init loop in initializeCPUInputs) desynchronises the CPU reference
 * from GPU data, causing intermittent single-element validation failures.
 * rows=K (must be mxBlock-aligned), cols=M/N (need not be).
 */
class MXGeneratorDeterminismTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

TEST_P(MXGeneratorDeterminismTest, GeneratorOutputIsDeterministic)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const size_t numPacked = (rows * cols + 1) / 2;
    const size_t numScales = (rows / mxBlock) * cols;

    std::vector<uint8_t> data1(numPacked);
    std::vector<uint8_t> data2(numPacked);
    std::vector<uint8_t> scale1(numScales, 0x00);
    std::vector<uint8_t> scale2(numScales, 0xFF); // sentinel: catches no-write if scale1==scale2 passes

    std::vector<size_t> emptySwizzle, emptyTile;

    DGen::generateMXInput(DGen::DataFormat::Fp4,
                    DGen::ScaleType::E8M0,
                    data1.data(), scale1.data(),
                    rows, cols, rows, isTranspose,
                    emptySwizzle, emptyTile,
                    mxBlock, 1, isMatrixA, "Bounded", -1.f, 1.f);

    DGen::generateMXInput(DGen::DataFormat::Fp4,
                    DGen::ScaleType::E8M0,
                    data2.data(), scale2.data(),
                    rows, cols, rows, isTranspose,
                    emptySwizzle, emptyTile,
                    mxBlock, 1, isMatrixA, "Bounded", -1.f, 1.f);

    EXPECT_EQ(data1, data2)
        << "FP4 data is non-deterministic";
    EXPECT_EQ(scale1, scale2)
        << "Scale data is non-deterministic; any post-generation overwrite will corrupt validation";

    bool allZero = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0; });
    bool allOnes = std::all_of(scale1.begin(), scale1.end(), [](uint8_t b){ return b == 0xFF; });
    EXPECT_FALSE(allZero) << "Scale buffer is all-zero — generator did not write";
    EXPECT_FALSE(allOnes) << "Scale buffer is all-0xFF (max UE8M0 value) — generator likely failed; bounded [-1,1] input should produce varied scales";
}

INSTANTIATE_TEST_SUITE_P(
    GeneratorDeterminism,
    MXGeneratorDeterminismTest,
    ::testing::Values(
        // rows=K, cols=M or N  (tensorA.sizes()={K,M}, tensorB.sizes()={K,N})
        std::make_tuple(1024u, 128u, 32, true,  true),  // transposed A
        std::make_tuple(1024u, 128u, 32, false, false), // non-transposed B
        std::make_tuple(1024u, 204u, 32, true,  true),  // M=204, non-32-aligned (was failing)
        std::make_tuple(1024u, 213u, 32, true,  true)   // M=213, non-32-aligned (was failing)
    )
);

// ============================================================================
// PreSwizzle scale tests
//
// Verify generateMXInput with preSwizzle produces scale data that is a
// permutation of the unswizzled layout. gfx950 FP4 MX kernels expect:
//   preSwizzle = {swizzleTileMN=32, tileK=8, subTileK=MiK/mxBlock}
//   preTile    = {tileK=8, swizzleTileMN=32}
// swizzleTileMN=32 is fixed (2 SIMDs * 16 lanes); subTileK=4 for MiK=128, mxBlock=32.
// ============================================================================

// Params: {rows, cols, mxBlock, isTranspose, isMatrixA}
class MXPreSwizzleTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, int, bool, bool>>
{
};

/** @brief Verify preSwizzle produces a non-trivial permutation of scale data. */
TEST_P(MXPreSwizzleTest, ScaleIsPermutationOfUnswizzled)
{
    auto [rows, cols, mxBlock, isTranspose, isMatrixA] = GetParam();

    const std::vector<size_t> preSwizzle = {32, 8, 4};
    const std::vector<size_t> preTile    = {8, 32};

    const uint64_t numElements  = rows * cols;
    const uint64_t numPacked    = (numElements + 1) / 2;
    const size_t   numScales    = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> dataNoShuf(numPacked, 0);
    std::vector<uint8_t> scaleNoShuf(numScales, 0);
    std::vector<uint8_t> dataShuf(numPacked, 0);
    std::vector<uint8_t> scaleShuf(numScales, 0);

    // Generate without preSwizzle
    DGen::generateMXInput(DGen::DataFormat::Fp4,
                    DGen::ScaleType::E8M0,
                    dataNoShuf.data(),
                    scaleNoShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    {}, {},
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // Generate with preSwizzle
    DGen::generateMXInput(DGen::DataFormat::Fp4,
                    DGen::ScaleType::E8M0,
                    dataShuf.data(),
                    scaleShuf.data(),
                    rows, cols, rows,
                    isTranspose,
                    preSwizzle, preTile,
                    mxBlock, 1, isMatrixA,
                    "Bounded", -1.0f, 1.0f);

    // The scale buffers must be different
    EXPECT_NE(scaleNoShuf, scaleShuf)
        << "Scale data was not shuffled for " << rows << "x" << cols
        << " (transpose=" << isTranspose << ", isMatrixA=" << isMatrixA << ")";

    // The shuffled scale must be a permutation: same multiset of bytes
    std::vector<uint8_t> sortedNoShuf = scaleNoShuf;
    std::vector<uint8_t> sortedShuf   = scaleShuf;
    std::sort(sortedNoShuf.begin(), sortedNoShuf.end());
    std::sort(sortedShuf.begin(), sortedShuf.end());
    EXPECT_EQ(sortedNoShuf, sortedShuf)
        << "Pre-shuffled scale is not a permutation of the unshuffled scale for "
        << rows << "x" << cols;

    // Data buffer must be identical (preSwizzle only affects scale, not data)
    EXPECT_EQ(dataNoShuf, dataShuf)
        << "Data buffer changed unexpectedly with preSwizzle for "
        << rows << "x" << cols;
}

INSTANTIATE_TEST_SUITE_P(
    FP4PreSwizzle,
    MXPreSwizzleTest,
    ::testing::Values(
        // rows, cols, mxBlock, isTranspose, isMatrixA
        // Test size constraints for preSwizzle {32,8,4} + preTile {8,32}:
        //   rows % 256 == 0  (scaleRows = rows/mxBlock must be divisible by tileK=8)
        //   cols % 32  == 0  (scaleCols must be divisible by swizzleTileMN=32)        std::make_tuple(256u,  256u,  32, true,  true),   // scale A transposed
        std::make_tuple(256u,  256u,  32, false, false),  // scale B non-transposed
        std::make_tuple(512u,  256u,  32, true,  true),   // larger scale A
        std::make_tuple(256u,  512u,  32, false, false),  // larger scale B
        std::make_tuple(4096u, 16384u, 32, true, true)    // benchmark-scale problem
    )
);

// ============================================================================
// FP6 and FP8 zero-frequency tests
//
// FP6 (E2M3 / E3M2) has 6 bits, of which 0x00 / 0x20 (positive / negative
// zero) are the only zero values, giving 2/64 = 3.125% baseline. FP8 (E4M3 /
// E5M2) has 256 distinct values with a single positive zero (and a negative
// zero), giving 2/256 = 0.78% baseline. With block-aware MX scaling on
// uniform [-1, 1] input the empirical zero rates stay close to those
// baselines (no rounding-to-zero collapse).
// ============================================================================

namespace
{
    bool isZeroFP6(uint8_t v)
    {
        v &= 0x3f;
        return v == 0x00u || v == 0x20u;
    }

    size_t countZerosFP6(uint8_t const* packed, size_t numElements)
    {
        size_t zeros = 0;
        for(size_t i = 0; i < numElements; ++i)
        {
            size_t   bit  = i * 6;
            size_t   byte = bit / 8;
            int      shift = bit % 8;
            uint16_t word  = packed[byte];
            if((bit + 6 + 7) / 8 > byte)
                word |= static_cast<uint16_t>(packed[byte + 1]) << 8;
            uint8_t v = static_cast<uint8_t>((word >> shift) & 0x3f);
            if(isZeroFP6(v))
                ++zeros;
        }
        return zeros;
    }

    size_t countZerosFP8(uint8_t const* data, size_t numElements)
    {
        size_t zeros = 0;
        for(size_t i = 0; i < numElements; ++i)
            if(data[i] == 0x00u || data[i] == 0x80u)
                ++zeros;
        return zeros;
    }
} // namespace

TEST(MXDataGenFP6Test, ZeroFrequencyE2M3WithinBounds)
{
    constexpr uint64_t rows = 1024, cols = 1024;
    constexpr int      mxBlock = 32;
    size_t const       packedBytes = (rows * cols * 6 + 7) / 8;
    size_t const       scaleBytes  = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> data(packedBytes, 0);
    std::vector<uint8_t> scale(scaleBytes, 0);
    DGen::generateMXInput(DGen::DataFormat::Fp6E2M3,
                          DGen::ScaleType::E8M0,
                          data.data(), scale.data(),
                          rows, cols, rows, true,
                          {}, {},
                          mxBlock, 1, true,
                          "Bounded", -1.0f, 1.0f);

    size_t zeros = countZerosFP6(data.data(), rows * cols);
    double frac  = static_cast<double>(zeros) / static_cast<double>(rows * cols);
    EXPECT_LT(frac, 0.10) << "FP6 E2M3 zero rate " << frac
                          << " should stay well under the FP4 ~13% baseline";
}

TEST(MXDataGenFP6Test, ZeroFrequencyE3M2WithinBounds)
{
    constexpr uint64_t rows = 1024, cols = 1024;
    constexpr int      mxBlock = 32;
    size_t const       packedBytes = (rows * cols * 6 + 7) / 8;
    size_t const       scaleBytes  = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> data(packedBytes, 0);
    std::vector<uint8_t> scale(scaleBytes, 0);
    DGen::generateMXInput(DGen::DataFormat::Fp6E3M2,
                          DGen::ScaleType::E8M0,
                          data.data(), scale.data(),
                          rows, cols, rows, true,
                          {}, {},
                          mxBlock, 1, true,
                          "Bounded", -1.0f, 1.0f);

    size_t zeros = countZerosFP6(data.data(), rows * cols);
    double frac  = static_cast<double>(zeros) / static_cast<double>(rows * cols);
    EXPECT_LT(frac, 0.10);
}

TEST(MXDataGenFP8Test, ZeroFrequencyE4M3WithinBounds)
{
    constexpr uint64_t rows = 512, cols = 512;
    constexpr int      mxBlock = 32;
    size_t const       dataBytes  = rows * cols;
    size_t const       scaleBytes = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> data(dataBytes, 0);
    std::vector<uint8_t> scale(scaleBytes, 0);
    DGen::generateMXInput(DGen::DataFormat::Fp8E4M3,
                          DGen::ScaleType::E8M0,
                          data.data(), scale.data(),
                          rows, cols, rows, true,
                          {}, {},
                          mxBlock, 1, true,
                          "Bounded", -1.0f, 1.0f);

    size_t zeros = countZerosFP8(data.data(), dataBytes);
    double frac  = static_cast<double>(zeros) / static_cast<double>(dataBytes);
    EXPECT_LT(frac, 0.05) << "FP8 E4M3 zero rate " << frac
                          << " should be very low (<5%)";
}

TEST(MXDataGenFP8Test, ZeroFrequencyE5M2WithinBounds)
{
    constexpr uint64_t rows = 512, cols = 512;
    constexpr int      mxBlock = 32;
    size_t const       dataBytes  = rows * cols;
    size_t const       scaleBytes = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> data(dataBytes, 0);
    std::vector<uint8_t> scale(scaleBytes, 0);
    DGen::generateMXInput(DGen::DataFormat::Fp8E5M2,
                          DGen::ScaleType::E8M0,
                          data.data(), scale.data(),
                          rows, cols, rows, true,
                          {}, {},
                          mxBlock, 1, true,
                          "Bounded", -1.0f, 1.0f);

    size_t zeros = countZerosFP8(data.data(), dataBytes);
    double frac  = static_cast<double>(zeros) / static_cast<double>(dataBytes);
    EXPECT_LT(frac, 0.05);
}

// ============================================================================
// `hpl` init mode now maps to Bounded(-0.5, 0.5) instead of the legacy
// TrigonometricFromFloat fallback. Sanity-check that the FP4 zero rate stays
// in the same ballpark as Bounded(-1, 1).
// ============================================================================
TEST(MXDataGenFP4Test, HplModeMapsToBounded)
{
    constexpr uint64_t rows = 1024, cols = 1024;
    constexpr int      mxBlock = 32;
    uint64_t const     numPacked = (rows * cols + 1) / 2;
    size_t const       numScales = ((rows + mxBlock - 1) / mxBlock) * cols;

    std::vector<uint8_t> data(numPacked, 0);
    std::vector<uint8_t> scale(numScales, 0);
    DGen::generateMXInput(DGen::DataFormat::Fp4,
                          DGen::ScaleType::E8M0,
                          data.data(), scale.data(),
                          rows, cols, rows, true,
                          {}, {},
                          mxBlock, 1, true,
                          "hpl");

    size_t zeros = countZerosFP4(data.data(), numPacked);
    double frac  = static_cast<double>(zeros) / static_cast<double>(rows * cols);
    EXPECT_LT(frac, 0.20) << "hpl mode should not collapse FP4 to ~50% zeros";
}

// ============================================================================
// gfx1250-style: empty preSwizzle / preTile must leave scales canonical, and
// generation must succeed for all 5 MX dtype variants.
// ============================================================================
TEST(MXDataGenGfx1250, NoSwizzleAllDtypes)
{
    struct Case
    {
        DGen::DataFormat fmt;
        size_t           bitsPerElement;
    };
    Case const cases[] = {
        {DGen::DataFormat::Fp4,     4},
        {DGen::DataFormat::Fp6E2M3, 6},
        {DGen::DataFormat::Fp6E3M2, 6},
        {DGen::DataFormat::Fp8E4M3, 8},
        {DGen::DataFormat::Fp8E5M2, 8},
    };
    constexpr uint64_t rows = 256, cols = 32;
    constexpr int      mxBlock = 32;
    size_t const       numScales = (rows / mxBlock) * cols;

    for(auto const& c : cases)
    {
        size_t dataBytes = (rows * cols * c.bitsPerElement + 7) / 8;
        std::vector<uint8_t> data(dataBytes, 0);
        std::vector<uint8_t> scale(numScales, 0);
        ASSERT_NO_THROW(DGen::generateMXInput(c.fmt,
                                              DGen::ScaleType::E8M0,
                                              data.data(), scale.data(),
                                              rows, cols, rows, true,
                                              {}, {},
                                              mxBlock, 1, true,
                                              "Bounded", -1.0f, 1.0f))
            << "Generation failed for fmt " << static_cast<int>(c.fmt);

        bool anyNonZero = false;
        for(uint8_t b : scale)
            if(b != 0)
            {
                anyNonZero = true;
                break;
            }
        EXPECT_TRUE(anyNonZero) << "All scales are zero for fmt " << static_cast<int>(c.fmt);
    }
}
