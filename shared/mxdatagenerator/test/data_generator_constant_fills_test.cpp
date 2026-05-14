// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Focused tests for the constant-fill and pathological DataInitMode variants
// added on top of the existing Bounded/Ones/Zeros/etc. set:
//
//   - Twos       : every dequantized element == 2.0
//   - NegOnes    : every dequantized element == -1.0
//   - MaxVals    : every dequantized element == DTYPE::dataMaxNormalNumber
//   - DenormMins : every dequantized element == smallest non-zero subnormal
//   - DenormMaxs : every dequantized element == largest subnormal
//   - NaNs       : every dequantized element is NaN
//   - Infs       : every dequantized element is +Inf
//                  (only valid when DTYPE::dataInfo.hasInf is true; otherwise
//                  the generator must throw)
//
// We intentionally avoid the full TYPED_TEST_SUITE machinery the legacy tests
// use because each new mode has data-type-specific expectations (the smallest
// representable subnormal differs per DTYPE, NaN handling routes through the
// scale for F4/F6, Inf is unrepresentable for non-E5M2 data, etc.).  A small
// hand-rolled per-DTYPE matrix is clearer than templating around all of that.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/bf16.hpp>
#include <mxDataGenerator/f32.hpp>
#include <mxDataGenerator/fp16.hpp>

using namespace DGen;

namespace
{
    // 32-element fastest-varying dim matches the canonical MX block size
    // (mxBlock = 32 for f8/f6, 32 for f4 too in the gfx950/1250 pipelines),
    // so blockScaling = 32 keeps the test honest about scale broadcasting.
    constexpr index_t kBlockScaling = 32;
    constexpr index_t kRows         = 32;
    constexpr index_t kCols         = 4;

    template <typename DTYPE>
    DataGenerator<DTYPE> generateConstant(DataInitMode initMode)
    {
        DataGeneratorOptions opts;
        opts.blockScaling = kBlockScaling;
        opts.initMode     = initMode;
        opts.forceDenorm  = false;
        // Use a column-major 2D shape so size/stride match the
        // "{row, col, rowStride=1, colStride=row}" layout the rest of
        // the suite uses.
        std::vector<index_t> sizes{kRows, kCols};
        std::vector<index_t> strides{1, kRows};
        DataGenerator<DTYPE> dgen;
        dgen.generate(sizes, strides, opts);
        return dgen;
    }

    template <typename DTYPE>
    void expectAllValues(DataInitMode initMode, float expected)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_EQ(ref[i], expected) << "element " << i;
        }
    }

    template <typename DTYPE>
    void expectAllNaN(DataInitMode initMode)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_TRUE(std::isnan(ref[i])) << "element " << i << " = " << ref[i];
        }
    }

    template <typename DTYPE>
    void expectAllInf(DataInitMode initMode)
    {
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_TRUE(std::isinf(ref[i]) && ref[i] > 0)
                << "element " << i << " = " << ref[i];
        }
    }
} // namespace

// ---------------------------------------------------------------------------
// Twos: dequantized == 2.0 for every supported DTYPE.
// ---------------------------------------------------------------------------
TEST(DataGeneratorConstantFills, TwosF4_E8M0)
{
    expectAllValues<ocp_e2m1_mxfp4>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF4_E4M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e4m3>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF4_E5M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e5m3>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF6_E2M3)
{
    expectAllValues<ocp_e2m3_mxfp6>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF6_E3M2)
{
    expectAllValues<ocp_e3m2_mxfp6>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF8_E4M3)
{
    expectAllValues<ocp_e4m3_mxfp8>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF8_E5M2)
{
    expectAllValues<ocp_e5m2_mxfp8>(Twos{}, 2.0f);
}

// ---------------------------------------------------------------------------
// NegOnes: dequantized == -1.0 for every supported DTYPE.  Valid for every
// data type since each has a sign bit.
// ---------------------------------------------------------------------------
TEST(DataGeneratorConstantFills, NegOnesF4_E8M0)
{
    expectAllValues<ocp_e2m1_mxfp4>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF4_E4M3)
{
    expectAllValues<ocp_e2m1_mxfp4_e4m3>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF6_E2M3)
{
    expectAllValues<ocp_e2m3_mxfp6>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF6_E3M2)
{
    expectAllValues<ocp_e3m2_mxfp6>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF8_E4M3)
{
    expectAllValues<ocp_e4m3_mxfp8>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF8_E5M2)
{
    expectAllValues<ocp_e5m2_mxfp8>(NegOnes{}, -1.0f);
}

// ---------------------------------------------------------------------------
// MaxVals: dequantized == data type's max normal number.
// ---------------------------------------------------------------------------
TEST(DataGeneratorConstantFills, MaxValsF4)
{
    // F4 max representable normal = 6.0
    expectAllValues<ocp_e2m1_mxfp4>(MaxVals{}, 6.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF6_E2M3)
{
    // F6 E2M3 max normal = 7.5
    expectAllValues<ocp_e2m3_mxfp6>(MaxVals{}, 7.5f);
}
TEST(DataGeneratorConstantFills, MaxValsF6_E3M2)
{
    // F6 E3M2 max normal = 28.0
    expectAllValues<ocp_e3m2_mxfp6>(MaxVals{}, 28.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF8_E4M3)
{
    // F8 E4M3 max normal = 448.0
    expectAllValues<ocp_e4m3_mxfp8>(MaxVals{}, 448.0f);
}
TEST(DataGeneratorConstantFills, MaxValsF8_E5M2)
{
    // F8 E5M2 max normal = 57344.0
    expectAllValues<ocp_e5m2_mxfp8>(MaxVals{}, 57344.0f);
}

// ---------------------------------------------------------------------------
// DenormMins: dequantized == smallest non-zero subnormal for the data type.
// We don't pin the exact float here -- different DTYPEs have different
// subnormal-min magnitudes -- but every element must be the same nonzero
// positive subnormal.
// ---------------------------------------------------------------------------
template <typename DTYPE>
void expectDenormMinAllEqualNonzero()
{
    auto dgen = generateConstant<DTYPE>(DenormMins{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_NE(ref[0], 0.0f);
    EXPECT_GT(ref[0], 0.0f);
    for(size_t i = 1; i < ref.size(); ++i)
    {
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
    }
}

TEST(DataGeneratorConstantFills, DenormMinsF4)
{
    expectDenormMinAllEqualNonzero<ocp_e2m1_mxfp4>();
}
TEST(DataGeneratorConstantFills, DenormMinsF6_E2M3)
{
    expectDenormMinAllEqualNonzero<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMinsF6_E3M2)
{
    expectDenormMinAllEqualNonzero<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMinsF8_E4M3)
{
    expectDenormMinAllEqualNonzero<ocp_e4m3_mxfp8>();
}
TEST(DataGeneratorConstantFills, DenormMinsF8_E5M2)
{
    expectDenormMinAllEqualNonzero<ocp_e5m2_mxfp8>();
}

// ---------------------------------------------------------------------------
// DenormMaxs: dequantized == largest positive subnormal for the data type.
// Just like DenormMins, only the relative ordering matters here -- it must
// be strictly greater than DenormMins for the same DTYPE on types where
// there is more than one subnormal value (F6 and up; F4 has a single
// subnormal so the two modes coincide).
// ---------------------------------------------------------------------------
template <typename DTYPE>
void expectDenormMaxAllEqualNonzero()
{
    auto dgen = generateConstant<DTYPE>(DenormMaxs{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_NE(ref[0], 0.0f);
    EXPECT_GT(ref[0], 0.0f);
    for(size_t i = 1; i < ref.size(); ++i)
    {
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
    }
}

TEST(DataGeneratorConstantFills, DenormMaxsF6_E2M3)
{
    expectDenormMaxAllEqualNonzero<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF6_E3M2)
{
    expectDenormMaxAllEqualNonzero<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF8_E4M3)
{
    expectDenormMaxAllEqualNonzero<ocp_e4m3_mxfp8>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF8_E5M2)
{
    expectDenormMaxAllEqualNonzero<ocp_e5m2_mxfp8>();
}

TEST(DataGeneratorConstantFills, DenormMinLessOrEqualDenormMax_F8_E4M3)
{
    auto dmin = generateConstant<ocp_e4m3_mxfp8>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<ocp_e4m3_mxfp8>(DenormMaxs{}).getReferenceFloat();
    ASSERT_FALSE(dmin.empty());
    ASSERT_FALSE(dmax.empty());
    EXPECT_LE(dmin[0], dmax[0]);
    EXPECT_LT(0.0f, dmin[0]);
}

// ---------------------------------------------------------------------------
// NaNs: every dequantized element is NaN.  For F4 / F6 this is encoded by
// the scale (no per-element NaN representation in the data type); for F8
// E4M3 / E5M2 it is encoded by the data byte; for F8 with E4M3 / E5M3
// scales it is again encoded by the scale.  All of those should look
// identical at the dequantized-float boundary.
// ---------------------------------------------------------------------------
TEST(DataGeneratorConstantFills, NaNsF4_E8M0)
{
    expectAllNaN<ocp_e2m1_mxfp4>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF4_E4M3)
{
    expectAllNaN<ocp_e2m1_mxfp4_e4m3>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF4_E5M3)
{
    expectAllNaN<ocp_e2m1_mxfp4_e5m3>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF6_E2M3)
{
    expectAllNaN<ocp_e2m3_mxfp6>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF6_E3M2)
{
    expectAllNaN<ocp_e3m2_mxfp6>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF8_E4M3)
{
    expectAllNaN<ocp_e4m3_mxfp8>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF8_E5M2)
{
    expectAllNaN<ocp_e5m2_mxfp8>(NaNs{});
}

// ---------------------------------------------------------------------------
// Infs: only F8 E5M2 has an Inf representation.  Every other DTYPE must
// throw out of `generate()` so a misuse fails loudly instead of silently
// producing 1.0s.
// ---------------------------------------------------------------------------
TEST(DataGeneratorConstantFills, InfsF8_E5M2)
{
    expectAllInf<ocp_e5m2_mxfp8>(Infs{});
}

template <typename DTYPE>
void expectInfsThrow()
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = Infs{};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<DTYPE> dgen;
    EXPECT_THROW(dgen.generate(sizes, strides, opts), std::runtime_error);
}

TEST(DataGeneratorConstantFills, InfsF4_Throws)
{
    expectInfsThrow<ocp_e2m1_mxfp4>();
}
TEST(DataGeneratorConstantFills, InfsF6_E2M3_Throws)
{
    expectInfsThrow<ocp_e2m3_mxfp6>();
}
TEST(DataGeneratorConstantFills, InfsF6_E3M2_Throws)
{
    expectInfsThrow<ocp_e3m2_mxfp6>();
}
TEST(DataGeneratorConstantFills, InfsF8_E4M3_Throws)
{
    // E4M3 lacks an Inf representation even though it is an 8-bit float.
    expectInfsThrow<ocp_e4m3_mxfp8>();
}

// ---------------------------------------------------------------------------
// RandInt: every dequantized element is an integer in [lo, hi] (exact, since
// integer values within the data type's range round-trip cleanly through
// satConvertToType).  Distribution is uniform; the test checks
// integerness + range bounds + non-degenerate spread.
// ---------------------------------------------------------------------------
template <typename DTYPE>
void expectRandIntInRange(int lo, int hi)
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{lo, hi};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<DTYPE> dgen;
    // Use a fixed seed so the spread assertion is deterministic.
    dgen.setSeed(123u);
    dgen.generate(sizes, strides, opts);
    auto ref = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));

    int distinctCount = 0;
    bool seen[256]    = {};
    for(size_t i = 0; i < ref.size(); ++i)
    {
        float const v = ref[i];
        EXPECT_FALSE(std::isnan(v)) << "element " << i;
        EXPECT_FALSE(std::isinf(v)) << "element " << i;
        // Integer-valued.
        EXPECT_EQ(v, std::trunc(v)) << "element " << i << " = " << v;
        // In range.
        EXPECT_GE(v, static_cast<float>(lo)) << "element " << i;
        EXPECT_LE(v, static_cast<float>(hi)) << "element " << i;
        // Track distinct values for the spread check below.
        int const slot = static_cast<int>(v) - lo;
        if(slot >= 0 && slot < 256 && !seen[slot])
        {
            seen[slot] = true;
            ++distinctCount;
        }
    }
    // For non-degenerate ranges, expect more than one distinct value across
    // 128 samples (sanity check that the PRNG is actually being driven).
    if(hi > lo)
    {
        EXPECT_GT(distinctCount, 1) << "RandInt produced a degenerate sample";
    }
}

TEST(DataGeneratorConstantFills, RandIntF4_Range_m4_4)
{
    // F4 max = 6.0; legacy random_int<hipblaslt_f4x2> uses [-4, 4].
    expectRandIntInRange<ocp_e2m1_mxfp4>(-4, 4);
}
TEST(DataGeneratorConstantFills, RandIntF6_E2M3_Range_m7_7)
{
    // F6 E2M3 max = 7.5; legacy random_int<hipblaslt_f6x16> uses [-7, 7].
    expectRandIntInRange<ocp_e2m3_mxfp6>(-7, 7);
}
TEST(DataGeneratorConstantFills, RandIntF6_E3M2_Range_m28_28)
{
    // F6 E3M2 max = 28.0; legacy random_int<hipblaslt_bf6x16> uses [-28, 28].
    expectRandIntInRange<ocp_e3m2_mxfp6>(-28, 28);
}
TEST(DataGeneratorConstantFills, RandIntF8_E4M3_Range_1_10)
{
    // Legacy default-template random_int<T>: [1, 10].
    expectRandIntInRange<ocp_e4m3_mxfp8>(1, 10);
}
TEST(DataGeneratorConstantFills, RandIntF8_E5M2_Range_1_10)
{
    expectRandIntInRange<ocp_e5m2_mxfp8>(1, 10);
}
TEST(DataGeneratorConstantFills, RandIntDegenerateRangeIsConstant)
{
    // lo == hi must produce that single value at every element.
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{3, 3};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<ocp_e4m3_mxfp8> dgen;
    dgen.generate(sizes, strides, opts);
    auto ref = dgen.getReferenceFloat();
    for(float v : ref)
        EXPECT_EQ(v, 3.0f);
}
TEST(DataGeneratorConstantFills, RandIntInvertedRangeThrows)
{
    DataGeneratorOptions opts;
    opts.blockScaling = kBlockScaling;
    opts.initMode     = RandInt{5, -5};
    opts.forceDenorm  = false;
    std::vector<index_t> sizes{kRows, kCols};
    std::vector<index_t> strides{1, kRows};
    DataGenerator<ocp_e4m3_mxfp8> dgen;
    EXPECT_THROW(dgen.generate(sizes, strides, opts), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Multi-byte (host) DTYPE coverage: bf16, fp16, f32.
//
// Regression guard for a bug in the original `generate_data_constant_byte`
// helper, which truncated the data bit-pattern to a single byte.  That was
// invisible to MX types (all 1 byte unpacked) but produced wildly wrong
// values for bf16 / fp16 / f32 -- e.g. Twos for bf16 came out as 0.0
// because bf16(2.0) = 0x4000 and only the low byte (0x00) was written.
// Once the helper widened to a uint64_t bit pattern + memcpy(byte_size),
// every constant fill below started returning the expected dequantized
// value.  Pin those expectations so the bug can never come back silently.
// ---------------------------------------------------------------------------
namespace
{
    template <typename DTYPE>
    void expectAllValuesUnscaled(DataInitMode initMode, float expected)
    {
        // Unscaled host types ignore blockScaling at the array level but the
        // generator still asserts size[0] % blockScaling == 0, so reuse the
        // 32-row shape from generateConstant().
        auto dgen = generateConstant<DTYPE>(initMode);
        auto ref  = dgen.getReferenceFloat();
        ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
        for(size_t i = 0; i < ref.size(); ++i)
        {
            EXPECT_EQ(ref[i], expected) << "element " << i;
        }
    }
} // namespace

// Twos -- this is the original smoking gun for the byte-truncation bug.
TEST(DataGeneratorConstantFills, TwosBf16)
{
    expectAllValuesUnscaled<bf16>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosFp16)
{
    expectAllValuesUnscaled<fp16>(Twos{}, 2.0f);
}
TEST(DataGeneratorConstantFills, TwosF32)
{
    expectAllValuesUnscaled<f32>(Twos{}, 2.0f);
}

// NegOnes -- bf16(-1.0) = 0xBF80; the high byte was being dropped.
TEST(DataGeneratorConstantFills, NegOnesBf16)
{
    expectAllValuesUnscaled<bf16>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesFp16)
{
    expectAllValuesUnscaled<fp16>(NegOnes{}, -1.0f);
}
TEST(DataGeneratorConstantFills, NegOnesF32)
{
    expectAllValuesUnscaled<f32>(NegOnes{}, -1.0f);
}

// MaxVals -- pulls the canonical max-normal mask from the DTYPE itself
// (bf16: 0x7F7F, fp16: 0x7BFF, f32: 0x7F7FFFFF), so a byte-truncated path
// would silently turn into a tiny value instead.
TEST(DataGeneratorConstantFills, MaxValsBf16)
{
    expectAllValuesUnscaled<bf16>(MaxVals{}, bf16::dataMaxNormalNumber);
}
TEST(DataGeneratorConstantFills, MaxValsFp16)
{
    expectAllValuesUnscaled<fp16>(MaxVals{}, fp16::dataMaxNormalNumber);
}
TEST(DataGeneratorConstantFills, MaxValsF32)
{
    expectAllValuesUnscaled<f32>(MaxVals{}, f32::dataMaxNormalNumber);
}

// DenormMins -- the canonical "smallest non-zero denorm" pattern is 0x1
// in the low byte for every DTYPE (the higher bytes must remain 0), so
// this happens to be the *one* constant fill that was already correct
// even with the original byte-truncated helper.  Pin it anyway so any
// future regression that disturbs the higher bytes is caught.
//
// We compute the expected value as an exact 2^(-bias - mantissa_bits)
// power of two via std::ldexp instead of relying on each DTYPE's
// `dataMinSubNormalNumber` literal -- some of those literals (notably
// fp16's) are truncated decimal approximations that don't bit-match the
// IEEE-correct value getReferenceFloat() produces.
TEST(DataGeneratorConstantFills, DenormMinsBf16)
{
    // bf16: bias=127, mantissa=7 bits -> 2^-(127-1+7) = 2^-133.
    expectAllValuesUnscaled<bf16>(DenormMins{}, std::ldexp(1.0f, -133));
}
TEST(DataGeneratorConstantFills, DenormMinsFp16)
{
    // fp16: bias=15, mantissa=10 bits -> 2^-(15-1+10) = 2^-24.
    expectAllValuesUnscaled<fp16>(DenormMins{}, std::ldexp(1.0f, -24));
}
TEST(DataGeneratorConstantFills, DenormMinsF32)
{
    // f32: bias=127, mantissa=23 bits -> 2^-(127-1+23) = 2^-149.
    expectAllValuesUnscaled<f32>(DenormMins{}, std::ldexp(1.0f, -149));
}

// DenormMaxs -- this is the second pattern that exercised the bug:
// bf16 max-denorm = 0x007F (low byte 0x7F, high byte 0x00) was previously
// fine on bf16 but *not* on fp16 (0x03FF) or f32 (0x007FFFFF), where the
// high mantissa bytes were dropped.
template <typename DTYPE>
void expectDenormMaxAllPositiveAtBoundary()
{
    auto dgen = generateConstant<DTYPE>(DenormMaxs{});
    auto ref  = dgen.getReferenceFloat();
    ASSERT_EQ(ref.size(), static_cast<size_t>(kRows * kCols));
    ASSERT_GT(ref[0], 0.0f);
    // Largest subnormal must be strictly less than the type's smallest
    // normal.  smallest normal = 2 * (largest subnormal LSB granularity),
    // and for any IEEE-style format largest-denorm == smallest-normal -
    // smallest-denorm.  So largest-denorm < smallest-normal is the cleanest
    // type-agnostic ordering invariant we can pin.
    EXPECT_LT(ref[0], DTYPE::dataMinSubNormalNumber * (1ull << 24));
    for(size_t i = 1; i < ref.size(); ++i)
        EXPECT_EQ(ref[i], ref[0]) << "element " << i;
}

TEST(DataGeneratorConstantFills, DenormMaxsBf16)
{
    expectDenormMaxAllPositiveAtBoundary<bf16>();
}
TEST(DataGeneratorConstantFills, DenormMaxsFp16)
{
    expectDenormMaxAllPositiveAtBoundary<fp16>();
}
TEST(DataGeneratorConstantFills, DenormMaxsF32)
{
    expectDenormMaxAllPositiveAtBoundary<f32>();
}

// Cross-check: DenormMaxs > DenormMins for every multi-byte type.
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_Bf16)
{
    auto dmin = generateConstant<bf16>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<bf16>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_Fp16)
{
    auto dmin = generateConstant<fp16>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<fp16>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}
TEST(DataGeneratorConstantFills, DenormMaxGreaterThanMin_F32)
{
    auto dmin = generateConstant<f32>(DenormMins{}).getReferenceFloat();
    auto dmax = generateConstant<f32>(DenormMaxs{}).getReferenceFloat();
    EXPECT_LT(dmin[0], dmax[0]);
}

// NaN / Inf -- bf16, fp16, f32 all encode both, so neither must throw.
TEST(DataGeneratorConstantFills, NaNsBf16)
{
    expectAllNaN<bf16>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsFp16)
{
    expectAllNaN<fp16>(NaNs{});
}
TEST(DataGeneratorConstantFills, NaNsF32)
{
    expectAllNaN<f32>(NaNs{});
}
TEST(DataGeneratorConstantFills, InfsBf16)
{
    expectAllInf<bf16>(Infs{});
}
TEST(DataGeneratorConstantFills, InfsFp16)
{
    expectAllInf<fp16>(Infs{});
}
TEST(DataGeneratorConstantFills, InfsF32)
{
    expectAllInf<f32>(Infs{});
}

// RandInt for multi-byte types -- this never went through the byte-truncated
// helper (it has its own memcpy of byte_size bytes), but exercise it anyway
// so coverage is uniform across DTYPEs.
TEST(DataGeneratorConstantFills, RandIntBf16_Range_m10_10)
{
    expectRandIntInRange<bf16>(-10, 10);
}
TEST(DataGeneratorConstantFills, RandIntFp16_Range_m10_10)
{
    expectRandIntInRange<fp16>(-10, 10);
}
TEST(DataGeneratorConstantFills, RandIntF32_Range_m100_100)
{
    expectRandIntInRange<f32>(-100, 100);
}
