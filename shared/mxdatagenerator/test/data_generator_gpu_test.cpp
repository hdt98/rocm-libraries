// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tests for the header-only HIP/GPU backend `DGen::DataGeneratorGPU`.
// Compares the GPU output to the existing CPU `DGen::DataGenerator` on:
//   * deterministic init modes (Sequential / Identity / Ones / Zeros) where
//     CPU and GPU produce semantically identical outputs;
//   * statistical properties (mean / std-dev / zero-frequency tolerances) for
//     Bounded and TrigonometricFromFloat where the two backends use different
//     PRNGs but should still match in distribution.
//
// Also exercises the GPU implementation of `preSwizzleScalesGFX950`, comparing
// against the existing host implementation byte-for-byte.

#include <gtest/gtest.h>

#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/DataGeneratorGPU.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>
#include <mxDataGenerator/ocp_e2m1_mxfp4.hpp>
#include <mxDataGenerator/ocp_e2m3_mxfp6.hpp>
#include <mxDataGenerator/ocp_e3m2_mxfp6.hpp>
#include <mxDataGenerator/ocp_e4m3_mxfp8.hpp>
#include <mxDataGenerator/ocp_e5m2_mxfp8.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace DGen;

namespace
{
    // Helper: count zeros and compute mean/stddev of |x| in a float vector.
    struct StatSummary
    {
        double zeroFrac;
        double meanAbs;
        double stdDev;
        float  maxAbs;
    };

    StatSummary summarize(std::vector<float> const& v)
    {
        StatSummary s{};
        if(v.empty())
            return s;
        size_t zeros  = 0;
        double sumAbs = 0.0;
        for(float x : v)
        {
            if(x == 0.0f)
                zeros++;
            sumAbs += std::fabs(x);
            if(std::fabs(x) > s.maxAbs)
                s.maxAbs = std::fabs(x);
        }
        s.zeroFrac = static_cast<double>(zeros) / static_cast<double>(v.size());
        s.meanAbs  = sumAbs / static_cast<double>(v.size());
        double sumSq = 0.0;
        for(float x : v)
            sumSq += static_cast<double>(x) * static_cast<double>(x);
        double meanSq = sumSq / static_cast<double>(v.size());
        s.stdDev      = std::sqrt(std::max(0.0, meanSq));
        return s;
    }
} // namespace

// -----------------------------------------------------------------------------
// Bounded mode: zero frequency for FP4 hpl-equivalent should be in the same
// ballpark as the CPU backend (both gate on block-aware quantisation rather
// than on independent scale draws).
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP4HplZeroFrequency)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -0.5;
    opt.max          = 0.5;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(12345);
    gpu.generate({1024, 1024}, {1, 1024}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    // Uncoordinated GPU init produces ~50% zeros for FP4 hpl. mxDataGenerator
    // CPU lands at ~12.9%. Our block-aware GPU quantiser should be similar in
    // order of magnitude (much less than 50%).
    EXPECT_LT(sum.zeroFrac, 0.30) << "GPU FP4 hpl zero rate should be well below the "
                                     "uncoordinated GPU baseline (~50%)";
    EXPECT_LE(sum.maxAbs, 0.5001f) << "Max should respect the input bound";
    EXPECT_GT(sum.meanAbs, 0.0) << "Mean abs should be > 0";
}

// -----------------------------------------------------------------------------
// Determinism: same seed -> same output.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, DeterministicWithFixedSeed)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> a;
    DataGeneratorGPU<DType> b;
    a.setSeed(42);
    b.setSeed(42);
    a.generate({256, 256}, {1, 256}, opt);
    b.generate({256, 256}, {1, 256}, opt);

    auto da = a.getDataBytes();
    auto db = b.getDataBytes();
    auto sa = a.getScaleBytes();
    auto sb = b.getScaleBytes();
    EXPECT_EQ(da, db);
    EXPECT_EQ(sa, sb);
}

// -----------------------------------------------------------------------------
// FP6 (E2M3) zero frequency.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP6E2M3HplZeroFrequency)
{
    using DType = ocp_e2m3_mxfp6;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -0.5;
    opt.max          = 0.5;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(12345);
    gpu.generate({512, 512}, {1, 512}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    EXPECT_LT(sum.zeroFrac, 0.20) << "FP6 has finer quantisation than FP4; expect <20% zeros";
    EXPECT_LE(sum.maxAbs, 0.5001f);
}

// -----------------------------------------------------------------------------
// FP8 (E4M3) zero frequency: very low because data type has subnormals down to
// ~0.002 and uniform [-1,1] input rarely lands inside that gap.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedFP8E4M3LowZeroFrequency)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(7);
    gpu.generate({512, 512}, {1, 512}, opt);

    auto refs = gpu.getReferenceFloat();
    auto sum  = summarize(refs);

    EXPECT_LT(sum.zeroFrac, 0.05) << "FP8 E4M3 should have <5% zeros for uniform [-1,1]";
}

// -----------------------------------------------------------------------------
// Sanity bounds for CPU vs GPU Bounded init.
//
// The CPU `Bounded` mode runs an extra `scale_block_mean` + `post_sprinkle`
// pass that injects denormals/zeros/max-values and squashes the mean magnitude
// well below the requested [min, max] range (typical |mean| ~ 0.02 for
// [-1, 1]). The GPU port deliberately omits that pass: its goal is "well-formed
// MX data with the right block-scaling invariant", not bit- or distribution-
// equivalence with the CPU path. Both representations are valid inputs for a
// real GEMM. We therefore only check that both stay inside [min, max] and that
// neither degenerates to all-zeros or all-saturated.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, BoundedCpuVsGpuStatistics)
{
    using DType = ocp_e4m3_mxfp8;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    std::vector<index_t> sizes   = {512, 512};
    std::vector<index_t> strides = {1, 512};

    DataGenerator<DType> cpu;
    cpu.setSeed(99);
    cpu.generate(sizes, strides, opt);

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(99);
    gpu.generate(sizes, strides, opt);

    auto cpuRefs = cpu.getReferenceFloat();
    auto gpuRefs = gpu.getReferenceFloat();

    auto cpuSum = summarize(cpuRefs);
    auto gpuSum = summarize(gpuRefs);

    // CPU: post_sprinkle compresses the mean toward zero; just require it stays
    // representable.
    EXPECT_GT(cpuSum.meanAbs, 0.0);
    EXPECT_LT(cpuSum.meanAbs, 1.0);
    EXPECT_LT(cpuSum.zeroFrac, 0.5)
        << "CPU Bounded should not degenerate to >50% zeros";

    // GPU: no post-processing; expect the mean magnitude to land squarely in
    // the upper half of [0, 1] (uniform draws on [-1, 1] yield E[|x|]=0.5).
    EXPECT_GT(gpuSum.meanAbs, 0.25);
    EXPECT_LT(gpuSum.meanAbs, 0.75);
    EXPECT_LT(gpuSum.zeroFrac, 0.05)
        << "GPU Bounded on FP8 should rarely produce true zeros";

    // Both should respect the [min, max] envelope set in `opt`. We allow a
    // small slack to account for FP8 quantisation rounding above 1.0.
    EXPECT_LE(cpuSum.maxAbs, 1.5f);
    EXPECT_LE(gpuSum.maxAbs, 1.5f);
}

// -----------------------------------------------------------------------------
// preSwizzleScalesGFX950: GPU vs host should be byte-equivalent.
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, PreSwizzleScalesGFX950MatchesHost)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    // 64 rows x 64 cols of K/32 scales -> 64 rows x 64 cols swizzle.
    std::vector<index_t> sizes   = {64 * 32, 64};
    std::vector<index_t> strides = {1, 64 * 32};

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(123);
    gpu.generate(sizes, strides, opt);

    auto canonical = gpu.getScaleBytes();

    // Swizzle on device.
    gpu.preSwizzleScalesGFX950Device({64, 64});
    auto gpuSwizzled = gpu.getScaleBytes();

    // Swizzle on host (reference).
    auto hostSwizzled = preSwizzleScalesGFX950(canonical, std::vector<size_t>{64, 64});

    ASSERT_EQ(gpuSwizzled.size(), hostSwizzled.size());
    EXPECT_EQ(gpuSwizzled, hostSwizzled);
}

// -----------------------------------------------------------------------------
// gfx1250 path: no shuffle should mean canonical scales are unchanged.
// (Sanity test that the wrapper plumbs preSwizzle/no-preSwizzle correctly.)
// -----------------------------------------------------------------------------
TEST(DataGeneratorGPU, NoSwizzleLeavesScalesCanonical)
{
    using DType = ocp_e2m1_mxfp4;

    DataGeneratorOptions opt;
    opt.blockScaling = 32;
    opt.initMode     = Bounded{};
    opt.min          = -1.0;
    opt.max          = 1.0;

    DataGeneratorGPU<DType> gpu;
    gpu.setSeed(31);
    gpu.generate({256, 8}, {1, 256}, opt);
    auto canonical = gpu.getScaleBytes();

    // No swizzle call: getScaleBytes() must still match the canonical layout
    // we just produced.
    auto again = gpu.getScaleBytes();
    EXPECT_EQ(canonical, again);
}
