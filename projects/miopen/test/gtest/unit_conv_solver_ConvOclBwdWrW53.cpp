/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <optional>

#include "unit_conv_solver.hpp"
#include "../lib_env_var.hpp"

#include <miopen/bfloat16.hpp>
#include <half/half.hpp>

// The ConvOclBwdWrW53 solver supports two kernel backends selected at runtime
// via these environment variables (read by ExecutionContext::DetectRocm()).
// We toggle both flags so each test fixture deterministically exercises one
// backend regardless of the host's defaults.
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_HIP_KERNELS)

namespace {

auto GetConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 1, 7, 7}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, datatype},
        // clang-format on
    };
}

// Grouped variant exercising the MIOpenGroupConvBwdWrW_LxG_P53{,Hip} kernel path.
// Channel and filter counts are chosen to be divisible by the group count so the
// solver applies cleanly: c = w_per_group * groups, k = filters * groups.
auto GetGroupedConvTestCases(miopenDataType_t datatype)
{
    using TestCase = miopen::unit_tests::ConvTestCase;

    return std::vector{
        // clang-format off
        TestCase{{16, 2, 7, 7}, {2, 1, 3, 3}, {0, 0}, {1, 1}, {1, 1}, /*groups=*/2, datatype},
        // clang-format on
    };
}

const auto& GetTestParams()
{
    static const auto params = [] {
        Gpu supported_gpus = Gpu::gfx900 | Gpu::gfx906 | Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx103X;
        auto p             = miopen::unit_tests::UnitTestConvSolverParams(supported_gpus);
        return p;
    }();
    return params;
}

// Helper that snapshots the current backend-selection env vars, forces a
// specific backend for the duration of a single test, and restores the
// original values on teardown. Designed to be mixed into a gtest fixture via
// SetUp/TearDown delegation.
class BackendEnvOverride
{
public:
    enum class Backend
    {
        Ocl,
        Hip,
    };

    // Capture the current env-var state so it can be restored later. Call
    // exactly once per fixture lifetime, before any Force() flips.
    void Snapshot()
    {
        prev_ocl_ = SnapshotBool(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS);
        prev_hip_ = SnapshotBool(MIOPEN_DEBUG_HIP_KERNELS);
    }

    // Flip env vars to select the requested backend. Safe to call multiple
    // times within a single Snapshot()/Restore() pair.
    void Force(Backend backend)
    {
        // Force exactly one backend on. The solver prefers HIP when both flags
        // are true, so we additionally disable the unused side to keep the
        // selection unambiguous and to give a clean failure mode if the
        // intended kernel is unavailable.
        switch(backend)
        {
        case Backend::Ocl:
            lib_env::update(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS, true);
            lib_env::update(MIOPEN_DEBUG_HIP_KERNELS, false);
            break;
        case Backend::Hip:
            lib_env::update(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS, false);
            lib_env::update(MIOPEN_DEBUG_HIP_KERNELS, true);
            break;
        }
    }

    // Convenience wrapper preserving the original Snapshot+Force semantics
    // used by WithBackend<>.
    void Apply(Backend backend)
    {
        Snapshot();
        Force(backend);
    }

    void Restore()
    {
        RestoreBool(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS, prev_ocl_);
        RestoreBool(MIOPEN_DEBUG_HIP_KERNELS, prev_hip_);
    }

private:
    static std::optional<bool> SnapshotBool(const lib_env::LibEnvVar& var)
    {
        if(static_cast<bool>(var))
        {
            return lib_env::value<bool>(var);
        }
        return std::nullopt;
    }

    static void RestoreBool(const lib_env::LibEnvVar& var, std::optional<bool> prev)
    {
        if(prev.has_value())
        {
            lib_env::update(var, *prev);
        }
        else
        {
            lib_env::clear(var);
        }
    }

    std::optional<bool> prev_ocl_;
    std::optional<bool> prev_hip_;
};

template <class Base, BackendEnvOverride::Backend kBackend>
class WithBackend : public Base
{
protected:
    void SetUp() override
    {
        env_override_.Apply(kBackend);
        Base::SetUp();
    }

    void TearDown() override
    {
        Base::TearDown();
        env_override_.Restore();
    }

private:
    BackendEnvOverride env_override_;
};

// Fixture mix-in for the cross-backend comparison tests. Snapshots env vars
// once at SetUp so a single test can flip backends multiple times via
// env_.Force() and still have the original state restored at TearDown.
template <class Base>
class CompareBackendsFixture : public Base
{
protected:
    void SetUp() override
    {
        env_.Snapshot();
        Base::SetUp();
    }

    void TearDown() override
    {
        Base::TearDown();
        env_.Restore();
    }

    template <typename T>
    void RunCompare(const miopen::solver::conv::ConvSolverInterface& solv,
                    miopenDataType_t datatype)
    {
        miopen::unit_tests::UnitTestConvSolverParams params;
        miopenConvAlgorithm_t algo;
        miopen::unit_tests::ConvTestCase conv_config;
        std::tie(params, algo, conv_config) = this->GetParam();

        // This test compares HIP- vs OCL-emitted versions of the same
        // ConvOclBwdWrW53 kernel, not GPU-vs-CPU-reference. Existing
        // MIOpen FP32 tests set tolerance multipliers in the 2-40x range
        // to absorb the larger error from a fundamentally different CPU
        // reference path, but here the only divergence is sub-epsilon
        // noise from accumulation order and LDS scheduling decisions
        // that differ between the OCL and HIP compilers (both emit
        // v_fma_f32 for the inner MAC). Worst observed FP32 error is
        // ~8.48e-8 (group conv); 0.75f gives threshold ~8.94e-8, ~5%
        // headroom over that floor and tight enough to catch any real
        // codegen regression. FP16/BFP16 use 0.25f for the same reason
        // -- the GPU-vs-GPU path has no need to tolerate CPU-reference
        // -level error.
        if(datatype == miopenFloat)
        {
            params.SetTolerance(Gpu::All, datatype, 0.75f);
        }
        else
        {
            params.SetTolerance(Gpu::All, datatype, 0.25f);
        }

        miopen::unit_tests::RunSolverWrwCompareBackends<T>(
            solv,
            params,
            conv_config,
            algo,
            [&] { env_.Force(BackendEnvOverride::Backend::Ocl); },
            [&] { env_.Force(BackendEnvOverride::Backend::Hip); });
    }

private:
    BackendEnvOverride env_;
};

} // namespace

// OpenCL backend (legacy kernel path)
using GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP16 =
    WithBackend<GPU_UnitTestConvSolverWrw_FP16, BackendEnvOverride::Backend::Ocl>;
using GPU_UnitTestConvSolverOclBwdWrW53_OCL_BFP16 =
    WithBackend<GPU_UnitTestConvSolverWrw_BFP16, BackendEnvOverride::Backend::Ocl>;
using GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP32 =
    WithBackend<GPU_UnitTestConvSolverWrw_FP32, BackendEnvOverride::Backend::Ocl>;

// HIP backend (new kernel path)
using GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP16 =
    WithBackend<GPU_UnitTestConvSolverWrw_FP16, BackendEnvOverride::Backend::Hip>;
using GPU_UnitTestConvSolverOclBwdWrW53_HIP_BFP16 =
    WithBackend<GPU_UnitTestConvSolverWrw_BFP16, BackendEnvOverride::Backend::Hip>;
using GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP32 =
    WithBackend<GPU_UnitTestConvSolverWrw_FP32, BackendEnvOverride::Backend::Hip>;

using CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_OCL_NONE =
    WithBackend<CPU_UnitTestConvSolverDevApplicabilityWrw_NONE, BackendEnvOverride::Backend::Ocl>;
using CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_HIP_NONE =
    WithBackend<CPU_UnitTestConvSolverDevApplicabilityWrw_NONE, BackendEnvOverride::Backend::Hip>;

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP16, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_OCL_BFP16, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP32, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP16, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HIP_BFP16, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP32, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_OCL_NONE, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

TEST_P(CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_HIP_NONE, ConvOclBwdWrW53)
{
    this->RunTest(miopen::solver::conv::ConvOclBwdWrW53{});
};

// Smoke tests - OpenCL backend
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_OCL_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Smoke tests - HIP backend
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_HIP_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Grouped per-backend smoke tests exercising the
// MIOpenGroupConvBwdWrW_LxG_P53{,Hip} kernel path against the CPU reference.
// Mirrors the ungrouped Smoke instantiations above.
INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke_G2,
    GPU_UnitTestConvSolverOclBwdWrW53_OCL_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetGroupedConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_OCL_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenFloat))));

INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke_G2,
    GPU_UnitTestConvSolverOclBwdWrW53_HIP_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetGroupedConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_HIP_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenFloat))));

// Device applicability tests - both backends
INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_OCL_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_HIP_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(GetConvTestCases(miopenFloat)[0])));

// Device applicability also evaluated against a grouped problem so the
// IsApplicable path doesn't drift untested for group_count > 1.
INSTANTIATE_TEST_SUITE_P(
    Smoke_G2,
    CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_OCL_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetGroupedConvTestCases(miopenFloat)[0])));

INSTANTIATE_TEST_SUITE_P(
    Smoke_G2,
    CPU_UnitTestConvSolverOclBwdWrW53DevApplicability_HIP_NONE,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(GetGroupedConvTestCases(miopenFloat)[0])));

// Cross-backend (OCL vs HIP) numeric comparison fixtures. Each test runs the
// same problem through both backends and diffs the GPU outputs directly,
// giving a tighter bound than the per-backend CPU-reference comparisons.
using GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP16 =
    CompareBackendsFixture<GPU_UnitTestConvSolverWrw_FP16>;
using GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_BFP16 =
    CompareBackendsFixture<GPU_UnitTestConvSolverWrw_BFP16>;
using GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP32 =
    CompareBackendsFixture<GPU_UnitTestConvSolverWrw_FP32>;

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP16, ConvOclBwdWrW53)
{
    this->template RunCompare<half_float::half>(miopen::solver::conv::ConvOclBwdWrW53{},
                                                miopenHalf);
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_BFP16, ConvOclBwdWrW53)
{
    this->template RunCompare<bfloat16>(miopen::solver::conv::ConvOclBwdWrW53{}, miopenBFloat16);
};

TEST_P(GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP32, ConvOclBwdWrW53)
{
    this->template RunCompare<float>(miopen::solver::conv::ConvOclBwdWrW53{}, miopenFloat);
};

// Ungrouped (group=1) instantiations.
INSTANTIATE_TEST_SUITE_P(Smoke_G1,
                         GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(Smoke_G1,
                         GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_BFP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke_G1,
                         GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetConvTestCases(miopenFloat))));

// Grouped (group=2) instantiations exercising the
// MIOpenGroupConvBwdWrW_LxG_P53{,Hip} kernel path.
INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP16,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Smoke_G2,
    GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoDirect),
                     testing::ValuesIn(GetGroupedConvTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(Smoke_G2,
                         GPU_UnitTestConvSolverOclBwdWrW53_HipVsOcl_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::Values(miopenConvolutionAlgoDirect),
                                          testing::ValuesIn(GetGroupedConvTestCases(miopenFloat))));
