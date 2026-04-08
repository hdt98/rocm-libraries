// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include "GEMMTestBase.hpp"
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

#include <common/Utilities.hpp>

namespace GEMMTests
{
    using namespace rocRoller;

    class GEMMStrideTestSuite : public BaseGEMMContextFixture<bool> // bool = materialStrides
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            auto device          = std::get<0>(this->GetParam());
            bool materialStrides = std::get<1>(this->GetParam());

            m_kernelOptions->materialStrides = materialStrides;
            return this->createContextForArch(device);
        }

        bool materialStrides() const
        {
            return std::get<1>(this->GetParam());
        }
    };

    TEST_P(GEMMStrideTestSuite, GPU_MaterialStrides_FP32)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        GEMMProblem gemm;

        gemm.loadPathA = Parameters::Solution::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = Parameters::Solution::LoadPath::BufferToLDSViaVGPR;

        basicGEMM<float>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        int numStrideRegisters   = countSubstring(generatedCode, "Assign stride register");
        int numStrideExpressions = countSubstring(generatedCode, "Assign stride expression");

        if(materialStrides())
        {
            EXPECT_GT(numStrideRegisters, 0) << "materialStrides=true: expected some runtime "
                                                "strides to be pre-computed into registers";
            EXPECT_GT(numStrideExpressions, 0)
                << "materialStrides=true: expected some strides to be kept as expressions";
        }
        else
        {
            EXPECT_EQ(numStrideRegisters, 0)
                << "materialStrides=false: expected no stride registers";
            EXPECT_GT(numStrideExpressions, 0)
                << "materialStrides=false: expected strides to be kept as expressions";
        }
    }

    INSTANTIATE_TEST_SUITE_P(GEMMStrideTest,
                             GEMMStrideTestSuite,
                             ::testing::Combine(currentGPUISA(), ::testing::Values(false, true)));

} // namespace GEMMTests
