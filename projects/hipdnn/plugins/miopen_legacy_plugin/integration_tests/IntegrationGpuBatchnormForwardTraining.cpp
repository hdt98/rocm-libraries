// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "IntegrationTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

enum class BatchnormTrainingScenario
{
    BASIC_FORWARD,      // No saved stats (Y output only)
    WITH_RUNNING_STATS, // Running stats only
    WITH_BATCH_STATS,   // Batch stats only
    FULL_TRAINING       // All stats (batch + running)
};

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w
                  << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

struct Batchnorm3dTestCase
{
    int64_t n;
    int64_t c;
    int64_t d;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm3dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " d:" << tc.d << " h:" << tc.h
                  << " w:" << tc.w << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, d, h, w};
    }
};

template <typename InputType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardTraining : public GraphVerifierTest<InputType, TestCaseType>
{
protected:
    void runGraphTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        runGraphTestWithScenario(tolerance, BatchnormTrainingScenario::FULL_TRAINING, layout);
    }

    void runGraphTestWithScenario(InputType tolerance,
                                  BatchnormTrainingScenario scenario,
                                  const TensorLayout& layout = TensorLayout::NCHW)
    {
        const TestCaseType& testCase = this->GetParam();
        
        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormForwardTrainingTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;
        auto dims = testCase.getDims();
        auto derivedDims = getDerivedShape(dims);

        // Create input tensor attributes
        auto xAttr = graph::makeTensorAttributes(
            "X", inputDataType, dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonAttr = graph::makeTensorAttributes(
            "epsilon", intermediateDataType, std::vector<int64_t>{1}, std::vector<int64_t>{1});
        epsilonAttr.set_uid(uid++);
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        // Conditionally setup running statistics based on scenario
        std::shared_ptr<graph::TensorAttributes> prevRunningMeanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> prevRunningVarianceTensorAttr;
        std::shared_ptr<graph::TensorAttributes> momentumTensorAttr;

        if(scenario == BatchnormTrainingScenario::WITH_RUNNING_STATS
           || scenario == BatchnormTrainingScenario::FULL_TRAINING)
        {
            auto prevRunningMeanAttr = graph::makeTensorAttributes(
                "prev_running_mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
            prevRunningMeanAttr.set_uid(uid++);
            prevRunningMeanTensorAttr =
                std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr = graph::makeTensorAttributes(
                "prev_running_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
            prevRunningVarianceAttr.set_uid(uid++);
            prevRunningVarianceTensorAttr =
                std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

            auto momentumAttr = graph::makeTensorAttributes(
                "momentum", intermediateDataType, std::vector<int64_t>{1}, std::vector<int64_t>{1});
            momentumAttr.set_uid(uid++);
            momentumTensorAttr =
                std::make_shared<graph::TensorAttributes>(std::move(momentumAttr));
        }

        // Create batchnorm attributes
        graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_training");

        if(prevRunningMeanTensorAttr && prevRunningVarianceTensorAttr && momentumTensorAttr)
        {
            bnAttrs.set_previous_running_stats(
                prevRunningMeanTensorAttr, prevRunningVarianceTensorAttr, momentumTensorAttr);
        }

        // Conditionally set mean/invVariance based on scenario
        if(scenario != BatchnormTrainingScenario::BASIC_FORWARD)
        {
            auto meanPlaceholder = std::make_shared<graph::TensorAttributes>();
            auto invVarPlaceholder = std::make_shared<graph::TensorAttributes>();
            bnAttrs.set_mean(meanPlaceholder);
            bnAttrs.set_inv_variance(invVarPlaceholder);
        }

        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yTensorAttr,
              meanTensorAttr,
              invVarianceTensorAttr,
              nextRunningMeanTensorAttr,
              nextRunningVarianceTensorAttr] =
            graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        // Set output tensor attributes
        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(inputDataType);
        yTensorAttr->set_dim(dims);
        yTensorAttr->set_stride(generateStrides(dims, layout.strideOrder));

        // Configure batch statistics outputs if they exist
        if(meanTensorAttr)
        {
            if(!meanTensorAttr->has_uid())
            {
                meanTensorAttr->set_uid(uid++);
            }
            meanTensorAttr->set_output(true);
            meanTensorAttr->set_data_type(intermediateDataType);
            meanTensorAttr->set_dim(derivedDims);
            meanTensorAttr->set_stride(generateStrides(derivedDims));
        }

        if(invVarianceTensorAttr)
        {
            if(!invVarianceTensorAttr->has_uid())
            {
                invVarianceTensorAttr->set_uid(uid++);
            }
            invVarianceTensorAttr->set_output(true);
            invVarianceTensorAttr->set_data_type(intermediateDataType);
            invVarianceTensorAttr->set_dim(derivedDims);
            invVarianceTensorAttr->set_stride(generateStrides(derivedDims));
        }

        // Configure running statistics outputs if they exist
        if(nextRunningMeanTensorAttr)
        {
            if(!nextRunningMeanTensorAttr->has_uid())
            {
                nextRunningMeanTensorAttr->set_uid(uid++);
            }
            nextRunningMeanTensorAttr->set_name("next_running_mean");
            nextRunningMeanTensorAttr->set_output(true);
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
            nextRunningMeanTensorAttr->set_dim(derivedDims);
            nextRunningMeanTensorAttr->set_stride(generateStrides(derivedDims));
        }

        if(nextRunningVarianceTensorAttr)
        {
            if(!nextRunningVarianceTensorAttr->has_uid())
            {
                nextRunningVarianceTensorAttr->set_uid(uid++);
            }
            nextRunningVarianceTensorAttr->set_name("next_running_variance");
            nextRunningVarianceTensorAttr->set_output(true);
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
            nextRunningVarianceTensorAttr->set_dim(derivedDims);
            nextRunningVarianceTensorAttr->set_stride(generateStrides(derivedDims));
        }

        CpuFpReferenceMiopenRmsValidation<InputType> validator(tolerance);
        this->verifyGraph(graphObj, testCase.seed, validator);
    }

    GraphTensorBundle generateBundle(hipdnn_frontend::graph::Graph& graph) override
    {
        auto bundle = GraphVerifierTest<InputType, TestCaseType>::generateBundle(graph);
        return bundle;
    }

    void initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Initialize tensors with custom ranges to match MIOpen's initialization strategy
        std::mt19937 gen(seed);
        
        // Track running stats to ensure prev/next pairs get same values
        std::map<std::string, unsigned int> runningStatSeeds;

        for(auto& [tensorId, tensorPtr] : bundle.tensors)
        {
            auto name = getTensorName(graph, tensorId);
            
            if(name == "epsilon")
            {
                // Epsilon: small positive value
                std::uniform_real_distribution<float> epsilonDist(1e-6f, 1e-4f);
                auto* data = static_cast<float*>(tensorPtr->rawHostData());
                data[0] = epsilonDist(gen);
            }
            else if(name == "momentum")
            {
                // Momentum: typical training value
                std::uniform_real_distribution<float> momentumDist(0.05f, 0.15f);
                auto* data = static_cast<float*>(tensorPtr->rawHostData());
                data[0] = momentumDist(gen);
            }
            else if(name == "prev_running_mean" || name == "next_running_mean")
            {
                // Running mean: prev and next must start with SAME values
                // because MIOpen's API uses IN/OUT parameter semantics
                if(runningStatSeeds.find("running_mean") == runningStatSeeds.end())
                {
                    runningStatSeeds["running_mean"] = seed + 1000;
                }
                bundle.randomizeTensor(tensorId, -2.0f, 2.0f, runningStatSeeds["running_mean"]);
            }
            else if(name == "prev_running_variance" || name == "next_running_variance")
            {
                // Running variance: prev and next must start with SAME values
                // because MIOpen's API uses IN/OUT parameter semantics
                if(runningStatSeeds.find("running_variance") == runningStatSeeds.end())
                {
                    runningStatSeeds["running_variance"] = seed + 2000;
                }
                bundle.randomizeTensor(tensorId, -2.0f, 2.0f, runningStatSeeds["running_variance"]);
            }
            else if(name == "scale" || name == "bias")
            {
                // Match MIOpen's initialization: -2.0 to 2.0
                bundle.randomizeTensor(tensorId, -2.0f, 2.0f, seed + static_cast<unsigned int>(tensorId));
            }
            else
            {
                // Default initialization for input and other tensors
                bundle.randomizeTensor(tensorId, -1.0f, 1.0f, seed + static_cast<unsigned int>(tensorId));
            }
        }
    }

private:
    std::string getTensorName(const hipdnn_frontend::graph::Graph& graph, int64_t tensorId) const
    {
        // Simple helper to find tensor name by ID
        std::string result;
        visitGraph(graph, [&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(tensorAttr->get_uid() == tensorId && !tensorAttr->get_name().empty())
                {
                    result = tensorAttr->get_name();
                    return;
                }
            }
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tensorAttr->get_uid() == tensorId && !tensorAttr->get_name().empty())
                {
                    result = tensorAttr->get_name();
                    return;
                }
            }
        });
        return result;
    }
};

// NCHW 2D
using BnFwdTrainNchwFp32 = BatchnormForwardTraining<float, float, Batchnorm2dTestCase>;
using BnFwdTrainNchwFp16Mixed = BatchnormForwardTraining<half, float, Batchnorm2dTestCase>;
using BnFwdTrainNchwBfp16Mixed = BatchnormForwardTraining<hip_bfloat16, float, Batchnorm2dTestCase>;

// NHWC 2D
using BnFwdTrainNhwcFp32 = BatchnormForwardTraining<float, float, Batchnorm2dTestCase>;
using BnFwdTrainNhwcFp16Mixed = BatchnormForwardTraining<half, float, Batchnorm2dTestCase>;
using BnFwdTrainNhwcBfp16Mixed = BatchnormForwardTraining<hip_bfloat16, float, Batchnorm2dTestCase>;

// NCDHW 3D
using BnFwdTrainNcdhwFp32 = BatchnormForwardTraining<float, float, Batchnorm3dTestCase>;
using BnFwdTrainNcdhwFp16Mixed = BatchnormForwardTraining<half, float, Batchnorm3dTestCase>;
using BnFwdTrainNcdhwBfp16Mixed =
    BatchnormForwardTraining<hip_bfloat16, float, Batchnorm3dTestCase>;

// NDHWC 3D
using BnFwdTrainNdhwcFp32 = BatchnormForwardTraining<float, float, Batchnorm3dTestCase>;
using BnFwdTrainNdhwcFp16Mixed = BatchnormForwardTraining<half, float, Batchnorm3dTestCase>;
using BnFwdTrainNdhwcBfp16Mixed =
    BatchnormForwardTraining<hip_bfloat16, float, Batchnorm3dTestCase>;

std::vector<Batchnorm2dTestCase> getBnFwdTrainingTestCases()
{
    unsigned int seed = std::random_device{}();

    return {
    //     {1, 3, 14, 14, seed},
    //     // {1, 256, 1, 1, seed},
    //     {2, 3, 1, 1, seed},
    //     {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed}
        // {32, 3, 14, 1, seed},
        // {64, 64, 112, 112, seed},
        // {64, 512, 14, 14, seed},
    };
}

std::vector<Batchnorm3dTestCase> getBnFwdTraining3dTestCases()
{
    unsigned int seed = std::random_device{}();

    return {
        {2, 3, 3, 1, 1, seed},
        // {16, 3, 8, 14, 14, seed},
    };
}

} // namespace

// ============================================================================
// NCHW 2D Tests
// ============================================================================

TEST_P(BnFwdTrainNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NCHW);
}

TEST_P(BnFwdTrainNchwFp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NCHW);
}

TEST_P(BnFwdTrainNchwBfp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NCHW);
}

// ============================================================================
// NHWC 2D Tests
// ============================================================================

TEST_P(BnFwdTrainNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NHWC);
}

TEST_P(BnFwdTrainNhwcFp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NHWC);
}

TEST_P(BnFwdTrainNhwcBfp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NHWC);
}

// ============================================================================
// NCDHW 3D Tests
// ============================================================================

TEST_P(BnFwdTrainNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NCDHW);
}

TEST_P(BnFwdTrainNcdhwFp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NCDHW);
}

TEST_P(BnFwdTrainNcdhwBfp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NCDHW);
}

// ============================================================================
// NDHWC 3D Tests
// ============================================================================

TEST_P(BnFwdTrainNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NDHWC);
}

TEST_P(BnFwdTrainNdhwcFp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NDHWC);
}

TEST_P(BnFwdTrainNdhwcBfp16Mixed, Correctness)
{
    runGraphTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NDHWC);
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(, BnFwdTrainNchwFp32, testing::ValuesIn(getBnFwdTrainingTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNchwFp16Mixed,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNchwBfp16Mixed,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

INSTANTIATE_TEST_SUITE_P(, BnFwdTrainNhwcFp32, testing::ValuesIn(getBnFwdTrainingTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNhwcFp16Mixed,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNhwcBfp16Mixed,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

INSTANTIATE_TEST_SUITE_P(, BnFwdTrainNcdhwFp32, testing::ValuesIn(getBnFwdTraining3dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNcdhwFp16Mixed,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNcdhwBfp16Mixed,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

INSTANTIATE_TEST_SUITE_P(, BnFwdTrainNdhwcFp32, testing::ValuesIn(getBnFwdTraining3dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNdhwcFp16Mixed,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         BnFwdTrainNdhwcBfp16Mixed,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));
