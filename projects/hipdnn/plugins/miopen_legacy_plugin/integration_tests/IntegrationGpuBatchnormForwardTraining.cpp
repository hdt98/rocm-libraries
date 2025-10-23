// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

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
class BatchnormForwardTraining : public ::testing::TestWithParam<TestCaseType>
{
    struct TensorBundle
    {
        TensorBundle(const std::vector<int64_t>& dims,
                     unsigned int seed = 1,
                     const TensorLayout& layout = TensorLayout::NCHW)
            : derivedDims(getDerivedShape(dims))
            , xTensor(dims, layout)
            , yTensor(dims, layout)
            , scaleTensor(derivedDims)
            , biasTensor(derivedDims)
            , runningMeanTensor(derivedDims)
            , runningVarianceTensor(derivedDims)
            , meanTensor(derivedDims)
            , invVarianceTensor(derivedDims)
            , momentumTensor({1})
            , epsilonTensor({1})
        {
            xTensor.fillWithRandomValues(
                static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
            yTensor.fillWithRandomValues(
                static_cast<InputType>(-100.0f), static_cast<InputType>(100.0f), seed);

            scaleTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            biasTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            // Initialize running statistics with random values
            // These simulate the state from a previous training iteration
            // In real training, this same buffer is reused across iterations, with MIOpen
            // updating it in-place: new = (1-momentum)*old + momentum*batch_stat
            runningMeanTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            runningVarianceTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);

            // Initialize output tensors with zeros
            meanTensor.fillWithValue(static_cast<IntermediateType>(0.0f));
            invVarianceTensor.fillWithValue(static_cast<IntermediateType>(0.0f));

            // Set momentum and epsilon values using random generation
            // This ensures we're testing with varied values rather than just the defaults
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> momentumDist(0.05f, 0.15f);
            std::uniform_real_distribution<float> epsilonDist(1e-6f, 1e-4f);

            momentumTensor.memory().hostData()[0] = static_cast<IntermediateType>(momentumDist(gen));
            epsilonTensor.memory().hostData()[0] = static_cast<IntermediateType>(epsilonDist(gen));
        }

        std::vector<int64_t> derivedDims;
        PinnedTensor<InputType> xTensor;
        PinnedTensor<InputType> yTensor;
        PinnedTensor<IntermediateType> scaleTensor;
        PinnedTensor<IntermediateType> biasTensor;
        PinnedTensor<IntermediateType> runningMeanTensor;
        PinnedTensor<IntermediateType> runningVarianceTensor;
        PinnedTensor<IntermediateType> meanTensor;
        PinnedTensor<IntermediateType> invVarianceTensor;
        PinnedTensor<IntermediateType> momentumTensor;
        PinnedTensor<IntermediateType> epsilonTensor;
    };

protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Uncomment if you want debug logging info.
        // setenv("HIPDNN_LOG_LEVEL", "info", 1);

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle and stream
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    std::unordered_map<int64_t, void*> createVariantPack(
        const graph::TensorAttributes& xTensorAttr,
        const graph::TensorAttributes& scaleTensorAttr,
        const graph::TensorAttributes& biasTensorAttr,
        const graph::TensorAttributes& prevRunningMeanTensorAttr,
        const graph::TensorAttributes& prevRunningVarianceTensorAttr,
        const graph::TensorAttributes& momentumTensorAttr,
        const graph::TensorAttributes& epsilonTensorAttr,
        const graph::TensorAttributes& yTensorAttr,
        const graph::TensorAttributes& meanTensorAttr,
        const graph::TensorAttributes& invVarianceTensorAttr,
        const graph::TensorAttributes& nextRunningMeanTensorAttr,
        const graph::TensorAttributes& nextRunningVarianceTensorAttr,
        TensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[scaleTensorAttr.get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[biasTensorAttr.get_uid()] = tensorBundle.biasTensor.memory().deviceData();
        
        // Use the SAME buffer for both prev and next running statistics
        // This matches real-world usage where the buffer is updated in-place across iterations
        variantPack[prevRunningMeanTensorAttr.get_uid()]
            = tensorBundle.runningMeanTensor.memory().deviceData();
        variantPack[prevRunningVarianceTensorAttr.get_uid()]
            = tensorBundle.runningVarianceTensor.memory().deviceData();
        variantPack[nextRunningMeanTensorAttr.get_uid()]
            = tensorBundle.runningMeanTensor.memory().deviceData();
        variantPack[nextRunningVarianceTensorAttr.get_uid()]
            = tensorBundle.runningVarianceTensor.memory().deviceData();
            
        variantPack[momentumTensorAttr.get_uid()]
            = tensorBundle.momentumTensor.memory().hostData();
        variantPack[epsilonTensorAttr.get_uid()]
            = tensorBundle.epsilonTensor.memory().hostData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();
        variantPack[meanTensorAttr.get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        variantPack[invVarianceTensorAttr.get_uid()]
            = tensorBundle.invVarianceTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenBatchnormFwd(TensorBundle& graphTensorBundle,
                               hipdnn_frontend::DataType inputDataType,
                               hipdnn_frontend::DataType intermediateDataType)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("BatchnormTrainingTest");

        int64_t uid = 1;
        auto xAttr = graph::makeTensorAttributes("X", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, graphTensorBundle.biasTensor);
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto prevRunningMeanAttr = graph::makeTensorAttributes(
            "prev_running_mean", intermediateDataType, graphTensorBundle.runningMeanTensor);
        prevRunningMeanAttr.set_uid(uid++);
        auto prevRunningMeanTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

        auto prevRunningVarianceAttr
            = graph::makeTensorAttributes("prev_running_variance",
                                          intermediateDataType,
                                          graphTensorBundle.runningVarianceTensor);
        prevRunningVarianceAttr.set_uid(uid++);
        auto prevRunningVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

        auto momentumAttr = graph::makeTensorAttributes(
            "momentum", intermediateDataType, graphTensorBundle.momentumTensor);
        momentumAttr.set_uid(uid++);
        auto momentumTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(momentumAttr));

        auto epsilonAttr = graph::makeTensorAttributes(
            "epsilon", intermediateDataType, graphTensorBundle.epsilonTensor);
        epsilonAttr.set_uid(uid++);
        auto epsilonTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_training");
        bnAttrs.set_previous_running_stats(
            prevRunningMeanTensorAttr, prevRunningVarianceTensorAttr, momentumTensorAttr);
        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yTensorAttr,
              meanTensorAttr,
              invVarianceTensorAttr,
              nextRunningMeanTensorAttr,
              nextRunningVarianceTensorAttr]
            = graph->batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }
        if(!meanTensorAttr->has_uid())
        {
            meanTensorAttr->set_uid(uid++);
        }
        if(!invVarianceTensorAttr->has_uid())
        {
            invVarianceTensorAttr->set_uid(uid++);
        }
        if(!nextRunningMeanTensorAttr->has_uid())
        {
            nextRunningMeanTensorAttr->set_uid(uid++);
        }
        if(!nextRunningVarianceTensorAttr->has_uid())
        {
            nextRunningVarianceTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(inputDataType);
        yTensorAttr->set_dim(graphTensorBundle.yTensor.dims());
        yTensorAttr->set_stride(graphTensorBundle.yTensor.strides());

        meanTensorAttr->set_data_type(intermediateDataType);
        meanTensorAttr->set_dim(graphTensorBundle.meanTensor.dims());
        meanTensorAttr->set_stride(graphTensorBundle.meanTensor.strides());

        invVarianceTensorAttr->set_data_type(intermediateDataType);
        invVarianceTensorAttr->set_dim(graphTensorBundle.invVarianceTensor.dims());
        invVarianceTensorAttr->set_stride(graphTensorBundle.invVarianceTensor.strides());

        nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
        nextRunningMeanTensorAttr->set_dim(graphTensorBundle.runningMeanTensor.dims());
        nextRunningMeanTensorAttr->set_stride(graphTensorBundle.runningMeanTensor.strides());

        nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
        nextRunningVarianceTensorAttr->set_dim(graphTensorBundle.runningVarianceTensor.dims());
        nextRunningVarianceTensorAttr->set_stride(graphTensorBundle.runningVarianceTensor.strides());

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack = createVariantPack(*xTensorAttr,
                                             *scaleTensorAttr,
                                             *biasTensorAttr,
                                             *prevRunningMeanTensorAttr,
                                             *prevRunningVarianceTensorAttr,
                                             *momentumTensorAttr,
                                             *epsilonTensorAttr,
                                             *yTensorAttr,
                                             *meanTensorAttr,
                                             *invVarianceTensorAttr,
                                             *nextRunningMeanTensorAttr,
                                             *nextRunningVarianceTensorAttr,
                                             graphTensorBundle);

        result = graph->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuBatchnormFwd(TensorBundle& cpuTensorBundle)
    {
        auto epsilon = static_cast<IntermediateType>(cpuTensorBundle.epsilonTensor.memory().hostData()[0]);
        auto momentum
            = static_cast<IntermediateType>(cpuTensorBundle.momentumTensor.memory().hostData()[0]);

        // For CPU reference, we use the same single buffer for both prev and next
        // The CPU reference will read the old values and write the updated values in-place
        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormFwdTraining(
            cpuTensorBundle.xTensor,
            cpuTensorBundle.scaleTensor,
            cpuTensorBundle.biasTensor,
            cpuTensorBundle.yTensor,
            epsilon,
            momentum,
            &cpuTensorBundle.meanTensor,
            &cpuTensorBundle.invVarianceTensor,
            &cpuTensorBundle.runningMeanTensor,
            &cpuTensorBundle.runningVarianceTensor,
            &cpuTensorBundle.runningMeanTensor,
            &cpuTensorBundle.runningVarianceTensor);
    }

    void runBatchnormTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        TestCaseType testCase = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        TensorBundle graphTensorBundle(testCase.getDims(), testCase.seed, layout);

        TensorBundle cpuTensorBundle(testCase.getDims(), testCase.seed, layout);

        runMiopenBatchnormFwd(graphTensorBundle, inputDataType, intermediateDataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();
        graphTensorBundle.meanTensor.memory().markDeviceModified();
        graphTensorBundle.invVarianceTensor.memory().markDeviceModified();
        graphTensorBundle.runningMeanTensor.memory().markDeviceModified();
        graphTensorBundle.runningVarianceTensor.memory().markDeviceModified();

        runCpuBatchnormFwd(cpuTensorBundle);

        CpuFpReferenceMiopenRmsValidation<InputType> cpuRefValidation(tolerance);
        CpuFpReferenceMiopenRmsValidation<IntermediateType> cpuRefValidationStats(
            static_cast<IntermediateType>(tolerance));

        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));
        EXPECT_TRUE(cpuRefValidationStats.allClose(cpuTensorBundle.meanTensor.memory(),
                                                   graphTensorBundle.meanTensor.memory()));
        EXPECT_TRUE(cpuRefValidationStats.allClose(cpuTensorBundle.invVarianceTensor.memory(),
                                                   graphTensorBundle.invVarianceTensor.memory()));
        EXPECT_TRUE(
            cpuRefValidationStats.allClose(cpuTensorBundle.runningMeanTensor.memory(),
                                          graphTensorBundle.runningMeanTensor.memory()));
        EXPECT_TRUE(
            cpuRefValidationStats.allClose(cpuTensorBundle.runningVarianceTensor.memory(),
                                          graphTensorBundle.runningVarianceTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

class IntegrationGpuBatchnormForwardTrainingNchwFp32
    : public BatchnormForwardTraining<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNchwBfp16
    : public BatchnormForwardTraining<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNchwFp16
    : public BatchnormForwardTraining<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNhwcFp32
    : public BatchnormForwardTraining<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNhwcBfp16
    : public BatchnormForwardTraining<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNhwcFp16
    : public BatchnormForwardTraining<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNcdhwFp32
    : public BatchnormForwardTraining<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNcdhwBfp16
    : public BatchnormForwardTraining<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNcdhwFp16
    : public BatchnormForwardTraining<half, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNdhwcFp32
    : public BatchnormForwardTraining<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNdhwcBfp16
    : public BatchnormForwardTraining<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardTrainingNdhwcFp16
    : public BatchnormForwardTraining<half, float, Batchnorm3dTestCase>
{
};

std::vector<Batchnorm2dTestCase> getBnFwdTrainingTestCases()
{
    unsigned int seed = std::random_device{}();

    return {
        {1, 3, 14, 14, seed},
        {1, 256, 1, 1, seed},
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

std::vector<Batchnorm3dTestCase> getBnFwdTraining3dTestCases()
{
    unsigned int seed = std::random_device{}();

    return {
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

} // namespace

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwFp32,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwBfp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwFp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcFp32,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcBfp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcFp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcFp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwFp32,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwBfp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwFp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcFp32,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcBfp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcFp16, Correctness)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcFp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));
