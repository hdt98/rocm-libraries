// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

enum class BatchnormTrainingScenario
{
    BASIC_FORWARD,       // No saved stats (Y output only) - MIO: (0,0)
    WITH_RUNNING_STATS,  // Batch + Running stats, no initial - MIO: (1,1)
    WITH_BATCH_STATS,    // Batch stats only - MIO: (0,1)  
    FULL_TRAINING        // All stats with initial - MIO: (1,1)
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
class BatchnormForwardTraining : public ::testing::TestWithParam<TestCaseType>
{
    struct GraphTensorBundle
    {
        GraphTensorBundle(const std::vector<int64_t>& dims,
                          BatchnormTrainingScenario scenario,
                          unsigned int seed = getGlobalTestSeed(),
                          const TensorLayout& layout = TensorLayout::NCHW)
            : scenario(scenario)
            , derivedDims(getDerivedShape(dims))
            , xTensor(dims, layout)
            , yTensor(dims, layout)
            , scaleTensor(derivedDims)
            , biasTensor(derivedDims)
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

            // Set epsilon value using random generation
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> epsilonDist(1e-6f, 1e-4f);
            epsilonTensor.memory().hostData()[0] = static_cast<IntermediateType>(epsilonDist(gen));

            // Initialize batch statistics for all scenarios except BASIC_FORWARD
            if(scenario != BatchnormTrainingScenario::BASIC_FORWARD)
            {
                meanTensor.emplace(derivedDims);
                invVarianceTensor.emplace(derivedDims);
                meanTensor->fillWithValue(static_cast<IntermediateType>(0.0f));
                invVarianceTensor->fillWithValue(static_cast<IntermediateType>(0.0f));
            }

            // Initialize running statistics tensors if needed (MIO_RUNNING_RESULT=1)
            if(scenario == BatchnormTrainingScenario::WITH_RUNNING_STATS
               || scenario == BatchnormTrainingScenario::FULL_TRAINING)
            {
                runningMeanTensor.emplace(derivedDims);
                runningVarianceTensor.emplace(derivedDims);
                momentumTensor.emplace(std::vector<int64_t>{1});

                // Initialize with random values simulating state from previous iteration
                runningMeanTensor->fillWithRandomValues(
                    static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

                runningVarianceTensor->fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                                            static_cast<IntermediateType>(1.0f),
                                                            seed);

                std::uniform_real_distribution<float> momentumDist(0.05f, 0.15f);
                momentumTensor->memory().hostData()[0]
                    = static_cast<IntermediateType>(momentumDist(gen));
            }
        }

        std::unordered_map<int64_t, void*>
            toDeviceVariantPack(const graph::TensorAttributes& xTensorAttr,
                                const graph::TensorAttributes& scaleTensorAttr,
                                const graph::TensorAttributes& biasTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& prevRunningMeanTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& prevRunningVarianceTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& momentumTensorAttr,
                                const graph::TensorAttributes& epsilonTensorAttr,
                                const graph::TensorAttributes& yTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& meanTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& invVarianceTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& nextRunningMeanTensorAttr,
                                const std::shared_ptr<graph::TensorAttributes>& nextRunningVarianceTensorAttr)
        {
            std::unordered_map<int64_t, void*> variantPack;
            variantPack[xTensorAttr.get_uid()] = xTensor.memory().deviceData();
            variantPack[scaleTensorAttr.get_uid()] = scaleTensor.memory().deviceData();
            variantPack[biasTensorAttr.get_uid()] = biasTensor.memory().deviceData();
            variantPack[epsilonTensorAttr.get_uid()] = epsilonTensor.memory().hostData();
            variantPack[yTensorAttr.get_uid()] = yTensor.memory().deviceData();

            // Add running statistics if both attributes and tensors exist
            if(prevRunningMeanTensorAttr && prevRunningVarianceTensorAttr && momentumTensorAttr
               && nextRunningMeanTensorAttr && nextRunningVarianceTensorAttr
               && runningMeanTensor.has_value() && runningVarianceTensor.has_value()
               && momentumTensor.has_value())
            {
                // Use the SAME buffer for both prev and next running statistics
                variantPack[prevRunningMeanTensorAttr->get_uid()]
                    = runningMeanTensor->memory().deviceData();
                variantPack[prevRunningVarianceTensorAttr->get_uid()]
                    = runningVarianceTensor->memory().deviceData();
                variantPack[nextRunningMeanTensorAttr->get_uid()]
                    = runningMeanTensor->memory().deviceData();
                variantPack[nextRunningVarianceTensorAttr->get_uid()]
                    = runningVarianceTensor->memory().deviceData();
                variantPack[momentumTensorAttr->get_uid()] = momentumTensor->memory().hostData();
            }

            // Add batch statistics if tensors were actually allocated
            if(meanTensorAttr && invVarianceTensorAttr && meanTensor.has_value()
               && invVarianceTensor.has_value())
            {
                variantPack[meanTensorAttr->get_uid()] = meanTensor->memory().deviceData();
                variantPack[invVarianceTensorAttr->get_uid()]
                    = invVarianceTensor->memory().deviceData();
            }

            return variantPack;
        }

        BatchnormTrainingScenario scenario;
        std::vector<int64_t> derivedDims;
        PinnedTensor<InputType> xTensor;
        PinnedTensor<InputType> yTensor;
        PinnedTensor<IntermediateType> scaleTensor;
        PinnedTensor<IntermediateType> biasTensor;
        PinnedTensor<IntermediateType> epsilonTensor;

        // Optional: batch statistics (MIO_SAVE_MEAN_VARIANCE=1)
        std::optional<PinnedTensor<IntermediateType>> meanTensor;
        std::optional<PinnedTensor<IntermediateType>> invVarianceTensor;

        // Optional: running statistics (MIO_RUNNING_RESULT=1)
        std::optional<PinnedTensor<IntermediateType>> runningMeanTensor;
        std::optional<PinnedTensor<IntermediateType>> runningVarianceTensor;
        std::optional<PinnedTensor<IntermediateType>> momentumTensor;
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

    void runMiopenBatchnormFwd(GraphTensorBundle& graphTensorBundle,
                               hipdnn_frontend::DataType inputDataType,
                               hipdnn_frontend::DataType intermediateDataType,
                               BatchnormTrainingScenario scenario)
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

        auto epsilonAttr = graph::makeTensorAttributes(
            "epsilon", intermediateDataType, graphTensorBundle.epsilonTensor);
        epsilonAttr.set_uid(uid++);
        auto epsilonTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        // Conditionally set up running statistics
        std::shared_ptr<graph::TensorAttributes> prevRunningMeanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> prevRunningVarianceTensorAttr;
        std::shared_ptr<graph::TensorAttributes> momentumTensorAttr;

        if(scenario == BatchnormTrainingScenario::WITH_RUNNING_STATS
           || scenario == BatchnormTrainingScenario::FULL_TRAINING)
        {
            auto prevRunningMeanAttr = graph::makeTensorAttributes(
                "prev_running_mean", intermediateDataType, *graphTensorBundle.runningMeanTensor);
            prevRunningMeanAttr.set_uid(uid++);
            prevRunningMeanTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr
                = graph::makeTensorAttributes("prev_running_variance",
                                              intermediateDataType,
                                              *graphTensorBundle.runningVarianceTensor);
            prevRunningVarianceAttr.set_uid(uid++);
            prevRunningVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(prevRunningVarianceAttr));

            auto momentumAttr = graph::makeTensorAttributes(
                "momentum", intermediateDataType, *graphTensorBundle.momentumTensor);
            momentumAttr.set_uid(uid++);
            momentumTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(momentumAttr));
        }

        graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_training");

        if(prevRunningMeanTensorAttr && prevRunningVarianceTensorAttr && momentumTensorAttr)
        {
            bnAttrs.set_previous_running_stats(
                prevRunningMeanTensorAttr, prevRunningVarianceTensorAttr, momentumTensorAttr);
        }

        // Pre-set mean/invVariance to signal Graph.hpp to create these outputs
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
              nextRunningVarianceTensorAttr]
            = graph->batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(inputDataType);
        yTensorAttr->set_dim(graphTensorBundle.yTensor.dims());
        yTensorAttr->set_stride(graphTensorBundle.yTensor.strides());

        // Only configure batch stats outputs if memory was allocated for them
        if(meanTensorAttr && graphTensorBundle.meanTensor.has_value())
        {
            if(!meanTensorAttr->has_uid())
            {
                meanTensorAttr->set_uid(uid++);
            }
            if(!invVarianceTensorAttr->has_uid())
            {
                invVarianceTensorAttr->set_uid(uid++);
            }
            
            meanTensorAttr->set_data_type(intermediateDataType);
            meanTensorAttr->set_dim(graphTensorBundle.meanTensor->dims());
            meanTensorAttr->set_stride(graphTensorBundle.meanTensor->strides());
            
            invVarianceTensorAttr->set_data_type(intermediateDataType);
            invVarianceTensorAttr->set_dim(graphTensorBundle.invVarianceTensor->dims());
            invVarianceTensorAttr->set_stride(graphTensorBundle.invVarianceTensor->strides());
        }

        // Only configure running stats outputs if memory was allocated for them
        if(nextRunningMeanTensorAttr && graphTensorBundle.runningMeanTensor.has_value())
        {
            if(!nextRunningMeanTensorAttr->has_uid())
            {
                nextRunningMeanTensorAttr->set_uid(uid++);
            }
            if(!nextRunningVarianceTensorAttr->has_uid())
            {
                nextRunningVarianceTensorAttr->set_uid(uid++);
            }
            
            nextRunningMeanTensorAttr->set_data_type(intermediateDataType);
            nextRunningMeanTensorAttr->set_dim(graphTensorBundle.runningMeanTensor->dims());
            nextRunningMeanTensorAttr->set_stride(graphTensorBundle.runningMeanTensor->strides());
            
            nextRunningVarianceTensorAttr->set_data_type(intermediateDataType);
            nextRunningVarianceTensorAttr->set_dim(graphTensorBundle.runningVarianceTensor->dims());
            nextRunningVarianceTensorAttr->set_stride(
                graphTensorBundle.runningVarianceTensor->strides());
        }

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

        auto variantPack = graphTensorBundle.toDeviceVariantPack(*xTensorAttr,
                                                                  *scaleTensorAttr,
                                                                  *biasTensorAttr,
                                                                  prevRunningMeanTensorAttr,
                                                                  prevRunningVarianceTensorAttr,
                                                                  momentumTensorAttr,
                                                                  *epsilonTensorAttr,
                                                                  *yTensorAttr,
                                                                  meanTensorAttr,
                                                                  invVarianceTensorAttr,
                                                                  nextRunningMeanTensorAttr,
                                                                  nextRunningVarianceTensorAttr);

        result = graph->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuBatchnormFwd(GraphTensorBundle& cpuTensorBundle,
                            [[maybe_unused]] BatchnormTrainingScenario scenario)
    {
        auto epsilon
            = static_cast<IntermediateType>(cpuTensorBundle.epsilonTensor.memory().hostData()[0]);

        double momentum = 0.0;
        if(cpuTensorBundle.momentumTensor.has_value())
        {
            momentum = static_cast<double>(
                static_cast<IntermediateType>(
                    cpuTensorBundle.momentumTensor->memory().hostData()[0]));
        }

        // Prepare pointers for optional outputs
        PinnedTensor<IntermediateType>* meanPtr
            = cpuTensorBundle.meanTensor.has_value() ? &(*cpuTensorBundle.meanTensor) : nullptr;
        PinnedTensor<IntermediateType>* invVariancePtr
            = cpuTensorBundle.invVarianceTensor.has_value() ? &(*cpuTensorBundle.invVarianceTensor)
                                                             : nullptr;
        PinnedTensor<IntermediateType>* runningMeanPtr
            = cpuTensorBundle.runningMeanTensor.has_value()
                  ? &(*cpuTensorBundle.runningMeanTensor)
                  : nullptr;
        PinnedTensor<IntermediateType>* runningVariancePtr
            = cpuTensorBundle.runningVarianceTensor.has_value()
                  ? &(*cpuTensorBundle.runningVarianceTensor)
                  : nullptr;

        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormFwdTraining(
            cpuTensorBundle.xTensor,
            cpuTensorBundle.scaleTensor,
            cpuTensorBundle.biasTensor,
            cpuTensorBundle.yTensor,
            epsilon,
            static_cast<IntermediateType>(momentum),
            meanPtr,
            invVariancePtr,
            runningMeanPtr,
            runningVariancePtr,
            runningMeanPtr,  // Use same buffer for prev and next
            runningVariancePtr);
    }

    void runBatchnormTest(InputType tolerance,
                          BatchnormTrainingScenario scenario,
                          const TensorLayout& layout = TensorLayout::NCHW)
    {
        TestCaseType testCase = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        GraphTensorBundle graphTensorBundle(testCase.getDims(), scenario, testCase.seed, layout);
        GraphTensorBundle cpuTensorBundle(testCase.getDims(), scenario, testCase.seed, layout);

        runMiopenBatchnormFwd(graphTensorBundle, inputDataType, intermediateDataType, scenario);

        // Mark modified tensors based on scenario
        graphTensorBundle.yTensor.memory().markDeviceModified();

        if(graphTensorBundle.meanTensor.has_value())
        {
            graphTensorBundle.meanTensor->memory().markDeviceModified();
        }
        if(graphTensorBundle.invVarianceTensor.has_value())
        {
            graphTensorBundle.invVarianceTensor->memory().markDeviceModified();
        }
        if(graphTensorBundle.runningMeanTensor.has_value())
        {
            graphTensorBundle.runningMeanTensor->memory().markDeviceModified();
        }
        if(graphTensorBundle.runningVarianceTensor.has_value())
        {
            graphTensorBundle.runningVarianceTensor->memory().markDeviceModified();
        }

        runCpuBatchnormFwd(cpuTensorBundle, scenario);

        // Validate outputs based on scenario
        CpuFpReferenceMiopenRmsValidation<InputType> cpuRefValidation(tolerance);
        CpuFpReferenceMiopenRmsValidation<IntermediateType> cpuRefValidationStats(
            static_cast<IntermediateType>(tolerance));

        // Y output is always present
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));

        // Validate batch statistics if present (not for BASIC_FORWARD)
        if(scenario != BatchnormTrainingScenario::BASIC_FORWARD)
        {
            EXPECT_TRUE(cpuRefValidationStats.allClose(cpuTensorBundle.meanTensor->memory(),
                                                       graphTensorBundle.meanTensor->memory()));
            EXPECT_TRUE(
                cpuRefValidationStats.allClose(cpuTensorBundle.invVarianceTensor->memory(),
                                               graphTensorBundle.invVarianceTensor->memory()));
        }

        // Validate running statistics if present
        if(scenario == BatchnormTrainingScenario::WITH_RUNNING_STATS
           || scenario == BatchnormTrainingScenario::FULL_TRAINING)
        {
            EXPECT_TRUE(
                cpuRefValidationStats.allClose(cpuTensorBundle.runningMeanTensor->memory(),
                                               graphTensorBundle.runningMeanTensor->memory()));
            EXPECT_TRUE(
                cpuRefValidationStats.allClose(cpuTensorBundle.runningVarianceTensor->memory(),
                                               graphTensorBundle.runningVarianceTensor->memory()));
        }
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
        // {2, 3, 1, 1, seed},
        // {32, 1, 14, 14, seed},
        // {32, 3, 1, 14, seed},
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

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp32, BasicForward)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::BASIC_FORWARD,
                     TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp32, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp32, WithRunningStats)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::WITH_RUNNING_STATS,
                     TensorLayout::NCHW);
}

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp32, WithBatchStats)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::WITH_BATCH_STATS,
                     TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwFp32,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwBfp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwBfp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNchwFp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNchwFp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcFp32, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcFp32,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcBfp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcBfp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNhwcFp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNhwcFp16,
                         testing::ValuesIn(getBnFwdTrainingTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwFp32, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwFp32,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwBfp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwBfp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNcdhwFp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNcdhwFp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcFp32, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<float>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcFp32,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcBfp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<hip_bfloat16>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcBfp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardTrainingNdhwcFp16, FullTraining)
{
    runBatchnormTest(batchnorm::getRmsToleranceTraining<half>(),
                     BatchnormTrainingScenario::FULL_TRAINING,
                     TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardTrainingNdhwcFp16,
                         testing::ValuesIn(getBnFwdTraining3dTestCases()));
