/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier:  MIT */

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_sdk/plugin/test_utils/MockGraph.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>
#include <miopen/miopen.h>

#include "HipdnnEnginePluginHandle.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "tests/common/ActivationCommon.hpp"
#include "tests/common/ConvolutionCommon.hpp"

#include <unordered_map>

using namespace miopen_legacy_plugin;
using namespace hipdnn_plugin;
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;

class TestMiopenConvFwdBiasActivPlanBuilder : public ::testing::Test
{
protected:
    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _dummyHandle;
};

enum class TypeKey
{
    X,
    W,
    BIAS,
    Y_CONV,
    Y_BIAS,
    Y,
    BIAS_COMPUTE,
    CONV_COMPUTE,
    ACTIV_COMPUTE
};

// NOLINTNEXTLINE(readability-identifier-naming)
const char* to_string(TypeKey tk)
{
    std::vector<const char*> keyMap{"x",
                                    "y",
                                    "bias",
                                    "y_conv",
                                    "y_bias",
                                    "y",
                                    "bias_compute_type",
                                    "conv_compute_type",
                                    "activation_compute_type"};

    return keyMap[static_cast<size_t>(tk)];
}

std::ostream& operator<<(std::ostream& os, TypeKey tk)
{
    std::vector<std::string> keyMap{"x",
                                    "y",
                                    "bias",
                                    "y_conv",
                                    "y_bias",
                                    "y",
                                    "bias_compute_type",
                                    "conv_compute_type",
                                    "activation_compute_type"};
    os << keyMap[static_cast<size_t>(tk)];
    return os;
}

struct ConvolutionBiasActivationTestParam
{
    test_conv_common::ConvTestCase convTestCase;
    hipdnn_sdk::utilities::TensorLayout layout;
    test_activation_common::ActivTestCase activTestCase;
    hipdnn_frontend::DataType defaultDataType;
    std::unordered_map<TypeKey, hipdnn_frontend::DataType> dataTypes;
    std::vector<TypeKey> virtualTensors;
    bool isApplicable;

    std::string label;

    friend std::ostream& operator<<(std::ostream& os, const ConvolutionBiasActivationTestParam& tc)
    {
        using namespace hipdnn_sdk::utilities;
        os << "Conv: " << tc.convTestCase;
        os << "\nlayout: "
           << (tc.layout.name.empty() ? vecToString(tc.layout.strideOrder) : tc.layout.name);
        os << "\nactiv: " << tc.activTestCase;
        os << "\ndefault DataType: " << tc.defaultDataType;
        os << "\nDataTypes:\n";
        for(const auto& [typeKey, dataType] : tc.dataTypes)
        {
            os << "\t" << typeKey << ": " << dataType << "\n";
        }
        os << "Virtual tensors: " << vecToString(tc.virtualTensors) << "\n";

        return os;
    }
};

class TestGpuMiopenConvFwdBiasActivPlanBuilder
    : public ::testing::TestWithParam<ConvolutionBiasActivationTestParam>
{
protected:
    void SetUp() override
    {

        namespace graph = hipdnn_frontend::graph;

        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(miopenCreate(&_handle.miopenHandle), miopenStatusSuccess);
        bool doBias = true;

        auto param = GetParam();
        auto& convTestCase = param.convTestCase;
        auto& activTestCase = param.activTestCase;
        auto& layout = param.layout;

        auto isVirtual = [&param](TypeKey key) {
            return std::find(param.virtualTensors.begin(), param.virtualTensors.end(), key)
                   != param.virtualTensors.end();
        };

        _graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        _graphObj.set_intermediate_data_type(param.defaultDataType);
        _graphObj.set_io_data_type(param.defaultDataType);

        // int64_t uid = 1;

        auto xTensorAttrObj
            = graph::makeTensorAttributes("x",
                                          param.dataTypes[TypeKey::X],
                                          convTestCase.xDims,
                                          generateStrides(convTestCase.xDims, layout.strideOrder));
        // xTensorAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xTensorAttrObj));
        xTensorAttr->set_is_virtual(isVirtual(TypeKey::X));

        auto wTensorAttrObj = graph::makeTensorAttributes(
            "w",
            param.dataTypes[TypeKey::W],
            convTestCase.wDims,
            generateStrides(convTestCase.wDims, param.layout.strideOrder));
        // wTensorAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wTensorAttrObj));
        wTensorAttr->set_is_virtual(isVirtual(TypeKey::W));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(convTestCase.convPrePadding);
        convAttrs.set_post_padding(convTestCase.convPostPadding);
        convAttrs.set_stride(convTestCase.convStride);
        convAttrs.set_dilation(convTestCase.convDilation);
        convAttrs.set_compute_data_type(param.dataTypes[TypeKey::CONV_COMPUTE]);

        auto yConvTensorAttr = _graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yConvTensorAttr->set_data_type(param.dataTypes[TypeKey::Y_CONV]);
        yConvTensorAttr->set_dim(convTestCase.yDims);
        yConvTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yConvTensorAttr->set_is_virtual(isVirtual(TypeKey::Y_CONV));
        // yConvTensorAttr->set_uid(uid++);
        std::shared_ptr<graph::TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            const auto biasDims = getDerivedShape(convTestCase.yDims);

            auto biasTensorAttrObj
                = graph::makeTensorAttributes("bias",
                                              param.dataTypes[TypeKey::BIAS],
                                              biasDims,
                                              generateStrides(biasDims, layout.strideOrder));
            // biasTensorAttr.set_uid(uid++);
            auto biasTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(biasTensorAttrObj));
            biasTensorAttr->set_is_virtual(isVirtual(TypeKey::BIAS));

            graph::PointwiseAttributes biasAttrs;
            biasAttrs.set_name("bias");
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(param.dataTypes[TypeKey::BIAS_COMPUTE]);

            yBiasTensorAttr = _graphObj.pointwise(yConvTensorAttr, biasTensorAttr, biasAttrs);
            yBiasTensorAttr->set_name("y_bias");
            yBiasTensorAttr->set_data_type(param.dataTypes[TypeKey::Y_BIAS]);
            yBiasTensorAttr->set_dim(convTestCase.yDims);
            yBiasTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
            yBiasTensorAttr->set_is_virtual(isVirtual(TypeKey::Y_BIAS));
            // yBiasTensorAttr->set_uid(uid++);
        }

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_name("activation_forward");
        activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
        activAttrs.set_compute_data_type(param.dataTypes[TypeKey::ACTIV_COMPUTE]);
        if(activTestCase.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
        }
        if(activTestCase.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
        }
        if(activTestCase.reluLowerClipSlope.has_value())
        {
            activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
        }
        if(activTestCase.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activTestCase.swishBeta.value());
        }
        if(activTestCase.eluAlpha.has_value())
        {
            activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
        }
        if(activTestCase.softplusBeta.has_value())
        {
            activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
        }

        auto yTensorAttr
            = _graphObj.pointwise(doBias ? yBiasTensorAttr : yConvTensorAttr, activAttrs);
        yTensorAttr->set_data_type(param.dataTypes[TypeKey::Y]);
        yTensorAttr->set_dim(convTestCase.yDims);
        yTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
        yTensorAttr->set_output(true);
        yTensorAttr->set_is_virtual(isVirtual(TypeKey::Y));

        auto status = _graphObj.validate();

        std::cout << static_cast<int>(status.code) << ": " << status.get_message() << "\n";

        ASSERT_TRUE(status.is_good());

        _isApplicable = param.isApplicable;
        // yTensorAttr->set_uid(uid);
    }

    void TearDown() override
    {
        if(_handle.miopenHandle != nullptr)
        {
            EXPECT_EQ(miopenDestroy(_handle.miopenHandle), miopenStatusSuccess);
        }
    }

    MiopenConvFwdBiasActivPlanBuilder _planBuilder;
    HipdnnEnginePluginHandle _handle;

    hipdnn_frontend::graph::Graph _graphObj;
    bool _isApplicable;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> xTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> wTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> convTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yConvTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> biasTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yBiasTensorAttr;
    // std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yTensorAttr;
};

TEST_P(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicable)
{
    auto graphBuffer = _graphObj.buildFlatbufferOperationGraph();
    auto graph = GraphWrapper(graphBuffer.data(), graphBuffer.size());

    EXPECT_EQ(_planBuilder.isApplicable(_handle, graph), _isApplicable);

    // // Test 2
    // EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));

    // // Test 3
    // HipdnnEnginePluginExecutionContext ctx;

    // EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    // EXPECT_TRUE(ctx.hasValidPlan());
}

test_conv_common::ConvTestCase validConvTestCase4d()
{
    return {{2, 3, 4, 4}, {1, 3, 2, 2}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, 0};
}

test_conv_common::ConvTestCase validConvTestCase5d()
{
    return {{2, 3, 4, 4, 4}, {1, 3, 2, 2, 2}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 0};
};

test_activation_common::ActivTestCase validActivTestCase()
{
    return {PointwiseMode::RELU_FWD,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt};
};

// struct ConvolutionBiasActivationTestParam
// {
//     test_conv_common::ConvTestCase convTestCase;
//     hipdnn_sdk::utilities::TensorLayout layout;
//     test_activation_common::ActivTestCase activTestCase;
//     hipdnn_frontend::DataType defaultDataType;
//     std::unordered_map<TypeKey, hipdnn_frontend::DataType> dataTypes;
//     bool isApplicable;

//     std::string label;
// };

std::vector<ConvolutionBiasActivationTestParam> testParams()
{
    using Param = ConvolutionBiasActivationTestParam;

    std::vector<Param> params = {Param{validConvTestCase4d(),
                                       TensorLayout::NCHW,
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       true,
                                       ""},
                                 Param{validConvTestCase5d(),
                                       TensorLayout::NCDHW,
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       true,
                                       ""},
                                 Param{validConvTestCase4d(),
                                       TensorLayout::NHWC,
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       true,
                                       ""},
                                 Param{validConvTestCase5d(),
                                       TensorLayout::NDHWC,
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       true,
                                       ""},
                                 Param{validConvTestCase4d(),
                                       TensorLayout{"", {0, 1, 2, 3}},
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       false,
                                       ""},
                                 Param{validConvTestCase5d(),
                                       TensorLayout{"", {0, 1, 2, 3, 4}},
                                       validActivTestCase(),
                                       hipdnn_frontend::DataType::FLOAT,
                                       {},
                                       {TypeKey::Y_CONV, TypeKey::Y_BIAS},
                                       false,
                                       ""}};

    return params;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdBiasActivPlanBuilder,
                         testing::ValuesIn(testParams()));

// TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsTrueForSupportedGraph)
// {
//     auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
//     hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

//     bool applicable = _planBuilder.isApplicable(_handle, graph);
//     EXPECT_TRUE(applicable);
// }

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForUnsupportedGraph)
{
    {
        auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvFwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvBwdGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
    {
        auto builder = hipdnn_sdk::test_utilities::createValidConvWrwGraph();
        hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

        bool applicable = _planBuilder.isApplicable(_dummyHandle, graph);
        EXPECT_FALSE(applicable);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, IsApplicableReturnsFalseForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        bool applicable = _planBuilder.isApplicable(_dummyHandle, mockGraph);
        EXPECT_FALSE(applicable);
    }
}

// TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableVariousLayouts)
// {
//     using namespace hipdnn_sdk::utilities;

//     std::vector<std::pair<std::vector<int64_t>, bool>> layoutsAndExpectedResults
//         = {{{3, 2, 1, 0}, true},
//            {{3, 0, 2, 1}, true},
//            {{4, 3, 2, 1, 0}, true},
//            {{4, 0, 3, 2, 1}, true},
//            {{0, 1, 2, 3}, false}};

//     for(const auto& [layoutOrder, isApplicable] : layoutsAndExpectedResults)
//     {
//         std::vector<int64_t> xDims(layoutOrder.size(), 16);
//         xDims[0] = 1;
//         auto xStrides = generateStrides(xDims, layoutOrder);
//         std::vector<int64_t> wDims(layoutOrder.size(), 3);
//         wDims[0] = 1;
//         wDims[1] = xDims[1];
//         auto wStrides = generateStrides(wDims, layoutOrder);

//         std::vector<int64_t> convPrePadding(layoutOrder.size() - 2, 0);
//         std::vector<int64_t> convPostPadding(layoutOrder.size() - 2, 0);
//         std::vector<int64_t> convStrides(layoutOrder.size() - 2, 1);
//         std::vector<int64_t> convDilation(layoutOrder.size() - 2, 1);

//         test_conv_common::ConvTestCase convTestCase(std::move(xDims),
//                                                     std::move(wDims),
//                                                     std::move(convPrePadding),
//                                                     std::move(convPostPadding),
//                                                     std::move(convStrides),
//                                                     std::move(convDilation),
//                                                     0);

//         auto yStrides = generateStrides(convTestCase.yDims, layoutOrder);
//         auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph(
//             convTestCase.xDims,
//             xStrides,
//             convTestCase.wDims,
//             wStrides,
//             convTestCase.yDims,
//             yStrides,
//             convTestCase.convPrePadding,
//             convTestCase.convPostPadding,
//             convTestCase.convStride,
//             convTestCase.convDilation);

//         hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

//         EXPECT_EQ(_planBuilder.isApplicable(_handle, graph), isApplicable)
//             << "Layout order " + vecToString(layoutOrder);
//     }
// }

// TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableFailsForVirtualInputOrOutput)
// {
//     auto setNamedTensorVirtuality = [](auto* tensors, const std::string& name, bool isVirtual) {
//         for(hipdnn_sdk::data_objects::TensorAttributes* tensor : *tensors)
//         {
//             if(tensor->name()->str() == name)
//             {
//                 tensor->mutate_virtual_(isVirtual);
//             }
//         }
//     };

//     std::vector<std::string> tensorNames = {"x", "w", "y"};
//     auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
//     auto graph = hipdnn_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());

//     for(const auto& tensorName : tensorNames)
//     {
//         setNamedTensorVirtuality(graph->mutable_tensors(), tensorName, true);

//         auto graphWrapper = GraphWrapper(builder.GetBufferPointer(), builder.GetSize());
//         EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper))
//             << ("With virtual " + tensorName);

//         setNamedTensorVirtuality(graph->mutable_tensors(), tensorName, false);
//     }
// }

// struct ConvolutionBiasActivationTestParam
// {
//     test_conv_common::ConvTestCase convTestCase;
//     hipdnn_sdk::data_objects::DataType input;
//     hipdnn_sdk::data_objects::DataType convolution;
//     hipdnn_sdk::data_objects::DataType bias;
//     hipdnn_sdk::data_objects::DataType activation;
//     bool isApplicable;
// };

// struct ConvFrontendGraph
// {
//     hipdnn_frontend::graph::Graph graphObj;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> xTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> wTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> convTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yConvTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> biasTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yBiasTensorAttr;
//     std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yTensorAttr;

// }

// ConvFrontendGraph
//     graphFromParams(const test_conv_common::ConvTestCase& convTestCase,
//                     ConvolutionBiasActivationDataTypes dataTypes,
//                     bool doBias)
// {
//     ConvFrontendGraph graph;

//     graph.graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

//     int64_t uid = 1;

//     auto xTensorAttr = graph::makeTensorAttributes(
//         "x", dataType, convTestCase.xDims, generateStrides(convTestCase.xDims, layout.strideOrder));
//     // xTensorAttr.set_uid(uid++);
//     graph.xTensorAttr
//         = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xTensorAttr));

//     auto wTensorAttr = graph::makeTensorAttributes(
//         "w", dataType, convTestCase.wDims, generateStrides(convTestCase.wDims, layout.strideOrder));
//     // wTensorAttr.set_uid(uid++);
//     auto graph.wTensorAttr
//         = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(wTensorAttr));

//     hipdnn_frontend::graph::ConvFpropAttributes convAttrs;
//     convAttrs.set_pre_padding(convTestCase.convPrePadding);
//     convAttrs.set_post_padding(convTestCase.convPostPadding);
//     convAttrs.set_stride(convTestCase.convStride);
//     convAttrs.set_dilation(convTestCase.convDilation);
//     convAttrs.set_compute_data_type(dataTypes.convolution);

//     graph.yConvTensorAttr = graphObj.conv_fprop(graph.xAttr, graph.wAttr, convAttrs);
//     yConvTensorAttr->set_data_type(dataType);
//     yConvTensorAttr->set_dim(convTestCase.yDims);
//     yConvTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
//     // yConvTensorAttr->set_uid(uid++);
//     if(doBias)
//     {
//         const auto biasDims = getDerivedShape(convTestCase.yDims);

//         auto biasTensorAttr = hipdnn_frontend::graph::makeTensorAttributes(
//             "bias", dataType, biasDims, generateStrides(biasDims, layout.strideOrder));
//         // biasTensorAttr.set_uid(uid++);
//         graph.biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasTensorAttr));

//         hipdnn_frontend::graph::PointwiseAttributes biasAttrs;
//         biasAttrs.set_name("bias");
//         biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
//         biasAttrs.set_compute_data_type(dataTypes.bias);

//         graph.yBiasAttr = graphObj.pointwise(graph.yConvAttr, graph.biasAttr, biasAttrs);
//         yBiasTensorAttr->set_name("y_bias");
//         yBiasTensorAttr->set_data_type(dataTypes.input);
//         yBiasTensorAttr->set_dim(convTestCase.yDims);
//         yBiasTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
//         // yBiasTensorAttr->set_uid(uid++);
//     }

//     hipdnn_frontend::graph::PointwiseAttributes activAttrs;
//     activAttrs.set_name("activation_forward");
//     activAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activTestCase.mode));
//     activAttrs.set_compute_data_type(dataTypes.activation);
//     if(activTestCase.reluLowerClip.has_value())
//     {
//         activAttrs.set_relu_lower_clip(activTestCase.reluLowerClip.value());
//     }
//     if(activTestCase.reluUpperClip.has_value())
//     {
//         activAttrs.set_relu_upper_clip(activTestCase.reluUpperClip.value());
//     }
//     if(activTestCase.reluLowerClipSlope.has_value())
//     {
//         activAttrs.set_relu_lower_clip_slope(activTestCase.reluLowerClipSlope.value());
//     }
//     if(activTestCase.swishBeta.has_value())
//     {
//         activAttrs.set_swish_beta(activTestCase.swishBeta.value());
//     }
//     if(activTestCase.eluAlpha.has_value())
//     {
//         activAttrs.set_elu_alpha(activTestCase.eluAlpha.value());
//     }
//     if(activTestCase.softplusBeta.has_value())
//     {
//         activAttrs.set_softplus_beta(activTestCase.softplusBeta.value());
//     }

//     graph.yTensorAttr
//         = graphObj.pointwise(doBias ? graph.yBiasTensorAttr : graph.yConvTensorAttr, activAttrs);
//     yTensorAttr->set_data_type(dataTypes.input);
//     yTensorAttr->set_dim(convTestCase.yDims);
//     yTensorAttr->set_stride(generateStrides(convTestCase.yDims, layout.strideOrder));
//     yTensorAttr->set_output(true);
//     // yTensorAttr->set_uid(uid);

//     return graph;
// }

// TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, IsApplicableValidComputeType)
// {
//     using namespace hipdnn_sdk::data_objects;

//     test_conv_common::ConvTestCase convTestCase(
//         {4, 4, 4, 4}, {2, 2, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, 0);

//     std::vector<ComputeTypeTestParams> computeTypes
//         = {{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, true},
//            {DataType::BFLOAT16, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT, true}};

//     auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
//     auto graph = hipdnn_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
// }

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, mockGraph),
                     hipdnn_plugin::HipdnnPluginException);
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));

        EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, mockGraph),
                     hipdnn_plugin::HipdnnPluginException);
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(_planBuilder.getWorkspaceSize(_dummyHandle, graph),
                 hipdnn_plugin::HipdnnPluginException);
}

// BRING THIS ONE BACK
// TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, GetWorkspaceSizeReturnsValueForSupportedGraph)
// {
//     auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
//     hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

//     EXPECT_NO_THROW(_planBuilder.getWorkspaceSize(_handle, graph));
// }

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForWrongNodeCountGraph)
{
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(1));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, ctx),
                     hipdnn_plugin::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
    {
        MockGraph mockGraph;
        EXPECT_CALL(mockGraph, nodeCount()).WillRepeatedly(::testing::Return(4));
        HipdnnEnginePluginExecutionContext ctx;

        EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, mockGraph, ctx),
                     hipdnn_plugin::HipdnnPluginException);
        EXPECT_FALSE(ctx.hasValidPlan());
    }
}

TEST_F(TestMiopenConvFwdBiasActivPlanBuilder, BuildPlanThrowsForUnsupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_THROW(_planBuilder.buildPlan(_dummyHandle, graph, ctx),
                 hipdnn_plugin::HipdnnPluginException);
    EXPECT_FALSE(ctx.hasValidPlan());
}

TEST_F(TestGpuMiopenConvFwdBiasActivPlanBuilder, BuildPlanCreatesValidPlanForSupportedGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidConvFwdBiasActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    HipdnnEnginePluginExecutionContext ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_handle, graph, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}
