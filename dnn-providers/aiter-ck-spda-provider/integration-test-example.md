# AITER CK SDPA POC - Integration Test Plan

## Overview

The AITER SDPA POC plugin (designed in `aiter-hipdnn-integration-design.md`) launches pre-compiled ASM Flash Attention v3 kernels for BF16 on gfx942. The plugin is not yet implemented, but we need a test that exercises the full hipDNN pipeline — graph construction, plugin dispatch, GPU kernel execution, and result validation against a CPU reference. This test will serve as the acceptance criteria for the POC.

## File to Create

**`dnn-providers/aiter-ck-spda-provider/integration_tests/IntegrationGpuAiterSdpaFwd.cpp`**

This file will be created when the plugin is implemented. The test code example below shows the intended implementation.

## Reusable Utilities

| Utility | File | Usage |
|---------|------|-------|
| `makeTensorAttributes()` | `frontend/include/hipdnn_frontend/Utilities.hpp` | Create tensor descriptors from `Tensor<T>` objects |
| `generateStrides()` | `data_sdk/include/hipdnn_data_sdk/utilities/ShapeUtilities.hpp` | Row-major strides from dims |
| `CpuFpReferenceSdpa::forward()` | `test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp` | CPU golden reference |
| `CpuFpReferenceValidation<T>` | `test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp` | `allClose()` comparison |
| `Tensor<T>` / `MigratableMemory<T>` | `data_sdk/include/hipdnn_data_sdk/utilities/Tensor.hpp` | Auto host/device memory |
| `Workspace` | `data_sdk/include/hipdnn_data_sdk/utilities/Workspace.hpp` | RAII device workspace |
| `SKIP_IF_NO_DEVICES()` | `test_sdk/include/hipdnn_test_sdk/utilities/TestUtilities.hpp` | Skip when no GPU |
| `Graph::sdpa()` | `frontend/include/hipdnn_frontend/Graph.hpp:2339` | SDPA graph node API |

## Test Code

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace
{

// ---------------------------------------------------------------------------
// Skip if not running on gfx942 (MI300X)
// ---------------------------------------------------------------------------
#define SKIP_IF_NOT_GFX942()                                                   \
    do                                                                         \
    {                                                                          \
        hipDeviceProp_t props;                                                 \
        if(hipGetDeviceProperties(&props, 0) != hipSuccess)                    \
        {                                                                      \
            GTEST_SKIP() << "Cannot query device properties";                  \
        }                                                                      \
        if(std::string(props.gcnArchName).find("gfx942") == std::string::npos) \
        {                                                                      \
            GTEST_SKIP() << "AITER SPDA POC requires gfx942, found: "          \
                         << props.gcnArchName;                                 \
        }                                                                      \
    } while(0)

// ---------------------------------------------------------------------------
// Plugin path resolution
// ---------------------------------------------------------------------------
std::string getAiterPluginPath()
{
    if(auto* envPath = std::getenv("HIPDNN_AITER_PLUGIN_PATH"))
    {
        return std::string(envPath);
    }
#ifdef AITER_PLUGIN_PATH
    return std::string(AITER_PLUGIN_PATH);
#else
    return std::string("./libaiter_plugin.so");
#endif
}

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------
struct SdpaTestCase
{
    int64_t batch;
    int64_t numHeads;
    int64_t numKvHeads;
    int64_t seqLenQ;
    int64_t seqLenKv;
    int64_t headDim;
    std::string description;

    friend std::ostream& operator<<(std::ostream& os, const SdpaTestCase& tc)
    {
        os << "SdpaTestCase{B=" << tc.batch << ", H=" << tc.numHeads
           << ", Hkv=" << tc.numKvHeads << ", Sq=" << tc.seqLenQ
           << ", Skv=" << tc.seqLenKv << ", D=" << tc.headDim
           << ", desc=" << tc.description << "}";
        return os;
    }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class IntegrationGpuAiterSdpaFwdBfp16 : public ::testing::TestWithParam<SdpaTestCase>
{
protected:
    using InputType = bfloat16;

    struct SdpaTensors
    {
        Tensor<InputType> qTensor;
        Tensor<InputType> kTensor;
        Tensor<InputType> vTensor;
        Tensor<InputType> oTensor;
        Tensor<InputType> oRefTensor;

        SdpaTensors(const std::vector<int64_t>& qDims,
                    const std::vector<int64_t>& kDims,
                    const std::vector<int64_t>& vDims)
            : qTensor(qDims)
            , kTensor(kDims)
            , vTensor(vDims)
            , oTensor({qDims[0], qDims[1], qDims[2], vDims[3]})
            , oRefTensor({qDims[0], qDims[1], qDims[2], vDims[3]})
        {
            auto seed = getGlobalTestSeed();
            qTensor.fillWithRandomValues(
                static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
            kTensor.fillWithRandomValues(
                static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed + 1);
            vTensor.fillWithRandomValues(
                static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed + 2);
            oTensor.fillWithValue(static_cast<InputType>(0.0f));
            oRefTensor.fillWithValue(static_cast<InputType>(0.0f));
        }
    };

    struct SdpaTensorAttrs
    {
        std::shared_ptr<TensorAttributes> q;
        std::shared_ptr<TensorAttributes> k;
        std::shared_ptr<TensorAttributes> v;
        std::shared_ptr<TensorAttributes> o;
    };

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        SKIP_IF_NOT_GFX942();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    void setupPlugin()
    {
        ASSERT_EQ(hipInit(0), hipSuccess);

        int device = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);

        auto pluginPath = getAiterPluginPath();
        const std::array<const char*, 1> paths = {pluginPath.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS)
            << "Failed to load plugin: " << pluginPath;

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    static std::pair<std::shared_ptr<Graph>, SdpaTensorAttrs>
        buildSdpaGraph(const SdpaTensors& tensors)
    {
        auto graph = std::make_shared<Graph>();
        graph->set_name("AiterSdpaFwdTest")
            .set_io_data_type(DataType::BFLOAT16)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        int64_t uid = 1;
        SdpaTensorAttrs attrs;

        auto qAttr = makeTensorAttributes("Q", getDataTypeEnumFromType<InputType>(), tensors.qTensor);
        qAttr.set_uid(uid++);
        attrs.q = std::make_shared<TensorAttributes>(std::move(qAttr));

        auto kAttr = makeTensorAttributes("K", getDataTypeEnumFromType<InputType>(), tensors.kTensor);
        kAttr.set_uid(uid++);
        attrs.k = std::make_shared<TensorAttributes>(std::move(kAttr));

        auto vAttr = makeTensorAttributes("V", getDataTypeEnumFromType<InputType>(), tensors.vTensor);
        vAttr.set_uid(uid++);
        attrs.v = std::make_shared<TensorAttributes>(std::move(vAttr));

        SdpaAttributes sdpaAttrs;
        sdpaAttrs.set_name("AiterSdpaFwd");
        sdpaAttrs.set_causal_mask(false);

        auto [oTensorAttr, statsAttr] = graph->sdpa(attrs.q, attrs.k, attrs.v, sdpaAttrs);
        (void)statsAttr;

        if(!oTensorAttr->has_uid())
        {
            oTensorAttr->set_uid(uid++);
        }

        const auto oDims = tensors.oTensor.dims();
        const auto oStrides = generateStrides(oDims);
        oTensorAttr->set_data_type(DataType::BFLOAT16)
            .set_dim(oDims)
            .set_stride(oStrides)
            .set_output(true);

        attrs.o = oTensorAttr;
        return {graph, attrs};
    }

    static std::unordered_map<int64_t, void*>
        createDeviceVariantPack(const SdpaTensorAttrs& attrs, SdpaTensors& tensors)
    {
        // deviceData() triggers hipMalloc + host-to-device copy
        return {
            {attrs.q->get_uid(), tensors.qTensor.memory().deviceData()},
            {attrs.k->get_uid(), tensors.kTensor.memory().deviceData()},
            {attrs.v->get_uid(), tensors.vTensor.memory().deviceData()},
            {attrs.o->get_uid(), tensors.oTensor.memory().deviceData()},
        };
    }

    void runPipeline(const std::shared_ptr<Graph>& graph,
                     const SdpaTensorAttrs& attrs,
                     SdpaTensors& tensors)
    {
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK)
            << "No engine accepted the SDPA graph. Is the AITER plugin loaded? "
            << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t workspaceSize = 0;
        result = graph->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0);
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = createDeviceVariantPack(attrs, tensors);

        result = graph->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Tell MigratableMemory that device has the latest data
        // so that hostData() triggers device-to-host copy
        tensors.oTensor.memory().markDeviceModified();
    }

    static void validateAgainstCpuReference(SdpaTensors& tensors)
    {
        CpuFpReferenceSdpa::forward(
            tensors.qTensor,
            tensors.kTensor,
            tensors.vTensor,
            tensors.oRefTensor,
            std::nullopt,
            static_cast<const TensorBase<float>*>(nullptr),
            false);

        // BF16 epsilon ~ 0.0078; Flash Attention tiling adds further error
        constexpr float kAbsTol = 1e-2f;
        constexpr float kRelTol = 1e-2f;
        CpuFpReferenceValidation<InputType> validator(kAbsTol, kRelTol);

        EXPECT_TRUE(validator.allClose(tensors.oRefTensor, tensors.oTensor))
            << "GPU output does not match CPU reference (atol=" << kAbsTol
            << ", rtol=" << kRelTol << ")";
    }

    hipdnnHandle_t _handle = nullptr;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test body
// ---------------------------------------------------------------------------
TEST_P(IntegrationGpuAiterSdpaFwdBfp16, ExecuteAndValidate)
{
    const auto& tc = GetParam();

    setupPlugin();

    std::vector<int64_t> qDims = {tc.batch, tc.numHeads, tc.seqLenQ, tc.headDim};
    std::vector<int64_t> kDims = {tc.batch, tc.numKvHeads, tc.seqLenKv, tc.headDim};
    std::vector<int64_t> vDims = {tc.batch, tc.numKvHeads, tc.seqLenKv, tc.headDim};

    SdpaTensors tensors(qDims, kDims, vDims);
    auto [graph, attrs] = buildSdpaGraph(tensors);

    runPipeline(graph, attrs, tensors);
    validateAgainstCpuReference(tensors);
}

// ---------------------------------------------------------------------------
// Test instances
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGpuAiterSdpaFwdBfp16,
    ::testing::Values(
        SdpaTestCase{1, 1, 1, 256, 256, 128, "SmallMha"},
        SdpaTestCase{2, 8, 8, 512, 512, 128, "MediumMha"},
        SdpaTestCase{1, 8, 2, 256, 256, 128, "GqaRatio4"}),
    [](const ::testing::TestParamInfo<SdpaTestCase>& info) {
        return info.param.description;
    });
```

## Key Design Decisions

1. **Device variant pack**: The existing `SdpaFwdTensorBundle::createVariantPack()` uses `hostData()` (for CPU tests). GPU execution requires `deviceData()` which triggers `hipMalloc` + H2D copy via `MigratableMemory`.

2. **Separate `oRefTensor`**: GPU kernel writes to `oTensor` on device. A separate `oRefTensor` holds the CPU reference for comparison.

3. **`markDeviceModified()` after execute**: Tells `MigratableMemory` the device copy is authoritative, so `allClose()`'s internal `hostData()` access triggers D2H copy.

4. **`hipDeviceSynchronize()` after execute**: Ensures kernel completion before reading results.

5. **Tolerance (atol=1e-2, rtol=1e-2)**: BF16 has ~7 mantissa bits. Flash Attention's tiled online-softmax introduces additional numerical divergence from the naive CPU implementation.

6. **Different seeds for Q/K/V**: Using `seed`, `seed+1`, `seed+2` ensures the tensors have distinct random values.

7. **`SKIP_IF_NOT_GFX942` macro**: Local to this test file since no existing utility exists for GPU architecture gating.

## Verification

Once the AITER plugin is implemented:

```bash
cd build
ninja aiter_plugin_integration_tests
HIPDNN_AITER_PLUGIN_PATH=./lib/libaiter_plugin.so \
    ./bin/aiter_plugin_integration_tests --gtest_filter="*AiterSdpa*"
```
