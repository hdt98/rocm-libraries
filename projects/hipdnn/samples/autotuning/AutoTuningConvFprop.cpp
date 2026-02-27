// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file AutoTuningConvFprop.cpp
/// @brief Sample demonstrating the full autotuning workflow for convolution forward propagation.
///
/// This sample shows how to:
///   Mode A - Select the winning plan and re-execute with it
///   Mode B - Benchmark all candidate plans and print a ranked results table
///   Mode C - Save the winning engine to an engine override config file

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>
#include <hipdnn_frontend/detail/EngineOverrideUtils.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;
using namespace hipdnn_data_sdk::utilities;

#define HIP_CHECK_AUTOTUNE(status)                                                             \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

/// Benchmarking result for a single plan
struct PlanBenchmarkResult
{
    int64_t index;
    std::string engineName;
    float avgTimeMs;
    int64_t workspaceSize;
    bool succeeded;
};

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);
    (void)config; // cpuValidation not used in autotuning sample

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    std::cout << "=== hipDNN Auto-Tuning Sample: Convolution Forward Propagation ===\n\n";

    // ── Problem setup ────────────────────────────────────────────────────────

    const auto inputType = DataType::HALF;

    constexpr int64_t n = 16; // Batch size
    constexpr int64_t c = 64; // Input channels
    constexpr int64_t h = 56; // Height
    constexpr int64_t w = 56; // Width
    constexpr int64_t k = 64; // Output channels
    constexpr int64_t r = 3;  // Filter height
    constexpr int64_t s = 3;  // Filter width
    constexpr int64_t padH = 1;
    constexpr int64_t padW = 1;
    constexpr int64_t strideH = 1;
    constexpr int64_t strideW = 1;
    constexpr int64_t dilH = 1;
    constexpr int64_t dilW = 1;

    TensorLayout layout = TensorLayout::NCHW;

    std::cout << "Problem: Conv2D FProp " << n << "x" << c << "x" << h << "x" << w << " * " << k
              << "x" << c << "x" << r << "x" << s << " (pad=" << padH << ", stride=" << strideH
              << ", dilation=" << dilH << ")\n\n";

    // ── Build the graph ──────────────────────────────────────────────────────

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(DataType::FLOAT);

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("conv_fprop_autotune");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({strideH, strideW});
    convAttributes.set_dilation({dilH, dilW});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    // Validate and build the operation graph (but not the execution plans yet)
    HIPDNN_FE_CHECK(graph->validate());
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    // ── Build ALL candidate plans ────────────────────────────────────────────

    HIPDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy::ALL));

    auto planCount = graph->get_execution_plan_count();
    std::cout << "Built " << planCount << " candidate execution plan(s).\n\n";

    if(planCount == 0)
    {
        std::cerr << "No execution plans available. Exiting.\n";
        HIPDNN_CHECK(hipdnnDestroy(handle));
        return 1;
    }

    // ── Allocate tensors and workspace ───────────────────────────────────────

    Tensor<half> xTensor(xAttr->get_dim(), layout);
    Tensor<half> wTensor(wAttr->get_dim(), layout);
    Tensor<half> yTensor(yAttr->get_dim(), layout);

    xTensor.fillWithRandomValues(static_cast<half>(0.0f), static_cast<half>(1.0f));
    wTensor.fillWithRandomValues(static_cast<half>(0.0f), static_cast<half>(1.0f));
    yTensor.fillWithValue(static_cast<half>(0.0f));

    // Find max workspace size across all plans
    int64_t maxWorkspaceSize = 0;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto ws = graph->get_workspace_size_plan_at_index(i);
        if(ws > maxWorkspaceSize)
        {
            maxWorkspaceSize = ws;
        }
    }
    Workspace workspace(static_cast<size_t>(maxWorkspaceSize));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();

    // ── Mode B: Benchmark all plans ──────────────────────────────────────────

    std::cout << "--- Mode B: Benchmarking all candidate plans ---\n\n";

    constexpr int warmupIterations = 1;
    constexpr int timedIterations = 10;

    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    HIP_CHECK_AUTOTUNE(hipEventCreate(&startEvent));
    HIP_CHECK_AUTOTUNE(hipEventCreate(&stopEvent));

    hipStream_t stream = nullptr;
    // Use default stream

    std::vector<PlanBenchmarkResult> results;
    results.reserve(static_cast<size_t>(planCount));

    for(int64_t i = 0; i < planCount; ++i)
    {
        PlanBenchmarkResult result;
        result.index = i;
        result.workspaceSize = graph->get_workspace_size_plan_at_index(i);

        std::string name;
        auto nameErr = graph->get_plan_name_at_index(i, name);
        result.engineName = nameErr.is_good() ? name : "unknown";

        // Warmup
        auto warmupStatus = graph->execute_plan_at_index(handle, variantPack, workspace.get(), i);
        if(warmupStatus.is_bad())
        {
            std::cout << "  Plan " << i << " (" << result.engineName
                      << "): FAILED - " << warmupStatus.get_message() << "\n";
            result.avgTimeMs = std::numeric_limits<float>::max();
            result.succeeded = false;
            results.push_back(std::move(result));
            continue;
        }

        // Additional warmup iterations
        for(int iter = 1; iter < warmupIterations; ++iter)
        {
            graph->execute_plan_at_index(handle, variantPack, workspace.get(), i);
        }
        HIP_CHECK_AUTOTUNE(hipDeviceSynchronize());

        // Timed iterations
        HIP_CHECK_AUTOTUNE(hipEventRecord(startEvent, stream));
        for(int iter = 0; iter < timedIterations; ++iter)
        {
            graph->execute_plan_at_index(handle, variantPack, workspace.get(), i);
        }
        HIP_CHECK_AUTOTUNE(hipEventRecord(stopEvent, stream));
        HIP_CHECK_AUTOTUNE(hipEventSynchronize(stopEvent));

        float totalMs = 0.0f;
        HIP_CHECK_AUTOTUNE(hipEventElapsedTime(&totalMs, startEvent, stopEvent));
        result.avgTimeMs = totalMs / static_cast<float>(timedIterations);
        result.succeeded = true;

        results.push_back(std::move(result));
    }

    // Sort by execution time (ascending)
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.avgTimeMs < b.avgTimeMs;
    });

    // Print ranked results table
    std::cout << "\n"
              << std::left << std::setw(6) << "Rank" << std::setw(8) << "Index"
              << std::setw(40) << "Engine Name" << std::setw(15) << "Avg Time (ms)"
              << std::setw(20) << "Workspace (bytes)"
              << "Status"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    int rank = 1;
    for(const auto& r : results)
    {
        std::cout << std::left << std::setw(6) << rank++ << std::setw(8) << r.index
                  << std::setw(40) << r.engineName << std::setw(15) << std::fixed
                  << std::setprecision(4)
                  << (r.succeeded ? r.avgTimeMs : 0.0f) << std::setw(20) << r.workspaceSize
                  << (r.succeeded ? "OK" : "FAILED") << "\n";
    }
    std::cout << "\n";

    // Find the winner (first succeeded plan, which is already at the front after sort)
    auto winnerIt
        = std::find_if(results.begin(), results.end(), [](const auto& r) { return r.succeeded; });

    if(winnerIt == results.end())
    {
        std::cerr << "No plans succeeded. Exiting.\n";
        HIP_CHECK_AUTOTUNE(hipEventDestroy(startEvent));
        HIP_CHECK_AUTOTUNE(hipEventDestroy(stopEvent));
        HIPDNN_CHECK(hipdnnDestroy(handle));
        return 1;
    }

    std::cout << "Winner: " << winnerIt->engineName << " (index " << winnerIt->index << ", "
              << winnerIt->avgTimeMs << " ms avg)\n\n";

    // ── Mode A: Select winner and re-execute ─────────────────────────────────

    std::cout << "--- Mode A: Selecting winner and re-executing ---\n";

    HIPDNN_FE_CHECK(graph->select_plan(winnerIt->index));

    // Reset output tensor
    yTensor.fillWithValue(static_cast<half>(0.0f));

    int64_t winnerWorkspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(winnerWorkspaceSize));
    Workspace winnerWorkspace(static_cast<size_t>(winnerWorkspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, winnerWorkspace.get()));
    std::cout << "Re-execution with winner plan completed successfully.\n\n";

    // ── Mode C: Save winner to engine override config ────────────────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    std::cout << "--- Mode C: Saving winner to engine override config ---\n";

    auto opInfo = engine_override::extractOperationInfo(*graph);
    if(opInfo.has_value())
    {
        engine_override::OperationRule rule;
        rule.op = opInfo->op;
        rule.engineName = winnerIt->engineName;
        rule.tensors = opInfo->tensors;

        engine_override::EngineOverrideConfig overrideConfig;
        overrideConfig.addRule(std::move(rule));

        std::string outputPath = "hipdnn_autotune_override.json";
        if(overrideConfig.save(outputPath))
        {
            std::cout << "Engine override config saved to: " << outputPath << "\n";
            std::cout << "To use: export HIPDNN_ENGINE_OVERRIDE_FILE=" << outputPath << "\n";
        }
        else
        {
            std::cerr << "Failed to save engine override config.\n";
        }
    }
    else
    {
        std::cerr << "Could not extract operation info from graph.\n";
    }
#else
    std::cout << "--- Mode C: Skipped (JSON support not compiled in) ---\n";
#endif

    std::cout << "\n=== Auto-tuning complete ===\n";

    HIP_CHECK_AUTOTUNE(hipEventDestroy(startEvent));
    HIP_CHECK_AUTOTUNE(hipEventDestroy(stopEvent));
    HIPDNN_CHECK(hipdnnDestroy(handle));

    return 0;
}
