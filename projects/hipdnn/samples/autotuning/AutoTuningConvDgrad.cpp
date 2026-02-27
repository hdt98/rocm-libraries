// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// @file AutoTuningConvDgrad.cpp
/// @brief Sample demonstrating the high-level autotuning API for convolution backward data (dgrad).
///
/// This sample shows how to use the three autotuning convenience methods on Graph:
///   Mode A - autotune_and_select(): autotune + select winner for subsequent execute() calls
///   Mode B - autotune(): benchmark all candidate plans and receive sorted results
///   Mode C - autotune_and_save(): autotune + save winner to engine override config JSON
///
/// Conv dgrad is used because it is more likely to produce non-zero workspace sizes,
/// making it a better demonstration of workspace handling.

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

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

int main(int argc, char* argv[])
{
    auto config = parseCommandLineArgs(argc, argv);
    (void)config; // cpuValidation not used in autotuning sample

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    std::cout << "=== hipDNN Auto-Tuning Sample: Convolution Backward Data (Dgrad) ===\n\n";

    // ── Problem setup (same dimensions as ConvDgrad.cpp sample) ─────────────

    const auto inputType = DataType::HALF;

    constexpr int64_t n = 16; // Batch size
    constexpr int64_t c = 16; // Number of dx channels
    constexpr int64_t h = 16; // Height
    constexpr int64_t w = 16; // Width
    constexpr int64_t k = 16; // Number of dy channels
    constexpr int64_t r = 3; // Filter height
    constexpr int64_t s = 3; // Filter width
    constexpr int64_t u = 1; // Height stride
    constexpr int64_t v = 1; // Width stride
    constexpr int64_t padH = 1;
    constexpr int64_t padW = 1;
    constexpr int64_t dilH = 1;
    constexpr int64_t dilW = 1;

    TensorLayout layout = TensorLayout::NCHW;

    // Output (dy dimensions) - computed based on input and conv parameters
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    std::cout << "Problem: Conv2D Dgrad dx=" << n << "x" << c << "x" << h << "x" << w
              << ", filter=" << k << "x" << c << "x" << r << "x" << s << ", dy=" << n << "x" << k
              << "x" << outH << "x" << outW << " (pad=" << padH << ", stride=" << u
              << ", dilation=" << dilH << ")\n\n";

    // ── Build the graph ──────────────────────────────────────────────────────

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(DataType::FLOAT);

    auto dyAttr = createTensor({n, k, outH, outW}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvDgradAttributes convAttributes;
    convAttributes.set_name("conv_dgrad_autotune");
    convAttributes.set_pre_padding({padH, padW});
    convAttributes.set_post_padding({padH, padW});
    convAttributes.set_stride({u, v});
    convAttributes.set_dilation({dilH, dilW});

    auto dxAttr = graph->conv_dgrad(dyAttr, wAttr, convAttributes);
    dxAttr->set_output(true);

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

    Tensor<half> dyTensor(dyAttr->get_dim(), layout);
    Tensor<half> wTensor(wAttr->get_dim(), layout);
    Tensor<half> dxTensor(dxAttr->get_dim(), layout);

    dyTensor.fillWithRandomValues(static_cast<half>(0.0f), static_cast<half>(1.0f));
    wTensor.fillWithRandomValues(static_cast<half>(0.0f), static_cast<half>(1.0f));
    dxTensor.fillWithValue(static_cast<half>(0.0f));

    // Use the convenience method to get max workspace across all plans
    auto maxWorkspaceSize = graph->get_max_workspace_size();
    std::cout << "Max workspace size across all plans: " << maxWorkspaceSize << " bytes\n\n";
    Workspace workspace(static_cast<size_t>(maxWorkspaceSize));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[dyAttr->get_uid()] = dyTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[dxAttr->get_uid()] = dxTensor.memory().deviceData();

    // ── Mode B: Benchmark all plans (autotune) ──────────────────────────────

    std::cout << "--- Mode B: Benchmarking all candidate plans ---\n\n";

    AutotuneConfig tuneConfig;
    tuneConfig.warmupIterations = 1;
    tuneConfig.timedIterations = 10;

    std::vector<AutotuneResult> results;
    HIPDNN_FE_CHECK(graph->autotune(handle, variantPack, workspace.get(), results, tuneConfig));

    // Print ranked results table
    std::cout << std::left << std::setw(6) << "Rank" << std::setw(8) << "Index" << std::setw(40)
              << "Engine Name" << std::setw(15) << "Avg Time (ms)" << std::setw(20)
              << "Workspace (bytes)" << "Status" << "\n";
    std::cout << std::string(100, '-') << "\n";

    int rank = 1;
    for(const auto& r : results)
    {
        std::cout << std::left << std::setw(6) << rank++ << std::setw(8) << r.planIndex
                  << std::setw(40) << r.engineName << std::setw(15) << std::fixed
                  << std::setprecision(4) << (r.succeeded ? r.avgTimeMs : 0.0f) << std::setw(20)
                  << r.workspaceSize << (r.succeeded ? "OK" : "FAILED") << "\n";
    }
    std::cout << "\n";

    // ── Mode A: Autotune + select winner ────────────────────────────────────

    std::cout << "--- Mode A: Autotuning and selecting winner ---\n";

    // Need to rebuild all plans since autotune consumed them via execute_plan_at_index
    // (select_plan is destructive). We rebuild for Mode A demonstration.
    HIPDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy::ALL));

    HIPDNN_FE_CHECK(graph->autotune_and_select(handle, variantPack, workspace.get(), tuneConfig));

    // Reset output tensor and re-execute with the selected winner
    dxTensor.fillWithValue(static_cast<half>(0.0f));

    int64_t winnerWorkspaceSize;
    HIPDNN_FE_CHECK(graph->get_workspace_size(winnerWorkspaceSize));
    Workspace winnerWorkspace(static_cast<size_t>(winnerWorkspaceSize));

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, winnerWorkspace.get()));
    std::cout << "Re-execution with winner plan completed successfully.\n\n";

    // ── Mode C: Autotune + save winner to JSON ──────────────────────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    std::cout << "--- Mode C: Autotuning and saving winner to engine override config ---\n";

    // Rebuild plans again for Mode C
    HIPDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy::ALL));

    std::string outputPath = "hipdnn_autotune_dgrad_override.json";
    HIPDNN_FE_CHECK(
        graph->autotune_and_save(handle, variantPack, workspace.get(), outputPath, tuneConfig));

    std::cout << "Engine override config saved to: " << outputPath << "\n";
    std::cout << "To use: export HIPDNN_ENGINE_OVERRIDE_FILE=" << outputPath << "\n";
#else
    std::cout << "--- Mode C: Skipped (JSON support not compiled in) ---\n";
#endif

    std::cout << "\n=== Auto-tuning complete ===\n";

    HIPDNN_CHECK(hipdnnDestroy(handle));

    return 0;
}
