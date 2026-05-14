// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Integration tests for the hipDNN autotune API.
// Standalone executable (not GTest): prints PASS/FAIL for each test.
//
// Test 4.1: End-to-end autotune -> execute
// Test 4.2: AUTO vs EXHAUSTIVE comparison
// Test 4.3: Config file persistence round-trip

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Retrieves the engine name for a given engine ID. Returns a hex-formatted
/// fallback string when the engine ID is not in the registered engine map.
static std::string getEngineName(int64_t engineId)
{
    try
    {
        return std::string(utilities::getEngineNameFromId(engineId));
    }
    catch(const std::out_of_range&)
    {
        std::ostringstream oss;
        oss << "engine_0x" << std::hex << std::uppercase << engineId;
        return oss.str();
    }
}

// ─── ConvGraph builder ──────────────────────────────────────────────────────

/// All state needed to run autotune scenarios against a convolution graph.
struct ConvGraphState
{
    std::shared_ptr<graph::Graph> graph;
    std::shared_ptr<graph::Tensor_attributes> xAttr;
    std::shared_ptr<graph::Tensor_attributes> wAttr;
    std::shared_ptr<graph::Tensor_attributes> yAttr;
    std::unique_ptr<utilities::Tensor<float>> xTensor;
    std::unique_ptr<utilities::Tensor<float>> wTensor;
    std::unique_ptr<utilities::Tensor<float>> yTensor;
    std::unordered_map<int64_t, void*> variantPack;
};

/// Creates a convolution fprop graph, validates it, builds the operation graph,
/// and allocates device tensors. Does NOT call build() or autotune().
static ConvGraphState buildConvGraph(hipdnnHandle_t handle)
{
    const int64_t n = 16;
    const int64_t c = 16;
    const int64_t h = 16;
    const int64_t w = 16;

    const int64_t k = 16;
    const int64_t r = 3;
    const int64_t s = 3;

    const int64_t padH = 1;
    const int64_t padW = 1;
    const int64_t strideH = 1;
    const int64_t strideW = 1;
    const int64_t dilH = 1;
    const int64_t dilW = 1;

    const auto inputType = hipdnn_frontend::DataType::FLOAT;
    const auto layout = utilities::TensorLayout::NCHW;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("integration_test_conv_fprop");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({strideH, strideW});
    convAttributes.set_dilation({dilH, dilW});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    auto xTensor = std::make_unique<utilities::Tensor<float>>(xAttr->get_dim(), layout);
    auto wTensor = std::make_unique<utilities::Tensor<float>>(wAttr->get_dim(), layout);
    auto yTensor = std::make_unique<utilities::Tensor<float>>(yAttr->get_dim(), layout);

    xTensor->fillWithRandomValues(0.0f, 1.0f);
    wTensor->fillWithRandomValues(0.0f, 1.0f);
    yTensor->fillWithValue(0.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor->memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor->memory().deviceData();
    variantPack[yAttr->get_uid()] = yTensor->memory().deviceData();

    ConvGraphState state;
    state.graph = std::move(graph);
    state.xAttr = std::move(xAttr);
    state.wAttr = std::move(wAttr);
    state.yAttr = std::move(yAttr);
    state.xTensor = std::move(xTensor);
    state.wTensor = std::move(wTensor);
    state.yTensor = std::move(yTensor);
    state.variantPack = std::move(variantPack);

    return state;
}

// ─── Test 4.1: End-to-End Autotune → Execute ───────────────────────────────

/// Autotunes with AUTO/SINGLE_SHOT, executes, and verifies non-zero output.
static bool testEndToEndAutotuneExecute(hipdnnHandle_t handle)
{
    std::cout << "\n=== Test 4.1: End-to-End Autotune -> Execute ===\n";

    auto state = buildConvGraph(handle);

    HIPDNN_FE_CHECK(state.graph->add_all_engines());

    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_max_workspace_size(maxWs));
    utilities::Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    auto tuneStatus = state.graph->autotune(handle, state.variantPack, workspace.get(), config);
    if(tuneStatus.is_bad())
    {
        std::cout << "  autotune() failed: " << tuneStatus.get_message() << "\n";
        std::cout << "  FAIL\n";
        return false;
    }

    int64_t ws = 0;
    HIPDNN_FE_CHECK(state.graph->get_workspace_size(ws));
    utilities::Workspace execWorkspace(static_cast<size_t>(ws));

    auto execStatus = state.graph->execute(handle, state.variantPack, execWorkspace.get());
    if(execStatus.is_bad())
    {
        std::cout << "  execute() failed: " << execStatus.get_message() << "\n";
        std::cout << "  FAIL\n";
        return false;
    }

    // Verify at least one output element is non-zero
    state.yTensor->memory().markDeviceModified();
    const auto* yHostPtr = state.yTensor->memory().hostData();
    const auto yDims = state.yAttr->get_dim();
    int64_t totalElements = 1;
    for(auto d : yDims)
    {
        totalElements *= d;
    }

    bool foundNonZero = false;
    for(int64_t i = 0; i < totalElements; ++i)
    {
        if(yHostPtr[i] != 0.0f)
        {
            foundNonZero = true;
            break;
        }
    }

    std::cout << "  Output tensor dims: [";
    for(size_t i = 0; i < yDims.size(); ++i)
    {
        std::cout << yDims[i] << (i + 1 < yDims.size() ? ", " : "");
    }
    std::cout << "], total elements: " << totalElements << "\n";

    std::cout << "  Output[0..4]: ";
    for(int i = 0; i < std::min(int64_t{5}, totalElements); ++i)
    {
        std::cout << yHostPtr[i] << " ";
    }
    std::cout << "\n";

    if(!foundNonZero)
    {
        std::cout << "  All output elements are zero (expected non-zero)\n";
        std::cout << "  FAIL\n";
        return false;
    }

    std::cout << "  Non-zero output verified.\n";
    std::cout << "  PASS\n";
    return true;
}

// ─── Test 4.2: AUTO vs EXHAUSTIVE Comparison ───────────────────────────────

/// Runs EXHAUSTIVE autotune and verifies the results vector.
static bool testExhaustiveComparison(hipdnnHandle_t handle)
{
    std::cout << "\n=== Test 4.2: AUTO vs EXHAUSTIVE Comparison ===\n";

    auto state = buildConvGraph(handle);

    HIPDNN_FE_CHECK(state.graph->add_all_engines());

    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_max_workspace_size(maxWs));
    utilities::Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.continueOnPrimingFailure = true;

    std::vector<AutotuneResult> results;
    auto tuneStatus
        = state.graph->autotune(handle, state.variantPack, workspace.get(), config, {}, &results);
    if(tuneStatus.is_bad())
    {
        std::cout << "  autotune() failed: " << tuneStatus.get_message() << "\n";
        std::cout << "  FAIL\n";
        return false;
    }

    // Print results table
    std::cout << "  " << std::left << std::setw(6) << "Rank" << std::setw(30) << "Engine"
              << std::setw(12) << "Min (ms)" << std::setw(12) << "Succeeded" << std::setw(14)
              << "Exhaustive" << "\n";
    std::cout << "  " << std::string(74, '-') << "\n";

    bool anySucceeded = false;
    bool anyRanExhaustive = false;

    for(const auto& result : results)
    {
        const std::string rankStr = result.rank >= 0 ? std::to_string(result.rank) : "FAIL";
        const std::string name
            = result.engineName.empty() ? getEngineName(result.engineId) : result.engineName;

        std::cout << "  " << std::left << std::setw(6) << rankStr << std::setw(30) << name;

        if(result.succeeded)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << result.minTimeMs;
            anySucceeded = true;
        }
        else
        {
            std::cout << std::setw(12) << "n/a";
        }

        std::cout << std::setw(12) << (result.succeeded ? "yes" : "no") << std::setw(14)
                  << (result.ranExhaustive ? "yes" : "no");

        if(!result.succeeded && !result.errorMessage.empty())
        {
            std::cout << " [" << result.errorMessage << "]";
        }
        std::cout << "\n";

        if(result.ranExhaustive)
        {
            anyRanExhaustive = true;
        }
    }

    std::cout << "  Total engines benchmarked: " << results.size() << "\n";

    if(!anySucceeded)
    {
        std::cout << "  No engine succeeded benchmarking.\n";
        std::cout << "  FAIL\n";
        return false;
    }

    if(!anyRanExhaustive)
    {
        std::cout << "  No engine ran exhaustive priming.\n";
        std::cout << "  FAIL\n";
        return false;
    }

    std::cout << "  Verified: at least one succeeded, at least one ran exhaustive.\n";
    std::cout << "  PASS\n";
    return true;
}

// ─── Test 4.3: Config File Persistence Round-Trip ──────────────────────────

/// Autotunes to a config file, then builds a new graph with the override config.
static bool testConfigFilePersistence(hipdnnHandle_t handle)
{
    std::cout << "\n=== Test 4.3: Config File Persistence Round-Trip ===\n";

    const std::filesystem::path configFile = "test_autotune_config.json";

    // Phase 1: Autotune and write config file
    {
        auto state = buildConvGraph(handle);

        HIPDNN_FE_CHECK(state.graph->add_all_engines());

        int64_t maxWs = 0;
        HIPDNN_FE_CHECK(state.graph->get_max_workspace_size(maxWs));
        utilities::Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::AUTO;
        config.strategy = AutotuneStrategy::SINGLE_SHOT;
        config.warmupIterations = 1;

        const AutotuneStorageConfig storageConfig{configFile, false};

        std::vector<AutotuneResult> results;
        auto tuneStatus = state.graph->autotune(
            handle, state.variantPack, workspace.get(), config, storageConfig, &results);
        if(tuneStatus.is_bad())
        {
            std::cout << "  Phase 1: autotune() failed: " << tuneStatus.get_message() << "\n";
            std::cout << "  FAIL\n";
            return false;
        }

        if(!std::filesystem::exists(configFile))
        {
            std::cout << "  Phase 1: Config file was not created at '" << configFile << "'.\n";
            std::cout << "  FAIL\n";
            return false;
        }

        // Find the winning engine
        if(results.empty() || !results[0].succeeded)
        {
            std::cout << "  Phase 1: No successful autotune result.\n";
            std::error_code ec;
            std::filesystem::remove(configFile, ec);
            std::cout << "  FAIL\n";
            return false;
        }

        std::cout << "  Phase 1: Autotune wrote config to '" << configFile << "'.\n";
        std::cout << "  Winning engine: "
                  << (results[0].engineName.empty() ? getEngineName(results[0].engineId)
                                                    : results[0].engineName)
                  << " (min time: " << std::fixed << std::setprecision(4) << results[0].minTimeMs
                  << " ms)\n";
    }

    // Phase 2: Build a new graph using the override config
    bool phase2Passed = false;
    {
        setenv("HIPDNN_ENGINE_OVERRIDE_FILE", configFile.c_str(), 1);

        auto state = buildConvGraph(handle);

        auto buildStatus = state.graph->build(handle);
        if(buildStatus.is_bad())
        {
            std::cout << "  Phase 2: build() with override config failed: "
                      << buildStatus.get_message() << "\n";
        }
        else
        {
            int64_t ws = 0;
            HIPDNN_FE_CHECK(state.graph->get_workspace_size(ws));
            utilities::Workspace execWorkspace(static_cast<size_t>(ws));

            auto execStatus = state.graph->execute(handle, state.variantPack, execWorkspace.get());
            if(execStatus.is_bad())
            {
                std::cout << "  Phase 2: execute() failed: " << execStatus.get_message() << "\n";
            }
            else
            {
                std::cout << "  Phase 2: build() + execute() with override config succeeded.\n";
                phase2Passed = true;
            }
        }

        unsetenv("HIPDNN_ENGINE_OVERRIDE_FILE");
    }

    // Clean up the temp file
    std::error_code ec;
    std::filesystem::remove(configFile, ec);

    if(phase2Passed)
    {
        std::cout << "  Config file cleaned up.\n";
        std::cout << "  PASS\n";
    }
    else
    {
        std::cout << "  FAIL\n";
    }

    return phase2Passed;
}

// ─── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    // Accept and ignore --verify-cpu (passed by add_hipdnn_sample_test())
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "  --verify-cpu  Accepted but ignored (compatibility with test harness)\n"
                      << "  --help, -h    Show this help\n";
            return EXIT_SUCCESS;
        }
        if(arg != "--verify-cpu" && arg != "-vc")
        {
            std::cerr << "Unknown argument: " << arg << " (use --help)\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "hipDNN Autotune Integration Tests\n";

    initializeFrontendLogging();

    // GPU availability check
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        std::cout << "SKIPPED: No GPU devices available.\n";
        return 0;
    }

    hipdnnHandle_t handle = nullptr;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    int passed = 0;
    int failed = 0;

    if(testEndToEndAutotuneExecute(handle))
    {
        ++passed;
    }
    else
    {
        ++failed;
    }

    if(testExhaustiveComparison(handle))
    {
        ++passed;
    }
    else
    {
        ++failed;
    }

    if(testConfigFilePersistence(handle))
    {
        ++passed;
    }
    else
    {
        ++failed;
    }

    HIPDNN_CHECK(hipdnnDestroy(handle));

    std::cout << "\n" << passed << " passed, " << failed << " failed.\n";

    if(failed == 0)
    {
        std::cout << "All integration tests passed.\n";
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Some integration tests FAILED.\n";
        return EXIT_FAILURE;
    }
}
