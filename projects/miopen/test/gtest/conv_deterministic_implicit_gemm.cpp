/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// ALMIOPEN-718: GemmKBlock threshold experiment for WrwV4R4Xdlops determinism.
//
// Goal: Prove that the determinism threshold is GemmKBlock > 2 (not > 1).
// ~33 configs x 3 data types (FP32, FP16, BF16) = ~99 tests.
// Focus on GKB=1 and GKB=2 boundary with extra configs around anomaly.
// Each test: 10 seeds x 10 iterations = 100 kernel runs.
// Non-fatal assertions (EXPECT) so all tests run for full analysis.

#include <cstring>
#include <gtest/gtest.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "../random.hpp"
#include "get_handle.hpp"
#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"
#include "gtest_common.hpp"

namespace {

using Direction = miopen::conv::Direction;

struct DeterministicTestConfig
{
    size_t N;
    size_t C;
    size_t K;
    size_t H;
    size_t W;
    size_t y;
    size_t x;
    size_t pad_h;
    size_t pad_w;
    size_t stride_h;
    size_t stride_w;
    size_t dilation_h;
    size_t dilation_w;

    friend std::ostream& operator<<(std::ostream& os, const DeterministicTestConfig& tc)
    {
        os << "N:" << tc.N << " C:" << tc.C << " K:" << tc.K;
        os << " H:" << tc.H << " W:" << tc.W;
        os << " y:" << tc.y << " x:" << tc.x;
        os << " pad:{" << tc.pad_h << "," << tc.pad_w << "}";
        os << " stride:{" << tc.stride_h << "," << tc.stride_w << "}";
        os << " dilation:{" << tc.dilation_h << "," << tc.dilation_w << "}";
        return os;
    }
};

/// Compute GemmKBlock using the same algorithm as the solver.
/// Returns GemmKBlock value, or 0 if computation fails.
static int ComputeGemmKBlock(const DeterministicTestConfig& cfg,
                             size_t Ho,
                             size_t Wo,
                             int GemmMPerBlock,
                             int GemmNPerBlock,
                             int GemmKPerBlock,
                             int GemmKPack,
                             int maxComputeUnits)
{
    const int N      = static_cast<int>(cfg.N);
    const int K      = static_cast<int>(cfg.K);
    const int C      = static_cast<int>(cfg.C);
    const int y      = static_cast<int>(cfg.y);
    const int x      = static_cast<int>(cfg.x);
    const int gemm_m = K;
    const int gemm_n = C * y * x;

    if(GemmMPerBlock == 0 || GemmNPerBlock == 0 || GemmKPack == 0)
        return 0;

    const int n_ho_wo = N * static_cast<int>(Ho) * static_cast<int>(Wo);

    if(n_ho_wo % (GemmKPack) != 0)
        return 0;
    if(gemm_m % GemmMPerBlock != 0 || gemm_n % GemmNPerBlock != 0)
        return 0;

    const int grid_size_without_split = (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);
    if(grid_size_without_split == 0)
        return 0;

    const int max_grid_size = 20 * maxComputeUnits;
    int gemm_k_block        = std::max(max_grid_size / grid_size_without_split, 1);
    gemm_k_block            = std::min(gemm_k_block, N);

    for(; gemm_k_block > 1; gemm_k_block--)
    {
        if(N % gemm_k_block != 0)
            continue;
        if(n_ho_wo % (gemm_k_block * GemmKPack) != 0)
            continue;
        const int gemm_k = n_ho_wo / (gemm_k_block * GemmKPack);
        if(gemm_k % GemmKPerBlock != 0)
            continue;
        break;
    }
    return std::max(1, gemm_k_block);
}

// 33 WrwV4R4Xdlops configs focused on GKB=1/2 boundary.
// Config format: {N, C, K, H, W, y, x, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w}
std::vector<DeterministicTestConfig> GetConfigWrwV4R4Xdlops()
{
    return {
        // === GKB=1 (deterministic, no AtomicAdd) ===
        {1, 192, 16, 28, 28, 1, 1, 0, 0, 1, 1, 1, 1}, // [0]
        {1, 128, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // [1]
        {1, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},  // [2]
        {1, 128, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [3]
        {1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1}, // [4]
        {4, 128, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [5]
        {2, 256, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [6]
        {2, 128, 64, 14, 14, 5, 5, 2, 2, 1, 1, 1, 1}, // [7]

        // === GKB=2 (deterministic, FP commutativity) ===
        {8, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1}, // [8]  verified
        {2, 64, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},  // [9]  verified
        {2, 64, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},  // [10] verified
        {16, 64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1}, // [11] verified
        {8, 32, 128, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [12] verified
        {4, 64, 32, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},  // [13]
        {2, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // [14]
        {2, 64, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // [15]

        // === Anomaly region: N=32 C=32 K=64 and variants ===
        // Test 40 FP16 anomaly was here: GKB computed as 2 but FAIL
        {32, 32, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [16] anomaly
        {32, 32, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // [17] anomaly variant
        {32, 64, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [18] anomaly variant
        {32, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [19] anomaly variant
        {16, 32, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [20] anomaly variant
        {16, 32, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // [21] anomaly variant
        {32, 32, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [22] anomaly variant
        {16, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // [23] anomaly variant

        // === GKB=3 boundary (should FAIL) ===
        {3, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},  // [24]
        {3, 128, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // [25]
        {6, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},  // [26]
        {3, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},  // [27]

        // === GKB=4+ (should FAIL) ===
        {4, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},  // [28]
        {4, 32, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},  // [29]
        {8, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},  // [30]
        {16, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // [31]
        {32, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1}, // [32]
    };
}

template <typename T, typename SolverType>
class GPU_WrwXdlDeterminism : public ::testing::TestWithParam<DeterministicTestConfig>
{
protected:
    static constexpr int NUM_ITERATIONS  = 10;
    static constexpr int NUM_INPUT_SEEDS = 10;

    void SetUp() override { prng::reset_seed(); }

    void RunTest()
    {
        const auto& config = GetParam();
        std::cout << "PRNG seed: " << 12345678 << std::endl;
        std::cout << "Testing configuration: " << config << std::endl;
        std::cout << "Data type: "
                  << (sizeof(T) == 4 ? "FP32"
                      : sizeof(T) == 2
                          ? (std::is_same<T, half_float::half>::value ? "FP16" : "BF16")
                          : "???")
                  << std::endl;

        auto& handle = get_handle();

        std::vector<size_t> input_dims  = {config.N, config.C, config.H, config.W};
        std::vector<size_t> weight_dims = {config.K, config.C, config.y, config.x};

        tensor<T> input{miopenTensorNCHW, input_dims};
        tensor<T> weights{miopenTensorNCHW, weight_dims};

        auto conv_desc = miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(config.pad_h), static_cast<int>(config.pad_w)},
            {static_cast<int>(config.stride_h), static_cast<int>(config.stride_w)},
            {static_cast<int>(config.dilation_h), static_cast<int>(config.dilation_w)},
            {0, 0},
            1,
            1.0};

        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{miopenTensorNCHW, output_desc.GetLengths()};

        const auto Ho = output_desc.GetLengths()[2];
        const auto Wo = output_desc.GetLengths()[3];
        std::cout << "Output dims: Ho=" << Ho << " Wo=" << Wo << " N*Ho*Wo=" << config.N * Ho * Wo
                  << std::endl;

        input.generate(
            [](auto...) { return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1)); });
        output.generate(
            [](auto...) { return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1)); });
        std::fill(weights.begin(), weights.end(), T{0});

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(output.data);

        SolverType solv{};
        auto ctx = miopen::ExecutionContext{&handle};

        auto problem = miopen::conv::ProblemDescription{
            output.desc, weights.desc, input.desc, conv_desc, Direction::BackwardWeights};

        problem.SetupFloats(ctx);

        if(!solv.IsApplicable(ctx, problem))
        {
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable on this GPU/config";
        }

        std::cout << "Using solver: " << solv.SolverDbId() << std::endl;

        Workspace wspace{};
        if(solv.MayNeedWorkspace())
        {
            wspace.resize(solv.GetWorkspaceSize(ctx, problem));
        }

        auto perf_config = solv.GetDefaultPerformanceConfig(ctx, problem);
        auto sol         = solv.GetSolution(ctx, problem, perf_config);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        std::cout << "Performance config: " << perf_config << std::endl;

        // Parse perf config to compute GemmKBlock
        {
            std::stringstream ss;
            ss << perf_config;
            std::string pc_str = ss.str();
            std::vector<int> pc_vals;
            std::stringstream ss2(pc_str);
            std::string token;
            while(std::getline(ss2, token, ','))
            {
                try
                {
                    pc_vals.push_back(std::stoi(token));
                }
                catch(...)
                {
                    pc_vals.push_back(0);
                }
            }
            if(pc_vals.size() >= 6)
            {
                int gkb = ComputeGemmKBlock(config,
                                            Ho,
                                            Wo,
                                            pc_vals[0],
                                            pc_vals[1],
                                            pc_vals[2],
                                            pc_vals[5],
                                            static_cast<int>(handle.GetMaxComputeUnits()));
                std::cout << "GemmKBlock=" << gkb << std::endl;
            }
        }

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);

        int seeds_passed = 0;
        int seeds_failed = 0;

        for(int seed = 0; seed < NUM_INPUT_SEEDS; ++seed)
        {
            prng::reset_seed(seed * 17 + 42);

            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            });
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            });

            in_dev  = handle.Write(input.data);
            wei_dev = handle.Write(weights.data);
            out_dev = handle.Write(output.data);

            std::vector<std::vector<T>> results;
            results.reserve(NUM_ITERATIONS);

            for(int i = 0; i < NUM_ITERATIONS; ++i)
            {
                std::fill(weights.begin(), weights.end(), T(0));
                wei_dev = handle.Write(weights.data);

                auto invoke_params =
                    miopen::conv::WrWInvokeParams{miopen::ConvWrwTensors{output.desc,
                                                                         out_dev.get(),
                                                                         input.desc,
                                                                         in_dev.get(),
                                                                         weights.desc,
                                                                         wei_dev.get()},
                                                  wspace.ptr(),
                                                  wspace.size(),
                                                  false};
                (invoker)(handle, invoke_params);
                handle.Finish();

                handle.ReadToVec(wei_dev, weights.data);
                results.push_back(weights.data);
            }

            const auto& reference = results[0];
            bool match            = true;
            size_t mismatch_count = 0;
            for(int i = 1; i < NUM_ITERATIONS; ++i)
            {
                const auto& current = results[i];
                if(reference.size() != current.size())
                {
                    match = false;
                    break;
                }
                for(size_t j = 0; j < reference.size(); ++j)
                {
                    if(std::memcmp(&reference[j], &current[j], sizeof(T)) != 0)
                    {
                        if(match)
                        {
                            std::cout << "  seed " << seed << ": MISMATCH at iteration " << i
                                      << ", element " << j << "/" << reference.size()
                                      << ": ref=" << reference[j] << " cur=" << current[j]
                                      << std::endl;
                        }
                        match = false;
                        mismatch_count++;
                    }
                }
            }

            if(match)
            {
                std::cout << "  seed " << seed << ": PASS (all " << NUM_ITERATIONS
                          << " iterations bit-exact)" << std::endl;
                seeds_passed++;
            }
            else
            {
                std::cout << "  seed " << seed << ": FAIL (" << mismatch_count
                          << " mismatches across iterations)" << std::endl;
                seeds_failed++;
            }
        }

        std::cout << "Summary: " << seeds_passed << " seeds PASS, " << seeds_failed
                  << " seeds FAIL out of " << NUM_INPUT_SEEDS << std::endl;

        EXPECT_EQ(seeds_failed, 0) << seeds_failed << " out of " << NUM_INPUT_SEEDS
                                   << " seeds had non-deterministic results";
    }
};

} // namespace

// FP32
using GPU_WrwXdl_FP32 =
    GPU_WrwXdlDeterminism<float, miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;
TEST_P(GPU_WrwXdl_FP32, DeterministicTest) { this->RunTest(); };
INSTANTIATE_TEST_SUITE_P(GKBThreshold,
                         GPU_WrwXdl_FP32,
                         testing::ValuesIn(GetConfigWrwV4R4Xdlops()));

// FP16
using GPU_WrwXdl_FP16 =
    GPU_WrwXdlDeterminism<half_float::half, miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;
TEST_P(GPU_WrwXdl_FP16, DeterministicTest) { this->RunTest(); };
INSTANTIATE_TEST_SUITE_P(GKBThreshold,
                         GPU_WrwXdl_FP16,
                         testing::ValuesIn(GetConfigWrwV4R4Xdlops()));

// BF16
using GPU_WrwXdl_BF16 =
    GPU_WrwXdlDeterminism<bfloat16, miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;
TEST_P(GPU_WrwXdl_BF16, DeterministicTest) { this->RunTest(); };
INSTANTIATE_TEST_SUITE_P(GKBThreshold,
                         GPU_WrwXdl_BF16,
                         testing::ValuesIn(GetConfigWrwV4R4Xdlops()));
