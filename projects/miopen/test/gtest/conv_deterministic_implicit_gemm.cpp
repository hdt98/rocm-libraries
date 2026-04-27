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

// ALMIOPEN-718: Bit-exact determinism tests for implicit GEMM convolution solvers.
// Runs each solver 10 times with identical input and verifies bit-exact output match.
//
// Each solver gets its own set of configs that are known to be applicable.
// Configs chosen to satisfy each solver's GEMM dimension divisibility requirements.

#include <cstring>
#include <cstdlib>
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

// ~100 test configs total. Each solver gets 8-20 configs to systematically test
// determinism across different parameters (kernel size, stride, dilation, batch, channels).
// BwdV1R1 gets extra configs to map the AtomicAdd determinism boundary.
//
// Config format: {N, C, K, H, W, y, x, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w}

// FwdV4R5Xdlops: xdlops forward (~20 configs)
std::vector<DeterministicTestConfig> GetConfigFwdV4R5Xdlops()
{
    return {
        // Original + varied batch/channel sizes
        {128, 16, 64, 54, 54, 3, 3, 1, 1, 1, 1, 1, 1}, // Jira original
        {64, 16, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1},  // smaller batch
        {32, 32, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1}, // larger C/K
        {16, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},  // small spatial
        {8, 32, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},   // small batch
        // Stride and dilation variations
        {32, 16, 64, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},  // stride 2
        {16, 32, 64, 28, 28, 3, 3, 0, 0, 1, 1, 2, 2},  // dilation 2
        {64, 16, 128, 16, 16, 3, 3, 0, 0, 1, 1, 1, 1}, // no padding
        // 5x5 kernel
        {16, 16, 64, 16, 16, 5, 5, 2, 2, 1, 1, 1, 1}, // 5x5 kernel
        {32, 16, 64, 28, 28, 5, 5, 2, 2, 1, 1, 1, 1}, // 5x5 larger
        // Additional configs
        {4, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 32, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 64, 14, 14, 3, 3, 0, 0, 1, 1, 1, 1},
        {8, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {2, 32, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 16, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {4, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 16, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 32, 128, 14, 14, 3, 3, 1, 1, 2, 2, 1, 1},
    };
}

// FwdV4R4Xdlops: xdlops forward (~15 configs)
std::vector<DeterministicTestConfig> GetConfigFwdV4R4Xdlops()
{
    return {
        {128, 48, 192, 13, 13, 1, 1, 0, 0, 1, 1, 1, 1},
        {256, 32, 128, 27, 27, 1, 1, 0, 0, 1, 1, 1, 1},
        {64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {32, 128, 64, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 64, 128, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {128, 16, 64, 54, 54, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 128, 28, 28, 1, 1, 0, 0, 2, 2, 1, 1},
        {32, 16, 64, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},
        // Additional
        {8, 128, 128, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 32, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {4, 64, 192, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {32, 32, 128, 14, 14, 3, 3, 0, 0, 1, 1, 1, 1},
        {64, 16, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
    };
}

// FwdV4R1: non-xdlops forward (~15 configs)
std::vector<DeterministicTestConfig> GetConfigFwdV4R1()
{
    return {
        {256, 32, 128, 27, 27, 1, 1, 0, 0, 1, 1, 1, 1},
        {128, 48, 192, 13, 13, 1, 1, 0, 0, 1, 1, 1, 1},
        {64, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {32, 32, 128, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {128, 16, 64, 54, 54, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 128, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},
        {128, 64, 64, 28, 28, 1, 1, 0, 0, 2, 2, 1, 1},
        {32, 16, 64, 28, 28, 5, 5, 2, 2, 1, 1, 1, 1},
        // Additional
        {16, 128, 64, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {8, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {4, 32, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 16, 128, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 64, 64, 28, 28, 3, 3, 0, 0, 1, 1, 1, 1},
    };
}

// BwdV1R1: CRITICAL — tests the AtomicAdd boundary (~30 configs)
// Rule: deterministic when stride >= dilation*(kernel_size-1)+1
std::vector<DeterministicTestConfig> GetConfigBwdV1R1()
{
    return {
        // === 1x1 kernel: always deterministic ===
        {32, 128, 12, 32, 32, 1, 1, 0, 0, 1, 1, 1, 1},
        {64, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 128, 64, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {8, 64, 32, 32, 32, 1, 1, 0, 0, 1, 1, 1, 1},
        {32, 32, 64, 16, 16, 1, 1, 0, 0, 2, 2, 1, 1},
        {4, 256, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {4, 128, 128, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {2, 64, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {128, 32, 64, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 64, 32, 28, 28, 1, 1, 0, 0, 1, 1, 1, 1},
        // === 3x3 stride=1: NON-DET (1<3) ===
        {16, 64, 64, 16, 16, 3, 3, 0, 0, 1, 1, 1, 1},
        {32, 32, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {4, 64, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 32, 8, 8, 3, 3, 0, 0, 1, 1, 1, 1},
        {2, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        // === 3x3 stride=2: NON-DET (2<3) ===
        {16, 64, 32, 16, 16, 3, 3, 1, 1, 2, 2, 1, 1},
        {8, 32, 64, 16, 16, 3, 3, 1, 1, 2, 2, 1, 1},
        // === 3x3 stride=3: DET (3>=3) ===
        {16, 64, 32, 16, 16, 3, 3, 0, 0, 3, 3, 1, 1},
        // === 5x5 stride=1: NON-DET (1<5) ===
        {16, 32, 64, 16, 16, 5, 5, 2, 2, 1, 1, 1, 1},
        {8, 64, 32, 14, 14, 5, 5, 2, 2, 1, 1, 1, 1},
        // === 5x5 stride=5: DET (5>=5) ===
        {8, 32, 64, 16, 16, 5, 5, 0, 0, 5, 5, 1, 1},
        // === dilation=2 ===
        {16, 64, 32, 16, 16, 3, 3, 0, 0, 1, 1, 2, 2},
        {8, 64, 32, 28, 28, 3, 3, 0, 0, 5, 5, 2, 2},
        // === 7x7 kernel ===
        {4, 32, 64, 16, 16, 7, 7, 3, 3, 1, 1, 1, 1},
    };
}

// BwdV4R1: disabled by default, enabled via env var (~15 configs)
std::vector<DeterministicTestConfig> GetConfigBwdV4R1()
{
    return {
        {16, 64, 64, 16, 16, 3, 3, 0, 0, 1, 1, 1, 1},
        {32, 32, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 128, 64, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {16, 64, 32, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},
        {32, 32, 64, 16, 16, 1, 1, 0, 0, 2, 2, 1, 1},
        {8, 32, 64, 16, 16, 5, 5, 2, 2, 1, 1, 1, 1},
        {16, 64, 32, 14, 14, 3, 3, 0, 0, 1, 1, 2, 2},
        // Additional
        {4, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {128, 32, 32, 8, 8, 1, 1, 0, 0, 1, 1, 1, 1},
        {8, 64, 64, 14, 14, 3, 3, 0, 0, 1, 1, 1, 1},
        {16, 32, 128, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {4, 128, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
    };
}

// WrwV4R4: non-xdlops weight gradient (~15 configs)
std::vector<DeterministicTestConfig> GetConfigWrwV4R4()
{
    return {
        {8, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {4, 128, 128, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 32, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 64, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {8, 128, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {4, 256, 64, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 64, 32, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},
        {4, 64, 64, 14, 14, 5, 5, 2, 2, 1, 1, 1, 1},
        // Additional
        {2, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {64, 32, 32, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1},
        {1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 32, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},
        {8, 32, 32, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},
    };
}

// WrwV4R4Xdlops: xdlops weight gradient (~30 configs)
// FOCUS: probe GemmKBlock > 1 boundary (gemm_k = N*Ho*Wo)
std::vector<DeterministicTestConfig> GetConfigWrwV4R4Xdlops()
{
    return {
        // === Original configs ===
        {1, 192, 16, 28, 28, 1, 1, 0, 0, 1, 1, 1, 1},  // Jira original PASS
        {4, 128, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // PASS
        {8, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1},   // FAIL (GemmKBlock>1)
        {8, 128, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},  // PASS
        {4, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},   // FAIL
        {16, 32, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1}, // FAIL
        {2, 256, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // PASS
        {1, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1},   // PASS
        {4, 64, 32, 28, 28, 3, 3, 1, 1, 2, 2, 1, 1},   // PASS
        {2, 128, 64, 14, 14, 5, 5, 2, 2, 1, 1, 1, 1},  // PASS
        // === Probing GemmKBlock boundary: vary N and spatial size ===
        // Small N*Ho*Wo → likely GemmKBlock=1 → deterministic
        {1, 128, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // N*Ho*Wo=196
        {1, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // 196
        {2, 64, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // 392
        {1, 128, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // 256
        // Medium N*Ho*Wo → GemmKBlock boundary
        {4, 64, 32, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // 784
        {2, 64, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // 512
        {4, 32, 64, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // 1024
        {8, 32, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // 1568
        // Large N*Ho*Wo → likely GemmKBlock>1 → non-deterministic
        {16, 64, 32, 16, 16, 1, 1, 0, 0, 1, 1, 1, 1}, // 4096
        {32, 32, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1}, // 6272
        {8, 64, 64, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},  // 1568
        // === 3x3 with varied N ===
        {1, 128, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1}, // small N
        {2, 64, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {16, 64, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {32, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        // === Different K values ===
        {4, 128, 128, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {4, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1},
        {8, 32, 128, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1},
        {1, 64, 128, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1},
    };
}

template <typename T, Direction CONV_DIR, typename SolverType>
class GPU_ConvDeterministicImplicitGemm : public ::testing::TestWithParam<DeterministicTestConfig>
{
protected:
    static constexpr int NUM_ITERATIONS = 10;

    void SetUp() override { prng::reset_seed(); }

    void RunTest()
    {
        const auto& config = GetParam();
        std::cout << "Testing configuration: " << config << std::endl;

        auto& handle = get_handle();

        // Create tensors (NCHW layout, non-grouped)
        std::vector<size_t> input_dims  = {config.N, config.C, config.H, config.W};
        std::vector<size_t> weight_dims = {config.K, config.C, config.y, config.x};

        tensor<T> input{miopenTensorNCHW, input_dims};
        tensor<T> weights{miopenTensorNCHW, weight_dims};

        // Create convolution descriptor with deterministic mode enabled
        auto conv_desc = miopen::ConvolutionDescriptor{
            2, // num spatial dims
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(config.pad_h), static_cast<int>(config.pad_w)},
            {static_cast<int>(config.stride_h), static_cast<int>(config.stride_w)},
            {static_cast<int>(config.dilation_h), static_cast<int>(config.dilation_w)},
            {0, 0}, // adj
            1,      // group_count
            1.0};   // lowp_quant

        // Enable deterministic mode
        conv_desc.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);
        ASSERT_TRUE(conv_desc.attribute.deterministic.Get() == 1);

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});
        tensor<T> output{miopenTensorNCHW, output_desc.GetLengths()};

        // Initialize tensors
        if constexpr(CONV_DIR == Direction::Forward)
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            weights.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            std::fill(output.begin(), output.end(), T(0));
        }
        else if constexpr(CONV_DIR == Direction::BackwardData)
        {
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            weights.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
            });
            std::fill(input.begin(), input.end(), T(0));
        }
        else // BackwardWeights
        {
            input.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.1), static_cast<T>(0.1));
            });
            output.generate([](auto...) {
                return prng::gen_A_to_B(static_cast<T>(-0.01), static_cast<T>(0.1));
            });
            std::fill(weights.begin(), weights.end(), T{0});
        }

        auto in_dev  = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Write(output.data);

        // Create solver and problem
        SolverType solv{};
        auto ctx = miopen::ExecutionContext{&handle};

        miopen::conv::ProblemDescription problem = [&]() {
            if constexpr(CONV_DIR == Direction::Forward)
                return miopen::conv::ProblemDescription{
                    input.desc, weights.desc, output.desc, conv_desc, CONV_DIR};
            else
                return miopen::conv::ProblemDescription{
                    output.desc, weights.desc, input.desc, conv_desc, CONV_DIR};
        }();

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

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);

        // Store results from each iteration
        std::vector<std::vector<T>> results;
        results.reserve(NUM_ITERATIONS);

        for(int i = 0; i < NUM_ITERATIONS; ++i)
        {
            // Reset result buffer before each run
            if constexpr(CONV_DIR == Direction::Forward)
            {
                std::fill(output.begin(), output.end(), T(0));
                out_dev = handle.Write(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                std::fill(input.begin(), input.end(), T(0));
                in_dev = handle.Write(input.data);
            }
            else
            {
                std::fill(weights.begin(), weights.end(), T(0));
                wei_dev = handle.Write(weights.data);
            }

            // Execute kernel
            if constexpr(CONV_DIR == Direction::Forward)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvFwdTensors{input.desc,
                                                                          in_dev.get(),
                                                                          weights.desc,
                                                                          wei_dev.get(),
                                                                          output.desc,
                                                                          out_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                auto invoke_params =
                    miopen::conv::DataInvokeParams{miopen::ConvDataTensors{output.desc,
                                                                           out_dev.get(),
                                                                           weights.desc,
                                                                           wei_dev.get(),
                                                                           input.desc,
                                                                           in_dev.get()},
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false};
                (invoker)(handle, invoke_params);
            }
            else // BackwardWeights
            {
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
            }

            handle.Finish();

            // Read result
            if constexpr(CONV_DIR == Direction::Forward)
            {
                handle.ReadToVec(out_dev, output.data);
                results.push_back(output.data);
            }
            else if constexpr(CONV_DIR == Direction::BackwardData)
            {
                handle.ReadToVec(in_dev, input.data);
                results.push_back(input.data);
            }
            else
            {
                handle.ReadToVec(wei_dev, weights.data);
                results.push_back(weights.data);
            }
        }

        // Verify bit-exact determinism: compare all iterations against first run
        const auto& reference = results[0];
        for(int i = 1; i < NUM_ITERATIONS; ++i)
        {
            const auto& current = results[i];
            ASSERT_EQ(reference.size(), current.size());

            bool match            = true;
            size_t first_mismatch = 0;
            for(size_t j = 0; j < reference.size(); ++j)
            {
                if(std::memcmp(&reference[j], &current[j], sizeof(T)) != 0)
                {
                    match          = false;
                    first_mismatch = j;
                    break;
                }
            }

            ASSERT_TRUE(match) << "Bit-exact mismatch at iteration " << i << ", element "
                               << first_mismatch << ": reference = " << reference[first_mismatch]
                               << ", current = " << current[first_mismatch];
        }

        std::cout << "✓ All " << NUM_ITERATIONS << " iterations produced bit-exact results"
                  << std::endl;
    }
};

// BwdV4R1 needs special setup: it's disabled by default via WORKAROUND_SWDEV_229277_227616_229195
template <typename T, typename SolverType>
class GPU_ConvDeterministicBwdV4R1
    : public GPU_ConvDeterministicImplicitGemm<T, Direction::BackwardData, SolverType>
{
protected:
    void SetUp() override
    {
        prng::reset_seed();
        // Enable the solver which is disabled by default due to a workaround
        setenv("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1", "1", 1);
    }
    void TearDown() override { unsetenv("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1"); }
};

} // namespace

// ============================================================================
// Forward solvers
// ============================================================================

using GPU_Deterministic_FwdV4R5Xdlops_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmForwardV4R5Xdlops>;
using GPU_Deterministic_FwdV4R5Xdlops_BFP16 =
    GPU_ConvDeterministicImplicitGemm<bfloat16,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmForwardV4R5Xdlops>;
using GPU_Deterministic_FwdV4R4Xdlops_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops>;
using GPU_Deterministic_FwdV4R1_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::Forward,
                                      miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd>;

// ============================================================================
// Backward Data solvers
// ============================================================================

using GPU_Deterministic_BwdV1R1_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::BackwardData,
                                      miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1>;
using GPU_Deterministic_BwdV4R1_FP32 =
    GPU_ConvDeterministicBwdV4R1<float, miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1>;

// ============================================================================
// Weight gradient (WrW) solvers
// ============================================================================

using GPU_Deterministic_WrwV4R4_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::BackwardWeights,
                                      miopen::solver::conv::ConvHipImplicitGemmV4R4WrW>;
using GPU_Deterministic_WrwV4R4Xdlops_FP32 =
    GPU_ConvDeterministicImplicitGemm<float,
                                      Direction::BackwardWeights,
                                      miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops>;

// ============================================================================
// Test definitions
// ============================================================================

TEST_P(GPU_Deterministic_FwdV4R5Xdlops_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_FwdV4R5Xdlops_BFP16, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_FwdV4R4Xdlops_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_FwdV4R1_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_BwdV1R1_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_BwdV4R1_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_WrwV4R4_FP32, DeterministicTest) { this->RunTest(); };
TEST_P(GPU_Deterministic_WrwV4R4Xdlops_FP32, DeterministicTest) { this->RunTest(); };

// ============================================================================
// Test instantiations — each solver gets its own applicable configs
// ============================================================================

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_FwdV4R5Xdlops_FP32,
                         testing::ValuesIn(GetConfigFwdV4R5Xdlops()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_FwdV4R5Xdlops_BFP16,
                         testing::ValuesIn(GetConfigFwdV4R5Xdlops()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_FwdV4R4Xdlops_FP32,
                         testing::ValuesIn(GetConfigFwdV4R4Xdlops()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_FwdV4R1_FP32,
                         testing::ValuesIn(GetConfigFwdV4R1()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_BwdV1R1_FP32,
                         testing::ValuesIn(GetConfigBwdV1R1()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_BwdV4R1_FP32,
                         testing::ValuesIn(GetConfigBwdV4R1()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_WrwV4R4_FP32,
                         testing::ValuesIn(GetConfigWrwV4R4()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Deterministic_WrwV4R4Xdlops_FP32,
                         testing::ValuesIn(GetConfigWrwV4R4Xdlops()));
