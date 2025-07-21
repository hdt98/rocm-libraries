/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include <miopen/ctc.hpp>
#include "get_handle.hpp"

namespace {

struct TestCaseValues
{
    const std::string msg;
    const miopenStatus_t status;

    const miopen::TensorDescriptor probsDesc;
    const miopen::TensorDescriptor gradientsDesc;

    const int* labels{nullptr};
    const int* labelLengths{nullptr};
    const int* inputLengths{nullptr};
};

static void check_vals(miopen::CTCLossDescriptor& ctc, TestCaseValues& vals)
{

    EXPECT_THROW(
        {
            try
            {
                ctc.CTCLoss(get_handle(),
                            vals.probsDesc,
                            nullptr,
                            vals.labels,
                            vals.labelLengths,
                            vals.inputLengths,
                            nullptr,
                            vals.gradientsDesc,
                            nullptr,
                            MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC,
                            nullptr,
                            0);
            }
            catch(const miopen::Exception& e)
            {
                if(vals.msg.length() > 0)
                {
                    EXPECT_THAT(e.message, ::testing::EndsWith(vals.msg));
                    EXPECT_THAT(e.what(), ::testing::EndsWith(vals.msg));
                }

                EXPECT_EQ(e.status, vals.status);
                throw;
            }
        },
        miopen::Exception);
}

TEST(CPU_CTCLoss_NONE, test_miopen_throw1)
{
    miopen::TensorDescriptor td_float{miopenFloat, std::vector{10, 3, 2}};
    miopen::TensorDescriptor td_half{miopenHalf, std::vector{10, 3, 2}};

    miopen::TensorDescriptor td_int32{miopenInt32, std::vector{10, 3, 2}};
    miopen::TensorDescriptor td_double{miopenDouble, std::vector{10, 3, 2}};
    miopen::TensorDescriptor td_bfloat{miopenBFloat16, std::vector{10, 3, 2}};

    miopen::TensorDescriptor td_wrong_len1{miopenFloat, std::vector{2, 1, 1}};
    miopen::TensorDescriptor td_wrong_len2{miopenFloat, std::vector{1, 2, 1}};
    miopen::TensorDescriptor td_wrong_len3{miopenFloat, std::vector{1, 1, 3}};

    TestCaseValues testvals[] = {
        // clang-format off

        // probDesc must be miopenFloat or miopenHalf
        {"", miopenStatusBadParm, td_int32,  td_float},
        {"", miopenStatusBadParm, td_double, td_float},
        {"", miopenStatusBadParm, td_bfloat, td_half},


        // probsDesc and gradientsDesc differs in length
        {"probs tensor's dimension does not match gradients tensor's dimension", miopenStatusUnknownError, td_float, td_wrong_len1},
        {"probs tensor's dimension does not match gradients tensor's dimension", miopenStatusUnknownError, td_float, td_wrong_len2},
        {"probs tensor's dimension does not match gradients tensor's dimension", miopenStatusUnknownError, td_float, td_wrong_len3}

        // clang-format on
    };

    miopen::CTCLossDescriptor ctc;
    for(auto& testval : testvals)
    {
        check_vals(ctc, testval);
    }
}

TEST(CPU_CTCLoss_NONE, test_miopen_throw2)
{
    constexpr int class_sz      = 2;
    constexpr int batch_size    = 10;
    constexpr int max_time_step = 5;

    miopen::TensorDescriptor td{miopenFloat, std::vector{max_time_step, batch_size, class_sz}};
    assert(td.size() == 3);

    std::vector<int> input_length_too_big;
    std::fill_n(std::back_inserter(input_length_too_big), batch_size, max_time_step + 1);

    std::vector<int> input_length1;
    std::fill_n(std::back_inserter(input_length1), batch_size, max_time_step);

    constexpr int label_len = 2;
    std::vector<int> label_lengths;
    std::fill_n(std::back_inserter(label_lengths), batch_size, label_len);

    std::vector<int> labels_too_long;
    for(int ll : label_lengths)
    {
        std::fill_n(std::back_inserter(labels_too_long), ll, class_sz);
    }

    std::vector<int> input_length2;
    std::fill_n(std::back_inserter(input_length2), batch_size, class_sz - 1);
    std::vector<int> labels_too_many;
    for(int ll : label_lengths)
    {
        std::fill_n(std::back_inserter(labels_too_many), ll, class_sz - 1);
    }

    TestCaseValues testvals[] = {
        // clang-format off

        // inputLengths[i] > max_time_step (aka probsDesc[0])
        {"Wrong input time step", miopenStatusUnknownError, td, td, nullptr, nullptr, input_length_too_big.data()},

        // labels[...] >= class_sz (aka probsDesc[2])
        {"Wrong label id", miopenStatusUnknownError, td, td, labels_too_long.data(), label_lengths.data(), input_length1.data()},

        // labelLengths[i] + repeat[i] > inputLengths[i]
        {"Error: label length exceeds input time step", miopenStatusUnknownError, td, td, labels_too_many.data(), label_lengths.data(), input_length2.data()}

        // clang-format on
    };

    miopen::CTCLossDescriptor ctc;
    for(auto& testval : testvals)
    {
        check_vals(ctc, testval);
    }
}
} // namespace
