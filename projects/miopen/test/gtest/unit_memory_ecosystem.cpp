// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>

#include "gtest_common.hpp"

#include "miopen/memory_ecosystem.hpp"

namespace
{
struct MemoryEcosystemTestCase
{
    MemoryEcosystemInfo info;
    std::vector<size_t> vram_blocks;
    std::vector<size_t> cpu_blocks;
    bool able;
};

bool False()
{
#ifndef _WIN32
    return false;
#else
// Does nothing in linux; expect always true
    return true;
#endif
}
}

struct TestNameGenerator
{
    std::string operator()(const auto& info)
    {
        const auto& tc = info.param;
        std::stringstream ss;

        ss << "input_" << GetRangeAsString(tc.input, "x") << "_lr_" << tc.lr << "_beta1_"
           << tc.beta1 << "_beta2_" << tc.beta2 << "_weight_decay_" << tc.weight_decay << "_eps_"
           << tc.eps << "_lr_" << tc.lr << "_lr_" << tc.lr << std::boolalpha << "_amsgrad_"
           << tc.amsgrad << "_maximize_" << tc.maximize << "_adamw_" << tc.adamw
           << "_use_step_tensor_" << tc.use_step_tensor << "_test_id_" << info.index;

        std::string str(ss.str());

        // Name format only supports letters, numbers and underscores.
        std::transform(str.begin(), str.end(), str.begin(), [](char c) -> char {
            return (c == '.') ? 'p' : (std::isalnum(c) ? c : '_');
        });

        return str;
    }
};

struct GPU_MemoryEcosystem_None
    : public ::testing::TestWithParam<MemoryEcosystemTestCase>
{
MemoryEcosystemInfo tmp_info;

    static auto IsAbleToAllocate(const MemoryEcosystemTestCase& testcase)
    {
        return MemoryEcosystem::AbleToAllocate(testcase.info, testcase.vram_blocks, testcase.cpu_blocks);
    }

    static auto NotAbleToAllocate(const MemoryEcosystemTestCase& testcase)
    {
        return !MemoryEcosystem::AbleToAllocate(testcase.info, testcase.vram_blocks, testcase.cpu_blocks);
    }
};

inline std::vector<MemoryEcosystemTestCase> AllocateCases()
{
    return {
        {{0, 8, 8, 16}, {3, 3}, {0}, true},
        {{0, 8, 8, 16}, {3, 3}, {2}, true},
        {{0, 8, 8, 16}, {9}, {0}, False()},
        {{0, 12, 8, 12}, {9}, {0}, true},
        {{0, 8, 8, 16}, {5, 5}, {3}, true},
        {{0, 8, 8, 16}, {5, 3, 2}, {3, 1}, true},
        {{0, 8, 8, 16}, {5, 5}, {3, 1}, False()},
        {{0, 8, 8, 16}, {4, 1, 4, 1, 4}, {0}, true},
        {{0, 8, 8, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, 12, 4, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, 12, 8, 12}, {4, 1, 4, 1, 4}, {3}, true},
        {{0, 8, 8, 16}, {4, 5, 2, 4}, {0}, False()},
        {{0, 8, 8, 16}, {5, 4, 2, 4}, {0}, true},
        {{0, 8, 8, 16}, {5, 4, 2, 4}, {1}, False()},
    };
};

TEST_P(GPU_MemoryEcosystem_None, AbleToAllocate)
{
    auto info = this->GetParam();
    tmp_info = info.info;

    if(info.able)
    {
        EXPECT_PRED1(IsAbleToAllocate, info);
    }
    else
    {
        EXPECT_PRED1(NotAbleToAllocate, info);
    }
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MemoryEcosystem_None,
                         testing::ValuesIn(AllocateCases()));
