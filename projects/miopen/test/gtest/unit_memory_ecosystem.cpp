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
    std::string operator()(const auto& _param)
    {
        const auto& param = _param.param;
        const auto& info = param.info;
        std::stringstream ss;

        ss << "adapter_" << info.adapter_index << "_dv_" << info.dedicated_vram
           << "_sr_" << info.shared_ram << "_dr_" << info.dedicated_ram
           << "_gpu";

        for(auto block : param.vram_blocks)
        {
            ss << "_" << block; 
        }

        ss << "_cpu";

        for(auto block : param.cpu_blocks)
        {
            ss << "_" << block; 
        }

        ss << "_expect_" << param.able;

        return ss.str();
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
                         testing::ValuesIn(AllocateCases()),
                        TestNameGenerator{});
