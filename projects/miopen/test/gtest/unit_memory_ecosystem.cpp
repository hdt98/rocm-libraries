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
    bool could;
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
        {{0, "r0_d33", 8, 8, 16}, {3, 3}, {0}, true},
        {{0, "r2_d33", 8, 8, 16}, {3, 3}, {2}, true},
        {{0, "r0_n9", 8, 8, 16}, {9}, {0}, False())},
        {{0, "r0_d9", 12, 8, 12}, {9}, {0}, true},
        {{0, "r3_d5_s5", 8, 8, 16}, {5, 5}, {3}, true},
        {{0, "r4_d53_s2", 8, 8, 16}, {5, 3, 2}, {3, 1}, true},
        {{0, "r4_d5_n5", 8, 8, 16}, {5, 5}, {3, 1}, False()},
        {{0, "r0_d41_s4_d1_s4", 8, 8, 16}, {4, 1, 4, 1, 4}, {0}, true},
        {{0, "r3_d41_s4_d1_n4__d44_s41_n1", 8, 8, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, "r3_d4141_n4", 12, 4, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, "r3_d4141_s4", 12, 8, 12}, {4, 1, 4, 1, 4}, {3}, true},
        {{0, "r0_d4_s5_d2_n4__d5_s4_s4_d2", 8, 8, 16}, {4, 5, 2, 4}, {0}, False()},
        {{0, "r1_d4_s5_d2_n4__d5_s4_n4", 8, 8, 16}, {4, 5, 2, 4}, {1}, False()},
    };
};

namespace
{
}

TEST_P(GPU_MemoryEcosystem_None, AbleToAllocate)
{
    auto info = this->GetParam();
    tmp_info = info.info;

    if(info.able)
    {
        EXPECT_PRED3(IsAbleToAllocate, info);
    }
    else
    {
        EXPECT_PRED3(NotAbleToAllocate, info);
    }
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MemoryEcosystem_None,
                         testing::ValuesIn(AllocateCases()));
