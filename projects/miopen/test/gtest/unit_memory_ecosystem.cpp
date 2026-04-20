// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "gtest_common.hpp"

#include "miopen/memory_ecosystem.hpp"

namespace
{

inline std::ostream& operator <<(std::ostream& os, const MemoryEcosystemInfo& info)
{
    return os << "adapter_" << (info.adapter_index < 0 ? "NA" : std::to_string(info.adapter_index)) << "_dv_" << info.dedicated_vram
              << "_sr_" << info.shared_ram << "_dr_" << info.dedicated_ram;
}

struct MemEcoGenericTestCase
{
    MemoryEcosystemInfo info;
    std::vector<size_t> vram_blocks;
    std::vector<size_t> cpu_blocks;
    bool                able;

    struct NameGenerator
    {
        std::string operator()(const auto& _param)
        {
            const MemEcoGenericTestCase& param = _param.param;
            const auto& _info = param.info;
            std::stringstream ss;

            ss << _info;
            ss << "_gpu";
            for(auto block : param.vram_blocks)
            {
                ss << "_" << block;
            }

            ss << "_cpu";
            for(auto block : param.cpu_blocks)
            {
                ss << "_" << block;
            }

            ss << "_able_" << param.able;

            return ss.str();
        }
    };
};

struct MemEcoSystemTestCase
{
    MemoryEcosystemInfo info;
    std::vector<size_t> vram_dedicated_percents;
    std::vector<size_t> vram_shared_percents;
    size_t              cpu_percent;   // % of shared mem

    struct NameGenerator
    {
        std::string operator()(const auto& _param)
        {
            const MemEcoSystemTestCase& param = _param.param;
            const auto& _info = param.info;
            std::stringstream ss;

            ss << _info;
            ss << "_dvrampcts";
            for(auto pct : param.vram_dedicated_percents)
            {
                ss << "_" << pct;
            }

            ss << "_svrampcts";
            for(auto pct : param.vram_shared_percents)
            {
                ss << "_" << pct;
            }

            ss << "_cpupct_" << param.cpu_percent;

            return ss.str();
        }
    };
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

struct GPU_MemoryEcosystemGeneric_None
    : public ::testing::TestWithParam<MemEcoGenericTestCase>
{
MemoryEcosystemInfo tmp_info;

    static auto IsAbleToAllocate(const MemEcoGenericTestCase& testcase)
    {
        return MemoryEcosystem::AbleToAllocate(testcase.info, testcase.vram_blocks, testcase.cpu_blocks);
    }

    static auto NotAbleToAllocate(const MemEcoGenericTestCase& testcase)
    {
        return !MemoryEcosystem::AbleToAllocate(testcase.info, testcase.vram_blocks, testcase.cpu_blocks);
    }
};

inline std::vector<MemEcoGenericTestCase> GenericCases()
{
    return {
        {{0, 8, 8, 16}, {3, 3}, {0}, true},
        {{0, 8, 8, 16}, {3, 3}, {2}, true},
        {{0, 8, 8, 16}, {9}, {0}, False()},
        {{0, 12, 8, 12}, {9}, {0}, true},
        {{0, 8, 8, 16}, {5, 5}, {3}, true},
        {{0, 8, 8, 16}, {5, 3, 2}, {4}, true},
        {{0, 8, 8, 16}, {5, 5}, {4}, False()},
        {{0, 8, 8, 16}, {4, 1, 4, 1, 4}, {0}, true},
        {{0, 8, 8, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, 12, 4, 16}, {4, 1, 4, 1, 4}, {3}, False()},
        {{0, 12, 8, 12}, {4, 1, 4, 1, 4}, {3}, true},
        {{0, 8, 8, 16}, {4, 5, 2, 4}, {0}, False()},
        {{0, 8, 8, 16}, {5, 4, 2, 4}, {0}, true},
        {{0, 8, 8, 16}, {5, 4, 2, 4}, {1}, False()},
    };
};

TEST_P(GPU_MemoryEcosystemGeneric_None, GenericAbleToAllocate)
{
    auto test_case = this->GetParam();

    if(test_case.able)
    {
        EXPECT_PRED1(IsAbleToAllocate, test_case);
    }
    else
    {
        EXPECT_PRED1(NotAbleToAllocate, test_case);
    }
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MemoryEcosystemGeneric_None,
                         testing::ValuesIn(GenericCases()),
                         MemEcoGenericTestCase::NameGenerator{});

struct GPU_MemoryEcosystemSystem_None
    : public ::testing::TestWithParam<MemEcoSystemTestCase>
{
MemoryEcosystemInfo tmp_info;

    static auto IsAbleToAllocate(const MemEcoGenericTestCase& testcase)
    {
        return MemoryEcosystem::AbleToAllocate(testcase.vram_blocks, testcase.cpu_blocks);
    }

    static auto NotAbleToAllocate(const MemEcoGenericTestCase& testcase)
    {
        return !MemoryEcosystem::AbleToAllocate(testcase.vram_blocks, testcase.cpu_blocks);
    }
};

inline std::vector<MemEcoSystemTestCase> SystemCases()
{
    const auto& info = MemoryEcosystem::GetMemoryEcosystemInfo(0);

    if(info.dedicated_vram == size_t{0})
    {
        return {
            {info, {0}, {0}, 0}
        };
    }

    std::vector<MemEcoSystemTestCase> system_cases = {
        {info, {50, 50}, {50, 50}, 0},    // should always be able
        {info, {50, 50}, {50, 50}, 10},   // should always be unable
        {info, {101}, {0}, 0},            // always able if SVRAM > DVRAM
        {info, {101}, {0}, 10},           // may be able if SVRAM > DVRAM
        {info, {0}, {101}, 0},            // always able if DVRAM > SVRAM
        {info, {0}, {101}, 10},           // may be able if DVRAM > SVRAM
    };

    // for(auto case : cases)

    return system_cases;
}

TEST_P(GPU_MemoryEcosystemSystem_None, SystemAbleToAllocate)
{
    const auto& test_case = this->GetParam();
    const auto& info = test_case.info;

    if(test_case.info.dedicated_vram == 0)
        GTEST_SKIP() << "Detailed VRAM config not available.";

    // We cannot know the system memory config at compile-time. Thus,
    // we cannot predetermine ability to allocate. Instead, we define
    // the expected behavior here:
    //
    // Blocks are assigned in this order:
    // 1.  cpu block->shared
    // 2a. gpu block->dedicated if available,
    // 2b. gpu block->shared
    //
    // All blocks are unmoveable for the duration of the test. 
    // If any block cannot fit, MemoryEcosystem shall return 'unable'.

    auto PerCent = [](size_t pct, size_t bytes) -> size_t {
        return (pct * bytes) / 100 - 1; // prevent roundoff errors
    };

    MemEcoGenericTestCase gen_case;
    bool                  able = true;

    gen_case.cpu_blocks.push_back(PerCent(test_case.cpu_percent, info.shared_ram) + 1);
    ASSERT_GE(info.shared_ram, gen_case.cpu_blocks[0]) << "CPU block cannot exceed shared mem";
    able = info.shared_ram >= gen_case.cpu_blocks[0];

    if(able)
    {
        for(auto pct : test_case.vram_dedicated_percents)
            gen_case.vram_blocks.push_back(PerCent(pct, info.dedicated_vram));
        for(auto pct : test_case.vram_shared_percents)
            gen_case.vram_blocks.push_back(PerCent(pct, info.shared_ram));

        size_t used_ded    = 0;
        size_t used_shared = gen_case.cpu_blocks[0];
        for(auto block : gen_case.vram_blocks)
        {
            if(info.dedicated_vram >= used_ded + block)
            {
                used_ded += block;
            }
            else if(info.shared_ram >= used_shared + block)
            {
                used_shared += block;
            }
            else
            {
                able = false;
                break;
            }
        }
    }

    if(able)
    {
        EXPECT_PRED1(IsAbleToAllocate, gen_case);
    }
    else
    {
        EXPECT_PRED1(NotAbleToAllocate, gen_case);
    }
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MemoryEcosystemSystem_None,
                         testing::ValuesIn(SystemCases()),
                         MemEcoSystemTestCase::NameGenerator{});
