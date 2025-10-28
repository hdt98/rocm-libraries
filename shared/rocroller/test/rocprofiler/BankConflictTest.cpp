/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Scheduling/LDSBankModel.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include "../catch/TestContext.hpp"
#include "Agent.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

namespace rocRollerTest
{
    std::vector<size_t>
        generateLDSAddresses(size_t workgroupSize, size_t strideMultiplier, size_t instrDwords)
    {
        std::vector<size_t> addresses;
        for(size_t workitemId = 0; workitemId < workgroupSize; ++workitemId)
        {
            size_t address = workitemId * (4 * strideMultiplier * instrDwords);
            addresses.push_back(address);
        }
        return addresses;
    }

    TEST_CASE("Rocprofiler LDS Microkernel", "[rocprofiler]")
    {
        // TODO: remove
        int ITERS = 16;
        if(const char* env_p = std::getenv("ITERS"))
            ITERS = atoi(env_p);

        auto workgroupSize = 64u;
        if(const char* env_p = std::getenv("WORKGROUP_SIZE"))
            workgroupSize = atoi(env_p);

        bool barrier = true;
        if(const char* env_p = std::getenv("BARRIER"))
            barrier = atoi(env_p) == 1 ? true : false;

        const std::vector<int>  instrSizes      = {1, 2, 4}; // b32, b64, b128
        const std::vector<int>  strides         = {1, 2, 4, 8, 16, 32, 64, 128}; // between threads
        const std::vector<bool> writeOperations = {false, true};

        for(auto instrDwords : instrSizes)
        {
            for(auto strideMultiplier : strides)
            {
                for(auto write : writeOperations)
                {
                    const auto name = "lds_microkernel_" + std::to_string(instrDwords * 32)
                                      + "b_stride" + std::to_string(strideMultiplier) + "_"
                                      + (write ? "write" : "read");

                    DYNAMIC_SECTION(name)
                    {
                        rocRoller::profiler::reset();

                        auto context = TestContext::ForTestDevice({}, name);

                        if(not context->targetArchitecture().target().isCDNA35GPU())
                        {
                            SKIP("LDS Bank Model only implemented for CDNA 3.5 GPUs");
                        }

                        auto command = std::make_shared<Command>();
                        auto k       = context->kernel();

                        k->setKernelDimensions(1);

                        const auto one  = std::make_shared<Expression::Expression>(1u);
                        const auto zero = std::make_shared<Expression::Expression>(0u);

                        auto workitemCount = Expression::literal(workgroupSize * 256 * 32);
                        k->setWorkgroupSize({workgroupSize, 1, 1});
                        k->setWorkitemCount({workitemCount, one, one});
                        k->setDynamicSharedMemBytes(zero);

                        context->schedule(k->preamble());
                        context->schedule(k->prolog());

                        auto kb = [&]() -> Generator<Instruction> {
                            const auto alignment = instrDwords;
                            const auto regCount  = 256 - 8; // leave a few

                            auto dst = Register::Value::Placeholder(
                                context.get(),
                                Register::Type::Vector,
                                DataType::Raw32,
                                regCount,
                                Register::AllocationOptions{
                                    .contiguousChunkWidth = Register::FULLY_CONTIGUOUS,
                                    .alignment            = alignment,
                                });
                            co_yield dst->allocate();

                            auto lds = Register::Value::AllocateLDS(
                                context.get(),
                                DataType::Raw32,
                                context->targetArchitecture().GetCapability(
                                    GPUCapability::MaxLdsSize)
                                    / 4);
                            auto ldsWithOffset = Register::Value::Placeholder(
                                context.get(), Register::Type::Vector, DataType::UInt32, 1);
                            auto workitemIndex = context->kernel()->workitemIndex()[0];
                            co_yield Expression::generate(
                                ldsWithOffset,
                                Expression::literal(lds->getLDSAllocation()->offset())
                                    + workitemIndex->expression()
                                          * Expression::literal((4 * strideMultiplier * alignment)
                                                                % lds->getLDSAllocation()->size()),
                                context.get());

                            auto getSubset
                                = [](size_t n, size_t m, size_t i) -> std::pair<size_t, size_t> {
                                // If run out of registers, wrap around
                                size_t num_complete_chunks = n / m;
                                if(num_complete_chunks == 0)
                                {
                                    return {0, 0};
                                }
                                size_t chunk_index = i % num_complete_chunks;
                                size_t start       = chunk_index * m;
                                return {start, start + m};
                            };

                            if(barrier)
                            {
                                co_yield context->mem()->barrier({});
                            }

                            for(int i = 0; i < ITERS; ++i)
                            {
                                const auto [start, end] = getSubset(regCount, instrDwords, i);
                                const auto numBytes     = instrDwords * 4;

                                if(write)
                                {
                                    co_yield context->mem()->storeLocal(
                                        ldsWithOffset,
                                        dst->subset(Generated(iota(start, end))),
                                        0,
                                        numBytes);
                                }
                                else
                                {
                                    co_yield context->mem()->loadLocal(
                                        dst->subset(Generated(iota(start, end))),
                                        ldsWithOffset,
                                        0,
                                        numBytes);
                                }
                            }

                            // TODO: simplify
                            const auto waitcnt_count  = 1;
                            const auto last_decrement = 0;

                            for(int i = waitcnt_count; i >= 0; --i)
                            {
                                if(i == last_decrement + 1)
                                {
                                    i -= last_decrement;
                                }
                                co_yield Instruction::Wait(
                                    WaitCount::DSCnt(context->targetArchitecture(), i));
                            }
                        };

                        context->schedule(kb());

                        context->schedule(k->postamble());
                        context->schedule(k->amdgpu_metadata());

                        CommandKernel commandKernel;
                        commandKernel.setContext(context.get());
                        commandKernel.generateKernel();

                        CommandArguments commandArgs = command->createArguments();

                        const auto latencies = rocRoller::profiler::loopUntilDispatchData(
                            [&]() { commandKernel.launchKernel(commandArgs.runtimeArguments()); });

                        INFO(toString(latencies));
                        Log::debug(toString(latencies));

                        REQUIRE(latencies.size() == 21);

                        GPUArchitectureGFX gfx = context->targetArchitecture().target().gfx;

                        auto baseAddresses
                            = generateLDSAddresses(workgroupSize, strideMultiplier, instrDwords);

                        LDSBankModel::RuntimeLDSInstruction ldsinstr;
                        ldsinstr.memoryOp.direction = write ? LDSBankModel::LdsDirection::Write
                                                            : LDSBankModel::LdsDirection::Read;
                        ldsinstr.dwords             = instrDwords;
                        ldsinstr.baseAddresses      = baseAddresses;

                        uint predictedCycles = LDSBankModel::getInstructionCycles(ldsinstr, gfx);

                        uint issueCycles = LDSBankModel::getInstructionIssueCycles(
                            ldsinstr.memoryOp, ldsinstr.dwords);
                        uint dataCycles = LDSBankModel::getInstructionDataCycles(ldsinstr, gfx);

                        std::stringstream info;

                        info << fmt::format("dwords {}, stride {}, {}\n",
                                            instrDwords,
                                            strideMultiplier,
                                            write ? "write" : "read");

                        uint64_t actualMaxLdsInstrCycles  = 0;
                        uint64_t actualLastSWaitcntCycles = 0;
                        for(const auto& data : latencies)
                        {
                            if((write && data.instruction.find("ds_write") != std::string::npos)
                               || (!write && data.instruction.find("ds_read") != std::string::npos))
                            {
                                actualMaxLdsInstrCycles
                                    = std::max(actualMaxLdsInstrCycles, data.meanLatency());
                            }
                            else if(data.instruction.find("s_waitcnt") != std::string::npos)
                            {
                                actualLastSWaitcntCycles = data.meanLatency();
                            }
                        }
                        info << fmt::format("  Actual s_waitcnt Cycles: {}\n",
                                            actualLastSWaitcntCycles);
                        info << fmt::format("  Actual LDS Instruction Cycles: {}\n",
                                            actualMaxLdsInstrCycles);
                        info << fmt::format("  Model Predicted Cycles: {}\n", predictedCycles);
                        info << fmt::format("    Issue Cycles: {}\n", issueCycles);
                        info << fmt::format("    Data Cycles: {}\n", dataCycles);

                        INFO(info.str());
                        Log::debug(info.str());

                        CHECK_THAT(actualLastSWaitcntCycles,
                                   Catch::Matchers::WithinAbs(predictedCycles, 1ul));
                        if(write && instrDwords == 4)
                            // ds_write_b128 requires queue info
                            CHECK_THAT(actualMaxLdsInstrCycles,
                                       Catch::Matchers::WithinAbs(predictedCycles, 12ul));
                        else
                            CHECK_THAT(actualMaxLdsInstrCycles,
                                       Catch::Matchers::WithinAbs(predictedCycles, 4ul));
                    }
                }
            }
        }
    }
}
