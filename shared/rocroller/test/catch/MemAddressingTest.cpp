/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CommandSolution_detail.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>
#include <common/Scheduling.hpp>

#include "CustomAssertions.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"

namespace MemAddressingTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;

    TEST_CASE("LoadLDSTile basic test", "[mem-addressing]")
    {
        SECTION("Float")
        {
            auto context = TestContext::ForTestDevice();
            auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

            example.setTileSize(128, 256, 8);
            example.setMFMA(32, 32, 2, 1);
            example.setUseLDS(true, false, false);

            auto command = example.getCommand();
            auto params
                = example
                      .getCommandParameters(); // TODO: ensure this not required to be called after getCommand()

            CommandKernel commandKernel(command, context.KernelName());
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);

            commandKernel.generateKernelGraph("");
            auto graph = commandKernel.getKernelGraph();

            for(auto inst : kernelInstructions(context.get(), command, graph))
            {
                context.get()->schedule(inst);
                if(inst.getModelledAddresses().has_value())
                {
                    auto addresses = inst.getModelledAddresses().value();
                    Log::info("addresses {}", addresses);

                    REQUIRE(addresses.size() == 64); // TODO: should this be 64 or 256?

                    for(auto addr : addresses)
                    {
                        CHECK(addr % 4 == 0);
                    }

                    for(size_t i = 1; i < addresses.size(); ++i)
                    {
                        int diff = addresses[i] - addresses[i - 1];
                        CHECK(diff % 4 == 0);
                        CHECK(diff > 0);
                    }
                }
            }
        }

        SECTION("FP4")
        {
            auto context = TestContext::ForTestDevice();
            auto example = rocRollerTest::Graphs::GEMM(rocRoller::DataType::FP4,
                                                       rocRoller::DataType::FP4,
                                                       rocRoller::DataType::Float,
                                                       rocRoller::DataType::Float);

            example.setTileSize(256, 256, 128);
            example.setMFMA(32, 32, 64, 1);
            example.setUseLDS(true, true, false);

            auto command = example.getCommand();
            // TODO: ensure this not required to be called after getCommand()
            auto params = example.getCommandParameters();

            CommandKernel commandKernel(command, context.KernelName());
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);

            commandKernel.generateKernelGraph("");
            auto graph = commandKernel.getKernelGraph();

            for(auto inst : kernelInstructions(context.get(), command, graph))
            {
                context.get()->schedule(inst);
                if(inst.getModelledAddresses().has_value())
                {
                    auto addresses = inst.getModelledAddresses().value();
                    Log::info("addresses {}", addresses);

                    REQUIRE(addresses.size() == 64); // TODO: should this be 64 or 256?

                    for(auto addr : addresses)
                    {
                        CHECK(addr % 4 == 0);
                    }

                    for(size_t i = 1; i < addresses.size(); ++i)
                    {
                        int diff = addresses[i] - addresses[i - 1];
                        CHECK(diff % 4 == 0);
                    }
                }
            }
        }
    }
}
