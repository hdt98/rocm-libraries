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
#include <rocRoller/TensorDescriptor.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>
#include <common/Scheduling.hpp>
#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include "CustomAssertions.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"

namespace MemAddressingTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;

    TEST_CASE("LDS Address Modelling", "[mem-addressing]")
    {
        SECTION("No padding")
        {
            auto context = TestContext::ForTestDevice();
            auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

            example.setUseLDS(true, true, false);
            example.setTranspose("T", "N");

            auto command = example.getCommand();
            auto params  = example.getCommandParameters();

            CommandKernel commandKernel(command, context.KernelName());
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);

            commandKernel.generateKernelGraph(""); // TODO: unused parameter
            auto graph = commandKernel.getKernelGraph();

            for(auto inst : kernelInstructions(context.get(), command, graph))
            {
                context.get()->schedule(inst);
                if(inst.getModelledAddresses().has_value())
                {
                    auto addresses = inst.getModelledAddresses().value();
                    Log::info("addresses {}", addresses);

                    // Check in rocgdb
                    CHECK(addresses
                          == std::vector<uint64_t>{
                              0,    256,  512,  768,  1024, 1280, 1536, 1792, 2048, 2304, 2560,
                              2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376,
                              5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 4,
                              260,  516,  772,  1028, 1284, 1540, 1796, 2052, 2308, 2564, 2820,
                              3076, 3332, 3588, 3844, 4100, 4356, 4612, 4868, 5124, 5380, 5636,
                              5892, 6148, 6404, 6660, 6916, 7172, 7428, 7684, 7940});
                }
            }

            auto [commandArgs, deviceA, deviceB, deviceC, deviceD]
                = example.getCommandArguments<float>();
            commandKernel.launchKernel(commandArgs.runtimeArguments());
        }
    }
}
