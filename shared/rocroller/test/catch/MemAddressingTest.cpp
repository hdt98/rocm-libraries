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
        struct Variants
        {
            std::string                              name;
            std::function<void(std::vector<size_t>)> validate;
            decltype(GEMMProblem::padA)              padA = GEMMProblem{}.padA;
            decltype(GEMMProblem::padB)              padB = GEMMProblem{}.padB;

            static void NoPadding(std::vector<size_t> addresses)
            {
                CHECK(addresses
                      == std::vector<size_t>{
                          0,    256,  512,  768,  1024, 1280, 1536, 1792, 2048, 2304, 2560,
                          2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376,
                          5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 4,
                          260,  516,  772,  1028, 1284, 1540, 1796, 2052, 2308, 2564, 2820,
                          3076, 3332, 3588, 3844, 4100, 4356, 4612, 4868, 5124, 5380, 5636,
                          5892, 6148, 6404, 6660, 6916, 7172, 7428, 7684, 7940});
            }

            static void YesPadding(std::vector<size_t> addresses)
            {
                INFO(addresses);
                std::vector<size_t> A_pattern
                    = {0,    256,  512,  768,  1024, 1280, 1536, 1792, 2052, 2308, 2564, 2820, 3076,
                       3332, 3588, 3844, 4104, 4360, 4616, 4872, 5128, 5384, 5640, 5896, 6156, 6412,
                       6668, 6924, 7180, 7436, 7692, 7948, 4,    260,  516,  772,  1028, 1284, 1540,
                       1796, 2056, 2312, 2568, 2824, 3080, 3336, 3592, 3848, 4108, 4364, 4620, 4876,
                       5132, 5388, 5644, 5900, 6160, 6416, 6672, 6928, 7184, 7440, 7696, 7952};
                std::vector<size_t> B_pattern
                    = {0,    256,  512,  768,  1028, 1284, 1540, 1796, 2056, 2312, 2568, 2824, 3084,
                       3340, 3596, 3852, 4112, 4368, 4624, 4880, 5140, 5396, 5652, 5908, 6168, 6424,
                       6680, 6936, 7196, 7452, 7708, 7964, 4,    260,  516,  772,  1032, 1288, 1544,
                       1800, 2060, 2316, 2572, 2828, 3088, 3344, 3600, 3856, 4116, 4372, 4628, 4884,
                       5144, 5400, 5656, 5912, 6172, 6428, 6684, 6940, 7200, 7456, 7712, 7968};
                CHECK((addresses == A_pattern || addresses == B_pattern));
            }
        };

        const auto variants
            = GENERATE(Variants{"No padding", Variants::NoPadding},
                       Variants("Yes padding", Variants::YesPadding, {32 * 64, 4}, {16 * 64, 4}));

        SECTION(variants.name)
        {
            auto context = TestContext::ForTestDevice();
            auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

            example.setUseLDS(true, true, false);
            example.setTranspose("T", "N");
            example.setPad(variants.padA, variants.padB);

            auto command = example.getCommand();
            auto params  = example.getCommandParameters();

            CommandKernel commandKernel(command, context.KernelName());
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);

            commandKernel.generateKernelGraph();
            auto graph = commandKernel.getKernelGraph();

            for(auto inst : kernelInstructions(context.get(), command, graph))
            {
                context.get()->schedule(inst);
                if(inst.getModelledAddresses().has_value())
                {
                    auto addresses = inst.getModelledAddresses().value();
                    variants.validate(addresses);
                }
            }

            auto [commandArgs, deviceA, deviceB, deviceC, deviceD]
                = example.getCommandArguments<float>();
            commandKernel.launchKernel(commandArgs.runtimeArguments());
        }
    }
}
