/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
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
        SECTION("Create LoadLDSTile operations from GEMM graph")
        {
            auto context = TestContext::ForTestDevice(
                KernelOptions({.dsObserver = DSObserverType::WeightlessDSMemObserver}));
            auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

            example.setTileSize(128, 256, 8);
            example.setMFMA(32, 32, 2, 1);
            example.setUseLDS(true, true, true);

            auto kgraph = example.getKernelGraph();
            auto params = example.getCommandParameters();

            params->unrollK           = 4;
            params->prefetch          = true;
            params->prefetchInFlight  = 2;
            params->prefetchLDSFactor = 2;
            params->prefetchMixMemOps = true;

            std::vector<GraphTransformPtr> transforms;
            transforms.push_back(std::make_shared<IdentifyParallelDimensions>());
            transforms.push_back(std::make_shared<OrderMemory>(false));
            transforms.push_back(std::make_shared<UpdateParameters>(params));
            transforms.push_back(std::make_shared<AddLDS>(params, context.get()));
            transforms.push_back(std::make_shared<LowerLinear>(context.get()));
            transforms.push_back(std::make_shared<LowerTile>(params, context.get()));
            transforms.push_back(std::make_shared<LowerTensorContraction>(params, context.get()));
            transforms.push_back(std::make_shared<Simplify>());
            transforms.push_back(std::make_shared<FuseExpressions>());
            transforms.push_back(std::make_shared<ConnectWorkgroups>(context.get()));
            transforms.push_back(
                std::make_shared<WorkgroupRemapXCC>(context.get(), params->workgroupRemapXCC));
            transforms.push_back(std::make_shared<UnrollLoops>(params, context.get()));
            transforms.push_back(std::make_shared<FuseLoops>());
            transforms.push_back(std::make_shared<RemoveDuplicates>());
            transforms.push_back(std::make_shared<OrderEpilogueBlocks>());
            transforms.push_back(std::make_shared<CleanLoops>());
            transforms.push_back(std::make_shared<AddPrefetch>(params, context.get()));
            transforms.push_back(std::make_shared<UpdateWavefrontParameters>(params));
            transforms.push_back(
                std::make_shared<AssignIndexExpressions>(context.get(), example.getCommand()));

            for(auto& t : transforms)
                kgraph = kgraph.transform(t);

            auto loadLDSTileTags = kgraph.control.getNodes<LoadLDSTile>().to<std::vector>();

            REQUIRE(loadLDSTileTags.size() > 0);

            for(auto tag : loadLDSTileTags)
            {
            }

            auto one = Expression::literal(1);

            auto kernel = context.get()->kernel();

            // IMPORTANT: Add command arguments to the kernel to avoid missing argument errors
            auto command = example.getCommand();
            kernel->addCommandArguments(command->getArguments());

            kernel->setWorkgroupSize({256, 1, 1});
            kernel->setWorkitemCount({one, one, one});

            // Generate and schedule all kernel sections
            context.get()->schedule(kernel->preamble());
            context.get()->schedule(kernel->prolog());
            context.get()->schedule(rocRoller::KernelGraph::generate(kgraph, kernel));
            context.get()->schedule(kernel->postamble());

            // Get the generated output
            std::string output = context.output();
            INFO(output);

            // Add meaningful test assertions
            // Check that LDS operations were generated (since we enabled LDS for A, B, and D)
            CHECK(output.find("ds_write") != std::string::npos);
            CHECK(output.find("ds_read") != std::string::npos);

            // Check that MFMA instructions were generated (since we set MFMA dimensions)
            CHECK(output.find("v_mfma") != std::string::npos);
        }
    }
}
