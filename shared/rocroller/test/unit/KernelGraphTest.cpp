// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <common/CommonGraphs.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

namespace KernelGraphTest
{

    class KernelGraphTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::SaveAssembly, true);
            CurrentGPUContextFixture::SetUp();

            fastArith = Expression::FastArithmetic(m_context);
        }

        void GPU_SAXPBY(bool reload);
    };

    class KernelGraphTest : public GenericContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            GenericContextFixture::SetUp();
            fastArith = Expression::FastArithmetic(m_context);
        }
    };

    // TODO: Add transforms and tests for VectorAdd with: 1. ForLoop
    // and 2. ForLoop+Unroll.
    //
    // The tests should also make sure the lowering fails if the
    // WorkitemCount is missing.
    //
    // See KernelGraphTestGPULoopSize :: MissingWorkitemCount
    //     KernelGraphTestGPULoopSize :: TestForLoop

    TEST_F(KernelGraphTest, BasicTranslateLinear)
    {
        auto example       = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto kgraph0       = example.getKernelGraph();
        auto commandParams = example.getCommandParameters();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }


        // Coordinate graph: 3 User (2 inputs + 1 output), 3 SubDimension, 5 Linear
        EXPECT_EQ(kgraph0.coordinates.getNodes<User>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<Linear>().to<std::vector>().size(), 5u);

        // Control graph: 1 Kernel, 2 LoadLinear(Int32), 3 Assign VGPR, 1 StoreLinear
        EXPECT_EQ(kgraph0.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<LoadLinear>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<Assign>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreLinear>().to<std::vector>().size(), 1u);


        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform      = std::make_shared<LowerLinear>(m_context);
        auto updateParametersTransform = std::make_shared<UpdateParameters>(commandParams);

        auto kgraph1 = kgraph0.transform(updateParametersTransform);
        auto kgraph2 = kgraph1.transform(lowerLinearTransform);
        // After LowerLinear: loads become LoadVGPR, store becomes StoreVGPR.
        // New coordinate node types: 5 VGPR, 3 Workgroup, 3 Workitem for tiling.
        EXPECT_EQ(kgraph2.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph2.control.getNodes<LoadVGPR>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph2.control.getNodes<Assign>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph2.control.getNodes<StoreVGPR>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph2.coordinates.getNodes<VGPR>().to<std::vector>().size(), 5u);
        EXPECT_EQ(kgraph2.coordinates.getNodes<Workgroup>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph2.coordinates.getNodes<Workitem>().to<std::vector>().size(), 3u);


        int  loopSize     = 16;
        auto loopSizeExpr = Expression::literal(loopSize);

        auto lowerLinerLoopTransform = std::make_shared<LowerLinearLoop>(loopSizeExpr, m_context);

        auto kgraph3 = kgraph2.transform(lowerLinerLoopTransform);
        // After LowerLinearLoop: 1 ForLoopOp + 2 Assign SGPR (init/increment);
        // 3 ForLoop + 1 Linear coordinate nodes added for the loop variable.
        EXPECT_EQ(kgraph3.control.getNodes<ForLoopOp>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph3.control.getNodes<Assign>().to<std::vector>().size(), 5u); // 3 VGPR + 2 SGPR
        EXPECT_EQ(kgraph3.coordinates.getNodes<ForLoop>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph3.coordinates.getNodes<Linear>().to<std::vector>().size(), 6u); // 5 original + 1 loop
    }

    TEST_F(KernelGraphTest, BasicTranslateScalar)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>(true);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }


        // Scalar variant: inputs map directly to VGPR, no SubDimension/Linear decomposition.
        // 5 VGPR coordinate nodes; no store operation in scalar variant.
        EXPECT_EQ(kgraph0.coordinates.getNodes<VGPR>().to<std::vector>().size(), 5u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 0u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<Linear>().to<std::vector>().size(), 0u);
        EXPECT_EQ(kgraph0.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<LoadVGPR>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<Assign>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreLinear>().to<std::vector>().size(), 0u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreVGPR>().to<std::vector>().size(), 0u);
    }

    TEST_F(KernelGraphTest, TranslateMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Int32);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }


        // Coordinate graph: 3 User, 4 SubDimension, 3 MacroTile (2 input + 1 output).
        EXPECT_EQ(kgraph0.coordinates.getNodes<User>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<MacroTile>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 6u);

        // Control graph: 1 Kernel, 2 LoadTiled, 1 TensorContraction, 1 StoreTiled.
        EXPECT_EQ(kgraph0.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<LoadTiled>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<TensorContraction>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreTiled>().to<std::vector>().size(), 1u);

        // Mapper: TensorContraction's LHS, RHS, DEST all connect to MacroTile coordinates.
        {
            auto tcTag = *kgraph0.control.getNodes<TensorContraction>().only();
            EXPECT_TRUE(kgraph0.coordinates.get<MacroTile>(
                kgraph0.mapper.get(tcTag, NaryArgument::LHS)).has_value());
            EXPECT_TRUE(kgraph0.coordinates.get<MacroTile>(
                kgraph0.mapper.get(tcTag, NaryArgument::RHS)).has_value());
            EXPECT_TRUE(kgraph0.coordinates.get<MacroTile>(
                kgraph0.mapper.get(tcTag, NaryArgument::DEST)).has_value());
        }
    }

    TEST_F(KernelGraphTest, TranslateUnscaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::None,
                                                             Operations::ScaleMode::None);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
        T_LOAD_TILED 1 Source 0
        Tensor.FP6.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
        T_LOAD_TILED 3 Source 2
        T_Mul 1 3 Value: Float
        Tensor.Float.d2 5, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
        T_STORE_TILED 6 Source 4 Dest 5
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 2);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(2, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, TranslateScaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::Separate,
                                                             Operations::ScaleMode::Separate);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
        T_LOAD_TILED 1 Source 0
        Tensor.E8M0.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
        T_LOAD_TILED 3 Source 2
        BlockScale(Separate, {1, 32}): Data: 1, Scale: 3
        Tensor.FP6.d2 5, (base=&80, sizes={&88 &96 }, strides={&104 &112 })
        T_LOAD_TILED 6 Source 5
        Tensor.E8M0.d2 7, (base=&120, sizes={&128 &136 }, strides={&144 &152 })
        T_LOAD_TILED 8 Source 7
        BlockScale(Separate, {32, 1}): Data: 6, Scale: 8
        T_Mul 4 9 Value: Float
        Tensor.Float.d2 11, (base=&160, sizes={&168 &176 }, strides={&184 &192 })
        T_STORE_TILED 12 Source 10 Dest 11
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 4);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(4, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, LowerTensor)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto kgraph0 = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform      = std::make_shared<UnrollLoops>(params, m_context);
        auto fuseLoopsTransform        = std::make_shared<FuseLoops>();
        auto removeDuplicatesTransform = std::make_shared<RemoveDuplicates>();

        auto cleanLoopsTransform = std::make_shared<CleanLoops>();
        auto updateWavefrontParametersTransform
            = std::make_shared<UpdateWavefrontParameters>(params);
        auto assignIndexExprsTransform
            = std::make_shared<AssignIndexExpressions>(m_context, example.getCommand());

        kgraph0      = kgraph0.transform(updateParametersTransform);
        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(lowerTensorContractionTransform);

        // Verify the number of Multiply nodes in the graph after lowerTile
        auto multiplyNodes = kgraph1.control.getNodes<Multiply>().to<std::vector>();
        EXPECT_EQ(multiplyNodes.size(), macK / waveK);

        // Verify number of loads
        auto loads = kgraph0.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 3); // A, B, C

        loads = kgraph1.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 4); // 1 for A, 2 for B (no LDS), 1 for C

        loads = kgraph1.control.getNodes<LoadLDSTile>().to<std::vector>();
        EXPECT_EQ(loads.size(), 2); // 2 for A

        auto forLoops = kgraph1.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(forLoops.size(), 5); // main: X, Y, K; epilogue: X, Y

        auto kgraphUnrolled = kgraph1.transform(unrollLoopsTransform);

        // Verify that loops have been unrolled
        auto unrolledForLoops = kgraphUnrolled.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(unrolledForLoops.size(), 14);

        auto kgraphFused = kgraphUnrolled.transform(fuseLoopsTransform);
        kgraphFused      = kgraphFused.transform(removeDuplicatesTransform);

        // Verify that loops have been fused
        auto fusedForLoops = kgraphFused.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(fusedForLoops.size(), 5);

        auto fusedLoads = kgraphFused.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(fusedLoads.size(), 17);

        // Verify that single iteration loops have been removed.
        auto kgraphClean     = kgraphFused.transform(cleanLoopsTransform);
        auto cleanedForLoops = kgraphClean.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(cleanedForLoops.size(), 1);

        // Verify that there is only a single StoreLDSTile node per K loop
        auto unrolled_kgraph_lds = kgraphUnrolled.transform(addLDSTransform);
        auto unrolledStoreLDS
            = unrolled_kgraph_lds.control.getNodes<StoreLDSTile>().to<std::vector>();
        auto kloops = kgraphUnrolled.control.getNodes<ForLoopOp>()
                          .filter([&](int node) {
                              auto loop = kgraphUnrolled.control.getNode<ForLoopOp>(node);
                              return loop.loopName == KLOOP;
                          })
                          .to<std::vector>();
        EXPECT_EQ(unrolledStoreLDS.size(), kloops.size());

        // Verify number of Assigns: A loads; A LDS loads; B loads; C load; D
        kgraph1           = kgraph1.transform(updateWavefrontParametersTransform);
        kgraph1           = kgraph1.transform(assignIndexExprsTransform);
        auto indexAssigns = kgraph1.control.getNodes<Assign>().to<std::vector>();
        EXPECT_EQ(indexAssigns.size(), 44);

        // Verify number of deallocated dimensions.  They may be merged into fewer deallocate nodes.
        auto addDeallocate = std::make_shared<AddDeallocateDataFlow>();
        auto kgraph2       = kgraph1.transform(addDeallocate);
        {
            std::set<int> deallocatedDims;
            auto          deallocates = kgraph2.control.getNodes<Deallocate>();
            for(auto deallocate : deallocates)
            {
                auto connections = kgraph2.mapper.getConnections(deallocate);
                for(auto const& c : connections)
                {
                    EXPECT_THAT(deallocatedDims, ::testing::Not(::testing::Contains(c.coordinate)));
                    deallocatedDims.insert(c.coordinate);
                }
            }
            EXPECT_EQ(deallocatedDims.size(), 61);
        }

        auto storeLDS = kgraphUnrolled.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(storeLDS.size(), 8);

        auto fusedStoreLDS = kgraphFused.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(fusedStoreLDS.size(), 1);

        // Verify number of Assigns after unroll/fuse/lds
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(updateWavefrontParametersTransform);
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(assignIndexExprsTransform);
        indexAssigns        = unrolled_kgraph_lds.control.getNodes<Assign>().to<std::vector>();
        EXPECT_EQ(indexAssigns.size(), 248);

        // Verify number of deallocated dimensions.  They may be merged into fewer deallocate nodes.
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(addDeallocate);
        {
            std::set<int> deallocatedDims;
            auto          deallocates = unrolled_kgraph_lds.control.getNodes<Deallocate>();
            for(auto deallocate : deallocates)
            {
                auto connections = unrolled_kgraph_lds.mapper.getConnections(deallocate);
                for(auto const& c : connections)
                {
                    EXPECT_THAT(deallocatedDims, ::testing::Not(::testing::Contains(c.coordinate)));
                    deallocatedDims.insert(c.coordinate);
                }
            }
            EXPECT_EQ(deallocatedDims.size(), 303);
        }
    }

    TEST_F(KernelGraphTest, InlineIncrement)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, true);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerLinearTransform      = std::make_shared<LowerLinear>(m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform = std::make_shared<UnrollLoops>(params, m_context);
        auto cleanLoopsTransform  = std::make_shared<CleanLoops>();
        auto updateWavefrontParametersTransform
            = std::make_shared<UpdateWavefrontParameters>(params);
        auto assignIndexExprsTransform
            = std::make_shared<AssignIndexExpressions>(m_context, example.getCommand());
        auto inlineInrecrementsTransform = std::make_shared<InlineIncrements>();

        kgraph = kgraph.transform(updateParametersTransform);
        kgraph = kgraph.transform(addLDSTransform);
        kgraph = kgraph.transform(lowerLinearTransform);
        kgraph = kgraph.transform(lowerTileTransform);
        kgraph = kgraph.transform(lowerTensorContractionTransform);

        // Usual lowering, should be able to inline everything.
        auto kgraph1 = kgraph.transform(unrollLoopsTransform);
        kgraph1      = kgraph1.transform(updateWavefrontParametersTransform);
        kgraph1      = kgraph1.transform(cleanLoopsTransform);
        kgraph1      = kgraph1.transform(assignIndexExprsTransform);

        auto pre1  = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();
        kgraph1    = kgraph1.transform(inlineInrecrementsTransform);
        auto post1 = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();

        EXPECT_TRUE(pre1.size() > 0);
        EXPECT_TRUE(post1.empty());
    }

    TEST_F(KernelGraphTest, TileAdd)
    {
        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();

        example.setTileSize(16, 8);
        example.setSubTileSize(4, 2);

        auto params  = example.getCommandParameters(512, 512);
        auto kgraph0 = example.getKernelGraph();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);


        // Coordinate graph: 3 User, 4 SubDimension, 5 MacroTile (1 LDS + 4 VGPR).
        EXPECT_EQ(kgraph0.coordinates.getNodes<User>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 6u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<MacroTile>().to<std::vector>().size(), 5u);
        {
            int ldsCount = 0, vgprCount = 0;
            for(auto id : kgraph0.coordinates.getNodes<MacroTile>())
            {
                auto mt = *kgraph0.coordinates.get<MacroTile>(id);
                if(mt.memoryType == MemoryType::LDS)
                    ldsCount++;
                else if(mt.memoryType == MemoryType::VGPR)
                    vgprCount++;
            }
            EXPECT_EQ(ldsCount, 1);  // A tile stored in LDS
            EXPECT_EQ(vgprCount, 4); // intermediate and result tiles in VGPR
        }

        // Control graph: 1 Kernel, 2 LoadTiled, 3 Assign VGPR (double-add ops), 1 StoreTiled.
        EXPECT_EQ(kgraph0.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<LoadTiled>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<Assign>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreTiled>().to<std::vector>().size(), 1u);

        auto addLDSTransform    = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform = std::make_shared<LowerTile>(params, m_context);
        auto assignIndexExprsTransform
            = std::make_shared<AssignIndexExpressions>(m_context, example.getCommand());

        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(assignIndexExprsTransform);

        namespace CG = rocRoller::KernelGraph::ControlGraph;
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadTiled>().to<std::vector>().size(), 2);
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadLDSTile>().to<std::vector>().size(), 1);
        ASSERT_EQ(kgraph1.control.getNodes<CG::StoreLDSTile>().to<std::vector>().size(), 1);
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform = std::make_shared<LowerLinear>(m_context);

        auto kgraph0 = translate(command);
        auto kgraph1 = kgraph0.transform(lowerLinearTransform);

        auto user0   = 1;
        auto subdim0 = 2;
        auto block0  = 25;
        auto thread0 = 26;

        // given block id and thread id, compute regular (user) index for first (0) dataflow array
        auto block_id  = Expression::literal(2);
        auto thread_id = Expression::literal(33);

        auto exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 64:U32)U32, 33:I)U32}, "
                  "CommandArgument(Tensor_0_stride_0)I64)I64}");

        auto stride = getStride(kgraph1.coordinates.getNode(subdim0));
        auto fa     = fastArith(stride);

        exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        sexpr = Expression::toString(fastArith(exprs[0]));
        EXPECT_EQ(sexpr, "{Split: Multiply(161:U32, Tensor_0_stride_0_0:I64)I64}");
    }

#if 0
    TEST_F(KernelGraphTestGPU, GPU_Translate03)
    {
        TIMER(t_total, "Translate03");
        TIMER(t_gpu, "Translate03::GPU");
        TIMER(t_hip, "Translate03::HIP");

        TIC(t_total);

        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 0)));

        command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        Operations::T_Store_Linear store_C(1, 5);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate03");

        auto kgraph2 = commandKernel.getKernelGraph();

        auto kernelNode = kgraph2.control.getRootOperation();

        {
            auto expected = getTag(kernelNode);
            auto outputs
                = kgraph2.control.getOutputs<Body>(getTag(kernelNode));
            EXPECT_EQ(2, outputs.size());

            auto outputs2 = kgraph2.control.getOutputs<Sequence>(
                getTag(kernelNode));
            EXPECT_EQ(0, outputs2.size());

            auto outputs3
                = kgraph2.control.getOutputs(getTag(kernelNode), Body{});

            auto outputTags3 = kgraph2.control.getOutputTags(getTag(kernelNode),
                                                             Body{});

            EXPECT_EQ(outputs3.size(), outputTags3.size());
            for(size_t i = 0; i < outputs3.size(); i++)
            {
                EXPECT_EQ(getTag(outputs3[i]), outputTags3[i]);
            }

            EXPECT_EQ(getTag(outputs[0]), getTag(outputs3[0]));

            auto inputs1
                = kgraph2.control.getInputs<Body>(getTag(outputs.at(0)));
            ASSERT_EQ(1, inputs1.size());

            auto actual1 = getTag(inputs1.at(0));
            EXPECT_EQ(actual1, expected);

            auto inputs2 = kgraph2.control.getInputs(getTag(outputs.at(0)),
                                                     Body{});
            ASSERT_EQ(1, inputs2.size());

            auto inputTags2 = kgraph2.control.getInputTags(getTag(outputs.at(0)),
                                                           Body{});

            EXPECT_EQ(inputs2.size(), inputTags2.size());
            for(size_t i = 0; i < inputs2.size(); i++)
            {
                EXPECT_EQ(getTag(inputs2[i]), inputTags2[i]);
            }

            auto actual2 = getTag(inputs2.at(0));
            EXPECT_EQ(actual1, actual2);

            auto inputs3 = kgraph2.control.getInputs<Sequence>(
                getTag(outputs.at(0)));
            EXPECT_EQ(0, inputs3.size());

            auto inputs4 = kgraph2.control.getInputs<Initialize>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs4.size());

            auto inputs5 = kgraph2.control.getInputs<ForLoopIncrement>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs5.size());
        }

        {
            std::ostringstream msg;
            msg << kgraph2.control;

            std::ostringstream msg2;
            kgraph2.control.toDOT(msg2, "krn");

            EXPECT_EQ(msg.str(), msg2.str());
        }

        TIC(t_hip);
        size_t nx = 64;

        RandomGenerator random(17629u);
        auto            a = random.vector<int>(nx, -100, 100);
        auto            b = random.vector<int>(nx, -100, 100);

        auto user0 = make_shared_device(a);
        auto user2 = make_shared_device(b);
        auto user4 = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);
        TOC(t_hip);

        KernelArguments runtimeArgs;
        runtimeArgs.append("user0", user0.get());
        runtimeArgs.append("user1", nx);
        runtimeArgs.append("user2", nx);
        runtimeArgs.append("user3", (size_t)1);
        runtimeArgs.append("user4", user2.get());
        runtimeArgs.append("user5", nx);
        runtimeArgs.append("user6", nx);
        runtimeArgs.append("user7", (size_t)1);
        runtimeArgs.append("user8", user4.get());
        runtimeArgs.append("user9", nx);
        runtimeArgs.append("user10", (size_t)1);

        TIC(t_gpu);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        TOC(t_gpu);

        TIC(t_hip);
        ASSERT_THAT(hipMemcpy(r.data(), user4.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));
        TOC(t_hip);

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i] * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        TIC(t_hip);
        user0.reset();
        user2.reset();
        user4.reset();
        TOC(t_hip);

        TOC(t_total);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }
#endif

    void KernelGraphTestGPU::GPU_SAXPBY(bool reload)
    {
        RandomGenerator random(1263u);

        size_t nx = 64;

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;
        int beta  = 33;

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto example = rocRollerTest::Graphs::VectorAdd<int>(true);
        auto command = example.getCommand();
        auto runtimeArgs
            = example.getRuntimeArguments(nx, d_alpha.get(), beta, d_a.get(), d_b.get(), d_c.get());

        CommandKernel commandKernel(command, testKernelName());
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // launch again, using saved assembly
        auto assemblyFileName = m_context->assemblyFileName();

        if(reload)
        {
            commandKernel.loadKernelFromAssembly(assemblyFileName, testKernelName());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }

        std::vector<int> r(nx);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(alpha, beta, a, b));

        ASSERT_LT(rnorm, 1.e-12);

        if(reload)
        {
            // load, using bad kernel name
            EXPECT_THROW(commandKernel.loadKernelFromAssembly(assemblyFileName, "SAXPBY_BAD"),
                         FatalError);

            // load, using non-existant file
            EXPECT_THROW(
                commandKernel.loadKernelFromAssembly(assemblyFileName + "_bad", testKernelName()),
                FatalError);

            std::filesystem::remove(assemblyFileName);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBY)
    {
        GPU_SAXPBY(false);
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYDebug)
    {
        // Make sure Debug mode doesn't introduce bad pointer
        // references in observers
        auto settings = Settings::getInstance();
        settings->set(Settings::LogLvl, LogLevel::Debug);
        GPU_SAXPBY(false);
        settings->reset();
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYLoadAssembly)
    {
        GPU_SAXPBY(true);
    }

    TEST_F(KernelGraphTestGPU, GPU_LeakyRelu)
    {
        auto command = std::make_shared<rocRoller::Command>();

        constexpr auto dataType = DataType::Float;

        auto xTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        auto xLoadTag   = command->addOperation(rocRoller::Operations::T_Load_Linear(xTensorTag));

        auto alphaScalarTag = command->addOperation(
            rocRoller::Operations::Scalar({dataType, PointerType::PointerGlobal}));
        auto alphaLoadTag
            = command->addOperation(rocRoller::Operations::T_Load_Scalar(alphaScalarTag));

        auto zeroLiteralTag = command->addOperation(rocRoller::Operations::Literal(0.0f));

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto condTag
            = execute.addXOp(rocRoller::Operations::E_GreaterThan(xLoadTag, zeroLiteralTag));
        auto productTag = execute.addXOp(rocRoller::Operations::E_Mul(xLoadTag, alphaLoadTag));
        auto reluTag
            = execute.addXOp(rocRoller::Operations::E_Conditional(condTag, xLoadTag, productTag));
        command->addOperation(std::move(execute));

        auto reluTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        command->addOperation(rocRoller::Operations::T_Store_Linear(reluTag, reluTensorTag));

        CommandKernel commandKernel(command, "LeakyRelu");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx    = 64;
        float  alpha = 0.9;

        RandomGenerator random(135679u);
        auto            a = random.vector<float>(nx, -5, 5);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device<float>(nx);
        auto d_alpha = make_shared_device<float>();

        std::vector<float> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(xTensorTag, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(xTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(xTensorTag, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(alphaScalarTag, ArgumentType::Value, d_alpha.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(reluTensorTag, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            x[i] = a[i] > 0 ? a[i] : alpha * a[i];
        }

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_LinearCopy)
    {
        auto command = std::make_shared<rocRoller::Command>();

        Operations::Tensor tensor_A(1, DataType::Int32);
        auto               tagTensorA
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_A)));

        Operations::Tensor tensor_B(1, DataType::Int32);
        auto               tagTensorB
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_B)));

        Operations::T_Load_Linear load_A(tagTensorA);
        auto                      tagLoadStore
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));
        Operations::T_Store_Linear store_B(tagLoadStore, tagTensorB);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_B)));

        CommandKernel commandKernel(command, "LinearCopy");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx = 64;

        RandomGenerator random(135679u);
        auto            a = random.vector<int>(nx, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i];

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    template <typename T>
    void CopyStrideOverride(CommandKernelPtr& commandKernel, bool override = false)
    {
        auto example = rocRollerTest::Graphs::TileCopy<T>();

        example.setTileSize(16, 4);
        example.setSubTileSize(4, 2);

        if(override)
        {
            example.setLiteralStrides({(size_t)0, (size_t)1});
        }

        size_t nx = 256;
        size_t ny = 128;

        RandomGenerator random(193674u);
        auto            ax = static_cast<T>(-100.);
        auto            ay = static_cast<T>(100.);
        auto            a  = random.vector<T>(nx * ny, ax, ay);

        std::vector<T> r(nx * ny, 0.);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<T>(nx * ny);

        auto command     = example.getCommand();
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get());

        std::string colName    = (override) ? "ColOverride" : "";
        std::string kernelName = "TensorTileCopy" + colName + TypeInfo<T>::Name();

        commandKernel = std::make_shared<CommandKernel>(command, kernelName);
        commandKernel->setContext(Context::ForDefaultHipDevice(kernelName));
        commandKernel->setCommandParameters(params);
        commandKernel->generateKernel();

        commandKernel->setLaunchParameters(launch);
        commandKernel->launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto   ref   = example.referenceSolution(a);
        double rnorm = relativeNormL2(r, ref);

        std::ostringstream msg;
        for(int i = 0; i < r.size(); i++)
        {
            msg << i << " " << r[i] << " | " << ref[i];
            if(r[i] != ref[i])
                msg << " * ";
            msg << std::endl;
        }

        ASSERT_LT(rnorm, 1.e-12) << msg.str();
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopy)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<int>(commandKernel);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideHalf)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<Half>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dword "))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dword "))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideFloat)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<float>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx2"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx2"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideDouble)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<double>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx4"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx4"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd)
    {
        size_t nx = 256; // tensor size x
        size_t ny = 512; // tensor size y

        RandomGenerator random(129674u);

        auto a = random.vector<int>(nx * ny, -100, 100);
        auto b = random.vector<int>(nx * ny, -100, 100);
        auto r = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();
        example.setTileSize(8, 64);
        example.setSubTileSize(2, 8);

        auto command     = example.getCommand();
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get(), d_c.get());
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);

        CommandKernel commandKernel(command, "TensorTileAdd");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.setLaunchParameters(launch);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(a, b));

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTest, CleanArguments)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto cleanArgumentsTransform = std::make_shared<CleanArguments>(m_context, command);

        auto kgraph = translate(command);

        auto beforePredicates = {HasSubstr("SubDimension{0, CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("SubDimension{0, CommandArgument(Tensor_2_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_2_size_0)I64}")};

        // Note that these searches do not include the close braces ("}").  This is because the
        // argument name will have a number appended which is subject to change
        // (Load_Linear_0_size_0 might become Load_Linear_0_size_0_2).
        auto afterPredicates = {
            HasSubstr("SubDimension{0, Tensor_0_size_0"),
            HasSubstr("SubDimension{0, Tensor_2_size_0"),
            HasSubstr("Linear{Tensor_0_size_0"),
            HasSubstr("Linear{Tensor_2_size_0"),
        };

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, pred);

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, Not(pred));
        }

        kgraph = kgraph.transform(cleanArgumentsTransform);

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, Not(pred));

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, pred);
        }
    }

    TEST_F(KernelGraphTest, Basic)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        // Control Graph
        int kernel_index = kgraph.control.addElement(Kernel());
        int loadA_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int loadB_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int body1_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadA_index});
        int body2_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadB_index});

        int op1_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(5)});
        int sequence1_index = kgraph.control.addElement(Sequence(), {loadA_index}, {op1_index});
        int sequence2_index = kgraph.control.addElement(Sequence(), {loadB_index}, {op1_index});

        int op2_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(7)});
        int sequence3_index = kgraph.control.addElement(Sequence(), {op1_index}, {op2_index});

        int op3_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(9)});
        int sequence4_index = kgraph.control.addElement(Sequence(), {op1_index}, {op3_index});
        int sequence5_index = kgraph.control.addElement(Sequence(), {op2_index}, {op3_index});

        int storeC_index    = kgraph.control.addElement(StoreLinear());
        int sequence6_index = kgraph.control.addElement(Sequence(), {op3_index}, {storeC_index});

        // Coordinate Graph
        int u1_index       = kgraph.coordinates.addElement(User());
        int sd1_index      = kgraph.coordinates.addElement(SubDimension());
        int split1_index   = kgraph.coordinates.addElement(Split(), {u1_index}, {sd1_index});
        int linear1_index  = kgraph.coordinates.addElement(Linear());
        int flatten1_index = kgraph.coordinates.addElement(Flatten(), {sd1_index}, {linear1_index});
        int dataflow1_index
            = kgraph.coordinates.addElement(DataFlow(), {u1_index}, {linear1_index});
        int buffer1_index = kgraph.coordinates.addElement(
            rocRoller::KernelGraph::CoordinateGraph::Buffer(), {u1_index}, {linear1_index});

        int u2_index       = kgraph.coordinates.addElement(User());
        int sd2_index      = kgraph.coordinates.addElement(SubDimension());
        int split2_index   = kgraph.coordinates.addElement(Split(), {u2_index}, {sd2_index});
        int linear2_index  = kgraph.coordinates.addElement(Linear());
        int flatten2_index = kgraph.coordinates.addElement(Flatten(), {sd2_index}, {linear2_index});
        int dataflow2_index
            = kgraph.coordinates.addElement(DataFlow(), {u2_index}, {linear2_index});

        int linear3_index   = kgraph.coordinates.addElement(Linear());
        int dataflow3_index = kgraph.coordinates.addElement(
            DataFlow(), {linear1_index, linear2_index}, {linear3_index});
        int linear4_index = kgraph.coordinates.addElement(Linear());
        int dataflow4_index
            = kgraph.coordinates.addElement(DataFlow(), {linear3_index}, {linear4_index});
        int linear5i_index  = kgraph.coordinates.addElement(Linear());
        int dataflow5_index = kgraph.coordinates.addElement(
            DataFlow(), {linear3_index, linear4_index}, {linear5i_index});

        int linear5o_index = kgraph.coordinates.addElement(Linear());
        int makeoutput1_index
            = kgraph.coordinates.addElement(MakeOutput(), {linear5i_index}, {linear5o_index});
        int sd5o_index   = kgraph.coordinates.addElement(SubDimension(0));
        int split3_index = kgraph.coordinates.addElement(Split(), {linear5o_index}, {sd5o_index});
        int u5o_index    = kgraph.coordinates.addElement(User({}, ""));
        int join1_index  = kgraph.coordinates.addElement(Join(), {sd5o_index}, {u5o_index});
        int dataflow6_index
            = kgraph.coordinates.addElement(DataFlow(), {linear5i_index}, {u5o_index});

        auto yamlData = toYAML(kgraph);
        auto graph2   = rocRoller::KernelGraph::fromYAML(yamlData);
        auto yaml2    = toYAML(graph2);
        EXPECT_EQ(yamlData, yaml2);


        // Coordinate graph: 3 User, 3 SubDimension, 6 Linear, 1 Buffer, 1 MakeOutput, 1 Join.
        EXPECT_EQ(kgraph.coordinates.getNodes<User>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph.coordinates.getNodes<Linear>().to<std::vector>().size(), 6u);

        // Control graph: 1 Kernel, 2 LoadLinear(Float), 3 Assign VGPR (literals 5/7/9), 1 StoreLinear.
        EXPECT_EQ(kgraph.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph.control.getNodes<LoadLinear>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph.control.getNodes<Assign>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph.control.getNodes<StoreLinear>().to<std::vector>().size(), 1u);
    }

    TEST_F(KernelGraphTest, UpdateParamsTMul)
    {
        auto command = std::make_shared<Command>();

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // B
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

        auto tagTensorD
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        auto kgraph0 = translate(command);


        // Coordinate graph: 3 User, 4 SubDimension, 3 MacroTile (pre-transform: no sizes).
        EXPECT_EQ(kgraph0.coordinates.getNodes<User>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<MacroTile>().to<std::vector>().size(), 3u);
        EXPECT_EQ(kgraph0.coordinates.getNodes<SubDimension>().to<std::vector>().size(), 6u);
        for(auto id : kgraph0.coordinates.getNodes<MacroTile>())
        {
            auto mt = *kgraph0.coordinates.get<MacroTile>(id);
            EXPECT_TRUE(mt.sizes.empty());
            EXPECT_EQ(mt.memoryType, MemoryType::None);
        }

        // Control graph: 1 Kernel, 2 LoadTiled(Float), 1 TensorContraction, 1 StoreTiled.
        EXPECT_EQ(kgraph0.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<LoadTiled>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<TensorContraction>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreTiled>().to<std::vector>().size(), 1u);

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        auto macTileA = MacroTile({mac_m, mac_k}, MemoryType::VGPR); // A
        auto macTileB = MacroTile({mac_k, mac_n}, MemoryType::VGPR); // B

        auto params = std::make_shared<CommandParameters>();

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);


        // After UpdateParameters: same control graph, but MacroTiles now have
        // concrete sizes {64,64} with memory types (2 VGPR inputs, 1 WAVE accumulator).
        EXPECT_EQ(kgraph0.control.getNodes<LoadTiled>().to<std::vector>().size(), 2u);
        EXPECT_EQ(kgraph0.control.getNodes<TensorContraction>().to<std::vector>().size(), 1u);
        EXPECT_EQ(kgraph0.control.getNodes<StoreTiled>().to<std::vector>().size(), 1u);
        {
            int vgprTiles = 0, waveTiles = 0;
            for(auto id : kgraph0.coordinates.getNodes<MacroTile>())
            {
                auto mt = *kgraph0.coordinates.get<MacroTile>(id);
                EXPECT_EQ(mt.sizes, (std::vector<int>{64, 64}));
                if(mt.memoryType == MemoryType::VGPR)
                    vgprTiles++;
                else if(mt.memoryType == MemoryType::WAVE)
                    waveTiles++;
            }
            EXPECT_EQ(vgprTiles, 2); // input tiles A and B
            EXPECT_EQ(waveTiles, 1); // output accumulator tile D
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_Conditional)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto kernel = kgraph.control.addElement(Kernel());
        auto unit   = Expression::literal(1);
        auto zero   = Expression::literal(0);

        auto test = m_context->kernel()->addArgument({"foo", DataType::Int32});

        auto                    destReg = kgraph.coordinates.addElement(Linear());
        Expression::DataFlowTag destRegTag{destReg, Register::Type::Vector, DataType::Int32};

        auto beforeConditionalAssign
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(0)});
        kgraph.control.addElement(Body(), {kernel}, {beforeConditionalAssign});

        auto conditional
            = kgraph.control.addElement(ConditionalOp{test < unit, "Test Conditional"});

        kgraph.control.addElement(Sequence(), {beforeConditionalAssign}, {conditional});

        auto trueOp    = kgraph.control.addElement(Assign{Register::Type::Vector, unit});
        auto trueBody  = kgraph.control.addElement(Body(), {conditional}, {trueOp});
        auto falseOp   = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        auto falseBody = kgraph.control.addElement(Else(), {conditional}, {falseOp});

        kgraph.mapper.connect(beforeConditionalAssign, destReg, NaryArgument::DEST);
        kgraph.mapper.connect(trueOp, destReg, NaryArgument::DEST);
        kgraph.mapper.connect(falseOp, destReg, NaryArgument::DEST);

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(m_context->kernel()->preamble());
        m_context->schedule(m_context->kernel()->prolog());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        if(m_context->targetArchitecture().HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
        {
            EXPECT_THAT(output(), testing::HasSubstr("s_cmp_lt_i32 s2, 1"));
        }
        else
        {
            EXPECT_THAT(output(), testing::HasSubstr("s_cmp_lt_i32 s3, 1"));
        }
        EXPECT_THAT(output(), testing::HasSubstr("s_cbranch_scc0")); //Branch for False
        EXPECT_THAT(output(), testing::HasSubstr("s_branch")); //Branch after True
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 1")); //True Body
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 0")); //False Body
    }

    TEST_F(KernelGraphTestGPU, GPU_ConditionalExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        std::vector<int> testValues = {22, 66};

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        // set result to testValues[0]
        auto assignTrueBranch1 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch1, dstVGPR, NaryArgument::DEST);
        auto assignTrueBranch2 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch2, dstVGPR, NaryArgument::DEST);

        // set result to testValues[1]
        auto assignFalseBranch = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[1])});
        kgraph.mapper.connect(assignFalseBranch, dstVGPR, NaryArgument::DEST);

        auto workgroupExpr = k->workgroupIndex().at(0)->expression();
        auto firstConditional
            = kgraph.control.addElement(ConditionalOp{workgroupExpr < one, "First Conditional"});
        auto secondConditional = kgraph.control.addElement(
            ConditionalOp{(workgroupExpr > one) && (workgroupExpr <= two), "Second Conditional"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {firstConditional});
        kgraph.control.addElement(Body(), {firstConditional}, {assignTrueBranch1});
        kgraph.control.addElement(Else(), {firstConditional}, {secondConditional});
        kgraph.control.addElement(Body(), {secondConditional}, {assignTrueBranch2});
        kgraph.control.addElement(Else(), {secondConditional}, {assignFalseBranch});
        kgraph.control.addElement(Sequence(), {firstConditional}, {storeIndex});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], testValues[0]);
            EXPECT_EQ(result[1], testValues[1]);
            EXPECT_EQ(result[2], testValues[0]);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_DoWhileExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{dstVGPR, Register::Type::Vector, DataType::UInt32});
        auto assignVGPR = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(assignVGPR, dstVGPR, NaryArgument::DEST);
        auto assignBody = kgraph.control.addElement(Assign{Register::Type::Vector, dfa + one});
        kgraph.mapper.connect(assignBody, dstVGPR, NaryArgument::DEST);
        auto workgroupExpr = k->workgroupIndex().at(0)->expression();

        auto condVGPR = kgraph.coordinates.addElement(VGPR());
        auto condDFT  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{condVGPR, Register::Type::Vector, DataType::UInt32});

        auto assignCond = kgraph.control.addElement(Assign{Register::Type::Vector, workgroupExpr});
        kgraph.mapper.connect(assignCond, condVGPR, NaryArgument::DEST);

        auto doWhile = kgraph.control.addElement(DoWhileOp{dfa < condDFT, "Test DoWhile"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {assignVGPR});
        kgraph.control.addElement(Body(), {kernel}, {assignCond});
        kgraph.control.addElement(Sequence(), {assignCond}, {doWhile});
        kgraph.control.addElement(Sequence(), {assignVGPR}, {doWhile});
        kgraph.control.addElement(Body(), {doWhile}, {assignBody});
        kgraph.control.addElement(Sequence(), {doWhile}, {storeIndex});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], 1);
            EXPECT_EQ(result[1], 1);
            EXPECT_EQ(result[2], 2);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_F(KernelGraphTest, WaitZero)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto kernel = kgraph.control.addElement(Kernel());
        auto wait   = kgraph.control.addElement(WaitZero());
        kgraph.control.addElement(Body(), {kernel}, {wait});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt"));

        EXPECT_THAT(output(), testing::HasSubstr("vmcnt(0)"));
        EXPECT_THAT(output(), testing::HasSubstr("lgkmcnt(0)"));
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasExpcnt))
        {
            EXPECT_THAT(output(), testing::HasSubstr("expcnt(0)"));
        }
    }

    TEST_F(KernelGraphTest, ReindexConditionalOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User({}, "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto conditional = kgraph.control.addElement(ConditionalOp{exprA > unit, "conditional"});
        kgraph.control.addElement(Sequence(), {loadA}, {conditional});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {conditional});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, conditional, reindexer);

        auto condition = kgraph.control.get<ConditionalOp>(conditional)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }

    TEST_F(KernelGraphTest, ReindexAssertOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User(Operations::OperationTag(0), "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto assertOp = kgraph.control.addElement(AssertOp{"assert", exprA > unit});
        kgraph.control.addElement(Sequence(), {loadA}, {assertOp});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {assertOp});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, assertOp, reindexer);

        auto condition = kgraph.control.get<AssertOp>(assertOp)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }

    TEST_F(KernelGraphTest, Transformer)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto kgraph0 = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform      = std::make_shared<UnrollLoops>(params, m_context);
        auto fuseLoopsTransform        = std::make_shared<FuseLoops>();
        auto removeDuplicatesTransform = std::make_shared<RemoveDuplicates>();

        auto cleanLoopsTransform = std::make_shared<CleanLoops>();
        auto assignIndexExprsTransform
            = std::make_shared<AssignIndexExpressions>(m_context, example.getCommand());

        kgraph0      = kgraph0.transform(updateParametersTransform);
        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(lowerTensorContractionTransform);

        //
        // Build transformer one by one
        //
        std::unordered_map<int, Transformer> transformers;
        for(auto op : kgraph1.control.getNodes())
            transformers.emplace(op, kgraph1.buildTransformer(op));

        //
        // Build all transformers at once
        //
        kgraph1.buildAllTransformers();

        //
        // The resulting transformers should be identical
        //
        for(auto op : kgraph1.control.getNodes())
        {
            auto const& expected = transformers.at(op).getIndexes();
            auto const& actual   = kgraph1.buildTransformer(op).getIndexes();

            ASSERT_EQ(expected.size(), actual.size());
            for(auto const& [dim, expr] : expected)
            {
                auto it = actual.find(dim);
                ASSERT_NE(it, actual.end());
                EXPECT_TRUE(Expression::identical(expr, it->second));
            }
        }
    }

    TEST_F(KernelGraphTest, RemoveSetCoordinate)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        int kernel = kgraph.control.addElement(Kernel());

        int nop1 = kgraph.control.addElement(NOP());
        int nop2 = kgraph.control.addElement(NOP());
        int nop3 = kgraph.control.addElement(NOP());
        int nop4 = kgraph.control.addElement(NOP());
        int nop5 = kgraph.control.addElement(NOP());
        int nop6 = kgraph.control.addElement(NOP());

        auto one = Expression::literal(1u);
        int  sc1 = kgraph.control.addElement(SetCoordinate(one));
        int  sc2 = kgraph.control.addElement(SetCoordinate(one));
        int  sc3 = kgraph.control.addElement(SetCoordinate(one));
        int  sc4 = kgraph.control.addElement(SetCoordinate(one));
        int  sc5 = kgraph.control.addElement(SetCoordinate(one));
        int  sc6 = kgraph.control.addElement(SetCoordinate(one));

        int dim = kgraph.coordinates.addElement(Adhoc());
        kgraph.mapper.connect<Adhoc>(sc1, dim);
        kgraph.mapper.connect<Adhoc>(sc2, dim);
        kgraph.mapper.connect<Adhoc>(sc3, dim);
        kgraph.mapper.connect<Adhoc>(sc4, dim);
        kgraph.mapper.connect<Adhoc>(sc5, dim);
        kgraph.mapper.connect<Adhoc>(sc6, dim);

        //  Original:
        //
        //          Kernel
        //            |
        //            |[body]
        //            v
        //           nop1
        //            |
        //            |[seq]
        //            v
        //           nop2 --------------
        //            |                |
        //            |[body]          |[seq]
        //            v                v
        //           sc1              sc2  -------------
        //            |                |               |
        //            |[seq]           |[body]         |[body]
        //            v                v               v
        //           sc3              nop3            sc4 ------------------
        //            |------------                    |                   |
        //            |           |                    |[seq]              |[seq]
        //            |[seq]      |[seq]               v                   v
        //            v           v                   nop4                nop5
        //           sc5         sc6
        //                        |
        //                        |[body]
        //                        v
        //                       nop6
        //

        kgraph.control.addElement(Body(), {kernel}, {nop1});
        kgraph.control.addElement(Sequence(), {nop1}, {nop2});
        kgraph.control.addElement(Body(), {nop2}, {sc1});
        kgraph.control.addElement(Sequence(), {sc1}, {sc3});
        kgraph.control.addElement(Sequence(), {sc3}, {sc5});
        kgraph.control.addElement(Sequence(), {sc3}, {sc6});
        kgraph.control.addElement(Body(), {sc6}, {nop6});

        kgraph.control.addElement(Sequence(), {nop2}, {sc2});
        kgraph.control.addElement(Body(), {sc2}, {nop3});
        kgraph.control.addElement(Body(), {sc2}, {sc4});
        kgraph.control.addElement(Sequence(), {sc4}, {nop4});
        kgraph.control.addElement(Sequence(), {sc4}, {nop5});

        auto removeSetCoordinate = std::make_shared<RemoveSetCoordinate>();
        auto kg2                 = kgraph.transform(removeSetCoordinate);

        //  After:
        //
        //          Kernel
        //            |
        //            |[body]
        //            v
        //           nop1
        //            |
        //            |[seq]
        //            v
        //           nop2 -------------------------------------
        //            |                |           |          |
        //            |[body]          |[seq]      |[seq]     |[seq]
        //            v                v           v          v
        //           nop6              nop3        nop4       nop5
        //


        // After RemoveSetCoordinate: all 6 SetCoordinate nodes removed,
        // 6 NOP nodes remain, and nop2 now directly parents nop3-nop6.
        EXPECT_EQ(kg2.control.getNodes<SetCoordinate>().to<std::vector>().size(), 0u);
        EXPECT_EQ(kg2.control.getNodes<NOP>().to<std::vector>().size(), 6u);
        EXPECT_EQ(kg2.control.getNodes<Kernel>().to<std::vector>().size(), 1u);
        // nop3, nop4, nop5, nop6 are now leaves (direct children of nop2 with no children of their own)
        {
            auto leaves = kg2.control.leaves().to<std::unordered_set>();
            std::unordered_set<int> leafNOPs;
            for(auto id : leaves)
                if(kg2.control.get<NOP>(id).has_value())
                    leafNOPs.insert(id);
            EXPECT_THAT(leafNOPs, UnorderedElementsAre(nop3, nop4, nop5, nop6));
        }
    }

    TEST_F(KernelGraphTest, StreamKTwoTileDPFirst)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        uint numTileM = 57;
        uint numTileN = 57;
        uint numTileK = 57;
        uint numWGs   = 128;

        auto kernel = kgraph.control.addElement(Kernel());
        auto [forKCoord, forKOp]
            = rangeFor(kgraph, Expression::literal(numTileK), rocRoller::KLOOP);

        auto user = kgraph.coordinates.addElement(User({}, "result"));

        auto tileM = kgraph.coordinates.addElement(
            MacroTileNumber(0, Expression::literal(numTileM), nullptr));
        auto tileN = kgraph.coordinates.addElement(
            MacroTileNumber(1, Expression::literal(numTileN), nullptr));
        auto tileK = kgraph.coordinates.addElement(
            MacroTileNumber(-1, Expression::literal(numTileK), nullptr));

        kgraph.coordinates.addElement(PassThrough(), {forKCoord}, {tileK});
        kgraph.coordinates.addElement(Flatten(), {tileM, tileN, tileK}, {user});

        auto dstVGPR = kgraph.coordinates.addElement(VGPR());
        auto wgDim   = kgraph.coordinates.addElement(Workgroup(0));
        auto wgExpr  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{wgDim, Register::Type::Vector, DataType::UInt32});
        auto assignWGNumber = kgraph.control.addElement(Assign{Register::Type::Vector, wgExpr});
        kgraph.mapper.connect(assignWGNumber, dstVGPR, NaryArgument::DEST);

        kgraph.coordinates.addElement(PassThrough(), {user}, {dstVGPR});

        auto storeOp = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeOp, user);
        kgraph.mapper.connect<VGPR>(storeOp, dstVGPR);

        auto preWaitOp  = kgraph.control.addElement(WaitZero());
        auto loopWaitOp = kgraph.control.addElement(WaitZero());

        kgraph.control.addElement(Body(), {kernel}, {preWaitOp});
        kgraph.control.addElement(Sequence(), {preWaitOp}, {forKOp});
        kgraph.control.addElement(Body(), {forKOp}, {assignWGNumber});
        kgraph.control.addElement(Sequence(), {assignWGNumber}, {storeOp});
        kgraph.control.addElement(Sequence(), {storeOp}, {loopWaitOp});

        CommandParametersPtr params           = std::make_shared<CommandParameters>();
        params->loopOverOutputTilesDimensions = {0, 1};

        // Helper to check loop order in a given kernel graph
        auto checkStreamKLoopOrder = [&](rocRoller::KernelGraph::KernelGraph& kg,
                                         const std::string&                   firstLoop,
                                         const std::string&                   secondLoop) {
            int body = -1, sequence = -1;
            for(auto const scope :
                filter(kg.control.isElemType<Scope>(),
                       kg.control.depthFirstVisit(kg.control.roots().only().value())))
            {
                auto bodies = kg.control.getOutputNodeIndices<Body>(scope).to<std::unordered_set>();
                auto sequences
                    = kg.control.getOutputNodeIndices<Sequence>(scope).to<std::unordered_set>();

                if(bodies.size() + sequences.size() == 1)
                    continue;

                AssertFatal(
                    bodies.size() == 1 && sequences.size() == 1,
                    "Expected one body and one sequence in scope, found {} bodies and {} sequences",
                    bodies.size(),
                    sequences.size());

                body     = *bodies.begin();
                sequence = *sequences.begin();
                break;
            }

            // Check for first loop through the Body edge
            for(auto const loop :
                filter(kg.control.isElemType<ForLoopOp>(), kg.control.depthFirstVisit(body)))
            {
                auto forloop = kg.control.get<ForLoopOp>(loop).value();
                EXPECT_EQ(forloop.loopName, firstLoop);
                break;
            }

            // Check for second loop through the Sequence edge
            for(auto const loop :
                filter(kg.control.isElemType<ForLoopOp>(), kg.control.depthFirstVisit(sequence)))
            {
                auto forloop = kg.control.get<ForLoopOp>(loop).value();
                EXPECT_EQ(forloop.loopName, secondLoop);
                break;
            }
        };

        // For streamKTwoTile, the SK loop is first
        params->streamK = StreamKMode::TwoTile;

        auto kgraphTwoTile = kgraph.transform(std::make_shared<AddStreamK>(
            m_context, params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs)));

        if(m_context->kernelOptions()->removeSetCoordinate)
            kgraphTwoTile = kgraphTwoTile.transform(std::make_shared<RemoveSetCoordinate>());

        checkStreamKLoopOrder(kgraphTwoTile, "SKStreamTileLoop", "DPStreamTileLoop");

        m_context->kernel()->resetArguments();

        // For streamKTwoTileDPFirst, the DP loop is first
        params->streamK = StreamKMode::TwoTileDPFirst;

        auto kgraphTwoTileDPFirst = kgraph.transform(std::make_shared<AddStreamK>(
            m_context, params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs)));

        if(m_context->kernelOptions()->removeSetCoordinate)
            kgraphTwoTileDPFirst
                = kgraphTwoTileDPFirst.transform(std::make_shared<RemoveSetCoordinate>());

        checkStreamKLoopOrder(kgraphTwoTileDPFirst, "DPStreamTileLoop", "SKStreamTileLoop");
    }
}
