
#include <common/SourceMatcher.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/KernelGraph/ControlGraph/DataDependenceDAG_detail.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller;
using namespace rocRoller::Expression;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::DataDependenceDAG::Detail;
using namespace Catch::Matchers;

namespace DataDependenceDAGDetailTest
{
    TEST_CASE("Object creation inserts all nodes from the control graph into the dependence graph",
              "[data-dependence-dag]")
    {
        auto graph = KernelGraph::KernelGraph();

        SECTION("Control graph with no nodes and no edges")
        {
            DataDependenceDAGDetail obj(graph);

            auto depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 0);
            CHECK(graph.control.getElementCount() == depDAG.getElementCount());
        }

        SECTION("Control graph with nodes and no edges")
        {
            graph.control.addElement(Kernel());

            DataDependenceDAGDetail obj(graph);

            auto depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 1);
            CHECK(graph.control.getElementCount() == depDAG.getElementCount());
        }

        SECTION("Control graph with nodes and edges")
        {
            auto kernel = graph.control.addElement(Kernel());
            auto nop    = graph.control.addElement(NOP());
            graph.control.addElement(Sequence(), {kernel}, {nop});

            DataDependenceDAGDetail obj(graph);

            auto depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 3);
            CHECK(depDAG.getElementCount() == 2);
        }
    }

    TEST_CASE("construct data dependence directed acyclic graph", "[data-dependence-dag]")
    {
        KernelGraph::KernelGraph graph;

        // computes Nth Fibonacci number
        auto kernel = graph.control.addElement(Kernel());

        auto loadN = graph.control.addElement(LoadVGPR(DataType::UInt32));
        auto vgprN = graph.coordinates.addElement(VGPR());
        auto userN = graph.coordinates.addElement(User());
        graph.mapper.connect<User>(loadN, userN);
        graph.mapper.connect<VGPR>(loadN, vgprN);

        // if(N <= 1)
        auto exprN       = dataFlowTag(vgprN, Register::Type::Scalar, DataType::UInt32);
        auto conditional = graph.control.addElement(ConditionalOp{exprN <= literal(1u)});

        graph.control.addElement(Body(), {kernel}, {loadN});
        graph.control.addElement(Sequence(), {loadN}, {conditional});

        auto vgprResult = graph.coordinates.addElement(VGPR());

        auto assignResult = graph.control.addElement(Assign{Register::Type::Scalar, exprN});
        graph.mapper.connect(assignResult, vgprResult, NaryArgument::DEST);

        // then, Result = N
        graph.control.addElement(Body(), {conditional}, {assignResult});

        auto assignF0 = graph.control.addElement(Assign{Register::Type::Scalar, literal(0u)});
        auto assignF1 = graph.control.addElement(Assign{Register::Type::Scalar, literal(1u)});
        auto vgprF0   = graph.coordinates.addElement(VGPR());
        auto vgprF1   = graph.coordinates.addElement(VGPR());
        graph.mapper.connect(assignF0, vgprF0, NaryArgument::DEST);
        graph.mapper.connect(assignF1, vgprF1, NaryArgument::DEST);

        // else, F0 = 0 -> F1 = 1 -> loop
        graph.control.addElement(Else(), {conditional}, {assignF0});
        graph.control.addElement(Sequence(), {assignF0}, {assignF1});

        auto vgprI = graph.coordinates.addElement(VGPR());
        auto exprI = dataFlowTag(vgprI, Register::Type::Scalar, DataType::UInt32);
        // for(I = 2; I <=N, I++)
        auto forLoop = graph.control.addElement(ForLoopOp(exprI <= exprN, "loop"));
        graph.mapper.connect(forLoop, vgprI, NaryArgument::DEST);
        auto forInit = graph.control.addElement(Assign{Register::Type::Scalar, literal(2u)});
        graph.mapper.connect(forInit, vgprI, NaryArgument::DEST);
        int forInc = graph.control.addElement(Assign{Register::Type::Scalar, exprI + literal(1u)});
        graph.mapper.connect(forInc, vgprI, NaryArgument::DEST);

        graph.control.addElement(Sequence(), {assignF1}, {forLoop});
        graph.control.addElement(Initialize(), {forLoop}, {forInit});
        graph.control.addElement(ForLoopIncrement(), {forLoop}, {forInc});

        auto exprF0   = dataFlowTag(vgprF0, Register::Type::Scalar, DataType::UInt32);
        auto exprF1   = dataFlowTag(vgprF1, Register::Type::Scalar, DataType::UInt32);
        auto assignF2 = graph.control.addElement(Assign{Register::Type::Scalar, exprF0 + exprF1});
        graph.mapper.connect(assignF2, vgprResult, NaryArgument::DEST);

        // F2 = F0 + F1
        graph.control.addElement(Body(), {forLoop}, {assignF2});

        auto assignF1ToF0 = graph.control.addElement(Assign{Register::Type::Scalar, exprF1});
        graph.mapper.connect(assignF1ToF0, vgprF0, NaryArgument::DEST);
        auto exprF2       = dataFlowTag(vgprResult, Register::Type::Scalar, DataType::UInt32);
        auto assignF2ToF1 = graph.control.addElement(Assign{Register::Type::Scalar, exprF2});
        graph.mapper.connect(assignF2ToF1, vgprF1, NaryArgument::DEST);

        // F0 = F1
        graph.control.addElement(Sequence(), {assignF2}, {assignF1ToF0});
        // F1 = F2
        graph.control.addElement(Sequence(), {assignF1ToF0}, {assignF2ToF1});

        auto storeResult = graph.control.addElement(StoreVGPR());
        auto userResult  = graph.coordinates.addElement(User());
        graph.mapper.connect<User>(storeResult, userResult);
        graph.mapper.connect<VGPR>(storeResult, vgprResult);

        graph.control.addElement(Sequence(), {conditional}, {storeResult});

        DataDependenceDAGDetail obj(graph);
        obj.constructDataDependenceDAG();

        auto depDAG = obj.getDataDependenceDAG();

        CHECK(graph.control.findEdge(kernel, loadN).has_value());
        CHECK(!depDAG.findEdge(kernel, loadN).has_value());

        // conditional if(N <= 1) reads loadN
        // WR (flow) dependency
        CHECK(graph.control.findEdge(loadN, conditional).has_value());
        CHECK(depDAG.findEdge(loadN, conditional).has_value());
        CHECK(obj.getBodyParent(loadN) == kernel);
        CHECK(obj.getBodyParent(conditional) == kernel);

        CHECK(graph.control.findEdge(conditional, assignResult).has_value());
        CHECK(!depDAG.findEdge(conditional, assignResult).has_value());

        CHECK(graph.control.findEdge(conditional, assignF0).has_value());
        CHECK(!depDAG.findEdge(conditional, assignF0).has_value());

        CHECK(graph.control.findEdge(assignF0, assignF1).has_value());
        CHECK(!depDAG.findEdge(assignF0, assignF1).has_value());

        CHECK(graph.control.findEdge(assignF1, forLoop).has_value());
        CHECK(!depDAG.findEdge(assignF1, forLoop).has_value());

        // WW (output) dependency
        CHECK(!graph.control.findEdge(forInit, forInc).has_value());
        CHECK(depDAG.findEdge(forInit, forInc).has_value());
        CHECK(obj.getBodyParent(forInit) == forLoop);
        CHECK(obj.getBodyParent(forInc) == forLoop);

        // F2 = F0 + F1 reads both F0 and F1,
        // but assignF0 and assignF2 have different body parents
        // and assignF1 and assignF2 also have different body parents,
        // meaning they belong to different basic blocks,
        // therefore, no dependence edges between them.
        CHECK(!graph.control.findEdge(assignF0, assignF2).has_value());
        CHECK(!depDAG.findEdge(assignF0, assignF2).has_value());
        CHECK(!graph.control.findEdge(assignF1, assignF2).has_value());
        CHECK(!depDAG.findEdge(assignF0, assignF2).has_value());
        CHECK(obj.getBodyParent(assignF0) == conditional);
        CHECK(obj.getBodyParent(assignF1) == conditional);
        CHECK(obj.getBodyParent(assignF2) == forLoop);

        // F0 = F1
        // RW (anti) dep
        CHECK(graph.control.findEdge(assignF2, assignF1ToF0).has_value());
        CHECK(depDAG.findEdge(assignF2, assignF1ToF0).has_value());
        CHECK(obj.getBodyParent(assignF2) == forLoop);
        CHECK(obj.getBodyParent(assignF1ToF0) == forLoop);

        // F1 = F2
        // WR (flow) dependency
        CHECK(!graph.control.findEdge(assignF2, assignF2ToF1).has_value());
        CHECK(depDAG.findEdge(assignF2, assignF2ToF1).has_value());
        CHECK(obj.getBodyParent(assignF2) == forLoop);
        CHECK(obj.getBodyParent(assignF2ToF1) == forLoop);
    }
}
