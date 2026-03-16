
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

    TEST_CASE("data dependence DAG detail member functions and states", "[data-dependence-dag]")
    {
        KernelGraph::KernelGraph graph;

        // computes Nth Fibonacci number
        // if(N <= 1)
        // then
        //   Result = N
        // else
        //   F0 = 0, F1 = 1
        //   for(I = 2; I <=N, I++)
        //     F2 = F0 + F1
        //     F0 = F1
        //     F1 = F2
        //   Result = F2

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
        // node id 12
        auto forInit = graph.control.addElement(Assign{Register::Type::Scalar, literal(2u)});
        graph.mapper.connect(forInit, vgprI, NaryArgument::DEST);
        // node id 13
        auto forLoop = graph.control.addElement(ForLoopOp(exprI <= exprN, "loop"));
        graph.mapper.connect(forLoop, vgprI, NaryArgument::DEST);
        // node id 14
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

        SECTION("constructDataDependenceDAG and test if the required dependence edges are present")
        {
            DataDependenceDAGDetail obj(graph);
            obj.constructDataDependenceDAG();

            auto depDAG = obj.getDataDependenceDAG();

            // kernel node has no data dependence with any other node
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(kernel).to<std::vector>();
                CHECK(inNodes.empty());

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(kernel).to<std::vector>();
                CHECK(outNodes.empty());
                CHECK(!depDAG.findEdge(kernel, loadN).has_value());
                CHECK(graph.control.findEdge(kernel, loadN).has_value());
            }

            // loadN
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(loadN).to<std::vector>();
                CHECK(inNodes.empty());

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(loadN).to<std::vector>();
                CHECK(outNodes.size() == 1);
                CHECK(outNodes[0] == conditional);
            }

            // conditional if(N <= 1) reads loadN
            // WR (flow) dependency
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(conditional).to<std::vector>();
                CHECK(inNodes.size() == 1);
                CHECK(inNodes[0] == loadN);
                CHECK(graph.control.findEdge(loadN, conditional).has_value());
                CHECK(depDAG.findEdge(loadN, conditional).has_value());
                CHECK(obj.getBodyParent(loadN) == kernel);
                CHECK(obj.getBodyParent(conditional) == kernel);

                auto outNodes
                    = depDAG.getOutputNodeIndices<Sequence>(conditional).to<std::vector>();
                CHECK(outNodes.empty());
            }

            for(auto inNode : depDAG.getInputNodeIndices<Sequence>(assignResult))
            {
                std::cout << "\t" << inNode << " -->" << std::endl;
            }
            for(auto outNode : depDAG.getOutputNodeIndices<Sequence>(assignResult))
            {
                std::cout << "\t--> " << outNode << std::endl;
            }

            CHECK(graph.control.findEdge(conditional, assignResult).has_value());
            CHECK(!depDAG.findEdge(conditional, assignResult).has_value());

            CHECK(graph.control.findEdge(conditional, assignF0).has_value());
            CHECK(!depDAG.findEdge(conditional, assignF0).has_value());

            CHECK(graph.control.findEdge(assignF0, assignF1).has_value());
            CHECK(!depDAG.findEdge(assignF0, assignF1).has_value());

            CHECK(graph.control.findEdge(assignF1, forLoop).has_value());
            CHECK(!depDAG.findEdge(assignF1, forLoop).has_value());

            // No WW (output) dependency between forInit and forInc
            // because they belong to different basic blocks.
            // Note: they have same body parent.
            CHECK(!graph.control.findEdge(forInit, forInc).has_value());
            CHECK(!depDAG.findEdge(forInit, forInc).has_value());
            CHECK(obj.getBodyParent(forInit) == forLoop);
            CHECK(obj.getBodyParent(forInc) == forLoop);
            CHECK(obj.belongToSameBasicBlock(forInit, forInc) == false);

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

        SECTION("process tracer records to add dependence edges if possible")
        {
            DataDependenceDAGDetail obj(graph);
            auto                    depDAG = obj.getDataDependenceDAG();

            auto tracer  = rocRoller::KernelGraph::ControlFlowRWTracer(graph);
            auto records = tracer.coordinatesReadWrite();
            std::cout << records.size() << std::endl;

            for(auto iter = records.begin(); iter != records.end();)
            {
                std::cout << iter->control << " " << iter->coordinate << " " << iter->rw
                          << std::endl;
                switch(iter->control)
                {
                case 2:
                {
                    CHECK(iter->control == loadN);
                    CHECK(iter->coordinate == vgprN);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    break;
                }
                case 3:
                {
                    CHECK(iter->control == conditional);
                    CHECK(iter->coordinate == vgprN);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    CHECK(!depDAG.findEdge(loadN, conditional).has_value());

                    obj.processReadWriteRecord(*iter);
                    depDAG = obj.getDataDependenceDAG();
                    // WR (flow) dependency
                    CHECK(depDAG.findEdge(loadN, conditional).has_value());
                    iter++;
                    break;
                }
                case 6:
                {
                    CHECK(iter->control == assignResult);
                    CHECK(iter->coordinate == vgprN);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    CHECK(!depDAG.findEdge(loadN, assignResult).has_value());
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    std::cout << iter->control << " " << iter->coordinate << " " << iter->rw
                              << std::endl;
                    depDAG = obj.getDataDependenceDAG();
                    CHECK(obj.getBodyParent(loadN) != obj.getBodyParent(assignResult));
                    CHECK(obj.getBodyParent(loadN) == kernel);
                    CHECK(obj.getBodyParent(assignResult) == conditional);
                    // Since loadN and assignResult belong to different basic-blocks,
                    // so, no WR dependence edge between them.
                    CHECK(!depDAG.findEdge(loadN, assignResult).has_value());

                    // a dependence edge can only be added between the nodes
                    // that belong to the same basic-block.
                    obj.addDependenceEdge(loadN, assignResult);
                    depDAG = obj.getDataDependenceDAG();
                    CHECK(!depDAG.findEdge(loadN, assignResult).has_value());

                    CHECK(iter->control == assignResult);
                    CHECK(iter->coordinate == vgprResult);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    break;
                }
                case 8:
                {
                    CHECK(iter->control == assignF0);
                    CHECK(iter->coordinate == vgprF0);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    break;
                }
                case 9:
                {
                    CHECK(iter->control == assignF1);
                    CHECK(iter->coordinate == vgprF1);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    break;
                }
                default:
                {
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    break;
                }
                }
            }
        }
    }
}
