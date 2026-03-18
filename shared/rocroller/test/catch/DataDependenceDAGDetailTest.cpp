
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

        // computes Nth Fibonacci number (assumes N to be an unsigned integer)
        // Result = N
        // F0 = 0, F1 = 1
        // for(I = 2; I <=N, I++)
        //   F2 = F0 + F1
        //   F0 = F1
        //   F1 = F2
        // Result = F2

        auto kernel = graph.control.addElement(Kernel());

        auto loadN = graph.control.addElement(LoadVGPR(DataType::UInt32));
        auto vgprN = graph.coordinates.addElement(VGPR());
        auto userN = graph.coordinates.addElement(User());
        graph.mapper.connect<User>(loadN, userN);
        graph.mapper.connect<VGPR>(loadN, vgprN);
        graph.control.addElement(Body(), {kernel}, {loadN});

        auto vgprResult   = graph.coordinates.addElement(VGPR());
        auto exprN        = dataFlowTag(vgprN, Register::Type::Scalar, DataType::UInt32);
        auto assignResult = graph.control.addElement(Assign{Register::Type::Scalar, exprN});
        graph.mapper.connect(assignResult, vgprResult, NaryArgument::DEST);
        graph.control.addElement(Sequence(), {loadN}, {assignResult});

        auto assignF0 = graph.control.addElement(Assign{Register::Type::Scalar, literal(0u)});
        auto assignF1 = graph.control.addElement(Assign{Register::Type::Scalar, literal(1u)});
        auto vgprF0   = graph.coordinates.addElement(VGPR());
        auto vgprF1   = graph.coordinates.addElement(VGPR());
        graph.mapper.connect(assignF0, vgprF0, NaryArgument::DEST);
        graph.mapper.connect(assignF1, vgprF1, NaryArgument::DEST);

        // "F0 = 0" -> "F1 = 1" -> loop
        graph.control.addElement(Sequence(), {assignResult}, {assignF0});
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

        graph.control.addElement(Sequence(), {forLoop}, {storeResult});

        SECTION("constructDataDependenceDAG and test if all dependence edges are valid")
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
                CHECK(outNodes.size() == 2);
                CHECK(outNodes[0] == assignResult);
                CHECK(outNodes[1] == forLoop);
                // WR (flow) dependency
                CHECK(graph.control.findEdge(loadN, assignResult).has_value());
                CHECK(depDAG.findEdge(loadN, assignResult).has_value());
                CHECK(obj.getBodyParent(loadN) == kernel);
                CHECK(obj.getBodyParent(assignResult) == kernel);
                // WR (flow) dependency
                CHECK(!graph.control.findEdge(loadN, forLoop).has_value());
                CHECK(depDAG.findEdge(loadN, forLoop).has_value());
                CHECK(obj.getBodyParent(loadN) == kernel);
                CHECK(obj.getBodyParent(forLoop) == kernel);
            }

            // assignResult
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignResult).to<std::vector>();
                CHECK(inNodes.size() == 1);
                CHECK(inNodes[0] == loadN);

                auto outNodes
                    = depDAG.getOutputNodeIndices<Sequence>(assignResult).to<std::vector>();
                CHECK(outNodes.empty());
            }

            // assignF0
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignF0).to<std::vector>();
                CHECK(inNodes.empty());

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(assignF0).to<std::vector>();
                CHECK(outNodes.empty());
            }

            // assignF1
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignF1).to<std::vector>();
                CHECK(inNodes.empty());

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(assignF1).to<std::vector>();
                CHECK(outNodes.empty());
            }

            // forLoop
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(forLoop).to<std::vector>();
                CHECK(inNodes.size() == 1);
                CHECK(inNodes[0] == loadN);

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(forLoop).to<std::vector>();
                CHECK(outNodes.empty());
            }

            // forInit
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(forInit).to<std::vector>();
                CHECK(inNodes.empty());

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(forInit).to<std::vector>();
                CHECK(outNodes.size() == 1);
                CHECK(outNodes[0] == forInc);
                // WW (output) dependency between forInit and forInc
                // because both write the vgprI coord and both have same body parent.
                // FIXME: when body parent will be defined by pair<parentID, ControlEdge>
                // instead of just parentID.
                CHECK(!graph.control.findEdge(forInit, forInc).has_value());
                CHECK(depDAG.findEdge(forInit, forInc).has_value());
                CHECK(obj.getBodyParent(forInit) == forLoop);
                CHECK(obj.getBodyParent(forInc) == forLoop);
            }

            // forInc
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(forInc).to<std::vector>();
                CHECK(inNodes.size() == 1);
                CHECK(inNodes[0] == forInit);

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(forInc).to<std::vector>();
                CHECK(outNodes.empty());
            }

            // assignF2
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignF2).to<std::vector>();
                CHECK(inNodes.empty());
                // F2 = F0 + F1 reads both F0 and F1,
                // but assignF0 and assignF2 have different body parents
                // and assignF1 and assignF2 also have different body parents,
                // meaning they belong to different basic blocks,
                // therefore, no dependence edges between them.
                CHECK(!graph.control.findEdge(assignF0, assignF2).has_value());
                CHECK(!depDAG.findEdge(assignF0, assignF2).has_value());
                CHECK(!graph.control.findEdge(assignF1, assignF2).has_value());
                CHECK(!depDAG.findEdge(assignF1, assignF2).has_value());
                CHECK(obj.getBodyParent(assignF0) == kernel);
                CHECK(obj.getBodyParent(assignF1) == kernel);
                CHECK(obj.getBodyParent(assignF2) == forLoop);

                auto outNodes = depDAG.getOutputNodeIndices<Sequence>(assignF2).to<std::vector>();
                CHECK(outNodes.size() == 2);
                CHECK(outNodes[0] == assignF1ToF0);
                CHECK(outNodes[1] == assignF2ToF1);
                // F2 = F0 + F1 reads both F0 and F1,
                // assignF1ToF0 and assignF2ToF1 write to F0 and F1 respectively,
                // and all three have same body parent,
                // so, there is anti (RW) dependence from assignF2 to assignF1ToF0,
                // and from assignF2 to assignF2ToF1.
                // Also, there is a flow (WR) dependence from assignF2 to
                // assignF2ToF1.
                CHECK(graph.control.findEdge(assignF2, assignF1ToF0).has_value());
                CHECK(depDAG.findEdge(assignF2, assignF1ToF0).has_value());
                CHECK(obj.getBodyParent(assignF2) == forLoop);
                CHECK(obj.getBodyParent(assignF1ToF0) == forLoop);
                CHECK(!graph.control.findEdge(assignF2, assignF2ToF1).has_value());
                CHECK(depDAG.findEdge(assignF2, assignF2ToF1).has_value());
                CHECK(obj.getBodyParent(assignF2) == forLoop);
                CHECK(obj.getBodyParent(assignF2ToF1) == forLoop);
            }

            // assignF1ToF0
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignF1ToF0).to<std::vector>();
                CHECK(inNodes.size() == 1);
                CHECK(inNodes[0] == assignF2);

                auto outNodes
                    = depDAG.getOutputNodeIndices<Sequence>(assignF1ToF0).to<std::vector>();
                CHECK(outNodes.size() == 1);
                CHECK(outNodes[0] == assignF2ToF1);
                // RW dep from "F0 = F1" to "F1 = F2"
                CHECK(graph.control.findEdge(assignF1ToF0, assignF2ToF1).has_value());
                CHECK(depDAG.findEdge(assignF1ToF0, assignF2ToF1).has_value());
                CHECK(obj.getBodyParent(assignF1ToF0) == forLoop);
                CHECK(obj.getBodyParent(assignF2ToF1) == forLoop);
            }

            // assignF2ToF1
            {
                auto inNodes = depDAG.getInputNodeIndices<Sequence>(assignF2ToF1).to<std::vector>();
                CHECK(inNodes.size() == 2);
                CHECK(inNodes[0] == assignF2);
                CHECK(inNodes[1] == assignF1ToF0);

                auto outNodes
                    = depDAG.getOutputNodeIndices<Sequence>(assignF2ToF1).to<std::vector>();
                CHECK(outNodes.empty());
            }

            for(auto inNode : depDAG.getInputNodeIndices<Sequence>(storeResult))
            {
                std::cout << "\t" << inNode << " -->" << std::endl;
            }
            for(auto outNode : depDAG.getOutputNodeIndices<Sequence>(storeResult))
            {
                std::cout << "\t--> " << outNode << std::endl;
            }
        }

        SECTION("process tracer records to add dependence edges if possible")
        {
            DataDependenceDAGDetail obj(graph);
            auto                    depDAG = obj.getDataDependenceDAG();

            auto tracer  = rocRoller::KernelGraph::ControlFlowRWTracer(graph);
            auto records = tracer.coordinatesReadWrite();

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

                    CHECK(iter->control != loadN);
                    break;
                }
                case 4:
                {
                    CHECK(iter->control == assignResult);
                    CHECK(iter->coordinate == vgprN);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    CHECK(!depDAG.findEdge(loadN, assignResult).has_value());
                    obj.processReadWriteRecord(*iter);
                    depDAG = obj.getDataDependenceDAG();
                    // WR (flow) dependency
                    CHECK(depDAG.findEdge(loadN, assignResult).has_value());
                    CHECK(obj.getBodyParent(loadN) == obj.getBodyParent(assignResult));
                    CHECK(obj.getBodyParent(loadN) == kernel);
                    CHECK(obj.getBodyParent(assignResult) == kernel);
                    iter++;

                    CHECK(iter->control == assignResult);
                    CHECK(iter->coordinate == vgprResult);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control != assignResult);
                    break;
                }
                case 6:
                {
                    CHECK(iter->control == assignF0);
                    CHECK(iter->coordinate == vgprF0);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    CHECK(iter->control != assignF0);
                    break;
                }
                case 7:
                {
                    CHECK(iter->control == assignF1);
                    CHECK(iter->coordinate == vgprF1);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;
                    CHECK(iter->control != assignF1);
                    break;
                }
                case 10:
                {
                    CHECK(iter->control == forLoop);
                    CHECK(iter->coordinate == vgprN);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control == forLoop);
                    CHECK(iter->coordinate == vgprI);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control != forLoop);
                    break;
                }
                case 11:
                {
                    CHECK(iter->control == forInit);
                    CHECK(iter->coordinate == vgprI);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control != forInit);
                    break;
                }
                case 12:
                {
                    CHECK(iter->control == forInc);
                    CHECK(iter->coordinate == vgprI);
                    CHECK(iter->rw == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::READ);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control == forInc);
                    CHECK(iter->coordinate == vgprI);
                    CHECK(iter->rw
                          == rocRoller::KernelGraph::ControlFlowRWTracer::ReadWrite::WRITE);
                    obj.processReadWriteRecord(*iter);
                    iter++;

                    CHECK(iter->control != forInc);
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
