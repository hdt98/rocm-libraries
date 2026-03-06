
#include <common/SourceMatcher.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/KernelGraph/ControlGraph/DataDependenceDAG_detail.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

using namespace rocRoller::KernelGraph::DataDependenceDAG::Detail;
using namespace Catch::Matchers;

namespace DataDependenceDAGDetailTest
{
    namespace CF = rocRoller::KernelGraph::ControlGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    TEST_CASE("Object creation inserts all nodes from the control graph into the dependence graph",
              "[data-dependence-dag]")
    {
        auto graph = rocRoller::KernelGraph::KernelGraph();

        SECTION("Control graph with no nodes and no edges")
        {
            DataDependenceDAGDetail obj(graph);
            auto                    depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 0);
            CHECK(graph.control.getElementCount() == depDAG.getElementCount());
        }

        SECTION("Control graph with nodes and no edges")
        {
            graph.control.addElement(CF::Kernel());

            DataDependenceDAGDetail obj(graph);
            auto                    depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 1);
            CHECK(graph.control.getElementCount() == depDAG.getElementCount());
        }

        SECTION("Control graph with nodes and edges")
        {
            auto kernel = graph.control.addElement(CF::Kernel());
            auto nop    = graph.control.addElement(CF::NOP());
            graph.control.addElement(CF::Sequence(), {kernel}, {nop});

            DataDependenceDAGDetail obj(graph);
            auto                    depDAG = obj.getDataDependenceDAG();
            CHECK(graph.control.getElementCount() == 3);
            CHECK(depDAG.getElementCount() == 2);
        }
    }
}
