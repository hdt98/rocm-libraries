// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/DataDependenceDAG.hpp>
#include <rocRoller/KernelGraph/ControlGraph/DataDependenceDAG_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    namespace DataDependenceDAG
    {
        using ReadWrite = ControlFlowRWTracer::ReadWrite;

        ControlGraph::ControlGraph ConstructDataDependenceDAG(KernelGraph const& graph)
        {
            using namespace Detail;

            DataDependenceDAGDetail obj(graph);
            obj.constructDataDependenceDAG();
            return obj.getDataDependenceDAG();
        }

        namespace Detail
        {
            DataDependenceDAGDetail::DataDependenceDAGDetail(KernelGraph const& graph)
                : m_graph(graph)
            {
                // Insert all control graph nodes into the data dependence DAG
                for(auto node : m_graph.control.getNodes())
                {
                    m_dependenceDAG.setElement(node, m_graph.control.getElement(node));
                }
            }

            ControlGraph::ControlGraph DataDependenceDAGDetail::getDataDependenceDAG()
            {
                return m_dependenceDAG;
            }

            void DataDependenceDAGDetail::constructDataDependenceDAG()
            {
                auto tracer  = ControlFlowRWTracer(m_graph);
                auto records = tracer.coordinatesReadWrite();

                // This assumes that the trace is ordered and records for the
                // same control operation are consecutive.
                std::unordered_set<int> seen;
                for(auto iter = records.begin(); iter != records.end();)
                {
                    auto currentControl = iter->control;

                    AssertFatal(seen.find(currentControl) == seen.end(),
                                "The records for the same control operation are not consecutive.",
                                ShowValue(currentControl));

                    for(; iter != records.end() && iter->control == currentControl; ++iter)
                    {
                        processReadWriteRecord(*iter);
                    }

                    seen.insert(currentControl);
                }
            }

            int DataDependenceDAGDetail::getBodyParent(int control)
            {
                if(auto iter = m_bodyParentCache.find(control); iter != m_bodyParentCache.end())
                    return iter->second;

                auto topSetCoordinate = getTopSetCoordinate(m_graph, control);
                auto bodyParent       = bodyParents(topSetCoordinate, m_graph).take(1).only();
                AssertFatal(bodyParent.has_value(),
                            "Control node has no body parent",
                            ShowValue(control),
                            ShowValue(topSetCoordinate));

                m_bodyParentCache.emplace(control, bodyParent.value());
                return bodyParent.value();
            }

            bool DataDependenceDAGDetail::belongToSameBasicBlock(int node1, int node2)
            {
                auto top1 = getTopSetCoordinate(m_graph, node1);
                auto top2 = getTopSetCoordinate(m_graph, node2);

                auto bodyParent1 = getBodyParent(top1);
                auto bodyParent2 = getBodyParent(top2);

                if(bodyParent1 != bodyParent2)
                    return false;

                auto isContainingEdge = [this](int tag) -> bool {
                    return m_graph.control.getElementType(tag) == Graph::ElementType::Edge
                           && !std::holds_alternative<ControlGraph::Sequence>(
                               m_graph.control.getEdge(tag));
                };

                auto path1 = m_graph.control
                                 .path<Graph::Direction::Downstream>(std::vector<int>{bodyParent1},
                                                                     std::vector<int>{top1})
                                 .filter(isContainingEdge)
                                 .to<std::vector>();
                auto path2 = m_graph.control
                                 .path<Graph::Direction::Downstream>(std::vector<int>{bodyParent2},
                                                                     std::vector<int>{top2})
                                 .filter(isContainingEdge)
                                 .to<std::vector>();

                AssertFatal(!path1.empty() && !path2.empty(),
                            "Each path must contain at least one containing edge",
                            ShowValue(path1.size()),
                            ShowValue(path2.size()));

                // Check if all elements in a path are of same edge type.
                auto allSameEdgeTypes = [this](const auto& path) {
                    return std::all_of(path.begin() + 1,
                                       path.end(),
                                       [firstIndex = m_graph.control.getEdge(path[0]).index(),
                                        this](const auto& e) {
                                           return m_graph.control.getEdge(e).index() == firstIndex;
                                       });
                };

                AssertFatal(allSameEdgeTypes(path1),
                            "path1 contains multiple types of containing edges");
                AssertFatal(allSameEdgeTypes(path2),
                            "path2 contains multiple types of containing edges");

                // Check if both paths hold the same edge types.
                if(m_graph.control.getEdge(path1[0]).index()
                   == m_graph.control.getEdge(path2[0]).index())
                    return true;

                return false;
            }

            void DataDependenceDAGDetail::addDependenceEdge(int source, int dest)
            {
                AssertFatal(source != dest, ShowValue(source), ShowValue(dest));

                if(!belongToSameBasicBlock(source, dest))
                    return;

                if(!m_dependenceDAG.findEdge(source, dest).has_value())
                {
                    m_dependenceDAG.addElement(ControlGraph::Sequence(), {source}, {dest});
                }
            }

            void DataDependenceDAGDetail::processReadWriteRecord(
                ControlFlowRWTracer::ReadWriteRecord const& record)
            {
                AssertFatal(record.rw != ReadWrite::Count,
                            ShowValue(record.control),
                            ShowValue(record.coordinate),
                            ShowValue(record.rw));

                if(auto writeIter = m_latestWriteToCoord.find(record.coordinate);
                   writeIter != m_latestWriteToCoord.end())
                {
                    AssertFatal(writeIter->second != record.control,
                                ShowValue(writeIter->second),
                                ShowValue(record.control),
                                ShowValue(record.coordinate),
                                ShowValue(record.rw));

                    // adds WW(output dep) or WR(flow dep) edge
                    addDependenceEdge(writeIter->second, record.control);
                }

                if(record.rw == ReadWrite::WRITE || record.rw == ReadWrite::READWRITE)
                {
                    for(auto const readControl : m_latestReadsToCoord[record.coordinate])
                    {
                        if(readControl == record.control)
                            continue;

                        // adds RW(anti dep) edges
                        addDependenceEdge(readControl, record.control);
                    }

                    // Since the current control node writes into this coord,
                    // the latest reads info needs to be reset.
                    m_latestReadsToCoord[record.coordinate].clear();
                    // update the latest write to coord
                    m_latestWriteToCoord[record.coordinate] = record.control;
                }

                if(record.rw == ReadWrite::READ || record.rw == ReadWrite::READWRITE)
                {
                    m_latestReadsToCoord[record.coordinate].insert(record.control);
                }
            }
        }
    }
}
