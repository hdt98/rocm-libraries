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

                    // The READ records should appear before the WRITE records,
                    // essentially first READ, then READWRITE and then WRITE.
                    auto prev = ReadWrite::READ;
                    for(; iter != records.end() && iter->control == currentControl; ++iter)
                    {
                        switch(iter->rw)
                        {
                        case ReadWrite::READ:
                            AssertFatal(prev == ReadWrite::READ);
                            break;
                        case ReadWrite::READWRITE:
                            AssertFatal(prev == ReadWrite::READ || prev == ReadWrite::READWRITE);
                            break;
                        case ReadWrite::WRITE:
                            break;
                        default:
                            Throw<FatalError>("Invalid ReadWrite.");
                        }

                        processReadWriteRecord(*iter);

                        prev = iter->rw;
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

            bool DataDependenceDAGDetail::addDependenceEdge(int source, int dest)
            {
                AssertFatal(source != dest, ShowValue(source), ShowValue(dest));

                if(getBodyParent(source) != getBodyParent(dest))
                    return false;

                if(!m_dependenceDAG.findEdge(source, dest).has_value())
                {
                    m_dependenceDAG.addElement(ControlGraph::Sequence(), {source}, {dest});
                }

                return true;
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
                    auto depAdded = addDependenceEdge(writeIter->second, record.control);

                    //TODO: Check the order and ensure that writeIter->second
                    //      happens before record.control.
                    //if(depAdded)
                    //{
                    //    auto order = m_graph.control.compareNodes(
                    //        UseCacheIfAvailable, writeIter->second, record.control);
                    //    AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                    //                    || order == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                    //                ShowValue(order),
                    //                ShowValue(writeIter->second),
                    //                ShowValue(record.control),
                    //                ShowValue(record.coordinate),
                    //                ShowValue(record.rw));
                    //}
                }

                if(record.rw == ReadWrite::WRITE || record.rw == ReadWrite::READWRITE)
                {
                    for(auto const readControl : m_latestReadsToCoord[record.coordinate])
                    {
                        if(readControl == record.control)
                            continue;

                        // adds RW(anti dep) edges
                        auto depAdded = addDependenceEdge(readControl, record.control);

                        //TODO: Check the order and ensure that writeIter->second
                        //      happens before record.control.
                        //if(depAdded)
                        //{
                        //    auto order = m_graph.control.compareNodes(
                        //        UseCacheIfAvailable, readControl, record.control);
                        //    AssertFatal(order == ControlGraph::NodeOrdering::LeftFirst
                        //                    || order
                        //                           == ControlGraph::NodeOrdering::RightInBodyOfLeft,
                        //                ShowValue(order),
                        //                ShowValue(readControl),
                        //                ShowValue(record.control),
                        //                ShowValue(record.coordinate),
                        //                ShowValue(record.rw));
                        //}
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
