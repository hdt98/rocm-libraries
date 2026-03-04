
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
            }

            ControlGraph::ControlGraph DataDependenceDAGDetail::getDataDependenceDAG()
            {
                return m_dependenceDAG;
            }

            void DataDependenceDAGDetail::constructDataDependenceDAG()
            {
                // Insert all control graph nodes into the data dependence DAG
                for(auto node : m_graph.control.getNodes())
                {
                    m_dependenceDAG.setElement(node, m_graph.control.getElement(node));
                }

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

            void DataDependenceDAGDetail::addDependenceEdge(int sourceControl, int destControl)
            {
                AssertFatal(
                    sourceControl != destControl, ShowValue(sourceControl), ShowValue(destControl));

                auto sourceBodyParent = getBodyParent(sourceControl);
                auto destBodyParent   = getBodyParent(destControl);

                if(sourceBodyParent != destBodyParent)
                    return;

                if(!m_dependenceDAG.findEdge(sourceControl, destControl).has_value())
                {
                    m_dependenceDAG.addElement(
                        ControlGraph::Sequence(), {sourceControl}, {destControl});
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

                    // adds WW(output dep) and WR(flow dep) edges
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
