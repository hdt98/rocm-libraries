// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;

        namespace LowerLinearDetail
        {
            void LowerLoadLinear(KernelGraph const&        original,
                                 KernelGraph&              graph,
                                 int                       tag,
                                 Expression::ExpressionPtr workgroupSizeX,
                                 GraphReindexer&           reindexer)
            {
                auto one       = Expression::literal(1u);
                auto oload     = *original.control.get<LoadLinear>(tag);
                auto userTag   = original.mapper.get<User>(tag);
                auto linearTag = original.mapper.get<Linear>(tag);

                auto vgpr      = graph.coordinates.addElement(VGPR());
                auto numTilesX = (graph.coordinates.get<User>(userTag)->size + workgroupSizeX - one)
                                 / workgroupSizeX;
                auto workgroup = graph.coordinates.addElement(Workgroup(0, numTilesX));
                auto workitem  = graph.coordinates.addElement(Workitem(0, workgroupSizeX));

                graph.coordinates.addElement(Tile(), {linearTag}, {workgroup, workitem});
                graph.coordinates.addElement(Forget(), {workgroup, workitem}, {vgpr});
                graph.coordinates.addElement(DataFlow(), {userTag}, {vgpr});

                graph.control.setElement(tag, LoadVGPR(oload.varType));

                Log::debug("KernelGraph::lowerLinear(): LoadLinear {} -> LoadVGPR {}: {} -> {}",
                           tag,
                           tag,
                           linearTag,
                           vgpr);

                reindexer.coordinates[linearTag] = vgpr;

                graph.mapper.purge(tag);
                graph.mapper.connect<User>(tag, userTag);
                graph.mapper.connect<VGPR>(tag, vgpr);
            }

            void LowerAssign(KernelGraph const& original,
                             KernelGraph&       graph,
                             int                tag,
                             GraphReindexer&    reindexer)
            {
                auto linearTag = original.mapper.get(tag, NaryArgument::DEST);
                if(not original.coordinates.get<Linear>(linearTag))
                    return;

                auto vgpr = graph.coordinates.addElement(VGPR());

                std::vector<int> inputs;
                for(auto input : original.coordinates.parentNodes(linearTag))
                    inputs.push_back(input);
                graph.coordinates.addElement(DataFlow(), inputs, std::vector<int>{vgpr});

                // Reindex DataFlowTags in the expression: input Linear tags -> VGPR tags.
                reindexExpressions(graph, tag, reindexer);

                reindexer.coordinates[linearTag] = vgpr;

                graph.mapper.purge(tag);
                graph.mapper.connect(tag, vgpr, NaryArgument::DEST);
            }

            void LowerStoreLinear(KernelGraph const&        original,
                                  KernelGraph&              graph,
                                  int                       tag,
                                  Expression::ExpressionPtr workgroupSizeX,
                                  GraphReindexer const&     reindexer)
            {
                auto one       = Expression::literal(1u);
                auto userTag   = original.mapper.get<User>(tag);
                auto linearTag = original.mapper.get<Linear>(tag);
                auto vgprTag   = reindexer.coordinates.at(linearTag);

                auto numTilesX = (graph.coordinates.get<User>(userTag)->size + workgroupSizeX - one)
                                 / workgroupSizeX;
                auto workgroup = graph.coordinates.addElement(Workgroup(0, numTilesX));
                auto workitem  = graph.coordinates.addElement(Workitem(0, workgroupSizeX));

                graph.coordinates.addElement(Inherit(), {vgprTag}, {workgroup, workitem});
                graph.coordinates.addElement(Flatten(), {workgroup, workitem}, {linearTag});
                graph.coordinates.addElement(DataFlow(), {vgprTag}, {userTag});

                graph.control.setElement(tag, StoreVGPR());

                Log::debug("KernelGraph::lowerLinear(): StoreLinear {} -> StoreVGPR {}: {} -> {}",
                           tag,
                           tag,
                           linearTag,
                           vgprTag);

                graph.mapper.purge(tag);
                graph.mapper.connect<User>(tag, userTag);
                graph.mapper.connect<VGPR>(tag, vgprTag);
            }

        } // namespace LowerLinearDetail

        KernelGraph LowerLinear::apply(KernelGraph const& original)
        {
            using namespace LowerLinearDetail;
            KernelGraph graph = original;

            auto workgroupSizeX = Expression::literal(m_context->kernel()->workgroupSize()[0]);

            GraphReindexer reindexer;

            for(auto tag : original.control.getNodes<LoadLinear>())
                LowerLoadLinear(original, graph, tag, workgroupSizeX, reindexer);

            for(auto const& [linear, vgpr] : reindexer.coordinates)
                Log::debug("KernelGraph::LowerLinear(): Linear {} -> VGPR {}", linear, vgpr);

            for(auto tag : original.control.getNodes<Assign>())
                LowerAssign(original, graph, tag, reindexer);

            for(auto tag : original.control.getNodes<StoreLinear>())
                LowerStoreLinear(original, graph, tag, workgroupSizeX, reindexer);

            // Drop DataFlow edges connected to Linear nodes.
            // Done last so that parentNodes lookups above still work.
            for(auto tag : original.coordinates.getEdges<DataFlowEdge>())
            {
                auto edge = original.coordinates.getEdge(tag);
                if(not std::get_if<DataFlow>(&std::get<DataFlowEdge>(edge)))
                    continue;
                auto loc       = original.coordinates.getLocation(tag);
                auto connected = loc.incoming;
                connected.insert(connected.end(), loc.outgoing.begin(), loc.outgoing.end());
                bool drop = std::any_of(connected.begin(), connected.end(), [&](int idx) {
                    return std::holds_alternative<Linear>(original.coordinates.getNode(idx));
                });
                if(drop)
                    graph.coordinates.deleteElement(tag);
            }

            return graph;
        }

    }
}
