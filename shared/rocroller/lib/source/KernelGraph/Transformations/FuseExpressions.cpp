#include <algorithm>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    std::optional<std::tuple<int, int>> findMultiplyAdd(KernelGraph const&             kgraph,
                                                        std::unordered_set<int> const& exclude)
    {
        for(auto const& parent : kgraph.control.getNodes<Assign>())
        {
            if(exclude.contains(parent))
                continue;

            // Find Assign nodes with Multiply expression
            auto multiplyAssign = kgraph.control.get<Assign>(parent);
            if(!std::holds_alternative<Expression::Multiply>(*multiplyAssign->expression))
                continue;

            // Constraint 1: Multiply must have exactly ONE Sequence child
            auto child = only(kgraph.control.getOutputNodeIndices<Sequence>(parent));
            if(!child)
                continue;

            // Constraint 2: The Sequence child must be an Assign node
            auto addAssign = kgraph.control.get<Assign>(*child);
            if(!addAssign)
                continue;

            // Verify the Multiply has a valid DEST coordinate
            auto dst = kgraph.mapper.get(parent, NaryArgument::DEST);
            AssertFatal(dst != -1, "Invalid connection.");

            // Constraint 3: The child Assign must be an Add expression
            if(!std::holds_alternative<Expression::Add>(*addAssign->expression))
                continue;

            return {{parent, *child}};
        }

        return {};
    }
    void reflow(KernelGraph& graph, int dim, std::vector<int> const& other_inputs)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        std::vector<int>        inputs;
        std::vector<int>        outputs;
        std::unordered_set<int> deletedEdges;
        // Use the same nested-variant inspection style that already exists in
        // RegisterTagManager.cpp instead of relying on getEdges<CT::DataFlow>().
        auto isPlainDataFlowEdge = [&graph](int edgeTag) {
            auto edge = graph.coordinates.getEdge(edgeTag);
            if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                return false;
            return std::holds_alternative<CT::DataFlow>(std::get<CT::DataFlowEdge>(edge));
        };
        // Collect incoming DataFlow edges where dim is a destination:
        //   [parents...] -> dim
        // Those parents become inputs to the replacement edge.
        for(auto edgeTag : graph.coordinates.getNeighbours<Graph::Direction::Upstream>(dim))
        {
            if(!graph.coordinates.get<CT::DataFlow>(edgeTag))
                continue;
            auto parents = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
            inputs.insert(inputs.end(), parents.begin(), parents.end());
            deletedEdges.insert(edgeTag);
        }
        // Standard case:
        // If the hypergraph exposes downstream neighbours for dim directly,
        // use that existing path first.
        auto downstreamEdges = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(dim);
        if(!downstreamEdges.empty())
        {
            for(auto edgeTag : downstreamEdges)
            {
                if(!graph.coordinates.get<CT::DataFlow>(edgeTag))
                    continue;
                auto children
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                outputs.insert(outputs.end(), children.begin(), children.end());
                deletedEdges.insert(edgeTag);
            }
        }
        else
        {
            // Multi-source hyperedge case:
            // dim may still be one source of a DataFlow edge even if
            // getNeighbours<Downstream>(dim) returns empty.
            auto allEdges
                = graph.coordinates.getEdges().filter(isPlainDataFlowEdge).to<std::vector>();
            for(auto edgeTag : allEdges)
            {
                if(deletedEdges.contains(edgeTag))
                    continue;
                auto sources = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
                if(std::find(sources.begin(), sources.end(), dim) == sources.end())
                    continue;
                auto children
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                outputs.insert(outputs.end(), children.begin(), children.end());
                deletedEdges.insert(edgeTag);
            }
        }
        inputs.insert(inputs.end(), other_inputs.begin(), other_inputs.end());
        for(auto edgeTag : deletedEdges)
            graph.coordinates.deleteElement(edgeTag);
        // Never create a dangling hyperedge.
        if(!inputs.empty() && !outputs.empty())
            graph.coordinates.addElement(CT::DataFlow(), inputs, outputs);
    }

    KernelGraph fuseMultiplyAdd(KernelGraph const& original)
    {
        using Expression::multiplyAdd;

        auto kgraph = original;

        std::unordered_set<int> excluded;

        for(;;)
        {
            auto candidate = findMultiplyAdd(kgraph, excluded);
            if(!candidate)
                break;

            auto const& [multiplyTag, addTag] = *candidate;
            excluded.emplace(multiplyTag);

            // Constraint 4: Add's LHS must equal Multiply's DEST
            auto [addLHS, addLHSDF] = getBinaryLHS<Expression::Add>(kgraph, addTag);
            auto mulDST             = getDEST(kgraph, multiplyTag);

            if(addLHS != mulDST)
                continue;

            // All constraints passed - perform the fusion
            auto addDST             = getDEST(kgraph, addTag);
            auto [addRHS, addRHSDF] = getBinaryRHS<Expression::Add>(kgraph, addTag);
            auto [mulLHS, mulLHSDF] = getBinaryLHS<Expression::Multiply>(kgraph, multiplyTag);
            auto [mulRHS, mulRHSDF] = getBinaryRHS<Expression::Multiply>(kgraph, multiplyTag);
            auto fma                = multiplyAdd(mulLHSDF, mulRHSDF, addRHSDF);

            // Reuse register type and count from Multiply
            auto fmaAssign       = *kgraph.control.get<Assign>(multiplyTag);
            fmaAssign.expression = fma;

            auto fmaTag = kgraph.control.addElement(fmaAssign);

            // Connect FMA; delete Multiply and Add operations
            reconnect<Graph::Direction::Downstream>(kgraph, -1, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, addTag);
            reconnect<Graph::Direction::Downstream>(kgraph, fmaTag, addTag);

            kgraph.control.deleteElement(multiplyTag);
            kgraph.control.deleteElement(addTag);

            // Tidy up coordinate graph using the fixed reflow function
            reflow(kgraph, addLHS, {addRHS});
            kgraph.coordinates.deleteElement(addLHS);

            // Tidy up connections
            kgraph.mapper.purge(multiplyTag);
            kgraph.mapper.purge(addTag);

            // Connect FMA
            kgraph.mapper.connect(fmaTag, addDST, NaryArgument::DEST);
        }

        return kgraph;
    }

    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        return fuseMultiplyAdd(original);
    }
}
