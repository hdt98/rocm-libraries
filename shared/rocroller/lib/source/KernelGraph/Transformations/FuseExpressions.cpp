// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <optional>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>
namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    namespace
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        std::optional<int> tryGetDataFlowTag(Expression::ExpressionPtr const& expr)
        {
            if(!expr)
                return std::nullopt;
            if(!std::holds_alternative<Expression::DataFlowTag>(*expr))
                return std::nullopt;
            return std::get<Expression::DataFlowTag>(*expr).tag;
        }
        bool containsDataFlowTag(Expression::ExpressionPtr const& expr);
        struct ContainsDataFlowTagVisitor
        {
            bool operator()(Expression::DataFlowTag const&) const
            {
                return true;
            }
            template <Expression::CUnary T>
            bool operator()(T const& expr) const
            {
                return containsDataFlowTag(expr.arg);
            }
            template <Expression::CBinary T>
            bool operator()(T const& expr) const
            {
                return containsDataFlowTag(expr.lhs) || containsDataFlowTag(expr.rhs);
            }
            template <Expression::CTernary T>
            bool operator()(T const& expr) const
            {
                return containsDataFlowTag(expr.lhs) || containsDataFlowTag(expr.r1hs)
                       || containsDataFlowTag(expr.r2hs);
            }
            template <Expression::CNary T>
            bool operator()(T const& expr) const
            {
                return std::ranges::any_of(expr.operands, [](auto const& operand) {
                    return containsDataFlowTag(operand);
                });
            }
            bool operator()(Expression::ScaledMatrixMultiply const& expr) const
            {
                return containsDataFlowTag(expr.matA) || containsDataFlowTag(expr.matB)
                       || containsDataFlowTag(expr.matC) || containsDataFlowTag(expr.scaleA)
                       || containsDataFlowTag(expr.scaleB);
            }
            template <Expression::CValue T>
            requires(!std::same_as<T, Expression::DataFlowTag>) bool operator()(T const&) const
            {
                return false;
            }
        };
        bool containsDataFlowTag(Expression::ExpressionPtr const& expr)
        {
            if(!expr)
                return false;
            return std::visit(ContainsDataFlowTagVisitor{}, *expr);
        }
        bool isSupportedFusedOperand(Expression::ExpressionPtr const& expr)
        {
            if(!expr)
                return false;
            // Direct DataFlowTags are safe.
            if(std::holds_alternative<Expression::DataFlowTag>(*expr))
                return true;
            // Also allow constant-only expressions. Reject anything that still depends on
            // coordinate dataflow, because this transform only rewires plain DataFlow edges.
            return !containsDataFlowTag(expr);
        }
        struct AddFusionMatch
        {
            Expression::ExpressionPtr otherExpr;
            std::vector<int>          otherInputs;
        };
        std::optional<AddFusionMatch> matchAddConsumer(int mulDST, Expression::Add const& addExpr)
        {
            auto lhsTag     = tryGetDataFlowTag(addExpr.lhs);
            auto rhsTag     = tryGetDataFlowTag(addExpr.rhs);
            bool lhsMatches = lhsTag.has_value() && *lhsTag == mulDST;
            bool rhsMatches = rhsTag.has_value() && *rhsTag == mulDST;
            // Reject:
            // - neither side matches the multiply result
            // - both sides match (e.g. Add(mul, mul))
            if(lhsMatches == rhsMatches)
                return std::nullopt;
            auto const& otherExpr = lhsMatches ? addExpr.rhs : addExpr.lhs;
            if(!isSupportedFusedOperand(otherExpr))
                return std::nullopt;
            std::vector<int> otherInputs;
            if(auto otherTag = tryGetDataFlowTag(otherExpr))
                otherInputs.push_back(*otherTag);
            return AddFusionMatch{otherExpr, otherInputs};
        }
        std::vector<int> plainDataFlowEdgesWithSource(KernelGraph const& graph, int dim)
        {
            auto isDataFlow = [&graph](int idx) {
                auto edge = graph.coordinates.getEdge(idx);
                if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                    return false;

                return std::holds_alternative<CT::DataFlow>(std::get<CT::DataFlowEdge>(edge));
            };

            std::vector<int> edges;
            for(auto edgeTag : graph.coordinates.getEdges().filter(isDataFlow))
            {
                auto sources = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
                if(std::find(sources.begin(), sources.end(), dim) != sources.end())
                    edges.push_back(edgeTag);
            }
            return edges;
        }
        std::vector<int> plainDataFlowEdgesWithTarget(KernelGraph const& graph, int dim)
        {
            auto isDataFlow = [&graph](int idx) {
                auto edge = graph.coordinates.getEdge(idx);
                if(!std::holds_alternative<CT::DataFlowEdge>(edge))
                    return false;

                return std::holds_alternative<CT::DataFlow>(std::get<CT::DataFlowEdge>(edge));
            };

            std::vector<int> edges;
            for(auto edgeTag : graph.coordinates.getEdges().filter(isDataFlow))
            {
                auto targets
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
                if(std::find(targets.begin(), targets.end(), dim) != targets.end())
                    edges.push_back(edgeTag);
            }
            return edges;
        }
        void sortAndUnique(std::vector<int>& values)
        {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
        bool hasSinglePlainDataFlowUse(KernelGraph const& graph,
                                       int                sourceDim,
                                       int                expectedOutputDim)
        {
            auto edges = plainDataFlowEdgesWithSource(graph, sourceDim);
            if(edges.size() != 1)
                return false;
            auto outputs
                = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edges.front());
            return outputs.size() == 1 && outputs.front() == expectedOutputDim;
        }
    } // namespace
    /**
       * @brief Look for {Assign Multiply(., .)} --Sequence--> {Assign Add(., .)}.
       *
       * Fusion is only legal if the multiply result has exactly one plain DataFlow use,
       * namely the Add result being replaced. This version keeps that legality rule but
       * implements it correctly for multi-source hyperedges by scanning all plain DataFlow
       * edges.
       */
    std::optional<std::tuple<int, int>> findMultiplyAdd(KernelGraph const&             kgraph,
                                                        std::unordered_set<int> const& exclude)
    {
        for(auto const& multiplyTag : kgraph.control.getNodes<Assign>())
        {
            if(exclude.contains(multiplyTag))
                continue;
            auto multiplyAssign = kgraph.control.get<Assign>(multiplyTag);
            if(!multiplyAssign || !multiplyAssign->expression)
                continue;
            if(!std::holds_alternative<Expression::Multiply>(*multiplyAssign->expression))
                continue;
            auto addTag = only(kgraph.control.getOutputNodeIndices<Sequence>(multiplyTag));
            if(!addTag)
                continue;
            auto addAssign = kgraph.control.get<Assign>(*addTag);
            if(!addAssign || !addAssign->expression)
                continue;
            if(!std::holds_alternative<Expression::Add>(*addAssign->expression))
                continue;
            auto mulDST = kgraph.mapper.get(multiplyTag, NaryArgument::DEST);
            auto addDST = kgraph.mapper.get(*addTag, NaryArgument::DEST);
            if(mulDST == -1 || addDST == -1)
                continue;
            auto const& mulExpr = std::get<Expression::Multiply>(*multiplyAssign->expression);
            auto const& addExpr = std::get<Expression::Add>(*addAssign->expression);
            // Keep the multiply operands simple and local to this fusion.
            if(!isSupportedFusedOperand(mulExpr.lhs) || !isSupportedFusedOperand(mulExpr.rhs))
                continue;
            auto match = matchAddConsumer(mulDST, addExpr);
            if(!match)
                continue;
            // Preserve the original legality rule, but compute it correctly.
            if(!hasSinglePlainDataFlowUse(kgraph, mulDST, addDST))
                continue;
            return {{multiplyTag, *addTag}};
        }
        return {};
    }
    /**
       * @brief Reconnect incoming and outgoing plain DataFlow edges from dimension.
       *
       * Inputs flowing into the eliminated dimension are augmented with any extra
       * DataFlowTag inputs from the surviving add operand. Only plain DataFlow edges
       * are rewritten here.
       */
    void reflow(KernelGraph& graph, int dim, std::vector<int> const& otherInputs)
    {
        std::vector<int> inputs;
        std::vector<int> outputs;
        auto             upstreamEdges   = plainDataFlowEdgesWithTarget(graph, dim);
        auto             downstreamEdges = plainDataFlowEdgesWithSource(graph, dim);
        for(auto edgeTag : upstreamEdges)
        {
            auto parents = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(edgeTag);
            inputs.insert(inputs.end(), parents.begin(), parents.end());
        }
        for(auto edgeTag : downstreamEdges)
        {
            auto children = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(edgeTag);
            outputs.insert(outputs.end(), children.begin(), children.end());
        }
        std::vector<int> edgesToDelete = upstreamEdges;
        edgesToDelete.insert(edgesToDelete.end(), downstreamEdges.begin(), downstreamEdges.end());
        sortAndUnique(edgesToDelete);
        for(auto edgeTag : edgesToDelete)
            graph.coordinates.deleteElement(edgeTag);
        inputs.insert(inputs.end(), otherInputs.begin(), otherInputs.end());
        sortAndUnique(inputs);
        sortAndUnique(outputs);
        if(!inputs.empty() && !outputs.empty())
            graph.coordinates.addElement(CT::DataFlow(), inputs, outputs);
    }
    /**
       * @brief Fuse Add(Multiply(., .), .) into MultiplyAdd(., ., .).
       */
    KernelGraph fuseMultiplyAdd(KernelGraph const& original)
    {
        using Expression::multiplyAdd;
        auto                    kgraph = original;
        std::unordered_set<int> excluded;
        for(;;)
        {
            auto candidate = findMultiplyAdd(kgraph, excluded);
            if(!candidate)
                break;
            auto const& [multiplyTag, addTag] = *candidate;
            excluded.emplace(multiplyTag);
            auto multiplyAssign = kgraph.control.get<Assign>(multiplyTag);
            auto addAssign      = kgraph.control.get<Assign>(addTag);
            if(!multiplyAssign || !addAssign || !multiplyAssign->expression
               || !addAssign->expression)
                continue;
            if(!std::holds_alternative<Expression::Multiply>(*multiplyAssign->expression))
                continue;
            if(!std::holds_alternative<Expression::Add>(*addAssign->expression))
                continue;
            auto mulDST = getDEST(kgraph, multiplyTag);
            auto addDST = getDEST(kgraph, addTag);
            if(mulDST == -1 || addDST == -1)
                continue;
            auto const& mulExpr = std::get<Expression::Multiply>(*multiplyAssign->expression);
            auto const& addExpr = std::get<Expression::Add>(*addAssign->expression);
            if(!isSupportedFusedOperand(mulExpr.lhs) || !isSupportedFusedOperand(mulExpr.rhs))
                continue;
            auto match = matchAddConsumer(mulDST, addExpr);
            if(!match)
                continue;
            auto fma = multiplyAdd(mulExpr.lhs, mulExpr.rhs, match->otherExpr);
            // The fused node computes the Add result, so it must inherit Add metadata,
            // not Multiply metadata.
            auto fmaAssign       = *addAssign;
            fmaAssign.expression = fma;
            auto fmaTag          = kgraph.control.addElement(fmaAssign);
            // Reconnect control edges around the fused node.
            reconnect<Graph::Direction::Downstream>(kgraph, -1, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, multiplyTag);
            reconnect<Graph::Direction::Upstream>(kgraph, fmaTag, addTag);
            reconnect<Graph::Direction::Downstream>(kgraph, fmaTag, addTag);
            kgraph.control.deleteElement(multiplyTag);
            kgraph.control.deleteElement(addTag);
            // Rewire coordinate dataflow and remove the dead multiply temporary.
            reflow(kgraph, mulDST, match->otherInputs);
            kgraph.coordinates.deleteElement(mulDST);
            kgraph.mapper.purge(multiplyTag);
            kgraph.mapper.purge(addTag);
            kgraph.mapper.connect(fmaTag, addDST, NaryArgument::DEST);
        }
        return kgraph;
    }
    KernelGraph FuseExpressions::apply(KernelGraph const& original)
    {
        return fuseMultiplyAdd(original);
    }
}
