#include <algorithm>
#include <map>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlEdge.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Cross.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>
#include <set>
#include <unordered_set>
#include <vector>
namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;
    namespace Expr = rocRoller::Expression;
    struct SubExprCollector
    {
        struct Entry
        {
            Expr::ExpressionPtr expr;
            int                 ownerTag; // which Assign control node owns this
        };
        std::vector<Entry> collected;
        void               collect(Expr::ExpressionPtr const& expr, int ownerTag)
        {
            if(!expr)
                return;
            collectImpl(*expr, expr, ownerTag);
        }

    private:
        void collectImpl(Expr::Expression const& node, Expr::ExpressionPtr const& ptr, int ownerTag)
        {
            std::visit(rocRoller::overloaded{[&](Expr::ScaledMatrixMultiply const&) {
                                                 // ScaledMatrixMultiply is too complex and
                                                 // tightly coupled to WaveTile state to factor out
                                             },
                                             [&](Expr::MatrixMultiply const&) {
                                                 // MatrixMultiply is tied to WaveTile state
                                             },
                                             [&](auto const& val) {
                                                 using T = std::decay_t<decltype(val)>;
                                                 if constexpr(Expr::CTernary<T>)
                                                 {
                                                     collected.push_back({ptr, ownerTag});
                                                     if(val.lhs)
                                                         collectImpl(*val.lhs, val.lhs, ownerTag);
                                                     if(val.r1hs)
                                                         collectImpl(*val.r1hs, val.r1hs, ownerTag);
                                                     if(val.r2hs)
                                                         collectImpl(*val.r2hs, val.r2hs, ownerTag);
                                                 }
                                                 else if constexpr(Expr::CBinary<T>)
                                                 {
                                                     collected.push_back({ptr, ownerTag});
                                                     if(val.lhs)
                                                         collectImpl(*val.lhs, val.lhs, ownerTag);
                                                     if(val.rhs)
                                                         collectImpl(*val.rhs, val.rhs, ownerTag);
                                                 }
                                                 else if constexpr(Expr::CUnary<T>)
                                                 {
                                                     collected.push_back({ptr, ownerTag});
                                                     if(val.arg)
                                                         collectImpl(*val.arg, val.arg, ownerTag);
                                                 }
                                                 else if constexpr(Expr::CNary<T>)
                                                 {
                                                     collected.push_back({ptr, ownerTag});
                                                     for(auto const& operand : val.operands)
                                                     {
                                                         if(operand)
                                                             collectImpl(
                                                                 *operand, operand, ownerTag);
                                                     }
                                                 }
                                                 // Leaf types (DataFlowTag, Register::ValuePtr,
                                                 // CommandArgumentValue, etc.) are not collected.
                                             }},
                       node);
        }
    };
    struct ReplaceSubExprWithDFTag
    {
        Expr::ExpressionPtr target;
        Expr::ExpressionPtr replacement; // a DataFlowTag expression
        Expr::ExpressionPtr call(Expr::ExpressionPtr const& expr) const
        {
            if(!expr)
                return nullptr;
            // If this entire sub-expression matches, replace it
            if(Expr::equivalent(expr, target))
                return replacement;
            return std::visit(
                rocRoller::overloaded{
                    [&](Expr::ScaledMatrixMultiply const& val) -> Expr::ExpressionPtr {
                        // Don't recurse into ScaledMatrixMultiply
                        return expr;
                    },
                    [&](auto const& val) -> Expr::ExpressionPtr {
                        using T = std::decay_t<decltype(val)>;
                        if constexpr(Expr::CTernary<T>)
                        {
                            auto newLHS  = call(val.lhs);
                            auto newR1HS = call(val.r1hs);
                            auto newR2HS = call(val.r2hs);
                            if(newLHS.get() != val.lhs.get() || newR1HS.get() != val.r1hs.get()
                               || newR2HS.get() != val.r2hs.get())
                            {
                                T cpy    = val;
                                cpy.lhs  = newLHS;
                                cpy.r1hs = newR1HS;
                                cpy.r2hs = newR2HS;
                                return std::make_shared<Expr::Expression>(cpy);
                            }
                        }
                        else if constexpr(Expr::CBinary<T>)
                        {
                            auto newLHS = call(val.lhs);
                            auto newRHS = call(val.rhs);
                            if(newLHS.get() != val.lhs.get() || newRHS.get() != val.rhs.get())
                            {
                                T cpy   = val;
                                cpy.lhs = newLHS;
                                cpy.rhs = newRHS;
                                return std::make_shared<Expr::Expression>(cpy);
                            }
                        }
                        else if constexpr(Expr::CUnary<T>)
                        {
                            auto newArg = call(val.arg);
                            if(newArg.get() != val.arg.get())
                            {
                                T cpy   = val;
                                cpy.arg = newArg;
                                return std::make_shared<Expr::Expression>(cpy);
                            }
                        }
                        else if constexpr(Expr::CNary<T>)
                        {
                            bool                             changed = false;
                            std::vector<Expr::ExpressionPtr> newOps;
                            newOps.reserve(val.operands.size());
                            for(auto const& op : val.operands)
                            {
                                auto newOp = call(op);
                                if(newOp.get() != op.get())
                                    changed = true;
                                newOps.push_back(newOp);
                            }
                            if(changed)
                            {
                                T cpy        = val;
                                cpy.operands = std::move(newOps);
                                return std::make_shared<Expr::Expression>(std::move(cpy));
                            }
                        }
                        // No change (including leaf types)
                        return expr;
                    }},
                *expr);
        }
    };
    std::vector<std::vector<int>> findSiblingAssignGroups(KernelGraph const& kgraph)
    {
        std::vector<std::vector<int>> groups;
        auto                          allNodes = kgraph.control.getNodes().to<std::vector>();
        for(auto parentTag : allNodes)
        {
            auto bodyChildren
                = kgraph.control.getOutputNodeIndices<Body>(parentTag).to<std::vector>();
            // Among the direct body children, also follow Sequence chains to
            // find all Assign nodes at this scope level.
            std::set<int> scope(bodyChildren.begin(), bodyChildren.end());
            scope = kgraph.control.followEdges<Sequence>(scope);
            std::vector<int> assignsInScope;
            for(auto tag : scope)
            {
                if(kgraph.control.get<Assign>(tag).has_value())
                {
                    assignsInScope.push_back(tag);
                }
            }
            if(assignsInScope.size() >= 2)
            {
                groups.push_back(std::move(assignsInScope));
            }
        }
        return groups;
    }
    bool isWorthFactoring(Expr::ExpressionPtr const& expr)
    {
        if(!expr)
            return false;
        // complexity() returns a heuristic cost; a simple Add is 2,
        // a Multiply is 4. We require >= 2 to skip trivial cases.
        return Expr::complexity(expr) >= 2;
    }
    KernelGraph CrossOperationCSE::apply(KernelGraph const& original)
    {
        auto kgraph = original;
        auto groups = findSiblingAssignGroups(kgraph);
        for(auto const& group : groups)
        {
            // Phase 1: Collect sub-expressions from each Assign in the group
            SubExprCollector                   collector;
            std::map<int, Expr::ExpressionPtr> assignExprs;
            for(auto tag : group)
            {
                auto maybeAssign = kgraph.control.get<Assign>(tag);
                if(!maybeAssign || !maybeAssign->expression)
                    continue;
                assignExprs[tag] = maybeAssign->expression;
                collector.collect(maybeAssign->expression, tag);
            }
            if(assignExprs.size() < 2)
                continue;
            // Phase 2: Group sub-expressions by equivalence and find
            // those shared by >= 2 different Assign operations.
            struct SharedSubExpr
            {
                Expr::ExpressionPtr representative;
                std::set<int>       ownerTags;
            };
            std::vector<SharedSubExpr> sharedSubExprs;
            for(auto const& entry : collector.collected)
            {
                if(!isWorthFactoring(entry.expr))
                    continue;
                bool found = false;
                for(auto& shared : sharedSubExprs)
                {
                    if(Expr::equivalent(entry.expr, shared.representative))
                    {
                        shared.ownerTags.insert(entry.ownerTag);
                        found = true;
                        break;
                    }
                }
                if(!found)
                {
                    sharedSubExprs.push_back({entry.expr, {entry.ownerTag}});
                }
            }
            std::erase_if(sharedSubExprs,
                          [](SharedSubExpr const& s) { return s.ownerTags.size() < 2; });
            if(sharedSubExprs.empty())
                continue;
            std::sort(sharedSubExprs.begin(),
                      sharedSubExprs.end(),
                      [](SharedSubExpr const& a, SharedSubExpr const& b) {
                          return Expr::complexity(a.representative)
                                 > Expr::complexity(b.representative);
                      });
            for(auto const& shared : sharedSubExprs)
            {
                // Re-check: after previous factorings in this group,
                // the sub-expression may no longer exist in enough owners
                int activeOwnerCount = 0;
                for(auto ownerTag : shared.ownerTags)
                {
                    auto it = assignExprs.find(ownerTag);
                    if(it != assignExprs.end()
                       && Expr::containsSubExpression(it->second, shared.representative))
                    {
                        activeOwnerCount++;
                    }
                }
                if(activeOwnerCount < 2)
                    continue;
                auto resType = Expr::resultType(shared.representative);
                // Skip if result type is None (deferred types must not be factored)
                if(resType.varType == DataType::None)
                    continue;
                int    newCoordTag = kgraph.coordinates.addElement(VGPR());
                Assign newAssign;
                newAssign.regType    = resType.regType;
                newAssign.expression = shared.representative;
                newAssign.valueCount = resType.valueCount;
                int newAssignTag     = kgraph.control.addElement(newAssign);
                // Map the new Assign to its VGPR destination coordinate
                kgraph.mapper.connect(newAssignTag, newCoordTag, NaryArgument::DEST);
                // Build the DataFlowTag expression that will replace the
                // shared sub-expression inside each owner Assign
                auto dfTagExpr = Expr::dataFlowTag(newCoordTag, resType.regType, resType.varType);
                // Wire up each owner: replace sub-expression with
                // DataFlowTag ref, add DataFlow edge, add Sequence edge
                for(auto ownerTag : shared.ownerTags)
                {
                    auto it = assignExprs.find(ownerTag);
                    if(it == assignExprs.end())
                        continue;
                    if(!Expr::containsSubExpression(it->second, shared.representative))
                        continue;
                    // Replace the sub-expression with the DataFlowTag ref
                    ReplaceSubExprWithDFTag replacer{shared.representative, dfTagExpr};
                    auto                    newExpr = replacer.call(it->second);
                    // Update the Assign node's expression in the control graph
                    auto updatedAssign       = *kgraph.control.get<Assign>(ownerTag);
                    updatedAssign.expression = newExpr;
                    kgraph.control.setElement(ownerTag, updatedAssign);
                    // Update our local tracking map
                    assignExprs[ownerTag] = newExpr;
                    // Add a DataFlow edge from newCoordTag to the owner's
                    // destination coordinate so the coordinate graph
                    // reflects the data dependency
                    auto ownerCoordTag = kgraph.mapper.get(ownerTag, NaryArgument::DEST);
                    if(ownerCoordTag != -1)
                    {
                        kgraph.coordinates.addElement(DataFlow(), {newCoordTag}, {ownerCoordTag});
                    }
                    kgraph.control.addElement(Sequence(), {newAssignTag}, {ownerTag});
                }
                Log::debug("CrossOperationCSE: factored sub-expr "
                           "(complexity={}) shared by {} ops into "
                           "Assign tag {} -> VGPR coord {}",
                           Expr::complexity(shared.representative),
                           activeOwnerCount,
                           newAssignTag,
                           newCoordTag);
            }
        }
        return kgraph;
    }
}
