// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        /**
         * Rewrite HyperGraph to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelGraph CleanArguments::apply(KernelGraph const& k)
        {
            auto kernel    = m_context->kernel();
            auto cleanExpr = [&](ExpressionPtr expr) { return FastArithmetic(m_context)(expr); };

            auto divideBySize = [&](KernelGraph const& graph, int dimTag) {
                auto dim  = graph.coordinates.getNode(dimTag);
                auto size = getSize(dim);
                if(size && not Expression::evaluationTimes(size)[EvaluationTime::Translate])
                {
                    auto resultType = resultVariableType(size);
                    if(resultType == DataType::Int32 || resultType == DataType::Int64
                       || resultType == DataType::UInt32 || resultType == DataType::UInt64)
                        enableDivideBy(size, m_context);
                }
            };

            KernelGraph graph = k;

            // Clean coordinate nodes.
            for(auto tag : k.coordinates.getNodes())
            {
                auto node    = k.coordinates.getNode(tag);
                bool changed = std::visit(
                    [&](auto& dim) -> bool {
                        auto update = [&](ExpressionPtr& field) -> bool {
                            auto newVal = cleanExpr(field);
                            if(not Expression::identical(newVal, field))
                            {
                                field = std::move(newVal);
                                return true;
                            }
                            return false;
                        };
                        bool modified = update(dim.size) | update(dim.stride) | update(dim.offset);

                        if constexpr(std::same_as<User, std::decay_t<decltype(dim)>>)
                        {
                            if(not kernel->hasArgument(dim.argumentName))
                            {
                                auto arg = findArgumentByName(m_command, dim.argumentName);
                                AssertFatal(arg, ShowValue(dim.argumentName));
                                kernel->addCommandArgument(arg);
                            }
                        }
                        return modified;
                    },
                    node);
                if(changed)
                    graph.coordinates.setElement(tag, node);
            }

            // Process coordinate transform edges (Tile/Flatten) to enable fast division.
            for(auto tag : k.coordinates.getEdges())
            {
                auto edge = k.coordinates.getEdge(tag);
                // Edge is variant<CoordinateTransformEdge, DataFlowEdge>; only the
                // CoordinateTransformEdge arm needs processing.
                if(auto* cte = std::get_if<CoordinateTransformEdge>(&edge))
                {
                    auto loc = k.coordinates.getLocation(tag);
                    std::visit(
                        [&](auto& e) {
                            using T = std::decay_t<decltype(e)>;
                            if constexpr(std::same_as<Tile, T>)
                            {
                                for(int i = 1; i < (int)loc.outgoing.size(); i++)
                                    divideBySize(k, loc.outgoing[i]);
                            }
                            else if constexpr(std::same_as<Flatten, T>)
                            {
                                for(int i = 1; i < (int)loc.incoming.size(); i++)
                                    divideBySize(k, loc.incoming[i]);
                            }
                        },
                        *cte);
                }
            }

            // Clean control nodes.
            for(auto tag : k.control.getNodes())
            {
                auto node    = k.control.getNode(tag);
                bool changed = std::visit(
                    [&](auto& op) -> bool {
                        using T = std::decay_t<decltype(op)>;
                        if constexpr(std::same_as<Assign, T>)
                        {
                            auto newExpr = cleanExpr(op.expression);
                            if(not Expression::identical(newExpr, op.expression))
                            {
                                op.expression = std::move(newExpr);
                                return true;
                            }
                        }
                        else if constexpr(
                            std::same_as<
                                ConditionalOp,
                                T> || std::same_as<AssertOp, T> || std::same_as<ForLoopOp, T>)
                        {
                            auto newCond = cleanExpr(op.condition);
                            if(not Expression::identical(newCond, op.condition))
                            {
                                op.condition = std::move(newCond);
                                return true;
                            }
                        }
                        return false;
                    },
                    node);
                if(changed)
                    graph.control.setElement(tag, node);
            }

            return graph;
        }

    }
}
