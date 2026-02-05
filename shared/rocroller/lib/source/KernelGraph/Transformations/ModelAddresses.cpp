/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ModelAddresses.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

namespace rocRoller::KernelGraph
{
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    KernelGraph ModelAddresses::apply(KernelGraph const& original)
    {
        KernelGraph graph = original;

        auto root = graph.control.roots().only();
        AssertFatal(root.has_value());

        auto allNodes
            = graph.control.depthFirstVisit(*root).filter(graph.control.isElemType<Operation>());

        for(const auto node : allNodes)
        {
            auto visitor = rocRoller::overloaded{
                [&](LoadLDSTile op) {
                    {
                        const auto addresses
                            = getLDSAddresses(graph, node, op.varType).to<std::vector>();

                        std::vector<size_t> normalizedAddresses;
                        auto minAddress = *std::min_element(addresses.begin(), addresses.end());
                        for(auto addr : addresses)
                        {
                            normalizedAddresses.push_back(addr - minAddress);
                        }

                        graph.modelledAddresses[node] = std::move(normalizedAddresses);
                    }
                },
                [&](auto op) {}};

            std::visit(visitor, graph.control.getNode(node));
        }
        return graph;
    }

    std::string ModelAddresses::name() const
    {
        return "ModelAddresses";
    }
}
