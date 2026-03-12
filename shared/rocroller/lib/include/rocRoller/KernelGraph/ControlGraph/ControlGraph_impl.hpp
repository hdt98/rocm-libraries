// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>

namespace rocRoller::KernelGraph::ControlGraph
{
    inline std::string toString(NodeOrdering n)
    {
        switch(n)
        {
        case NodeOrdering::LeftFirst:
            return "LeftFirst";
        case NodeOrdering::LeftInBodyOfRight:
            return "LeftInBodyOfRight";
        case NodeOrdering::Undefined:
            return "Undefined";
        case NodeOrdering::RightInBodyOfLeft:
            return "RightInBodyOfLeft";
        case NodeOrdering::RightFirst:
            return "RightFirst";

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline std::ostream& operator<<(std::ostream& stream, NodeOrdering n)
    {
        return stream << toString(n);
    }

    inline std::string toString(CacheStatus c)
    {
        switch(c)
        {
        case CacheStatus::Invalid:
            return "Invalid";
        case CacheStatus::Partial:
            return "Partial";
        case CacheStatus::Valid:
            return "Valid";
        default:
            break;
        }
        Throw<FatalError>("Invalid CacheStatus");
    }

    inline std::ostream& operator<<(std::ostream& stream, CacheStatus c)
    {
        return stream << toString(c);
    }

    static_assert(CCountedEnum<NodeOrdering>);

    inline std::string abbrev(NodeOrdering n)
    {
        switch(n)
        {
        case NodeOrdering::LeftFirst:
            return " LF";
        case NodeOrdering::LeftInBodyOfRight:
            return "LIB";
        case NodeOrdering::Undefined:
            return "und";
        case NodeOrdering::RightInBodyOfLeft:
            return "RIB";
        case NodeOrdering::RightFirst:
            return " RF";

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline NodeOrdering opposite(NodeOrdering order)
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            return NodeOrdering::RightFirst;
        case NodeOrdering::LeftInBodyOfRight:
            return NodeOrdering::RightInBodyOfLeft;
        case NodeOrdering::Undefined:
            return NodeOrdering::Undefined;
        case NodeOrdering::RightInBodyOfLeft:
            return NodeOrdering::LeftInBodyOfRight;
        case NodeOrdering::RightFirst:
            return NodeOrdering::LeftFirst;

        case NodeOrdering::Count:
        default:
            break;
        }
        Throw<FatalError>("Invalid NodeOrdering");
    }

    inline NodeOrdering ControlGraph::lookupOrder(CacheOnlyPolicy const, int nodeA, int nodeB) const
    {
        auto toDense = [this](int tag) -> int {
            if(tag < 0 || static_cast<size_t>(tag) >= m_cacheTagToDense.size())
                return -1;
            return m_cacheTagToDense[static_cast<size_t>(tag)];
        };
        int const a = toDense(nodeA);
        int const b = toDense(nodeB);
        if(a < 0 || b < 0 || m_cacheWords == 0)
            return NodeOrdering::Undefined;
        auto test = [this, a, b](std::vector<uint64_t> const& mat) -> bool {
            auto const* row = mat.data() + static_cast<size_t>(a) * m_cacheWords;
            return ((row[static_cast<size_t>(b) >> 6] >> (b & 63)) & 1ull) != 0;
        };
        if(test(m_cacheAfter))
            return NodeOrdering::LeftFirst;
        if(test(m_cacheBefore))
            return NodeOrdering::RightFirst;
        if(test(m_cacheInBody))
            return NodeOrdering::RightInBodyOfLeft;
        if(test(m_cacheContaining))
            return NodeOrdering::LeftInBodyOfRight;
        return NodeOrdering::Undefined;
    }

    inline Generator<int> ControlGraph::nodesAfter(int node) const
    {
        populateOrderCache();
        if(node < 0 || static_cast<size_t>(node) >= m_cacheTagToDense.size())
            co_return;
        int const dense = m_cacheTagToDense[static_cast<size_t>(node)];
        if(dense < 0)
            co_return;
        auto const* row = m_cacheAfter.data() + static_cast<size_t>(dense) * m_cacheWords;
        int const   n   = static_cast<int>(m_cacheDenseToTag.size());
        for(size_t w = 0; w < m_cacheWords; ++w)
        {
            uint64_t bits = row[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int d   = static_cast<int>(w * 64 + bit);
                if(d < n)
                    co_yield m_cacheDenseToTag[d];
                bits &= bits - 1;
            }
        }
    }

    inline Generator<int> ControlGraph::nodesBefore(int node) const
    {
        populateOrderCache();
        if(node < 0 || static_cast<size_t>(node) >= m_cacheTagToDense.size())
            co_return;
        int const dense = m_cacheTagToDense[static_cast<size_t>(node)];
        if(dense < 0)
            co_return;
        auto const* row = m_cacheBefore.data() + static_cast<size_t>(dense) * m_cacheWords;
        int const   n   = static_cast<int>(m_cacheDenseToTag.size());
        for(size_t w = 0; w < m_cacheWords; ++w)
        {
            uint64_t bits = row[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int d   = static_cast<int>(w * 64 + bit);
                if(d < n)
                    co_yield m_cacheDenseToTag[d];
                bits &= bits - 1;
            }
        }
    }

    inline Generator<int> ControlGraph::nodesInBody(int node) const
    {
        populateOrderCache();
        if(node < 0 || static_cast<size_t>(node) >= m_cacheTagToDense.size())
            co_return;
        int const dense = m_cacheTagToDense[static_cast<size_t>(node)];
        if(dense < 0)
            co_return;
        auto const* row = m_cacheInBody.data() + static_cast<size_t>(dense) * m_cacheWords;
        int const   n   = static_cast<int>(m_cacheDenseToTag.size());
        for(size_t w = 0; w < m_cacheWords; ++w)
        {
            uint64_t bits = row[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int d   = static_cast<int>(w * 64 + bit);
                if(d < n)
                    co_yield m_cacheDenseToTag[d];
                bits &= bits - 1;
            }
        }
    }

    inline Generator<int> ControlGraph::nodesContaining(int node) const
    {
        populateOrderCache();
        if(node < 0 || static_cast<size_t>(node) >= m_cacheTagToDense.size())
            co_return;
        int const dense = m_cacheTagToDense[static_cast<size_t>(node)];
        if(dense < 0)
            co_return;
        auto const* row = m_cacheContaining.data() + static_cast<size_t>(dense) * m_cacheWords;
        int const   n   = static_cast<int>(m_cacheDenseToTag.size());
        for(size_t w = 0; w < m_cacheWords; ++w)
        {
            uint64_t bits = row[w];
            while(bits)
            {
                int bit = std::countr_zero(bits);
                int d   = static_cast<int>(w * 64 + bit);
                if(d < n)
                    co_yield m_cacheDenseToTag[d];
                bits &= bits - 1;
            }
        }
    }

    template <typename T>
    requires(std::constructible_from<Operation, T>) inline std::set<
        std::pair<int, int>> ControlGraph::ambiguousNodes() const
    {
        std::set<std::pair<int, int>> badNodes;

        auto memNodes = getNodes<T>().template to<std::set>();
        for(auto iter = memNodes.begin(); iter != memNodes.end(); iter++)
        {
            std::set otherNodes(std::next(iter), memNodes.end());
            for(auto node : otherNodes)
            {
                if(compareNodes(rocRoller::UpdateCache, *iter, node) == NodeOrdering::Undefined)
                {
                    badNodes.insert(std::make_pair(*iter, node));
                }
            }
        }
        return badNodes;
    }

    template <typename T>
    requires(std::constructible_from<ControlGraph::Element,
                                     T>) inline std::optional<T> ControlGraph::get(int tag) const
    {
        auto x = getElement(tag);
        if constexpr(CIsAnyOf<T, Operation, ControlEdge>)
        {
            if(std::holds_alternative<T>(x))
            {
                return std::get<T>(x);
            }
        }
        else
        {
            if constexpr(std::constructible_from<ControlEdge, T>)
            {
                if(std::holds_alternative<ControlEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                    {
                        return std::get<T>(std::get<ControlEdge>(x));
                    }
                }
            }
            if constexpr(std::constructible_from<Operation, T>)
            {
                if(std::holds_alternative<Operation>(x))
                {
                    if(std::holds_alternative<T>(std::get<Operation>(x)))
                    {
                        return std::get<T>(std::get<Operation>(x));
                    }
                }
            }
        }
        return {};
    }

    template <CControlEdge Edge, std::convertible_to<int>... Nodes>
    void ControlGraph::chain(int a, int b, Nodes... remaining)
    {
        addElement(Edge(), {a}, {b});

        if constexpr(sizeof...(remaining) > 0)
            chain<Edge>(b, remaining...);
    }

    template <typename T>
    requires(std::constructible_from<Operation, T>) inline bool isOperation(auto const& x)
    {
        if(std::holds_alternative<Operation>(x))
        {
            if(std::holds_alternative<T>(std::get<Operation>(x)))
                return true;
        }

        return false;
    }

    template <typename T>
    requires(std::constructible_from<ControlEdge, T>) inline bool isEdge(auto const& x)
    {
        if(std::holds_alternative<ControlEdge>(x))
        {
            if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                return true;
        }

        return false;
    }

    inline std::string name(ControlGraph::Element const& el)
    {
        return ControlGraph::ElementName(el);
    }

    template <CForwardRangeOf<int> Range>
    inline void ControlGraph::orderMemoryNodes(Range const& aControlStack,
                                               Range const& bControlStack,
                                               bool         ordered)
    {
        int src  = -1;
        int dest = -1;
        for(int i = 0; (i < aControlStack.size()) && (i < bControlStack.size()); ++i)
        {
            if(aControlStack.at(i) != bControlStack.at(i))
            {
                auto setCoordA = get<SetCoordinate>(aControlStack.at(i));
                auto setCoordB = get<SetCoordinate>(bControlStack.at(i));
                if(ordered)
                {
                    src  = aControlStack.at(i);
                    dest = bControlStack.at(i);
                }
                else if(setCoordA && setCoordB
                        && evaluationTimes(setCoordA->value)[Expression::EvaluationTime::Translate]
                        && evaluationTimes(setCoordB->value)[Expression::EvaluationTime::Translate])
                {
                    src  = aControlStack.at(i);
                    dest = bControlStack.at(i);
                    if(getUnsignedInt(evaluate(setCoordB->value))
                       < getUnsignedInt(evaluate(setCoordA->value)))
                    {
                        src  = bControlStack.at(i);
                        dest = aControlStack.at(i);
                    }
                }
                else
                {
                    src  = std::min(aControlStack.at(i), bControlStack.at(i));
                    dest = std::max(aControlStack.at(i), bControlStack.at(i));
                }
                break;
            }
        }
        if(src == -1 || dest == -1)
        {
            int aIndex = aControlStack.size() - 1;
            int bIndex = bControlStack.size() - 1;
            if(aControlStack.size() > bControlStack.size())
            {
                aIndex = bIndex;
            }
            else
            {
                bIndex = aIndex;
            }
            src  = std::min(aControlStack.at(aIndex), bControlStack.at(bIndex));
            dest = std::max(aControlStack.at(aIndex), bControlStack.at(bIndex));
        }

        addElement(Sequence(), {src}, {dest});
    }
}
