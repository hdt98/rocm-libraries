// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <bit>
#include <cstdint>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_fwd.hpp>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/Graph/Hypergraph_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlEdge.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/Policy.hpp>
#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller
{
    /**
     * Control flow routines.
     *
     * Control flow is represented as a graph.  Nodes in the control flow graph represent
     * operations (like load/store or a for loop).  Edges in the graph encode dependencies
     * between nodes.
     *
     * The graph answers the question:
     * What are the series of operations needed to solve the problem?
     *
     * Each node of the graph, when traversed, will help generate assembly code.
     * It relies on coordinate transform graph expressions.
     *
     * The control graph should begin with a single Kernel node.
     *
     * There are two main categories of edges: Sequence edges and Body-like edges.
     *
     * From a given node, each kind of Body-like edge denodes a separate body
     * of that node.
     * For example, a Conditional node may have Body edges and Else edges.
     * The nodes downstream of the Body edges represent the true case, and the
     * nodes downstream of the Else edges represent the false case.
     */
    namespace KernelGraph::ControlGraph
    {
        using Bitset = std::vector<uint64_t>;

        inline void bitsetSet(Bitset& bs, int id)
        {
            size_t word = static_cast<size_t>(id) >> 6;
            if(word >= bs.size())
                bs.resize(word + 1, 0);
            bs[word] |= uint64_t(1) << (id & 63);
        }
        inline bool bitsetTest(Bitset const& bs, int id)
        {
            size_t word = static_cast<size_t>(id) >> 6;
            return word < bs.size() && ((bs[word] >> (id & 63)) & 1);
        }
        inline void bitsetOr(Bitset& dst, Bitset const& src)
        {
            if(dst.size() < src.size())
                dst.resize(src.size(), 0);
            for(size_t i = 0; i < src.size(); ++i)
                dst[i] |= src[i];
        }
        inline bool bitsetAny(Bitset const& bs)
        {
            for(auto w : bs)
                if(w)
                    return true;
            return false;
        }
        inline bool bitsetOverlaps(Bitset const& a, Bitset const& b)
        {
            size_t n = std::min(a.size(), b.size());
            for(size_t i = 0; i < n; ++i)
                if(a[i] & b[i])
                    return true;
            return false;
        }
        inline std::vector<int> bitsetToSortedVec(Bitset const& bs)
        {
            std::vector<int> result;
            for(size_t w = 0; w < bs.size(); ++w)
            {
                uint64_t bits = bs[w];
                while(bits)
                {
                    int bit = std::countr_zero(bits);
                    result.push_back(static_cast<int>(w * 64 + bit));
                    bits &= bits - 1;
                }
            }
            return result;
        }

        enum class NodeOrdering
        {
            LeftFirst = 0,
            LeftInBodyOfRight,
            Undefined,
            RightInBodyOfLeft,
            RightFirst,
            Count
        };

        enum class CacheStatus
        {
            Invalid = 0, //< Cache is empty
            Partial, //< Cache does not have all the orders between nodes
            Valid, //< Cache has all orders of nodes
            Count
        };

        /**
         * Return a full representation of 'n'
         */
        std::string   toString(NodeOrdering n);
        std::ostream& operator<<(std::ostream& stream, NodeOrdering n);

        /**
         * Return a full representation of 'c'
         */
        std::string   toString(CacheStatus c);
        std::ostream& operator<<(std::ostream& stream, CacheStatus c);

        /**
         * Return a 3-character representation of 'n'.
         */
        std::string abbrev(NodeOrdering n);

        /**
         * If ordering `order` applies to (a, b), return the ordering that applies to (b, a).
         */
        NodeOrdering opposite(NodeOrdering order);

        struct NodeOrders
        {
            Bitset after;
            Bitset before;
            Bitset inBody;
            Bitset containing;
            void   clear()
            {
                after.clear();
                before.clear();
                inBody.clear();
                containing.clear();
            }
        };

        /**
         * Control flow graph.
         *
         * Nodes in the graph represent operations.  Edges describe
         * dependencies.
         */
        class ControlGraph : public Graph::Hypergraph<Operation, ControlEdge, false>
        {
        public:
            using Base     = Graph::Hypergraph<Operation, ControlEdge, false>;
            ControlGraph() = default;
            template <typename T>
            requires(std::constructible_from<ControlGraph::Element, T>) std::optional<T> get(
                int tag)
            const;
            NodeOrdering compareNodes(UpdateCachePolicy const, int nodeA, int nodeB) const;
            NodeOrdering compareNodes(CacheOnlyPolicy const, int nodeA, int nodeB) const;
            NodeOrdering compareNodes(UseCacheIfAvailablePolicy const, int nodeA, int nodeB) const;
            NodeOrdering compareNodes(IgnoreCachePolicy const, int nodeA, int nodeB) const;
            Generator<int> nodesAfter(int node) const;
            Generator<int> nodesBefore(int node) const;
            Generator<int> nodesInBody(int node) const;
            Generator<int> nodesContaining(int node) const;
            std::string    nodeOrderTableString() const;
            std::string    nodeOrderTableString(std::set<int> const& nodes) const;
            std::unordered_map<int, std::unordered_map<int, NodeOrdering>> nodeOrderTable() const;
            template <typename T>
            requires(std::constructible_from<Operation, T>)
                std::set<std::pair<int, int>> ambiguousNodes()
            const;
            template <CForwardRangeOf<int> Range>
            void orderMemoryNodes(Range const& aControlStack,
                                  Range const& bControlStack,
                                  bool         ordered = true);
            template <typename T>
            inline std::predicate<int> auto isElemType() const
            {
                return [this](int x) -> bool { return get<T>(x).has_value(); };
            }
            template <CControlEdge Edge, std::convertible_to<int>... Nodes>
            void         chain(int a, int b, Nodes... remaining);
            virtual bool isModificationAllowed(int index) const override;
            void         setRestricted()
            {
                m_changesRestricted = true;
            }

        private:
            virtual void clearCache(Graph::GraphModification modification) override;
            void         populateOrderCache() const;
            void         validateOrderCache() const;
            NodeOrdering lookupOrder(CacheOnlyPolicy const, int nodeA, int nodeB) const;
            NodeOrdering lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const;
            /**
                 * Convert graph tag to dense cache index, or -1 if not present.
                 */
            int denseOfTag(int tag) const;
            /**
                 * Select the row-major cache matrix corresponding to a NodeOrdering.
                 * Resolved once per write call so inner loops have no branching.
                 */
            std::vector<uint64_t>& selectCacheMatrix(NodeOrdering order) const;
            // Dense index -> graph node tag.
            mutable std::vector<int> m_cacheDenseToTag;
            // Graph tag -> dense index. Single generic lookup structure.
            mutable std::unordered_map<int, int> m_cacheTagToDense;
            // Number of uint64_t words per row in the dense bit matrices.
            mutable size_t m_cacheWords = 0;
            // Row-major dense bit matrices (row = node_dense, bit = other_dense).
            mutable std::vector<uint64_t> m_cacheAfter;
            mutable std::vector<uint64_t> m_cacheBefore;
            mutable std::vector<uint64_t> m_cacheInBody;
            mutable std::vector<uint64_t> m_cacheContaining;
            mutable CacheStatus           m_cacheStatus       = CacheStatus::Invalid;
            mutable bool                  m_changesRestricted = false;
        };

        std::string name(ControlGraph::Element const& el);

        /**
         * @brief Determine if x holds an Operation of type T.
         */
        template <typename T>
        bool isOperation(auto const& x);

        /**
         * @brief Determine if x holds a ControlEdge of type T.
         */
        template <typename T>
        bool isEdge(auto const& x);
    }
}

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_impl.hpp>
