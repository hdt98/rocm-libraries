// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>
#include <unordered_set>

namespace rocRoller::KernelGraph::ControlGraph
{

    namespace
    {
        inline void rowOr(uint64_t* dst, uint64_t const* src, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                dst[i] |= src[i];
        }
        inline void rowAndNot(uint64_t* dst, uint64_t const* mask, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                dst[i] &= ~mask[i];
        }
        inline bool rowAny(uint64_t const* row, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if(row[i] != 0)
                    return true;
            return false;
        }
        inline bool rowOverlap(uint64_t const* a, uint64_t const* b, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if((a[i] & b[i]) != 0)
                    return true;
            return false;
        }
        inline void collectNonZeroWordIndices(Bitset const& bs, std::vector<int>& out)
        {
            out.clear();
            for(size_t w = 0; w < bs.size(); ++w)
            {
                if(bs[w] != 0)
                    out.push_back(static_cast<int>(w));
            }
        }
        inline void rowOrWords(uint64_t* dst, Bitset const& src, std::vector<int> const& words)
        {
            for(int w : words)
                dst[w] |= src[static_cast<size_t>(w)];
        }
        template <typename Func>
        inline void forEachSetBit(Bitset const& bits, int bitLimit, Func&& f)
        {
            for(size_t w = 0; w < bits.size(); ++w)
            {
                uint64_t word = bits[w];
                while(word)
                {
                    int bit = std::countr_zero(word);
                    int id  = static_cast<int>(w * 64 + bit);
                    if(id < bitLimit)
                        f(id);
                    word &= word - 1;
                }
            }
        }
        template <typename Func>
        inline void forEachSetBitRow(uint64_t const* row, size_t words, int bitLimit, Func&& f)
        {
            for(size_t w = 0; w < words; ++w)
            {
                uint64_t word = row[w];
                while(word)
                {
                    int bit = std::countr_zero(word);
                    int id  = static_cast<int>(w * 64 + bit);
                    if(id < bitLimit)
                        f(id);
                    word &= word - 1;
                }
            }
        }
    } // namespace

    size_t ControlGraph::wordsForBits(size_t bitCount)
    {
        return (bitCount + 63) >> 6;
    }
    int ControlGraph::denseOfTag(int tag) const
    {
        auto it = m_cacheTagToDense.find(tag);
        return it == m_cacheTagToDense.end() ? -1 : it->second;
    }
    uint64_t* ControlGraph::rowPtr(std::vector<uint64_t>& matrix, int denseRow) const
    {
        return matrix.data() + static_cast<size_t>(denseRow) * m_cacheWordsPerRow;
    }
    uint64_t const* ControlGraph::rowPtr(std::vector<uint64_t> const& matrix, int denseRow) const
    {
        return matrix.data() + static_cast<size_t>(denseRow) * m_cacheWordsPerRow;
    }
    bool
        ControlGraph::bitTest(std::vector<uint64_t> const& matrix, int denseRow, int denseCol) const
    {
        auto const* row = rowPtr(matrix, denseRow);
        size_t      w   = static_cast<size_t>(denseCol) >> 6;
        return ((row[w] >> (denseCol & 63)) & 1ULL) != 0;
    }
    void ControlGraph::ensureTransposeCache() const
    {
        if(m_transposeValid)
            return;
        size_t const nodeCount   = m_cacheDenseToTag.size();
        size_t const matrixWords = nodeCount * m_cacheWordsPerRow;
        m_cacheAfterT.assign(matrixWords, 0);
        m_cacheInBodyT.assign(matrixWords, 0);
        auto setBitInRow = [&](std::vector<uint64_t>& matrix, int row, int col) {
            auto* dst = rowPtr(matrix, row);
            dst[static_cast<size_t>(col) >> 6] |= uint64_t(1) << (col & 63);
        };
        for(int i = 0; i < static_cast<int>(nodeCount); ++i)
        {
            auto const* afterRow  = rowPtr(m_cacheAfter, i);
            auto const* inBodyRow = rowPtr(m_cacheInBody, i);
            forEachSetBitRow(afterRow, m_cacheWordsPerRow, static_cast<int>(nodeCount), [&](int j) {
                setBitInRow(m_cacheAfterT, j, i);
            });
            forEachSetBitRow(inBodyRow,
                             m_cacheWordsPerRow,
                             static_cast<int>(nodeCount),
                             [&](int j) { setBitInRow(m_cacheInBodyT, j, i); });
        }
        m_transposeValid = true;
    }

    //=========================================================================
    // isModificationAllowed (unchanged)
    //=========================================================================
    bool ControlGraph::isModificationAllowed(int index) const
    {
        if(not Settings::getInstance()->get(Settings::EnforceGraphConstraints))
            return true;
        if(not m_changesRestricted)
            return true;
        auto const& el = getElement(index);
        if(std::holds_alternative<Operation>(el))
        {
            return std::visit(
                [](auto&& arg) {
                    using OpType = std::decay_t<decltype(arg)>;
                    return !(
                        std::is_same_v<OpType, ForLoopOp> or std::is_same_v<OpType, SetCoordinate>);
                },
                std::get<Operation>(el));
        }
        else
        {
            return true;
        }
    }

    std::unordered_map<int, std::unordered_map<int, NodeOrdering>>
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        std::vector<int>                                               tags = m_cacheDenseToTag;
        std::sort(tags.begin(), tags.end());
        for(size_t i = 0; i < tags.size(); ++i)
        {
            for(size_t j = i + 1; j < tags.size(); ++j)
            {
                auto ord = lookupOrder(CacheOnly, tags[i], tags[j]);
                if(ord != NodeOrdering::Undefined)
                    table[tags[i]][tags[j]] = ord;
            }
        }
        return table;
    }
    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();
        if(nodes.empty())
            return "Empty order cache.\n";
        std::ostringstream msg;
        int                maxNode = *nodes.rbegin();
        int                width   = maxNode > 0
                                         ? static_cast<int>(std::ceil(std::log10(static_cast<double>(maxNode + 1))))
                                         : 1;
        width                      = std::max(width, 3);
        msg << std::setw(width) << " "
            << "\\";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;
        for(int i : nodes)
        {
            msg << std::endl << std::setw(width) << i << "|";
            for(int j : nodes)
            {
                if(i == j)
                {
                    msg << " ";
                    auto oldFill = msg.fill('-');
                    msg << std::setw(width) << "-";
                    msg.fill(oldFill);
                }
                else
                {
                    msg << " " << std::setw(width) << abbrev(lookupOrder(CacheOnly, i, j));
                }
            }
            msg << " | " << std::setw(width) << i;
        }
        msg << std::endl
            << std::setw(width) << " "
            << "|";
        for(int n : nodes)
            msg << " " << std::setw(width) << n;
        msg << std::endl;
        return msg.str();
    }
    std::string ControlGraph::nodeOrderTableString() const
    {
        populateOrderCache();
        TIMER(t, "nodeOrderTable");
        std::set<int> nodes(m_cacheDenseToTag.begin(), m_cacheDenseToTag.end());
        return nodeOrderTableString(nodes);
    }
    void ControlGraph::clearCache(Graph::GraphModification modification)
    {
        Hypergraph<Operation, ControlEdge, false>::clearCache(modification);
        if(modification == Graph::GraphModification::AddElement
           && m_cacheStatus != CacheStatus::Invalid)
        {
            m_cacheStatus = CacheStatus::Partial;
        }
        else
        {
            m_cacheStatus = CacheStatus::Invalid;
        }
        m_cacheAfter.clear();
        m_cacheInBody.clear();
        m_cacheAfterT.clear();
        m_cacheInBodyT.clear();
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWordsPerRow = 0;
        m_transposeValid   = false;
    }
    void ControlGraph::validateOrderCache() const
    {
        int const nodeCount = static_cast<int>(m_cacheDenseToTag.size());
        for(int i = 0; i < nodeCount; ++i)
        {
            auto const* afterRow  = rowPtr(m_cacheAfter, i);
            auto const* inBodyRow = rowPtr(m_cacheInBody, i);
            // Same-direction conflict: cannot be both "after" and "inBody" simultaneously.
            AssertFatal(!rowOverlap(afterRow, inBodyRow, m_cacheWordsPerRow),
                        "Node has conflicting orders (after & inBody)");
            // Self relation is invalid.
            size_t   diagWord = static_cast<size_t>(i) >> 6;
            uint64_t diagMask = uint64_t(1) << (i & 63);
            AssertFatal((afterRow[diagWord] & diagMask) == 0, "Self relation in after");
            AssertFatal((inBodyRow[diagWord] & diagMask) == 0, "Self relation in inBody");
            // Cross-direction conflicts:
            // after(i,j) conflicts with after(j,i) and inBody(j,i)
            forEachSetBitRow(afterRow, m_cacheWordsPerRow, nodeCount, [&](int j) {
                AssertFatal(!bitTest(m_cacheAfter, j, i),
                            "Conflicting orders: after and before simultaneously");
                AssertFatal(!bitTest(m_cacheInBody, j, i),
                            "Conflicting orders: after and containing simultaneously");
            });
            // inBody(i,j) conflicts with inBody(j,i) and after(j,i)
            forEachSetBitRow(inBodyRow, m_cacheWordsPerRow, nodeCount, [&](int j) {
                AssertFatal(!bitTest(m_cacheInBody, j, i),
                            "Conflicting orders: inBody and containing simultaneously");
                AssertFatal(!bitTest(m_cacheAfter, j, i),
                            "Conflicting orders: inBody and before simultaneously");
            });
        }
    }
    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");
        if(m_cacheStatus == CacheStatus::Valid)
            return;
        m_cacheAfter.clear();
        m_cacheInBody.clear();
        m_cacheAfterT.clear();
        m_cacheInBodyT.clear();
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWordsPerRow = 0;
        m_transposeValid   = false;
        // Dense id assignment for all control-graph nodes.
        // This decouples storage width from sparse graph tags.
        m_cacheDenseToTag = getNodes<Operation>().template to<std::vector>();
        if(m_cacheDenseToTag.empty())
        {
            m_cacheStatus = CacheStatus::Valid;
            return;
        }
        m_cacheTagToDense.reserve(m_cacheDenseToTag.size());
        for(size_t i = 0; i < m_cacheDenseToTag.size(); ++i)
            m_cacheTagToDense.emplace(m_cacheDenseToTag[i], static_cast<int>(i));
        size_t const nodeCount   = m_cacheDenseToTag.size();
        m_cacheWordsPerRow       = wordsForBits(nodeCount);
        size_t const matrixWords = nodeCount * m_cacheWordsPerRow;
        m_cacheAfter.assign(matrixWords, 0);
        m_cacheInBody.assign(matrixWords, 0);
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Expected edge types: Sequence(0), Initialize(1), "
                      "ForLoopIncrement(2), Body(3), Else(4).");
        constexpr int kSequence = 0;
        constexpr int kInit     = 1;
        constexpr int kInc      = 2;
        constexpr int kBody     = 3;
        constexpr int kElse     = 4;
        using ChildBuckets      = std::array<std::vector<int>, std::variant_size_v<ControlEdge>>;
        std::vector<ChildBuckets> childrenByType(nodeCount);
        std::vector<int>          indegree(nodeCount, 0);
        using GD = Graph::Direction;
        // One-time typed adjacency build in dense id space.
        for(size_t parentDense = 0; parentDense < nodeCount; ++parentDense)
        {
            int parentTag = m_cacheDenseToTag[parentDense];
            for(int edgeTag : getNeighbours<GD::Downstream>(parentTag))
            {
                int   edgeType = static_cast<int>(getEdge(edgeTag).index());
                auto& bucket   = childrenByType[parentDense][edgeType];
                for(int childTag : getNeighbours<GD::Downstream>(edgeTag))
                {
                    auto it = m_cacheTagToDense.find(childTag);
                    if(it == m_cacheTagToDense.end())
                        continue;
                    bucket.push_back(it->second);
                }
            }
            // Deduplicate to avoid redundant downstream work.
            for(auto& bucket : childrenByType[parentDense])
            {
                std::sort(bucket.begin(), bucket.end());
                bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
                for(int childDense : bucket)
                    ++indegree[childDense];
            }
        }
        // Kahn topo order.
        std::deque<int> ready;
        for(size_t i = 0; i < nodeCount; ++i)
        {
            if(indegree[i] == 0)
                ready.push_back(static_cast<int>(i));
        }
        std::vector<int> topo;
        topo.reserve(nodeCount);
        while(!ready.empty())
        {
            int node = ready.front();
            ready.pop_front();
            topo.push_back(node);
            for(auto const& bucket : childrenByType[node])
            {
                for(int child : bucket)
                {
                    if(--indegree[child] == 0)
                        ready.push_back(child);
                }
            }
        }
        AssertFatal(topo.size() == nodeCount,
                    "Cycle detected in control graph; populateOrderCache requires DAG.");
        // allDescendants[row] memoizes full descendant set of row node.
        std::vector<uint64_t> allDescendants(matrixWords, 0);
        // Reused temporaries.
        Bitset           initNodes(m_cacheWordsPerRow, 0);
        Bitset           bodyNodes(m_cacheWordsPerRow, 0);
        Bitset           elseNodes(m_cacheWordsPerRow, 0);
        Bitset           incNodes(m_cacheWordsPerRow, 0);
        Bitset           seqNodes(m_cacheWordsPerRow, 0);
        Bitset           tmpNodes(m_cacheWordsPerRow, 0);
        Bitset           laterNodes(m_cacheWordsPerRow, 0);
        Bitset           allNodes(m_cacheWordsPerRow, 0);
        std::vector<int> nonZeroWords;
        auto             dSet = [](Bitset& bs, int denseNode) {
            bs[static_cast<size_t>(denseNode) >> 6] |= uint64_t(1) << (denseNode & 63);
        };
        auto gatherGroup = [&](int nodeDense, int edgeType, Bitset& out) {
            std::fill(out.begin(), out.end(), 0);
            for(int childDense : childrenByType[nodeDense][edgeType])
            {
                dSet(out, childDense);
                rowOr(out.data(), rowPtr(allDescendants, childDense), m_cacheWordsPerRow);
            }
        };
        // Forward-only write: no opposite writes in hot path.
        auto writeOneToSetForward = [&](int nodeA, Bitset const& nodesB, bool toAfterMatrix) {
            if(!bitsetAny(nodesB))
                return;
            collectNonZeroWordIndices(nodesB, nonZeroWords);
            auto* row = toAfterMatrix ? rowPtr(m_cacheAfter, nodeA) : rowPtr(m_cacheInBody, nodeA);
            rowOrWords(row, nodesB, nonZeroWords);
        };
        auto writeSetToSetForward
            = [&](Bitset const& nodesA, Bitset const& nodesB, bool toAfterMatrix) {
                  if(!bitsetAny(nodesA) || !bitsetAny(nodesB))
                      return;
                  collectNonZeroWordIndices(nodesB, nonZeroWords);
                  forEachSetBit(nodesA, static_cast<int>(nodeCount), [&](int nodeA) {
                      auto* row = toAfterMatrix ? rowPtr(m_cacheAfter, nodeA)
                                                : rowPtr(m_cacheInBody, nodeA);
                      rowOrWords(row, nodesB, nonZeroWords);
                  });
              };
        // Bottom-up DP in reverse topo order.
        for(auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            int node = *it;
            gatherGroup(node, kInit, initNodes);
            gatherGroup(node, kBody, bodyNodes);
            gatherGroup(node, kElse, elseNodes);
            gatherGroup(node, kInc, incNodes);
            gatherGroup(node, kSequence, seqNodes);
            // Make partitions disjoint in precedence order:
            // Init -> Body -> Else -> Inc -> Sequence
            rowAndNot(bodyNodes.data(), initNodes.data(), m_cacheWordsPerRow);
            std::copy(initNodes.begin(), initNodes.end(), tmpNodes.begin());
            rowOr(tmpNodes.data(), bodyNodes.data(), m_cacheWordsPerRow);
            rowAndNot(elseNodes.data(), tmpNodes.data(), m_cacheWordsPerRow);
            rowOr(tmpNodes.data(), elseNodes.data(), m_cacheWordsPerRow);
            rowAndNot(incNodes.data(), tmpNodes.data(), m_cacheWordsPerRow);
            rowOr(tmpNodes.data(), incNodes.data(), m_cacheWordsPerRow);
            rowAndNot(seqNodes.data(), tmpNodes.data(), m_cacheWordsPerRow);
            // Remove self from all partitions defensively.
            auto clearSelf = [&](Bitset& bs) {
                bs[static_cast<size_t>(node) >> 6] &= ~(uint64_t(1) << (node & 63));
            };
            clearSelf(initNodes);
            clearSelf(bodyNodes);
            clearSelf(elseNodes);
            clearSelf(incNodes);
            clearSelf(seqNodes);
            // Parent/body relations:
            writeOneToSetForward(node, initNodes, false);
            writeOneToSetForward(node, bodyNodes, false);
            writeOneToSetForward(node, elseNodes, false);
            writeOneToSetForward(node, incNodes, false);
            // Parent/sequence relation:
            writeOneToSetForward(node, seqNodes, true);
            // Cross partition precedence (cumulative form reduces calls):
            // init < body|else|inc|seq
            std::copy(bodyNodes.begin(), bodyNodes.end(), laterNodes.begin());
            rowOr(laterNodes.data(), elseNodes.data(), m_cacheWordsPerRow);
            rowOr(laterNodes.data(), incNodes.data(), m_cacheWordsPerRow);
            rowOr(laterNodes.data(), seqNodes.data(), m_cacheWordsPerRow);
            writeSetToSetForward(initNodes, laterNodes, true);
            // body < else|inc|seq
            std::copy(elseNodes.begin(), elseNodes.end(), laterNodes.begin());
            rowOr(laterNodes.data(), incNodes.data(), m_cacheWordsPerRow);
            rowOr(laterNodes.data(), seqNodes.data(), m_cacheWordsPerRow);
            writeSetToSetForward(bodyNodes, laterNodes, true);
            // else < inc|seq
            std::copy(incNodes.begin(), incNodes.end(), laterNodes.begin());
            rowOr(laterNodes.data(), seqNodes.data(), m_cacheWordsPerRow);
            writeSetToSetForward(elseNodes, laterNodes, true);
            // inc < seq
            writeSetToSetForward(incNodes, seqNodes, true);
            // Memo full descendants for parent use.
            std::fill(allNodes.begin(), allNodes.end(), 0);
            rowOr(allNodes.data(), initNodes.data(), m_cacheWordsPerRow);
            rowOr(allNodes.data(), bodyNodes.data(), m_cacheWordsPerRow);
            rowOr(allNodes.data(), elseNodes.data(), m_cacheWordsPerRow);
            rowOr(allNodes.data(), incNodes.data(), m_cacheWordsPerRow);
            rowOr(allNodes.data(), seqNodes.data(), m_cacheWordsPerRow);
            std::copy(allNodes.begin(), allNodes.end(), rowPtr(allDescendants, node));
        }
        // Remove diagonal bits.
        for(int i = 0; i < static_cast<int>(nodeCount); ++i)
        {
            size_t   w    = static_cast<size_t>(i) >> 6;
            uint64_t mask = ~(uint64_t(1) << (i & 63));
            rowPtr(m_cacheAfter, i)[w] &= mask;
            rowPtr(m_cacheInBody, i)[w] &= mask;
        }
        validateOrderCache();
        m_cacheStatus = CacheStatus::Valid;
    }

    //=========================================================================
    // lookupOrder(IgnoreCache) -- unchanged algorithm, no cache involved
    //=========================================================================
    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");
        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;
        auto const              getOrderIfParent = [&](int edge) {
            return std::holds_alternative<Sequence>(getEdge(edge))
                                    ? NodeOrdering::LeftFirst
                                    : NodeOrdering::RightInBodyOfLeft;
        };
        auto const getOrderOfDescendants = [&](int edgeA, int edgeB) {
            auto edgeAElem = getEdge(edgeA);
            auto edgeBElem = getEdge(edgeB);
            AssertFatal(edgeAElem.index() != edgeBElem.index(),
                        "edgeA and edgeB types should not be the same");
            return std::visit(
                rocRoller::overloaded{
                    [&](auto const&) {
                        AssertFatal(false, "Unhandled edge type in getOrderOfDescendants");
                        return NodeOrdering::Undefined;
                    },
                    [&](Body const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    },
                    [&](Sequence const&) { return opposite(NodeOrdering::LeftFirst); },
                    [&](Initialize const&) { return NodeOrdering::LeftFirst; },
                    [&](ForLoopIncrement const&) {
                        return std::holds_alternative<Sequence>(edgeBElem)
                                   ? NodeOrdering::LeftFirst
                                   : opposite(NodeOrdering::LeftFirst);
                    },
                    [&](Else const&) {
                        return std::holds_alternative<Initialize>(edgeBElem)
                                       || std::holds_alternative<Body>(edgeBElem)
                                   ? opposite(NodeOrdering::LeftFirst)
                                   : NodeOrdering::LeftFirst;
                    }},
                edgeAElem);
        };
        std::unordered_map<int, int> A_ancestors;
        std::vector<int>             stk{nodeA};
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();
            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeB)
                        return opposite(getOrderIfParent(edge));
                    A_ancestors.insert({parent, edge});
                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }
        AssertFatal(stk.empty());
        stk.push_back(nodeB);
        visited_nodes.clear();
        while(!stk.empty())
        {
            auto node = stk.back();
            stk.pop_back();
            for(auto edge : getNeighbours<GD::Upstream>(node))
            {
                for(auto parent : getNeighbours<GD::Upstream>(edge))
                {
                    if(parent == nodeA)
                        return getOrderIfParent(edge);
                    if(A_ancestors.contains(parent))
                    {
                        if(getEdge(A_ancestors.at(parent)).index() != getEdge(edge).index())
                        {
                            return getOrderOfDescendants(A_ancestors.at(parent), edge);
                        }
                    }
                    if(!visited_nodes.contains(parent))
                    {
                        visited_nodes.insert(parent);
                        stk.push_back(parent);
                    }
                }
            }
        }
        return NodeOrdering::Undefined;
    }
    //=========================================================================
    // compareNodes variants
    //=========================================================================
    static void validateNodes(ControlGraph const& control, int nodeA, int nodeB)
    {
        AssertFatal(nodeA != nodeB, ShowValue(nodeA));
        AssertFatal(control.getElementType(nodeA) == Graph::ElementType::Node
                        && control.getElementType(nodeB) == Graph::ElementType::Node,
                    ShowValue(control.getElementType(nodeA)),
                    ShowValue(control.getElementType(nodeB)));
    }
    NodeOrdering ControlGraph::compareNodes(CacheOnlyPolicy const, int nodeA, int nodeB) const
    {
        AssertFatal(m_cacheStatus == CacheStatus::Valid);
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(CacheOnly, nodeA, nodeB);
    }
    NodeOrdering ControlGraph::compareNodes(UpdateCachePolicy const, int nodeA, int nodeB) const
    {
        if(m_cacheStatus != CacheStatus::Valid)
            populateOrderCache();
        return compareNodes(CacheOnly, nodeA, nodeB);
    }
    NodeOrdering
        ControlGraph::compareNodes(UseCacheIfAvailablePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return m_cacheStatus == CacheStatus::Valid ? compareNodes(CacheOnly, nodeA, nodeB)
                                                   : compareNodes(IgnoreCache, nodeA, nodeB);
    }
    NodeOrdering ControlGraph::compareNodes(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        validateNodes(*this, nodeA, nodeB);
        return lookupOrder(IgnoreCache, nodeA, nodeB);
    }
}
