// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <algorithm>

#include <bitset>
#include <cmath>
#include <iomanip>

namespace rocRoller::KernelGraph::ControlGraph
{
    namespace
    {
        inline size_t wordsForBits(size_t bits)
        {
            return (bits + 63) >> 6;
        }
        inline uint64_t* matrixRow(std::vector<uint64_t>& mat, size_t words, int row)
        {
            return mat.data() + static_cast<size_t>(row) * words;
        }
        inline uint64_t const* matrixRow(std::vector<uint64_t> const& mat, size_t words, int row)
        {
            return mat.data() + static_cast<size_t>(row) * words;
        }
        inline void rowOr(uint64_t* dst, uint64_t const* src, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                dst[i] |= src[i];
        }
        inline bool rowAny(uint64_t const* row, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if(row[i])
                    return true;
            return false;
        }
        inline bool rowOverlap(uint64_t const* a, uint64_t const* b, size_t words)
        {
            for(size_t i = 0; i < words; ++i)
                if(a[i] & b[i])
                    return true;
            return false;
        }
        template <typename F>
        inline void forEachSetBit(uint64_t const* row, size_t words, F&& fn)
        {
            for(size_t w = 0; w < words; ++w)
            {
                uint64_t bits = row[w];
                while(bits)
                {
                    int bit = std::countr_zero(bits);
                    fn(static_cast<int>(w * 64 + bit));
                    bits &= bits - 1;
                }
            }
        }
        template <typename F>
        inline void forEachSetBit(Bitset const& bs, F&& fn)
        {
            for(size_t w = 0; w < bs.size(); ++w)
            {
                uint64_t bits = bs[w];
                while(bits)
                {
                    int bit = std::countr_zero(bits);
                    fn(static_cast<int>(w * 64 + bit));
                    bits &= bits - 1;
                }
            }
        }
        inline void denseSet(Bitset& bs, int dense)
        {
            bs[static_cast<size_t>(dense) >> 6] |= uint64_t(1) << (dense & 63);
        }
        inline void denseClear(Bitset& bs, int dense)
        {
            bs[static_cast<size_t>(dense) >> 6] &= ~(uint64_t(1) << (dense & 63));
        }
        inline void denseOr(Bitset& dst, Bitset const& src)
        {
            for(size_t i = 0; i < dst.size(); ++i)
                dst[i] |= src[i];
        }
        inline void denseAndNot(Bitset& dst, Bitset const& mask)
        {
            for(size_t i = 0; i < dst.size(); ++i)
                dst[i] &= ~mask[i];
        }
    }

    std::unordered_map<int, std::unordered_map<int, NodeOrdering>>
        ControlGraph::nodeOrderTable() const
    {
        populateOrderCache();
        std::unordered_map<int, std::unordered_map<int, NodeOrdering>> table;
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        for(int aDense = 0; aDense < n; ++aDense)
        {
            int const   aTag      = m_cacheDenseToTag[aDense];
            auto const* afterRow  = matrixRow(m_cacheAfter, m_cacheWords, aDense);
            auto const* inBodyRow = matrixRow(m_cacheInBody, m_cacheWords, aDense);
            forEachSetBit(afterRow, m_cacheWords, [&](int bDense) {
                if(bDense >= n)
                    return;
                int const bTag = m_cacheDenseToTag[bDense];
                if(aTag < bTag)
                    table[aTag][bTag] = NodeOrdering::LeftFirst;
            });
            forEachSetBit(inBodyRow, m_cacheWords, [&](int bDense) {
                if(bDense >= n)
                    return;
                int const bTag = m_cacheDenseToTag[bDense];
                if(aTag < bTag)
                    table[aTag][bTag] = NodeOrdering::RightInBodyOfLeft;
                else if(bTag < aTag)
                    table[bTag][aTag] = NodeOrdering::LeftInBodyOfRight;
            });
        }
        return table;
    }

    std::string ControlGraph::nodeOrderTableString(std::set<int> const& nodes) const
    {
        populateOrderCache();
        if(nodes.empty())
        {
            return "Empty order cache.\n";
        }

        std::ostringstream msg;

        int width = std::ceil(std::log10(static_cast<float>(*nodes.rbegin())));
        width     = std::max(width, 3);

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
        std::set<int> nodes;
        int const     n = static_cast<int>(m_cacheDenseToTag.size());
        for(int d = 0; d < n; ++d)
        {
            int const   tag           = m_cacheDenseToTag[d];
            auto const* afterRow      = matrixRow(m_cacheAfter, m_cacheWords, d);
            auto const* beforeRow     = matrixRow(m_cacheBefore, m_cacheWords, d);
            auto const* inBodyRow     = matrixRow(m_cacheInBody, m_cacheWords, d);
            auto const* containingRow = matrixRow(m_cacheContaining, m_cacheWords, d);
            if(rowAny(afterRow, m_cacheWords) || rowAny(beforeRow, m_cacheWords)
               || rowAny(inBodyRow, m_cacheWords) || rowAny(containingRow, m_cacheWords))
            {
                nodes.insert(tag);
            }
            auto addRowNodes = [&](uint64_t const* row) {
                forEachSetBit(row, m_cacheWords, [&](int otherDense) {
                    if(otherDense < n)
                        nodes.insert(m_cacheDenseToTag[otherDense]);
                });
            };
            addRowNodes(afterRow);
            addRowNodes(beforeRow);
            addRowNodes(inBodyRow);
            addRowNodes(containingRow);
        }
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
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWords = 0;
        m_cacheAfter.clear();
        m_cacheBefore.clear();
        m_cacheInBody.clear();
        m_cacheContaining.clear();
    }

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
            //
            // Theoretically, add/delete Body edge should be disallowed. But sometimes
            // delete Body edges is OK (e.g., Simplify), and currently there is no way
            // to know if this is called in a valid or invalid use case.
            //
            return true;
        }
    }

    void ControlGraph::validateOrderCache() const
    {
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        for(int r = 0; r < n; ++r)
        {
            auto const* afterRow      = matrixRow(m_cacheAfter, m_cacheWords, r);
            auto const* beforeRow     = matrixRow(m_cacheBefore, m_cacheWords, r);
            auto const* inBodyRow     = matrixRow(m_cacheInBody, m_cacheWords, r);
            auto const* containingRow = matrixRow(m_cacheContaining, m_cacheWords, r);
            AssertFatal(!rowOverlap(afterRow, beforeRow, m_cacheWords),
                        "Node has conflicting orders (after & before)");
            AssertFatal(!rowOverlap(afterRow, inBodyRow, m_cacheWords),
                        "Node has conflicting orders (after & inBody)");
            AssertFatal(!rowOverlap(afterRow, containingRow, m_cacheWords),
                        "Node has conflicting orders (after & containing)");
            AssertFatal(!rowOverlap(beforeRow, inBodyRow, m_cacheWords),
                        "Node has conflicting orders (before & inBody)");
            AssertFatal(!rowOverlap(beforeRow, containingRow, m_cacheWords),
                        "Node has conflicting orders (before & containing)");
            AssertFatal(!rowOverlap(inBodyRow, containingRow, m_cacheWords),
                        "Node has conflicting orders (inBody & containing)");
        }
    }

    using NodeOrderField = Bitset NodeOrders::*;
    // Maps ordering enum to the exact bitset field in NodeOrders.
    // This lets writeOrderCache compute the field once per call,
    // not once per element in hot loops.
    inline NodeOrderField orderField(NodeOrdering order)
    {
        switch(order)
        {
        case NodeOrdering::LeftFirst:
            return &NodeOrders::after;
        case NodeOrdering::RightFirst:
            return &NodeOrders::before;
        case NodeOrdering::RightInBodyOfLeft:
            return &NodeOrders::inBody;
        case NodeOrdering::LeftInBodyOfRight:
            return &NodeOrders::containing;
        default:
            break;
        }
        AssertFatal(false, "Invalid NodeOrdering", ShowValue(order));
        return &NodeOrders::after;
    }
    inline void clearBit(Bitset& bs, int id)
    {
        size_t const word = static_cast<size_t>(id) >> 6;
        if(word < bs.size())
            bs[word] &= ~(uint64_t(1) << (id & 63));
    }

    void ControlGraph::populateOrderCache() const
    {
        TIMER(t, "populateOrderCache");
        if(m_cacheStatus == CacheStatus::Valid)
            return;
        m_cacheDenseToTag.clear();
        m_cacheTagToDense.clear();
        m_cacheWords = 0;
        m_cacheAfter.clear();
        m_cacheBefore.clear();
        m_cacheInBody.clear();
        m_cacheContaining.clear();
        static_assert(std::variant_size_v<ControlEdge> == 5,
                      "Assumes Sequence(0), Initialize(1), ForLoopIncrement(2), Body(3), Else(4).");
        constexpr uint8_t kSequence      = 0;
        constexpr uint8_t kInit          = 1;
        constexpr uint8_t kInc           = 2;
        constexpr uint8_t kBody          = 3;
        constexpr uint8_t kElse          = 4;
        constexpr size_t  kEdgeTypeCount = std::variant_size_v<ControlEdge>;
        using GD                         = Graph::Direction;
        auto nodeTags                    = getNodes().to<std::vector>();
        if(nodeTags.empty())
        {
            m_cacheStatus = CacheStatus::Valid;
            return;
        }
        int const nodeCount = static_cast<int>(nodeTags.size());
        int const maxTag    = *std::max_element(nodeTags.begin(), nodeTags.end());
        // Dense mapping.
        m_cacheDenseToTag = nodeTags;
        m_cacheTagToDense.assign(static_cast<size_t>(maxTag) + 1, -1);
        for(int d = 0; d < nodeCount; ++d)
            m_cacheTagToDense[static_cast<size_t>(m_cacheDenseToTag[d])] = d;
        m_cacheWords            = wordsForBits(static_cast<size_t>(nodeCount));
        size_t const matrixSize = static_cast<size_t>(nodeCount) * m_cacheWords;
        m_cacheAfter.assign(matrixSize, 0);
        m_cacheBefore.assign(matrixSize, 0);
        m_cacheInBody.assign(matrixSize, 0);
        m_cacheContaining.assign(matrixSize, 0);
        // allDesc[row=u_dense] contains all descendants of u (dense bits).
        std::vector<uint64_t>                                     allDesc(matrixSize, 0);
        std::vector<std::array<std::vector<int>, kEdgeTypeCount>> childrenByType(nodeCount);
        std::vector<int>                                          indegree(nodeCount, 0);
        // Build adjacency (node -> child nodes by outgoing edge type).
        for(int u = 0; u < nodeCount; ++u)
        {
            int const parentTag = m_cacheDenseToTag[u];
            for(int edgeTag : getNeighbours<GD::Downstream>(parentTag))
            {
                uint8_t const edgeType = static_cast<uint8_t>(getEdge(edgeTag).index());
                auto&         bucket   = childrenByType[u][edgeType];
                for(int childTag : getNeighbours<GD::Downstream>(edgeTag))
                {
                    if(childTag < 0 || childTag > maxTag)
                        continue;
                    int const v = m_cacheTagToDense[static_cast<size_t>(childTag)];
                    if(v >= 0)
                        bucket.push_back(v);
                }
            }
        }
        // Deduplicate buckets and compute indegree after dedup.
        for(int u = 0; u < nodeCount; ++u)
        {
            for(auto& bucket : childrenByType[u])
            {
                std::sort(bucket.begin(), bucket.end());
                bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
                for(int v : bucket)
                    ++indegree[v];
            }
        }
        // Kahn topological order.
        std::vector<int> q;
        q.reserve(nodeCount);
        for(int i = 0; i < nodeCount; ++i)
            if(indegree[i] == 0)
                q.push_back(i);
        std::vector<int> topo;
        topo.reserve(nodeCount);
        for(size_t head = 0; head < q.size(); ++head)
        {
            int const u = q[head];
            topo.push_back(u);
            for(auto const& bucket : childrenByType[u])
            {
                for(int v : bucket)
                {
                    if(--indegree[v] == 0)
                        q.push_back(v);
                }
            }
        }
        AssertFatal(static_cast<int>(topo.size()) == nodeCount,
                    "ControlGraph must be acyclic for order-cache population.");
        // Reused temporaries: fixed-size dense bitsets.
        std::array<Bitset, kEdgeTypeCount> typedDesc;
        for(auto& bs : typedDesc)
            bs.assign(m_cacheWords, 0);
        Bitset seen(m_cacheWords, 0);
        Bitset bodyLike(m_cacheWords, 0);
        Bitset afterInit(m_cacheWords, 0);
        Bitset afterBody(m_cacheWords, 0);
        Bitset afterElse(m_cacheWords, 0);
        auto   mergeChildrenDesc = [&](Bitset& dst, std::vector<int> const& childDenseList) {
            for(int child : childDenseList)
            {
                denseSet(dst, child);
                rowOr(dst.data(), matrixRow(allDesc, m_cacheWords, child), m_cacheWords);
            }
        };
        // Reverse topo => children processed before parents.
        for(auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            int const u = *it;
            for(auto& bs : typedDesc)
                std::fill(bs.begin(), bs.end(), 0);
            std::fill(seen.begin(), seen.end(), 0);
            std::fill(bodyLike.begin(), bodyLike.end(), 0);
            std::fill(afterInit.begin(), afterInit.end(), 0);
            std::fill(afterBody.begin(), afterBody.end(), 0);
            std::fill(afterElse.begin(), afterElse.end(), 0);
            mergeChildrenDesc(typedDesc[kInit], childrenByType[u][kInit]);
            mergeChildrenDesc(typedDesc[kBody], childrenByType[u][kBody]);
            mergeChildrenDesc(typedDesc[kElse], childrenByType[u][kElse]);
            mergeChildrenDesc(typedDesc[kInc], childrenByType[u][kInc]);
            mergeChildrenDesc(typedDesc[kSequence], childrenByType[u][kSequence]);
            // Make partitions disjoint by precedence:
            // Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.
            // This removes contradictory cross-writes in merged-path corner cases.
            seen = typedDesc[kInit];
            denseAndNot(typedDesc[kBody], seen);
            denseOr(seen, typedDesc[kBody]);
            denseAndNot(typedDesc[kElse], seen);
            denseOr(seen, typedDesc[kElse]);
            denseAndNot(typedDesc[kInc], seen);
            denseOr(seen, typedDesc[kInc]);
            denseAndNot(typedDesc[kSequence], seen);
            auto& initNodes = typedDesc[kInit];
            auto& bodyNodes = typedDesc[kBody];
            auto& elseNodes = typedDesc[kElse];
            auto& incNodes  = typedDesc[kInc];
            auto& seqNodes  = typedDesc[kSequence];
            // bodyLike = init | body | else | inc
            bodyLike = initNodes;
            denseOr(bodyLike, bodyNodes);
            denseOr(bodyLike, elseNodes);
            denseOr(bodyLike, incNodes);
            // Parent relationships.
            writeOrderCache(u, bodyLike, NodeOrdering::RightInBodyOfLeft);
            writeOrderCache(u, seqNodes, NodeOrdering::LeftFirst);
            // Cross-partition relationships.
            // IMPORTANT correctness fix: afterInit excludes initNodes itself.
            afterInit = bodyNodes;
            denseOr(afterInit, elseNodes);
            denseOr(afterInit, incNodes);
            denseOr(afterInit, seqNodes);
            writeOrderCache(initNodes, afterInit, NodeOrdering::LeftFirst);
            afterBody = elseNodes;
            denseOr(afterBody, incNodes);
            denseOr(afterBody, seqNodes);
            writeOrderCache(bodyNodes, afterBody, NodeOrdering::LeftFirst);
            afterElse = incNodes;
            denseOr(afterElse, seqNodes);
            writeOrderCache(elseNodes, afterElse, NodeOrdering::LeftFirst);
            writeOrderCache(incNodes, seqNodes, NodeOrdering::LeftFirst);
            // Cache all descendants for parent computations.
            auto* allRow = matrixRow(allDesc, m_cacheWords, u);
            for(size_t w = 0; w < m_cacheWords; ++w)
                allRow[w] = bodyLike[w] | seqNodes[w];
            // Defensive: no self descendant.
            allRow[static_cast<size_t>(u) >> 6] &= ~(uint64_t(1) << (u & 63));
        }
        validateOrderCache();
        m_cacheStatus = CacheStatus::Valid;
    }

    /*
void ControlGraph::populateOrderCache() const
  {
      TIMER(t, "populateOrderCache");
      if(m_cacheStatus == CacheStatus::Valid)
          return;
      m_orderCache.clear();
      m_descendentCache.clear();
      static_assert(std::variant_size_v<ControlEdge> == 5,
                    "Edge types assumed: Sequence(0), Initialize(1), ForLoopIncrement(2), "
                    "Body(3), Else(4). Update populateOrderCache if this changes.");
      constexpr int kEdgeTypeCount = std::variant_size_v<ControlEdge>;
      // Variant indices in ControlEdge.
      constexpr uint8_t kSequence = 0;
      constexpr uint8_t kInit     = 1;
      constexpr uint8_t kInc      = 2;
      constexpr uint8_t kBody     = 3;
      constexpr uint8_t kElse     = 4;
      // Per-node topology in dense-node-index space (not graph tag space).
      // We keep dense indexing to avoid hash lookups in hot loops.
      struct NodeTopology
      {
          // Child node indices grouped by outgoing edge type.
          std::array<std::vector<int>, kEdgeTypeCount> childrenByTypeIdx;
          // Immediate parents of this node: {parentNodeIndex, edgeType(parent->this)}.
          std::vector<std::pair<int, uint8_t>> parents;
      };
      // Memoized descendants grouped by "first edge type" from the source node.
      struct DescInfo
      {
          std::array<Bitset, kEdgeTypeCount> byFirstEdge;
          Bitset                             all; // union of all byFirstEdge sets
          uint8_t                            dfsState = 0; // 0=unvisited,1=visiting,2=done
      };
      using GD = Graph::Direction;
      auto nodeTags = getNodes().to<std::vector>();
      if(nodeTags.empty())
      {
          m_cacheStatus = CacheStatus::Valid;
          return;
      }
      // Optional micro-optimization:
      // Process larger tags first so "set one bit in many rows" tends to grow bitsets once.
      std::sort(nodeTags.begin(), nodeTags.end(), std::greater<int>{});
      int const nodeCount = static_cast<int>(nodeTags.size());
      // Tag -> dense index.
      std::unordered_map<int, int> tagToIdx;
      tagToIdx.reserve(nodeTags.size());
      for(int i = 0; i < nodeCount; ++i)
          tagToIdx.emplace(nodeTags[i], i);
      std::vector<NodeTopology> topo(nodeCount);
      // Build parent and childrenByType adjacency once from graph edges.
      // This avoids repeated getNeighbours() calls inside per-node ancestor traversals.
      for(int edge : getEdges().to<std::vector>())
      {
          uint8_t const edgeType = static_cast<uint8_t>(getEdge(edge).index());
          auto const parents = getNeighbours<GD::Upstream>(edge);
          auto const kids    = getNeighbours<GD::Downstream>(edge);
          for(int pTag : parents)
          {
              auto pIt = tagToIdx.find(pTag);
              if(pIt == tagToIdx.end())
                  continue;
              int const pIdx = pIt->second;
              auto&     bucket = topo[pIdx].childrenByTypeIdx[edgeType];
              for(int cTag : kids)
              {
                  auto cIt = tagToIdx.find(cTag);
                  if(cIt == tagToIdx.end())
                      continue;
                  int const cIdx = cIt->second;
                  bucket.push_back(cIdx);
                  topo[cIdx].parents.emplace_back(pIdx, edgeType);
              }
          }
      }
      // Descendant cache by first-edge-type, computed once per node.
      std::vector<DescInfo> desc(nodeCount);
      auto buildDescByFirstEdge = [&](auto&& self, int nodeIdx) -> DescInfo const& {
          auto& info = desc[nodeIdx];
          if(info.dfsState == 2)
              return info;
          // If this fires, graph has a cycle in node-level reachability.
          // Current cache logic (old and new) assumes acyclic control structure.
          AssertFatal(info.dfsState != 1,
                      "ControlGraph must be acyclic while building order cache.",
                      ShowValue(nodeTags[nodeIdx]));
          info.dfsState = 1;
          for(uint8_t edgeType = 0; edgeType < kEdgeTypeCount; ++edgeType)
          {
              for(int childIdx : topo[nodeIdx].childrenByTypeIdx[edgeType])
              {
                  int const childTag = nodeTags[childIdx];
                  // Direct child belongs to this first-edge-type bucket.
                  bitsetSet(info.byFirstEdge[edgeType], childTag);
                  // Everything reachable from that child also belongs to same first-edge bucket.
                  auto const& childInfo = self(self, childIdx);
                  bitsetOr(info.byFirstEdge[edgeType], childInfo.all);
              }
              bitsetOr(info.all, info.byFirstEdge[edgeType]);
          }
          info.dfsState = 2;
          return info;
      };
      for(int i = 0; i < nodeCount; ++i)
          (void)buildDescByFirstEdge(buildDescByFirstEdge, i);
      // Select which NodeOrders bitset a NodeOrdering writes to.
      auto selectField = [](NodeOrdering order) -> Bitset NodeOrders::* {
          switch(order)
          {
          case NodeOrdering::LeftFirst:
              return &NodeOrders::after;
          case NodeOrdering::RightFirst:
              return &NodeOrders::before;
          case NodeOrdering::RightInBodyOfLeft:
              return &NodeOrders::inBody;
          case NodeOrdering::LeftInBodyOfRight:
              return &NodeOrders::containing;
          default:
              break;
          }
          AssertFatal(false, "Invalid order: ", ShowValue(order));
          return &NodeOrders::after;
      };
      // Writes A -> nodeB (many-to-one) efficiently:
      // 1) set one bit in each left row
      // 2) one bulk OR into right row's opposite set
      auto writeManyToOne = [&](Bitset const& leftNodes, int nodeB, NodeOrdering order) {
          auto const lhsField = selectField(order);
          auto const rhsField = selectField(opposite(order));
          bool any = false;
          for(size_t w = 0; w < leftNodes.size(); ++w)
          {
              uint64_t bits = leftNodes[w];
              while(bits)
              {
                  int const bit     = std::countr_zero(bits);
                  int const leftTag = static_cast<int>(w * 64 + bit);
                  bitsetSet((m_orderCache[leftTag].*lhsField), nodeB);
                  bits &= bits - 1;
                  any = true;
              }
          }
          if(any)
              bitsetOr((m_orderCache[nodeB].*rhsField), leftNodes);
      };
      auto clearBit = [](Bitset& bs, int id) {
          size_t const word = static_cast<size_t>(id) >> 6;
          if(word < bs.size())
              bs[word] &= ~(uint64_t(1) << (id & 63));
      };
      m_orderCache.reserve(nodeTags.size());
      // Stamp+mask visitation:
      // - stamp tells if ancestor was visited for current target node
      // - mask dedups by (ancestor, viaEdgeType), not just ancestor
      // This is important because same ancestor can be reached through different first-edge types.
      std::vector<uint32_t> seenStamp(nodeCount, 0);
      std::vector<uint8_t>  seenMask(nodeCount, 0);
      uint32_t              stamp = 0;
      std::vector<std::pair<int, uint8_t>> stack;
      stack.reserve(64);
      // Main algorithm (your idea):
      // For each target node:
      //   - traverse ancestors upward
      //   - add direct ancestor relation based on incoming edge type
      //   - add all descendants of ancestor that are known to execute before target
      for(int nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx)
      {
          int const nodeTag = nodeTags[nodeIdx];
          ++stamp;
          Bitset before;     // nodes definitely before nodeTag
          Bitset containing; // nodes that contain nodeTag in body
          stack.clear();
          for(auto const& p : topo[nodeIdx].parents)
              stack.push_back(p);
          while(!stack.empty())
          {
              auto const [ancIdx, viaEdgeType] = stack.back();
              stack.pop_back();
              if(seenStamp[ancIdx] != stamp)
              {
                  seenStamp[ancIdx] = stamp;
                  seenMask[ancIdx]  = 0;
              }
              uint8_t const edgeMask = static_cast<uint8_t>(1u << viaEdgeType);
              if(seenMask[ancIdx] & edgeMask)
                  continue;
              seenMask[ancIdx] |= edgeMask;
              int const ancTag = nodeTags[ancIdx];
              // Ancestor relation.
              if(viaEdgeType == kSequence)
                  bitsetSet(before, ancTag); // ancestor is before target
              else
                  bitsetSet(containing, ancTag); // target is in ancestor's body
              // Descendants of ancestor that are before this target, based on edge precedence:
              // Initialize < Body < Else < ForLoopIncrement < Sequence
              auto const& byType = desc[ancIdx].byFirstEdge;
              switch(viaEdgeType)
              {
              case kInit:
                  // Nothing before Initialize bucket.
                  break;
              case kBody:
                  bitsetOr(before, byType[kInit]);
                  break;
              case kElse:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  break;
              case kInc:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  bitsetOr(before, byType[kElse]);
                  break;
              case kSequence:
                  bitsetOr(before, byType[kInit]);
                  bitsetOr(before, byType[kBody]);
                  bitsetOr(before, byType[kElse]);
                  bitsetOr(before, byType[kInc]);
                  break;
              default:
                  AssertFatal(false, "Unhandled edge type index", ShowValue(viaEdgeType));
              }
              // Continue climbing ancestors.
              for(auto const& p : topo[ancIdx].parents)
                  stack.push_back(p);
          }
          // Defensive: avoid self relation if malformed graph introduces it.
          clearBit(before, nodeTag);
          clearBit(containing, nodeTag);
          // Materialize two relation families.
          writeManyToOne(before, nodeTag, NodeOrdering::LeftFirst);
          writeManyToOne(containing, nodeTag, NodeOrdering::RightInBodyOfLeft);
      }
      validateOrderCache();
      m_cacheStatus = CacheStatus::Valid;
      m_descendentCache.clear();
  }
*/

    //void ControlGraph::populateOrderCache() const
    //{
    //    TIMER(t, "populateOrderCache");

    //    if(m_cacheStatus == CacheStatus::Valid)
    //        return;

    //    m_orderCache.clear();

    //    auto rootNodes = roots().to<std::vector>();

    //    populateOrderCache(rootNodes);
    //    validateOrderCache();
    //    //sortOrderCache();

    //    m_cacheStatus = CacheStatus::Valid;
    //    //
    //    // m_descendentCache is only used to help build m_orderCache,
    //    // and it must be cleared after finish building m_orderCache
    //    // to ensure no stale data being used when building m_orderCache
    //    // next time.
    //    //
    //    m_descendentCache.clear();
    //}

    void ControlGraph::writeOrderCache(Bitset const& nodesA,
                                       Bitset const& nodesB,
                                       NodeOrdering  order) const
    {
        if(!bitsetAny(nodesA) || !bitsetAny(nodesB))
            return;
        auto selectMatrix = [this](NodeOrdering o) -> std::vector<uint64_t>& {
            switch(o)
            {
            case NodeOrdering::LeftFirst:
                return m_cacheAfter;
            case NodeOrdering::RightFirst:
                return m_cacheBefore;
            case NodeOrdering::RightInBodyOfLeft:
                return m_cacheInBody;
            case NodeOrdering::LeftInBodyOfRight:
                return m_cacheContaining;
            default:
                break;
            }
            AssertFatal(false, "Invalid order", ShowValue(o));
            return m_cacheAfter;
        };
        auto&     matAB = selectMatrix(order);
        auto&     matBA = selectMatrix(opposite(order));
        int const n     = static_cast<int>(m_cacheDenseToTag.size());
        forEachSetBit(nodesA, [&](int aDense) {
            if(aDense >= n)
                return;
            rowOr(matrixRow(matAB, m_cacheWords, aDense), nodesB.data(), m_cacheWords);
        });
        forEachSetBit(nodesB, [&](int bDense) {
            if(bDense >= n)
                return;
            rowOr(matrixRow(matBA, m_cacheWords, bDense), nodesA.data(), m_cacheWords);
        });
    }
    void ControlGraph::writeOrderCache(int nodeA, Bitset const& nodesB, NodeOrdering order) const
    {
        if(!bitsetAny(nodesB))
            return;
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        AssertFatal(nodeA >= 0 && nodeA < n, ShowValue(nodeA), ShowValue(n));
        auto selectMatrix = [this](NodeOrdering o) -> std::vector<uint64_t>& {
            switch(o)
            {
            case NodeOrdering::LeftFirst:
                return m_cacheAfter;
            case NodeOrdering::RightFirst:
                return m_cacheBefore;
            case NodeOrdering::RightInBodyOfLeft:
                return m_cacheInBody;
            case NodeOrdering::LeftInBodyOfRight:
                return m_cacheContaining;
            default:
                break;
            }
            AssertFatal(false, "Invalid order", ShowValue(o));
            return m_cacheAfter;
        };
        auto& matAB = selectMatrix(order);
        auto& matBA = selectMatrix(opposite(order));
        rowOr(matrixRow(matAB, m_cacheWords, nodeA), nodesB.data(), m_cacheWords);
        forEachSetBit(nodesB, [&](int bDense) {
            if(bDense >= n)
                return;
            auto* row = matrixRow(matBA, m_cacheWords, bDense);
            row[static_cast<size_t>(nodeA) >> 6] |= uint64_t(1) << (nodeA & 63);
        });
    }
    void ControlGraph::writeOrderCache(int nodeA, int nodeB, NodeOrdering order) const
    {
        int const n = static_cast<int>(m_cacheDenseToTag.size());
        AssertFatal(nodeA >= 0 && nodeA < n, ShowValue(nodeA), ShowValue(n));
        AssertFatal(nodeB >= 0 && nodeB < n, ShowValue(nodeB), ShowValue(n));
        auto selectMatrix = [this](NodeOrdering o) -> std::vector<uint64_t>& {
            switch(o)
            {
            case NodeOrdering::LeftFirst:
                return m_cacheAfter;
            case NodeOrdering::RightFirst:
                return m_cacheBefore;
            case NodeOrdering::RightInBodyOfLeft:
                return m_cacheInBody;
            case NodeOrdering::LeftInBodyOfRight:
                return m_cacheContaining;
            default:
                break;
            }
            AssertFatal(false, "Invalid order", ShowValue(o));
            return m_cacheAfter;
        };
        auto& matAB = selectMatrix(order);
        auto& matBA = selectMatrix(opposite(order));
        auto* rowAB = matrixRow(matAB, m_cacheWords, nodeA);
        auto* rowBA = matrixRow(matBA, m_cacheWords, nodeB);
        rowAB[static_cast<size_t>(nodeB) >> 6] |= uint64_t(1) << (nodeB & 63);
        rowBA[static_cast<size_t>(nodeA) >> 6] |= uint64_t(1) << (nodeA & 63);
    }

    NodeOrdering ControlGraph::lookupOrder(IgnoreCachePolicy const, int nodeA, int nodeB) const
    {
        TIMER(t, "ControlGraph::lookupOrder");

        using GD = Graph::Direction;
        std::unordered_set<int> visited_nodes;

        // Decide order of A and B when A is parent of B
        auto const getOrderIfParent = [&](int edge) {
            return std::holds_alternative<Sequence>(getEdge(edge))
                       ? NodeOrdering::LeftFirst
                       : NodeOrdering::RightInBodyOfLeft;
        };

        // Decide order of A and B when A and B are descendants of a node
        // And order of edge type :
        // Initialize -> Body -> Else -> ForLoopIncrement -> Sequence.
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

        // {key, value} = {node index, edge index}
        std::unordered_map<int, int> A_ancestors;
        // stack
        std::vector<int> stk{nodeA};

        // Traverse upstream from A to collect all ancestors of A
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

        // Traverse upstream from B to find common ancestors to decide order
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
                        // If this is a common ancestor, compare the types of both edges to
                        // know the order
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
