// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/MergeConditionalLoads.hpp>

using namespace rocRoller;
using namespace rocRoller::Expression;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

// Alias to avoid ambiguity between the KernelGraph namespace and the KernelGraph class.
using KG = rocRoller::KernelGraph::KernelGraph;

namespace
{
    /**
     * Build a minimal KernelGraph containing:
     *
     *   Kernel
     *     └─Body─► ForLoopOp
     *                 └─Body─► loadA  (MATRIX_A, memType)
     *                        ► loadB  (MATRIX_B, memType)
     *
     * Both tiles share the same sizes and dataType.
     */
    struct MinimalGraph
    {
        KG  graph;
        int kernel;
        int forLoop;
        int loadA, loadB;
        int tileA, tileB;
        int userA, userB;
    };

    MinimalGraph buildGraph(MemoryType       memType = MemoryType::WAVE_LDS,
                            std::vector<int> sizes   = {32, 32})
    {
        MinimalGraph mg;
        auto&        g = mg.graph;

        // Control nodes
        mg.kernel = g.control.addElement(Kernel());

        // ForLoopOp needs a valid ExpressionPtr condition.
        // Use a simple literal comparison: 0 < 10 (always true at translate time).
        auto forLoopCoord = g.coordinates.addElement(ForLoop());
        auto counter = dataFlowTag(forLoopCoord, Register::Type::Scalar, DataType::Int32);
        mg.forLoop = g.control.addElement(ForLoopOp{counter < literal(10), "forLoopK"});

        mg.loadA = g.control.addElement(LoadTiled());
        mg.loadB = g.control.addElement(LoadTiled());

        // Control edges
        g.control.addElement(Body(), {mg.kernel}, {mg.forLoop});
        g.control.addElement(Body(), {mg.forLoop}, {mg.loadA});
        g.control.addElement(Sequence(), {mg.loadA}, {mg.loadB});

        // Coordinate nodes — MacroTile A
        MacroTile tileADim;
        tileADim.layoutType = LayoutType::MATRIX_A;
        tileADim.memoryType = memType;
        tileADim.sizes      = sizes;
        mg.tileA            = g.coordinates.addElement(tileADim);

        // MacroTile B
        MacroTile tileBDim;
        tileBDim.layoutType = LayoutType::MATRIX_B;
        tileBDim.memoryType = memType;
        tileBDim.sizes      = sizes;
        mg.tileB            = g.coordinates.addElement(tileBDim);

        // User A
        mg.userA = g.coordinates.addElement(User("tensorA"));

        // User B
        mg.userB = g.coordinates.addElement(User("tensorB"));

        // Mapper connections (plain TypeAndSubDimension — not yet merged)
        g.mapper.connect<MacroTile>(mg.loadA, mg.tileA);
        g.mapper.connect<User>(mg.loadA, mg.userA);
        g.mapper.connect<MacroTile>(mg.loadB, mg.tileB);
        g.mapper.connect<User>(mg.loadB, mg.userB);

        return mg;
    }
} // anonymous namespace

// ---------------------------------------------------------------------------
// No-candidate paths (transform is a no-op)
// ---------------------------------------------------------------------------

TEST_CASE("MergeConditionalLoads - no candidates: empty graph", "[merge-conditional-loads]")
{
    KG graph;
    graph.control.addElement(Kernel());

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(graph);

    auto loads = result.control.getNodes<LoadTiled>().to<std::vector>();
    CHECK(loads.empty());
}

TEST_CASE("MergeConditionalLoads - no candidates: single MATRIX_A only",
          "[merge-conditional-loads]")
{
    KG   graph;
    auto kernel     = graph.control.addElement(Kernel());
    auto flCoord    = graph.coordinates.addElement(ForLoop());
    auto counter    = dataFlowTag(flCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoopTag = graph.control.addElement(ForLoopOp{counter < literal(8), "k"});
    auto loadA      = graph.control.addElement(LoadTiled());

    graph.control.addElement(Body(), {kernel}, {forLoopTag});
    graph.control.addElement(Body(), {forLoopTag}, {loadA});

    MacroTile tileA;
    tileA.layoutType = LayoutType::MATRIX_A;
    tileA.memoryType = MemoryType::WAVE_LDS;
    tileA.sizes      = {32, 32};
    auto tileATag    = graph.coordinates.addElement(tileA);

    auto userATag = graph.coordinates.addElement(User("A"));

    graph.mapper.connect<MacroTile>(loadA, tileATag);
    graph.mapper.connect<User>(loadA, userATag);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(graph);

    // Still exactly one LoadTiled — no merge possible
    auto loads = result.control.getNodes<LoadTiled>().to<std::vector>();
    CHECK(loads.size() == 1);
    CHECK(result.mapper.getWaveGroup<User>(loadA, 0) == -1);
}

TEST_CASE("MergeConditionalLoads - no candidates: mismatched memoryType",
          "[merge-conditional-loads]")
{
    KG   graph;
    auto kernel     = graph.control.addElement(Kernel());
    auto flCoord    = graph.coordinates.addElement(ForLoop());
    auto counter    = dataFlowTag(flCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoopTag = graph.control.addElement(ForLoopOp{counter < literal(8), "k"});
    auto loadA      = graph.control.addElement(LoadTiled());
    auto loadB      = graph.control.addElement(LoadTiled());

    graph.control.addElement(Body(), {kernel}, {forLoopTag});
    graph.control.addElement(Body(), {forLoopTag}, {loadA});
    graph.control.addElement(Sequence(), {loadA}, {loadB});

    MacroTile tileA;
    tileA.layoutType = LayoutType::MATRIX_A;
    tileA.memoryType = MemoryType::WAVE_LDS;
    tileA.sizes      = {32, 32};
    auto tileATag    = graph.coordinates.addElement(tileA);

    MacroTile tileB;
    tileB.layoutType = LayoutType::MATRIX_B;
    tileB.memoryType = MemoryType::WAVE_Direct2LDS; // different — no merge
    tileB.sizes      = {32, 32};
    auto tileBTag    = graph.coordinates.addElement(tileB);

    auto userATag = graph.coordinates.addElement(User("A"));
    auto userBTag = graph.coordinates.addElement(User("B"));

    graph.mapper.connect<MacroTile>(loadA, tileATag);
    graph.mapper.connect<User>(loadA, userATag);
    graph.mapper.connect<MacroTile>(loadB, tileBTag);
    graph.mapper.connect<User>(loadB, userBTag);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(graph);

    CHECK(result.mapper.getWaveGroup<User>(loadA, 0) == -1);
    CHECK(result.mapper.getWaveGroup<User>(loadB, 0) == -1);
}

TEST_CASE("MergeConditionalLoads - no candidates: ops in different ForLoopK",
          "[merge-conditional-loads]")
{
    KG   graph;
    auto kernel  = graph.control.addElement(Kernel());
    auto flCoord = graph.coordinates.addElement(ForLoop());
    auto counter = dataFlowTag(flCoord, Register::Type::Scalar, DataType::Int32);

    auto forLoopA = graph.control.addElement(ForLoopOp{counter < literal(8), "kA"});
    auto forLoopB = graph.control.addElement(ForLoopOp{counter < literal(8), "kB"});
    auto loadA    = graph.control.addElement(LoadTiled());
    auto loadB    = graph.control.addElement(LoadTiled());

    graph.control.addElement(Body(), {kernel}, {forLoopA});
    graph.control.addElement(Sequence(), {forLoopA}, {forLoopB});
    graph.control.addElement(Body(), {forLoopA}, {loadA});
    graph.control.addElement(Body(), {forLoopB}, {loadB});

    MacroTile tileA;
    tileA.layoutType = LayoutType::MATRIX_A;
    tileA.memoryType = MemoryType::WAVE_LDS;
    tileA.sizes      = {32, 32};
    auto tileATag    = graph.coordinates.addElement(tileA);

    MacroTile tileB;
    tileB.layoutType = LayoutType::MATRIX_B;
    tileB.memoryType = MemoryType::WAVE_LDS;
    tileB.sizes      = {32, 32};
    auto tileBTag    = graph.coordinates.addElement(tileB);

    auto userATag = graph.coordinates.addElement(User("A"));
    auto userBTag = graph.coordinates.addElement(User("B"));

    graph.mapper.connect<MacroTile>(loadA, tileATag);
    graph.mapper.connect<User>(loadA, userATag);
    graph.mapper.connect<MacroTile>(loadB, tileBTag);
    graph.mapper.connect<User>(loadB, userBTag);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(graph);

    // loadA and loadB are in different ForLoopOp parents — no merge
    CHECK(result.mapper.getWaveGroup<User>(loadA, 0) == -1);
    CHECK(result.mapper.getWaveGroup<User>(loadB, 0) == -1);
}

// ---------------------------------------------------------------------------
// Merge path — structural assertions
// ---------------------------------------------------------------------------

TEST_CASE("MergeConditionalLoads - merges WAVE_LDS pair: WaveGroupBranch connections",
          "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(mg.graph);

    // loadA must have WaveGroupBranch{0}→userA and WaveGroupBranch{1}→userB
    CHECK(result.mapper.getWaveGroup<User>(mg.loadA, 0) == mg.userA);
    CHECK(result.mapper.getWaveGroup<User>(mg.loadA, 1) == mg.userB);

    // MacroTile connection still points to tileA
    CHECK(result.mapper.get<MacroTile>(mg.loadA) == mg.tileA);

    // Adhoc waveGroup coordinate is connected and valid
    int waveGroupAdhoc = result.mapper.get<Adhoc>(mg.loadA);
    REQUIRE(waveGroupAdhoc != -1);
    auto adhocOpt = result.coordinates.get<Adhoc>(waveGroupAdhoc);
    REQUIRE(adhocOpt);
    CHECK(adhocOpt->name() == "waveGroup");
}

TEST_CASE("MergeConditionalLoads - merges WAVE_LDS pair: loadB replaced by NOP",
          "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(mg.graph);

    // loadB replaced by NOP
    auto elem = result.control.getElement(mg.loadB);
    CHECK(std::holds_alternative<NOP>(std::get<Operation>(elem)));

    // Exactly one LoadTiled remains
    auto loads = result.control.getNodes<LoadTiled>().to<std::vector>();
    CHECK(loads.size() == 1);
    CHECK(loads[0] == mg.loadA);
}

TEST_CASE("MergeConditionalLoads - merges WAVE_LDS pair: tileB removed from coordinates",
          "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(mg.graph);

    // tileB must have been removed from the coordinate graph
    CHECK(!result.coordinates.get<MacroTile>(mg.tileB));
}

TEST_CASE(
    "MergeConditionalLoads - merges WAVE_LDS pair: waveGroup Assign inserted before ForLoopK",
    "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(mg.graph);

    // At least one Assign must exist (the waveGroup prolog Assign)
    auto assigns = result.control.getNodes<Assign>().to<std::vector>();
    REQUIRE(!assigns.empty());

    // One Assign must have its DEST connected to the waveGroup Adhoc coordinate
    int  waveGroupAdhoc      = result.mapper.get<Adhoc>(mg.loadA);
    bool foundWaveGroupAssign = false;
    for(int assignTag : assigns)
    {
        for(auto const& c : result.mapper.getConnections(assignTag))
        {
            if(c.coordinate == waveGroupAdhoc)
            {
                foundWaveGroupAssign = true;
                break;
            }
        }
        if(foundWaveGroupAssign)
            break;
    }
    CHECK(foundWaveGroupAssign);
}

TEST_CASE("MergeConditionalLoads - merges WAVE_Direct2LDS pair", "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_Direct2LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(mg.graph);

    CHECK(result.mapper.getWaveGroup<User>(mg.loadA, 0) == mg.userA);
    CHECK(result.mapper.getWaveGroup<User>(mg.loadA, 1) == mg.userB);
    CHECK(result.mapper.get<MacroTile>(mg.loadA) == mg.tileA);

    auto elem = result.control.getElement(mg.loadB);
    CHECK(std::holds_alternative<NOP>(std::get<Operation>(elem)));
}

TEST_CASE("MergeConditionalLoads - idempotent: second apply finds no new candidates",
          "[merge-conditional-loads]")
{
    auto mg = buildGraph(MemoryType::WAVE_LDS);

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto after1 = transform.apply(mg.graph);
    auto after2 = transform.apply(after1);

    // Still exactly one LoadTiled after two passes
    auto loads = after2.control.getNodes<LoadTiled>().to<std::vector>();
    CHECK(loads.size() == 1);

    // WaveGroupBranch connections unchanged
    CHECK(after2.mapper.getWaveGroup<User>(mg.loadA, 0) == mg.userA);
    CHECK(after2.mapper.getWaveGroup<User>(mg.loadA, 1) == mg.userB);
}

TEST_CASE("MergeConditionalLoads - two independent pairs in same ForLoopK",
          "[merge-conditional-loads]")
{
    KG   graph;
    auto kernel     = graph.control.addElement(Kernel());
    auto flCoord    = graph.coordinates.addElement(ForLoop());
    auto counter    = dataFlowTag(flCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoopTag = graph.control.addElement(ForLoopOp{counter < literal(16), "k"});

    auto loadA1 = graph.control.addElement(LoadTiled());
    auto loadB1 = graph.control.addElement(LoadTiled());
    auto loadA2 = graph.control.addElement(LoadTiled());
    auto loadB2 = graph.control.addElement(LoadTiled());

    graph.control.addElement(Body(), {kernel}, {forLoopTag});
    graph.control.addElement(Body(), {forLoopTag}, {loadA1});
    graph.control.addElement(Sequence(), {loadA1}, {loadB1});
    graph.control.addElement(Sequence(), {loadB1}, {loadA2});
    graph.control.addElement(Sequence(), {loadA2}, {loadB2});

    // Helper to add a tile+user pair and connect to a LoadTiled
    auto addPair = [&](int loadTag, LayoutType layout, std::string argName) {
        MacroTile tile;
        tile.layoutType = layout;
        tile.memoryType = MemoryType::WAVE_LDS;
        tile.sizes      = {16, 16};
        int tileTag     = graph.coordinates.addElement(tile);
        User user;
        user.argumentName = argName;
        int userTag       = graph.coordinates.addElement(user);
        graph.mapper.connect<MacroTile>(loadTag, tileTag);
        graph.mapper.connect<User>(loadTag, userTag);
        return userTag;
    };

    int userA1 = addPair(loadA1, LayoutType::MATRIX_A, "A1");
    int userB1 = addPair(loadB1, LayoutType::MATRIX_B, "B1");
    int userA2 = addPair(loadA2, LayoutType::MATRIX_A, "A2");
    int userB2 = addPair(loadB2, LayoutType::MATRIX_B, "B2");

    rocRoller::KernelGraph::MergeConditionalLoads transform(nullptr, nullptr);
    auto result = transform.apply(graph);

    // Exactly 2 LoadTiled ops must remain (one per pair)
    auto loads = result.control.getNodes<LoadTiled>().to<std::vector>();
    CHECK(loads.size() == 2);

    // Exactly 2 NOPs (one per replaced B-side)
    auto nops = result.control.getNodes<NOP>().to<std::vector>();
    CHECK(nops.size() == 2);

    // Each surviving LoadTiled must have both WaveGroupBranch connections and an Adhoc
    for(int load : loads)
    {
        CHECK(result.mapper.getWaveGroup<User>(load, 0) != -1);
        CHECK(result.mapper.getWaveGroup<User>(load, 1) != -1);
        CHECK(result.mapper.get<Adhoc>(load) != -1);
    }

    // The two pairs must have merged the correct user connections
    // (loadA1→userA1/userB1, loadA2→userA2/userB2 — order determined by search)
    std::set<int> wg0Users, wg1Users;
    for(int load : loads)
    {
        wg0Users.insert(result.mapper.getWaveGroup<User>(load, 0));
        wg1Users.insert(result.mapper.getWaveGroup<User>(load, 1));
    }
    CHECK(wg0Users.count(userA1) == 1);
    CHECK(wg0Users.count(userA2) == 1);
    CHECK(wg1Users.count(userB1) == 1);
    CHECK(wg1Users.count(userB2) == 1);
}
