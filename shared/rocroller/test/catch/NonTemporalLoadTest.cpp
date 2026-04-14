// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestContext.hpp"
#include "CustomMatchers.hpp"

#include <common/CommonGraphs.hpp>
#include <common/SourceMatcher.hpp>
#include <common/mxDataGen.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/SetNonTemporal.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sstream>

using namespace rocRoller;
using Catch::Matchers::ContainsSubstring;

// ---------------------------------------------------------------------------
// Step 1 — BufferInstructionOptions: bool nt
// ---------------------------------------------------------------------------

TEST_CASE("BufferInstructionOptions: nt defaults to false", "[non-temporal-loads]")
{
    BufferInstructionOptions opts;
    CHECK(opts.nt == false);
}

TEST_CASE("BufferInstructionOptions: nt=true appears in toString", "[non-temporal-loads]")
{
    BufferInstructionOptions opts;
    opts.nt = true;
    CHECK_THAT(toString(opts), ContainsSubstring("nt"));
}

TEST_CASE("BufferInstructionOptions: nt=false does not indicate true in toString",
          "[non-temporal-loads]")
{
    // ShowValue(options.nt) produces "\toptions.nt = 0\n" or "\toptions.nt = 1\n".
    BufferInstructionOptions opts;
    auto                     str = toString(opts);
    CHECK_THAT(str, ContainsSubstring("nt"));                    // key is present
    CHECK_FALSE(ContainsSubstring("options.nt = 1").match(str)); // not set to true
}

// ---------------------------------------------------------------------------
// Step 2 — loadBuffer() emits "nt"
// ---------------------------------------------------------------------------

static std::string AssembleLoadBuffer(ContextPtr ctx, BufferInstructionOptions buffOpts)
{
    auto k = ctx->kernel();
    k->setKernelName("nt_loadbuffer_test");
    k->setKernelDimensions(1);
    k->addArgument({"buf", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly});

    ctx->schedule(k->preamble());
    ctx->schedule(k->prolog());

    auto kb = [&]() -> Generator<Instruction> {
        Register::ValuePtr s_ptr;
        co_yield ctx->argLoader()->getValue("buf", s_ptr);

        auto v_dest
            = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::Float, 1);
        co_yield v_dest->allocate();

        auto bufDesc = Register::Value::Placeholder(
            ctx, Register::Type::Scalar, DataType::Raw32, 4);
        co_yield bufDesc->allocate();

        co_yield ctx->mem()->loadBuffer(v_dest, s_ptr, 0, bufDesc, buffOpts, 4, false);
    };

    ctx->schedule(kb());
    ctx->schedule(k->postamble());
    ctx->schedule(k->amdgpu_metadata());
    return ctx->instructions()->toString();
}

TEST_CASE("loadBuffer: nt modifier emitted when buffOpts.nt=true", "[non-temporal-loads]")
{
    auto                     context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    BufferInstructionOptions opts;
    opts.nt   = true;
    auto asm_ = AssembleLoadBuffer(context.get(), opts);
    CHECK_THAT(NormalizedSource(asm_), ContainsSubstring("buffer_load"));
    // Verify the nt modifier appears on a buffer_load line (not just in the kernel name).
    // Per-line check avoids a false positive from "nt_loadbuffer_test" in the .amdhsa_kernel
    // section, which also contains " nt" as a prefix of the kernel symbol name.
    {
        bool found = false;
        std::istringstream ss(NormalizedSource(asm_));
        std::string        ln;
        while(std::getline(ss, ln))
            if(ln.find("buffer_load") != std::string::npos && ln.find(" nt") != std::string::npos)
                found = true;
        CHECK(found);
    }
}

TEST_CASE("loadBuffer: no nt modifier when buffOpts.nt=false", "[non-temporal-loads]")
{
    auto                     context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    BufferInstructionOptions opts; // nt=false by default
    auto                     asm_ = AssembleLoadBuffer(context.get(), opts);
    CHECK_THAT(NormalizedSource(asm_), ContainsSubstring("buffer_load"));
    // Verify the nt modifier is absent on all buffer_load lines.
    // Per-line check avoids false matches against the kernel name ("nt_loadbuffer_test").
    {
        std::istringstream ss(NormalizedSource(asm_));
        std::string        ln;
        while(std::getline(ss, ln))
            if(ln.find("buffer_load") != std::string::npos)
                CHECK(ln.find(" nt") == std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Step 3 — bufferLoad2LDS() emits "nt"
// ---------------------------------------------------------------------------

// Note: always sets buffOpts.lds = true internally, regardless of the caller's value,
// because bufferLoad2LDS AssertFatals if lds is not set.
static std::string AssembleBufferLoad2LDS(ContextPtr ctx, BufferInstructionOptions buffOpts)
{
    auto k = ctx->kernel();
    k->setKernelName("nt_load2lds_test");
    k->setKernelDimensions(1);
    k->addArgument({"buf", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly});

    ctx->schedule(k->preamble());
    ctx->schedule(k->prolog());

    auto kb = [&]() -> Generator<Instruction> {
        Register::ValuePtr s_ptr;
        co_yield ctx->argLoader()->getValue("buf", s_ptr);

        auto bufDesc = Register::Value::Placeholder(
            ctx, Register::Type::Scalar, DataType::Raw32, 4);
        co_yield bufDesc->allocate();

        buffOpts.lds = true; // required by bufferLoad2LDS AssertFatal
        auto soffset = Register::Value::Literal(0);
        co_yield ctx->mem()->bufferLoad2LDS(s_ptr, bufDesc, buffOpts, 4, soffset);
    };

    ctx->schedule(kb());
    ctx->schedule(k->postamble());
    ctx->schedule(k->amdgpu_metadata());
    return ctx->instructions()->toString();
}

TEST_CASE("bufferLoad2LDS: nt modifier emitted when buffOpts.nt=true", "[non-temporal-loads]")
{
    auto                     context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    BufferInstructionOptions opts;
    opts.nt   = true;
    auto asm_ = AssembleBufferLoad2LDS(context.get(), opts);
    CHECK_THAT(NormalizedSource(asm_), ContainsSubstring("buffer_load"));
    CHECK_THAT(NormalizedSource(asm_), ContainsSubstring(" nt "));
}

TEST_CASE("bufferLoad2LDS: no nt modifier when buffOpts.nt=false", "[non-temporal-loads]")
{
    auto                     context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    BufferInstructionOptions opts;
    opts.nt   = false;
    auto asm_ = AssembleBufferLoad2LDS(context.get(), opts);
    CHECK_THAT(NormalizedSource(asm_), ContainsSubstring("buffer_load"));
    // " nt " (space-nt-space) is the correct token: bufferLoad2LDS always sets lds=true so "nt"
    // is followed by "lds" in the modifier list, never at end-of-line.
    // The kernel name "nt_load2lds_test" contains " nt_" (underscore), not " nt ", so no
    // false positive.
    CHECK(NormalizedSource(asm_).find(" nt ") == std::string::npos);
}

// ---------------------------------------------------------------------------
// Step 4 — CommandParameters: nonTemporalA / nonTemporalB
// ---------------------------------------------------------------------------

TEST_CASE("CommandParameters: nonTemporalA and nonTemporalB default to false",
          "[non-temporal-loads]")
{
    CommandParameters params;
    CHECK(params.nonTemporalA == false);
    CHECK(params.nonTemporalB == false);
}

TEST_CASE("CommandParameters: nonTemporalA and nonTemporalB are independent",
          "[non-temporal-loads]")
{
    CommandParameters p1;
    p1.nonTemporalA = true;
    CHECK(p1.nonTemporalA == true);
    CHECK(p1.nonTemporalB == false);

    CommandParameters p2;
    p2.nonTemporalB = true;
    CHECK(p2.nonTemporalA == false);
    CHECK(p2.nonTemporalB == true);
}

TEST_CASE("CommandParameters: toString includes nonTemporalA and nonTemporalB",
          "[non-temporal-loads]")
{
    CommandParameters params;
    params.nonTemporalA = true;
    params.nonTemporalB = false;
    auto str            = params.toString();
    CHECK_THAT(str, ContainsSubstring("nonTemporalA"));
    CHECK_THAT(str, ContainsSubstring("nonTemporalB"));
}

// ---------------------------------------------------------------------------
// Step 5 — MacroTile: bool nonTemporal field
// ---------------------------------------------------------------------------

using namespace rocRoller::KernelGraph::CoordinateGraph;
namespace KG = rocRoller::KernelGraph;

TEST_CASE("MacroTile: nonTemporal defaults to false", "[non-temporal-loads]")
{
    MacroTile tile;
    CHECK(tile.nonTemporal == false);
}

TEST_CASE("MacroTile: nonTemporal can be set independently of layoutType", "[non-temporal-loads]")
{
    MacroTile tileA;
    tileA.layoutType  = LayoutType::MATRIX_A;
    tileA.nonTemporal = true;
    CHECK(tileA.nonTemporal == true);

    MacroTile tileB;
    tileB.layoutType  = LayoutType::MATRIX_B;
    tileB.nonTemporal = false;
    CHECK(tileB.nonTemporal == false);
}

// ---------------------------------------------------------------------------
// Step 6 — SetNonTemporal transformation
// ---------------------------------------------------------------------------

// Build a minimal graph with one MATRIX_A and one MATRIX_B MacroTile,
// apply SetNonTemporal, and verify that each tile's nonTemporal flag
// matches the corresponding CommandParameters field.
static KG::KernelGraph MakeGraphWithABTiles(int& tagA, int& tagB)
{
    KG::KernelGraph graph;
    tagA = graph.coordinates.addElement(
        MacroTile({64, 64}, LayoutType::MATRIX_A, {}, MemoryType::WAVE));
    tagB = graph.coordinates.addElement(
        MacroTile({64, 64}, LayoutType::MATRIX_B, {}, MemoryType::WAVE));
    return graph;
}

TEST_CASE("SetNonTemporal: nonTemporalA=true marks MATRIX_A only", "[non-temporal-loads]")
{
    int  tagA, tagB;
    auto graph = MakeGraphWithABTiles(tagA, tagB);

    auto params          = std::make_shared<CommandParameters>();
    params->nonTemporalA = true;
    params->nonTemporalB = false;

    auto result = KG::SetNonTemporal(params).apply(graph);

    CHECK(result.coordinates.get<MacroTile>(tagA)->nonTemporal == true);
    CHECK(result.coordinates.get<MacroTile>(tagB)->nonTemporal == false);
}

TEST_CASE("SetNonTemporal: nonTemporalB=true marks MATRIX_B only", "[non-temporal-loads]")
{
    int  tagA, tagB;
    auto graph = MakeGraphWithABTiles(tagA, tagB);

    auto params          = std::make_shared<CommandParameters>();
    params->nonTemporalA = false;
    params->nonTemporalB = true;

    auto result = KG::SetNonTemporal(params).apply(graph);

    CHECK(result.coordinates.get<MacroTile>(tagA)->nonTemporal == false);
    CHECK(result.coordinates.get<MacroTile>(tagB)->nonTemporal == true);
}

TEST_CASE("SetNonTemporal: both false leaves all tiles unchanged", "[non-temporal-loads]")
{
    int  tagA, tagB;
    auto graph = MakeGraphWithABTiles(tagA, tagB);

    auto params          = std::make_shared<CommandParameters>();
    params->nonTemporalA = false;
    params->nonTemporalB = false;

    auto result = KG::SetNonTemporal(params).apply(graph);

    CHECK(result.coordinates.get<MacroTile>(tagA)->nonTemporal == false);
    CHECK(result.coordinates.get<MacroTile>(tagB)->nonTemporal == false);
}

// ---------------------------------------------------------------------------
// Step 7 — Assembly inspection (translate-time, no GPU)
// Three load paths: BufferToVGPR (7a), BufferToLDSViaVGPR (7b), BufferToLDS (7c)
// ---------------------------------------------------------------------------

// Helper: build and generate a GEMM kernel with given load paths and nt flags,
// return the normalized assembly string.
static std::string GemmAssembly(SolutionParams::LoadPath loadPathA,
                                SolutionParams::LoadPath loadPathB,
                                bool                     ntA,
                                bool                     ntB)
{
    auto context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto problem      = example.getProblem();
    problem.loadPathA = loadPathA;
    problem.loadPathB = loadPathB;
    problem.macM      = 64;
    problem.macN      = 64;
    problem.macK      = 64;
    example.setProblem(problem);

    auto command         = example.getCommand();
    auto params          = example.getCommandParameters();
    params->nonTemporalA = ntA;
    params->nonTemporalB = ntB;
    CommandKernel commandKernel(command, context.KernelName());
    commandKernel.setContext(context.get());
    commandKernel.setCommandParameters(params);
    commandKernel.generateKernel();

    return NormalizedSource(context.output());
}

// --- Sub-case 7a: BufferToVGPR (hits loadMacroTileWAVE) ---

TEST_CASE("loadMacroTileWAVE: nonTemporalA=true emits nt on buffer_load, BufferToVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToVGPR,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/true,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

TEST_CASE("loadMacroTileWAVE: nonTemporalA=false emits no nt, BufferToVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToVGPR,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/false,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") == std::string::npos);
}

TEST_CASE("loadMacroTileWAVE: nonTemporalB=true emits nt on buffer_load, BufferToVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToVGPR,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/false,
                             /*ntB=*/true);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

// --- Sub-case 7b: BufferToLDSViaVGPR (hits loadMacroTileVGPR for global→staging step) ---

TEST_CASE("loadMacroTileVGPR: nonTemporalA=true emits nt on buffer_load, BufferToLDSViaVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             /*ntA=*/true,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

TEST_CASE("loadMacroTileVGPR: nonTemporalA=false emits no nt, BufferToLDSViaVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             /*ntA=*/false,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") == std::string::npos);
}

TEST_CASE("loadMacroTileVGPR: nonTemporalB=true emits nt on buffer_load, BufferToLDSViaVGPR",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             SolutionParams::LoadPath::BufferToLDSViaVGPR,
                             /*ntA=*/false,
                             /*ntB=*/true);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

// --- Sub-case 7c: BufferToLDS (hits loadMacroTileDirect2LDS) ---

TEST_CASE("loadMacroTileDirect2LDS: nonTemporalA=true emits nt on buffer_load, BufferToLDS",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDS,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/true,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

TEST_CASE("loadMacroTileDirect2LDS: nonTemporalA=false emits no nt, BufferToLDS",
          "[non-temporal-loads]")
{
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDS,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/false,
                             /*ntB=*/false);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") == std::string::npos);
}

TEST_CASE(
    "loadMacroTileDirect2LDS: nonTemporalB=true emits nt on buffer_load, BufferToLDS+VGPRforB",
    "[non-temporal-loads]")
{
    // A uses Direct2LDS, B uses BufferToVGPR; only B is nonTemporal.
    auto asm_ = GemmAssembly(SolutionParams::LoadPath::BufferToLDS,
                             SolutionParams::LoadPath::BufferToVGPR,
                             /*ntA=*/false,
                             /*ntB=*/true);
    CHECK(asm_.find("buffer_load") != std::string::npos);
    CHECK(asm_.find(" nt") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Step 7d — GlobalToVGPR (WAVE_FROM_GLOBAL) + nonTemporal=true must AssertFatal
//
// The nt modifier only applies to buffer_load instructions. The WAVE_FROM_GLOBAL
// path emits global_load instead, so SetNonTemporal + generateKernel() must abort.
// ---------------------------------------------------------------------------

TEST_CASE("loadMacroTileWAVE: nonTemporalA=true with GlobalToVGPR asserts at code generation",
          "[non-temporal-loads]")
{
    auto context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto problem      = example.getProblem();
    problem.loadPathA = SolutionParams::LoadPath::GlobalToVGPR;
    problem.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
    problem.macM      = 64;
    problem.macN      = 64;
    problem.macK      = 64;
    example.setProblem(problem);

    auto command         = example.getCommand();
    auto params          = example.getCommandParameters();
    params->nonTemporalA = true;
    params->nonTemporalB = false;

    CommandKernel commandKernel(command, context.KernelName());
    commandKernel.setContext(context.get());
    commandKernel.setCommandParameters(params);

    CHECK_THROWS_AS(commandKernel.generateKernel(), FatalError);
}

// ---------------------------------------------------------------------------
// Step 7e — non-GFX950 silencing: nonTemporalA/B flags are cleared with a
// warning when the target architecture does not support the nt modifier.
// ---------------------------------------------------------------------------

TEST_CASE("nonTemporalA=true is silently cleared on non-GFX950 targets", "[non-temporal-loads]")
{
    auto context = TestContext::ForTarget({GPUArchitectureGFX::GFX942});
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto problem      = example.getProblem();
    problem.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
    problem.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
    problem.macM      = 64;
    problem.macN      = 64;
    problem.macK      = 64;
    example.setProblem(problem);

    auto command         = example.getCommand();
    auto params          = example.getCommandParameters();
    params->nonTemporalA = true;

    CommandKernel commandKernel(command, context.KernelName());
    commandKernel.setContext(context.get());
    commandKernel.setCommandParameters(params);
    commandKernel.generateKernel();

    // nt modifier must be absent: flag is silently cleared for non-GFX950.
    CHECK(NormalizedSource(context.output()).find(" nt") == std::string::npos);
}

// ---------------------------------------------------------------------------
// Helpers for step 8 GPU tests (also used by the debug count test below)
// ---------------------------------------------------------------------------

struct GEMMRunResult
{
    std::string        assembly;
    std::vector<float> d;
};

// Count the number of lines in `assembly` that contain both "buffer_load" and " nt".
static int CountBufferLoadNT(std::string const& assembly)
{
    std::istringstream stream(assembly);
    std::string        line;
    int                count = 0;
    while(std::getline(stream, line))
    {
        if(line.find("buffer_load") != std::string::npos
           && line.find(" nt") != std::string::npos)
            ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Step 8 — GPU correctness (requires GFX950; submit via: sbatch --constraint=GFX950 ...)
//
// For each load path:
//   - Reference kernel (ntA=false, ntB=false): verify NO "nt" in assembly (including
//     C-tile loads, which must never receive the modifier), then launch and capture D.
//   - NT kernel (ntA=true or ntB=true): verify "nt" IS present in assembly, then
//     launch and capture D.
//   - Compare D outputs: nt modifier must be transparent to numerical correctness.
// ---------------------------------------------------------------------------

// Generate, assemble-check, and run one GEMM variant on real hardware.
// Uses its own TestContext so .output() contains only this kernel's assembly.
static GEMMRunResult RunGEMMOnDevice(SolutionParams::LoadPath loadPathA,
                                     SolutionParams::LoadPath loadPathB,
                                     bool                     ntA,
                                     bool                     ntB)
{
    auto context = TestContext::ForTestDevice();

    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto problem      = example.getProblem();
    problem.loadPathA = loadPathA;
    problem.loadPathB = loadPathB;
    problem.macM      = 64;
    problem.macN      = 64;
    problem.macK      = 64;
    example.setProblem(problem);

    auto command         = example.getCommand();
    auto params          = example.getCommandParameters();
    params->nonTemporalA = ntA;
    params->nonTemporalB = ntB;

    CommandKernel commandKernel(command, context.KernelName(ntA, ntB));
    commandKernel.setContext(context.get());
    commandKernel.setCommandParameters(params);
    commandKernel.generateKernel();

    auto [commandArgs, deviceA, deviceB, deviceC, deviceD]
        = example.getCommandArguments<float>();
    commandKernel.launchKernel(commandArgs.runtimeArguments());

    int                M = problem.m, N = problem.n;
    std::vector<float> result(M * N);
    CHECK_THAT(
        hipMemcpy(result.data(), deviceD.get(), M * N * sizeof(float), hipMemcpyDefault),
        HasHipSuccess(0));

    return {NormalizedSource(context.output()), std::move(result)};
}

TEST_CASE("GEMM GPU correctness: nonTemporalA=true, BufferToVGPR",
          "[non-temporal-loads][gpu]")
{
    {
        auto ctx = TestContext::ForTestDevice();
        if(ctx->targetArchitecture().target().gfx != GPUArchitectureGFX::GFX950)
            SKIP("nt modifier is only supported on GFX950");
    }

    auto ref = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToVGPR,
                               SolutionParams::LoadPath::BufferToVGPR,
                               /*ntA=*/false, /*ntB=*/false);
    auto nt  = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToVGPR,
                               SolutionParams::LoadPath::BufferToVGPR,
                               /*ntA=*/true, /*ntB=*/false);

    // BufferToVGPR uses loadMacroTileWAVE: one buffer_load_dword per lane per K-block,
    // with the inner K-loop unrolled macK/waveK = 64/2 = 32 times.
    // ref has 0 nt loads; nt kernel has exactly 32 nt loads (A tile only).
    const int expectedWaveNT = /*macK / waveK=*/ 64 / 2;
    CHECK(CountBufferLoadNT(ref.assembly) == 0);
    CHECK(CountBufferLoadNT(nt.assembly) == expectedWaveNT);

    // Numerical correctness: nt is transparent.
    CHECK(ref.d == nt.d);
}

TEST_CASE("GEMM GPU correctness: nonTemporalA=true, BufferToLDSViaVGPR",
          "[non-temporal-loads][gpu]")
{
    {
        auto ctx = TestContext::ForTestDevice();
        if(ctx->targetArchitecture().target().gfx != GPUArchitectureGFX::GFX950)
            SKIP("nt modifier is only supported on GFX950");
    }

    auto ref = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                               SolutionParams::LoadPath::BufferToLDSViaVGPR,
                               /*ntA=*/false, /*ntB=*/false);
    auto nt  = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToLDSViaVGPR,
                               SolutionParams::LoadPath::BufferToLDSViaVGPR,
                               /*ntA=*/true, /*ntB=*/false);

    // BufferToLDSViaVGPR uses loadMacroTileVGPR: the full macM×macK A tile is loaded
    // once per outer K iteration by all workgroup threads. With workgroupSize=256 threads,
    // DataType::Float, and buffer_load_dwordx4 (16 bytes = 4 floats per instruction):
    //   macM * macK / (workgroupSize * 4) = 64*64 / (256*4) = 4 nt instructions.
    // NOTE: The divisor 4 assumes buffer_load_dwordx4 packing (4 floats per instruction).
    // If the code generator chooses a narrower instruction width, this count will differ.
    const int expectedVGPRNT = (64 * 64) / (256 * 4);
    CHECK(CountBufferLoadNT(ref.assembly) == 0);
    CHECK(CountBufferLoadNT(nt.assembly) == expectedVGPRNT);

    CHECK(ref.d == nt.d);
}

TEST_CASE("GEMM GPU correctness: nonTemporalA=true, BufferToLDS",
          "[non-temporal-loads][gpu]")
{
    {
        auto ctx = TestContext::ForTestDevice();
        if(ctx->targetArchitecture().target().gfx != GPUArchitectureGFX::GFX950)
            SKIP("nt modifier is only supported on GFX950");
    }

    auto ref = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToLDS,
                               SolutionParams::LoadPath::BufferToVGPR,
                               /*ntA=*/false, /*ntB=*/false);
    auto nt  = RunGEMMOnDevice(SolutionParams::LoadPath::BufferToLDS,
                               SolutionParams::LoadPath::BufferToVGPR,
                               /*ntA=*/true, /*ntB=*/false);

    // BufferToLDS uses loadMacroTileDirect2LDS: the full macM×macK A tile is loaded
    // directly to LDS once per outer K iteration by all workgroup threads. Same formula
    // as BufferToLDSViaVGPR: macM*macK / (workgroupSize * 4) = 64*64 / (256*4) = 4.
    // NOTE: The divisor 4 assumes buffer_load_dwordx4 packing (4 floats per instruction).
    const int expectedLDSNT = (64 * 64) / (256 * 4);
    CHECK(CountBufferLoadNT(ref.assembly) == 0);
    CHECK(CountBufferLoadNT(nt.assembly) == expectedLDSNT);

    CHECK(ref.d == nt.d);
}
