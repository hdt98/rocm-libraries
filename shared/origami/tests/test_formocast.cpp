// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>
#include <origami/hardware.hpp>
#include <queue>

using Catch::Approx;
using namespace origami;

// Construct a hardware_t matching the old Formocast byte-blob values for a given
// architecture.  The compute_clock and memory_clock are the model-tuned values
// that the legacy blobs encoded (they are NOT necessarily the physical clocks).
static hardware_t make_test_hardware(hardware_t::architecture_t arch)
{
    auto constants = hardware_t::get_arch_constants(arch);

    size_t N_CU = 0, L2_cap = 0, lds = 65536;
    double compute_ghz = 0.0, mem_ghz = 0.0;

    switch (arch) {
        case hardware_t::architecture_t::gfx950:
            N_CU = 256; L2_cap = 4194304;
            compute_ghz = 1.8;  mem_ghz = 1.9;
            break;
        case hardware_t::architecture_t::gfx942:
            N_CU = 304; L2_cap = 4194304;
            compute_ghz = 1.1;  mem_ghz = 1.3;
            break;
        case hardware_t::architecture_t::gfx1201:
            N_CU = 64;  L2_cap = 8388608;
            compute_ghz = 2.35; mem_ghz = 1.258;
            break;
        default:
            throw std::runtime_error("Unsupported test architecture");
    }

    return hardware_t(arch, N_CU, lds, constants, L2_cap, compute_ghz, mem_ghz);
}

// Helper function to create a ProblemInfo
Formocast::ProblemInfo make_problem_info(double M, double N, double K,
                                         uint32_t bpeA = 2, uint32_t bpeB = 2,
                                         uint32_t bpeD = 2, uint32_t bpeCompute = 2,
                                         bool transA = true, bool transB = false,
                                         double numBatches = 1,
                                         data_type_t dataType = data_type_t::BFloat16)
{
    Formocast::ProblemInfo problem;
    problem.M = M;
    problem.N = N;
    problem.K = K;
    problem.NumBatches = numBatches;
    problem.bpeA = bpeA;
    problem.bpeB = bpeB;
    problem.bpeD = bpeD;
    problem.bpeCompute = bpeCompute;
    problem.transA = transA;
    problem.transB = transB;
    problem.swizzleTensorA = false;
    problem.swizzleTensorB = false;
    problem.dataType = dataType;
    return problem;
}

// Helper function to create a SizeMapping
Formocast::SizeMapping make_size_mapping(int MT0 = 128, int MT1 = 128, int depthU = 32,
                                         int miM = 32, int miN = 32, int miK = 8,
                                         int waveNum = 4, int globalSplitU = 1)
{
    Formocast::SizeMapping mapping;
    mapping.macroTile[0] = MT0;
    mapping.macroTile[1] = MT1;
    mapping.macroTile[2] = 1;
    mapping.matrixInstruction[0] = miM;
    mapping.matrixInstruction[1] = miN;
    mapping.matrixInstruction[2] = miK;
    mapping.matrixInstruction[3] = 1;
    mapping.depthU = depthU;
    mapping.waveNum = waveNum;
    mapping.globalSplitU = globalSplitU;
    mapping.workGroupMapping = 8;
    mapping.CUOccupancy = 2;
    mapping.PrefetchGlobalRead = 2;
    mapping.MathClocksUnrolledLoop = 2048;
    mapping.grvwA = 4;
    mapping.grvwB = 4;
    mapping.gwvwD = 4;
    mapping.VectorWidthA = 4;
    mapping.VectorWidthB = 4;
    mapping.LocalSplitU = 1;
    mapping.DirectToVgprA = false;
    mapping.DirectToVgprB = false;
    mapping.DirectToLdsA = false;
    mapping.DirectToLdsB = false;
    mapping.NumLoadsCoalescedA = 1;
    mapping.NumLoadsCoalescedB = 1;
    mapping.waveGroup[0] = 2;
    mapping.waveGroup[1] = 2;
    mapping.globalAccumulation = 0;
    mapping.globalSplitUCoalesced = false;
    mapping.globalSplitUWorkGroupMappingRoundRobin = false;
    return mapping;
}

TEST_CASE("Formocast: Hardware setup via hardware_t", "[formocast]") {
    SECTION("gfx950 hardware_t is valid") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        REQUIRE(hw.arch == hardware_t::architecture_t::gfx950);
        REQUIRE(hw.N_CU == 256);
    }

    SECTION("gfx942 hardware_t is valid") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx942);
        REQUIRE(hw.arch == hardware_t::architecture_t::gfx942);
        REQUIRE(hw.N_CU == 304);
    }

    SECTION("gfx1201 hardware_t is valid") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx1201);
        REQUIRE(hw.arch == hardware_t::architecture_t::gfx1201);
        REQUIRE(hw.N_CU == 64);
    }
}

TEST_CASE("Formocast: Basic problem and solution setup", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Set valid problem") {
        auto problem = make_problem_info(1024, 1024, 1024);
        REQUIRE_NOTHROW(simulator.setProblem(problem));
        REQUIRE(simulator.problem.M == 1024);
        REQUIRE(simulator.problem.N == 1024);
        REQUIRE(simulator.problem.K == 1024);
    }

    SECTION("Set problem with zero dimension throws exception") {
        auto problem = make_problem_info(0, 1024, 1024);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);

        problem = make_problem_info(1024, 0, 1024);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);

        problem = make_problem_info(1024, 1024, 0);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);
    }

    SECTION("Set solution mapping") {
        auto mapping = make_size_mapping();
        REQUIRE_NOTHROW(simulator.setSolution(mapping));
        REQUIRE(simulator.sizeMapping.macroTile[0] == 128);
        REQUIRE(simulator.sizeMapping.macroTile[1] == 128);
        REQUIRE(simulator.sizeMapping.depthU == 32);
    }
}

TEST_CASE("Formocast: Performance prediction", "[formocast]") {

    SECTION("Basic performance prediction for square problem") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping(128, 128, 32);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds > 0);
        REQUIRE(perf.microSeconds < 1000000);
        REQUIRE(perf.hitRate >= 0);
        REQUIRE(perf.hitRate <= 100);
    }

    SECTION("Performance prediction for rectangular problem") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx942);
        Formocast simulator(hw);
        auto problem = make_problem_info(2048, 512, 1024);
        auto mapping = make_size_mapping(256, 64, 32);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds == Approx(68.640434).epsilon(0.01));
        REQUIRE(perf.hitRate == Approx(70.0).epsilon(0.01));
    }

    SECTION("Performance prediction with batched problem") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(512, 512, 512, 2, 2, 2, 2, true, false, 4);
        auto mapping = make_size_mapping(128, 128, 32);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds == Approx(23.165081).epsilon(0.01));
        REQUIRE(perf.hitRate == Approx(62.5).epsilon(0.01));
    }

    SECTION("Performance prediction with GlobalSplitU") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(1024, 1024, 2048);
        auto mapping = make_size_mapping(128, 128, 32, 32, 32, 8, 4, 2);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds > 0);
    }
}

TEST_CASE("Formocast: Cache hit rate computation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("L1 cache hit rate computation") {
        auto l1_hit = simulator.computeL1CacheHitRate(
            hw, 128, 128, 2, 2, 0, 0, 4, 4,
            false, false, false, false,
            4, 4, true, false,
            1024, 1024, 1, 1,
            256, 2, 2
        );

        REQUIRE(l1_hit.tile0HitRate == Approx(0.5).epsilon(0.01));
        REQUIRE(l1_hit.tile1HitRate == Approx(0.5).epsilon(0.01));
    }

    SECTION("L2 cache hit rate computation") {
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping();

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto l2_hit = simulator.computeL2CacheHitRate(
            1024, 1024, 1024, hw, 1, 8, 1, 2, 2, 0, 0, false
        );

        REQUIRE(l2_hit.totalHitRate == Approx(0.4375).epsilon(0.01));
        REQUIRE(l2_hit.tile0HitRate == Approx(0.875).epsilon(0.01));
        REQUIRE(l2_hit.tile1HitRate == Approx(0.0).epsilon(0.01));
    }

    SECTION("L3 cache hit rate computation") {
        auto l3_hit = simulator.computeL3CacheHitRate(
            1024, 1024, 1024, hw, 2, 2, 0, 0, 8, 8, 8, 8
        );

        REQUIRE(l3_hit.totalHitRate == Approx(0.875).epsilon(0.01));
        REQUIRE(l3_hit.tile0HitRate == Approx(0.875).epsilon(0.01));
        REQUIRE(l3_hit.tile1HitRate == Approx(0.875).epsilon(0.01));
    }
}

TEST_CASE("Formocast: Store performance calculation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);
    double store, store_edge;

    SECTION("Calculate store performance with edge case") {
        simulator.calculateStorePerformance(
            1000, 1000, 1, 128, 128, 4, 2, hw, 304, 38, store, store_edge
        );

        REQUIRE(store == Approx(2.75062).epsilon(0.01));
        REQUIRE(store_edge == Approx(7.01728).epsilon(0.01));
    }

    SECTION("Calculate store performance with different GWVWD") {
        double store1, store_edge1, store2, store_edge2;

        simulator.calculateStorePerformance(
            1024, 1024, 1, 128, 128, 1, 2, hw, 304, 38, store1, store_edge1
        );

        simulator.calculateStorePerformance(
            1024, 1024, 1, 128, 128, 4, 2, hw, 304, 38, store2, store_edge2
        );

        // GWVWD=1 should have higher cost
        REQUIRE(store1 > store2);

        REQUIRE(store1 == Approx(6.259750).epsilon(0.01));
        REQUIRE(store2 == Approx(2.679505).epsilon(0.01));
    }
}

TEST_CASE("Formocast: Tie-breaker comparison", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);
    auto problem = make_problem_info(1024, 1024, 1024);
    simulator.setProblem(problem);

    SECTION("Standalone tie-breaker function for skinny N") {
        bool result = compareConfigTieBreaker(
            1024, 32, 1024, 1,
            128, 32, 64, 4,
            64, 32, 64, 4
        );

        REQUIRE(result == true);
    }

    SECTION("Standalone tie-breaker function for skinny M") {
        bool result = compareConfigTieBreaker(
            32, 1024, 1024, 1,
            32, 128, 64, 4,
            32, 64, 64, 4
        );

        REQUIRE(result == true);
    }

    SECTION("Standalone tie-breaker function for wide & short K") {
        bool result = compareConfigTieBreaker(
            1024, 1024, 512, 1,
            128, 128, 32, 8,
            128, 128, 32, 4
        );

        REQUIRE(result == true);
    }

    SECTION("Tie-breaker returns false for equal configs") {
        bool result = compareConfigTieBreaker(
            1024, 1024, 1024, 1,
            128, 128, 32, 4,
            128, 128, 32, 4
        );

        REQUIRE(result == false);
    }
}

TEST_CASE("Formocast: GSU overhead calculation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);
    auto problem = make_problem_info(1024, 1024, 1024);
    auto mapping = make_size_mapping();

    simulator.setProblem(problem);
    simulator.setSolution(mapping);

    SECTION("GSU overhead with MultipleBuffer method") {
        double gsu_overhead = simulator.calculateGlobalSplitUOverhead(
            1024, 1024, 1024, 1, 2, 2,
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );

        REQUIRE(gsu_overhead == Approx(1.979094).epsilon(0.01));
    }

    SECTION("GSU overhead with MBSK method") {
        double gsu_overhead = simulator.calculateGlobalSplitUOverhead(
            1024, 1024, 1024, 1, 2, 3,
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );

        REQUIRE(gsu_overhead == Approx(2.155789).epsilon(0.01));
    }

    SECTION("No GSU overhead when GlobalSplitU=1") {
        double gsu_overhead = simulator.calculateGlobalSplitUOverhead(
            1024, 1024, 1024, 1, 1, 2,
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );

        REQUIRE(gsu_overhead == 0);
    }

    SECTION("GSU overhead with small M and N values") {
        double gsu_overhead = simulator.calculateGlobalSplitUOverhead(
            2, 4, 1024, 1, 2, 2,
            problem, hw, 304, 38, 32, 32, 256, 1.0, 1.0
        );

        REQUIRE(gsu_overhead == Approx(0.0000151).epsilon(0.01));
    }
}

TEST_CASE("Formocast: LSU overhead calculation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);
    auto problem = make_problem_info(1024, 1024, 1024);

    simulator.setProblem(problem);

    SECTION("LSU overhead calculation") {
        double lsu_overhead = simulator.calculateLocalSplitUOverhead(
            128, 128, 2, 4, 256, problem, hw
        );

        REQUIRE(lsu_overhead == Approx(1.066667).epsilon(0.01));
    }

    SECTION("LSU overhead increases with larger LSU value") {
        double lsu_overhead1 = simulator.calculateLocalSplitUOverhead(
            128, 128, 2, 4, 256, problem, hw
        );

        double lsu_overhead2 = simulator.calculateLocalSplitUOverhead(
            128, 128, 4, 4, 256, problem, hw
        );

        REQUIRE(lsu_overhead2 > lsu_overhead1);

        REQUIRE(lsu_overhead1 == Approx(1.066667).epsilon(0.01));
        REQUIRE(lsu_overhead2 == Approx(1.351111).epsilon(0.01));
    }
}

TEST_CASE("Formocast: FIFO queue operations", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Check global read FIFO full - no stall") {
        std::deque<int> fifo;
        int result = simulator.getGlobalReadQueueFullStallCycles(100, fifo, 8, 4, false, false);
        REQUIRE(result == 100);
        REQUIRE(fifo.size() == 0);
    }

    SECTION("Check global read FIFO full - with stall") {
        std::deque<int> fifo;
        fifo.push_back(10);
        fifo.push_back(11);
        fifo.push_back(12);
        fifo.push_back(13);
        fifo.push_back(15);
        fifo.push_back(16);
        fifo.push_back(17);
        fifo.push_back(18);
        fifo.push_back(20);
        fifo.push_back(21);
        fifo.push_back(22);
        fifo.push_back(23);
        fifo.push_back(25);
        fifo.push_back(26);
        fifo.push_back(27);
        fifo.push_back(28);
        // stall
        int result = simulator.getGlobalReadQueueFullStallCycles(29, fifo, 8, 4, true, false);
        REQUIRE(result == 32);
        REQUIRE(fifo.size() == 20);
    }

    SECTION("Check local read FIFO full - no stall") {
        std::queue<int> fifo;
        // no stall
        int result = simulator.getLocalReadQueueFullStallCycles(100, fifo, 8, 4, false, 1.0);
        REQUIRE(result == 100);
        REQUIRE(fifo.size() == 1);
    }

    SECTION("Check local read FIFO full - with bank conflict") {
        std::queue<int> fifo;
        fifo.push(10);
        fifo.push(20);
        fifo.push(30);
        fifo.push(40);
        // no stall
        int result = simulator.getLocalReadQueueFullStallCycles(100, fifo, 8, 4, true, 1.5);
        REQUIRE(result == 100);
        REQUIRE(fifo.size() == 4);
    }

    SECTION("Check local read completion - queue not full") {
        std::queue<int> fifo;
        fifo.push(50);
        fifo.push(60);
        int result = simulator.getLocalReadCompletionCycle(100, fifo, 2);
        REQUIRE(result == 100);
        REQUIRE(fifo.size() == 2);
    }

    SECTION("Check local read completion - remove finished reads") {
        std::queue<int> fifo;
        fifo.push(50);
        fifo.push(60);
        fifo.push(70);
        int result = simulator.getLocalReadCompletionCycle(100, fifo, 2);
        REQUIRE(result == 100);
        REQUIRE(fifo.size() == 2);
        REQUIRE(fifo.front() == 60);
    }

    SECTION("Check local read completion - wait for unfinished reads") {
        std::queue<int> fifo;
        fifo.push(110);
        fifo.push(120);
        fifo.push(130);
        int result = simulator.getLocalReadCompletionCycle(100, fifo, 1);
        REQUIRE(result == 120);
        REQUIRE(fifo.size() == 1);
    }

    SECTION("Push local read write with bank conflict") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 8, 1.5, true, 0);

        REQUIRE(fifo.size() == 1);
        REQUIRE(fifo.front() == 111); // 100 + 11
    }

    SECTION("Push local read write without bank conflict") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 8, 1.0, true, 0);

        REQUIRE(fifo.size() == 1);
        REQUIRE(fifo.front() == 110); // 100 + 10
    }

    SECTION("Push local write without bank conflict - bpr=8") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 8, 1.0, false, 0);

        REQUIRE(fifo.size() == 1);
        // For gfx950: LocalWriteBaseLatencyB64=10, LocalWriteConflictMultiplierB64=2
        // latency = 10 + 1.0 * 2 = 12
        REQUIRE(fifo.front() == 112); // 100 + 12
    }

    SECTION("Push local write with bank conflict - bpr=8") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 8, 1.5, false, 0);

        REQUIRE(fifo.size() == 1);
        // latency = 10 + 1.5 * 2 = 13
        REQUIRE(fifo.front() == 113); // 100 + 13
    }

    SECTION("Push local write without bank conflict - bpr=16") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 16, 1.0, false, 0);

        REQUIRE(fifo.size() == 1);
        // For gfx950: LocalWriteBaseLatencyB128=10, LocalWriteConflictMultiplierB128=4
        // latency = 10 + 1.0 * 4 = 14
        REQUIRE(fifo.front() == 114); // 100 + 14
    }

    SECTION("Push local write with bank conflict - bpr=16") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 16, 1.5, false, 0);

        REQUIRE(fifo.size() == 1);
        // latency = 10 + 1.5 * 4 = 16
        REQUIRE(fifo.front() == 116); // 100 + 16
    }

    SECTION("Push local write without bank conflict - bpr=4") {
        std::queue<int> fifo;

        simulator.pushLocalReadWrite(100, fifo, 4, 1.0, false, 0);

        REQUIRE(fifo.size() == 1);
        // For gfx950: LocalWriteBaseLatencyB32=10, LocalWriteConflictMultiplierB32=1
        // latency = 10 + 1.0 * 1 = 11
        REQUIRE(fifo.front() == 111); // 100 + 11
    }

    SECTION("Get local write queue full stall cycles - numWaves != 4") {
        int result1 = simulator.getLocalWriteQueueFullStallCycles(100, 50, 3, 8, 2);
        REQUIRE(result1 == 103);

        int result2 = simulator.getLocalWriteQueueFullStallCycles(100, 70, 3, 8, 2);
        REQUIRE(result2 == 103);

        int result3 = simulator.getLocalWriteQueueFullStallCycles(100, 99, 3, 8, 2);
        REQUIRE(result3 == 103);
    }

    SECTION("Get local write queue full stall cycles - numWaves == 4, bpWrite == 16") {
        int result1 = simulator.getLocalWriteQueueFullStallCycles(100, 50, 5, 16, 4);
        REQUIRE(result1 == 105);

        int result2 = simulator.getLocalWriteQueueFullStallCycles(100, 99, 5, 16, 4);
        REQUIRE(result2 == 110);
    }

    SECTION("Get local write queue full stall cycles - edge cases") {
        int result1 = simulator.getLocalWriteQueueFullStallCycles(100, 100, 10, 8, 2);
        REQUIRE(result1 == 110);
    }
}

TEST_CASE("Formocast: Bank conflict analysis", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Analyze bank conflicts from VGPR state") {
        std::vector<std::unordered_map<std::string, int64_t>> vgprState(64);

        for (int i = 0; i < 64; i++) {
            vgprState[i]["vgprLocalReadAddrA"] = i * 32;
            vgprState[i]["vgprLocalReadAddrB"] = i * 16;
        }

        auto result = simulator.analyzeBankConflictsFromVGPR(
            vgprState, "vgprLocalReadAddrA", "vgprLocalReadAddrB", 4, 8
        );

        REQUIRE(result.ratioA == Approx(8.0).epsilon(0.01));
        REQUIRE(result.ratioB == Approx(2.0).epsilon(0.01));
    }
}

TEST_CASE("Formocast: Memory access costs calculation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Calculate memory access costs") {
        Formocast::CacheHitRates hr;
        hr.A_L1_hit = 0.9;
        hr.B_L1_hit = 0.9;
        hr.A_L2_hit = 0.7;
        hr.B_L2_hit = 0.7;
        hr.A_L3_hit = 0.5;
        hr.B_L3_hit = 0.5;
        hr.totalL2HitRate = 0.8;
        hr.totalL3HitRate = 0.6;

        auto mem_costs = simulator.calculateMemoryAccessCosts(
            128, 128, hw, hr,
            1000.0, 800.0, 600.0,
            false, false,
            100.0, 100.0,
            50.0, 25.0, 10.0,
            50.0, 25.0, 10.0
        );

        REQUIRE(mem_costs.mem_overall == Approx(0.157801).epsilon(0.0001));
        REQUIRE(mem_costs.mem_l1 == Approx(0.1).epsilon(0.0001));
        REQUIRE(mem_costs.mem_l2 == Approx(0.0555556).epsilon(0.0001));
        REQUIRE(mem_costs.mem_l3 == Approx(0.00210526).epsilon(0.0001));
        REQUIRE(mem_costs.mem_hbm == Approx(0.000140351).epsilon(0.0001));
        REQUIRE(mem_costs.l1_hit == Approx(0.9).epsilon(0.0001));
    }
}

TEST_CASE("Formocast: Loop overall calculation", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Loop overall with prefetch") {
        Formocast::MemoryAccessCosts mem;
        mem.mem_overall = 10.0;

        double loop = simulator.getLoopOverall(mem, 8.0, 10, 2.0);

        REQUIRE(loop == Approx(98.0).epsilon(0.0001));
    }

    SECTION("Loop overall without prefetch") {
        Formocast::MemoryAccessCosts mem;
        mem.mem_overall = 10.0;

        double loop = simulator.getLoopOverall(mem, 8.0, 10, 1.0);

        REQUIRE(loop == Approx(100.0).epsilon(0.0001));
    }
}

TEST_CASE("Formocast: Occupancy resolution", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    SECTION("Occupancy comparison: occupancy=2 perf should be <= occupancy=1 perf") {
        double perf_occ1 = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 2, 1);
        double perf_occ2 = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 2, 2);

        REQUIRE(perf_occ2 <= perf_occ1);

        REQUIRE(perf_occ1 == Approx(201.7).epsilon(0.01));
        REQUIRE(perf_occ2 == Approx(130.0).epsilon(0.01));
    }

    SECTION("NumTiles comparison: numTile=1 perf should be <= numTile=2 perf") {
        double perf_tile1 = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 1, 1);
        double perf_tile2 = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 2, 1);

        REQUIRE(perf_tile1 <= perf_tile2);

        REQUIRE(perf_tile1 == Approx(100.0).epsilon(0.01));
        REQUIRE(perf_tile2 == Approx(201.7).epsilon(0.01));
    }
}

TEST_CASE("Formocast: Tie-breaker info", "[formocast]") {
    auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
    Formocast simulator(hw);

    auto problem = make_problem_info(1024, 1024, 1024);
    auto mapping = make_size_mapping();

    simulator.setProblem(problem);
    simulator.setSolution(mapping);

    // Run prediction to populate perfInfo
    simulator.predictedPerformance();

    SECTION("Get tie-breaker info") {
        auto tbInfo = simulator.getTieBreakerInfo();

        REQUIRE(tbInfo.mt0 > 0);
        REQUIRE(tbInfo.mt1 > 0);
        REQUIRE(tbInfo.du > 0);
        REQUIRE(tbInfo.perf > 0);
    }

    SECTION("Get min tie-breaker info") {
        auto minTbInfo = simulator.getMinTieBreakerInfo();

        REQUIRE(minTbInfo.mt0 > 0);
        REQUIRE(minTbInfo.mt1 > 0);
        REQUIRE(minTbInfo.du > 0);
    }

    SECTION("Compare tie-breaker configs") {
        Formocast::MinTieBreakerInfo config1;
        config1.mt0 = 128;
        config1.mt1 = 128;
        config1.du = 32;
        config1.svw = 4;

        Formocast::MinTieBreakerInfo config2;
        config2.mt0 = 64;
        config2.mt1 = 64;
        config2.du = 32;
        config2.svw = 4;

        REQUIRE(simulator.isBetter(config1, config1) == false);
    }
}

TEST_CASE("Formocast: Edge cases and error handling", "[formocast]") {
    SECTION("Very small problem size") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(32, 32, 32);
        auto mapping = make_size_mapping(128, 128, 32);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds == Approx(1e+07).epsilon(0.01));
    }

    SECTION("Very large problem size") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(8192, 8192, 8192);
        auto mapping = make_size_mapping(256, 256, 64);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds == Approx(847.421).epsilon(0.01));
    }

    SECTION("Problem with GlobalSplitU=2") {
        auto hw = make_test_hardware(hardware_t::architecture_t::gfx950);
        Formocast simulator(hw);
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping(128, 128, 32, 32, 32, 8, 4, 2);

        simulator.setProblem(problem);
        simulator.setSolution(mapping);

        auto perf = simulator.predictedPerformance();

        REQUIRE(perf.microSeconds == Approx(23.5146).epsilon(0.01));
    }
}
