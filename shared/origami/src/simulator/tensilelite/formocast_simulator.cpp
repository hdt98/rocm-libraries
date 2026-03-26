// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <origami/simulator/tensilelite/formocast_simulator.hpp>
#include <origami/math.hpp>
#include <origami/simulator/tensilelite/formocast.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cassert>
#include <iomanip>

namespace origami
{
    using math::safe_ceil_div;

    static double getPrefetchPerformance(int      grvwa,
                                         int      grvwb,
                                         int      bpeA,
                                         int      bpeB,
                                         uint32_t depthU,
                                         int      waveNum,
                                         double   MT0,
                                         double   MT1,
                                         double   math_frequency,
                                         int      numAccPerWave)
    {
        const double others = 220 + numAccPerWave * 4;

        int numGRA = MT0 * depthU * bpeA / (waveNum * 64) / grvwa;
        int numGRB = MT1 * depthU * bpeB / (waveNum * 64) / grvwb;

        //issue 2nd prefetch
        double grCycles2 = numGRA * 4 / waveNum;
        grCycles2 += numGRB * 4 / waveNum;
        return (grCycles2 + others + 1024*depthU/64) / math_frequency;
    }



    double Formocast::getLoopOverall(const MemoryAccessCosts& mem, double math, uint32_t loopCnt, double pgr) const
    {
        double loop_overall;

        if(pgr > 1 && loopCnt > 0)
            loop_overall = std::max(math, mem.mem_overall) * (loopCnt - 1) + (math);
        else
            loop_overall = std::max(math, mem.mem_overall) * loopCnt;
        return loop_overall;
    }


    void Formocast::calculateStorePerformance(double M, double N, double NumBatches, double MT0, double MT1, uint32_t GWVWD, uint32_t bpeD, const hardware_t& hw, uint32_t WGs_per_tile, uint32_t WGs_per_tile_XCD, double &store, double &store_edge) const
    {
        const double math_freq = hw.compute_clock_ghz * 1000.0;
        const double mem_freq  = hw.mem_frequency_mhz;

        double D_L1_req = 0.0;
        double D_L2_req = 0.0;
        double D_L3_req = 0.0;
        double D_L1_edge_req, D_L2_edge_req, D_L3_edge_req;
        double total_store_req1
            = simulator::calculateStoreL1Request(M, N, MT0, MT1, GWVWD, D_L1_req, D_L1_edge_req);
        double total_store_req2
            = simulator::calculateStoreL2Request(M, N, MT0, MT1, GWVWD, D_L2_req, D_L2_edge_req);
        double total_store_req3 = simulator::calculateStoreL3Request(M, N, MT0, MT1, D_L3_req, D_L3_edge_req);

        double L2WriteBandWidthPerCU = hw.L2_write_arb_eff * 128 * 16 / WGs_per_tile_XCD;
        double L3BandWidthPerCU      = hw.L3_bandwidth / WGs_per_tile;
        double HBMBandWidthPerCU     = hw.hbm_bandwidth / WGs_per_tile;
        double L1WBus                = static_cast<double>(hw.L1_write_bus_width_per_cu);
        double L2WBus                = static_cast<double>(hw.L2_write_bus_width_per_cu);
        double D_L1_clk              = D_L1_req * 64 / L1WBus;
        double D_L2_clk = D_L2_req * 64 / std::min(L2WBus, L2WriteBandWidthPerCU);
        double D_L3_clk = D_L3_req * 64 / L3BandWidthPerCU;
        // TODO: D_hbm_clk use D_L3_req.
        double D_hbm_clk     = 0 * 64 / HBMBandWidthPerCU;
        double D_L1_clk_edge = D_L1_edge_req * 64 / L1WBus;
        double D_L2_clk_edge
            = D_L2_edge_req * 64 / std::min(L2WBus, L2WriteBandWidthPerCU);
        double D_L3_clk_edge  = D_L3_edge_req * 64 / L3BandWidthPerCU;
        double D_hbm_clk_edge = 0 * 64 / HBMBandWidthPerCU;
        double D_L1_clk_total = total_store_req1 * 64 / L1WBus;
        double D_L2_clk_total
            = total_store_req2 * 64 / std::min(L2WBus, L2WriteBandWidthPerCU);
        double D_L3_clk_total  = total_store_req3 * 64 / L3BandWidthPerCU;
        double D_hbm_clk_total = 0 * 64 / HBMBandWidthPerCU;

        double store_edge_overall = ((D_L1_clk_edge + D_L2_clk_edge) / math_freq)
                                    + ((D_L3_clk_edge + D_hbm_clk_edge) / mem_freq);
        double store_non_edge_overall
            = ((D_L1_clk + D_L2_clk) / math_freq) + ((D_L3_clk + D_hbm_clk) / mem_freq);
        double store_total = ((D_L1_clk_total + D_L2_clk_total) / math_freq)
                             + ((D_L3_clk_total + D_hbm_clk_total) / mem_freq);
        // Use the max of edge/non-edge store
        store      = store_non_edge_overall;
        store_edge = store_edge_overall;
        store = (GWVWD==1) ? store*2: store;
        store = (GWVWD==2) ? store*1.5: store;

        store_edge = (GWVWD==1) ? store_edge*2: store_edge;
        store_edge = (GWVWD==2) ? store_edge*1.5: store_edge;
    }

    Formocast::L1CacheHitRate
    Formocast::computeL1CacheHitRate(const hardware_t& hw,
                                          double MT0, double MT1, uint32_t bpeA, uint32_t bpeB,
                                          int NTA, int NTB, uint32_t GRVWA, uint32_t GRVWB,
                                          bool DTVA, bool DTVB, bool isSwizzleA, bool isSwizzleB,
                                          uint32_t VWA, uint32_t VWB, bool transA, bool transB,
                                          double lda, double ldb, int NLCA, int NLCB,
                                          uint32_t threadnum, uint32_t NumWave0, uint32_t NumWave1) const
    {
        auto hr = simulator::computeL1CacheHitRate(
            static_cast<double>(hw.L1_capacity), static_cast<double>(hw.L1_cache_line_size),
            static_cast<double>(hw.L1_bus_width_per_cu),
            MT0, MT1, bpeA, bpeB, NTA, NTB, GRVWA, GRVWB, DTVA, DTVB,
            isSwizzleA, isSwizzleB, VWA, VWB, transA, transB, lda, ldb,
            NLCA, NLCB, threadnum, NumWave0, NumWave1, hw.arch == hardware_t::architecture_t::gfx942);

        L1CacheHitRate result;
        result.tile0HitRate = hr.tile0HitRate;
        result.tile1HitRate = hr.tile1HitRate;
        return result;
    }

    Formocast::L3CacheHitRate
    Formocast::computeL3CacheHitRate(double M, double N, double K, const hardware_t& hw,
                                          uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                          int N_WGs_total, int M_WGs_total, int N_WGs_per_tile, int M_WGs_per_tile) const
    {
        auto hr = simulator::computeL3CacheHitRate(
            M, N, K, static_cast<double>(hw.L3_capacity), static_cast<double>(hw.N_CU),
            bpeA, bpeB, NTA, NTB,
            N_WGs_total, M_WGs_total, N_WGs_per_tile, M_WGs_per_tile);

        L3CacheHitRate result;
        result.totalHitRate = hr.totalHitRate;
        result.tile0HitRate = hr.tile0HitRate;
        result.tile1HitRate = hr.tile1HitRate;
        return result;
    }

    double Formocast::calculateGlobalSplitUOverhead(double M, double N, double K,
                                                        double NumBatches, double GlobalSplitU,
                                                        uint32_t gsuMethod, ProblemInfo problem,
                                                        const hardware_t& hw,
                                                        uint32_t WGs_per_tile, uint32_t WGs_per_tile_XCD,
                                                        double MT0, double MT1, uint32_t numWGs, double vgprCheck,
                                                        double storeGSU) const
    {
        const double boost_freq = hw.boost_clock_ghz * 1000.0;
        double gsu_overall = 0.0;

        if(gsuMethod == 2 && GlobalSplitU > 1) //MB
        {
            gsu_overall = simulator::getMultipleBufferOverhead(
                M, N, GlobalSplitU, NumBatches,
                problem.bpeCompute, problem.bpeD, hw.hbm_bandwidth,
                static_cast<double>(hw.L1_cache_line_size), static_cast<double>(hw.N_CU),
                boost_freq, hw.mem_frequency_mhz,
                hw.L2_write_arb_eff, hw.L2_read_arb_eff,
                hw.L3_bandwidth, static_cast<double>(hw.L1_bus_width_per_cu),
                static_cast<double>(hw.L2_bus_width_per_cu),
                static_cast<double>(hw.L1_write_bus_width_per_cu),
                static_cast<double>(hw.L2_write_bus_width_per_cu)
            );
        }
        else if(gsuMethod == 3 && GlobalSplitU > 1) //MBSK
        {
            gsu_overall = simulator::getMultipleBufferSingleKernelOverhead(
                GlobalSplitU, MT0, MT1, problem.bpeCompute,
                static_cast<double>(hw.N_CU), numWGs, boost_freq,
                hw.L2_read_arb_eff, static_cast<double>(hw.L1_bus_width_per_cu),
                static_cast<double>(hw.L2_bus_width_per_cu),
                storeGSU
            );
        }

        return gsu_overall;
    }

    double Formocast::calculateLocalSplitUOverhead(double MT0, double MT1, double lsu,
                                                     uint32_t svw, uint32_t numThreads,
                                                     ProblemInfo problem,
                                                     const hardware_t& hw) const
    {
        return simulator::getLocalSplitKOverhead(MT0, MT1, lsu, svw, numThreads,
                                         problem.bpeCompute, hw.compute_clock_ghz * 1000.0);
    }

    Formocast::MemoryAccessCosts
    Formocast::calculateMemoryAccessCosts(double MT0, double MT1,
                                   const hardware_t& hw,
                                   const CacheHitRates& hr,
                                   double L2BandWidthPerCU, double L3BandWidthPerCU, double HBMBandWidthPerCU,
                                   bool isSwizzleA, bool isSwizzleB,
                                   double A_L1_req, double B_L1_req,
                                   double A_L2_req, double A_L3_req, double A_hbm_req,
                                   double B_L2_req, double B_L3_req, double B_hbm_req) const
    {
        const double math_freq = hw.compute_clock_ghz * 1000.0;
        const double mem_freq  = hw.mem_frequency_mhz;
        const double L1Bus     = static_cast<double>(hw.L1_bus_width_per_cu);
        const double L2Bus     = static_cast<double>(hw.L2_bus_width_per_cu);
        MemoryAccessCosts mem;

        double A_L1_clk = A_L1_req * 64 / L1Bus;
        double A_L2_clk;
        if(isSwizzleA)
            A_L2_clk = A_L2_req * 128 / std::min(L2BandWidthPerCU, L2Bus);
        else
            A_L2_clk = A_L2_req * 128 / std::min(L2BandWidthPerCU, L2Bus);
        double A_L3_clk = A_L3_req * 128 / L3BandWidthPerCU;
        double A_hbm_clk = A_hbm_req * 128 / HBMBandWidthPerCU;

        double B_L1_clk = B_L1_req * 64 / L1Bus;
        double B_L2_clk;
        if(isSwizzleB)
            B_L2_clk = B_L2_req * 128 / std::min(L2BandWidthPerCU, L2Bus);
        else
            B_L2_clk = B_L2_req * 128 / std::min(L2BandWidthPerCU, L2Bus);
        double B_L3_clk = B_L3_req * 128 / L3BandWidthPerCU;
        double B_hbm_clk = B_hbm_req * 128 / HBMBandWidthPerCU;

        A_L1_clk = A_L1_req * hr.A_L1_hit * 64 / L1Bus;
        A_L3_clk = A_L3_req * 64 / L3BandWidthPerCU;
        A_hbm_clk = A_hbm_req * 8 / HBMBandWidthPerCU;
        B_L1_clk = B_L1_req * hr.B_L1_hit * 64 / L1Bus;
        B_L3_clk = B_L3_req * 64 / L3BandWidthPerCU;
        B_hbm_clk = B_hbm_req * 8 / HBMBandWidthPerCU;

        double L1_overall   = (A_L1_clk + B_L1_clk) / math_freq;
        double L2_overall   = (A_L2_clk + B_L2_clk) / math_freq;
        double L3_overall   = (A_L3_clk + B_L3_clk) / mem_freq;
        double hbm_overall  = (A_hbm_clk + B_hbm_clk) / mem_freq;
        mem.mem_overall     = L1_overall + L2_overall + L3_overall + hbm_overall;

        mem.mem_l1 = L1_overall;
        mem.mem_l2 = L2_overall;//std::max(mem.mem_l1, L2_overall);
        mem.mem_l3 = L3_overall;//std::max(mem.mem_l2, L3_overall);
        mem.mem_hbm = hbm_overall;//std::max(mem.mem_l3, hbm_overall);
        mem.l1_hit = (hr.A_L1_hit * MT0 + hr.B_L1_hit * MT1) / (MT0 + MT1);
        mem.l2_hit = hr.totalL2HitRate;
        mem.l3_hit = hr.totalL3HitRate;
        //for debug
        mem.A_L1_req = A_L1_req;
        mem.B_L1_req = B_L1_req;
        mem.A_L2_req = A_L2_req;
        mem.B_L2_req = B_L2_req;

        return mem;
    }

    double Formocast::resolveOccupancy(const hardware_t& hw, double perf, double prefetch, double mathCost, double storeCost, uint32_t num_tiles, uint32_t CUOccupancy) const
    {
        if ((num_tiles > 1)  && CUOccupancy >= 2)
        {
            perf = (prefetch + mathCost)
                    + (mathCost + storeCost)
                       * (num_tiles - 1);
        }
        else
        {
            perf *= num_tiles;
            perf += 1.7*(num_tiles-1);
        }
        return perf;
    }

    bool Formocast::isBetter(ProblemInfo problem, TieBreakerInfo previousSolution) const
    {
        auto currSol = getTieBreakerInfo();

        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currSol.mt0, currSol.mt1, currSol.du, currSol.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    // NB:
    //  isBetter return
    //   True means current is better than previous
    //   False doesn't means worse, means tie (equal) --> IMPORTANT note since this would be used in std::sort
    bool Formocast::isBetter(TieBreakerInfo previousSolution, TieBreakerInfo currentSolution) const
    {
        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currentSolution.mt0, currentSolution.mt1, currentSolution.du, currentSolution.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    Formocast::TieBreakerInfo Formocast::getTieBreakerInfo() const
    {
        return perfInfo;
    }

    // NB:
    //  isBetter return
    //   True means current is better than previous
    //   False doesn't means worse, means tie (EQUAL) --> IMPORTANT note since this would be used in std::sort
    bool Formocast::isBetter(MinTieBreakerInfo previousSolution, MinTieBreakerInfo currentSolution) const
    {
        // just early return, return false means equal
        if(previousSolution == currentSolution)
            return false;

        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currentSolution.mt0, currentSolution.mt1, currentSolution.du, currentSolution.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    Formocast::MinTieBreakerInfo Formocast::getMinTieBreakerInfo() const
    {
        MinTieBreakerInfo tbInfo;

        tbInfo.mt0 = perfInfo.mt0;
        tbInfo.mt1 = perfInfo.mt1;
        tbInfo.du  = perfInfo.du;
        tbInfo.svw = perfInfo.svw;

        return tbInfo;
    }

    Formocast::PredictedPerformance
    Formocast::predictedPerformance(void) const
    {
        PredictedPerformance pp;

        // 1. Problem Dimension Calculation
        double M = problem.M;
        double N = problem.N;
        double NumBatches = problem.NumBatches;
        double K = problem.K;
        bool transA = problem.transA;
        bool transB = problem.transB;
        uint32_t bpeA    = problem.bpeA;
        uint32_t bpeB    = problem.bpeB;
        uint32_t bpeD    = problem.bpeD;
        // swizzle settings
        bool     isSwizzleA = problem.swizzleTensorA;
        bool     isSwizzleB = problem.swizzleTensorB;

        // 2. Hardware Parameter Extraction
        const hardware_t& hw = hw_;
        const double math_freq = hw.compute_clock_ghz * 1000.0;
        const double mem_freq  = hw.mem_frequency_mhz;

        // 3. Variables directly from sizeMapping

        // Basic tile and workgroup configuration
        double MT0 = sizeMapping.macroTile[0];
        double MT1 = sizeMapping.macroTile[1];
        int      WGM = sizeMapping.workGroupMapping != 0 ? sizeMapping.workGroupMapping : 1;
        int      CUOccupancy = sizeMapping.CUOccupancy;
        uint32_t depthU = sizeMapping.depthU;

        // Global split
        bool     isGSUWGMRR = sizeMapping.globalSplitUWorkGroupMappingRoundRobin;
        uint32_t gsuMethod = sizeMapping.globalAccumulation;

        // Prefetch and memory access configuration
        int      PGR = sizeMapping.PrefetchGlobalRead;

        // Wave and global read configuration
        uint32_t GRVWA = sizeMapping.grvwA;
        uint32_t GRVWB = sizeMapping.grvwB;
        uint32_t GWVWD = sizeMapping.gwvwD;
        uint32_t VWA   = sizeMapping.VectorWidthA;
        uint32_t VWB   = sizeMapping.VectorWidthB;
        uint32_t waveNum  = sizeMapping.waveNum;
        uint32_t NumWave0 = sizeMapping.waveGroup[0];
        uint32_t NumWave1 = sizeMapping.waveGroup[1];
        uint32_t NumThreads = hw.wavefront_size * waveNum;

        // Matrix instruction and VGPR configuration
        int miSize = sizeMapping.matrixInstruction[0];
        bool DTVA = sizeMapping.DirectToVgprA;
        bool DTVB = sizeMapping.DirectToVgprB;

        // NLCA/B is used for non-TN cases to calculate load requests.
        int NLCA = sizeMapping.NumLoadsCoalescedA;
        int NLCB = sizeMapping.NumLoadsCoalescedB;

        //GlobalSplitU
        uint32_t GlobalSplitU = sizeMapping.globalSplitU;
        //LocalSplitU
        int LSU = sizeMapping.LocalSplitU;

        //DirectToLdsA
        bool DirectToLdsA = sizeMapping.DirectToLdsA;
        //DirectToLdsB
        bool DirectToLdsB = sizeMapping.DirectToLdsB;

        // Clock calculation
        // TODO: No need to check minMathClock if we guarantee that MathClocksUnrolledLoop is correct.
        double math_clk = sizeMapping.MathClocksUnrolledLoop;

        // 3.1 Early terminate. FIXME: Can filter most of the solutions with an outside function.
        // FIXME: add an extra function to reject the solutions first.
        if (GlobalSplitU == 0)
        {
            // FIXME: Need to support streamK kernels.
            GlobalSplitU = 1;
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if ((M < 128 && MT0 - M >= 16) || (N < 128 && MT1 - N >= 16))
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if ((M >= 128 && MT0 - M >= 32) || (N >= 128 && MT1 - N >= 32))
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if(problem.dataType == data_type_t::BFloat16 || problem.dataType == data_type_t::Half)
        {
            if(problem.bpeA == 2 && problem.bpeB == 2)
                if (((K >= 64 && depthU <=32) || (K <= 32 && depthU > 32) || (K > 32 && depthU > K)) && NumBatches < hw.N_CU && sizeMapping.matrixInstruction[2] >= 32)
                {
                    pp.microSeconds = 9999999.9;
                    pp.hitRate = 0;
                    return pp;
                }
        }
        if(DirectToLdsA && M < MT0)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if(DirectToLdsB && N < MT1)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }

        // 4. Derived Problem/Workgroup Dimensions
        double K_AfterGSU = safe_ceil_div(static_cast<uint32_t>(K), static_cast<uint32_t>(GlobalSplitU));
        uint32_t M_WGs_total = safe_ceil_div(static_cast<uint32_t>(M), static_cast<uint32_t>(MT0));
        uint32_t N_WGs_total = safe_ceil_div(static_cast<uint32_t>(N), static_cast<uint32_t>(MT1));
        int N_WGs_per_tile_XCD = std::min((uint32_t)WGM, N_WGs_total);
        int M_WGs_per_tile_XCD
            = std::min(M_WGs_total, static_cast<uint32_t>(safe_ceil_div(static_cast<int>(hw.N_CU / hw.NUM_XCD), N_WGs_per_tile_XCD)));
        int M_WGs_per_tile = std::min(M_WGs_total, static_cast<uint32_t>(safe_ceil_div(static_cast<int>(hw.N_CU), N_WGs_per_tile_XCD)));
        int N_WGs_per_tile
            = std::min(N_WGs_total, static_cast<uint32_t>(N_WGs_per_tile_XCD * safe_ceil_div(M_WGs_per_tile, M_WGs_total)));
        uint32_t numberWGs = M_WGs_total * N_WGs_total * NumBatches * GlobalSplitU;
        uint32_t WGs_per_tile = std::min(static_cast<uint32_t>(hw.N_CU), numberWGs);
        uint32_t WGs_per_tile_XCD = safe_ceil_div(WGs_per_tile, static_cast<uint32_t>(hw.NUM_XCD));
        uint32_t num_tiles = safe_ceil_div(numberWGs, static_cast<uint32_t>(hw.N_CU));
        uint32_t loopCnt = K_AfterGSU / depthU;
        uint32_t K_tail = K_AfterGSU - (loopCnt * depthU);
        PGR = (std::floor(K_AfterGSU/depthU > 1)) ? sizeMapping.PrefetchGlobalRead : int(K_AfterGSU/depthU);
        int      PLR = (std::floor(K_AfterGSU/sizeMapping.LocalSplitU/depthU) < 1) ? 0: 1;//sizeMapping.PrefetchLocalRead;

        if (PLR == 0)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        // 5. Cache Hit Rates and Bandwidths
        CacheHitRates cache_hits;
        L1CacheHitRate l1 = computeL1CacheHitRate(hw,
                                                MT0, MT1, bpeA, bpeB,
                                                0, 0, GRVWA, GRVWB,
                                                DTVA, DTVB, isSwizzleA, isSwizzleB,
                                                VWA, VWB, transA, transB,
                                                M, N, NLCA, NLCB,
                                                NumThreads, NumWave0, NumWave1);
        L2CacheHitRate l2 = computeL2CacheHitRate(M,
                                                N,
                                                K_AfterGSU,
                                                hw,
                                                GlobalSplitU,
                                                WGM,
                                                NumBatches,
                                                bpeA,
                                                bpeB,
                                                0,
                                                0,
                                                isGSUWGMRR);
        L3CacheHitRate l3 = computeL3CacheHitRate(M, N, K, hw,
                                                bpeA, bpeB, 0, 0,
                                                N_WGs_total, M_WGs_total, N_WGs_per_tile, M_WGs_per_tile);

        cache_hits.A_L1_hit = l1.tile0HitRate;
        cache_hits.B_L1_hit = l1.tile1HitRate;
        cache_hits.A_L2_hit = l2.tile0HitRate;
        cache_hits.B_L2_hit = l2.tile1HitRate;
        cache_hits.A_L3_hit = l3.tile0HitRate;
        cache_hits.B_L3_hit = l3.tile1HitRate;
        cache_hits.totalL2HitRate = l2.totalHitRate;
        cache_hits.totalL3HitRate = l3.totalHitRate;

        // 6. Calculate Store Performance (D matrix writes)
        double store, store_edge;
        calculateStorePerformance(M, N, NumBatches, MT0, MT1, GWVWD, bpeD, hw, WGs_per_tile, WGs_per_tile_XCD, store, store_edge);

        // 7. Calculate GSU Overhead
        double storeGSU = store * 2; //FIXME: incorrect
        auto vgprUsageCheck = MT0 * MT1 / miSize / miSize;
        double gsu_overall = calculateGlobalSplitUOverhead(M, N, K, NumBatches, GlobalSplitU, gsuMethod,
                                                  problem, hw, WGs_per_tile, WGs_per_tile_XCD,
                                                  MT0, MT1, numberWGs, vgprUsageCheck, storeGSU);

        // 8. Calcupate LSU Overhead
        double lsu_overall = calculateLocalSplitUOverhead(MT0, MT1, LSU, GWVWD, NumThreads, problem, hw);

        // 9. Calculate Memory Access and Math Costs
        double L2BandWidthPerCU     = hw.L2_read_arb_eff * 128 * 16 / WGs_per_tile_XCD;
        if (L2BandWidthPerCU > hw.L2_read_arb_eff * 128 * 16 / (static_cast<double>(hw.N_CU)/hw.NUM_XCD))
            L2BandWidthPerCU = hw.L2_read_arb_eff * 128 * 16 / (static_cast<double>(hw.N_CU)/hw.NUM_XCD);
        double L3BandWidthPerCU     = hw.L3_bandwidth / WGs_per_tile;
        double HBMBandWidthPerCU    = hw.hbm_bandwidth / WGs_per_tile;

        // Calculate load requests and memory access costs before calling calculateMemoryAccessCosts
        double tcc_ea0_coalscedA;
        double tcc_ea0_coalscedB;
        const double L1CacheLS = static_cast<double>(hw.L1_cache_line_size);
        const double L1BusCU  = static_cast<double>(hw.L1_bus_width_per_cu);
        double A_L1_req = simulator::getLoadRequest(std::min(MT0, M), depthU, L1CacheLS,
                                         GRVWA, bpeA, DTVA,
                                         transA,
                                         isSwizzleA,
                                         VWA,
                                         L1BusCU,
                                         NLCA,
                                         NumWave1,
                                         tcc_ea0_coalscedA);

        double B_L1_req = simulator::getLoadRequest(std::min(MT1, N), depthU, L1CacheLS,
                                         GRVWB, bpeB, DTVB,
                                         !transB,
                                         isSwizzleB,
                                         VWB,
                                         L1BusCU,
                                         NLCB,
                                         NumWave0,
                                         tcc_ea0_coalscedB);

        double A_L2_req = A_L1_req * (1 - cache_hits.A_L1_hit) / 2 * tcc_ea0_coalscedA;
        double A_L3_req = A_L2_req * (1 - cache_hits.A_L2_hit) / tcc_ea0_coalscedA;
        double A_hbm_req = A_L3_req * (1 - cache_hits.A_L3_hit);
        double B_L2_req = B_L1_req * (1 - cache_hits.B_L1_hit) / 2 * tcc_ea0_coalscedB;
        double B_L3_req = B_L2_req * (1 - cache_hits.B_L2_hit) / tcc_ea0_coalscedB;
        double B_hbm_req = B_L3_req * (1 - cache_hits.B_L3_hit);

        MemoryAccessCosts mem_costs = calculateMemoryAccessCosts(std::min(MT0, M), std::min(MT1, N),
                                                                hw,
                                                                cache_hits,
                                                                L2BandWidthPerCU, L3BandWidthPerCU, HBMBandWidthPerCU,
                                                                isSwizzleA, isSwizzleB,
                                                                A_L1_req, B_L1_req,
                                                                A_L2_req, A_L3_req, A_hbm_req,
                                                                B_L2_req, B_L3_req, B_hbm_req);

        // 10. Calculate Prefetch Performance
        int numAccPerWave = MT0 * MT1 / waveNum / hw.wavefront_size;
        double prefetch      = getPrefetchPerformance(GRVWA,
                                                 GRVWB,
                                                 bpeA,
                                                 bpeB,
                                                 depthU,
                                                 waveNum,
                                                 MT0,
                                                 MT1,
                                                 math_freq,
                                                 numAccPerWave);
        double preLoopCost = hw.initial_cost + prefetch;

        // 11. Calculate loop Performance
        double math_overall = math_clk / math_freq;
        double loop_overall = getLoopOverall(mem_costs, math_overall, loopCnt, PGR);

        loop_overall += loopCnt*0.2;
        // 12. Aggregate Performance: pre-loop + unrolled-loop + post-loop
        double perf = preLoopCost + loop_overall + store;
        if (num_tiles > 1)
        {
            // consider edge percentage
            double edge_percentage = 0.0;
            if (M_WGs_total * MT0 > M)
            {
                edge_percentage = 1 / (double)M_WGs_total;
            }
            store = edge_percentage * store_edge + (1 - edge_percentage) * store;
            perf = preLoopCost + loop_overall + store;
        }
        else { store = std::max(store_edge, store); perf = prefetch + loop_overall + store;}

        // 13. Handle Tail Loop
        double tail_overall = 0.0;
        if (K_tail > 0)
        {
            // FIXME: need to add new opt.
            tail_overall = (mem_costs.mem_overall*K_tail/depthU + math_overall) + prefetch*2;
            perf += tail_overall;
        }

        // 14. Add LSU Reduction Part
        perf += lsu_overall;

        // 15. Apply CU Occupancy
        perf = resolveOccupancy(hw, perf, prefetch, loop_overall + tail_overall, store, num_tiles, CUOccupancy);

        // 16. Add GSU Reduction Part
        perf += gsu_overall;

        if (int(M) % int(MT0) != 0)
            perf = perf + std::max(store_edge, store);
        pp.microSeconds = perf;
        pp.hitRate = cache_hits.totalL2HitRate * 100;

        perfInfo.memory = mem_costs;
        perfInfo.math = math_overall;
        perfInfo.svw = GWVWD;
        perfInfo.perf = perf;
        perfInfo.preloop = preLoopCost;
        perfInfo.loop = loop_overall;
        perfInfo.tail = tail_overall;
        perfInfo.store = store;
        perfInfo.gsu = gsu_overall;
        perfInfo.lsu = lsu_overall;
        perfInfo.mt0 = MT0;
        perfInfo.mt1 = MT1;
        perfInfo.du = depthU;

        return pp;
    }

    Formocast::L2CacheHitRate
        Formocast::computeL2CacheHitRate(uint32_t M,
                                                   uint32_t N,
                                                   uint32_t K,
                                                   const hardware_t& hw,
                                                   uint32_t gsu,
                                                   int32_t  wgm,
                                                   uint32_t batches,
                                                   uint32_t bpeA,
                                                   uint32_t bpeB,
                                                   int32_t  NTA,
                                                   int32_t  NTB,
                                                   bool     isGSUWGMRR) const
    {
        uint32_t MT0 = sizeMapping.macroTile[0];
        uint32_t MT1 = sizeMapping.macroTile[1];
        uint32_t depthU = sizeMapping.depthU;

        // can't make these two unsigned, they could be -1 in special cases
        int XCC = sizeMapping.workGroupMappingXCC;
        int XCCG = (sizeMapping.workGroupMappingXCCGroup < 0)? static_cast<int>(hw.N_CU) : sizeMapping.workGroupMappingXCCGroup;

        auto hr = simulator::computeL2CacheHitRate(
            M, N, K, MT0, MT1, depthU,
            static_cast<uint32_t>(hw.L2_capacity), static_cast<uint32_t>(hw.N_CU),
            static_cast<uint32_t>(hw.NUM_XCD),
            XCC, XCCG, gsu, wgm, batches, bpeA, bpeB, NTA, NTB, isGSUWGMRR);

        L2CacheHitRate hitRate;
        hitRate.totalHitRate = hr.totalHitRate;
        hitRate.tile0HitRate = hr.tile0HitRate;
        hitRate.tile1HitRate = hr.tile1HitRate;

        return hitRate;
    }

    void Formocast::setProblem(ProblemInfo p)
    {
        problem = p;
        if (problem.M == 0 || problem.N == 0 || problem.K == 0)
            throw std::runtime_error(
                "Problem size is invalid");
    }

    void Formocast::setSolution(SizeMapping sm)
    {
        sizeMapping = sm;
    }

    Formocast::Formocast(const hardware_t& hw)
        : hw_(hw)
    {
    }

    int Formocast::getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead, int numWaves, bool isStall, bool isSgprOffset) const
    {
        return simulator::getGlobalReadQueueFullStallCycles(currentCycle, fifo, bpRead, numWaves, isStall, isSgprOffset);
    }

    int Formocast::getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, int numLR) const
    {
        return simulator::getLocalReadCompletionCycle(currentCycle, fifo, numLR);
    }

    int Formocast::getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall, double bankConflict) const
    {
        int lrStallLatencyBuffer;
        if (!isStall){
            lrStallLatencyBuffer = 1;
        }
        else if (bpRead == 16) {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_.local_read_latency_b128, hw_.local_read_conflict_b128, bankConflict);
        } else if (bpRead == 8) {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_.local_read_latency_b64, hw_.local_read_conflict_b64, bankConflict);
        } else {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_.local_read_latency_b32, hw_.local_read_conflict_b32, bankConflict);
        }
        return simulator::getLocalReadQueueFullStallCycles(currentCycle, fifo, bpRead, numWaves, lrStallLatencyBuffer);
    }

    int Formocast::getLocalWriteQueueFullStallCycles(int currentCycle, int previousLW, int issueCycles, int bpWrite, int numWaves) const
    {
        int issuePenality = 0;
        if(numWaves == 4)
        {
            // only 2 simds can be issued at the same time. if 4 simds, need 1 extra issue cycle.
            issuePenality = issueCycles;
        }
        if(currentCycle - previousLW >= issueCycles)
        {
            // if the space is enough, no stall.
            return currentCycle + issueCycles;
        }
        return currentCycle + issueCycles + issuePenality;
    }

    void Formocast::pushLocalReadWrite(int currentCycle, std::queue<int>& fifo, int bpr, double bankConflict, bool isLocalRead, int numPreviousLRs)
    {
        int lrMemLatency;
        if (isLocalRead) {
            if (bpr == 16) {
                lrMemLatency = simulator::getLocalReadLatency(hw_.local_read_latency_b128, hw_.local_read_conflict_b128, bankConflict);
                if (numPreviousLRs <= 4) {
                    lrMemLatency += 2 * numPreviousLRs;
                }
                else {
                    lrMemLatency += 2 * 4;
                }
            } else if (bpr == 8) {
                lrMemLatency = simulator::getLocalReadLatency(hw_.local_read_latency_b64, hw_.local_read_conflict_b64, bankConflict);
            } else {
                lrMemLatency = simulator::getLocalReadLatency(hw_.local_read_latency_b32, hw_.local_read_conflict_b32, bankConflict);
            }
        }
        else {
            if (bpr == 16) {
                lrMemLatency = simulator::getLocalWriteLatency(hw_.local_write_latency_b128, hw_.local_write_conflict_b128, bankConflict);
            } else if (bpr == 8) {
                lrMemLatency = simulator::getLocalWriteLatency(hw_.local_write_latency_b64, hw_.local_write_conflict_b64, bankConflict);
            } else {
                lrMemLatency = simulator::getLocalWriteLatency(hw_.local_write_latency_b32, hw_.local_write_conflict_b32, bankConflict);
            }
        }
        fifo.push(currentCycle + lrMemLatency);
    }

    // Standalone tie-breaker function implementation
    bool compareConfigTieBreaker(
        double M, double N, double K, size_t batch,
        double mt0_a, double mt1_a, double du_a, int svw_a,
        double mt0_b, double mt1_b, double du_b, int svw_b
    )
    {
        // Skinny N case: N <= 32 && M >= 1024 && K >= 1024
        // Prefer configurations with larger "skinny ratio" when mt1 matches N
        if (N <= 32 && M >= 1024 && K >= 1024) {
            auto skinny_a = mt0_a * du_a / M;
            auto skinny_b = mt0_b * du_b / M;
            if (mt1_a == N && mt1_b == N && skinny_a != skinny_b) {
                return skinny_a > skinny_b;
            }
        }

        // Skinny M case: M <= 32 && N >= 1024 && K >= 1024
        // Prefer configurations with larger "skinny ratio" when mt0 matches M
        if (M <= 32 && N >= 1024 && K >= 1024) {
            auto skinny_a = mt1_a * du_a / N;
            auto skinny_b = mt1_b * du_b / N;
            if (mt0_a == M && mt0_b == M && skinny_a != skinny_b) {
                return skinny_a > skinny_b;
            }
        }

        // Wide & short K case: batch == 1 && K <= 512 && M >= 1024 && N >= 1024
        // Prefer larger store vector width for better memory efficiency
        if (batch == 1 && K <= 512 && M >= 1024 && N >= 1024) {
            if (svw_a != svw_b) {
                return svw_a > svw_b;
            }
        }

        // No preference - configurations are considered equal
        return false;
    }

    // Helper function to analyze bank conflicts from VGPR states
    Formocast::BankConflictResult Formocast::analyzeBankConflictsFromVGPR(
        const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
        const std::string& vgprLocalReadAddrA,
        const std::string& vgprLocalReadAddrB,
        int LocalReadBytesA,
        int LocalReadBytesB)
    {
        BankConflictResult result;
        int NUM_THREADS_TO_SIMULATE = hw_.wavefront_size;
        int NUM_BANKS = 32;
        int BANK_WIDTH = 4;
        result.ratioA = simulator::analyzeBankConflictsFromVGPR(vgprState, vgprLocalReadAddrA, NUM_THREADS_TO_SIMULATE, NUM_BANKS, BANK_WIDTH, LocalReadBytesA);
        result.ratioB = simulator::analyzeBankConflictsFromVGPR(vgprState, vgprLocalReadAddrB, NUM_THREADS_TO_SIMULATE, NUM_BANKS, BANK_WIDTH, LocalReadBytesB);
        return result;
    }
} // namespace origami
