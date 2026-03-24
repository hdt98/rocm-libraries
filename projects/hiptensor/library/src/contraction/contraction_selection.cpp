/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include <algorithm>
#include <numeric>

#ifndef CHECK_HIP_ALLOC
#define CHECK_HIP_ALLOC(status)               \
    if(status != hipSuccess)                  \
    {                                         \
        return HIPTENSOR_STATUS_ALLOC_FAILED; \
    }
#endif

#include <ck/stream_config.hpp>

#include "contraction_selection.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "performance.hpp"
#include "util.hpp"

namespace hiptensor
{
    hiptensorStatus_t bruteForceModel(ContractionSolution**              winner,
                                      std::vector<ContractionSolution*>& candidates,
                                      hiptensorDataType_t                typeA,
                                      std::vector<std::size_t> const&    a_ms_ks_lengths,
                                      std::vector<std::size_t> const&    a_ms_ks_strides,
                                      std::vector<int32_t> const&        a_ms_ks_modes,
                                      hiptensorDataType_t                typeB,
                                      std::vector<std::size_t> const&    b_ns_ks_lengths,
                                      std::vector<std::size_t> const&    b_ns_ks_strides,
                                      std::vector<int32_t> const&        b_ns_ks_modes,
                                      hiptensorDataType_t                typeD,
                                      std::vector<std::size_t> const&    d_ms_ns_lengths,
                                      std::vector<std::size_t> const&    d_ms_ns_strides,
                                      std::vector<int32_t> const&        d_ms_ns_modes,
                                      hiptensorDataType_t                typeE,
                                      std::vector<std::size_t> const&    e_ms_ns_lengths,
                                      std::vector<std::size_t> const&    e_ms_ns_strides,
                                      std::vector<int32_t> const&        e_ms_ns_modes,
                                      hiptensorComputeDescriptor_t       computeType,
                                      ContractionUnaryOps const&         unaryOps,
                                      const uint64_t                     workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementsFromLengths(a_ms_ks_lengths) * hiptensorDataTypeSize(typeA);
        auto sizeB = elementsFromLengths(b_ns_ks_lengths) * hiptensorDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementsFromLengths(d_ms_ns_lengths) * hiptensorDataTypeSize(typeD);
        }
        auto sizeE = elementsFromLengths(e_ms_ns_lengths) * hiptensorDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *wspace;

        /*
         * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
         * ```
         * alphaF = hiptensor::readVal<float>(
         *      alpha, convertToComputeType(HipTensorDataType_v<typename Traits::ComputeDataT>));
         * ```
         * Hence, the `alpha` and `bete` need to point to a ComputeData value
         */
        ScalarData alpha;
        ScalarData beta;
        if(computeType == HIPTENSOR_COMPUTE_DESC_C32F || computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            writeVal(&alpha, computeType, {computeType, 1.02, 1.03});
            writeVal(&beta, computeType, {computeType, 1.04, 1.05});
        }
        else
        {
            writeVal(&alpha, computeType, ScalarData(computeType, 1.02));
            writeVal(&beta, computeType, ScalarData(computeType, 1.03));
        }

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

        std::string          best_op_name;
        ContractionSolution* bestSolution = nullptr;
        PerfMetrics          bestMetrics  = {
            0,
            "",
            0,
            0,
            0,
        };

        std::vector<float> sol_times(candidates.size(), std::numeric_limits<float>::max());
        std::vector<int>   indices(candidates.size());
        std::iota(indices.begin(), indices.end(), 0);
        int idx = 0;
        for(auto* solution : candidates)
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();

            auto [errorCode, time] = (*solution)(&alpha,
                                                 A_d,
                                                 B_d,
                                                 &beta,
                                                 D_d,
                                                 E_d,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 e_ms_ns_modes,
                                                 unaryOps,
                                                 wspace,
                                                 workspaceSize,
                                                 StreamConfig{
                                                     nullptr, // stream id
                                                     true, // time_kernel
                                                     0, // log_level
                                                     options->coldRuns(), // cold_niters
                                                     options->hotRuns(), // nrepeat
                                                 });
            if(errorCode == HIPTENSOR_STATUS_SUCCESS && time > 0)
            {
                // Make sure to time the kernels
                int32_t m, n, k;
                std::tie(m, n, k) = solution->problemDims();
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = solution->problemBytes();

                PerfMetrics metrics = {
                    solution->uid(), // id
                    solution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                using hiptensor::Logger;
                auto& logger = Logger::instance();

                // Log brute force timings for actor critic training
                if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE)
                {
                    // Log Kernel performances access
                    char msg[256];
                    snprintf(msg,
                             sizeof(msg),
                             "KernelId: %zu, KernelName: %s, AvgTime: %0.3f ms",
                             solution->uid(),
                             solution->kernelName().c_str(),
                             time);

                    logger->logHeuristics("BRUTE_FORCE_KERNEL_PERF", msg);
                }

                if(metrics > bestMetrics)
                {
                    bestSolution = solution;
                    bestMetrics  = metrics;
                }

                sol_times[idx] = time;
            }

            idx++;
        }

        CHECK_HIP_ALLOC(hipFree(A_d));
        CHECK_HIP_ALLOC(hipFree(B_d));
        CHECK_HIP_ALLOC(hipFree(D_d));
        CHECK_HIP_ALLOC(hipFree(E_d));
        CHECK_HIP_ALLOC(hipFree(wspace));

        *winner = bestSolution;

        //Sort candidates based on performance (from fastest to slowest)
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return sol_times[i] < sol_times[j];
        });
        std::vector<ContractionSolution*> tmpCandidates = candidates;
        candidates.clear();
        for(auto idx : indices)
            candidates.push_back(tmpCandidates[idx]);

        if(bestSolution == nullptr)
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
        else
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761397456090142ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2137990535294127184ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2137990535294127184ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1440761397456090142ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7799114060056978341ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1440761397456090142ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2855318965775908903ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10464053660216715737ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10322445194860155826ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761397456090142ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761397456090142ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2137990535294127184ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1440761397456090142ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7799114060056978341ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1440761397456090142ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10322445194860155826ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10464053660216715737ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10322445194860155826ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2137990534574907378ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14843745479191061099ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745479191061099ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745479191061099ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990534574907378ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7802847154886620598ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761397456090842ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761397456090842ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745479095717335ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745479095717335ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745479095717335ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745479095717335ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 7418684639400149524ull;
                }
                // m1n1k1
                else if(rank == 1)
                // if (rank == 1 || (rank == 1 && (a_ms_ks_lengths[3] == 1 || b_ns_ks_lengths[3] == 1)))
                {
                    unique_id = 7418684639400149524ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7802847154844028020ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684639400149524ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7418684639400149524ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7418684639400149524ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7802847154844028020ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440748527608857548ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745530650938202ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7802847185945830819ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818818903560452ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10464049561898994947ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440748527608857548ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745530650938202ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745530650938202ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7802847185945830819ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818818903560452ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10464049561898994947ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10464049561898994947ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2137990547371739022ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440748527571594356ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7418684562619456330ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684562619456330ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990547371739022ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2855318850119830510ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14843745530651266429ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745530651266429ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745530651266429ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745530651266429ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745530651266429ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745530651266429ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 13816607374468162736ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 7418684562505742304ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7802847185945766836ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684562505742304ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7418684562505742304ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7418684562505742304ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7418684562505742304ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761399884165504ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761399884165504ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745480098865124ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745480098865124ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7802847063887535332ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761399899554757ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761399899554757ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745485562042314ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745485562042314ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745485562042314ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745485562042314ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 2137990568415498270ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 7418684605442327629ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7418684605442327629ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684605442327629ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7418684605442327629ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7418684605442327629ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7418684605442327629ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1440761399884232257ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761399884232257ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1440761399884232257ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745480098915493ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745480098915493ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745480098915493ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 14843745480098915493ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 14843745480098915493ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684605148177952ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745480098915493ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1440761399884232257ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7802847063887519958ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float,
                                float,
                                float,
                                float,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799114058105151833ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761399899620868ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7799114058105151833ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 14843745485562238603ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745485562238603ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2137990568422245213ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 1440761399899620868ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 7418684605442261131ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7418684605442261131ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684605442261131ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7418684605442261131ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7418684605442261131ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7418684605442261131ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2137990568363823167ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7418684605154406951ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745480098848942ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2137990568363823167ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7802847063887535143ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1440761399899559459ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 14843745485562041486ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2137990568415498584ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 14843745485562041486ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 14843745485562041486ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7418684605442327694ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7418684605442327694ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5324987995348915333ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 18357727884545307985ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653994140800ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5864638737111407708ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13477196687221110573ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17963153062924488271ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5543115929776525336ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5543115929776525336ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1702163653986076371ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5324987995339838616ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653986076371ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 18357727884539305700ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 18357727884539305700ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653986076371ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 18357727884539305700ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 5543115929791243003ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5543115929791243003ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5543115929791243003ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5543115929791243003ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5543115929791243003ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5543115929791243003ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5478519037123088736ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1702163653994139125ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5864638737111407439ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 18357727884545307206ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16199389493181343293ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 18357727884539299315ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 18357727884539299315ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 18357727884539299315ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 18357727884539299315ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 18357727884539299315ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5478519037096761531ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 16199389493187916904ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 5543115929791244242ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5543115929791244242ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5543115929791244242ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5543115929791244242ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5543115929791244242ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5543115929791244242ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 11370543435050271066ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 10717070564994925940ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11370543435050271066ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 693497703183633080ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipFloatComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 11370543435030959225ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 11370543435030959225ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 11795761762493949769ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 693497703792998170ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 11795761762493949769ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 11795761762493949769ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 11795761762493949769ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 11795761762493949769ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10893407295027284090ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 5985644680422688852ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 9552735252268041381ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10893407295027284090ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5985644680422688852ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5985644680422688852ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5985644680422688852ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipDoubleComplex>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 10893407295038740933ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 10893407295038740933ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 10893407295038740933ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 10893407295038740933ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 10893407295038740933ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 10893407295038740933ull;
                }
            }
            else
            {
                bool dim1 = std::count(a_ms_ks_lengths.cbegin(), a_ms_ks_lengths.cend(), 1)
                            || std::count(b_ns_ks_lengths.cbegin(), b_ns_ks_lengths.cend(), 1);

                // rank2 dim1 case
                if(rank == 2 && dim1)
                {
                    unique_id = 5985644680464894601ull;
                }
                // m1n1k1
                else if(rank == 1)
                {
                    unique_id = 5070367300278634316ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5070367300278634316ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5070367300278634316ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 5070367300278634316ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5070367300278634316ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5070367300278634316ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    hiptensorStatus_t
        actorCriticModel(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         hiptensorComputeDescriptor_t                            computeType,
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
           && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
                                                                candidates,
                                                                typeA,
                                                                a_ms_ks_lengths,
                                                                a_ms_ks_strides,
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
                                                                    candidates,
                                                                    typeA,
                                                                    a_ms_ks_lengths,
                                                                    a_ms_ks_strides,
                                                                    a_ms_ks_modes,
                                                                    typeB,
                                                                    b_ns_ks_lengths,
                                                                    b_ns_ks_strides,
                                                                    b_ns_ks_modes,
                                                                    typeD,
                                                                    d_ms_ns_lengths,
                                                                    d_ms_ns_strides,
                                                                    d_ms_ns_modes,
                                                                    typeE,
                                                                    e_ms_ns_lengths,
                                                                    e_ms_ns_strides,
                                                                    e_ms_ns_modes,
                                                                    workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
                                                             candidates,
                                                             typeA,
                                                             a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
                                                             workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>::selectWinner(winner,
                                                              candidates,
                                                              typeA,
                                                              a_ms_ks_lengths,
                                                              a_ms_ks_strides,
                                                              a_ms_ks_modes,
                                                              typeB,
                                                              b_ns_ks_lengths,
                                                              b_ns_ks_strides,
                                                              b_ns_ks_modes,
                                                              typeD,
                                                              d_ms_ns_lengths,
                                                              d_ms_ns_strides,
                                                              d_ms_ns_modes,
                                                              typeE,
                                                              e_ms_ns_lengths,
                                                              e_ms_ns_strides,
                                                              e_ms_ns_modes,
                                                              workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_32F && typeB == HIPTENSOR_C_32F && typeD == HIPTENSOR_C_32F
                && typeE == HIPTENSOR_C_32F && computeType == HIPTENSOR_COMPUTE_DESC_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
                                                                       candidates,
                                                                       typeA,
                                                                       a_ms_ks_lengths,
                                                                       a_ms_ks_strides,
                                                                       a_ms_ks_modes,
                                                                       typeB,
                                                                       b_ns_ks_lengths,
                                                                       b_ns_ks_strides,
                                                                       b_ns_ks_modes,
                                                                       typeD,
                                                                       d_ms_ns_lengths,
                                                                       d_ms_ns_strides,
                                                                       d_ms_ns_modes,
                                                                       typeE,
                                                                       e_ms_ns_lengths,
                                                                       e_ms_ns_strides,
                                                                       e_ms_ns_modes,
                                                                       workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_C_64F && typeB == HIPTENSOR_C_64F && typeD == HIPTENSOR_C_64F
                && typeE == HIPTENSOR_C_64F && computeType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7799114060056978336ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818370467038ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818818370467038ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816606913356302831ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 7799114060056978336ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7799114060056978336ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818370467038ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818818370467038ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606913356302831ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816606913356302831ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    // Acotor-Critic model for unary ops
    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524427612935211ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17935524427612935211ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524427612935211ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17935524427612935211ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524427612935211ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524427612935211ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12026634380613989518ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606913347014500ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816606913347014500ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606913347014500ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606913347014500ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818814716308406ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15752964499459229608ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15752964499459229608ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15752964499459229608ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15752964499459229608ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752964499459229608ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15752964499459229608ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818370466973ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8631818818370466973ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818818370466973ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606913356301639ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606913356301639ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818818370466973ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15752963963305608309ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752963963305608309ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7799620307985751398ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818903560455ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816607374467706038ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799620307985751398ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752963963305608309ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7799620307985751398ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818903560455ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818818903560455ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816607374467706038ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816607374467706038ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15752963964616289564ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799620308515659795ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15752963964616289564ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799620308515659795ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752963964616289564ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15752963964616289564ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818916211369ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816607374243476251ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816607374243476251ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816607374243476251ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816607374243476251ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818818916211369ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 15752963963303840864ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799620307985946959ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15752963963303840864ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15752963963303840864ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752963963303840864ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 7799620307985946959ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818818927221532ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816607374468162743ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818818927221532ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816607374468162743ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816607374468162743ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818818927221532ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 7799114058106622813ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17935524478513395624ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524478513395624ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17935524478513395624ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524478513395624ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524478513395624ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818820589571050ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606924198920687ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816606924198920687ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606924198920687ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606924198920687ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816606924198920687ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524478459836898ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17935524478459836898ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524478459836898ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17935524478459836898ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524478459836898ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524478459836898ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 12026634250626031350ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606924240052004ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818820570004292ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606924240052004ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606924240052004ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816606924240052004ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524478513328745ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17935524478513328745ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524478513328745ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17935524478513328745ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524478513328745ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524478513328745ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818820589504811ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606924198855982ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13816606924198855982ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606924198855982ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606924198855982ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 13816606924198855982ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524478459900707ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 15752964492878320049ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 15752964492878320049ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 15752964492878320049ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 15752964492878320049ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 15752964492878320049ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818820571505029ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8631818820571505029ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818820571505029ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 8631818820571505029ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 8631818820571505029ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818820571505029ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524478513392152ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 17935524478513392152ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524478513392152ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 17935524478513392152ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524478513392152ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524478513392152ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818820589570860ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13816606924198924073ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818820589570860ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606924198924073ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606924198924073ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818820589570860ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17935524478459838250ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 7799114058105218391ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 17935524478459838250ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7799114058105218391ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 17935524478459838250ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 17935524478459838250ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 8631818820570004366ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 8631818820570004366ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 8631818820570004366ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 13816606924240051306ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13816606924240051306ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 8631818820570004366ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 7421365356354019374ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1702163653994140800ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653994140800ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16199389493181341988ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 2789232765572696187ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 2789232765572696187ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 2789232765572696187ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 2789232765572696187ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 2789232765572696187ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1702163653986076398ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1702163653986076398ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653986076398ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1702163653986076398ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1702163653986076398ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653986076398ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 17963153064137026984ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1058341445605340736ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1058341445605340736ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1058341445605340736ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1058341445605340736ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1058341445605340736ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16199389493181343293ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1702163653994139125ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653994139125ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1702163653994139125ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1702163653994139125ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653994139125ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 1645499542390481235ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 5543115929776526095ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 5543115929776526095ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1645499542390481235ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 5543115929776526095ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 5543115929776526095ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelectionUnaryOps<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hiptensorDataType_t                                     typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hiptensorDataType_t                                     typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hiptensorDataType_t                                     typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hiptensorDataType_t                                     typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            auto   rank      = getRank(a_ms_ks_strides);
            size_t unique_id = 0;

            auto& options = HiptensorOptions::instance();
            if(options->isColMajorStrides())
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16199389493187916911ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 1702163653986075129ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 1702163653986075129ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 1702163653986075129ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 1702163653986075129ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 1702163653986075129ull;
                }
            }
            else
            {
                // m1n1k1
                if(rank == 1)
                {
                    unique_id = 16199389493187916911ull;
                }
                // m2n2k2
                else if(rank == 2)
                {
                    unique_id = 13477196687211894914ull;
                }
                // m3n3k3
                else if(rank == 3)
                {
                    unique_id = 13477196687211894914ull;
                }
                // m4n4k4
                else if(rank == 4)
                {
                    unique_id = 16199389493187916911ull;
                }
                // m5n5k5
                else if(rank == 5)
                {
                    unique_id = 13477196687211894914ull;
                }
                // m6n6k6
                else if(rank == 6)
                {
                    unique_id = 16199389493187916911ull;
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    hiptensorStatus_t
        actorCriticModelUnaryOps(ContractionSolution**                                   winner,
                                 std::unordered_map<size_t, ContractionSolution*> const& candidates,
                                 hiptensorDataType_t                                     typeA,
                                 std::vector<std::size_t> const& a_ms_ks_lengths,
                                 std::vector<std::size_t> const& a_ms_ks_strides,
                                 std::vector<int32_t> const&     a_ms_ks_modes,
                                 hiptensorDataType_t             typeB,
                                 std::vector<std::size_t> const& b_ns_ks_lengths,
                                 std::vector<std::size_t> const& b_ns_ks_strides,
                                 std::vector<int32_t> const&     b_ns_ks_modes,
                                 hiptensorDataType_t             typeD,
                                 std::vector<std::size_t> const& d_ms_ns_lengths,
                                 std::vector<std::size_t> const& d_ms_ns_strides,
                                 std::vector<int32_t> const&     d_ms_ns_modes,
                                 hiptensorDataType_t             typeE,
                                 std::vector<std::size_t> const& e_ms_ns_lengths,
                                 std::vector<std::size_t> const& e_ms_ns_strides,
                                 std::vector<int32_t> const&     e_ms_ns_modes,
                                 hiptensorComputeDescriptor_t    computeType,
                                 const uint64_t                  workspaceSize)
    {
        if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == NONE_TYPE
           && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<_Float16,
                                                _Float16,
                                                _Float16,
                                                _Float16,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16F && typeB == HIPTENSOR_R_16F && typeD == HIPTENSOR_R_16F
                && typeE == HIPTENSOR_R_16F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<_Float16,
                                                _Float16,
                                                _Float16,
                                                _Float16,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_16BF && typeB == HIPTENSOR_R_16BF && typeD == HIPTENSOR_R_16BF
                && typeE == HIPTENSOR_R_16BF && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                hip_bfloat16,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                _Float16>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                _Float16>::selectWinner(winner,
                                                                        candidates,
                                                                        typeA,
                                                                        a_ms_ks_lengths,
                                                                        a_ms_ks_strides,
                                                                        a_ms_ks_modes,
                                                                        typeB,
                                                                        b_ns_ks_lengths,
                                                                        b_ns_ks_strides,
                                                                        b_ns_ks_modes,
                                                                        typeD,
                                                                        d_ms_ns_lengths,
                                                                        d_ms_ns_strides,
                                                                        d_ms_ns_modes,
                                                                        typeE,
                                                                        e_ms_ns_lengths,
                                                                        e_ms_ns_strides,
                                                                        e_ms_ns_modes,
                                                                        workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                hip_bfloat16>::selectWinner(winner,
                                                                            candidates,
                                                                            typeA,
                                                                            a_ms_ks_lengths,
                                                                            a_ms_ks_strides,
                                                                            a_ms_ks_modes,
                                                                            typeB,
                                                                            b_ns_ks_lengths,
                                                                            b_ns_ks_strides,
                                                                            b_ns_ks_modes,
                                                                            typeD,
                                                                            d_ms_ns_lengths,
                                                                            d_ms_ns_strides,
                                                                            d_ms_ns_modes,
                                                                            typeE,
                                                                            e_ms_ns_lengths,
                                                                            e_ms_ns_strides,
                                                                            e_ms_ns_modes,
                                                                            workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_R_16BF)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                hip_bfloat16>::selectWinner(winner,
                                                                            candidates,
                                                                            typeA,
                                                                            a_ms_ks_lengths,
                                                                            a_ms_ks_strides,
                                                                            a_ms_ks_modes,
                                                                            typeB,
                                                                            b_ns_ks_lengths,
                                                                            b_ns_ks_strides,
                                                                            b_ns_ks_modes,
                                                                            typeD,
                                                                            d_ms_ns_lengths,
                                                                            d_ms_ns_strides,
                                                                            d_ms_ns_modes,
                                                                            typeE,
                                                                            e_ms_ns_lengths,
                                                                            e_ms_ns_strides,
                                                                            e_ms_ns_modes,
                                                                            workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_32F && typeB == HIPTENSOR_R_32F && typeD == HIPTENSOR_R_32F
                && typeE == HIPTENSOR_R_32F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<float,
                                                float,
                                                float,
                                                float,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::SCALE,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::BILINEAR,
                                                float>::selectWinner(winner,
                                                                     candidates,
                                                                     typeA,
                                                                     a_ms_ks_lengths,
                                                                     a_ms_ks_strides,
                                                                     a_ms_ks_modes,
                                                                     typeB,
                                                                     b_ns_ks_lengths,
                                                                     b_ns_ks_strides,
                                                                     b_ns_ks_modes,
                                                                     typeD,
                                                                     d_ms_ns_lengths,
                                                                     d_ms_ns_strides,
                                                                     d_ms_ns_modes,
                                                                     typeE,
                                                                     e_ms_ns_lengths,
                                                                     e_ms_ns_strides,
                                                                     e_ms_ns_modes,
                                                                     workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == NONE_TYPE
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::SCALE,
                                                double>::selectWinner(winner,
                                                                      candidates,
                                                                      typeA,
                                                                      a_ms_ks_lengths,
                                                                      a_ms_ks_strides,
                                                                      a_ms_ks_modes,
                                                                      typeB,
                                                                      b_ns_ks_lengths,
                                                                      b_ns_ks_strides,
                                                                      b_ns_ks_modes,
                                                                      typeD,
                                                                      d_ms_ns_lengths,
                                                                      d_ms_ns_strides,
                                                                      d_ms_ns_modes,
                                                                      typeE,
                                                                      e_ms_ns_lengths,
                                                                      e_ms_ns_strides,
                                                                      e_ms_ns_modes,
                                                                      workspaceSize);
        }
        else if(typeA == HIPTENSOR_R_64F && typeB == HIPTENSOR_R_64F && typeD == HIPTENSOR_R_64F
                && typeE == HIPTENSOR_R_64F && computeType == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return ActorCriticSelectionUnaryOps<double,
                                                double,
                                                double,
                                                double,
                                                ContractionOpId_t::BILINEAR,
                                                double>::selectWinner(winner,
                                                                      candidates,
                                                                      typeA,
                                                                      a_ms_ks_lengths,
                                                                      a_ms_ks_strides,
                                                                      a_ms_ks_modes,
                                                                      typeB,
                                                                      b_ns_ks_lengths,
                                                                      b_ns_ks_strides,
                                                                      b_ns_ks_modes,
                                                                      typeD,
                                                                      d_ms_ns_lengths,
                                                                      d_ms_ns_strides,
                                                                      d_ms_ns_modes,
                                                                      typeE,
                                                                      e_ms_ns_lengths,
                                                                      e_ms_ns_strides,
                                                                      e_ms_ns_modes,
                                                                      workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
