/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Topology & Advanced P2P Test Suite
 * Tests: PCIe topology, multi-hop P2P, NUMA awareness, cross-socket performance
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>
#include <map>
#include <set>
#include <chrono>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // ----------------------------------------------------------------------------
    // Test 1: PCIe Topology Discovery
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, PCIeTopologyDiscovery)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== PCIe Topology Discovery ===" << std::endl;

        // Build P2P connectivity matrix
        std::vector<std::vector<bool>> p2p_matrix(numDevices, std::vector<bool>(numDevices, false));

        for(int i = 0; i < numDevices; ++i)
        {
            for(int j = 0; j < numDevices; ++j)
            {
                if(i == j)
                {
                    p2p_matrix[i][j] = true;
                    continue;
                }

                int canAccess = 0;
                hipDeviceCanAccessPeer(&canAccess, i, j);
                p2p_matrix[i][j] = (canAccess != 0);
            }
        }

        // Print connectivity matrix
        hipblaslt_cout << "P2P Connectivity Matrix:" << std::endl;
        hipblaslt_cout << "     ";
        for(int j = 0; j < numDevices; ++j)
            hipblaslt_cout << "GPU" << j << " ";
        hipblaslt_cout << std::endl;

        for(int i = 0; i < numDevices; ++i)
        {
            hipblaslt_cout << "GPU" << i << ": ";
            for(int j = 0; j < numDevices; ++j)
            {
                hipblaslt_cout << (p2p_matrix[i][j] ? " Y  " : " N  ");
            }
            hipblaslt_cout << std::endl;
        }

        // Analyze topology patterns
        int direct_connections = 0;
        for(int i = 0; i < numDevices; ++i)
        {
            for(int j = i + 1; j < numDevices; ++j)
            {
                if(p2p_matrix[i][j] && p2p_matrix[j][i])
                    direct_connections++;
            }
        }

        hipblaslt_cout << "Total direct P2P connections: " << direct_connections << std::endl;
        hipblaslt_cout << "Maximum possible connections: " << (numDevices * (numDevices - 1)) / 2 << std::endl;

        // Check for full mesh, partial mesh, or no connectivity
        int total_possible = numDevices * (numDevices - 1);
        int actual_connections = 0;
        for(int i = 0; i < numDevices; ++i)
        {
            for(int j = 0; j < numDevices; ++j)
            {
                if(i != j && p2p_matrix[i][j])
                    actual_connections++;
            }
        }

        float connectivity_ratio = static_cast<float>(actual_connections) / total_possible;
        hipblaslt_cout << "Connectivity ratio: " << connectivity_ratio * 100.0f << "%" << std::endl;

        if(connectivity_ratio == 1.0f)
            hipblaslt_cout << "Topology: Full mesh (all-to-all P2P)" << std::endl;
        else if(connectivity_ratio > 0.5f)
            hipblaslt_cout << "Topology: High connectivity (partial mesh)" << std::endl;
        else if(connectivity_ratio > 0.0f)
            hipblaslt_cout << "Topology: Low connectivity (sparse)" << std::endl;
        else
            hipblaslt_cout << "Topology: No P2P support" << std::endl;
    }

    // ----------------------------------------------------------------------------
    // Test 2: PCIe Switch vs Direct Connection Detection
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, PCIeSwitchDetection)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== PCIe Switch Detection ===" << std::endl;

        // Measure P2P bandwidth between all GPU pairs
        const size_t transfer_size = 256 * 1024 * 1024; // 256 MB
        const int num_iterations = 5;

        std::vector<std::vector<double>> bandwidth_matrix(numDevices, std::vector<double>(numDevices, 0.0));

        for(int src = 0; src < numDevices; ++src)
        {
            hipSetDevice(src);

            for(int dst = 0; dst < numDevices; ++dst)
            {
                if(src == dst) continue;

                int canAccess = 0;
                hipDeviceCanAccessPeer(&canAccess, dst, src);
                if(!canAccess) continue;

                // Enable P2P
                hipSetDevice(dst);
                hipError_t err = hipDeviceEnablePeerAccess(src, 0);
                if(err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
                    continue;

                // Allocate buffers
                hipSetDevice(src);
                float* d_src;
                hipMalloc(&d_src, transfer_size);

                hipSetDevice(dst);
                float* d_dst;
                hipMalloc(&d_dst, transfer_size);

                // Warm-up
                hipMemcpyPeer(d_dst, dst, d_src, src, transfer_size);
                hipDeviceSynchronize();

                // Timed transfers
                auto start = std::chrono::high_resolution_clock::now();
                for(int iter = 0; iter < num_iterations; ++iter)
                {
                    hipMemcpyPeer(d_dst, dst, d_src, src, transfer_size);
                }
                hipDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double bandwidth_gbps = (transfer_size * num_iterations) / (elapsed.count() * 1e9);
                bandwidth_matrix[src][dst] = bandwidth_gbps;

                hipSetDevice(src);
                hipFree(d_src);
                hipSetDevice(dst);
                hipFree(d_dst);

                hipblaslt_cout << "GPU " << src << " -> GPU " << dst << ": "
                               << bandwidth_gbps << " GB/s" << std::endl;
            }
        }

        // Analyze bandwidth patterns to detect switches
        // Direct connections typically have higher bandwidth than switch-based
        std::vector<double> all_bandwidths;
        for(int i = 0; i < numDevices; ++i)
        {
            for(int j = 0; j < numDevices; ++j)
            {
                if(i != j && bandwidth_matrix[i][j] > 0.0)
                    all_bandwidths.push_back(bandwidth_matrix[i][j]);
            }
        }

        if(!all_bandwidths.empty())
        {
            double max_bw = *std::max_element(all_bandwidths.begin(), all_bandwidths.end());
            double min_bw = *std::min_element(all_bandwidths.begin(), all_bandwidths.end());
            double avg_bw = std::accumulate(all_bandwidths.begin(), all_bandwidths.end(), 0.0) / all_bandwidths.size();

            hipblaslt_cout << "P2P Bandwidth Statistics:" << std::endl;
            hipblaslt_cout << "  Max: " << max_bw << " GB/s" << std::endl;
            hipblaslt_cout << "  Min: " << min_bw << " GB/s" << std::endl;
            hipblaslt_cout << "  Avg: " << avg_bw << " GB/s" << std::endl;

            // Large variance suggests mixed direct/switch connections
            if(max_bw > min_bw * 1.5)
            {
                hipblaslt_cout << "Detected mixed topology (likely PCIe switches present)" << std::endl;
            }
            else
            {
                hipblaslt_cout << "Uniform topology (likely all direct or all through switch)" << std::endl;
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Test 3: Multi-Hop P2P Transfer
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, MultiHopP2PTransfer)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 3) GTEST_SKIP() << "Requires 3+ GPUs for multi-hop";

        hipblaslt_cout << "=== Multi-Hop P2P Transfer ===" << std::endl;

        const size_t data_size = 64 * 1024 * 1024; // 64 MB

        // Test chain: GPU 0 -> GPU 1 -> GPU 2
        std::vector<float*> d_data(3);

        for(int i = 0; i < 3; ++i)
        {
            hipSetDevice(i);
            hipMalloc(&d_data[i], data_size);
        }

        // Initialize data on GPU 0
        hipSetDevice(0);
        std::vector<float> h_data(data_size / sizeof(float));
        for(size_t i = 0; i < h_data.size(); ++i)
            h_data[i] = static_cast<float>(i % 1000);

        hipMemcpy(d_data[0], h_data.data(), data_size, hipMemcpyHostToDevice);

        // Enable P2P for chain
        for(int i = 0; i < 2; ++i)
        {
            int canAccess = 0;
            hipDeviceCanAccessPeer(&canAccess, i + 1, i);
            if(canAccess)
            {
                hipSetDevice(i + 1);
                hipDeviceEnablePeerAccess(i, 0);
            }
        }

        // Multi-hop transfer: 0 -> 1 -> 2
        auto start = std::chrono::high_resolution_clock::now();

        hipMemcpyPeer(d_data[1], 1, d_data[0], 0, data_size);
        hipDeviceSynchronize();

        hipMemcpyPeer(d_data[2], 2, d_data[1], 1, data_size);
        hipDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();

        // Verify data on GPU 2
        hipSetDevice(2);
        std::vector<float> h_result(data_size / sizeof(float));
        hipMemcpy(h_result.data(), d_data[2], data_size, hipMemcpyDeviceToHost);

        bool correct = true;
        for(size_t i = 0; i < h_result.size(); ++i)
        {
            if(std::abs(h_result[i] - h_data[i]) > 0.001f)
            {
                correct = false;
                break;
            }
        }

        EXPECT_TRUE(correct) << "Multi-hop transfer data mismatch!";

        std::chrono::duration<double> elapsed = end - start;
        double bandwidth = (2.0 * data_size) / (elapsed.count() * 1e9); // 2 hops

        hipblaslt_cout << "Multi-hop transfer (GPU 0 -> 1 -> 2): " << elapsed.count() * 1000.0
                       << " ms" << std::endl;
        hipblaslt_cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;

        // Cleanup
        for(int i = 0; i < 3; ++i)
        {
            hipSetDevice(i);
            hipFree(d_data[i]);
        }
    }

    // ----------------------------------------------------------------------------
    // Test 4: NUMA Node Affinity Detection
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, NUMANodeAffinity)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== NUMA Node Affinity Detection ===" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            hipblaslt_cout << "GPU " << dev << " (" << prop.name << "):" << std::endl;
            hipblaslt_cout << "  PCIe Bus ID: " << prop.pciBusID << std::endl;
            hipblaslt_cout << "  PCIe Device ID: " << prop.pciDeviceID << std::endl;
            hipblaslt_cout << "  PCIe Domain ID: " << prop.pciDomainID << std::endl;

            // Try to determine NUMA node (platform-specific)
            // This is a heuristic - real implementation would use hwloc or similar
            int numa_hint = prop.pciBusID / 64; // Simple heuristic
            hipblaslt_cout << "  Estimated NUMA node: " << numa_hint << std::endl;
        }

        // Test cross-NUMA vs same-NUMA transfer performance
        if(numDevices >= 2)
        {
            const size_t test_size = 128 * 1024 * 1024; // 128 MB

            for(int src = 0; src < std::min(numDevices, 2); ++src)
            {
                for(int dst = src + 1; dst < std::min(numDevices, 4); ++dst)
                {
                    int canAccess = 0;
                    hipDeviceCanAccessPeer(&canAccess, dst, src);
                    if(!canAccess) continue;

                    hipSetDevice(src);
                    float* d_src;
                    hipMalloc(&d_src, test_size);

                    hipSetDevice(dst);
                    float* d_dst;
                    hipMalloc(&d_dst, test_size);
                    hipDeviceEnablePeerAccess(src, 0);

                    // Timed transfer
                    auto start = std::chrono::high_resolution_clock::now();
                    hipMemcpyPeer(d_dst, dst, d_src, src, test_size);
                    hipDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> elapsed = end - start;
                    double bandwidth = test_size / (elapsed.count() * 1e9);

                    hipblaslt_cout << "GPU " << src << " -> GPU " << dst << ": "
                                   << bandwidth << " GB/s (" << elapsed.count() * 1000.0 << " ms)"
                                   << std::endl;

                    hipSetDevice(src);
                    hipFree(d_src);
                    hipSetDevice(dst);
                    hipFree(d_dst);
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Test 5: Cross-Socket vs Same-Socket Performance
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, CrossSocketPerformance)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 4) GTEST_SKIP() << "Requires 4+ GPUs for socket testing";

        hipblaslt_cout << "=== Cross-Socket vs Same-Socket Performance ===" << std::endl;

        // Assumption: GPUs 0-3 might be on socket 0, GPUs 4-7 on socket 1 (if 8 GPUs)
        // This is heuristic - actual socket detection requires system-specific code

        const int64_t M = 1024, N = 1024, K = 1024;
        std::vector<double> execution_times(numDevices);

        // Test computation time on each GPU
        for(int dev = 0; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            float *d_a, *d_b, *d_c, *d_d;
            hipMalloc(&d_a, M * K * sizeof(float));
            hipMalloc(&d_b, K * N * sizeof(float));
            hipMalloc(&d_c, M * N * sizeof(float));
            hipMalloc(&d_d, M * N * sizeof(float));

            std::vector<float> h_a(M * K, 1.0f);
            std::vector<float> h_b(K * N, 2.0f);
            std::vector<float> h_c(M * N, 0.0f);

            hipMemcpy(d_a, h_a.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_b.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(d_c, h_c.data(), M * N * sizeof(float), hipMemcpyHostToDevice);

            hipblasLtMatrixLayout_t matA, matB, matC, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, M, N, M);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);

            hipblasLtMatmulHeuristicResult_t heuristicResult[1];
            int returnedAlgoCount = 0;
            hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD,
                                           pref, 1, heuristicResult, &returnedAlgoCount);

            if(returnedAlgoCount > 0)
            {
                void* d_workspace = nullptr;
                if(heuristicResult[0].workspaceSize > 0)
                    hipMalloc(&d_workspace, heuristicResult[0].workspaceSize);

                float alpha = 1.0f, beta = 0.0f;

                // Warmup
                hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                               &beta, d_c, matC, d_d, matD,
                               &heuristicResult[0].algo, d_workspace,
                               heuristicResult[0].workspaceSize, 0);
                hipDeviceSynchronize();

                // Timed execution
                auto start = std::chrono::high_resolution_clock::now();
                for(int iter = 0; iter < 10; ++iter)
                {
                    hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB,
                                   &beta, d_c, matC, d_d, matD,
                                   &heuristicResult[0].algo, d_workspace,
                                   heuristicResult[0].workspaceSize, 0);
                }
                hipDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                execution_times[dev] = elapsed.count() / 10.0; // Average per iteration

                if(d_workspace) hipFree(d_workspace);
            }

            hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matC);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);

            hipblaslt_cout << "GPU " << dev << " execution time: "
                           << execution_times[dev] * 1000.0 << " ms" << std::endl;
        }

        // Analyze performance variance (cross-socket GPUs may show different performance)
        double avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / numDevices;
        double variance = 0.0;
        for(auto t : execution_times)
            variance += (t - avg_time) * (t - avg_time);
        variance /= numDevices;

        hipblaslt_cout << "Average execution time: " << avg_time * 1000.0 << " ms" << std::endl;
        hipblaslt_cout << "Variance: " << variance * 1e6 << " ms²" << std::endl;

        if(variance > avg_time * avg_time * 0.01) // >1% variance
        {
            hipblaslt_cout << "Significant performance variance detected (possible cross-socket effect)" << std::endl;
        }
    }

    // ----------------------------------------------------------------------------
    // Test 6: P2P Bandwidth Saturation Test
    // ----------------------------------------------------------------------------
    TEST(MultiGPUTopology, P2PBandwidthSaturation)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "=== P2P Bandwidth Saturation Test ===" << std::endl;

        // Test increasing transfer sizes to find bandwidth saturation point
        std::vector<size_t> transfer_sizes = {
            1024,              // 1 KB
            1024 * 1024,       // 1 MB
            16 * 1024 * 1024,  // 16 MB
            64 * 1024 * 1024,  // 64 MB
            256 * 1024 * 1024, // 256 MB
            512 * 1024 * 1024  // 512 MB
        };

        int src = 0, dst = 1;
        int canAccess = 0;
        hipDeviceCanAccessPeer(&canAccess, dst, src);
        if(!canAccess)
        {
            GTEST_SKIP() << "P2P not supported between GPU 0 and GPU 1";
        }

        hipSetDevice(dst);
        hipDeviceEnablePeerAccess(src, 0);

        for(auto size : transfer_sizes)
        {
            hipSetDevice(src);
            float* d_src;
            hipMalloc(&d_src, size);

            hipSetDevice(dst);
            float* d_dst;
            hipMalloc(&d_dst, size);

            // Warmup
            hipMemcpyPeer(d_dst, dst, d_src, src, size);
            hipDeviceSynchronize();

            // Timed transfer
            const int iterations = 20;
            auto start = std::chrono::high_resolution_clock::now();
            for(int iter = 0; iter < iterations; ++iter)
            {
                hipMemcpyPeer(d_dst, dst, d_src, src, size);
            }
            hipDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            double bandwidth = (size * iterations) / (elapsed.count() * 1e9);

            hipblaslt_cout << "Transfer size: " << (size / 1024.0 / 1024.0) << " MB, "
                           << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

            hipSetDevice(src);
            hipFree(d_src);
            hipSetDevice(dst);
            hipFree(d_dst);
        }
    }

} // namespace
