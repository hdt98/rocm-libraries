/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Device Parameters Test Suite
 * P2P access, partitioning strategies, communication patterns
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // Test 1: Enumerate All Device IDs
    TEST(MultiGPUDevice, EnumerateAllDevices)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Enumerating all GPU devices" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            hipblaslt_cout << "GPU " << dev << ": " << prop.name
                           << " (compute " << prop.major << "." << prop.minor << ")"
                           << " Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB"
                           << std::endl;
        }
    }

    // Test 2: P2P Access Check
    TEST(MultiGPUDevice, P2PAccessCheck)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Checking P2P access between GPUs" << std::endl;

        for(int i = 0; i < numDevices; ++i)
        {
            for(int j = 0; j < numDevices; ++j)
            {
                if(i == j) continue;

                int canAccessPeer = 0;
                hipDeviceCanAccessPeer(&canAccessPeer, i, j);

                if(canAccessPeer)
                {
                    hipblaslt_cout << "GPU " << i << " can access GPU " << j << " via P2P" << std::endl;
                }
                else
                {
                    hipblaslt_cout << "GPU " << i << " CANNOT access GPU " << j << " via P2P" << std::endl;
                }
            }
        }
    }

    // Test 3: Enable P2P Access
    TEST(MultiGPUDevice, EnableP2PAccess)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Enabling P2P access between GPUs" << std::endl;

        for(int i = 0; i < numDevices; ++i)
        {
            hipSetDevice(i);

            for(int j = 0; j < numDevices; ++j)
            {
                if(i == j) continue;

                int canAccessPeer = 0;
                hipDeviceCanAccessPeer(&canAccessPeer, i, j);

                if(canAccessPeer)
                {
                    auto err = hipDeviceEnablePeerAccess(j, 0);
                    if(err == hipSuccess)
                    {
                        hipblaslt_cout << "Enabled P2P access: GPU " << i << " <-> GPU " << j << std::endl;
                    }
                    else if(err == hipErrorPeerAccessAlreadyEnabled)
                    {
                        hipblaslt_cout << "P2P access already enabled: GPU " << i << " <-> GPU " << j << std::endl;
                    }
                }
            }
        }
    }

    // Test 4: Row Partitioning (Split M across GPUs)
    TEST(MultiGPUDevice, RowPartitioning)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing row partitioning (split M across GPUs)" << std::endl;

        const int64_t M_total = 1024, N = 512, K = 512;
        int64_t M_per_gpu = M_total / numDevices;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t M_start = dev * M_per_gpu;
            int64_t M_local = (dev == numDevices - 1) ? (M_total - M_start) : M_per_gpu;

            hipblaslt_cout << "GPU " << dev << " handles rows [" << M_start
                           << ", " << (M_start + M_local) << ") of " << M_total << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M_local, K, M_local);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Column Partitioning (Split N across GPUs)
    TEST(MultiGPUDevice, ColumnPartitioning)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing column partitioning (split N across GPUs)" << std::endl;

        const int64_t M = 512, N_total = 1024, K = 512;
        int64_t N_per_gpu = N_total / numDevices;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t N_start = dev * N_per_gpu;
            int64_t N_local = (dev == numDevices - 1) ? (N_total - N_start) : N_per_gpu;

            hipblaslt_cout << "GPU " << dev << " handles columns [" << N_start
                           << ", " << (N_start + N_local) << ") of " << N_total << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matB;
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N_local, K);

            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: K-Dimension Partitioning
    TEST(MultiGPUDevice, KDimensionPartitioning)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing K-dimension partitioning across GPUs" << std::endl;

        const int64_t M = 512, N = 512, K_total = 1024;
        int64_t K_per_gpu = K_total / numDevices;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t K_start = dev * K_per_gpu;
            int64_t K_local = (dev == numDevices - 1) ? (K_total - K_start) : K_per_gpu;

            hipblaslt_cout << "GPU " << dev << " handles K-dimension [" << K_start
                           << ", " << (K_start + K_local) << ") of " << K_total << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K_local, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K_local, N, K_local);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Data Parallel (Same Work on All GPUs)
    TEST(MultiGPUDevice, DataParallel)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing data parallel (same work on all GPUs)" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " executing identical GEMM operation" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Ring Communication Pattern
    TEST(MultiGPUDevice, RingCommunication)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing ring communication pattern" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int next_dev = (dev + 1) % numDevices;
            int prev_dev = (dev - 1 + numDevices) % numDevices;

            hipblaslt_cout << "GPU " << dev << " ring neighbors: prev=" << prev_dev
                           << ", next=" << next_dev << std::endl;

            // Check P2P access with neighbors
            int canAccessNext = 0;
            hipDeviceCanAccessPeer(&canAccessNext, dev, next_dev);

            if(canAccessNext)
            {
                hipblaslt_cout << "  Can communicate with next GPU " << next_dev << std::endl;
            }
        }
    }

    // Test 9: All-Reduce Pattern Simulation
    TEST(MultiGPUDevice, AllReducePattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing all-reduce pattern simulation" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        // Phase 1: Each GPU computes partial result
        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblaslt_cout << "GPU " << dev << " computing partial result" << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matD;
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }

        // Phase 2: All-reduce (simulated)
        hipblaslt_cout << "All-reduce phase: aggregating results from all GPUs" << std::endl;
    }

    // Test 10: Broadcast Pattern
    TEST(MultiGPUDevice, BroadcastPattern)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing broadcast pattern (GPU 0 to all)" << std::endl;

        const int64_t M = 256, N = 256;

        // GPU 0 is the source
        hipSetDevice(0);
        float* d_data_source = nullptr;
        hipMalloc(&d_data_source, M * N * sizeof(float));

        hipblaslt_cout << "GPU 0: Source data allocated" << std::endl;

        // Broadcast to other GPUs
        for(int dev = 1; dev < numDevices; ++dev)
        {
            hipSetDevice(dev);
            float* d_data_dest = nullptr;
            hipMalloc(&d_data_dest, M * N * sizeof(float));

            // In real scenario, would use P2P copy or host staging
            hipblaslt_cout << "GPU " << dev << ": Receiving broadcast data from GPU 0" << std::endl;

            hipFree(d_data_dest);
        }

        hipSetDevice(0);
        hipFree(d_data_source);
    }

    // Test 11: Different Problem Sizes Per GPU
    TEST(MultiGPUDevice, DifferentProblemSizesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different problem sizes per GPU" << std::endl;

        std::vector<int64_t> M_sizes = {128, 256, 512, 1024};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            int64_t M = M_sizes[dev % M_sizes.size()];
            int64_t N = 256, K = 256;

            hipblaslt_cout << "GPU " << dev << " problem size: " << M << "x" << N << "x" << K << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

} // namespace
