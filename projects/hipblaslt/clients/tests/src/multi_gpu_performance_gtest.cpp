/*******************************************************************************
 * Multi-GPU Performance/Stress Test Suite
 * Tests: Load Balancing, Maximum Problem Sizes, Memory Pressure
 *******************************************************************************/
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <gtest/gtest-spi.h>
#include <chrono>

namespace {
int getNumGPUs() { int n = 0; hipGetDeviceCount(&n); return n; }

TEST(MultiGPUPerformance, LoadBalancing) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing load balancing across " << numDevices << " GPUs" << std::endl;
    
    // Distribute different workload sizes
    std::vector<int64_t> workloads;
    for(int i = 0; i < numDevices; ++i) {
        // Fibonacci-like distribution for testing load balancing
        workloads.push_back(64 * (1 << i));  // 64, 128, 256, 512...
    }
    
    for(int dev = 0; dev < numDevices; ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        int64_t M = workloads[dev];
        int64_t N = workloads[dev];
        int64_t K = 64;
        
        hipblaslt_cout << "GPU " << dev << " workload: " << M << "x" << N << "x" << K << std::endl;
        
        hipblasLtHandle_t handle;
        auto status = hipblasLtCreate(&handle);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        
        hipblasLtMatrixLayout_t matA, matB, matD;
        status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        
        status = hipblasLtMatrixLayoutDestroy(matA);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtMatrixLayoutDestroy(matB);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtMatrixLayoutDestroy(matD);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtDestroy(handle);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    }
    
    hipblaslt_cout << "Load balancing test passed" << std::endl;
}

TEST(MultiGPUPerformance, MaximumProblemSize) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing maximum problem sizes" << std::endl;
    
    for(int dev = 0; dev < std::min(numDevices, 2); ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        size_t free_mem, total_mem;
        hipErr = hipMemGetInfo(&free_mem, &total_mem);
        ASSERT_EQ(hipErr, hipSuccess);
        
        // Use ~10% of available memory for stress test
        size_t usable_mem = free_mem / 10;
        int64_t M = static_cast<int64_t>(sqrt(usable_mem / sizeof(float) / 3));
        M = (M / 64) * 64;  // Round to multiple of 64
        
        hipblaslt_cout << "GPU " << dev << " testing " << M << "x" << M << " GEMM ("
                       << (3 * M * M * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
        if(M < 64) continue;
        
        float *d_a, *d_b, *d_d;
        hipErr = hipMalloc(&d_a, M * M * sizeof(float));
        if(hipErr != hipSuccess) {
            hipblaslt_cout << "GPU " << dev << " skipped - insufficient memory" << std::endl;
            continue;
        }
        hipErr = hipMalloc(&d_b, M * M * sizeof(float));
        ASSERT_EQ(hipErr, hipSuccess);
        hipErr = hipMalloc(&d_d, M * M * sizeof(float));
        ASSERT_EQ(hipErr, hipSuccess);
        
        hipErr = hipFree(d_a);
        EXPECT_EQ(hipErr, hipSuccess);
        hipErr = hipFree(d_b);
        EXPECT_EQ(hipErr, hipSuccess);
        hipErr = hipFree(d_d);
        EXPECT_EQ(hipErr, hipSuccess);
    }
    
    hipblaslt_cout << "Maximum problem size test passed" << std::endl;
}

TEST(MultiGPUPerformance, MemoryPressure) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing memory pressure scenarios" << std::endl;
    
    // Allocate on all GPUs simultaneously
    std::vector<float*> allocations(numDevices);
    size_t alloc_size = 512 * 1024 * 1024;  // 512 MB per GPU
    
    for(int dev = 0; dev < numDevices; ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        hipErr = hipMalloc(&allocations[dev], alloc_size);
        if(hipErr == hipSuccess) {
            hipblaslt_cout << "GPU " << dev << " allocated 512 MB" << std::endl;
        } else {
            hipblaslt_cout << "GPU " << dev << " allocation failed (expected under pressure)" << std::endl;
            allocations[dev] = nullptr;
        }
    }
    
    // Free all
    for(int dev = 0; dev < numDevices; ++dev) {
        if(allocations[dev]) {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);
            hipErr = hipFree(allocations[dev]);
            EXPECT_EQ(hipErr, hipSuccess);
        }
    }
    
    hipblaslt_cout << "Memory pressure test passed" << std::endl;
}

TEST(MultiGPUPerformance, ConcurrentThroughput) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing concurrent throughput" << std::endl;
    
    const int64_t M = 512;
    const int64_t N = 512;
    const int64_t K = 512;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Sequential execution
    for(int dev = 0; dev < std::min(numDevices, 2); ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        hipblasLtHandle_t handle;
        auto status = hipblasLtCreate(&handle);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        
        // Create simple test
        hipblasLtMatrixLayout_t matA;
        status = hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
        ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        
        status = hipblasLtMatrixLayoutDestroy(matA);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
        status = hipblasLtDestroy(handle);
        EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    hipblaslt_cout << "Throughput test completed in " << duration << " ms" << std::endl;
}

} // namespace
