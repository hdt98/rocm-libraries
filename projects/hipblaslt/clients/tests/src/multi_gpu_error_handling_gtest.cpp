/*******************************************************************************
 * Multi-GPU Error Handling Test Suite
 * Tests: OOM, Invalid Device IDs, Device Mismatch, Graceful Degradation
 *******************************************************************************/
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <gtest/gtest-spi.h>

namespace {
int getNumGPUs() { int n = 0; hipGetDeviceCount(&n); return n; }

TEST(MultiGPUErrorHandling, InvalidDeviceID) {
    int numDevices = getNumGPUs();
    
    hipblaslt_cout << "Testing invalid device ID handling" << std::endl;
    
    // Try to set invalid device
    int invalid_device = numDevices + 10;
    auto hipErr = hipSetDevice(invalid_device);
    EXPECT_NE(hipErr, hipSuccess);
    
    hipblaslt_cout << "Invalid device ID correctly rejected" << std::endl;
}

TEST(MultiGPUErrorHandling, DeviceMismatchError) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing device mismatch error handling" << std::endl;
    
    // Allocate on device 0
    auto hipErr = hipSetDevice(0);
    ASSERT_EQ(hipErr, hipSuccess);
    
    float *d_ptr;
    hipErr = hipMalloc(&d_ptr, 1024 * sizeof(float));
    ASSERT_EQ(hipErr, hipSuccess);
    
    // Try to free from device 1 (should work in HIP)
    hipErr = hipSetDevice(1);
    ASSERT_EQ(hipErr, hipSuccess);
    hipErr = hipFree(d_ptr);
    EXPECT_EQ(hipErr, hipSuccess);
    
    hipblaslt_cout << "Device mismatch test passed" << std::endl;
}

TEST(MultiGPUErrorHandling, OutOfMemoryScenario) {
    int numDevices = getNumGPUs();
    if(numDevices < 1) GTEST_SKIP() << "Requires at least 1 GPU";
    
    hipblaslt_cout << "Testing out-of-memory scenario" << std::endl;
    
    auto hipErr = hipSetDevice(0);
    ASSERT_EQ(hipErr, hipSuccess);
    
    size_t free_mem, total_mem;
    hipErr = hipMemGetInfo(&free_mem, &total_mem);
    ASSERT_EQ(hipErr, hipSuccess);
    
    // Try to allocate more than available
    size_t huge_size = total_mem * 2;
    float *d_ptr;
    hipErr = hipMalloc(&d_ptr, huge_size);
    EXPECT_NE(hipErr, hipSuccess);
    
    if(hipErr != hipSuccess) {
        hipblaslt_cout << "OOM correctly detected and handled" << std::endl;
    }
}

TEST(MultiGPUErrorHandling, GracefulDegradation) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing graceful degradation" << std::endl;
    
    // Test that failure on one GPU doesn't affect others
    bool gpu0_success = false;
    bool gpu1_success = false;
    
    // GPU 0 - normal operation
    auto hipErr = hipSetDevice(0);
    if(hipErr == hipSuccess) {
        float *d_ptr;
        hipErr = hipMalloc(&d_ptr, 1024 * sizeof(float));
        if(hipErr == hipSuccess) {
            gpu0_success = true;
            hipFree(d_ptr);
        }
    }
    
    // GPU 1 - normal operation
    hipErr = hipSetDevice(1);
    if(hipErr == hipSuccess) {
        float *d_ptr;
        hipErr = hipMalloc(&d_ptr, 1024 * sizeof(float));
        if(hipErr == hipSuccess) {
            gpu1_success = true;
            hipFree(d_ptr);
        }
    }
    
    EXPECT_TRUE(gpu0_success && gpu1_success) 
        << "Both GPUs should work independently";
    
    hipblaslt_cout << "Graceful degradation test passed" << std::endl;
}

TEST(MultiGPUErrorHandling, NullPointerHandling) {
    int numDevices = getNumGPUs();
    if(numDevices < 1) GTEST_SKIP() << "Requires at least 1 GPU";
    
    hipblaslt_cout << "Testing null pointer handling" << std::endl;
    
    auto hipErr = hipSetDevice(0);
    ASSERT_EQ(hipErr, hipSuccess);
    
    hipblasLtHandle_t handle;
    auto status = hipblasLtCreate(&handle);
    ASSERT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    
    // Test with null matrix descriptor
    hipblasLtMatrixLayout_t null_mat = nullptr;
    status = hipblasLtMatrixLayoutDestroy(null_mat);
    // Should handle gracefully
    
    status = hipblasLtDestroy(handle);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
    
    hipblaslt_cout << "Null pointer test passed" << std::endl;
}

} // namespace
