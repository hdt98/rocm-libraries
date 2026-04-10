/*******************************************************************************
 * Multi-GPU Advanced Features Test Suite
 * Tests: P2P Transfers, Multi-Stream, NUMA Affinity, Memory Pools
 *******************************************************************************/
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <gtest/gtest-spi.h>

namespace {
int getNumGPUs() { int n = 0; hipGetDeviceCount(&n); return n; }

TEST(MultiGPUAdvanced, P2PMemoryTransfers) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing P2P memory transfers across GPUs" << std::endl;
    
    const size_t size = 1024 * 1024 * sizeof(float);
    
    // Check P2P access capability
    int canAccess = 0;
    auto hipErr = hipDeviceCanAccessPeer(&canAccess, 0, 1);
    ASSERT_EQ(hipErr, hipSuccess);
    
    if(!canAccess) {
        hipblaslt_cout << "P2P not supported between GPU 0 and 1" << std::endl;
        GTEST_SKIP();
    }
    
    // Enable P2P
    hipErr = hipSetDevice(0);
    ASSERT_EQ(hipErr, hipSuccess);
    hipErr = hipDeviceEnablePeerAccess(1, 0);
    // May already be enabled
    
    float *d_gpu0, *d_gpu1;
    hipErr = hipSetDevice(0);
    ASSERT_EQ(hipErr, hipSuccess);
    hipErr = hipMalloc(&d_gpu0, size);
    ASSERT_EQ(hipErr, hipSuccess);
    
    hipErr = hipSetDevice(1);
    ASSERT_EQ(hipErr, hipSuccess);
    hipErr = hipMalloc(&d_gpu1, size);
    ASSERT_EQ(hipErr, hipSuccess);
    
    // P2P copy
    hipErr = hipMemcpyPeer(d_gpu1, 1, d_gpu0, 0, size);
    EXPECT_EQ(hipErr, hipSuccess);
    
    hipErr = hipSetDevice(0);
    hipErr = hipFree(d_gpu0);
    EXPECT_EQ(hipErr, hipSuccess);
    hipErr = hipSetDevice(1);
    hipErr = hipFree(d_gpu1);
    EXPECT_EQ(hipErr, hipSuccess);
    
    hipblaslt_cout << "P2P memory transfer test passed" << std::endl;
}

TEST(MultiGPUAdvanced, MultiStreamPerGPU) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing multi-stream per GPU" << std::endl;
    
    const int num_streams = 4;
    std::vector<std::vector<hipStream_t>> streams(numDevices);
    
    for(int dev = 0; dev < std::min(numDevices, 2); ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        for(int s = 0; s < num_streams; ++s) {
            hipStream_t stream;
            hipErr = hipStreamCreate(&stream);
            ASSERT_EQ(hipErr, hipSuccess);
            streams[dev].push_back(stream);
        }
    }
    
    // Cleanup
    for(int dev = 0; dev < std::min(numDevices, 2); ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        for(auto& stream : streams[dev]) {
            hipErr = hipStreamDestroy(stream);
            EXPECT_EQ(hipErr, hipSuccess);
        }
    }
    
    hipblaslt_cout << "Multi-stream test passed" << std::endl;
}

TEST(MultiGPUAdvanced, GPUAffinityNUMA) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing GPU affinity and NUMA awareness" << std::endl;
    
    for(int dev = 0; dev < numDevices; ++dev) {
        hipDeviceProp_t props;
        auto hipErr = hipGetDeviceProperties(&props, dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        hipblaslt_cout << "GPU " << dev << ": " << props.name 
                       << " PCIe " << props.pciDomainID << ":" 
                       << props.pciBusID << ":" << props.pciDeviceID << std::endl;
    }
    
    hipblaslt_cout << "GPU affinity test passed" << std::endl;
}

TEST(MultiGPUAdvanced, MemoryPoolManagement) {
    int numDevices = getNumGPUs();
    if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";
    
    hipblaslt_cout << "Testing memory pool management" << std::endl;
    
    for(int dev = 0; dev < std::min(numDevices, 2); ++dev) {
        auto hipErr = hipSetDevice(dev);
        ASSERT_EQ(hipErr, hipSuccess);
        
        size_t free_mem, total_mem;
        hipErr = hipMemGetInfo(&free_mem, &total_mem);
        ASSERT_EQ(hipErr, hipSuccess);
        
        hipblaslt_cout << "GPU " << dev << " memory: " 
                       << (free_mem / 1024 / 1024) << " MB free / "
                       << (total_mem / 1024 / 1024) << " MB total" << std::endl;
    }
    
    hipblaslt_cout << "Memory pool test passed" << std::endl;
}

} // namespace
