// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/common_header.hpp"

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#include <hip/hip_runtime.h>

#if __clang_major__ >= 20
#include "ck/utility/amd_buffer_addressing_builtins.hpp"
#else
#include "ck/utility/amd_buffer_addressing.hpp"
#endif

namespace ck {
namespace prefetch_op_util {

// template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
struct GlobalPrefetchDataOp
{
    // addr needs to point to global memory!
    __device__ __forceinline__ void operator()(const void* addr) const
    {
#if defined(__gfx1250__)
        // NOTE: There's a bug in AM/GOPHER for gfx1250 when prefetching into L1, so we disable it
        // for now!
        __builtin_amdgcn_global_prefetch(
            addr,
            static_cast<index_t>(AmdBufferCoherenceEnum::GLC)); // static_cast<index_t>(coherence));
#else
        // ignore - not supported
        (void)addr;
#endif
    }
};

// template <AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
struct FlatPrefetchDataOp
{
    __device__ __forceinline__ void operator()(const void* addr) const
    {
#if defined(__gfx1250__)
        // NOTE: There's a bug in AM/GOPHER for gfx1250 when prefetching into L1, so we disable it
        // for now!
        __builtin_amdgcn_flat_prefetch(
            addr,
            static_cast<index_t>(AmdBufferCoherenceEnum::GLC)); // static_cast<index_t>(coherence));
#else
        // ignore - not supported
        (void)addr;
#endif
    }
};

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, typename PrefetchOp>
__global__ void
kernel_with_prefetch(const T* src, T* dst, const T* scalar_data, bool enable_prefetch)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate number of 64B cachelines needed to cover num_scalars elements
    constexpr index_t cachelineSize              = 64;
    constexpr index_t elements_per_cachelineSize = cachelineSize / sizeof(T);
    constexpr unsigned int cachelinesNeeded =
        (NUM_SCALARS + elements_per_cachelineSize - 1) / elements_per_cachelineSize;

    const char* byte_addr = reinterpret_cast<const char*>(scalar_data);

    // Prefetch all scalar data at once
    if(tid < cachelinesNeeded)
    {
        if(enable_prefetch)
        {
            // Prefetch the cacheline
            PrefetchOp{}(byte_addr + tid * cachelineSize);
        }
    }

    T sum = 0;
    if(tid < NUM_THREADS)
    {
        sum = src[tid]; // load from global mem to give time for prefetch to finish or be close to
                        // finishs
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < NUM_THREADS)
    {
        // Access prefetched scalar data
        for(uint32_t i = 0; i < NUM_SCALARS; i++)
        {
            sum += scalar_data[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, typename PrefetchOp>
__global__ void kernel_with_prefetch_and_shared_mem(const T* src,
                                                    T* dst,
                                                    const T* scalar_data,
                                                    bool enable_prefetch)
{
    __shared__ T sharedMem[32];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate number of 64B cachelines needed to cover num_scalars elements
    constexpr index_t cachelineSize              = 64;
    constexpr index_t elements_per_cachelineSize = cachelineSize / sizeof(T);
    constexpr unsigned int cachelinesNeeded =
        (NUM_SCALARS + elements_per_cachelineSize - 1) / elements_per_cachelineSize;

    bool use_shared_mem = tid % 2 == 1;

    const void* byte_addr;
    if(use_shared_mem)
    {
        byte_addr = reinterpret_cast<const void*>(sharedMem);
    }
    else
    {
        uintptr_t base   = reinterpret_cast<uintptr_t>(scalar_data);
        uintptr_t offset = base + (tid / 2) * cachelineSize;
        byte_addr        = reinterpret_cast<const void*>(offset);
    }

    // Prefetch all scalar data at once
    if(tid < cachelinesNeeded * 2)
    {
        if(enable_prefetch)
        {
            // Prefetch the cacheline
            PrefetchOp{}(byte_addr);
        }
        else
        {
            (void)byte_addr;
        }
    }

    T sum = 0;
    if(tid < NUM_THREADS)
    {
        sum = src[tid]; // load from global mem to give time for prefetch to finish or be close to
                        // finishs
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < NUM_THREADS)
    {
        // Access prefetched scalar data
        for(uint32_t i = 0; i < NUM_SCALARS; i++)
        {
            sum += scalar_data[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename PrefetchKernel, typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool test_prefetch_impl(bool time_kernels,
                        const PrefetchKernel& prefetch_kernel,
                        const std::string& kernel_name)
{
    // TODO: maybe add more prefetch instructions inside kernel to support more values
    assert(NUM_SCALARS / sizeof(T) < (128 * 32));
    constexpr index_t num_elements = NUM_THREADS;
    constexpr index_t num_scalars  = NUM_SCALARS;
    constexpr index_t block_size   = 256;
    constexpr index_t grid_size    = (num_elements + block_size - 1) / block_size;

    std::cout << "Testing " << kernel_name << " to L1/L2 cache for type: " << typeid(T).name()
              << std::endl;
    std::cout << "Elements: " << num_elements << ", Scalars: " << num_scalars << std::endl;

    // Host data
    std::vector<T> h_src(num_elements);
    std::vector<T> h_scalar(num_scalars);
    std::vector<T> h_dst_with_prefetch_chunks(num_elements);
    std::vector<T> h_expected(num_elements);

    // Initialize data
    for(index_t i = 0; i < num_elements; i++)
    {
        h_src[i] = static_cast<T>(i % 100);
    }

    T scalar_sum = 0;
    for(index_t i = 0; i < num_scalars; i++)
    {
        h_scalar[i] = static_cast<T>(i + 1);
        scalar_sum += h_scalar[i];
    }

    // Expected results
    for(index_t i = 0; i < num_elements; i++)
    {
        h_expected[i] = h_src[i] + scalar_sum;
    }

    // Device memory
    DeviceMem d_src(sizeof(T) * num_elements);
    DeviceMem d_scalar(sizeof(T) * num_scalars);
    DeviceMem d_dst_with_prefetch_chunks(sizeof(T) * num_elements);

    d_src.ToDevice(h_src.data());
    d_scalar.ToDevice(h_scalar.data());

    hipStream_t stream;
    hip_check_error(hipStreamCreate(&stream));

    if(time_kernels)
    {
        ck::static_for<0, 2, 1>{}([&](auto static_i) {
            constexpr bool prefetch_enabled = static_i == 0;
            std::cout << "PREFETCH " << (prefetch_enabled ? "ENABLED!" : "DISABLED!") << std::endl;

            constexpr int num_warmup     = 1;
            constexpr int num_iterations = 10;

            // Warmup runs
            for(int i = 0; i < num_warmup; i++)
            {
                prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const T*>(d_src.GetDeviceBuffer()),
                    static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
                    static_cast<const T*>(d_scalar.GetDeviceBuffer()),
                    prefetch_enabled);
            }
            hip_check_error(hipStreamSynchronize(stream));

            // Performance measurement
            hipEvent_t start, stop;
            hip_check_error(hipEventCreate(&start));
            hip_check_error(hipEventCreate(&stop));

            hip_check_error(hipEventRecord(start, stream));
            for(int i = 0; i < num_iterations; i++)
            {
                prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
                    static_cast<const T*>(d_src.GetDeviceBuffer()),
                    static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
                    static_cast<const T*>(d_scalar.GetDeviceBuffer()),
                    prefetch_enabled);
            }
            hip_check_error(hipEventRecord(stop, stream));

            hip_check_error(hipStreamSynchronize(stream));

            float elapsed_ms = 0;
            hip_check_error(hipEventElapsedTime(&elapsed_ms, start, stop));

            float avg_time_us       = (elapsed_ms * 1000.0f) / num_iterations;
            float total_bytes       = (num_elements * sizeof(T) + num_scalars * sizeof(T)); // read
            float bandwidth_gb_s    = (total_bytes / (avg_time_us * 1e-6)) / 1e9;
            float ops_per_iteration = num_elements * num_scalars; // adds
            float gflops            = (ops_per_iteration / (avg_time_us * 1e-6)) / 1e9;

            std::cout << "  Performance: " << std::endl;
            std::cout << "    Average kernel time: " << avg_time_us << " us" << std::endl;
            std::cout << "    Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
            std::cout << "    Compute throughput: " << gflops << " GFLOPS" << std::endl;

            hip_check_error(hipEventDestroy(start));
            hip_check_error(hipEventDestroy(stop));
        });
    }
    else
    {
        prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const T*>(d_src.GetDeviceBuffer()),
            static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
            static_cast<const T*>(d_scalar.GetDeviceBuffer()),
            true);

        hip_check_error(hipStreamSynchronize(stream));
    }

    // Copy results back
    d_dst_with_prefetch_chunks.FromDevice(h_dst_with_prefetch_chunks.data());

    // Verify results
    bool pass = ck::utils::check_err(h_dst_with_prefetch_chunks, h_expected);

    std::cout << "  Correctness: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    hip_check_error(hipStreamDestroy(stream));

    return pass;
}

} // namespace prefetch_op_util
} // namespace ck
