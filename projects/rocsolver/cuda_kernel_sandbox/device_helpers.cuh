/* **************************************************************************
 * Device Helper Functions for rocSOLVER Kernel Sandbox
 *
 * Contains device utility functions used by rocSOLVER kernels:
 * - aabs: absolute value (works for complex)
 * - swap: exchange two values
 * - load_ptr_batch: load pointer for batched operations
 * *************************************************************************/

#pragma once

#include "rocsolver_types.cuh"

ROCSOLVER_BEGIN_NAMESPACE

// Absolute value helper for real types
template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val);
}

// Absolute value helper for complex types (returns sum of abs of real and imag)
template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val.real()) + std::abs(val.imag());
}

// Swap helper
template <typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

// Swap vector elements (used by some kernels)
template <typename T>
__device__ __host__ void
    swap(const rocblas_int n, T* a, const rocblas_int inca, T* b, const rocblas_int incb)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        swap(a[inca * tid], b[incb * tid]);
}

// Load pointer for batched operations (strided version)
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
load_ptr_batch(T* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p + block * stride + offset;
}

// Load pointer for batched operations (array of pointers version)
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
load_ptr_batch(T* const* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

template <typename T, typename I>
__forceinline__ __device__ __host__ T*
load_ptr_batch(T** p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

// CUDA warpSize is 32 (vs AMD's 64)
#ifndef warpSize
#define warpSize 32
#endif

// Device-compatible min/max (std::min/max are host-only without --expt-relaxed-constexpr)
template <typename T>
__device__ __host__ __forceinline__ T device_min(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
__device__ __host__ __forceinline__ T device_max(T a, T b)
{
    return (a > b) ? a : b;
}

// Device-compatible abs
template <typename T>
__device__ __host__ __forceinline__ T device_abs(T a)
{
    return (a < T(0)) ? -a : a;
}

// Device-compatible log (use CUDA intrinsics)
__device__ __host__ __forceinline__ float device_log(float a)
{
#ifdef __CUDA_ARCH__
    return logf(a);
#else
    return std::log(a);
#endif
}

__device__ __host__ __forceinline__ double device_log(double a)
{
#ifdef __CUDA_ARCH__
    return log(a);
#else
    return std::log(a);
#endif
}

/** IAMAX finds the maximum element of a given vector.
    MAX_THDS should be 128, 256, 512, or 1024, and sval should
    be a shared array of size MAX_THDS. **/
template <int MAX_THDS, typename T, typename I, typename S>
__device__ void iamax(const I tid, const I n, T* A, const I incA, S* sval)
{
    // local memory setup
    S val1, val2;

    // read into shared memory while doing initial step
    // (each thread reduce as many elements as needed to cover the original array)
    val1 = 0;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        val2 = aabs<S>(A[i * incA]);
        if(val1 < val2)
            val1 = val2;
    }
    sval[tid] = val1;
    __syncthreads();

    if(n <= 1)
        return;

    // Reduction on the shared memory array
    // (We halve the number of active threads at each step)
#pragma unroll
    for(I i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
        {
            val2 = sval[tid + i];
            if(val1 < val2)
                sval[tid] = val1 = val2;
        }
        __syncthreads();
    }

    // Final warp reduction (no sync needed within a warp)
    if(tid < warpSize)
    {
        if(warpSize >= 64)
        {
            val2 = sval[tid + 64];
            if(val1 < val2)
                sval[tid] = val1 = val2;
        }
        val2 = sval[tid + 32];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 16];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 8];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 4];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 2];
        if(val1 < val2)
            sval[tid] = val1 = val2;
        val2 = sval[tid + 1];
        if(val1 < val2)
            sval[tid] = val1 = val2;
    }
    // after the reduction, the maximum of the elements is in sval[0]
}

/** IAMAX with index: finds the maximum element and its index.
    MAX_THDS should be 64, 128, 256, 512, or 1024, and sval and sidx should
    be shared arrays of size MAX_THDS. **/
template <int MAX_THDS, typename T, typename I, typename S>
__device__ void iamax(const I tid, const I n, T* A, const I incA, S* sval, I* sidx)
{
    S val1, val2;
    I idx1, idx2;

    val1 = 0;
    idx1 = INT_MAX;
    for(I i = tid; i < n; i += MAX_THDS)
    {
        val2 = aabs<S>(A[i * incA]);
        idx2 = i + 1; // 1-based index
        if(val1 < val2 || idx1 == INT_MAX)
        {
            val1 = val2;
            idx1 = idx2;
        }
    }
    sval[tid] = val1;
    sidx[tid] = idx1;
    __syncthreads();

    if(n <= 1)
        return;

#pragma unroll
    for(I i = MAX_THDS / 2; i > warpSize; i /= 2)
    {
        if(tid < i)
        {
            val2 = sval[tid + i];
            idx2 = sidx[tid + i];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        __syncthreads();
    }

    if(tid < warpSize)
    {
        if(warpSize >= 64 && MAX_THDS >= 128)
        {
            val2 = sval[tid + 64];
            idx2 = sidx[tid + 64];
            if((val1 < val2) || (val1 == val2 && idx1 > idx2))
            {
                sval[tid] = val1 = val2;
                sidx[tid] = idx1 = idx2;
            }
        }
        val2 = sval[tid + 32];
        idx2 = sidx[tid + 32];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 16];
        idx2 = sidx[tid + 16];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 8];
        idx2 = sidx[tid + 8];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 4];
        idx2 = sidx[tid + 4];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 2];
        idx2 = sidx[tid + 2];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
        val2 = sval[tid + 1];
        idx2 = sidx[tid + 1];
        if((val1 < val2) || (val1 == val2 && idx1 > idx2))
        {
            sval[tid] = val1 = val2;
            sidx[tid] = idx1 = idx2;
        }
    }
    // after the reduction, the maximum and its index are in sval[0] and sidx[0]
}

// Kernel to reset info array to a given value
template <typename T, typename S>
ROCSOLVER_KERNEL void reset_info(T* info, const S n, const T val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
        info[idx] = val;
}

// Helper to get machine epsilon
template <typename T>
__device__ __host__ T get_epsilon()
{
    // Single precision epsilon
    if constexpr(std::is_same_v<T, float>)
        return 1.19209290e-07f;
    // Double precision epsilon
    else if constexpr(std::is_same_v<T, double>)
        return 2.2204460492503131e-16;
    else
        return T(0);
}

// Helper to get safe minimum (1/sfmin doesn't overflow)
template <typename T>
__device__ __host__ T get_safemin()
{
    if constexpr(std::is_same_v<T, float>)
        return 1.17549435e-38f;
    else if constexpr(std::is_same_v<T, double>)
        return 2.2250738585072014e-308;
    else
        return T(0);
}

ROCSOLVER_END_NAMESPACE
