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

ROCSOLVER_END_NAMESPACE
