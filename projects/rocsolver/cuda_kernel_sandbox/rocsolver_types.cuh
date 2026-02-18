/* **************************************************************************
 * Minimal Type Definitions for rocSOLVER Kernel Sandbox
 *
 * Provides only the types needed to compile individual rocSOLVER kernels
 * with nvcc. This is not a complete implementation - add types as needed
 * for other kernels.
 * *************************************************************************/

#pragma once

#include <cstdint>
#include <cmath>
#include <complex>
#include <type_traits>

// Core rocBLAS types
using rocblas_stride = int64_t;
using rocblas_int = int32_t;

// rocblas_handle is unused in kernel code - stub it
using rocblas_handle = void*;

// rocblas_status for return values
enum rocblas_status {
    rocblas_status_success = 0,
    rocblas_status_invalid_handle = 1,
    rocblas_status_not_implemented = 2,
    rocblas_status_invalid_pointer = 3,
    rocblas_status_invalid_size = 4,
    rocblas_status_memory_error = 5,
    rocblas_status_internal_error = 6,
    rocblas_status_perf_degraded = 7,
    rocblas_status_size_query_mismatch = 8,
    rocblas_status_size_increased = 9,
    rocblas_status_size_unchanged = 10,
    rocblas_status_invalid_value = 11,
    rocblas_status_continue = 12,
    rocblas_status_check_numerics_fail = 13,
    rocblas_status_excluded_from_build = 14,
    rocblas_status_arch_mismatch = 15,
};

// Complex number type
template <typename T>
struct rocblas_complex_num {
    T x;  // real part
    T y;  // imaginary part

    __device__ __host__ rocblas_complex_num(T r = T(0), T i = T(0)) : x(r), y(i) {}
    __device__ __host__ T real() const { return x; }
    __device__ __host__ T imag() const { return y; }
};

using rocblas_float_complex = rocblas_complex_num<float>;
using rocblas_double_complex = rocblas_complex_num<double>;

// Type trait for complex detection
template <typename T>
struct is_rocblas_complex : std::false_type {};

template <>
struct is_rocblas_complex<rocblas_float_complex> : std::true_type {};

template <>
struct is_rocblas_complex<rocblas_double_complex> : std::true_type {};

template <typename T>
inline constexpr bool rocblas_is_complex = is_rocblas_complex<T>::value;

// Namespace macros - simplified for sandbox
#define ROCSOLVER_BEGIN_NAMESPACE namespace rocsolver {
#define ROCSOLVER_END_NAMESPACE }

// Kernel macro
#define ROCSOLVER_KERNEL __global__

// Size constants for getf2
#define GETF2_SSKER_MAX_M 512
#define GETF2_SSKER_MAX_N 64
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2

// Unreachable macro
#define ROCSOLVER_UNREACHABLE() __builtin_unreachable()

// rocblas_erange - eigenvalue range specifier (for STEBZ)
enum rocblas_erange {
    rocblas_erange_all = 171,   // all eigenvalues
    rocblas_erange_value = 172, // eigenvalues in (vl, vu] interval
    rocblas_erange_index = 173  // eigenvalues with index il to iu
};

// rocblas_eorder - eigenvalue ordering (for STEBZ)
enum rocblas_eorder {
    rocblas_eorder_blocks = 181, // ordered within split blocks
    rocblas_eorder_entire = 182  // ordered across entire matrix
};

// Block size constants for STEBZ
#define STEBZ_SPLIT_THDS 256
#define IBISEC_BLKS 64
#define IBISEC_THDS 128

// Generic block size for kernel launches
#define BS1 256
