// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif
#include <hip/hip_bfloat16.h>

#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"

#include "stride_array.hpp"

template <typename src_data_t, typename dst_data_t, int use_tf32 = 0>
inline __device__ __host__ dst_data_t cast_to(const src_data_t& val)
{
    return static_cast<dst_data_t>(val);
}
// __hip_bfloat16 ↔ double
template <>
inline __device__ __host__ __hip_bfloat16 cast_to(const double& val)
{
    return __hip_bfloat16(static_cast<float>(val));
}
template <>
inline __device__ __host__ double cast_to(const __hip_bfloat16& val)
{
    return static_cast<double>(static_cast<float>(val));
}
// half ↔ double
template <>
inline __device__ __host__ half cast_to(const double& val)
{
    return __float2half(static_cast<float>(val));
}
template <>
inline __device__ __host__ double cast_to(const half& val)
{
    return static_cast<double>(__half2float(val));
}
// __hip_bfloat16 ↔ float
template <>
inline __device__ __host__ __hip_bfloat16 cast_to(const float& val)
{
    return __hip_bfloat16(val);
}
template <>
inline __device__ __host__ float cast_to(const __hip_bfloat16& val)
{
    return static_cast<float>(val);
}
// half ↔ float
template <>
inline __device__ __host__ half cast_to(const float& val)
{
    return __float2half(val);
}
template <>
inline __device__ __host__ float cast_to(const half& val)
{
    return __half2float(val);
}

template <>
inline __device__ __host__ int8_t cast_to(const int32_t& val)
{
    return static_cast<int8_t>(val & 0xff);
}

template <>
inline __device__ __host__ float cast_to<float, float, 1>(const float& val)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {val};

    u.int32 = u.int32 & 0xffffe000;
    return u.fp32;
}

template <>
inline __device__ __host__ double cast_to<float, double, 1>(const float& val)
{
    return static_cast<double>(cast_to<float, float, 1>(val));
}

inline __device__ __host__ bool IsZero(double val) { return val == 0.0; }

inline __device__ __host__ bool IsOne(double val) { return val == 1.0; }

// Type trait: does the target GPU provide a native atomicAdd for this type?
// float/double are universally supported. half/__hip_bfloat16 require gfx90a+
// (CDNA2) or RDNA3+. The solver passes -DNAIVE_CONV_HAS_16BIT_FLOAT_ATOMIC=1
// when the target GPU supports 16-bit float atomics. Using if constexpr on
// this trait prevents the compiler from instantiating atomicAdd for
// unsupported types.
template <typename T>
struct has_native_atomic_add
{
    static constexpr bool value = false;
};
template <>
struct has_native_atomic_add<float>
{
    static constexpr bool value = true;
};
template <>
struct has_native_atomic_add<double>
{
    static constexpr bool value = true;
};
#if NAIVE_CONV_HAS_16BIT_FLOAT_ATOMIC
template <>
struct has_native_atomic_add<half>
{
    static constexpr bool value = true;
};
template <>
struct has_native_atomic_add<__hip_bfloat16>
{
    static constexpr bool value = true;
};
#endif

// Precompute valid filter range for FWD (and WRW spatial checks).
//
// For FWD: cur = stride * o_pos - pad + dil * iy, need 0 <= cur < i_size.
// Returns iy_start, iy_end (inclusive). Loop: for(iy = iy_start; iy <= iy_end; iy++)
// If no valid positions, iy_start > iy_end.
inline __device__ __host__ void fwd_filter_range(
    int o_pos, int pad, int dil, int stride, int f, int i_size, int& iy_start, int& iy_end)
{
    int tmp = stride * o_pos - pad;
    // cur = tmp + dil * iy >= 0  →  iy >= ceil(-tmp / dil) when tmp < 0
    if(tmp >= 0)
        iy_start = 0;
    else
        iy_start = (-tmp + dil - 1) / dil;

    // cur = tmp + dil * iy < i_size  →  iy <= (i_size - 1 - tmp) / dil
    int upper = i_size - 1 - tmp;
    if(upper < 0)
        iy_end = -1;
    else
        iy_end = upper / dil;
    if(iy_end >= f)
        iy_end = f - 1;
}

// Compute gcd for BWD-d filter range precomputation
inline __device__ __host__ int device_gcd(int a, int b)
{
    while(b != 0)
    {
        int t = b;
        b     = a % b;
        a     = t;
    }
    return a;
}

// Precompute the valid filter range for one spatial dimension in BWD-d.
//
// For BWD-d, given input position `i_pos`, padding `pad`, dilation `dil`,
// stride `s`, filter size `f`, and output size `o`, we need filter positions
// `iy` such that:
//   (1) cur_o = (i_pos + pad - dil * iy)  is non-negative
//   (2) cur_o % s == 0                    (stride alignment)
//   (3) cur_o / s < o                     (within output bounds)
//
// Returns: iy_start, iy_end (inclusive), iy_step — loop as:
//   for(int iy = iy_start; iy <= iy_end; iy += iy_step)
//       cur_o = (tmp - dil * iy) / s;    // guaranteed valid, no checks needed
//
// If no valid positions exist, iy_start > iy_end.
inline __device__ __host__ void bwd_filter_range(
    int i_pos, int pad, int dil, int s, int f, int o, int& iy_start, int& iy_end, int& iy_step)
{
    int tmp = i_pos + pad;

    // Raw bounds from non-negativity and output-range constraints
    // (1) tmp - dil*iy >= 0            → iy <= tmp / dil
    // (3) (tmp - dil*iy) / s <= o - 1  → tmp - dil*iy <= (o-1)*s → iy >= (tmp - (o-1)*s) / dil
    int iy_lo = (tmp - (o - 1) * s);
    // Ceiling division for positive result: ceil(a/b) when a may be negative
    iy_lo = (iy_lo > 0) ? (iy_lo + dil - 1) / dil : 0;
    iy_lo = (iy_lo < 0) ? 0 : iy_lo;

    int iy_hi = tmp / dil;
    if(iy_hi >= f)
        iy_hi = f - 1;

    // (2) Stride alignment: need (tmp - dil * iy) % s == 0
    int g   = device_gcd(dil, s);
    iy_step = s / g;

    // Find first iy >= iy_lo that satisfies (tmp - dil * iy) % s == 0
    int remainder = tmp % s;
    // We need dil * iy ≡ remainder (mod s)
    // Since g = gcd(dil, s), this has a solution iff g | remainder
    if(remainder % g != 0)
    {
        // No valid positions at all
        iy_start = 1;
        iy_end   = 0;
        return;
    }

    // Find the smallest non-negative iy satisfying the congruence
    // dil * iy ≡ remainder (mod s)
    // → (dil/g) * iy ≡ (remainder/g) (mod s/g)
    int s_red = s / g;
    int d_red = (dil / g) % s_red;
    int r_red = (remainder / g) % s_red;
    // Brute force for small s_red (s is typically 1-4)
    int base = 0;
    for(int t = 0; t < s_red; t++)
    {
        if((d_red * t) % s_red == r_red)
        {
            base = t;
            break;
        }
    }

    // First valid iy at or above iy_lo
    if(base < iy_lo)
    {
        int skip = (iy_lo - base + iy_step - 1) / iy_step;
        base += skip * iy_step;
    }
    iy_start = base;
    iy_end   = iy_hi;
}

template <typename dst_data_t, typename acc_data_t>
inline __device__ void applyalphaBetaUpdate(dst_data_t* __restrict__ p_array,
                                            const acc_data_t value,
                                            double alpha,
                                            double beta,
                                            size_t index)
{
    if(IsOne(alpha) && IsZero(beta))
    {
        p_array[index] = cast_to<acc_data_t, dst_data_t>(value);
        return;
    }
    // cast_to<src, dst>
    acc_data_t val_alpha_beta =
        cast_to<double, acc_data_t>(alpha) * value +
        cast_to<dst_data_t, acc_data_t>(p_array[index]) * cast_to<double, acc_data_t>(beta);
    p_array[index] = cast_to<acc_data_t, dst_data_t>(val_alpha_beta);
}

/// \todo remove template parameter 'bool ASSUME_PACKED' in a follow up PR
/// --amberhassaan
/// Notes (Amber):
/// - The following code used to assume that group (G) is an implicit
/// dimension, i.e. c= c_per_group * group and k = k_per_group * group. This is not
/// true for non-packed case because group (G) dimension needs to have its stride
/// explicitly specified for address math to make sense. This is also how
/// composable_kernel (CK) treats G dimension. Which is why nchw should be ngchw,
/// and nhwc should be nhwgc. Same follows for the 3D case.
///
/// - strides here are stored right to left, i.e., for NHWC, stride for N is
/// at index 3 while stride for C is at index 0. This is different from how the
/// tensor descriptors store strides, which is always NCHW order, left-to-right.

/// alpha and beta are double to ensure high precision.

#define DEFINE_2D_NAIVE_CONV_KERNEL(                                                                                     \
    direction, tensor_layout, src_data_t, acc_data_t, dst_data_t, use_tf32)                                              \
    extern "C" __global__ void                                                                                           \
        naive_conv_ab_packed_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t##_##use_tf32(    \
            src_data_t* __restrict__ p_in,                                                                               \
            src_data_t* __restrict__ p_wei,                                                                              \
            double alpha,                                                                                                \
            double beta,                                                                                                 \
            dst_data_t* __restrict__ p_out,                                                                              \
            Strides5D in_strides,                                                                                        \
            Strides5D wei_strides,                                                                                       \
            Strides5D out_strides,                                                                                       \
            int hi,                                                                                                      \
            int wi,                                                                                                      \
            int n,                                                                                                       \
            int k_per_group,                                                                                             \
            int c_per_group,                                                                                             \
            int ho,                                                                                                      \
            int wo,                                                                                                      \
            int sy,                                                                                                      \
            int sx,                                                                                                      \
            int dy,                                                                                                      \
            int dx,                                                                                                      \
            int py,                                                                                                      \
            int px,                                                                                                      \
            int fy,                                                                                                      \
            int fx,                                                                                                      \
            int group)                                                                                                   \
    {                                                                                                                    \
        naive_conv_##direction##_##tensor_layout<true,                                                                   \
                                                 src_data_t,                                                             \
                                                 acc_data_t,                                                             \
                                                 dst_data_t,                                                             \
                                                 use_tf32>(p_in,                                                         \
                                                           p_wei,                                                        \
                                                           alpha,                                                        \
                                                           beta,                                                         \
                                                           p_out,                                                        \
                                                           in_strides,                                                   \
                                                           wei_strides,                                                  \
                                                           out_strides,                                                  \
                                                           hi,                                                           \
                                                           wi,                                                           \
                                                           n,                                                            \
                                                           k_per_group,                                                  \
                                                           c_per_group,                                                  \
                                                           ho,                                                           \
                                                           wo,                                                           \
                                                           sy,                                                           \
                                                           sx,                                                           \
                                                           dy,                                                           \
                                                           dx,                                                           \
                                                           py,                                                           \
                                                           px,                                                           \
                                                           fy,                                                           \
                                                           fx,                                                           \
                                                           group);                                                       \
    }                                                                                                                    \
    extern "C" __global__ void                                                                                           \
        naive_conv_ab_nonpacked_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t##_##use_tf32( \
            src_data_t* __restrict__ p_in,                                                                               \
            src_data_t* __restrict__ p_wei,                                                                              \
            double alpha,                                                                                                \
            double beta,                                                                                                 \
            dst_data_t* __restrict__ p_out,                                                                              \
            Strides5D in_strides,                                                                                        \
            Strides5D wei_strides,                                                                                       \
            Strides5D out_strides,                                                                                       \
            int hi,                                                                                                      \
            int wi,                                                                                                      \
            int n,                                                                                                       \
            int k_per_group,                                                                                             \
            int c_per_group,                                                                                             \
            int ho,                                                                                                      \
            int wo,                                                                                                      \
            int sy,                                                                                                      \
            int sx,                                                                                                      \
            int dy,                                                                                                      \
            int dx,                                                                                                      \
            int py,                                                                                                      \
            int px,                                                                                                      \
            int fy,                                                                                                      \
            int fx,                                                                                                      \
            int group)                                                                                                   \
    {                                                                                                                    \
        naive_conv_##direction##_##tensor_layout<false,                                                                  \
                                                 src_data_t,                                                             \
                                                 acc_data_t,                                                             \
                                                 dst_data_t,                                                             \
                                                 use_tf32>(p_in,                                                         \
                                                           p_wei,                                                        \
                                                           alpha,                                                        \
                                                           beta,                                                         \
                                                           p_out,                                                        \
                                                           in_strides,                                                   \
                                                           wei_strides,                                                  \
                                                           out_strides,                                                  \
                                                           hi,                                                           \
                                                           wi,                                                           \
                                                           n,                                                            \
                                                           k_per_group,                                                  \
                                                           c_per_group,                                                  \
                                                           ho,                                                           \
                                                           wo,                                                           \
                                                           sy,                                                           \
                                                           sx,                                                           \
                                                           dy,                                                           \
                                                           dx,                                                           \
                                                           py,                                                           \
                                                           px,                                                           \
                                                           fy,                                                           \
                                                           fx,                                                           \
                                                           group);                                                       \
    }

#define DEFINE_3D_NAIVE_CONV_KERNEL(                                                                                     \
    direction, tensor_layout, src_data_t, acc_data_t, dst_data_t, use_tf32)                                              \
    extern "C" __global__ void                                                                                           \
        naive_conv_ab_packed_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t##_##use_tf32(    \
            src_data_t* __restrict__ p_in,                                                                               \
            src_data_t* __restrict__ p_wei,                                                                              \
            double alpha,                                                                                                \
            double beta,                                                                                                 \
            dst_data_t* __restrict__ p_out,                                                                              \
            Strides6D in_strides,                                                                                        \
            Strides6D wei_strides,                                                                                       \
            Strides6D out_strides,                                                                                       \
            int di,                                                                                                      \
            int hi,                                                                                                      \
            int wi,                                                                                                      \
            int n,                                                                                                       \
            int k_per_group,                                                                                             \
            int c_per_group,                                                                                             \
            int do_,                                                                                                     \
            int ho,                                                                                                      \
            int wo,                                                                                                      \
            int sz,                                                                                                      \
            int sy,                                                                                                      \
            int sx,                                                                                                      \
            int dz,                                                                                                      \
            int dy,                                                                                                      \
            int dx,                                                                                                      \
            int pz,                                                                                                      \
            int py,                                                                                                      \
            int px,                                                                                                      \
            int fz,                                                                                                      \
            int fy,                                                                                                      \
            int fx,                                                                                                      \
            int group)                                                                                                   \
    {                                                                                                                    \
        naive_conv_##direction##_##tensor_layout<true,                                                                   \
                                                 src_data_t,                                                             \
                                                 acc_data_t,                                                             \
                                                 dst_data_t,                                                             \
                                                 use_tf32>(p_in,                                                         \
                                                           p_wei,                                                        \
                                                           alpha,                                                        \
                                                           beta,                                                         \
                                                           p_out,                                                        \
                                                           in_strides,                                                   \
                                                           wei_strides,                                                  \
                                                           out_strides,                                                  \
                                                           di,                                                           \
                                                           hi,                                                           \
                                                           wi,                                                           \
                                                           n,                                                            \
                                                           k_per_group,                                                  \
                                                           c_per_group,                                                  \
                                                           do_,                                                          \
                                                           ho,                                                           \
                                                           wo,                                                           \
                                                           sz,                                                           \
                                                           sy,                                                           \
                                                           sx,                                                           \
                                                           dz,                                                           \
                                                           dy,                                                           \
                                                           dx,                                                           \
                                                           pz,                                                           \
                                                           py,                                                           \
                                                           px,                                                           \
                                                           fz,                                                           \
                                                           fy,                                                           \
                                                           fx,                                                           \
                                                           group);                                                       \
    }                                                                                                                    \
    extern "C" __global__ void                                                                                           \
        naive_conv_ab_nonpacked_##direction##_##tensor_layout##_##src_data_t##_##acc_data_t##_##dst_data_t##_##use_tf32( \
            src_data_t* __restrict__ p_in,                                                                               \
            src_data_t* __restrict__ p_wei,                                                                              \
            double alpha,                                                                                                \
            double beta,                                                                                                 \
            dst_data_t* __restrict__ p_out,                                                                              \
            Strides6D in_strides,                                                                                        \
            Strides6D wei_strides,                                                                                       \
            Strides6D out_strides,                                                                                       \
            int di,                                                                                                      \
            int hi,                                                                                                      \
            int wi,                                                                                                      \
            int n,                                                                                                       \
            int k_per_group,                                                                                             \
            int c_per_group,                                                                                             \
            int do_,                                                                                                     \
            int ho,                                                                                                      \
            int wo,                                                                                                      \
            int sz,                                                                                                      \
            int sy,                                                                                                      \
            int sx,                                                                                                      \
            int dz,                                                                                                      \
            int dy,                                                                                                      \
            int dx,                                                                                                      \
            int pz,                                                                                                      \
            int py,                                                                                                      \
            int px,                                                                                                      \
            int fz,                                                                                                      \
            int fy,                                                                                                      \
            int fx,                                                                                                      \
            int group)                                                                                                   \
    {                                                                                                                    \
        naive_conv_##direction##_##tensor_layout<false,                                                                  \
                                                 src_data_t,                                                             \
                                                 acc_data_t,                                                             \
                                                 dst_data_t,                                                             \
                                                 use_tf32>(p_in,                                                         \
                                                           p_wei,                                                        \
                                                           alpha,                                                        \
                                                           beta,                                                         \
                                                           p_out,                                                        \
                                                           in_strides,                                                   \
                                                           wei_strides,                                                  \
                                                           out_strides,                                                  \
                                                           di,                                                           \
                                                           hi,                                                           \
                                                           wi,                                                           \
                                                           n,                                                            \
                                                           k_per_group,                                                  \
                                                           c_per_group,                                                  \
                                                           do_,                                                          \
                                                           ho,                                                           \
                                                           wo,                                                           \
                                                           sz,                                                           \
                                                           sy,                                                           \
                                                           sx,                                                           \
                                                           dz,                                                           \
                                                           dy,                                                           \
                                                           dx,                                                           \
                                                           pz,                                                           \
                                                           py,                                                           \
                                                           px,                                                           \
                                                           fz,                                                           \
                                                           fy,                                                           \
                                                           fx,                                                           \
                                                           group);                                                       \
    }
