/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"

#include "stride_array.hpp"

// hcc seems need __device__ __host__ together to compile, and no extern "C"
typedef union value_bf16_fp32_t
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} value_bf16_fp32_t;

inline __device__ __host__ float convert_bf16_to_fp32(ushort src_val)
{
    value_bf16_fp32_t target_val;
    target_val.ushortx2 = make_ushort2(0, src_val);
    return target_val.f32;
}

inline __device__ __host__ ushort convert_fp32_to_bf16(float src_val)
{
    value_bf16_fp32_t target_val;
    target_val.f32 = src_val;

    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
    }
    return target_val.ushortvec[1];
}

template <typename src_data_t, typename dst_data_t, int use_tf32 = 0>
inline __device__ __host__ dst_data_t cast_to(const src_data_t& val)
{
    return static_cast<dst_data_t>(val);
}
template <>
inline __device__ __host__ ushort cast_to(const double& val)
{
    return convert_fp32_to_bf16(static_cast<float>(val));
}
template <>
inline __device__ __host__ double cast_to(const ushort& val)
{
    return static_cast<double>(convert_bf16_to_fp32(val));
}
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
