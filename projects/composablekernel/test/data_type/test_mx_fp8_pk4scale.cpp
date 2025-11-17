// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::bhalf8_t;
using ck::bhalf_t;
using ck::f8_ocp_t;
using ck::f8x8_ocp_t;
using ck::float8_t;
using ck::half8_t;
using ck::half_t;
using ck::type_convert;

/**
 * @brief Device version of "wave-wise FP8 to T(FP16/BF16) conversion".
 *
 * This function performs packed 8 conversions from FP8 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */

template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_convert(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }

#if CK_MX_ARCH_1250
    using T8 = typename ck::vector_type<T, 8>::type;
    // scale_sel = 1, 3, 5, 7 will use p_scale values in lane[16:31]
    ck::index_t lid = __lane_id();
    uint32_t scale  = (lid < 16) ? uint32_t(0) : p_scale[lid - 16];

    // Each iteration take care of 16 x 16 matrix
    // itr-0 use scale [7:0]
    // itr-1 use scale [23:16]
    // itr-2 use scale [15:8]
    // itr-3 use scale [31:24]
    ck::static_for<0, N / 16, 1>{}([&](auto it) {
        f8x8_ocp_t vf8{type_convert<f8_ocp_t>(Val)}; // 2.0f
        auto vT8 = ck::pk4scaled_type_convert<T8, f8x8_ocp_t, it * 2 + 1>(scale, vf8);

        // write to p_mat
        ck::static_for<0, 8, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 16 + ((lid >> 4) & 1) * 8 + ii] =
                vT8[static_cast<int>(ii)];
        });
    });
#endif
}

/* helper function to convert ith scale in packed form to a float */
static inline float convert_exponent_to_float(uint32_t exp4, int i)
{
    return ck::bit_cast<float>((exp4 >> (i * 8) & 0xFF) << 23);
}

/* Float16: "wave-wise FP8 to FP16 conversion" */
TEST(MXFP8, DevicePackedScaledConvertFP16)
{
    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 64;
    constexpr float Val = 2.0f;
    uint32_t v_scal     = (126u << 24) | (127u << 16) | (128u << 8) | (129u); //[0.5|1.|2.|4.]
    std::vector<half_t> out(M * N, -1.0f);
    std::vector<uint32_t> scale(M, v_scal);

    DeviceMem device_out(M * N * sizeof(half_t));
    DeviceMem device_scale(M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert<M, N, Val>
        <<<1, 32>>>(static_cast<half_t*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* n:  [0:15]  [16:31]  [32:47]  [48:63]
            8.0f     2.0f     4.0f    1.0f  */
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<half_t>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 16; n < 32; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<half_t>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<half_t>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 48; n < 64; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<half_t>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
    }
}

/* Bfloat16: "wave-wise FP8 to BF16 conversion" */
TEST(MXFP8, DevicePackedScaledConvertBF16)
{
    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 64;
    constexpr float Val = 2.0f;
    uint32_t v_scal     = (126u << 24) | (127u << 16) | (128u << 8) | (129u); //[0.5|1.|2.|4.]
    std::vector<bhalf_t> out(M * N, -1.0f);
    std::vector<uint32_t> scale(M, v_scal);

    DeviceMem device_out(M * N * sizeof(bhalf_t));
    DeviceMem device_scale(M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert<M, N, Val>
        <<<1, 32>>>(static_cast<bhalf_t*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* n:  [0:15]  [16:31]  [32:47]  [48:63]
            8.0f     2.0f     4.0f    1.0f  */
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<bhalf_t>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 16; n < 32; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<bhalf_t>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<bhalf_t>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 48; n < 64; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<bhalf_t>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n << std::endl;
        }
    }
}

/* Float32 "wave-wise FP8 to FP32 conversion" */
/**
 * @brief Device version of "wave-wise FP8 to FP32 conversion".
 *
 * This function performs packed 8 conversions from FP8 values to float32 values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 *
 * @param p_mat Pointer to the output array where the converted float32 values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val>
__global__ void test_packed_scaled_convert_fp32(float* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }
#if CK_MX_ARCH_1250
    ck::index_t lid = __lane_id();
    // scale_sel = 0, 2, 4, 6 will use p_scale values in lane[0:15]
    uint32_t scale = (lid < 16) ? p_scale[lid] : uint32_t(0);

    // Each iteration take care of 16 x 32 matrix
    // itr-0 use scale [7:0]
    // itr-1 use scale [23:16]
    // itr-2 use scale [15:8]
    // itr-3 use scale [31:24]
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        f8x8_ocp_t vf8_1{type_convert<f8_ocp_t>(Val)}; // 2.0f
        f8x8_ocp_t vf8_2{type_convert<f8_ocp_t>(Val)}; // 2.0f
        auto vfloat8_1 = ck::pk4scaled_type_convert<float8_t, f8x8_ocp_t, it * 2>(scale, vf8_1);
        auto vfloat8_2 = ck::pk4scaled_type_convert<float8_t, f8x8_ocp_t, it * 2>(scale, vf8_2);

        // write to p_mat
        ck::static_for<0, 8, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vfloat8_1[static_cast<int>(ii)];
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii + 8] =
                vfloat8_2[static_cast<int>(ii)];
        });
    });
#endif
}

TEST(MXFP8, DevicePackedScaledConvertFP32)
{
    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128; // Block 32 share a scale factor, 4 scale factors available
    constexpr float Val = 2.0f;
    uint32_t v_scal     = (126u << 24) | (127u << 16) | (128u << 8) | (129u); //[0.5|1.|2.|4.]
    std::vector<float> out(M * N, -1.0f);
    std::vector<uint32_t> scale(M, v_scal);

    DeviceMem device_out(M * N * sizeof(float));
    DeviceMem device_scale(M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert_fp32<M, N, Val>
        <<<1, 32>>>(static_cast<float*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    /* n:  [0:31]  [32:63]  [64:95]  [96:127]
            8.0f     2.0f     4.0f    1.0f  */
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < 32; n++)
        {
            EXPECT_EQ(out[m * N + n], convert_exponent_to_float(scale[m], 0) * Val)
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 32; n < 64; n++)
        {
            EXPECT_EQ(out[m * N + n], convert_exponent_to_float(scale[m], 2) * Val)
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 64; n < 96; n++)
        {
            EXPECT_EQ(out[m * N + n], convert_exponent_to_float(scale[m], 1) * Val)
                << "m: " << m << ", n: " << n << std::endl;
        }
        for(int n = 96; n < 128; n++)
        {
            EXPECT_EQ(out[m * N + n], convert_exponent_to_float(scale[m], 3) * Val)
                << "m: " << m << ", n: " << n << std::endl;
        }
    }
}
