// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::bhalf8_t;
using ck::bhalf_t;
using ::ck::DeviceMem;
using ck::f4_t;
using ck::f4x8_t;
using ck::float8_t;
using ck::half8_t;
using ck::half_t;
using ck::type_convert;

/* helper function to convert ith scale in packed form to a float */
static inline float convert_exponent_to_float(uint32_t exp4, int i)
{
    return ck::bit_cast<float>((exp4 >> (i * 8) & 0xFF) << 23);
}

/**
 * @brief Device version of "wave-wise FP4 to FP32 conversion".
 *
 * This function performs packed 8 conversions from FP4 values to T values in a wave.
 * One packed scale parameter can hold scale factor for 4 conversion calls.
 * See how template parameter Scale_sel used to select scale in the packed form.
 *
 * @param p_mat Pointer to the output array where the converted T values will be stored.
 * @param p_scale Pointer to the scale array.
 *
 */
template <int M, int N, float Val, typename T>
__global__ void test_packed_scaled_convert_fp32(T* p_mat, uint32_t* p_scale)
{
    if(p_mat == nullptr || p_scale == nullptr)
    {
        return;
    }
#if CK_MX_ARCH_1250
    using T8        = typename ck::vector_type<T, 8>::type;
    ck::index_t lid = __lane_id();
    uint32_t scale  = p_scale[lid];

    // Each iteration take care of 16 x 32 matrix
    // itr-0, scale_op-0, use scale[th0:15]   [7:0]-th0:15, [15:8]-th16:32
    // itr-1, scale_op-1, use scale[th16:31]  [7:0]-th0:15, [15:8]-th16:32
    // itr-2, scale_op-2, use scale[th0:15]   [23:16]-th0:15, [31:24]-th16:32
    // itr-3, scale_op-3, use scale[th16:31]  [23:16]-th0:15, [31:24]-th16:32
    ck::static_for<0, 4, 1>{}([&](auto it) { // 4 scale factor test
        // 16x32 sub-matrix will be processed by a wave
        f4x8_t vf4_1{
            ck::f4x2_pk_t{}.pack(type_convert<f4_t>(Val), type_convert<f4_t>(Val))}; // 1.0f
        f4x8_t vf4_2{vf4_1};                                                         // 1.0f
        auto vT8_1 = ck::pk4scaled_type_convert<T8, f4x8_t, it>(scale, vf4_1);
        auto vT8_2 = ck::pk4scaled_type_convert<T8, f4x8_t, it>(scale, vf4_2);

        // write to p_mat
        ck::static_for<0, 8, 1>{}([&](auto ii) {
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii] =
                vT8_1[static_cast<int>(ii)];
            p_mat[(lid & 0x0F) * N + it * 32 + ((lid >> 4) & 1) * 16 + ii + 8] =
                vT8_2[static_cast<int>(ii)];
        });
    });
#endif
}

template <typename T>
void validate(T* out, int M, int N, float Val, uint32_t* scale)
{
    /* n:  [0:31]  [32:63]  [64:95]  [96:127]
            4.0f     1.0f     2.0f    0.5f  */
    for(int m = 0; m < M; m++)
    {
        /* n = [0:31] */
        for(int n = 0; n < 16; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [32:63] */
        for(int n = 32; n < 48; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 0) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 1) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [64:95] */
        for(int n = 64; n < 80; n++)
        {
            EXPECT_EQ(out[m * N + n], type_convert<T>(convert_exponent_to_float(scale[m], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
        /* n = [96:127] */
        for(int n = 96; n < 112; n++)
        {
            EXPECT_EQ(out[m * N + n],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 2) * Val))
                << "m: " << m << ", n: " << n << std::endl;
            EXPECT_EQ(out[m * N + n + 16],
                      type_convert<T>(convert_exponent_to_float(scale[m + M], 3) * Val))
                << "m: " << m << ", n: " << n + 16 << std::endl;
        }
    }
}

template <typename T>
class MXFP4TypedTest : public ::testing::Test
{
};

using TestTypes = ::testing::Types<float, half_t, bhalf_t>;
TYPED_TEST_SUITE(MXFP4TypedTest, TestTypes);

/* Typed test: "wave-wise FP4 to FP32/FP16/BFP16 conversion" */
TYPED_TEST(MXFP4TypedTest, DevicePackedScaledConvert)
{
    using T = TypeParam;

    // matrix shape M x N
    constexpr int M     = 16;
    constexpr int N     = 128; // Block 32 share a scale factor, 4 scale factors available
    constexpr float Val = 1.0f;
    std::vector<T> out(M * N, -1.0f);
    std::vector<uint32_t> scale(2 * M);
    for(int m = 0; m < M; m++)
    {
        scale[m]     = (128u << 24) | (128u << 16) | (129u << 8) | (129u); //[2.|2.|4.|4.]
        scale[m + M] = (126u << 24) | (126u << 16) | (127u << 8) | (127u); //[0.5|0.5|1.|1.]
    }

    DeviceMem device_out(M * N * sizeof(T));
    DeviceMem device_scale(2 * M * sizeof(uint32_t));
    device_scale.ToDevice(scale.data());

    test_packed_scaled_convert_fp32<M, N, Val>
        <<<1, 32>>>(static_cast<T*>(device_out.GetDeviceBuffer()),
                    static_cast<uint32_t*>(device_scale.GetDeviceBuffer()));

    device_out.FromDevice(out.data());

    validate(out.data(), M, N, Val, scale.data());
}
