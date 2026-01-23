// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <vector>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/pk_fp6.hpp"
#include "ck_tile/host.hpp"

using ck_tile::bf16_t;
using ck_tile::bf16x16_t;
using ck_tile::fp16_t;
using ck_tile::fp16x16_t;
using ck_tile::fp32_t;
using ck_tile::fp32x16_t;
using ck_tile::number;
using ck_tile::pk_bf6_t;
using ck_tile::pk_fp6_t;

template <
    typename SRC,
    typename PK6,
    typename DST,
    bool is_device,
    std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool> = true>
CK_TILE_HOST void test_convert();

// ============================================================================
// FP6 (E2M3) Tests
// ============================================================================

TEST(PackedFp6, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_fp6_t>::has_inf(), false);

    // FP6 E2M3: bias=1, range ~[0.125, 7.0]
    // Test using the binary constants directly
    pk_fp6_t zero_pk        = ck_tile::numeric<pk_fp6_t>::zero();
    pk_fp6_t min_pk         = ck_tile::numeric<pk_fp6_t>::min();
    pk_fp6_t max_pk         = ck_tile::numeric<pk_fp6_t>::max();
    pk_fp6_t lowest_pk      = ck_tile::numeric<pk_fp6_t>::lowest();
    pk_fp6_t epsilon_pk     = ck_tile::numeric<pk_fp6_t>::epsilon();
    pk_fp6_t round_error_pk = ck_tile::numeric<pk_fp6_t>::round_error();
    pk_fp6_t denorm_min_pk  = ck_tile::numeric<pk_fp6_t>::denorm_min();
    EXPECT_FLOAT_EQ(zero_pk.to_float(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(min_pk.to_float(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(max_pk.to_float(1.0f), 7.5f);
    EXPECT_FLOAT_EQ(lowest_pk.to_float(1.0f), -7.5f);
    EXPECT_FLOAT_EQ(epsilon_pk.to_float(1.0f), 0.125f);
    EXPECT_FLOAT_EQ(round_error_pk.to_float(1.0f), 0.125f);
    EXPECT_FLOAT_EQ(denorm_min_pk.to_float(1.0f), 0.125f);
}

TEST(PackedFp6, Fill)
{
    std::vector<pk_fp6_t> v_fp6(2);
    ck_tile::FillUniformDistribution<pk_fp6_t>{1.f, 1.f}(v_fp6);
    pk_fp6_t expected;
    expected.set_element(0, 0b001000);
    EXPECT_EQ(v_fp6[0].get_element(0), expected.get_element(0));
    EXPECT_EQ(v_fp6[0].get_element(6), expected.get_element(0));
    EXPECT_EQ(v_fp6[1].get_element(15), expected.get_element(0));
}

TEST(PackedFp6, ConvertBasic)
{
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(0.0f), 0b000000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(-0.0f), 0b100000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(1.0f), 0b001000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_fp6_t>(-1.0f), 0b101000);

    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(1.0f).get_element(0), 0b001000);
    EXPECT_EQ(ck_tile::type_convert<pk_fp6_t>(-1.0f).get_element(0), 0b101000);

    EXPECT_EQ(pk_fp6_t(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(pk_fp6_t(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(pk_fp6_t(1.0f).get_element(0), 0b001000);
    EXPECT_EQ(pk_fp6_t(-1.0f).get_element(0), 0b101000);
}

TEST(PackedFp6, ConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

TEST(PackedFp6, ConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_fp6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_fp6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_fp6_t, fp32_t, is_device>();
}

// ============================================================================
// BF6 (E3M2) Tests
// ============================================================================

TEST(PackedBf6, NumericLimits)
{
    EXPECT_EQ(ck_tile::numeric<pk_bf6_t>::has_inf(), false);

    // BF6 E3M2: bias=3, range ~[0.25, 28.0]
    // Test using the binary constants directly
    pk_bf6_t zero_pk        = ck_tile::numeric<pk_bf6_t>::zero();
    pk_bf6_t min_pk         = ck_tile::numeric<pk_bf6_t>::min();
    pk_bf6_t max_pk         = ck_tile::numeric<pk_bf6_t>::max();
    pk_bf6_t lowest_pk      = ck_tile::numeric<pk_bf6_t>::lowest();
    pk_bf6_t epsilon_pk     = ck_tile::numeric<pk_bf6_t>::epsilon();
    pk_bf6_t round_error_pk = ck_tile::numeric<pk_bf6_t>::round_error();
    pk_bf6_t denorm_min_pk  = ck_tile::numeric<pk_bf6_t>::denorm_min();

    EXPECT_FLOAT_EQ(zero_pk.to_float(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(min_pk.to_float(1.0f), 0.25f);
    EXPECT_FLOAT_EQ(max_pk.to_float(1.0f), 28.0f);
    EXPECT_FLOAT_EQ(lowest_pk.to_float(1.0f), -28.0f);
    EXPECT_FLOAT_EQ(epsilon_pk.to_float(1.0f), 0.0625f);
    EXPECT_FLOAT_EQ(round_error_pk.to_float(1.0f), 0.0625f);
    EXPECT_FLOAT_EQ(denorm_min_pk.to_float(1.0f), 0.0625f);
}

TEST(PackedBf6, fill)
{
    std::vector<pk_bf6_t> v_bf6(2);
    ck_tile::FillUniformDistribution<pk_bf6_t>{1.f, 1.f}(v_bf6);
    pk_bf6_t expected;
    // 1.0f in BF6 E3M2: sign=0, exp=011, mant=00 = 0b001100
    expected.set_element(0, 0b001100);
    EXPECT_EQ(v_bf6[0].get_element(0), expected.get_element(0));
    EXPECT_EQ(v_bf6[0].get_element(6), expected.get_element(0));
    EXPECT_EQ(v_bf6[1].get_element(15), expected.get_element(0));
}

TEST(PackedBf6, ConvertBasic)
{
    // Test basic float to bf6 conversion
    // BF6 E3M2 format: sign(1) + exp(3) + mant(2)
    // 0.0f:  0 000 00 = 0b000000
    // -0.0f: 1 000 00 = 0b100000
    // 1.0f:  0 011 00 = 0b001100
    // -1.0f: 1 011 00 = 0b101100

    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(0.0f), 0b000000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(-0.0f), 0b100000);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(1.0f), 0b001100);
    EXPECT_EQ(ck_tile::convert_to_type<pk_bf6_t>(-1.0f), 0b101100);

    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(1.0f).get_element(0), 0b001100);
    EXPECT_EQ(ck_tile::type_convert<pk_bf6_t>(-1.0f).get_element(0), 0b101100);

    EXPECT_EQ(pk_bf6_t(0.0f).get_element(0), 0b000000);
    EXPECT_EQ(pk_bf6_t(-0.0f).get_element(0), 0b100000);
    EXPECT_EQ(pk_bf6_t(1.0f).get_element(0), 0b001100);
    EXPECT_EQ(pk_bf6_t(-1.0f).get_element(0), 0b101100);
}

TEST(PackedBf6, ConvertHost)
{
    constexpr bool is_device = false;
    test_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

TEST(PackedBf6, ConvertDevice)
{
    constexpr bool is_device = true;
    test_convert<fp32_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, fp16_t, is_device>();
    test_convert<fp32_t, pk_bf6_t, bf16_t, is_device>();
    test_convert<fp16_t, pk_bf6_t, fp32_t, is_device>();
    test_convert<bf16_t, pk_bf6_t, fp32_t, is_device>();
}

// ============================================================================
// Cross-word boundary tests (for 6-bit packing)
// ============================================================================

TEST(PackedFp6, CrossWordBoundary)
{
    // Test elements that span across uint32_t boundaries
    // Element at bit position 5 (5*6 = 30 bits) spans words 0 and 1
    pk_fp6_t val;

    // Elements that might span boundaries
    val.set_element(5, 0b001010);  // bit offset 30, spans word 0-1
    val.set_element(10, 0b001100); // bit offset 60, spans word 1-2
    val.set_element(15, 0b010000); // bit offset 90, spans word 2-3

    EXPECT_EQ(val.unpack(number<5>{}), 0b001010);
    EXPECT_EQ(val.unpack(number<10>{}), 0b001100);
    EXPECT_EQ(val.unpack(number<15>{}), 0b010000);
}

TEST(PackedBf6, CrossWordBoundary)
{
    // Test elements that span across uint32_t boundaries
    pk_bf6_t val;

    val.set_element(5, 0b001010);
    val.set_element(10, 0b001100);
    val.set_element(15, 0b010000);

    EXPECT_EQ(val.unpack(number<5>{}), 0b001010);
    EXPECT_EQ(val.unpack(number<10>{}), 0b001100);
    EXPECT_EQ(val.unpack(number<15>{}), 0b010000);
}

// ============================================================================
// Implementation
// ============================================================================

#define toF32(x) ck_tile::type_convert<float>(x)
#define toPF6(x) ck_tile::type_convert<pk_fp6_t>(x)
#define toBF6(x) ck_tile::type_convert<pk_bf6_t>(x)
#define toSRC(x) ck_tile::type_convert<SRC>(x)
#define toDST(x) ck_tile::type_convert<DST>(x)
#define toDSTx16(x) ck_tile::type_convert<DSTx16_t>(x)

template <typename Kernel, typename... Args>
__global__ void MyKernel(Args... args)
{
    Kernel{}(args...);
}

template <typename SRC, typename PK6, typename DST, int N>
struct SrcPk6Dst
{
    CK_TILE_HOST_DEVICE void operator()(const SRC* src, DST* dst) const
    {
#if CK_TILE_AVX512F_WA
        // Use arrays of two 8-element vectors only for float to avoid AVX-512 on non-supporting
        // CPUs For smaller types (fp16, bf16), 16-element vectors are fine with AVX2
        constexpr bool UseSrcx8 = std::is_same_v<SRC, float>;
        constexpr bool UseDstx8 = std::is_same_v<DST, float>;

        using SRCx8_t  = ck_tile::ext_vector_t<SRC, 8>;
        using DSTx8_t  = ck_tile::ext_vector_t<DST, 8>;
        using SRCx16_t = std::conditional_t<UseSrcx8, SRCx8_t[2], ck_tile::ext_vector_t<SRC, 16>>;
        using DSTx16_t = std::conditional_t<UseDstx8, DSTx8_t[2], ck_tile::ext_vector_t<DST, 16>>;
#else
        // Use regular 16-element vectors when AVX-512 is available
        using SRCx16_t = ck_tile::ext_vector_t<SRC, 16>;
        using DSTx16_t = ck_tile::ext_vector_t<DST, 16>;
#endif

        ck_tile::static_for<0, N, 16>{}([&](auto i) {
#if CK_TILE_AVX512F_WA
            // Load input
            SRCx16_t input16{};
            if constexpr(UseSrcx8)
            {
                // Load as two 8-element vectors
                for(int j = 0; j < 8; ++j)
                {
                    input16[0][j] = src[i + j];
                    input16[1][j] = src[i + j + 8];
                }
            }
            else
            {
                // Load as single 16-element vector
                for(int j = 0; j < 16; ++j)
                    input16[j] = src[i + j];
            }

            PK6 pk6_packed;
            if constexpr(std::is_same_v<PK6, pk_fp6_t>)
                pk6_packed = toPF6(input16);
            else
                pk6_packed = toBF6(input16);

            // Convert to output
            DSTx16_t output16{};
            if constexpr(UseDstx8)
            {
                ck_tile::type_convert<DSTx8_t[2]>(pk6_packed, output16);
            }
            else
            {
                output16 = toDSTx16(pk6_packed);
            }

            // Store output
            if constexpr(UseDstx8)
            {
                for(int j = 0; j < 8; ++j)
                {
                    dst[i + j]     = output16[0][j];
                    dst[i + j + 8] = output16[1][j];
                }
            }
            else
            {
                for(int j = 0; j < 16; ++j)
                    dst[i + j] = output16[j];
            }
#else
            // Standard 16-element vector path when AVX-512 is available
            SRCx16_t input16{};
            for(int j = 0; j < 16; ++j)
                input16[j] = src[i + j];

            PK6 pk6_packed{};
            if constexpr(std::is_same_v<PK6, pk_fp6_t>)
                pk6_packed = toPF6(input16);
            else
                pk6_packed = toBF6(input16);
            DSTx16_t output16 = toDSTx16(pk6_packed);

            for(int j = 0; j < 16; ++j)
                dst[i + j] = output16[j];
#endif
        });
    }
};

template <typename SRC,
          typename PK6,
          typename DST,
          bool is_device,
          std::enable_if_t<std::is_same_v<PK6, pk_fp6_t> || std::is_same_v<PK6, pk_bf6_t>, bool>>
CK_TILE_HOST void test_convert()
{
    constexpr int N = 32;

    // FP6 E2M3 test values: bias=1, range [0.125, 7.5]
    constexpr std::array<float, N> fp6_test_data = {
        0.f,   0.125f, 0.25f, 0.375f, 0.5f,  0.625f, 0.75f, 0.875f, 1.f,  1.25f, 1.5f,
        1.75f, 2.f,    2.25f, 2.5f,   2.75f, 3.f,    3.5f,  4.f,    4.5f, 5.f,   5.5f,
        6.f,   6.5f,   7.f,   7.5f,   -1.f,  -2.f,   -3.f,  -5.f,   -7.f, -7.5f};
    // Expected values after FP6 quantization
    constexpr std::array<float, N> fp6_ref_data = {
        0.f,   0.125f, 0.25f, 0.375f, 0.5f,  0.625f, 0.75f, 0.875f, 1.f,  1.25f, 1.5f,
        1.75f, 2.f,    2.25f, 2.5f,   2.75f, 3.f,    3.5f,  4.f,    4.5f, 5.f,   5.5f,
        6.f,   6.5f,   7.f,   7.5f,   -1.f,  -2.f,   -3.f,  -5.f,   -7.f, -7.5f};

    // BF6 E3M2 test values: bias=3, range [0.0625, 28]
    constexpr std::array<float, N> bf6_test_data = {
        0.f,   0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.f,
        1.25f, 1.5f,    1.75f,  2.f,     2.5f,  3.f,    3.5f, 4.f,    5.f,   6.f,    7.f,
        8.f,   10.f,    12.f,   14.f,    16.f,  24.f,   -1.f, -2.f,   -4.f,  -28.f};
    // Expected values after BF6 quantization
    constexpr std::array<float, N> bf6_ref_data = {
        0.f,   0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.f,
        1.25f, 1.5f,    1.75f,  2.f,     2.5f,  3.f,    3.5f, 4.f,    5.f,   6.f,    7.f,
        8.f,   10.f,    12.f,   14.f,    16.f,  24.f,   -1.f, -2.f,   -4.f,  -28.f};

    // Select test data based on PK6 type
    const auto& test_data = (std::is_same_v<PK6, pk_fp6_t> ? fp6_test_data : bf6_test_data);
    const auto& ref_data  = (std::is_same_v<PK6, pk_fp6_t> ? fp6_ref_data : bf6_ref_data);

    std::array<SRC, N> in;
    std::array<DST, N> ref, out;

    // prepare input and ground truth in host
    for(int i = 0; i < N; ++i)
    {
        in[i]  = toSRC(test_data[i]);
        ref[i] = toDST(ref_data[i]);
        EXPECT_EQ(test_data[i], toF32(in[i]));
        EXPECT_EQ(ref_data[i], toF32(ref[i]));
    }

    using job = SrcPk6Dst<SRC, PK6, DST, N>;

    if constexpr(is_device)
    {
        auto in_d  = std::make_unique<ck_tile::DeviceMem>(in.size() * sizeof(SRC));
        auto out_d = std::make_unique<ck_tile::DeviceMem>(out.size() * sizeof(DST));
        in_d->ToDevice(in.data());

        MyKernel<job><<<1, 1>>>(reinterpret_cast<const SRC*>(in_d->GetDeviceBuffer()),
                                reinterpret_cast<DST*>(out_d->GetDeviceBuffer()));

        out_d->FromDevice(out.data());
    }
    else
    {
        job{}(in.data(), out.data());
    }

    for(int i = 0; i < N; ++i)
        EXPECT_EQ(ref[i], out[i]) << "i:" << i;
}
