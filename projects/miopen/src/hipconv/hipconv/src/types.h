#pragma once

#include "hipconv/conv2d_params.hpp"
#include <hip/hip_fp8.h>

using fp16_t   = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

using bf16_t = __bf16;
using fp32_t = float;
using fp8_t  = __hip_fp8_e4m3;
using bf8_t  = __hip_fp8_e5m2;

template <hipconv::DataType type>
struct ToTypeImpl;
template <>
struct ToTypeImpl<hipconv::DataType::fp16>
{
    using type = fp16_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::bf16>
{
    using type = bf16_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::fp32>
{
    using type = fp32_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::fp8>
{
    using type = fp8_t;
};
template <>
struct ToTypeImpl<hipconv::DataType::bf8>
{
    using type = bf8_t;
};

template <hipconv::DataType type>
using ToType = typename ToTypeImpl<type>::type;

inline auto mantissa_bits(hipconv::DataType type)
{
    switch(type)
    {
    case hipconv::DataType::fp16:
        return 10;
    case hipconv::DataType::bf16:
        return 7;
    case hipconv::DataType::fp32:
        return 23;
    case hipconv::DataType::fp8:
        return 3;
    case hipconv::DataType::bf8:
        return 2;
    }
}
