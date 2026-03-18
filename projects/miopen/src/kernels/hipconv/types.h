#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using fp16_t = _Float16;
using bf16_t = __bf16;
using fp32_t = float;

enum class KernelDataType
{
    fp16,
    bf16,
};

template <KernelDataType type>
struct ToTypeImpl;
template <>
struct ToTypeImpl<KernelDataType::fp16>
{
    using type = fp16_t;
};
template <>
struct ToTypeImpl<KernelDataType::bf16>
{
    using type = bf16_t;
};

template <KernelDataType type>
using ToType = typename ToTypeImpl<type>::type;
