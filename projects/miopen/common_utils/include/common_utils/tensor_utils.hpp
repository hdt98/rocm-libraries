// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_COMMON_UTILS_TENSOR_UTILS_HPP
#define GUARD_COMMON_UTILS_TENSOR_UTILS_HPP

#include <miopen/miopen.h>

#include <cassert>
#include <cstddef>
#include <ostream>
#include <tuple>
#include <vector>

// Convenience wrappers around the public MIOpen tensor descriptor API.
// These replace direct use of miopen::deref(tensorDesc).Method() with
// clean single-call functions that use only the public C API.

namespace tensor_utils {

inline int GetNumDims(miopenTensorDescriptor_t desc)
{
    int size = 0;
    miopenGetTensorDescriptorSize(desc, &size);
    return size;
}

inline miopenDataType_t GetType(miopenTensorDescriptor_t desc)
{
    miopenDataType_t dt;
    miopenGetTensorDescriptor(desc, &dt, nullptr, nullptr);
    return dt;
}

inline std::vector<size_t> GetLengths(miopenTensorDescriptor_t desc)
{
    int ndim = GetNumDims(desc);
    std::vector<size_t> lens(ndim);
    miopenGetTensorDescriptorV2(desc, nullptr, lens.data(), nullptr);
    return lens;
}

inline std::vector<size_t> GetStrides(miopenTensorDescriptor_t desc)
{
    int ndim = GetNumDims(desc);
    std::vector<size_t> strides(ndim);
    miopenGetTensorDescriptorV2(desc, nullptr, nullptr, strides.data());
    return strides;
}

inline size_t GetNumBytes(miopenTensorDescriptor_t desc)
{
    size_t numBytes = 0;
    miopenGetTensorNumBytes(desc, &numBytes);
    return numBytes;
}

inline size_t GetElementSize(miopenTensorDescriptor_t desc)
{
    auto lens = GetLengths(desc);
    size_t n  = 1;
    for(auto l : lens)
        n *= l;
    return n;
}

inline miopenTensorLayout_t GetLayout(miopenTensorDescriptor_t desc)
{
    miopenTensorLayout_t layout;
    miopenGetTensorLayout(desc, &layout);
    return layout;
}

inline size_t GetElementSpace(miopenTensorDescriptor_t desc)
{
    size_t space = 0;
    miopenGetTensorElementSpace(desc, &space);
    return space;
}

inline bool IsPacked(miopenTensorDescriptor_t desc)
{
    bool packed = false;
    miopenIsTensorPacked(desc, &packed);
    return packed;
}

inline size_t GetVectorLength(miopenTensorDescriptor_t desc)
{
    size_t vlen = 0;
    miopenGetTensorVectorLength(desc, &vlen);
    return vlen;
}

// Returns the byte size of a MIOpen data type.
inline size_t GetTypeSize(miopenDataType_t dt)
{
    switch(dt)
    {
    case miopenHalf: return 2;
    case miopenFloat: return 4;
    case miopenDouble: return 8;
    case miopenBFloat16: return 2;
    case miopenInt8: return 1;
    case miopenInt32: return 4;
    case miopenInt64: return 8;
    case miopenFloat8_fnuz: return 1;
    case miopenBFloat8_fnuz: return 1;
    default: return 0;
    }
}

// Unpack a vector into a tuple of N values.
// Replacement for miopen::tien<N>(vec).
template <std::size_t N, typename T>
auto Tien(const std::vector<T>& v)
{
    assert(v.size() >= N);
    if constexpr(N == 2)
        return std::make_tuple(v[0], v[1]);
    else if constexpr(N == 3)
        return std::make_tuple(v[0], v[1], v[2]);
    else if constexpr(N == 4)
        return std::make_tuple(v[0], v[1], v[2], v[3]);
    else if constexpr(N == 5)
        return std::make_tuple(v[0], v[1], v[2], v[3], v[4]);
}

// Given a spatial dimension count and a vector of lengths (or strides),
// return a tuple of (N, C, D, H, W) with D=1 for 2D tensors.
// Replacement for miopen::GetNCDHW.
template <class TElement>
auto GetNCDHW(unsigned spatial_dims, const std::vector<TElement>& data)
{
    if(spatial_dims == 3)
    {
        assert(data.size() >= 5);
        return std::make_tuple(data[0], data[1], data[2], data[3], data[4]);
    }
    else
    {
        assert(data.size() >= 4);
        return std::make_tuple(
            data[0], data[1], static_cast<TElement>(1), data[2], data[3]);
    }
}

// Print a range with a separator, replacement for miopen::LogRange.
template <typename Container>
std::ostream& LogRange(std::ostream& os, const Container& c, const char* sep)
{
    bool first = true;
    for(const auto& v : c)
    {
        if(!first)
            os << sep;
        os << v;
        first = false;
    }
    return os;
}

} // namespace tensor_utils

// tensor_view_t utilities -- provide handle-based wrappers for building
// tensor views without requiring miopen::TensorDescriptor internals.
#include "../../src/kernels/tensor_view.hpp"

namespace tensor_utils {

// Build a tensor_view_t<N> from an opaque tensor descriptor handle.
// Replacement for miopen::get_inner_expanded_tv<N>(miopen::deref(desc)).
template <int N>
inline tensor_view_t<N> GetInnerExpandedTv(miopenTensorDescriptor_t desc)
{
    auto dims    = GetLengths(desc);
    auto strides = GetStrides(desc);

    tensor_view_t<N> tv{};
    for(int i = 0; i < N; ++i)
    {
        if(dims.empty())
        {
            tv.stride[i] = 0;
            tv.size[i]   = 0;
        }
        else if(static_cast<size_t>(i) < dims.size())
        {
            tv.stride[i] = strides[i];
            tv.size[i]   = dims[i];
        }
        else
        {
            tv.stride[i] = strides.back();
            tv.size[i]   = 1;
        }
    }
    return tv;
}

} // namespace tensor_utils

#endif // GUARD_COMMON_UTILS_TENSOR_UTILS_HPP
