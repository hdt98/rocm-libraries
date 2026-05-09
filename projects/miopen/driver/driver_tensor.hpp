// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_DRIVER_TENSOR_HPP
#define GUARD_MIOPEN_DRIVER_TENSOR_HPP

#include <miopen/miopen.h>

#include <cassert>
#include <cstddef>
#include <ostream>
#include <tuple>
#include <vector>

// Driver-local wrappers around the public MIOpen tensor descriptor API.
// These replace direct use of miopen::deref(tensorDesc).Method() with
// clean single-call functions that use only the public C API.

namespace driver_tensor {

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

inline std::vector<int> GetLengths(miopenTensorDescriptor_t desc)
{
    int ndim = GetNumDims(desc);
    std::vector<int> lens(ndim);
    miopenGetTensorDescriptor(desc, nullptr, lens.data(), nullptr);
    return lens;
}

inline std::vector<int> GetStrides(miopenTensorDescriptor_t desc)
{
    int ndim = GetNumDims(desc);
    std::vector<int> strides(ndim);
    miopenGetTensorDescriptor(desc, nullptr, nullptr, strides.data());
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
        // 5D tensor: data has at least 5 elements
        assert(data.size() >= 5);
        return std::make_tuple(data[0], data[1], data[2], data[3], data[4]);
    }
    else
    {
        // 4D tensor: insert depth=1 between C and H
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

} // namespace driver_tensor

#endif // GUARD_MIOPEN_DRIVER_TENSOR_HPP
