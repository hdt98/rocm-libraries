/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

// Driver-local tensor view utilities.
//
// These are self-contained copies of tensor_view_t / tensor_layout_t and
// associated helpers (originally from src/kernels/tensor_view.hpp and
// src/include/miopen/tensor_view_utils.hpp). They are rewritten to operate
// on plain lengths/strides vectors obtained via the public C API rather than
// the internal miopen::TensorDescriptor class.

#ifndef GUARD_DRIVER_TENSOR_VIEW_HPP
#define GUARD_DRIVER_TENSOR_VIEW_HPP

#include "driver_tensor.hpp"

#include <cstdint>
#include <initializer_list>
#include <vector>

template <int N>
struct tensor_layout_t;

template <int N>
struct tensor_view_t
{
    constexpr uint64_t get_tensor_view_idx(const tensor_layout_t<N>& tensor_layout)
    {
        static_assert(N > 0);
        uint64_t idx = 0;
        for(auto i = 0; i < N; ++i)
        {
            idx += stride[i] * tensor_layout.layout[i];
        }
        return idx;
    }
    uint64_t stride[N];
    uint64_t size[N];
};

template <int N>
struct tensor_layout_t
{
    constexpr tensor_layout_t(const tensor_view_t<N>& tensor_view, uint64_t idx)
    {
        static_assert(N > 0);
        uint64_t temp = idx;
        if constexpr(N == 1)
        {
            layout[0] = idx;
        }
        else
        {
            for(auto i = N - 1; i > 1; --i)
            {
                layout[i] = temp % tensor_view.size[i];
                temp      = temp / tensor_view.size[i];
            }
            layout[1] = temp % tensor_view.size[1];
            layout[0] = temp / tensor_view.size[1];
        }
    }

    constexpr tensor_layout_t(std::initializer_list<uint64_t> layout_)
    {
        static_assert(N > 0);
        for(auto i = 0; i < N; ++i)
        {
            layout[i] = layout_.begin()[i];
        }
    }

    // Copy constructor from another tensor_layout_t of the same N.
    constexpr tensor_layout_t(const tensor_layout_t& other)
    {
        for(auto i = 0; i < N; ++i)
            layout[i] = other.layout[i];
    }

    uint64_t layout[N];
};

// Build a tensor_view_t<N> from a tensor descriptor using the public API.
// Equivalent to miopen::get_inner_expanded_tv<N>(miopen::deref(desc)).
template <int N>
inline tensor_view_t<N> get_inner_expanded_tv(miopenTensorDescriptor_t desc)
{
    auto dims    = driver_tensor::GetLengths(desc);
    auto strides = driver_tensor::GetStrides(desc);

    tensor_view_t<N> tensor_view{};
    for(int i = 0; i < N; ++i)
    {
        if(dims.empty())
        {
            tensor_view.stride[i] = 0;
            tensor_view.size[i]   = 0;
        }
        else if(i < static_cast<int>(dims.size()))
        {
            tensor_view.stride[i] = strides[i];
            tensor_view.size[i]   = dims[i];
        }
        else
        {
            tensor_view.stride[i] = strides.back();
            tensor_view.size[i]   = 1;
        }
    }
    return tensor_view;
}

template <int N>
inline void slice_tv(tensor_view_t<N>& tensor_view, int32_t sliceCount, const int32_t* slices)
{
    for(int32_t i = 0; i < sliceCount; i++)
    {
        int32_t dim   = slices[4 * i + 0];
        int32_t start = slices[4 * i + 1];
        int32_t end   = slices[4 * i + 2];
        int32_t step  = slices[4 * i + 3];

        if(end > static_cast<int32_t>(tensor_view.size[dim]))
            end = tensor_view.size[dim];

        auto len = end - start;

        tensor_view.size[dim] = (len + step - 1) / step;
        tensor_view.stride[dim] *= step;
    }
}

template <int N>
inline tensor_view_t<N - 1> get_tv_without_dim(const tensor_view_t<N>& origin_tv, int selected_dim)
{
    tensor_view_t<N - 1> res{};
    for(int i = 0; i < N; ++i)
    {
        if(i == selected_dim)
            continue;
        if(i < selected_dim)
        {
            res.size[i]   = origin_tv.size[i];
            res.stride[i] = origin_tv.stride[i];
        }
        else
        {
            res.size[i - 1]   = origin_tv.size[i];
            res.stride[i - 1] = origin_tv.stride[i];
        }
    }
    return res;
}

#endif // GUARD_DRIVER_TENSOR_VIEW_HPP
