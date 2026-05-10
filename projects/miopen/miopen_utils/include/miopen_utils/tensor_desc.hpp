/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP
#define GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP

// RAII wrapper around miopenTensorDescriptor_t that uses only the public
// MIOpen C API.  Replaces direct use of miopen::TensorDescriptor (an
// internal, MIOPEN_INTERNALS_EXPORT type) in driver, test, and
// miopen_utils code.

#include <miopen/miopen.h>

#include <cassert>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class TensorDesc
{
    miopenTensorDescriptor_t handle_ = nullptr;

    void create()
    {
        auto s = miopenCreateTensorDescriptor(&handle_);
        if(s != miopenStatusSuccess)
            throw std::runtime_error("miopenCreateTensorDescriptor failed");
    }

    void destroy() noexcept
    {
        if(handle_)
        {
            miopenDestroyTensorDescriptor(handle_);
            handle_ = nullptr;
        }
    }

    void copy_from(miopenTensorDescriptor_t src)
    {
        if(!src)
            return;

        int ndim = 0;
        miopenGetTensorDescriptorSize(src, &ndim);
        if(ndim == 0)
            return;

        miopenDataType_t dt;
        std::vector<size_t> lens(ndim), strs(ndim);
        miopenGetTensorDescriptorV2(src, &dt, lens.data(), strs.data());
        miopenSetTensorDescriptorV2(handle_, dt, ndim, lens.data(), strs.data());
    }

public:
    // ---------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------

    /// Default: creates an empty descriptor handle.
    TensorDesc() { create(); }

    /// From data type + lengths (packed strides computed automatically).
    TensorDesc(miopenDataType_t type, const std::vector<size_t>& lens)
    {
        create();
        miopenSetTensorDescriptorV2(
            handle_, type, static_cast<int>(lens.size()), lens.data(), nullptr);
    }

    /// From data type + lengths + explicit strides.
    TensorDesc(miopenDataType_t type,
               const std::vector<size_t>& lens,
               const std::vector<size_t>& strides)
    {
        create();
        assert(lens.size() == strides.size());
        miopenSetTensorDescriptorV2(
            handle_, type, static_cast<int>(lens.size()), lens.data(), strides.data());
    }

    /// From data type + layout + lengths (strides derived from layout).
    TensorDesc(miopenDataType_t type,
               miopenTensorLayout_t layout,
               const std::vector<size_t>& lens)
    {
        create();
        std::vector<int> ilens(lens.begin(), lens.end());
        miopenSetNdTensorDescriptorWithLayout(
            handle_,
            type,
            layout,
            ilens.data(),
            static_cast<int>(ilens.size()));
    }

    /// Convenience: 4D from individual dimensions (packed).
    TensorDesc(miopenDataType_t type, size_t n, size_t c, size_t h, size_t w)
        : TensorDesc(type, std::vector<size_t>{n, c, h, w})
    {
    }

    /// Convenience: 5D from individual dimensions (packed).
    TensorDesc(miopenDataType_t type, size_t n, size_t c, size_t d, size_t h, size_t w)
        : TensorDesc(type, std::vector<size_t>{n, c, d, h, w})
    {
    }

    /// Deep copy from an existing opaque handle.
    explicit TensorDesc(miopenTensorDescriptor_t src)
    {
        create();
        copy_from(src);
    }

    // ---------------------------------------------------------------
    // Copy / move
    // ---------------------------------------------------------------

    TensorDesc(const TensorDesc& other)
    {
        create();
        copy_from(other.handle_);
    }

    TensorDesc& operator=(const TensorDesc& other)
    {
        if(this != &other)
            copy_from(other.handle_);
        return *this;
    }

    TensorDesc(TensorDesc&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }

    TensorDesc& operator=(TensorDesc&& other) noexcept
    {
        if(this != &other)
        {
            destroy();
            handle_       = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~TensorDesc() { destroy(); }

    // ---------------------------------------------------------------
    // Getters (all use public API)
    // ---------------------------------------------------------------

    std::vector<size_t> GetLengths() const
    {
        int ndim = GetNumDims();
        std::vector<size_t> v(ndim);
        miopenGetTensorDescriptorV2(handle_, nullptr, v.data(), nullptr);
        return v;
    }

    std::vector<size_t> GetStrides() const
    {
        int ndim = GetNumDims();
        std::vector<size_t> v(ndim);
        miopenGetTensorDescriptorV2(handle_, nullptr, nullptr, v.data());
        return v;
    }

    miopenDataType_t GetType() const
    {
        miopenDataType_t dt;
        miopenGetTensorDescriptorV2(handle_, &dt, nullptr, nullptr);
        return dt;
    }

    int GetNumDims() const
    {
        int n = 0;
        miopenGetTensorDescriptorSize(handle_, &n);
        return n;
    }

    /// Total number of elements in the tensor (product of all dimension lengths).
    /// For a 4D tensor with dims [N,C,H,W] this returns N*C*H*W.
    size_t GetElementSize() const
    {
        auto lens = GetLengths();
        size_t n  = 1;
        for(auto l : lens)
            n *= l;
        return n;
    }

    /// Total number of element slots needed to store the tensor in memory,
    /// including any gaps introduced by non-packed strides.  For a packed
    /// tensor this equals GetElementSize(); for a strided tensor it may
    /// be larger.  Use this to allocate the backing data buffer.
    size_t GetElementSpace() const
    {
        size_t s = 0;
        miopenGetTensorElementSpace(handle_, &s);
        return s;
    }

    size_t GetNumBytes() const
    {
        size_t n = 0;
        miopenGetTensorNumBytes(handle_, &n);
        return n;
    }

    bool IsPacked() const
    {
        bool p = false;
        miopenIsTensorPacked(handle_, &p);
        return p;
    }

    size_t GetVectorLength() const
    {
        size_t v = 0;
        miopenGetTensorVectorLength(handle_, &v);
        return v;
    }

    /// Returns the tensor layout enum (miopenTensorNCHW, miopenTensorNHWC, etc.).
    miopenTensorLayout_t GetLayout() const
    {
        miopenTensorLayout_t layout;
        miopenGetTensorLayout(handle_, &layout);
        return layout;
    }

    /// Returns the tensor layout as a human-readable string ("NCHW", "NHWC", etc.).
    std::string GetLayout_str() const
    {
        switch(GetLayout())
        {
        case miopenTensorNCHW: return "NCHW";
        case miopenTensorNHWC: return "NHWC";
        case miopenTensorNCHWc4: return "NCHWc4";
        case miopenTensorNCHWc8: return "NCHWc8";
        case miopenTensorNCDHW: return "NCDHW";
        case miopenTensorNDHWC: return "NDHWC";
        default: return "Unknown";
        }
    }

    /// Compute linear index from multi-dimensional coordinates using strides.
    template <typename... Ts>
    size_t GetIndex(Ts... xs) const
    {
        auto strides    = GetStrides();
        size_t coords[] = {static_cast<size_t>(xs)...};
        size_t idx      = 0;
        size_t n        = sizeof...(xs);
        assert(n <= strides.size());
        for(size_t i = 0; i < n; ++i)
            idx += coords[i] * strides[i];
        return idx;
    }

    /// Debug string: "layout dim[...] stride[...]"
    std::string ToString() const
    {
        auto lens = GetLengths();
        auto strs = GetStrides();
        std::ostringstream ss;
        ss << GetLayout_str() << " dim[";
        for(size_t i = 0; i < lens.size(); ++i)
        {
            if(i > 0)
                ss << ',';
            ss << lens[i];
        }
        ss << "] stride[";
        for(size_t i = 0; i < strs.size(); ++i)
        {
            if(i > 0)
                ss << ',';
            ss << strs[i];
        }
        ss << ']';
        return ss.str();
    }

    // ---------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------

    bool operator==(const TensorDesc& other) const
    {
        return GetType() == other.GetType() && GetLengths() == other.GetLengths() &&
               GetStrides() == other.GetStrides();
    }

    bool operator!=(const TensorDesc& other) const { return !(*this == other); }

    // ---------------------------------------------------------------
    // C API interop
    // ---------------------------------------------------------------

    /// Get the raw handle for passing to MIOpen C API functions.
    miopenTensorDescriptor_t get() const { return handle_; }

    /// Implicit conversion so code like miopenSetTensor(h, desc, ...) works.
    operator miopenTensorDescriptor_t() const { return handle_; }
};

#endif // GUARD_MIOPEN_UTILS_TENSOR_DESC_HPP
