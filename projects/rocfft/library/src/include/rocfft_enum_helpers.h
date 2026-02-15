// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_ENUM_HELPERS
#define ROCFFT_ENUM_HELPERS
#include "rocfft/rocfft.h"
#include <string_view>

// transform-type-related helper
constexpr bool dft_is_real(const rocfft_transform_type& type)
{
    return type == rocfft_transform_type_real_forward || type == rocfft_transform_type_real_inverse;
}
constexpr bool dft_is_complex(const rocfft_transform_type& type)
{
    return type == rocfft_transform_type_complex_forward
           || type == rocfft_transform_type_complex_inverse;
}
constexpr bool dft_is_forward(const rocfft_transform_type& type)
{
    return type == rocfft_transform_type_complex_forward
           || type == rocfft_transform_type_real_forward;
}
constexpr bool dft_is_inverse(const rocfft_transform_type& type)
{
    return type == rocfft_transform_type_complex_inverse
           || type == rocfft_transform_type_real_inverse;
}

// array-type-related helpers
constexpr bool array_type_is_real(const rocfft_array_type& array_type)
{
    return array_type == rocfft_array_type_real;
}
constexpr bool array_type_is_hermitian(const rocfft_array_type& array_type)
{
    return array_type == rocfft_array_type_hermitian_interleaved
           || array_type == rocfft_array_type_hermitian_planar;
}
constexpr bool array_type_is_complex_but_not_hermitian(const rocfft_array_type& array_type)
{
    return array_type == rocfft_array_type_complex_interleaved
           || array_type == rocfft_array_type_complex_planar;
}
constexpr bool array_type_is_complex(const rocfft_array_type& array_type)
{
    return array_type_is_complex_but_not_hermitian(array_type)
           || array_type_is_hermitian(array_type);
}
constexpr bool array_type_is_interleaved(const rocfft_array_type& array_type)
{
    return array_type == rocfft_array_type_complex_interleaved
           || array_type == rocfft_array_type_hermitian_interleaved;
}
constexpr bool array_type_is_planar(const rocfft_array_type& array_type)
{
    return array_type == rocfft_array_type_complex_planar
           || array_type == rocfft_array_type_hermitian_planar;
}

// precision-related helpers

constexpr size_t real_type_size(const rocfft_precision& precision)
{
    switch(precision)
    {
    case rocfft_precision_half:
        return 2;
    case rocfft_precision_single:
        return 4;
    case rocfft_precision_double:
        return 8;
    }
}

constexpr size_t complex_type_size(const rocfft_precision& precision)
{
    return real_type_size(precision) * 2;
}

constexpr std::string_view precision_name(const rocfft_precision& precision)
{
    switch(precision)
    {
    case rocfft_precision_half:
        return "half";
    case rocfft_precision_single:
        return "single";
    case rocfft_precision_double:
        return "double";
    }
}

constexpr size_t element_size(const rocfft_precision&  precision,
                              const rocfft_array_type& array_type)
{
    return array_type_is_complex(array_type) ? complex_type_size(precision)
                                             : real_type_size(precision);
}

namespace
{
    // offset a pointer by a number of elements, given the elements'
    // precision and type (complex or not)
    void* ptr_offset(void* p, size_t elems, rocfft_precision precision, rocfft_array_type type)
    {
        return static_cast<char*>(p) + elems * element_size(precision, type);
    }
}

#endif // ROCFFT_ENUM_HELPERS
