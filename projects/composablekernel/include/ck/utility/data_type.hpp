// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/statically_indexed_array.hpp"
#include <vector>
namespace ck {

using bhalf_t = ushort;
using half_t  = _Float16;
using int4_t  = _BitInt(4);
using uint4_t = unsigned _BitInt(4);
using f8_t    = _BitInt(8);
using bf8_t   = unsigned _BitInt(8);

enum class MTX_FMT
{
    MTX_FMT_FP8_E4M3, // FP8
    MTX_FMT_FP8_E5M2, // BF8
    MTX_FMT_FP6_E2M3, // FP6
    MTX_FMT_FP6_E3M2, // BF6
    MTX_FMT_FP4_E2M1, // FP4
    MTX_FMT_DEFAULT,
};

inline constexpr auto next_pow2(uint32_t x)
{
    // Precondition: x > 1.
    return x > 1u ? (1u << (32u - __builtin_clz(x - 1u))) : x;
}

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_t, bf8_t, bool
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value || is_same<T, int8_t>::value ||
           is_same<T, uint8_t>::value || is_same<T, f8_t>::value || is_same<T, bf8_t>::value ||
           is_same<T, bool>::value || is_same<T, __bf16>::value || is_same<T, __fp16>::value;
}

// vector_type
template <typename T, index_t N, typename Enable = void>
struct vector_type;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<T __attribute__((ext_vector_type(V))), N>;

// Caution: DO NOT REMOVE
// intentionally have only declaration but no definition to cause compilation failure when trying to
// instantiate this template. The purpose is to catch user's mistake when trying to make "vector of
// vectors"
template <typename T, index_t V, index_t N>
struct vector_type<vector_type<T, V>, N>;

// vector_type_maker
// This is the right way to handle "vector of vectors": making a bigger vector instead
template <typename T, index_t N>
struct vector_type_maker
{
    using type = vector_type<T, N>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<T __attribute__((ext_vector_type(N1))), N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N0, index_t N1>
struct vector_type_maker<vector_type<T, N1>, N0>
{
    using type = vector_type<T, N0 * N1>;
};

template <typename T, index_t N>
using vector_type_maker_t = typename vector_type_maker<T, N>::type;

template <typename T, index_t N>
__host__ __device__ constexpr auto make_vector_type(Number<N>)
{
    return typename vector_type_maker<T, N>::type{};
}

// scalar_type
template <typename TV>
struct scalar_type;

// is_scalar_type
template <typename TV>
struct is_scalar_type
{
    static constexpr bool value = (scalar_type<remove_cvref_t<TV>>::vector_size == 1);
};

// has_same_scalar_type
template <typename X, typename Y>
using has_same_scalar_type = is_same<typename scalar_type<remove_cvref_t<X>>::type,
                                     typename scalar_type<remove_cvref_t<Y>>::type>;

template <typename T, index_t N>
struct scalar_type<T __attribute__((ext_vector_type(N)))>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct scalar_type<vector_type<T, N>>
{
    using type                           = T;
    static constexpr index_t vector_size = N;
};

//
template <>
struct scalar_type<double>
{
    using type                           = double;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<float>
{
    using type                           = float;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<half_t>
{
    using type                           = half_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bhalf_t>
{
    using type                           = bhalf_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int32_t>
{
    using type                           = int32_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<int8_t>
{
    using type                           = int8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<uint8_t>
{
    using type                           = uint8_t;
    static constexpr index_t vector_size = 1;
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct scalar_type<int4_t>
{
    using type                           = int4_t;
    static constexpr index_t vector_size = 1;
};
#endif

template <>
struct scalar_type<f8_t>
{
    using type                           = f8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_t>
{
    using type                           = bf8_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

template <typename T>
struct vector_type<T, 1, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union
    {
        T d1_;
        StaticallyIndexedArray<T, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

__device__ int static err = 0;
template <typename T>
struct vector_type<T, 2, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));

    using type = d2_t;

    union
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));

    using type = d4_t;

    union
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));

    using type = d8_t;

    union
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));

    using type = d16_t;

    union
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));

    using type = d32_t;

    union
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));

    using type = d64_t;

    union
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 128, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));

    using type = d128_t;

    union
    {
        d128_t d128_;
        StaticallyIndexedArray<d1_t, 128> d1x128_;
        StaticallyIndexedArray<d2_t, 64> d2x64_;
        StaticallyIndexedArray<d4_t, 32> d4x32_;
        StaticallyIndexedArray<d8_t, 16> d8x16_;
        StaticallyIndexedArray<d16_t, 8> d16x8_;
        StaticallyIndexedArray<d32_t, 4> d32x4_;
        StaticallyIndexedArray<d64_t, 2> d64x2_;
        StaticallyIndexedArray<d128_t, 1> d128x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value || is_same<X, d128_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x128_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x64_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x32_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x16_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x8_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x4_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x2_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 256, typename std::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d16_t __attribute__((ext_vector_type(16)));
    typedef T d32_t __attribute__((ext_vector_type(32)));
    typedef T d64_t __attribute__((ext_vector_type(64)));
    typedef T d128_t __attribute__((ext_vector_type(128)));
    typedef T d256_t __attribute__((ext_vector_type(256)));

    using type = d256_t;

    union
    {
        d256_t d256_;
        StaticallyIndexedArray<d1_t, 256> d1x256_;
        StaticallyIndexedArray<d2_t, 128> d2x128_;
        StaticallyIndexedArray<d4_t, 64> d4x64_;
        StaticallyIndexedArray<d8_t, 32> d8x32_;
        StaticallyIndexedArray<d16_t, 16> d16x16_;
        StaticallyIndexedArray<d32_t, 8> d32x8_;
        StaticallyIndexedArray<d64_t, 4> d64x4_;
        StaticallyIndexedArray<d128_t, 2> d128x2_;
        StaticallyIndexedArray<d256_t, 1> d256x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                is_same<X, d8_t>::value || is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                is_same<X, d64_t>::value || is_same<X, d128_t>::value || is_same<X, d256_t>::value,
            "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x256_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x128_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x64_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x32_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x16_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x8_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x4_;
        }
        else if constexpr(is_same<X, d128_t>::value)
        {
            return data_.d128x2_;
        }
        else if constexpr(is_same<X, d256_t>::value)
        {
            return data_.d256x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T, index_t N>
struct non_native_vector_base
{
    using type = non_native_vector_base<T, N>;

    __host__ __device__ non_native_vector_base()            = default;
    __host__ __device__ non_native_vector_base(const type&) = default;
    __host__ __device__ non_native_vector_base(type&&)      = default;
    __host__ __device__ ~non_native_vector_base()           = default;

    T d[N];
};

// non-native vector_type implementation
template <typename T>
struct vector_type<T, 1, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using type = d1_t;

    union alignas(next_pow2(1 * sizeof(T)))
    {
        d1_t d1_;
        StaticallyIndexedArray<d1_t, 1> d1x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value,
                      "Something went wrong, please check src and dst types.");

        return data_.d1x1_;
    }
};

template <typename T>
struct vector_type<T, 2, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;

    using type = d2_t;

    union alignas(next_pow2(2 * sizeof(T)))
    {
        d2_t d2_;
        StaticallyIndexedArray<d1_t, 2> d1x2_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x2_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 4, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d4_t = non_native_vector_base<T, 4>;

    using type = d4_t;

    union alignas(next_pow2(4 * sizeof(T)))
    {
        d4_t d4_;
        StaticallyIndexedArray<d1_t, 4> d1x4_;
        StaticallyIndexedArray<d2_t, 2> d2x2_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x4_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x2_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d4_t = non_native_vector_base<T, 4>;
    using d8_t = non_native_vector_base<T, 8>;

    using type = d8_t;

    union alignas(next_pow2(8 * sizeof(T)))
    {
        d8_t d8_;
        StaticallyIndexedArray<d1_t, 8> d1x8_;
        StaticallyIndexedArray<d2_t, 4> d2x4_;
        StaticallyIndexedArray<d4_t, 2> d4x2_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x8_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x2_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;

    using type = d16_t;

    union alignas(next_pow2(16 * sizeof(T)))
    {
        d16_t d16_;
        StaticallyIndexedArray<d1_t, 16> d1x16_;
        StaticallyIndexedArray<d2_t, 8> d2x8_;
        StaticallyIndexedArray<d4_t, 4> d4x4_;
        StaticallyIndexedArray<d8_t, 2> d8x2_;
        StaticallyIndexedArray<d16_t, 1> d16x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x16_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x2_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 32, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;

    using type = d32_t;

    union alignas(next_pow2(32 * sizeof(T)))
    {
        d32_t d32_;
        StaticallyIndexedArray<d1_t, 32> d1x32_;
        StaticallyIndexedArray<d2_t, 16> d2x16_;
        StaticallyIndexedArray<d4_t, 8> d4x8_;
        StaticallyIndexedArray<d8_t, 4> d8x4_;
        StaticallyIndexedArray<d16_t, 2> d16x2_;
        StaticallyIndexedArray<d32_t, 1> d32x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x32_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x16_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x8_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x4_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x2_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 64, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d16_t = non_native_vector_base<T, 16>;
    using d32_t = non_native_vector_base<T, 32>;
    using d64_t = non_native_vector_base<T, 64>;

    using type = d64_t;

    union alignas(next_pow2(64 * sizeof(T)))
    {
        d64_t d64_;
        StaticallyIndexedArray<d1_t, 64> d1x64_;
        StaticallyIndexedArray<d2_t, 32> d2x32_;
        StaticallyIndexedArray<d4_t, 16> d4x16_;
        StaticallyIndexedArray<d8_t, 8> d8x8_;
        StaticallyIndexedArray<d16_t, 4> d16x4_;
        StaticallyIndexedArray<d32_t, 2> d32x2_;
        StaticallyIndexedArray<d64_t, 1> d64x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d16_t>::value || is_same<X, d32_t>::value ||
                          is_same<X, d64_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x64_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x32_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x16_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x8_;
        }
        else if constexpr(is_same<X, d16_t>::value)
        {
            return data_.d16x4_;
        }
        else if constexpr(is_same<X, d32_t>::value)
        {
            return data_.d32x2_;
        }
        else if constexpr(is_same<X, d64_t>::value)
        {
            return data_.d64x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 3>
{
    using d1_t = T;
    typedef T d3_t __attribute__((ext_vector_type(3)));

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 6>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d6_t __attribute__((ext_vector_type(6)));

    using type = d6_t;

    union
    {
        d6_t d6_;
        StaticallyIndexedArray<d1_t, 6> d1x6_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d3_t, 2> d3x2_;
        StaticallyIndexedArray<d6_t, 1> d6x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x6_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x2_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 12>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    using type = d12_t;

    union
    {
        d12_t d12_;
        StaticallyIndexedArray<d1_t, 12> d1x12_;
        StaticallyIndexedArray<d2_t, 6> d2x6_;
        StaticallyIndexedArray<d3_t, 4> d3x4_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d6_t, 2> d6x2_;
        StaticallyIndexedArray<d12_t, 1> d12x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d12_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x12_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x6_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x4_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x2_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 18>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d18_t __attribute__((ext_vector_type(18)));

    using type = d18_t;

    union
    {
        d18_t d18_;
        StaticallyIndexedArray<d1_t, 18> d1x18_;
        StaticallyIndexedArray<d2_t, 9> d2x9_;
        StaticallyIndexedArray<d3_t, 6> d3x6_;
        StaticallyIndexedArray<d6_t, 3> d6x3_;
        StaticallyIndexedArray<d9_t, 2> d9x2_;
        StaticallyIndexedArray<d18_t, 1> d18x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d6_t>::value ||
                          is_same<X, d9_t>::value || is_same<X, d18_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x18_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x2_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 24>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d24_t __attribute__((ext_vector_type(242)));
    using type = d24_t;

    union
    {
        d24_t d24_;
        StaticallyIndexedArray<d1_t, 24> d1x24_;
        StaticallyIndexedArray<d2_t, 12> d2x12_;
        StaticallyIndexedArray<d3_t, 8> d3x8_;
        StaticallyIndexedArray<d4_t, 6> d4x6_;
        StaticallyIndexedArray<d6_t, 4> d6x4_;
        StaticallyIndexedArray<d8_t, 3> d8x3_;
        StaticallyIndexedArray<d12_t, 2> d12x2_;
        StaticallyIndexedArray<d24_t, 1> d24x12_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d3_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                          is_same<X, d12_t>::value || is_same<X, d24_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x24_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x12_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x8_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x6_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x4_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x3_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x2_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 36>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    using type = d36_t;

    union
    {
        d36_t d36_;
        StaticallyIndexedArray<d1_t, 36> d1x36_;
        StaticallyIndexedArray<d2_t, 18> d2x18_;
        StaticallyIndexedArray<d3_t, 12> d3x12_;
        StaticallyIndexedArray<d4_t, 9> d4x9_;
        StaticallyIndexedArray<d6_t, 6> d6x6_;
        StaticallyIndexedArray<d9_t, 4> d9x4_;
        StaticallyIndexedArray<d12_t, 3> d12x3_;
        StaticallyIndexedArray<d18_t, 2> d18x2_;
        StaticallyIndexedArray<d36_t, 1> d36x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x36_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x18_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x12_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x9_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x6_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x4_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x3_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x2_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 72>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d24_t __attribute__((ext_vector_type(24)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    using type = d72_t;

    union
    {
        d72_t d72_;
        StaticallyIndexedArray<d1_t, 72> d1x72_;
        StaticallyIndexedArray<d2_t, 36> d2x36_;
        StaticallyIndexedArray<d3_t, 24> d3x24_;
        StaticallyIndexedArray<d4_t, 18> d4x18_;
        StaticallyIndexedArray<d6_t, 12> d6x12_;
        StaticallyIndexedArray<d8_t, 9> d8x9_;
        StaticallyIndexedArray<d9_t, 8> d9x8_;
        StaticallyIndexedArray<d12_t, 6> d12x6_;
        StaticallyIndexedArray<d18_t, 4> d18x4_;
        StaticallyIndexedArray<d24_t, 3> d24x3_;
        StaticallyIndexedArray<d36_t, 2> d36x2_;
        StaticallyIndexedArray<d72_t, 1> d72x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d8_t>::value ||
                is_same<X, d9_t>::value || is_same<X, d12_t>::value || is_same<X, d18_t>::value ||
                is_same<X, d24_t>::value || is_same<X, d36_t>::value || is_same<X, d72_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x72_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x36_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x24_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x18_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x12_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x9_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x8_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x6_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x4_;
        }
        else if constexpr(is_same<X, d24_t>::value)
        {
            return data_.d24x3_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x2_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 144>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d6_t __attribute__((ext_vector_type(6)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    typedef T d12_t __attribute__((ext_vector_type(12)));
    typedef T d18_t __attribute__((ext_vector_type(18)));
    typedef T d36_t __attribute__((ext_vector_type(36)));
    typedef T d48_t __attribute__((ext_vector_type(48)));
    typedef T d72_t __attribute__((ext_vector_type(72)));
    typedef T d144_t __attribute__((ext_vector_type(144)));
    using type = d144_t;

    union
    {
        d144_t d144_;
        StaticallyIndexedArray<d1_t, 144> d1x144_;
        StaticallyIndexedArray<d2_t, 72> d2x72_;
        StaticallyIndexedArray<d3_t, 48> d3x48_;
        StaticallyIndexedArray<d4_t, 36> d4x36_;
        StaticallyIndexedArray<d6_t, 24> d6x24_;
        StaticallyIndexedArray<d9_t, 16> d9x16_;
        StaticallyIndexedArray<d12_t, 12> d12x12_;
        StaticallyIndexedArray<d18_t, 8> d18x8_;
        StaticallyIndexedArray<d36_t, 4> d36x4_;
        StaticallyIndexedArray<d48_t, 4> d48x3_;
        StaticallyIndexedArray<d72_t, 2> d72x2_;
        StaticallyIndexedArray<d144_t, 1> d144x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(
            is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value ||
                is_same<X, d4_t>::value || is_same<X, d6_t>::value || is_same<X, d9_t>::value ||
                is_same<X, d12_t>::value || is_same<X, d18_t>::value || is_same<X, d36_t>::value ||
                is_same<X, d48_t>::value || is_same<X, d72_t>::value || is_same<X, d144_t>::value,
            "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x144_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x72_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x482_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x36_;
        }
        else if constexpr(is_same<X, d6_t>::value)
        {
            return data_.d6x24_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x16_;
        }
        else if constexpr(is_same<X, d12_t>::value)
        {
            return data_.d12x12_;
        }
        else if constexpr(is_same<X, d18_t>::value)
        {
            return data_.d18x8_;
        }
        else if constexpr(is_same<X, d36_t>::value)
        {
            return data_.d36x4_;
        }
        else if constexpr(is_same<X, d48_t>::value)
        {
            return data_.d48x3_;
        }
        else if constexpr(is_same<X, d72_t>::value)
        {
            return data_.d72x2_;
        }
        else if constexpr(is_same<X, d144_t>::value)
        {
            return data_.d144x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 5>
{
    using d1_t = T;
    typedef T d5_t __attribute__((ext_vector_type(5)));
    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d5_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d5_t>::value, "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 10>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d5_t __attribute__((ext_vector_type(5)));
    typedef T d10_t __attribute__((ext_vector_type(10)));
    using type = d10_t;

    union
    {
        d10_t d10_;
        StaticallyIndexedArray<d1_t, 10> d1x10_;
        StaticallyIndexedArray<d2_t, 5> d2x5_;
        StaticallyIndexedArray<d5_t, 2> d5x2_;
        StaticallyIndexedArray<d10_t, 1> d10x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d5_t>::value || is_same<X, d10_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x10_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x2_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 20>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));
    typedef T d10_t __attribute__((ext_vector_type(10)));
    typedef T d20_t __attribute__((ext_vector_type(20)));
    using type = d20_t;

    union
    {
        d20_t d20_;
        StaticallyIndexedArray<d1_t, 20> d1x20_;
        StaticallyIndexedArray<d2_t, 10> d2x10_;
        StaticallyIndexedArray<d4_t, 5> d4x5_;
        StaticallyIndexedArray<d5_t, 4> d5x4_;
        StaticallyIndexedArray<d10_t, 2> d10x2_;
        StaticallyIndexedArray<d20_t, 1> d20x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d5_t>::value ||
                          is_same<X, d10_t>::value || is_same<X, d20_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x20_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x10_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x5_;
        }
        else if constexpr(is_same<X, d5_t>::value)
        {
            return data_.d5x4_;
        }
        else if constexpr(is_same<X, d10_t>::value)
        {
            return data_.d10x2_;
        }
        else if constexpr(is_same<X, d20_t>::value)
        {
            return data_.d20x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 9>
{
    using d1_t = T;
    typedef T d3_t __attribute__((ext_vector_type(3)));
    typedef T d9_t __attribute__((ext_vector_type(9)));
    using type = d9_t;

    union
    {
        d9_t d9_;
        StaticallyIndexedArray<d1_t, 9> d1x9_;
        StaticallyIndexedArray<d3_t, 3> d3x3_;
        StaticallyIndexedArray<d9_t, 1> d9x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d3_t>::value || is_same<X, d9_t>::value,
                      "wrong!");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x9_;
        }
        else if constexpr(is_same<X, d3_t>::value)
        {
            return data_.d3x3_;
        }
        else if constexpr(is_same<X, d9_t>::value)
        {
            return data_.d9x1_;
        }
        else
        {
            return err;
        }
    }
};
using int64_t = long;

// fp64
using double2_t = typename vector_type<double, 2>::type;
using double4_t = typename vector_type<double, 4>::type;

// fp32
using float2_t  = typename vector_type<float, 2>::type;
using float4_t  = typename vector_type<float, 4>::type;
using float8_t  = typename vector_type<float, 8>::type;
using float16_t = typename vector_type<float, 16>::type;
using float32_t = typename vector_type<float, 32>::type;
using float64_t = typename vector_type<float, 64>::type;

// fp16
using half2_t  = typename vector_type<half_t, 2>::type;
using half4_t  = typename vector_type<half_t, 4>::type;
using half8_t  = typename vector_type<half_t, 8>::type;
using half16_t = typename vector_type<half_t, 16>::type;
using half32_t = typename vector_type<half_t, 32>::type;
using half64_t = typename vector_type<half_t, 64>::type;

using half3_t  = typename vector_type<half_t, 3>::type;
using half6_t  = typename vector_type<half_t, 6>::type;
using half9_t  = typename vector_type<half_t, 9>::type;
using half18_t = typename vector_type<half_t, 18>::type;
using half36_t = typename vector_type<half_t, 36>::type;
using half5_t  = typename vector_type<half_t, 5>::type;
using half10_t = typename vector_type<half_t, 10>::type;

// bfp16
using bhalf2_t  = typename vector_type<bhalf_t, 2>::type;
using bhalf4_t  = typename vector_type<bhalf_t, 4>::type;
using bhalf8_t  = typename vector_type<bhalf_t, 8>::type;
using bhalf16_t = typename vector_type<bhalf_t, 16>::type;
using bhalf32_t = typename vector_type<bhalf_t, 32>::type;
using bhalf64_t = typename vector_type<bhalf_t, 64>::type;

using bhalf3_t  = typename vector_type<bhalf_t, 3>::type;
using bhalf6_t  = typename vector_type<bhalf_t, 6>::type;
using bhalf9_t  = typename vector_type<bhalf_t, 9>::type;
using bhalf18_t = typename vector_type<bhalf_t, 18>::type;
using bhalf36_t = typename vector_type<bhalf_t, 36>::type;
using bhalf5_t  = typename vector_type<bhalf_t, 5>::type;
using bhalf10_t = typename vector_type<bhalf_t, 10>::type;

using bf16x2_t  = typename vector_type<__bf16, 2>::type;
using bf16x4_t  = typename vector_type<__bf16, 4>::type;
using bf16x8_t  = typename vector_type<__bf16, 8>::type;
using bf16x16_t = typename vector_type<__bf16, 16>::type;
using bf16x32_t = typename vector_type<__bf16, 32>::type;
using bf16x64_t = typename vector_type<__bf16, 64>::type;

using bf16x3_t  = typename vector_type<__bf16, 3>::type;
using bf16x6_t  = typename vector_type<__bf16, 6>::type;
using bf16x9_t  = typename vector_type<__bf16, 9>::type;
using bf16x18_t = typename vector_type<__bf16, 18>::type;
using bf16x36_t = typename vector_type<__bf16, 36>::type;
using bf16x5_t  = typename vector_type<__bf16, 5>::type;
using bf16x10_t = typename vector_type<__bf16, 10>::type;
// i32
using int32x2_t  = typename vector_type<int32_t, 2>::type;
using int32x4_t  = typename vector_type<int32_t, 4>::type;
using int32x8_t  = typename vector_type<int32_t, 8>::type;
using int32x16_t = typename vector_type<int32_t, 16>::type;
using int32x32_t = typename vector_type<int32_t, 32>::type;
using int32x64_t = typename vector_type<int32_t, 64>::type;

using int32x3_t  = typename vector_type<int32_t, 3>::type;
using int32x5_t  = typename vector_type<int32_t, 5>::type;
using int32x9_t  = typename vector_type<int32_t, 9>::type;
using int32x18_t = typename vector_type<int32_t, 18>::type;

// i8
using int8x2_t  = typename vector_type<int8_t, 2>::type;
using int8x4_t  = typename vector_type<int8_t, 4>::type;
using int8x8_t  = typename vector_type<int8_t, 8>::type;
using int8x16_t = typename vector_type<int8_t, 16>::type;
using int8x32_t = typename vector_type<int8_t, 32>::type;
using int8x64_t = typename vector_type<int8_t, 64>::type;

using int8x3_t  = typename vector_type<int8_t, 3>::type;
using int8x6_t  = typename vector_type<int8_t, 6>::type;
using int8x9_t  = typename vector_type<int8_t, 9>::type;
using int8x12_t = typename vector_type<int8_t, 12>::type;
using int8x18_t = typename vector_type<int8_t, 18>::type;
using int8x24_t = typename vector_type<int8_t, 24>::type;
using int8x36_t = typename vector_type<int8_t, 36>::type;
using int8x72_t = typename vector_type<int8_t, 72>::type;
using int8x5_t  = typename vector_type<int8_t, 5>::type;
using int8x10_t = typename vector_type<int8_t, 10>::type;
using int8x20_t = typename vector_type<int8_t, 20>::type;

// f8
using f8x2_t  = typename vector_type<f8_t, 2>::type;
using f8x4_t  = typename vector_type<f8_t, 4>::type;
using f8x8_t  = typename vector_type<f8_t, 8>::type;
using f8x12_t = typename vector_type<f8_t, 12>::type;
using f8x16_t = typename vector_type<f8_t, 16>::type;
using f8x24_t = typename vector_type<f8_t, 24>::type;
using f8x32_t = typename vector_type<f8_t, 32>::type;
using f8x64_t = typename vector_type<f8_t, 64>::type;

using f8x3_t  = typename vector_type<f8_t, 3>::type;
using f8x6_t  = typename vector_type<f8_t, 6>::type;
using f8x9_t  = typename vector_type<f8_t, 9>::type;
using f8x12_t = typename vector_type<f8_t, 12>::type;
using f8x18_t = typename vector_type<f8_t, 18>::type;
using f8x24_t = typename vector_type<f8_t, 24>::type;
using f8x36_t = typename vector_type<f8_t, 36>::type;
using f8x72_t = typename vector_type<f8_t, 72>::type;
using f8x5_t  = typename vector_type<f8_t, 5>::type;
using f8x10_t = typename vector_type<f8_t, 10>::type;
using f8x20_t = typename vector_type<f8_t, 20>::type;

// bf8
using bf8x2_t  = typename vector_type<bf8_t, 2>::type;
using bf8x4_t  = typename vector_type<bf8_t, 4>::type;
using bf8x8_t  = typename vector_type<bf8_t, 8>::type;
using bf8x16_t = typename vector_type<bf8_t, 16>::type;
using bf8x32_t = typename vector_type<bf8_t, 32>::type;
using bf8x64_t = typename vector_type<bf8_t, 64>::type;

using bf8x3_t  = typename vector_type<bf8_t, 3>::type;
using bf8x6_t  = typename vector_type<bf8_t, 6>::type;
using bf8x9_t  = typename vector_type<bf8_t, 9>::type;
using bf8x12_t = typename vector_type<bf8_t, 12>::type;
using bf8x18_t = typename vector_type<bf8_t, 18>::type;
using bf8x24_t = typename vector_type<bf8_t, 24>::type;
using bf8x36_t = typename vector_type<bf8_t, 36>::type;
using bf8x72_t = typename vector_type<bf8_t, 72>::type;
using bf8x5_t  = typename vector_type<bf8_t, 5>::type;
using bf8x10_t = typename vector_type<bf8_t, 10>::type;
using bf8x20_t = typename vector_type<bf8_t, 20>::type;
// u8
using uint8x2_t  = typename vector_type<uint8_t, 2>::type;
using uint8x4_t  = typename vector_type<uint8_t, 4>::type;
using uint8x8_t  = typename vector_type<uint8_t, 8>::type;
using uint8x16_t = typename vector_type<uint8_t, 16>::type;
using uint8x32_t = typename vector_type<uint8_t, 32>::type;
using uint8x64_t = typename vector_type<uint8_t, 64>::type;

using uint8x3_t  = typename vector_type<uint8_t, 3>::type;
using uint8x6_t  = typename vector_type<uint8_t, 6>::type;
using uint8x9_t  = typename vector_type<uint8_t, 9>::type;
using uint8x12_t = typename vector_type<uint8_t, 12>::type;
using uint8x18_t = typename vector_type<uint8_t, 18>::type;
using uint8x24_t = typename vector_type<uint8_t, 24>::type;
using uint8x36_t = typename vector_type<uint8_t, 36>::type;
using uint8x72_t = typename vector_type<uint8_t, 72>::type;
using uint8x5_t  = typename vector_type<uint8_t, 5>::type;
using uint8x10_t = typename vector_type<uint8_t, 10>::type;
using uint8x20_t = typename vector_type<uint8_t, 20>::type;

//__fp16
using fp16x2_t  = typename vector_type<__fp16, 2>::type;
using fp16x4_t  = typename vector_type<__fp16, 4>::type;
using fp16x8_t  = typename vector_type<__fp16, 8>::type;
using fp16x16_t = typename vector_type<__fp16, 16>::type;

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using int4x8_t  = typename vector_type<int8_t, 4>::type;
using int4x16_t = typename vector_type<int8_t, 8>::type;
using int4x32_t = typename vector_type<int8_t, 16>::type;
using int4x64_t = typename vector_type<int8_t, 32>::type;

using int4x24_t  = typename vector_type<int8_t, 12>::type;
using int4x48_t  = typename vector_type<int8_t, 24>::type;
using int4x72_t  = typename vector_type<int8_t, 36>::type;
using int4x144_t = typename vector_type<int8_t, 72>::type;
using int4x40_t  = typename vector_type<int8_t, 20>::type;
#endif

template <MTX_FMT MX_TYPE = MTX_FMT::MTX_FMT_DEFAULT>
struct MxType_t
{
    using type_t = unsigned _BitInt(8);
    using vec_t  = int32x8_t; // used for wmma
    MxType_t()   = default;
    MxType_t(type_t data) { m_data = data; }
    MxType_t(const MxType_t& other) { m_data = other.m_data; }
    type_t m_data;
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP4_E2M1> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP4_E2M1;
    static constexpr index_t vec_size         = 4;
    static constexpr index_t dwords_per_wmmak = 8;
    static constexpr index_t BITS             = 4;
    // the below function is used for data preparation in host side; not used in device side
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        std::vector<type_t> in_packed;
        std::size_t vec_size = in.size();
        in_packed.reserve(vec_size);
        for(std::size_t i = 0; i < vec_size; i += 2)
        {
            type_t val0       = in[i];
            type_t val1       = in[i + 1];
            in_packed[i >> 1] = val0 | (val1 << 4);
        }

        return in_packed;
    }
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP6_E3M2> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType = MTX_FMT::MTX_FMT_FP6_E3M2;
    // this is used per K-dimension how many int32_t need to load; used in wmma_op shader
    static constexpr index_t vec_size         = 6;
    static constexpr index_t dwords_per_wmmak = 12;
    static constexpr index_t BITS             = 6;
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        std::vector<type_t> in_packed;
        std::size_t vec_size = in.size();
        in_packed.reserve(vec_size);
        std::size_t index = 0;
        for(std::size_t i = 0; i < vec_size; i += 4)
        {
            type_t val0        = in[i];
            type_t val1        = in[i + 1];
            type_t val2        = in[i + 2];
            type_t val3        = in[i + 3];
            in_packed[index++] = val0 | (val1 & 0x3) << 6;
            in_packed[index++] = (val1 & 0x3C) >> 2 | (val2 & 0xf) << 4;
            in_packed[index++] = (val2 & 0x3C) >> 4 | (val3 & 0x3f) << 2;
        }
        return in_packed;
    }
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP6_E2M3> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP6_E2M3;
    static constexpr index_t vec_size         = 6;
    static constexpr index_t dwords_per_wmmak = 12;
    static constexpr index_t BITS             = 6;
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in)
    {
        return MxType_t<MTX_FMT::MTX_FMT_FP6_E3M2>::compact_to_raw(in);
    }
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP8_E4M3> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    using type_t                              = f8_t;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP8_E4M3;
    static constexpr index_t vec_size         = 8;
    static constexpr index_t dwords_per_wmmak = 16;
    static constexpr index_t BITS             = 8;
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in) { return in; }
};

template <>
struct MxType_t<MTX_FMT::MTX_FMT_FP8_E5M2> : public MxType_t<MTX_FMT::MTX_FMT_DEFAULT>
{
    using parent = MxType_t<MTX_FMT::MTX_FMT_DEFAULT>;
    using parent::parent;
    using type_t                              = bf8_t;
    static constexpr MTX_FMT RawType          = MTX_FMT::MTX_FMT_FP8_E5M2;
    static constexpr index_t vec_size         = 8;
    static constexpr index_t dwords_per_wmmak = 16;
    static constexpr index_t BITS             = 8;
    static std::vector<type_t> compact_to_raw(const std::vector<type_t>& in) { return in; }
};

template <typename T>
struct is_mx_type_t : std::false_type
{
};

// Specialize the template for ck::MxType_t
template <MTX_FMT MX_TYPE>
struct is_mx_type_t<MxType_t<MX_TYPE>> : std::true_type
{
};

// Helper variable template
template <typename T>
inline constexpr bool is_mx_type_t_v = is_mx_type_t<T>::value;

template <typename T, typename = void>
struct view_type
{
    using srcType  = T;
    using viewType = T;
};

template <typename T>
struct view_type<T, enable_if_t<is_mx_type_t<T>::value>>
{
    using srcType  = typename T::type_t;
    using viewType = int32_t;
};

template <typename T>
using src_type_t = typename view_type<T>::srcType;

template <typename T>
using view_type_t = typename view_type<T>::viewType;

template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }

    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }

    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }

    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_t Min() { return f8_t(binary_min); }

    __host__ __device__ static constexpr f8_t Max() { return f8_t(binary_max); }

    __host__ __device__ static constexpr f8_t Lowest() { return f8_t(binary_lowest); }

    __host__ __device__ static constexpr f8_t QuietNaN() { return f8_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_t Min() { return bf8_t(binary_min); }

    __host__ __device__ static constexpr bf8_t Max() { return bf8_t(binary_max); }

    __host__ __device__ static constexpr bf8_t Lowest() { return bf8_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_t QuietNaN() { return bf8_t(binary_qnan); }
};

template <typename T>
struct NumericUtils
{
};

template <>
struct NumericUtils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    using bitwise_type                  = uint32_t;
};

template <>
struct NumericUtils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    using bitwise_type                  = uint16_t;
};

template <>
struct NumericUtils<f8_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    // static constexpr int bias = 8; // negative zero nan mode
    static constexpr int bias = 7; // ieee mode
};

template <>
struct NumericUtils<bf8_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    // static constexpr int bias = 16; // negative zero nan mode
    static constexpr int bias = 15; // ieee mode
};

template <>
struct NumericUtils<bhalf_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 7;
    static constexpr int bias = 128; // negative zero nan mode
    // static constexpr int bias = 127; // ieee mode
};

// the below is used for MX data format
template <>
struct NumericUtils<MxType_t<MTX_FMT::MTX_FMT_FP4_E2M1>>
{
    static constexpr int exp              = 2;
    static constexpr int mant             = 1;
    static constexpr float absmin_nonzero = 0.5;
    static constexpr float absmax         = 6.f;
};

template <>
struct NumericUtils<MxType_t<MTX_FMT::MTX_FMT_FP6_E3M2>>
{
    static constexpr int exp              = 3;
    static constexpr int mant             = 2;
    static constexpr float absmin_nonzero = 0.0625;
    static constexpr float absmax         = 28.f;
};

template <>
struct NumericUtils<MxType_t<MTX_FMT::MTX_FMT_FP6_E2M3>>
{
    static constexpr int exp              = 2;
    static constexpr int mant             = 3;
    static constexpr float absmin_nonzero = 0.125;
    static constexpr float absmax         = 7.5;
};
} // namespace ck
