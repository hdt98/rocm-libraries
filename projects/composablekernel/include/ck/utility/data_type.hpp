// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/e8m0.hpp"
#include "ck/utility/statically_indexed_array.hpp"
#include <vector>
#define CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 1

/// Definitions from <cstdint>, <cmath> conflict with
/// /opt/rocm/include/hip/amd_detail/amd_hip_vector_types.h.

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
using int8_t   = signed char;
using uint8_t  = unsigned char;
using int16_t  = signed short;
using uint16_t = unsigned short;
using float_t  = float;
#endif // __HIPCC_RTC__

namespace ck {
#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
using byte = unsigned char;
#else
using std::byte;
#endif

#if defined(__clang_major__) && (__clang_major__ >= 21)
using bhalf_t = __bf16;
#else
using bhalf_t = ushort;
#endif
using half_t  = _Float16;
using int4_t  = _BitInt(4);
using uint4_t = unsigned _BitInt(4);
using f4_t    = unsigned _BitInt(4);
using f6_t    = _BitInt(6);          // e2m3 format
using bf6_t   = unsigned _BitInt(6); // e3m2 format

struct f4x2_pk_t
{
    using type = uint8_t;
    type data;
    f4x2_pk_t() : data{type{}} {}
    f4x2_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline type unpack(Number<I>) const
    {
        static_assert(I < 2, "Index is out of range.");
        if constexpr(I == 0)
            return data & 0b00001111;
        else
            return (data >> 4);
    }

    __host__ __device__ inline type pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }
};

struct f6x16_pk_t
{
    // store 16 elements of f6_t in an array of 3 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 3>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    f6x16_pk_t() : data{type{}} {}
    f6x16_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline f6_t unpack(Number<I>)
    {
        static_assert(I < 16, "Index out of range for 16 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 3;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<f6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 16 f6_t values, place its 6 bits in the correct position
        ck::static_for<0, 16, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 3;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct f6x32_pk_t
{
    // store 32 elements of f6_t in an array of 6 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 6>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    f6x32_pk_t() : data{type{}} {}
    f6x32_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline f6_t unpack(Number<I>)
    {
        static_assert(I < 32, "Index out of range for 32 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 6;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<f6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 32 f6_t values, place its 6 bits in the correct position
        ck::static_for<0, 32, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 6;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct bf6x16_pk_t
{
    // store 16 elements of bf6_t in an array of 3 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 3>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    bf6x16_pk_t() : data{type{}} {}
    bf6x16_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline bf6_t unpack(Number<I>)
    {
        static_assert(I < 16, "Index out of range for 16 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 3;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<bf6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 16 bf6_t values, place its 6 bits in the correct position
        ck::static_for<0, 16, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 3;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

struct bf6x32_pk_t
{
    // store 32 elements of bf6_t in an array of 6 uint32_t
    using element_type = uint32_t;
    using type         = StaticallyIndexedArray_v2<element_type, 6>;
    type data;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    bf6x32_pk_t() : data{type{}} {}
    bf6x32_pk_t(type init) : data{init} {}

    template <index_t I>
    __host__ __device__ inline bf6_t unpack(Number<I>)
    {
        static_assert(I < 32, "Index out of range for 32 f6_t elements.");

        constexpr int num_bits_elem     = 6;
        constexpr int num_bits_vec_elem = 32;
        constexpr int vector_size       = 6;
        constexpr int bit_pos           = I * num_bits_elem;
        constexpr int arr_idx           = bit_pos / num_bits_vec_elem;
        constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
        uint32_t bits                   = data.At(Number<arr_idx>{}) >> bit_offset;
        constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;

        if constexpr(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (data.At(Number<arr_idx + 1>{}) & ((1u << overhang) - 1))
                    << (num_bits_elem - overhang);
        }

        return static_cast<bf6_t>(bits & 0x3F);
    }

    __host__ __device__ inline type pack(const test_vec_t& x)
    {
        type packed{};

        // for each of the 32 bf6_t values, place its 6 bits in the correct position
        ck::static_for<0, 32, 1>{}([&](auto i) {
            uint32_t bits                   = static_cast<uint32_t>(x[static_cast<int>(i)]) & 0x3F;
            constexpr int num_bits_elem     = 6;
            constexpr int num_bits_vec_elem = 32;
            constexpr int vector_size       = 6;
            constexpr int bit_pos           = i * num_bits_elem;
            constexpr int arr_index         = bit_pos / num_bits_vec_elem;
            constexpr int bit_offset        = bit_pos % num_bits_vec_elem;
            constexpr int overhang          = bit_offset + num_bits_elem - num_bits_vec_elem;
            uint32_t old_value              = packed.At(Number<arr_index>{});

            // insert bits into the current 32-bit block
            old_value |= (bits << bit_offset);
            packed.At(Number<arr_index>{}) = old_value;

            // if it crosses into the next block, shift the remainder
            if constexpr(overhang > 0 && (arr_index + 1) < vector_size)
            {
                uint32_t next_value = packed.At(Number<arr_index + 1>{});
                next_value |= (bits >> (num_bits_elem - overhang));
                packed.At(Number<arr_index + 1>{}) = next_value;
            }
        });

        return packed;
    }
};

// custom data type - pack int4 data
struct pk_i4_t
{
    using type = int8_t;
    type data;
    __host__ __device__ constexpr pk_i4_t() : data{type{}} {}
    __host__ __device__ constexpr pk_i4_t(type init) : data{init} {}
};

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

// native types: double, float, _Float16, ushort, int32_t, int8_t, uint8_t, f8_fnuz_t, bf8_fnuz_t,
// native types: bool, f4_t, f6_t, bf6_t
template <typename T>
inline constexpr bool is_native_type()
{
    return is_same<T, double>::value || is_same<T, float>::value || is_same<T, half_t>::value ||
           is_same<T, bhalf_t>::value || is_same<T, int32_t>::value || is_same<T, int8_t>::value ||
           is_same<T, uint8_t>::value || is_same<T, f8_fnuz_t>::value ||
           is_same<T, bf8_fnuz_t>::value || is_same<T, bool>::value || is_same<T, f4_t>::value ||
           is_same<T, f6_t>::value || is_same<T, bf6_t>::value || is_same<T, __bf16>::value ||
           is_same<T, __fp16>::value;
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
struct scalar_type<pk_i4_t>
{
    using type                           = pk_i4_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_fnuz_t>
{
    using type                           = f8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_fnuz_t>
{
    using type                           = bf8_fnuz_t;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<f8_ocp_t>
{
    using type                           = f8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bf8_ocp_t>
{
    using type                           = bf8_ocp_t::data_type;
    static constexpr index_t vector_size = 1;
};

template <>
struct scalar_type<bool>
{
    using type                           = bool;
    static constexpr index_t vector_size = 1;
};

template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 2, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 3, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d3_t __attribute__((ext_vector_type(3)));

    using type = d3_t;

    union
    {
        d3_t d3_;
        StaticallyIndexedArray<d1_t, 3> d1x3_;
        StaticallyIndexedArray<d2_t, 1> d2x1_;
        StaticallyIndexedArray<d3_t, 1> d3x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value || is_same<X, d3_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x3_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x1_;
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
struct vector_type<T, 4, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 5, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d5_t __attribute__((ext_vector_type(5)));

    using type = d5_t;

    union
    {
        d5_t d5_;
        StaticallyIndexedArray<d1_t, 5> d1x5_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d5_t, 1> d5x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value || is_same<X, d5_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x5_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
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
struct vector_type<T, 7, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d2_t __attribute__((ext_vector_type(2)));
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d7_t __attribute__((ext_vector_type(7)));

    using type = d7_t;

    union
    {
        d7_t d7_;
        StaticallyIndexedArray<d1_t, 7> d1x7_;
        StaticallyIndexedArray<d2_t, 3> d2x3_;
        StaticallyIndexedArray<d4_t, 1> d4x1_;
        StaticallyIndexedArray<d7_t, 1> d7x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d2_t>::value ||
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
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
                          is_same<X, d4_t>::value || is_same<X, d7_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x7_;
        }
        else if constexpr(is_same<X, d2_t>::value)
        {
            return data_.d2x3_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x1_;
        }
        else if constexpr(is_same<X, d7_t>::value)
        {
            return data_.d7x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 8, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 13, typename ck::enable_if_t<is_native_type<T>()>>
{
    using d1_t = T;
    typedef T d4_t __attribute__((ext_vector_type(4)));
    typedef T d8_t __attribute__((ext_vector_type(8)));
    typedef T d13_t __attribute__((ext_vector_type(13)));

    using type = d13_t;

    union
    {
        d13_t d13_;
        StaticallyIndexedArray<d1_t, 13> d1x13_;
        StaticallyIndexedArray<d4_t, 3> d4x3_;
        StaticallyIndexedArray<d8_t, 1> d8x1_;
        StaticallyIndexedArray<d13_t, 1> d13x1_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d13_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value)
        {
            return data_.d1x13_;
        }
        else if constexpr(is_same<X, d4_t>::value)
        {
            return data_.d4x3_;
        }
        else if constexpr(is_same<X, d8_t>::value)
        {
            return data_.d8x1_;
        }
        else if constexpr(is_same<X, d13_t>::value)
        {
            return data_.d13x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 16, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 32, typename ck::enable_if_t<is_native_type<T>()>>
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
    } data_ = {d32_t{0}};

    __attribute__((host)) __attribute__((device)) constexpr vector_type() {}

    __attribute__((host)) __attribute__((device)) constexpr vector_type(type v) { (void)v; }

    // __host__ __device__ constexpr vector_type() : data_{type{0}} {}

    // __host__ __device__ constexpr vector_type(type v) : data_{v} {}

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
struct vector_type<T, 64, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 128, typename ck::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 256, typename ck::enable_if_t<is_native_type<T>()>>
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

template <typename T, index_t N, typename Enable = void>
struct non_native_vector_base;

template <typename T>
struct nnvb_data_t_selector
{
    using type = unsigned _BitInt(8 * sizeof(T));
};

template <>
struct nnvb_data_t_selector<f8_ocp_t>
{
    using type = f8_ocp_t::data_type;
};

template <>
struct nnvb_data_t_selector<bf8_ocp_t>
{
    using type = bf8_ocp_t::data_type;
};

template <>
struct nnvb_data_t_selector<f6x16_pk_t>
{
    using type = f6x16_pk_t::type;
};

template <>
struct nnvb_data_t_selector<f6x32_pk_t>
{
    using type = f6x32_pk_t::type;
};

template <>
struct nnvb_data_t_selector<bf6x16_pk_t>
{
    using type = bf6x16_pk_t::type;
};

template <>
struct nnvb_data_t_selector<bf6x32_pk_t>
{
    using type = bf6x32_pk_t::type;
};

template <>
struct nnvb_data_t_selector<pk_i4_t>
{
    using type = pk_i4_t::type;
};

template <typename T, index_t N>
struct non_native_vector_base<
    T,
    N,
    ck::enable_if_t<sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8>>
{
    using data_t = typename nnvb_data_t_selector<T>::type; // select data_t based on the size of T
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    using data_v = data_t __attribute__((ext_vector_type(N)));
    using type   = non_native_vector_base<T, N>;

    union alignas(next_pow2(N * sizeof(T)))
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray<data_t, N> dxN;
        StaticallyIndexedArray<T, N> dTxN;
        StaticallyIndexedArray<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a) : data_{data_v(a)} {}
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same_v<X, data_t> || is_same_v<X, T> || is_same_v<X, data_v>,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same_v<X, data_t>)
        {
            return data_.dxN;
        }
        else if constexpr(is_same_v<X, T>)
        {
            return data_.dTxN;
        }
        else if constexpr(is_same_v<X, data_v>)
        {
            return data_.dNx1;
        }
        else
        {
            return err;
        }
    }
};

// implementation for f6x16 and f6x32
template <typename T, index_t N>
struct non_native_vector_base<T, N, ck::enable_if_t<sizeof(T) == 12 || sizeof(T) == 24>>
{
    using data_t =
        typename nnvb_data_t_selector<T>::type; // select data_t based on declared base type
    using element_t = typename T::element_type; // select element_t based on declared element type
    static_assert(sizeof(T) == sizeof(data_t), "non_native_vector_base storage size mismatch");
    static constexpr size_t size_factor =
        sizeof(data_t) / sizeof(element_t); // f6x16: 12/4 = 3, f6x32: 24/4 = 6
    using data_v = element_t __attribute__((ext_vector_type(N * size_factor)));
    using type   = non_native_vector_base<T, N>;

    union alignas(next_pow2(N * sizeof(T)))
    {
        data_v dN; // storage vector;
        StaticallyIndexedArray<data_t, N> dxN;
        StaticallyIndexedArray<T, N> dTxN;
        StaticallyIndexedArray<data_v, 1> dNx1;
    } data_;

    __host__ __device__ constexpr non_native_vector_base(data_t a)
        : data_{data_v(a.At(Number<0>{}))}
    {
    }
    __host__ __device__ constexpr non_native_vector_base(T f)
        : non_native_vector_base(bit_cast<data_t>(f))
    {
    }
    __host__ __device__ constexpr non_native_vector_base() : non_native_vector_base(T{}){};
    __host__ __device__ constexpr non_native_vector_base(data_v v) : data_{v} {}

    __host__ __device__ constexpr operator data_v() const { return data_.dN; }
    __host__ __device__ constexpr operator data_t() const
    {
        if constexpr(N == 1)
        {
            return data_.dxN[Number<0>{}];
        }
        else
        {
            return data_.dxN; // XXX this should cause an error
        }
    }
    __host__ __device__ constexpr operator T() const
    {
        if constexpr(N == 1)
        {
            return data_.dTxN[Number<0>{}];
        }
        else
        {
            return data_.dTxN; // XXX this should cause an error
        }
    }
};

template <typename T, index_t N>
struct scalar_type<non_native_vector_base<T, N>>;

template <index_t N>
struct scalar_type<non_native_vector_base<f8_ocp_t, N>>
{
    using type = typename non_native_vector_base<f8_ocp_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

template <index_t N>
struct scalar_type<non_native_vector_base<bf8_ocp_t, N>>
{
    using type = typename non_native_vector_base<bf8_ocp_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

template <typename T>
struct vector_type<T, 6, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 12, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 18, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 24, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 36, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 72, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 144, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 10, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 20, typename std::enable_if_t<is_native_type<T>()>>
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
struct vector_type<T, 9, typename std::enable_if_t<is_native_type<T>()>>
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
template <index_t N>
struct scalar_type<non_native_vector_base<pk_i4_t, N>>
{
    using type = typename non_native_vector_base<pk_i4_t, N>::data_t;

    static constexpr index_t vector_size = N;
};

// non-native vector_type implementation
template <typename T>
struct vector_type<T, 1, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using type     = d1_nnv_t;

    union alignas(next_pow2(1 * sizeof(T)))
    {
        d1_t d1_;
        StaticallyIndexedArray<d1_t, 1> d1x1_;
        d1_nnv_t d1_nnv_;
    } data_;

    __host__ __device__ constexpr vector_type() : data_{d1_t{}} {}

    __host__ __device__ constexpr vector_type(type v) : data_{v} {}

    template <typename X>
    __host__ __device__ constexpr const auto& AsType() const
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }

    template <typename X>
    __host__ __device__ constexpr auto& AsType()
    {
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
        {
            return data_.d1x1_;
        }
        else
        {
            return err;
        }
    }
};

template <typename T>
struct vector_type<T, 2, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;

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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
struct vector_type<T, 4, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;

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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
struct vector_type<T, 8, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;

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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
struct vector_type<T, 16, typename ck::enable_if_t<!is_native_type<T>()>>
{
    using d1_t     = T;
    using d1_nnv_t = non_native_vector_base<T, 1>;
    using d2_t     = non_native_vector_base<T, 2>;
    using d4_t     = non_native_vector_base<T, 4>;
    using d8_t     = non_native_vector_base<T, 8>;
    using d16_t    = non_native_vector_base<T, 16>;

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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
        static_assert(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value ||
                          is_same<X, d2_t>::value || is_same<X, d4_t>::value ||
                          is_same<X, d8_t>::value || is_same<X, d16_t>::value,
                      "Something went wrong, please check src and dst types.");

        if constexpr(is_same<X, d1_t>::value || is_same<X, d1_nnv_t>::value)
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
struct vector_type<T, 32, typename ck::enable_if_t<!is_native_type<T>()>>
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
struct vector_type<T, 64, typename ck::enable_if_t<!is_native_type<T>()>>
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
struct vector_type<T, 128, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d8_t   = non_native_vector_base<T, 8>;
    using d16_t  = non_native_vector_base<T, 16>;
    using d32_t  = non_native_vector_base<T, 32>;
    using d64_t  = non_native_vector_base<T, 64>;
    using d128_t = non_native_vector_base<T, 128>;

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
struct vector_type<T, 256, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d8_t   = non_native_vector_base<T, 8>;
    using d16_t  = non_native_vector_base<T, 16>;
    using d32_t  = non_native_vector_base<T, 32>;
    using d64_t  = non_native_vector_base<T, 64>;
    using d128_t = non_native_vector_base<T, 128>;
    using d256_t = non_native_vector_base<T, 256>;

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

template <typename T>
struct vector_type<T, 3, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d3_t = non_native_vector_base<T, 3>;

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
struct vector_type<T, 6, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d2_t = non_native_vector_base<T, 2>;
    using d3_t = non_native_vector_base<T, 3>;
    using d6_t = non_native_vector_base<T, 6>;

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
struct vector_type<T, 12, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d12_t = non_native_vector_base<T, 12>;
    using type  = d12_t;

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
struct vector_type<T, 18, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d18_t = non_native_vector_base<T, 18>;

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
struct vector_type<T, 24, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d12_t = non_native_vector_base<T, 12>;
    using d24_t = non_native_vector_base<T, 242>;
    using type  = d24_t;

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
struct vector_type<T, 36, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d12_t = non_native_vector_base<T, 12>;
    using d18_t = non_native_vector_base<T, 18>;
    using d36_t = non_native_vector_base<T, 36>;
    using type  = d36_t;

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
struct vector_type<T, 72, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d3_t  = non_native_vector_base<T, 3>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d6_t  = non_native_vector_base<T, 6>;
    using d8_t  = non_native_vector_base<T, 8>;
    using d9_t  = non_native_vector_base<T, 9>;
    using d12_t = non_native_vector_base<T, 12>;
    using d18_t = non_native_vector_base<T, 18>;
    using d24_t = non_native_vector_base<T, 24>;
    using d36_t = non_native_vector_base<T, 36>;
    using d72_t = non_native_vector_base<T, 72>;
    using type  = d72_t;

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
struct vector_type<T, 144, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t   = T;
    using d2_t   = non_native_vector_base<T, 2>;
    using d3_t   = non_native_vector_base<T, 3>;
    using d4_t   = non_native_vector_base<T, 4>;
    using d6_t   = non_native_vector_base<T, 6>;
    using d9_t   = non_native_vector_base<T, 9>;
    using d12_t  = non_native_vector_base<T, 12>;
    using d18_t  = non_native_vector_base<T, 18>;
    using d36_t  = non_native_vector_base<T, 36>;
    using d48_t  = non_native_vector_base<T, 48>;
    using d72_t  = non_native_vector_base<T, 72>;
    using d144_t = non_native_vector_base<T, 144>;
    using type   = d144_t;

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
struct vector_type<T, 5, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d5_t = non_native_vector_base<T, 5>;
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
struct vector_type<T, 10, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d5_t  = non_native_vector_base<T, 5>;
    using d10_t = non_native_vector_base<T, 10>;
    using type  = d10_t;

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
struct vector_type<T, 20, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t  = T;
    using d2_t  = non_native_vector_base<T, 2>;
    using d4_t  = non_native_vector_base<T, 4>;
    using d5_t  = non_native_vector_base<T, 5>;
    using d10_t = non_native_vector_base<T, 10>;
    using d20_t = non_native_vector_base<T, 20>;
    using type  = d20_t;

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
struct vector_type<T, 9, typename std::enable_if_t<!is_native_type<T>()>>
{
    using d1_t = T;
    using d3_t = non_native_vector_base<T, 3>;
    using d9_t = non_native_vector_base<T, 9>;
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

// i16
using int16x2_t = typename vector_type<int16_t, 2>::type;
using int16x4_t = typename vector_type<int16_t, 4>::type;
using int16x8_t = typename vector_type<int16_t, 8>::type;

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
using f8x2_fnuz_t  = typename vector_type<f8_fnuz_t, 2>::type;
using f8x4_fnuz_t  = typename vector_type<f8_fnuz_t, 4>::type;
using f8x8_fnuz_t  = typename vector_type<f8_fnuz_t, 8>::type;
using f8x16_fnuz_t = typename vector_type<f8_fnuz_t, 16>::type;
using f8x32_fnuz_t = typename vector_type<f8_fnuz_t, 32>::type;
using f8x64_fnuz_t = typename vector_type<f8_fnuz_t, 64>::type;

using f8x3_fnuz_t  = typename vector_type<f8_fnuz_t, 3>::type;
using f8x5_fnuz_t  = typename vector_type<f8_fnuz_t, 5>::type;
using f8x6_fnuz_t  = typename vector_type<f8_fnuz_t, 6>::type;
using f8x9_fnuz_t  = typename vector_type<f8_fnuz_t, 9>::type;
using f8x10_fnuz_t = typename vector_type<f8_fnuz_t, 10>::type;
using f8x12_fnuz_t = typename vector_type<f8_fnuz_t, 12>::type;
using f8x18_fnuz_t = typename vector_type<f8_fnuz_t, 18>::type;
using f8x20_fnuz_t = typename vector_type<f8_fnuz_t, 20>::type;
using f8x24_fnuz_t = typename vector_type<f8_fnuz_t, 24>::type;
using f8x36_fnuz_t = typename vector_type<f8_fnuz_t, 36>::type;
using f8x72_fnuz_t = typename vector_type<f8_fnuz_t, 72>::type;

// bf8
using bf8x2_fnuz_t  = typename vector_type<bf8_fnuz_t, 2>::type;
using bf8x4_fnuz_t  = typename vector_type<bf8_fnuz_t, 4>::type;
using bf8x8_fnuz_t  = typename vector_type<bf8_fnuz_t, 8>::type;
using bf8x16_fnuz_t = typename vector_type<bf8_fnuz_t, 16>::type;
using bf8x32_fnuz_t = typename vector_type<bf8_fnuz_t, 32>::type;
using bf8x64_fnuz_t = typename vector_type<bf8_fnuz_t, 64>::type;

using bf8x3_fnuz_t  = typename vector_type<bf8_fnuz_t, 3>::type;
using bf8x5_fnuz_t  = typename vector_type<bf8_fnuz_t, 5>::type;
using bf8x6_fnuz_t  = typename vector_type<bf8_fnuz_t, 6>::type;
using bf8x9_fnuz_t  = typename vector_type<bf8_fnuz_t, 9>::type;
using bf8x10_fnuz_t = typename vector_type<bf8_fnuz_t, 10>::type;
using bf8x12_fnuz_t = typename vector_type<bf8_fnuz_t, 12>::type;
using bf8x18_fnuz_t = typename vector_type<bf8_fnuz_t, 18>::type;
using bf8x20_fnuz_t = typename vector_type<bf8_fnuz_t, 20>::type;
using bf8x24_fnuz_t = typename vector_type<bf8_fnuz_t, 24>::type;
using bf8x36_fnuz_t = typename vector_type<bf8_fnuz_t, 36>::type;
using bf8x72_fnuz_t = typename vector_type<bf8_fnuz_t, 72>::type;

// f8
using f8x2_ocp_t  = typename vector_type<f8_ocp_t, 2>::type;
using f8x4_ocp_t  = typename vector_type<f8_ocp_t, 4>::type;
using f8x8_ocp_t  = typename vector_type<f8_ocp_t, 8>::type;
using f8x16_ocp_t = typename vector_type<f8_ocp_t, 16>::type;
using f8x32_ocp_t = typename vector_type<f8_ocp_t, 32>::type;
using f8x64_ocp_t = typename vector_type<f8_ocp_t, 64>::type;

using f8x3_ocp_t  = typename vector_type<f8_ocp_t, 3>::type;
using f8x5_ocp_t  = typename vector_type<f8_ocp_t, 5>::type;
using f8x6_ocp_t  = typename vector_type<f8_ocp_t, 6>::type;
using f8x9_ocp_t  = typename vector_type<f8_ocp_t, 9>::type;
using f8x10_ocp_t = typename vector_type<f8_ocp_t, 10>::type;
using f8x12_ocp_t = typename vector_type<f8_ocp_t, 12>::type;
using f8x18_ocp_t = typename vector_type<f8_ocp_t, 18>::type;
using f8x20_ocp_t = typename vector_type<f8_ocp_t, 20>::type;
using f8x24_ocp_t = typename vector_type<f8_ocp_t, 24>::type;
using f8x36_ocp_t = typename vector_type<f8_ocp_t, 36>::type;
using f8x72_ocp_t = typename vector_type<f8_ocp_t, 72>::type;

// bf8
using bf8x2_ocp_t  = typename vector_type<bf8_ocp_t, 2>::type;
using bf8x4_ocp_t  = typename vector_type<bf8_ocp_t, 4>::type;
using bf8x8_ocp_t  = typename vector_type<bf8_ocp_t, 8>::type;
using bf8x16_ocp_t = typename vector_type<bf8_ocp_t, 16>::type;
using bf8x32_ocp_t = typename vector_type<bf8_ocp_t, 32>::type;
using bf8x64_ocp_t = typename vector_type<bf8_ocp_t, 64>::type;

using bf8x3_ocp_t  = typename vector_type<bf8_ocp_t, 3>::type;
using bf8x5_ocp_t  = typename vector_type<bf8_ocp_t, 5>::type;
using bf8x6_ocp_t  = typename vector_type<bf8_ocp_t, 6>::type;
using bf8x9_ocp_t  = typename vector_type<bf8_ocp_t, 9>::type;
using bf8x10_ocp_t = typename vector_type<bf8_ocp_t, 10>::type;
using bf8x12_ocp_t = typename vector_type<bf8_ocp_t, 12>::type;
using bf8x18_ocp_t = typename vector_type<bf8_ocp_t, 18>::type;
using bf8x20_ocp_t = typename vector_type<bf8_ocp_t, 20>::type;
using bf8x24_ocp_t = typename vector_type<bf8_ocp_t, 24>::type;
using bf8x36_ocp_t = typename vector_type<bf8_ocp_t, 36>::type;
using bf8x72_ocp_t = typename vector_type<bf8_ocp_t, 72>::type;

#if CK_FP8_TYPE_OCP
// f8
using f8x2_t  = f8x2_ocp_t;
using f8x4_t  = f8x4_ocp_t;
using f8x8_t  = f8x8_ocp_t;
using f8x16_t = f8x16_ocp_t;
using f8x32_t = f8x32_ocp_t;
using f8x64_t = f8x64_ocp_t;

using f8x3_t  = f8x3_ocp_t;
using f8x5_t  = f8x5_ocp_t;
using f8x6_t  = f8x6_ocp_t;
using f8x9_t  = f8x9_ocp_t;
using f8x10_t = f8x10_ocp_t;
using f8x12_t = f8x12_ocp_t;
using f8x18_t = f8x18_ocp_t;
using f8x20_t = f8x20_ocp_t;
using f8x24_t = f8x24_ocp_t;
using f8x36_t = f8x36_ocp_t;
using f8x72_t = f8x72_ocp_t;
// bf8
using bf8x2_t  = bf8x2_ocp_t;
using bf8x4_t  = bf8x4_ocp_t;
using bf8x8_t  = bf8x8_ocp_t;
using bf8x16_t = bf8x16_ocp_t;
using bf8x32_t = bf8x32_ocp_t;
using bf8x64_t = bf8x64_ocp_t;

using bf8x3_t  = bf8x3_ocp_t;
using bf8x5_t  = bf8x5_ocp_t;
using bf8x6_t  = bf8x6_ocp_t;
using bf8x9_t  = bf8x9_ocp_t;
using bf8x10_t = bf8x10_ocp_t;
using bf8x12_t = bf8x12_ocp_t;
using bf8x18_t = bf8x18_ocp_t;
using bf8x20_t = bf8x20_ocp_t;
using bf8x24_t = bf8x24_ocp_t;
using bf8x36_t = bf8x36_ocp_t;
using bf8x72_t = bf8x72_ocp_t;

#elif CK_FP8_TYPE_FNUZ
// f8
using f8x2_t  = f8x2_fnuz_t;
using f8x4_t  = f8x4_fnuz_t;
using f8x8_t  = f8x8_fnuz_t;
using f8x16_t = f8x16_fnuz_t;
using f8x32_t = f8x32_fnuz_t;
using f8x64_t = f8x64_fnuz_t;

using f8x3_t  = f8x3_fnuz_t;
using f8x5_t  = f8x5_fnuz_t;
using f8x6_t  = f8x6_fnuz_t;
using f8x9_t  = f8x9_fnuz_t;
using f8x10_t = f8x10_fnuz_t;
using f8x12_t = f8x12_fnuz_t;
using f8x18_t = f8x18_fnuz_t;
using f8x20_t = f8x20_fnuz_t;
using f8x24_t = f8x24_fnuz_t;
using f8x36_t = f8x36_fnuz_t;
using f8x72_t = f8x72_fnuz_t;

// bf8
using bf8x2_t  = bf8x2_fnuz_t;
using bf8x4_t  = bf8x4_fnuz_t;
using bf8x8_t  = bf8x8_fnuz_t;
using bf8x16_t = bf8x16_fnuz_t;
using bf8x32_t = bf8x32_fnuz_t;
using bf8x64_t = bf8x64_fnuz_t;

using bf8x3_t  = bf8x3_fnuz_t;
using bf8x5_t  = bf8x5_fnuz_t;
using bf8x6_t  = bf8x6_fnuz_t;
using bf8x9_t  = bf8x9_fnuz_t;
using bf8x10_t = bf8x10_fnuz_t;
using bf8x12_t = bf8x12_fnuz_t;
using bf8x18_t = bf8x18_fnuz_t;
using bf8x20_t = bf8x20_fnuz_t;
using bf8x24_t = bf8x24_fnuz_t;
using bf8x36_t = bf8x36_fnuz_t;
using bf8x72_t = bf8x72_fnuz_t;
#endif

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
using int4x8_t  = typename vector_type<int32_t, 1>::type;
using int4x16_t = typename vector_type<int32_t, 2>::type;
using int4x32_t = typename vector_type<int32_t, 4>::type;
using int4x64_t = typename vector_type<int32_t, 8>::type;

using int4x24_t  = typename vector_type<int32_t, 3>::type;
using int4x48_t  = typename vector_type<int32_t, 6>::type;
using int4x72_t  = typename vector_type<int32_t, 9>::type;
using int4x144_t = typename vector_type<int32_t, 18>::type;
using int4x40_t  = typename vector_type<int32_t, 5>::type;

using uint4x8_t  = typename vector_type<uint32_t, 1>::type;
using uint4x16_t = typename vector_type<uint32_t, 2>::type;
using uint4x32_t = typename vector_type<uint32_t, 4>::type;
using uint4x64_t = typename vector_type<uint32_t, 8>::type;
#endif

template <MTX_FMT MX_TYPE = MTX_FMT::MTX_FMT_DEFAULT>
struct MxType_t
{
    using type_t = unsigned _BitInt(8);
    using vec_t  = int32x8_t; // used for wmma
    MxType_t()   = default;
    MxType_t(type_t data) { m_data = data; }
    MxType_t(f8_ocp_t data) { m_data = data.data; }
    MxType_t(bf8_ocp_t data) { m_data = data.data; }
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
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 2)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            in_packed.push_back((val0 & 0xf) | (val1 << 4));
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
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 4)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            type_t val2 = in[i + 2];
            type_t val3 = in[i + 3];
            in_packed.push_back((val0 & 0x3f) | ((val1 & 0x3) << 6));
            in_packed.push_back((val1 & 0x3C) >> 2 | ((val2 & 0xf) << 4));
            in_packed.push_back((val2 & 0x3C) >> 4 | ((val3 & 0x3f) << 2));
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
        std::vector<type_t> in_packed;
        std::size_t in_vec_size = in.size();
        in_packed.reserve(in_vec_size);
        for(std::size_t i = 0; i < in_vec_size; i += 4)
        {
            type_t val0 = in[i];
            type_t val1 = in[i + 1];
            type_t val2 = in[i + 2];
            type_t val3 = in[i + 3];
            in_packed.push_back((val0 & 0x3f) | ((val1 & 0x3) << 6));
            in_packed.push_back((val1 & 0x3C) >> 2 | ((val2 & 0xf) << 4));
            in_packed.push_back((val2 & 0x3C) >> 4 | ((val3 & 0x3f) << 2));
        }
        return in_packed;
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

template <typename T, typename = void>
struct DataPerVGPR;

template <typename T>
struct DataPerVGPR<T, enable_if_t<is_same_v<T, half_t> || is_same_v<T, bhalf_t>>>
{
    static constexpr index_t value = 2;
};

template <typename T>
struct DataPerVGPR<T, enable_if_t<is_same_v<T, f8_t> || is_same_v<T, bf8_t>>>
{
    static constexpr index_t value = 4;
};

// f4
using f4x2_t  = typename vector_type<f4x2_pk_t, 1>::type;
using f4x4_t  = typename vector_type<f4x2_pk_t, 2>::type;
using f4x8_t  = typename vector_type<f4x2_pk_t, 4>::type;
using f4x16_t = typename vector_type<f4x2_pk_t, 8>::type;
using f4x32_t = typename vector_type<f4x2_pk_t, 16>::type;
using f4x64_t = typename vector_type<f4x2_pk_t, 32>::type;

// f6
using f6x16_t = typename vector_type<f6x16_pk_t, 1>::type;
using f6x32_t = typename vector_type<f6x32_pk_t, 1>::type;

// bf6
using bf6x16_t = typename vector_type<bf6x16_pk_t, 1>::type;
using bf6x32_t = typename vector_type<bf6x32_pk_t, 1>::type;

// pack int4
using pk_i4x2_t = typename vector_type<pk_i4_t, 2>::type;
using pk_i4x4_t = typename vector_type<pk_i4_t, 4>::type;
using pk_i4x8_t = typename vector_type<pk_i4_t, 8>::type;

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<int32_t>
{
    __host__ __device__ static constexpr int32_t Lowest() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Min() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Max() noexcept { return 2147483647; }

    __host__ __device__ static constexpr int32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int32_t QuietNaN() { return 0; }
};
template <>
struct NumericLimits<int16_t>
{
    __host__ __device__ static constexpr int16_t Lowest() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Min() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Max() noexcept { return 32767; }

    __host__ __device__ static constexpr int16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<int8_t>
{
    __host__ __device__ static constexpr int8_t Lowest() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Min() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Max() noexcept { return 127; }

    __host__ __device__ static constexpr int8_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int8_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint32_t>
{
    __host__ __device__ static constexpr uint32_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Max() noexcept { return 4294967295U; }

    __host__ __device__ static constexpr uint32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint16_t>
{
    __host__ __device__ static constexpr uint16_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Max() noexcept { return 65535U; }

    __host__ __device__ static constexpr uint16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<float>
{
    static constexpr unsigned int binary_min    = 0x00800000;
    static constexpr unsigned int binary_max    = 0x7F7FFFFF;
    static constexpr unsigned int binary_lowest = 0xFF7FFFFF;
    static constexpr unsigned int binary_qnan   = 0xFFC00001;
    static constexpr unsigned int binary_inf    = 0x7F8000000;

    __host__ __device__ static constexpr float Min() { return bit_cast<float>(binary_min); }

    __host__ __device__ static constexpr float Max() { return bit_cast<float>(binary_max); }

    __host__ __device__ static constexpr float Lowest() { return bit_cast<float>(binary_lowest); }

    __host__ __device__ static constexpr float QuietNaN() { return bit_cast<float>(binary_qnan); }

    __host__ __device__ static constexpr float Infinity() { return bit_cast<float>(binary_inf); }
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
struct NumericLimits<f8_fnuz_t>
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

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
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

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<f4_t>
{
    static constexpr uint8_t binary_min_normal    = 0x2; // 0b0010
    static constexpr uint8_t binary_max_normal    = 0x7; // 0b0111
    static constexpr uint8_t binary_lowest_normal = 0xF; // 0b1111
    static constexpr uint8_t binary_min_subnorm   = 0x1; // 0b0001
    static constexpr uint8_t binary_max_subnorm   = 0x1; // 0b0001

    static constexpr float data_max_normal_number    = 6;
    static constexpr float data_min_subnormal_number = 0.5;

    __host__ __device__ static constexpr f4_t Min() { return f4_t(binary_min_normal); }
    __host__ __device__ static constexpr f4_t Max() { return f4_t(binary_max_normal); }
    __host__ __device__ static constexpr f4_t Lowest() { return f4_t(binary_lowest_normal); }
    __host__ __device__ static constexpr f4_t MinSubnorm() { return f4_t(binary_min_subnorm); }
    __host__ __device__ static constexpr f4_t MaxSubnorm() { return f4_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<f6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x07; // 0b000111

    static constexpr float data_max_normal_number    = 7.5;
    static constexpr float data_min_subnormal_number = 0.125;

    __host__ __device__ static constexpr f6_t Min() { return f6_t(binary_min_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Max() { return f6_t(binary_max_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Lowest()
    {
        return f6_t(binary_lowest_normal & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MinSubnorm()
    {
        return f6_t(binary_min_subnorm & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MaxSubnorm()
    {
        return f6_t(binary_max_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<bf6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x03; // 0b000011

    static constexpr float data_max_normal_number    = 28;
    static constexpr float data_min_subnormal_number = 0.0625;

    __host__ __device__ static constexpr bf6_t Min() { return bf6_t(binary_min_normal); }
    __host__ __device__ static constexpr bf6_t Max() { return bf6_t(binary_max_normal); }
    __host__ __device__ static constexpr bf6_t Lowest() { return bf6_t(binary_lowest_normal); }
    __host__ __device__ static constexpr bf6_t MinSubnorm() { return bf6_t(binary_min_subnorm); }
    __host__ __device__ static constexpr bf6_t MaxSubnorm() { return bf6_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<e8m0_bexp_t>
{
    static constexpr e8m0_bexp_t binary_min  = 0x00; // 0b00000000
    static constexpr e8m0_bexp_t binary_max  = 0xFE; // 0b11111110
    static constexpr e8m0_bexp_t binary_qnan = 0xFF; // 0b11111111
    static constexpr e8m0_bexp_t binary_1    = 0x7F; // 0b01111111
    static constexpr e8m0_bexp_t binary_2    = 0x80; // 0b10000000
    static constexpr e8m0_bexp_t binary_3    = 0x82; // 0b10000010
    static constexpr e8m0_bexp_t binary_135  = 0x87; // 0b10000111
    static constexpr e8m0_bexp_t binary_142  = 0x8E; // 0b10001110

    __host__ __device__ static constexpr e8m0_bexp_t Min() { return e8m0_bexp_t(binary_min); }
    __host__ __device__ static constexpr e8m0_bexp_t Max() { return e8m0_bexp_t(binary_max); }
    __host__ __device__ static constexpr e8m0_bexp_t QuietNaN() { return e8m0_bexp_t(binary_qnan); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_1() { return e8m0_bexp_t(binary_1); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_2() { return e8m0_bexp_t(binary_2); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_3() { return e8m0_bexp_t(binary_3); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_135()
    {
        return e8m0_bexp_t(binary_135);
    }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_142()
    {
        return e8m0_bexp_t(binary_142);
    }
};
#else
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
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<f4_t>
{
    static constexpr uint8_t binary_min_normal    = 0x2; // 0b0010
    static constexpr uint8_t binary_max_normal    = 0x7; // 0b0111
    static constexpr uint8_t binary_lowest_normal = 0xF; // 0b1111
    static constexpr uint8_t binary_min_subnorm   = 0x1; // 0b0001
    static constexpr uint8_t binary_max_subnorm   = 0x1; // 0b0001

    static constexpr float data_max_normal_number    = 6;
    static constexpr float data_min_subnormal_number = 0.5;

    __host__ __device__ static constexpr f4_t Min() { return f4_t(binary_min_normal); }
    __host__ __device__ static constexpr f4_t Max() { return f4_t(binary_max_normal); }
    __host__ __device__ static constexpr f4_t Lowest() { return f4_t(binary_lowest_normal); }
    __host__ __device__ static constexpr f4_t MinSubnorm() { return f4_t(binary_min_subnorm); }
    __host__ __device__ static constexpr f4_t MaxSubnorm() { return f4_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<f6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x07; // 0b000111

    static constexpr float data_max_normal_number    = 7.5;
    static constexpr float data_min_subnormal_number = 0.125;

    __host__ __device__ static constexpr f6_t Min() { return f6_t(binary_min_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Max() { return f6_t(binary_max_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Lowest()
    {
        return f6_t(binary_lowest_normal & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MinSubnorm()
    {
        return f6_t(binary_min_subnorm & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MaxSubnorm()
    {
        return f6_t(binary_max_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<bf6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x03; // 0b000011

    static constexpr float data_max_normal_number    = 28;
    static constexpr float data_min_subnormal_number = 0.0625;

    __host__ __device__ static constexpr bf6_t Min() { return bf6_t(binary_min_normal); }
    __host__ __device__ static constexpr bf6_t Max() { return bf6_t(binary_max_normal); }
    __host__ __device__ static constexpr bf6_t Lowest() { return bf6_t(binary_lowest_normal); }
    __host__ __device__ static constexpr bf6_t MinSubnorm() { return bf6_t(binary_min_subnorm); }
    __host__ __device__ static constexpr bf6_t MaxSubnorm() { return bf6_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<e8m0_bexp_t>
{
    static constexpr e8m0_bexp_t binary_min  = 0x00; // 0b00000000
    static constexpr e8m0_bexp_t binary_max  = 0xFE; // 0b11111110
    static constexpr e8m0_bexp_t binary_qnan = 0xFF; // 0b11111111
    static constexpr e8m0_bexp_t binary_1    = 0x7F; // 0b01111111
    static constexpr e8m0_bexp_t binary_2    = 0x80; // 0b10000000
    static constexpr e8m0_bexp_t binary_3    = 0x82; // 0b10000010
    static constexpr e8m0_bexp_t binary_135  = 0x87; // 0b10000111
    static constexpr e8m0_bexp_t binary_142  = 0x8E; // 0b10001110

    __host__ __device__ static constexpr e8m0_bexp_t Min() { return e8m0_bexp_t(binary_min); }
    __host__ __device__ static constexpr e8m0_bexp_t Max() { return e8m0_bexp_t(binary_max); }
    __host__ __device__ static constexpr e8m0_bexp_t QuietNaN() { return e8m0_bexp_t(binary_qnan); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_1() { return e8m0_bexp_t(binary_1); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_2() { return e8m0_bexp_t(binary_2); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_3() { return e8m0_bexp_t(binary_3); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_135()
    {
        return e8m0_bexp_t(binary_135);
    }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_142()
    {
        return e8m0_bexp_t(binary_142);
    }
};
#endif

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
    static constexpr bool has_inf       = true;
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
    static constexpr bool has_inf       = true;
    using bitwise_type                  = uint16_t;
};

template <>
struct NumericUtils<bhalf_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 7;
    static constexpr int bias = 128; // negative zero nan mode
    // static constexpr int bias = 127; // ieee mode
};

template <>
struct NumericUtils<f8_fnuz_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 8; // negative zero nan mode
    // static constexpr int bias = 7; // ieee mode
    static constexpr bool has_inf = false;
};

template <>
struct NumericUtils<bf8_fnuz_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 16; // negative zero nan mode
    // static constexpr int bias = 15; // ieee mode
    static constexpr bool has_inf = false;
};
template <>
struct NumericUtils<f8_ocp_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 7;
};

template <>
struct NumericUtils<bf8_ocp_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 15;
};

template <>
struct NumericUtils<f4_t>
{
    static constexpr int exp           = 2;
    static constexpr int mant          = 1;
    static constexpr int bias          = 1;
    static constexpr uint32_t sr_shift = 10;

    static constexpr int unbiased_exp_min = 0;
    static constexpr int unbiased_exp_max = 2;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 3;

    static constexpr uint8_t positive_zero_mask = 0b0000;
    static constexpr uint8_t negative_zero_mask = 0b1000;

    static constexpr uint8_t one_mask      = 0b0010;
    static constexpr uint8_t set_sign_mask = 0b0111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b0111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b1111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b0001;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b1001;

    static constexpr bool has_inf = false;

    using bitwise_type = uint8_t;
};

template <>
struct NumericUtils<f6_t>
{
    static constexpr int exp           = 2;
    static constexpr int mant          = 3;
    static constexpr int bias          = 1;
    static constexpr uint32_t sr_shift = 12;

    static constexpr int unbiased_exp_min = 0;
    static constexpr int unbiased_exp_max = 2;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 3;

    static constexpr uint8_t positive_zero_mask = 0b000000;
    static constexpr uint8_t negative_zero_mask = 0b100000;

    static constexpr uint8_t set_sign_mask = 0b011111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b011111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b111111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b000111;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b100111;

    static constexpr bool has_inf  = false;
    static constexpr bool has_nan  = false;
    static constexpr bool has_zero = true;

    using bitwise_type = uint8_t;
};

template <>
struct NumericUtils<bf6_t>
{
    static constexpr int exp           = 3;
    static constexpr int mant          = 2;
    static constexpr int bias          = 3;
    static constexpr uint32_t sr_shift = 11;

    static constexpr int unbiased_exp_min = -2;
    static constexpr int unbiased_exp_max = 4;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 7;

    static constexpr uint8_t positive_zero_mask = 0b000000;
    static constexpr uint8_t negative_zero_mask = 0b100000;

    static constexpr uint8_t set_sign_mask = 0b011111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b011111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b111111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b000011;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b100011;

    static constexpr bool has_inf  = false;
    static constexpr bool has_nan  = false;
    static constexpr bool has_zero = true;

    using bitwise_type = uint8_t;
};

template <>
struct NumericUtils<e8m0_bexp_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 0;
    static constexpr int bias = 127;

    static constexpr int unbiased_exp_min = -127;
    static constexpr int unbiased_exp_max = 127;
    static constexpr int biased_exp_min   = 0;
    static constexpr int biased_exp_max   = 254;

    using bitwise_type = uint8_t;
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
