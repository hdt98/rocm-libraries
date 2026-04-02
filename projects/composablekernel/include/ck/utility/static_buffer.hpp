// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "statically_indexed_array.hpp"
#include <utility> // for std::index_sequence

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck {

// static buffer for scalar
template <AddressSpaceEnum AddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue> // TODO remove this bool, no longer needed
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    template <typename... Ys>
    __host__ __device__ constexpr StaticBuffer& operator=(const Tuple<Ys...>& y)
    {
        static_assert(base::Size() == sizeof...(Ys), "wrong! size not the same");
        StaticBuffer& x = *this;
        static_for<0, base::Size(), 1>{}([&](auto i) { x(i) = y[i]; });
        return x;
    }

    __host__ __device__ constexpr StaticBuffer& operator=(const T& y)
    {
        StaticBuffer& x = *this;
        static_for<0, base::Size(), 1>{}([&](auto i) { x(i) = y; });
        return x;
    }

    __host__ __device__ static constexpr AddressSpaceEnum GetAddressSpace() { return AddressSpace; }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I> i) const
    {
        return base::operator[](i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I> i)
    {
        return base::operator()(i);
    }

    // ========================================================================
    // RUNTIME INDEXING (NEW API - backward compatible)
    // ========================================================================
    // NOTE: Using named methods (At/at) instead of operator[] to avoid
    // ambiguity with the existing operator[](Number<I>) since Number<I>
    // implicitly converts to index_t.

    // Runtime read access - generates identical assembly when index is constexpr
    // Implementation: Recursive constexpr dispatch (optimized away by compiler)
    __host__ __device__ __forceinline__ const T& At(index_t i) const
    {
        return runtime_at_impl(*this, i, Number<0>{}, Number<N>{});
    }

    // Runtime write access - generates identical assembly when index is constexpr
    __host__ __device__ __forceinline__ T& At(index_t i)
    {
        return runtime_at_impl(*this, i, Number<0>{}, Number<N>{});
    }

private:
    // Helper: Binary search dispatch for runtime indexing
    // When i is constexpr, compiler optimizes to direct access
    template <typename Self, index_t Begin, index_t End>
    __host__ __device__ static constexpr auto& runtime_at_impl(
        Self& self, index_t i, Number<Begin>, Number<End>)
    {
        if constexpr (End - Begin == 1) {
            // Base case: single element
            return self(Number<Begin>{});
        } else {
            // Recursive case: binary split
            constexpr index_t Mid = (Begin + End) / 2;
            if (i < Mid) {
                return runtime_at_impl(self, i, Number<Begin>{}, Number<Mid>{});
            } else {
                return runtime_at_impl(self, i, Number<Mid>{}, Number<End>{});
            }
        }
    }

public:
    // ========================================================================

    __host__ __device__ void Set(T x)
    {
        static_for<0, N, 1>{}([&](auto i) { operator()(i) = T{x}; });
    }

    __host__ __device__ void Clear() { Set(T{0}); }
};

// ====================================================================================
// OffsetTable: Compile-time offset computation for runtime indexing
// ====================================================================================

template <index_t N, typename OffsetFunc>
struct OffsetTable
{
    index_t data[N];

    // Compute all offsets at compile time using parameter pack expansion
    template <index_t... Is>
    __host__ __device__ constexpr OffsetTable(OffsetFunc f, std::index_sequence<Is...>)
        : data{f(Number<Is>{})...}
    {
    }

    __host__ __device__ constexpr index_t operator[](index_t i) const { return data[i]; }
};

template <index_t N, typename OffsetFunc>
__host__ __device__ constexpr auto make_offset_table(OffsetFunc f)
{
    return OffsetTable<N, OffsetFunc>(f, std::make_index_sequence<N>{});
}

// ====================================================================================

// static buffer for vector
template <AddressSpaceEnum AddressSpace,
          typename S,
          index_t NumOfVector,
          index_t ScalarPerVector,
          bool InvalidElementUseNumericalZeroValue, // TODO remove this bool, no longer needed,
          typename enable_if<is_scalar_type<S>::value, bool>::type = false>
struct StaticBufferTupleOfVector
    : public StaticallyIndexedArray<vector_type<S, ScalarPerVector>, NumOfVector>
{
    using V    = typename vector_type<S, ScalarPerVector>::type;
    using base = StaticallyIndexedArray<vector_type<S, ScalarPerVector>, NumOfVector>;

    static constexpr auto s_per_v   = Number<ScalarPerVector>{};
    static constexpr auto num_of_v_ = Number<NumOfVector>{};
    static constexpr auto s_per_buf = s_per_v * num_of_v_;

    __host__ __device__ constexpr StaticBufferTupleOfVector() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum GetAddressSpace() { return AddressSpace; }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }

    __host__ __device__ static constexpr index_t Size() { return s_per_buf; };

    // Get S
    // i is offset of S
    template <index_t I>
    __host__ __device__ constexpr const S& operator[](Number<I> i) const
    {
        constexpr auto i_v = i / s_per_v;
        constexpr auto i_s = i % s_per_v;

        return base::operator[](i_v).template AsType<S>()[i_s];
    }

    // Set S
    // i is offset of S
    template <index_t I>
    __host__ __device__ constexpr S& operator()(Number<I> i) [[clang::lifetimebound]]
    {
        constexpr auto i_v = i / s_per_v;
        constexpr auto i_s = i % s_per_v;

        return base::operator()(i_v).template AsType<S>()(i_s);
    }

    // Get X
    // i is offset of S, not X. i should be aligned to X
    template <typename X,
              index_t I,
              typename enable_if<has_same_scalar_type<S, X>::value || !is_native_type<S>(),
                                 bool>::type = false>
    __host__ __device__ constexpr auto GetAsType(Number<I> i) const
    {
        constexpr auto s_per_x = Number<scalar_type<remove_cvref_t<X>>::vector_size>{};

        static_assert(s_per_v % s_per_x == 0, "wrong! V must  one or multiple X");
        static_assert(i % s_per_x == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;
        constexpr auto i_x = (i % s_per_v) / s_per_x;

        return base::operator[](i_v).template AsType<X>()[i_x];
    }

    // Set X
    // i is offset of S, not X. i should be aligned to X
    template <typename X,
              index_t I,
              typename enable_if<has_same_scalar_type<S, X>::value || !is_native_type<S>(),
                                 bool>::type = false>
    __host__ __device__ constexpr void SetAsType(Number<I> i, X x)
    {
        constexpr auto s_per_x = Number<scalar_type<remove_cvref_t<X>>::vector_size>{};

        static_assert(s_per_v % s_per_x == 0, "wrong! V must contain one or multiple X");
        static_assert(i % s_per_x == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;
        constexpr auto i_x = (i % s_per_v) / s_per_x;

        base::operator()(i_v).template AsType<X>()(i_x) = x;
    }

    // Get read access to vector_type V
    // i is offset of S, not V. i should be aligned to V
    template <index_t I>
    __host__ __device__ constexpr const auto& GetVectorTypeReference(Number<I> i) const
    {
        static_assert(i % s_per_v == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;

        return base::operator[](i_v);
    }

    // Get write access to vector_type V
    // i is offset of S, not V. i should be aligned to V
    template <index_t I>
    __host__ __device__ constexpr auto& GetVectorTypeReference(Number<I> i)
    {
        static_assert(i % s_per_v == 0, "wrong!");

        constexpr auto i_v = i / s_per_v;

        return base::operator()(i_v);
    }

    __host__ __device__ void Clear()
    {
        constexpr index_t NumScalars = NumOfVector * ScalarPerVector;

        static_for<0, NumScalars, 1>{}([&](auto i) { SetAsType(i, S{0}); });
    }
};

template <AddressSpaceEnum AddressSpace, typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<AddressSpace, T, N, true>{};
}

template <AddressSpaceEnum AddressSpace, typename T, long_index_t N>
__host__ __device__ constexpr auto make_static_buffer(LongNumber<N>)
{
    return StaticBuffer<AddressSpace, T, N, true>{};
}

} // namespace ck
#pragma clang diagnostic pop
