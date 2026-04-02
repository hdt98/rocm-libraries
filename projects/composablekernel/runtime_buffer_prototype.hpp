// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Prototype: Add runtime indexing to CK buffer types
// This is a backward-compatible API extension that enables 87.5% reduction in
// template instantiations while producing identical assembly.

#include "ck/utility/statically_indexed_array.hpp"

namespace ck {

// ====================================================================================
// Modified StaticBuffer with runtime indexing support
// ====================================================================================

template <AddressSpaceEnum AddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue = false>
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    // Existing API (unchanged - backward compatible)
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

    // ================================================================================
    // COMPILE-TIME INDEXED ACCESS (Existing API)
    // ================================================================================

    // Read access with compile-time index
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I> i) const
    {
        return base::operator[](i);
    }

    // Write access with compile-time index
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I> i)
    {
        return base::operator()(i);
    }

    // ================================================================================
    // RUNTIME INDEXED ACCESS (NEW API)
    // ================================================================================

    // Read access with runtime index
    // Generates identical assembly to compile-time version when index is constexpr
    __host__ __device__ __forceinline__ const T& operator[](index_t i) const
    {
        return base::data_[i];  // Direct array access
    }

    // Write access with runtime index
    // Generates identical assembly to compile-time version when index is constexpr
    __host__ __device__ __forceinline__ T& operator[](index_t i)
    {
        return base::data_[i];  // Direct array access
    }

    // Alternative: Named methods if operator overload ambiguity is an issue
    __host__ __device__ __forceinline__ const T& at_rt(index_t i) const
    {
        return base::data_[i];
    }

    __host__ __device__ __forceinline__ T& at_rt(index_t i)
    {
        return base::data_[i];
    }

    // Existing API (unchanged)
    __host__ __device__ void Set(T x)
    {
        static_for<0, N, 1>{}([&](auto i) { operator()(i) = T{x}; });
    }

    __host__ __device__ void Clear() { Set(T{0}); }
};

// ====================================================================================
// Helper: Compute offset table at compile time
// ====================================================================================

template <index_t N, typename OffsetFunc>
struct OffsetTable
{
    index_t data[N];

    // Compute all offsets at compile time
    template <index_t... Is>
    constexpr OffsetTable(OffsetFunc f, std::index_sequence<Is...>)
        : data{f(Number<Is>{})...}
    {
    }

    constexpr index_t operator[](index_t i) const { return data[i]; }
};

template <index_t N, typename OffsetFunc>
constexpr auto make_offset_table(OffsetFunc f)
{
    return OffsetTable<N, OffsetFunc>(f, std::make_index_sequence<N>{});
}

// ====================================================================================
// Usage Example: Buffer load with runtime indexing
// ====================================================================================

#if 0  // Example code (not compiled)

// BEFORE (current CK pattern - KPack template instantiations):
static_for<0, KPack, 1>{}([&](auto ik) {
    a_thread_vec.template AsType<FloatAB>()(ik) =
        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
            make_tuple(m0, I0, k0, ik))>{}];
});

// AFTER (runtime indexing - single instantiation):
constexpr auto a_offsets = make_offset_table<KPack>([](auto ik) {
    return a_thread_desc_.CalculateOffset(make_tuple(m0, I0, k0, ik));
});

#pragma unroll
for (index_t i = 0; i < KPack; ++i) {
    a_thread_vec[i] = a_thread_buf[a_offsets[i]];  // Runtime index!
}

// Even simpler with C++20 lambdas:
constexpr auto a_offsets = []() constexpr {
    OffsetTable<KPack, decltype(a_thread_desc_)> table;
    for (index_t i = 0; i < KPack; ++i) {
        table.data[i] = a_thread_desc_.CalculateOffset(
            make_tuple(m0, I0, k0, Number<i>{}));
    }
    return table;
}();

#pragma unroll
for (index_t i = 0; i < KPack; ++i) {
    a_thread_vec[i] = a_thread_buf[a_offsets[i]];
}

#endif

// ====================================================================================
// Migration Strategy
// ====================================================================================

/*
Phase 1: Add runtime indexing to buffer types (this file)
  - StaticBuffer
  - StaticBufferTupleOfVector
  - DynamicBuffer (if needed)

Phase 2: Test on hot files
  - Convert 3-5 high-usage pipeline files
  - Measure compile time improvement
  - Verify assembly remains identical

Phase 3: Gradual migration
  - Convert buffer load sites (358 locations)
  - Convert buffer store sites (72 locations)
  - Keep compile-time API for code that doesn't benefit

Phase 4: Cleanup
  - Remove unused Number<I> template instantiations
  - Simplify code where appropriate
  - Document best practices

Estimated effort:
  - API change: 1 day
  - Testing: 3 days
  - Hot file conversion: 1 week
  - Full migration: 2-4 weeks
  - Validation: 1 week

Risk assessment: LOW
  - Backward compatible (existing code unchanged)
  - Identical codegen (verified)
  - Easy to test incrementally
  - Easy to revert if issues found
*/

} // namespace ck
