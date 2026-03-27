// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

namespace detail {

// Lookup table to store precomputed indices for all 1D access values
template <index_t NumAccesses, index_t nDim>
struct sfc_index_lookup_table
{
    multi_index<nDim> data[NumAccesses > 0 ? NumAccesses : 1];

    CK_TILE_HOST_DEVICE constexpr const multi_index<nDim>& operator[](index_t i) const
    {
        return data[i];
    }
};

// Compute a single index given 1D access index - used during table construction
template <index_t nDim, bool SnakeCurved, typename Strides, typename OrderedAccessLengths>
CK_TILE_HOST_DEVICE constexpr auto
sfc_compute_single_index(index_t idx_1d, Strides strides, OrderedAccessLengths ordered_lengths)
{
    // Step 1: Convert 1D index to N-D ordered coordinates using strides
    multi_index<nDim> ordered_access_idx;
    index_t remaining = idx_1d;
    static_for<0, nDim, 1>{}([&](auto i) {
        ordered_access_idx(i) = remaining / strides[i];
        remaining             = remaining % strides[i];
    });

    // Step 2: Compute forward_sweep - whether each dimension is in forward direction
    statically_indexed_array<bool, nDim> forward_sweep;
    forward_sweep(number<0>{}) = true;
    index_t cumulative         = ordered_access_idx[number<0>{}];
    static_for<1, nDim, 1>{}([&](auto i) {
        forward_sweep(i) = cumulative % 2 == 0;
        cumulative       = cumulative * ordered_lengths[i] + ordered_access_idx[i];
    });

    // Step 3: Apply snake curve transformation
    multi_index<nDim> ordered_idx;
    static_for<0, nDim, 1>{}([&](auto i) {
        if constexpr(!SnakeCurved)
        {
            ordered_idx(i) = ordered_access_idx[i];
        }
        else
        {
            ordered_idx(i) = forward_sweep[i] ? ordered_access_idx[i]
                                              : ordered_lengths[i] - 1 - ordered_access_idx[i];
        }
    });

    return ordered_idx;
}

// Precompute all indices into a lookup table using a constexpr loop
template <index_t NumAccesses,
          index_t nDim,
          bool SnakeCurved,
          typename Strides,
          typename OrderedAccessLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess>
CK_TILE_HOST_DEVICE constexpr auto sfc_compute_all_indices(Strides strides,
                                                           OrderedAccessLengths ordered_lengths,
                                                           DimAccessOrder dim_order,
                                                           ScalarsPerAccess scalars)
{
    sfc_index_lookup_table<NumAccesses, nDim> table{};

    for(index_t idx_1d = 0; idx_1d < NumAccesses; ++idx_1d)
    {
        auto ordered_idx =
            sfc_compute_single_index<nDim, SnakeCurved>(idx_1d, strides, ordered_lengths);

        // Reorder and scale
        auto reordered     = container_reorder_given_old2new(ordered_idx, dim_order);
        table.data[idx_1d] = reordered * scalars;
    }

    return table;
}

} // namespace detail

template <typename TensorLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess,
          bool SnakeCurved = true> // # of scalars per access in each dimension
struct space_filling_curve
{
    static constexpr index_t TensorSize =
        reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{});
    static_assert(0 < TensorSize,
                  "space_filling_curve should be used to access a non-empty tensor");

    static constexpr index_t nDim = TensorLengths::size();

    using Index = multi_index<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, multiplies<>{}, number<1>{});

    static constexpr auto access_lengths   = TensorLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    // Precompute access strides at class level
    static constexpr auto access_strides =
        container_reverse_exclusive_scan(ordered_access_lengths, multiplies<>{}, number<1>{});

    // Number of access indices
    static constexpr index_t NumAccesses = TensorSize / ScalarPerVector;

    // Precompute ALL indices into a lookup table - computed once at class instantiation
    static constexpr auto index_table =
        detail::sfc_compute_all_indices<NumAccesses, nDim, SnakeCurved>(
            access_strides, ordered_access_lengths, dim_access_order, ScalarsPerAccess{});

    static constexpr auto to_index_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(ordered_access_lengths)),
        make_tuple(typename arithmetic_sequence_gen<0, nDim, 1>::type{}),
        make_tuple(sequence<0>{}));

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_access()
    {
        static_assert(TensorLengths::size() == ScalarsPerAccess::size());
        static_assert(TensorLengths{} % ScalarsPerAccess{} ==
                      typename uniform_sequence_gen<TensorLengths::size(), 0>::type{});

        return reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{}) / ScalarPerVector;
    }

    template <index_t AccessIdx1dHead, index_t AccessIdx1dTail>
    static CK_TILE_HOST_DEVICE constexpr auto get_step_between(number<AccessIdx1dHead>,
                                                               number<AccessIdx1dTail>)
    {
        static_assert(AccessIdx1dHead >= 0 && AccessIdx1dHead < get_num_of_access(),
                      "1D index out of range");
        static_assert(AccessIdx1dTail >= 0 && AccessIdx1dTail < get_num_of_access(),
                      "1D index out of range");

        constexpr auto idx_head = get_index(number<AccessIdx1dHead>{});
        constexpr auto idx_tail = get_index(number<AccessIdx1dTail>{});
        return idx_tail - idx_head;
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_forward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d < get_num_of_access(), "1D index should be larger than 0");
        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d + 1>{});
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_backward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d > 0, "1D index should be larger than 0");

        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d - 1>{});
    }

    // Simple lookup from precomputed table - O(1) with no template instantiation overhead
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr Index _get_index(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d >= 0 && AccessIdx1d < NumAccesses, "Index out of bounds");
        return index_table[AccessIdx1d];
    }

    // FIXME: return tuple of number<>, which is compile time only variable
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_index(number<AccessIdx1d>)
    {
        constexpr auto idx = _get_index(number<AccessIdx1d>{});

        return generate_tuple([&](auto i) { return number<idx[i]>{}; }, number<nDim>{});
    }
};

} // namespace ck_tile
