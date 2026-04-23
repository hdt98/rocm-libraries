// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Definitions for the public traversal API and detail fold-expression
///        helpers declared in `transform_graph.hpp`.
///
/// Auto-included from the bottom of `transform_graph.hpp`. Do not include
/// directly.
// IWYU pragma: private, include "ck_tile/experimental/core/transform/transform_graph.hpp"
#pragma once

#include "ck_tile/experimental/core/transform/transform_graph.hpp"

namespace ck_tile::core::transform {
namespace detail {

template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void applySingleTransform(static_array<index_t, G.num_slots>& slots)
{
    constexpr index_t t      = G.num_transforms - 1 - I; // pop order (top of stack first)
    constexpr auto transform = G.transforms[t];

    static_array<index_t, MAX_TENSOR_DIMS> t_input{};
    for(index_t d = 0; d < transform.ndim_input; ++d)
    {
        t_input[d] = slots[G.t_input_slots[t][d]];
    }

    static_array<index_t, MAX_TENSOR_DIMS> t_output{};
    TransformImpl<transform.type>::mapIndices(transform, t_output.elems, t_input.elems);

    for(index_t d = 0; d < transform.ndim_output; ++d)
    {
        slots[G.t_output_slots[t][d]] = t_output[d];
    }
}

template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void applyAllTransforms(static_array<index_t, G.num_slots>& slots,
                                                      sequence<Is...>)
{
    (applySingleTransform<G, Is>(slots), ...);
}

template <TransformGraph G, index_t T>
CK_TILE_HOST_DEVICE constexpr index_t tryInputLengthAt(index_t slot)
{
    constexpr auto transform = G.transforms[T];
    for(index_t d = 0; d < transform.ndim_input; ++d)
    {
        if(G.t_input_slots[T][d] == slot)
        {
            return TransformImpl<transform.type>::inputLength(transform, d);
        }
    }
    return -1; // not found at this transform
}

template <TransformGraph G, index_t... Ts>
CK_TILE_HOST_DEVICE constexpr index_t inputDimLengthDispatch(index_t slot, sequence<Ts...>)
{
    index_t result = -1;
    auto check     = [&](index_t candidate) {
        if(result == -1 && candidate != -1)
        {
            result = candidate;
        }
    };
    (check(tryInputLengthAt<G, G.num_transforms - 1 - Ts>(slot)), ...);
    return result;
}

template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void
reverseApplySingleTransform(static_array<index_t, G.num_slots>& slots)
{
    constexpr index_t t      = I; // forward array order (base → top)
    constexpr auto transform = G.transforms[t];

    static_array<index_t, MAX_TENSOR_DIMS> t_output{};
    for(index_t d = 0; d < transform.ndim_output; ++d)
    {
        t_output[d] = slots[G.t_output_slots[t][d]];
    }

    static_array<index_t, MAX_TENSOR_DIMS> t_input{};
    TransformImpl<transform.type>::reverseMapIndices(transform, t_input.elems, t_output.elems);

    for(index_t d = 0; d < transform.ndim_input; ++d)
    {
        slots[G.t_input_slots[t][d]] = t_input[d];
    }
}

template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void
reverseApplyAllTransforms(static_array<index_t, G.num_slots>& slots, sequence<Is...>)
{
    (reverseApplySingleTransform<G, Is>(slots), ...);
}

} // namespace detail

template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void map(index_t* output, const index_t* input)
{
    static_array<index_t, G.num_slots> slots{};

    for(index_t i = 0; i < G.ndim_input; ++i)
    {
        slots[G.input_slots[i]] = input[i];
    }

    detail::applyAllTransforms<G>(slots, make_index_sequence<G.num_transforms>{});

    for(index_t i = 0; i < G.ndim_output; ++i)
    {
        output[i] = slots[G.output_slots[i]];
    }
}

template <detail::TransformGraph G, index_t N>
CK_TILE_HOST_DEVICE constexpr index_t calculateOffset(const static_array<index_t, N>& input)
{
    static_assert(N == G.ndim_input,
                  "calculateOffset: input coordinate size must match graph's ndim_input");
    static_assert(G.ndim_output == 1,
                  "calculateOffset: requires ndim_output == 1 (use map() for N->M)");
    index_t output;
    map<G>(&output, input.elems);
    return output;
}

template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr index_t inputDimLength(index_t i)
{
    index_t slot = G.input_slots[i];
    return detail::inputDimLengthDispatch<G>(slot, make_index_sequence<G.num_transforms>{});
}

template <detail::TransformGraph G>
constexpr bool isGraphBijective()
{
    for(index_t t = 0; t < G.num_transforms; ++t)
        if(!G.transforms[t].is_bijective)
            return false;
    return true;
}

template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void reverseMap(index_t* input, const index_t* output)
{
    static_assert(
        []() constexpr {
            for(index_t t = 0; t < G.num_transforms; ++t)
                if(!G.transforms[t].is_bijective)
                    return false;
            return true;
        }(),
        "reverseMap: all transforms must be bijective");
    static_array<index_t, G.num_slots> slots{};

    for(index_t i = 0; i < G.ndim_output; ++i)
    {
        slots[G.output_slots[i]] = output[i];
    }

    detail::reverseApplyAllTransforms<G>(slots, make_index_sequence<G.num_transforms>{});

    for(index_t i = 0; i < G.ndim_input; ++i)
    {
        input[i] = slots[G.input_slots[i]];
    }
}

template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr static_array<index_t, G.ndim_input>
reverseCalculateOffset(index_t offset)
{
    static_assert(G.ndim_output == 1, "reverseCalculateOffset: requires ndim_output == 1");
    static_array<index_t, G.ndim_input> result{};
    reverseMap<G>(result.elems, &offset);
    return result;
}

} // namespace ck_tile::core::transform
