// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<sequence<...>, ...>
// TopDimensionHiddenIds> : sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize,
          typename GuaranteedVectorLengths_,
          typename GuaranteedVectorSrides_>
struct tensor_descriptor_tiled : public tensor_descriptor<Transforms,
                                                          LowerDimensionHiddenIdss,
                                                          UpperDimensionHiddenIdss,
                                                          TopDimensionHiddenIds,
                                                          ElementSpaceSize,
                                                          GuaranteedVectorLengths_,
                                                          GuaranteedVectorSrides_>
{
    using Base = tensor_descriptor<Transforms,
                                   LowerDimensionHiddenIdss,
                                   UpperDimensionHiddenIdss,
                                   TopDimensionHiddenIds,
                                   ElementSpaceSize,
                                   GuaranteedVectorLengths_,
                                   GuaranteedVectorSrides_>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_descriptor_tiled() = default;

    CK_TILE_HOST_DEVICE static constexpr bool is_tiled() { return true; }

    using Base::Base;
};

template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize,
          typename GuaranteedVectorLengths,
          typename GuaranteedVectorStrides>
CK_TILE_HOST_DEVICE static void print(const tensor_descriptor_tiled<Transforms,
                                                              LowerDimensionHiddenIdss,
                                                              UpperDimensionHiddenIdss,
                                                              TopDimensionHiddenIds,
                                                              ElementSpaceSize,
                                                              GuaranteedVectorLengths,
                                                              GuaranteedVectorStrides>& descriptor)
{
    printf("tensor_descriptor_tiled{\n");
    // first print the tensor adaptor part of the descriptor using the base class print
    using Base = typename tensor_descriptor_tiled<Transforms,
                                            LowerDimensionHiddenIdss,
                                            UpperDimensionHiddenIdss,
                                            TopDimensionHiddenIds,
                                            ElementSpaceSize,
                                            GuaranteedVectorLengths,
                                            GuaranteedVectorStrides>::Base;
    print(static_cast<const Base&>(descriptor));
    printf("element_space_size_: %ld,\n", static_cast<long>(descriptor.get_element_space_size()));
    printf("guaranteed_vector_lengths: ");
    print(GuaranteedVectorLengths{});
    printf(",\nguaranteed_vector_strides: ");
    print(GuaranteedVectorStrides{});
    printf("}\n}\n");
}

template <typename Adaptor, typename ElementSpaceSize>
CK_TILE_HOST_DEVICE constexpr auto
make_tensor_descriptor_tiled_from_adaptor(const Adaptor& adaptor,
                                    const ElementSpaceSize& element_space_size)
{
    constexpr index_t NDimHidden = Adaptor::get_num_of_hidden_dimension();

    return tensor_descriptor_tiled<remove_cvref_t<decltype(adaptor.get_transforms())>,
                             remove_cvref_t<decltype(adaptor.get_lower_dimension_hidden_idss())>,
                             remove_cvref_t<decltype(adaptor.get_upper_dimension_hidden_idss())>,
                             remove_cvref_t<decltype(adaptor.get_top_dimension_hidden_ids())>,
                             remove_cvref_t<decltype(element_space_size)>,
                             typename uniform_sequence_gen<NDimHidden, -1>::type,
                             typename uniform_sequence_gen<NDimHidden, -1>::type>{
        adaptor, element_space_size};
}

template <typename OldTensorDescriptor,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
CK_TILE_HOST_DEVICE constexpr auto
transform_tensor_descriptor_tiled(const OldTensorDescriptor& old_tensor_desc,
                            const NewTransforms& new_transforms,
                            NewLowerDimensionOldTopIdss,
                            NewUpperDimensionNewTopIdss)
{
    const auto element_space_size = old_tensor_desc.get_element_space_size();

    const auto new_tensor_adaptor = transform_tensor_adaptor(old_tensor_desc,
                                                             new_transforms,
                                                             NewLowerDimensionOldTopIdss{},
                                                             NewUpperDimensionNewTopIdss{});

    constexpr index_t NDimHiddenOld = OldTensorDescriptor::get_num_of_hidden_dimension();
    constexpr index_t NDimHiddenNew = decltype(new_tensor_adaptor)::get_num_of_hidden_dimension();

    using NewGuaranteedVectorLengths = typename sequence_merge<
        typename OldTensorDescriptor::GuaranteedVectorLengths,
        typename uniform_sequence_gen<NDimHiddenNew - NDimHiddenOld, -1>::type>::type;

    using NewGuaranteedVectorStrides = typename sequence_merge<
        typename OldTensorDescriptor::GuaranteedVectorStrides,
        typename uniform_sequence_gen<NDimHiddenNew - NDimHiddenOld, -1>::type>::type;

    return tensor_descriptor_tiled<
        remove_cvref_t<decltype(new_tensor_adaptor.get_transforms())>,
        remove_cvref_t<decltype(new_tensor_adaptor.get_lower_dimension_hidden_idss())>,
        remove_cvref_t<decltype(new_tensor_adaptor.get_upper_dimension_hidden_idss())>,
        remove_cvref_t<decltype(new_tensor_adaptor.get_top_dimension_hidden_ids())>,
        remove_cvref_t<decltype(element_space_size)>,
        NewGuaranteedVectorLengths,
        NewGuaranteedVectorStrides>{new_tensor_adaptor, element_space_size};
}

namespace detail {

template <typename Lengths, typename Strides, index_t I, typename AccOld>
CK_TILE_HOST_DEVICE constexpr auto calculate_element_space_size_impl_recursive(const Lengths& lengths,
                                                                     const Strides& strides,
                                                                     number<I> i,
                                                                     AccOld acc_old)
{
    auto acc_new = acc_old + (lengths[i] - number<1>{}) * strides[i];

    if constexpr(i.value < Lengths::size() - 1)
    {
        return calculate_element_space_size_impl_recursive(lengths, strides, i + number<1>{}, acc_new);
    }
    else
    {
        return acc_new;
    }
}

} // namespace detail

/*
 * These functions create naive tensor descriptor
 */

// Lengths..., Strides... could be:
//   1) index_t, which is known at run-time, or
//   2) number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) long_number<>
template <typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength                                   = -1,
          index_t GuaranteedLastDimensionVectorStride                                   = -1,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_descriptor_tiled(const tuple<Lengths...>& lengths,
                             const tuple<Strides...>& strides,
                             number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                             number<GuaranteedLastDimensionVectorStride> = number<-1>{})
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_embed_transform(lengths, strides));

    constexpr auto low_dim_hidden_idss = make_tuple(sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size =
        detail::calculate_element_space_size_impl_recursive(lengths, strides, number<0>{}, long_number<1>{});

    using GuaranteedVectorLengths =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type,
                                sequence<GuaranteedLastDimensionVectorLength>>::type;

    using GuaranteedVectorStrides =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type,
                                sequence<GuaranteedLastDimensionVectorStride>>::type;

    return tensor_descriptor_tiled<remove_cv_t<decltype(transforms)>,
                             remove_cv_t<decltype(low_dim_hidden_idss)>,
                             remove_cv_t<decltype(up_dim_hidden_idss)>,
                             remove_cv_t<decltype(visible_dim_hidden_ids)>,
                             remove_cv_t<decltype(element_space_size)>,
                             GuaranteedVectorLengths,
                             GuaranteedVectorStrides>{transforms, element_space_size};
}

// tensor descriptor with offset, the offset will not be added into element space size
// only have an information of the starting offset, and will impact on offset calculation
template <typename... Lengths,
          typename... Strides,
          typename offset,
          index_t GuaranteedLastDimensionVectorLength                                   = -1,
          index_t GuaranteedLastDimensionVectorStride                                   = -1,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_descriptor_tiled_with_offset(const tuple<Lengths...>& lengths,
                                         const tuple<Strides...>& strides,
                                         const offset& os,
                                         number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                                         number<GuaranteedLastDimensionVectorStride> = number<-1>{})
{
    const auto desc_0 = [&]() {
        const auto element_space_size = detail::calculate_element_space_size_impl_recursive(
            lengths, strides, number<0>{}, long_number<1>{});

        const auto transforms = make_tuple(make_offset_transform(element_space_size, os));

        constexpr auto low_dim_hidden_idss = make_tuple(sequence<0>{});

        constexpr auto up_dim_hidden_idss = make_tuple(sequence<1>{});

        constexpr auto visible_dim_hidden_ids = sequence<1>{};

        using GuaranteedVectorLengths =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    sequence<GuaranteedLastDimensionVectorLength>>::type;

        using GuaranteedVectorStrides =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    sequence<GuaranteedLastDimensionVectorStride>>::type;

        return tensor_descriptor_tiled<remove_cv_t<decltype(transforms)>,
                                 remove_cv_t<decltype(low_dim_hidden_idss)>,
                                 remove_cv_t<decltype(up_dim_hidden_idss)>,
                                 remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                 remove_cv_t<decltype(element_space_size)>,
                                 GuaranteedVectorLengths,
                                 GuaranteedVectorStrides>{transforms, element_space_size};
    }();

    constexpr index_t N = sizeof...(Lengths);

    return transform_tensor_descriptor_tiled(
        desc_0,
        make_tuple(make_embed_transform(lengths, strides)),
        make_tuple(sequence<0>{}),
        make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}));
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) long_number<>
template <typename... Lengths, index_t GuaranteedLastDimensionVectorLength = -1>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_descriptor_tiled_packed(const tuple<Lengths...>& lengths,
                                    number<GuaranteedLastDimensionVectorLength> = number<-1>{})
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_unmerge_transform(lengths));

    constexpr auto low_dim_hidden_idss = make_tuple(sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size = container_reduce(lengths, multiplies<>{}, long_number<1>{});

    constexpr index_t first_dim_length = []() {
        if constexpr(is_constant_v<remove_cvref_t<decltype(element_space_size)>>)
            return decltype(element_space_size)::value;
        else
            return -1;
    }();
    using last_t                      = remove_cvref_t<decltype(lengths.template get<N - 1>())>;
    constexpr index_t last_dim_length = []() {
        if constexpr(is_constant_v<last_t>)
            return std::max(last_t::value, GuaranteedLastDimensionVectorLength);
        else
            return -1;
    }();

    using GuaranteedVectorLengths =
        typename sequence_merge<sequence<first_dim_length>,
                                typename uniform_sequence_gen<N - 1, -1>::type,
                                sequence<last_dim_length>>::type;

    using GuaranteedVectorStrides =
        typename sequence_merge<sequence<1>,
                                typename uniform_sequence_gen<N - 1, -1>::type,
                                sequence<1>>::type;

    return tensor_descriptor_tiled<remove_cv_t<decltype(transforms)>,
                             remove_cv_t<decltype(low_dim_hidden_idss)>,
                             remove_cv_t<decltype(up_dim_hidden_idss)>,
                             remove_cv_t<decltype(visible_dim_hidden_ids)>,
                             remove_cv_t<decltype(element_space_size)>,
                             GuaranteedVectorLengths,
                             GuaranteedVectorStrides>{transforms, element_space_size};
}

template <typename... Lengths,
          typename... Strides,
          typename Offset,
          index_t GuaranteedLastDimensionVectorLength                                   = -1,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto make_naive_tensor_descriptor_tiled_packed_with_offset(
    const tuple<Lengths...>& lengths,
    const Offset& offset,
    number<GuaranteedLastDimensionVectorLength> = number<-1>{})
{
    const auto desc_0 = [&]() {
        const auto element_space_size = container_reduce(lengths, multiplies<>{}, long_number<1>{});

        const auto transforms = make_tuple(make_offset_transform(element_space_size, offset));

        constexpr auto low_dim_hidden_idss = make_tuple(sequence<0>{});

        constexpr auto up_dim_hidden_idss = make_tuple(sequence<1>{});

        constexpr auto visible_dim_hidden_ids = sequence<1>{};

        using GuaranteedVectorLengths =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    sequence<GuaranteedLastDimensionVectorLength>>::type;

        using GuaranteedVectorStrides =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type, sequence<1>>::type;

        return tensor_descriptor_tiled<remove_cv_t<decltype(transforms)>,
                                 remove_cv_t<decltype(low_dim_hidden_idss)>,
                                 remove_cv_t<decltype(up_dim_hidden_idss)>,
                                 remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                 remove_cv_t<decltype(element_space_size)>,
                                 GuaranteedVectorLengths,
                                 GuaranteedVectorStrides>{transforms, element_space_size};
    }();

    constexpr index_t N = sizeof...(Lengths);

    return transform_tensor_descriptor_tiled(
        desc_0,
        make_tuple(make_unmerge_transform(lengths)),
        make_tuple(sequence<0>{}),
        make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}));
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) number<>, which is known at compile-time
// align could be:
//   1) index_t, or
//   2) number<>
template <typename... Lengths, typename Align>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_descriptor_tiled_aligned(const tuple<Lengths...>& lengths, Align align)
{
    constexpr auto I1 = number<1>{};

    constexpr index_t N = sizeof...(Lengths);

    const auto stride_n_minus_2 = integer_least_multiple(lengths[number<N - 1>{}], align);

    auto strides = generate_tuple(
        [&](auto i) {
            if constexpr(i.value == N - 1)
            {
                return I1;
            }
            else if constexpr(i.value == N - 2)
            {
                return number<stride_n_minus_2>{};
            }
            else
            {
                return container_reduce(lengths,
                                        multiplies<>{},
                                        number<stride_n_minus_2>{},
                                        i + I1,
                                        number<N - 1>{},
                                        I1);
            }
        },
        number<N>{});

    return make_naive_tensor_descriptor_tiled(lengths, strides);
}

} // namespace ck_tile
