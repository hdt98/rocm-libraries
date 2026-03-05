#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/poly/polymorpher_helper.h>
#include <mint/tensor/tensor_descriptor.h>

namespace mint {
namespace tensor {

// bot dim: N-d
template <class DimAlias, DimAlias... kDimAliases>
MINT_HOST_DEVICE constexpr auto make_aliased_tensor_descriptor(
    const sequence<DimAlias, kDimAliases...>&,
    const nd_index<sizeof...(kDimAliases)>& lengths) {
  constexpr index_t ndim = sizeof...(kDimAliases);
  constexpr auto dim_aliases = array<DimAlias, ndim>{kDimAliases...};
  constexpr auto dim_pairs = nd_array<index_t, 0, 2, 2>{};
  constexpr auto alias_to_morpher = []() {
    static_map<index_t, index_t, ndim> ret;
    for (index_t i = 0; i < ndim; i++)
      ret[i] = i;
    return ret;
  }();
  constexpr auto alias_to_morpher_local_dim = [&dim_aliases]() {
    static_map<DimAlias, nd_index<2>, ndim> ret;
    for (index_t i = 0; i < ndim; i++)
      ret[dim_aliases[i]] = {0, i};
    return ret;
  }();

  const auto polymorpher_tmp = poly::
      make_polymorpher<dim_pairs, alias_to_morpher, alias_to_morpher_local_dim>(
          mint::make_tuple((static_cast<void>(kDimAliases), poly::none{})...),
          lengths);

  return tensor_descriptor<
      remove_cvref_t<decltype(polymorpher_tmp)>,
      dim_aliases,
      dim_aliases>{polymorpher_tmp};
}

template <class PolyMorpher>
MINT_HOST_DEVICE constexpr auto make_tensor_descriptor(PolyMorpher& poly) {
  return tensor_descriptor<
      PolyMorpher,
      PolyMorpher::top_dim_aliases(),
      PolyMorpher::bottom_dim_aliases()>{poly};
}

template <index_t kTopNDim, nd_index<kTopNDim> kTopDims>
MINT_HOST_DEVICE constexpr auto make_packed_tensor_descriptor_z2(
    integral_constant<nd_index<kTopNDim>, kTopDims>) {
  const index_t kBotSize = [&]() {
    index_t cnt = 1;
    for (index_t i = 0; i < kTopNDim; i++) {
      cnt *= kTopDims[i];
    }
    return cnt;
  }();
  constexpr auto m0 = poly::make_z2_pass_through_morpher(constant<kTopDims>{});
  constexpr auto m1 = reshape_bottom(m0, constant<nd_index<1>{kBotSize}>{});
  constexpr auto poly = make_z2_polymorpher_default_alias(m1);
  return make_tensor_descriptor(poly);
}

// bot dim : 1d
// packed
template <class DimAlias, DimAlias kOffsetDimAlias, DimAlias... kDimAliases>
MINT_HOST_DEVICE constexpr auto make_aliased_naive_packed_tensor_descriptor(
    sequence<DimAlias, kDimAliases...>,
    integral_constant<DimAlias, kOffsetDimAlias>,
    const nd_index<sizeof...(kDimAliases)>& lengths) {
  if constexpr (sizeof...(kDimAliases) == 0) {
    constexpr auto morphers = mint::make_tuple(poly::insert_length_one{});
    constexpr auto dim_pairs = nd_array<index_t, 0, 2, 2>{};
    constexpr auto all_lengths = nd_index<1>{1};
    constexpr auto morpher_aliases = nd_index<1>{0};
    constexpr auto dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    constexpr auto poly = poly::polymorpher<
        remove_cvref_t<decltype(morphers)>,
        dim_pairs,
        remove_cvref_t<decltype(all_lengths)>,
        morpher_aliases,
        dim_aliases>{morphers, all_lengths};

    constexpr auto top_dim_aliases = array<DimAlias, 0>{};
    constexpr auto bot_dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    return tensor_descriptor<
        remove_cvref_t<decltype(poly)>,
        top_dim_aliases,
        bot_dim_aliases>{poly};
  } else {
    constexpr index_t top_ndim = sizeof...(kDimAliases);
    constexpr index_t all_ndim = top_ndim + 1;
    constexpr auto dim_aliases =
        array<DimAlias, all_ndim>{kOffsetDimAlias, kDimAliases...};

    constexpr auto dim_pairs = nd_array<index_t, 0, 2, 2>{};

    constexpr auto alias_to_morpher = []() {
      static_map<index_t, index_t, 1> ret;
      ret[0] = 0;
      return ret;
    }();

    constexpr auto alias_to_morpher_local_dim = [&dim_aliases]() {
      static_map<DimAlias, nd_index<2>, all_ndim> ret;
      for (index_t i = 0; i < all_ndim; i++)
        ret[dim_aliases[i]] = {0, i};
      return ret;
    }();

    const auto morpher = poly::merge<nd_index<top_ndim>>{lengths};

    const auto all_lengths = [&lengths]() {
      nd_index<top_ndim + 1> ret;
      ret[0] = 1;
      static_for_n<top_ndim>()([&](auto i) {
        ret[i + 1] = lengths[i];
        ret[0] *= lengths[i];
      });
      return ret;
    }();

    const auto polymorpher_tmp = poly::make_polymorpher<
        dim_pairs,
        alias_to_morpher,
        alias_to_morpher_local_dim>(mint::make_tuple(morpher), all_lengths);

    constexpr auto top_dim_aliases = array<DimAlias, top_ndim>{kDimAliases...};

    constexpr auto bot_dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    return tensor_descriptor<
        remove_cvref_t<decltype(polymorpher_tmp)>,
        top_dim_aliases,
        bot_dim_aliases>{polymorpher_tmp};
  }
}

// bot dim : 1d
// packed
template <class DimAlias, DimAlias kOffsetDimAlias, DimAlias... kDimAliases>
MINT_HOST_DEVICE constexpr auto make_aliased_naive_tensor_descriptor(
    sequence<DimAlias, kDimAliases...>,
    integral_constant<DimAlias, kOffsetDimAlias>,
    const nd_index<sizeof...(kDimAliases)>& lengths,
    const nd_index<sizeof...(kDimAliases)>& strides) {
  if constexpr (sizeof...(kDimAliases) == 0) {
    constexpr auto morphers = mint::make_tuple(poly::insert_length_one{});
    constexpr auto dim_pairs = nd_array<index_t, 0, 2, 2>{};
    constexpr auto all_lengths = nd_index<1>{1};
    constexpr auto morpher_aliases = nd_index<1>{0};
    constexpr auto dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    constexpr auto poly = poly::polymorpher<
        remove_cvref_t<decltype(morphers)>,
        dim_pairs,
        remove_cvref_t<decltype(all_lengths)>,
        morpher_aliases,
        dim_aliases>{morphers, all_lengths};

    constexpr auto top_dim_aliases = array<DimAlias, 0>{};
    constexpr auto bot_dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    return tensor_descriptor<
        remove_cvref_t<decltype(poly)>,
        top_dim_aliases,
        bot_dim_aliases>{poly};
  } else {
    constexpr index_t top_ndim = sizeof...(kDimAliases);
    constexpr index_t all_ndim = top_ndim + 1;
    constexpr auto dim_aliases =
        array<DimAlias, all_ndim>{kOffsetDimAlias, kDimAliases...};

    constexpr auto dim_pairs = nd_array<index_t, 0, 2, 2>{};

    constexpr auto alias_to_morpher = []() {
      static_map<index_t, index_t, 1> ret;
      ret[0] = 0;
      return ret;
    }();

    constexpr auto alias_to_morpher_local_dim = [&dim_aliases]() {
      static_map<DimAlias, nd_index<2>, all_ndim> ret;
      for (index_t i = 0; i < all_ndim; i++)
        ret[dim_aliases[i]] = {0, i};
      return ret;
    }();

    const auto morpher = poly::project<nd_index<top_ndim>>{strides};

    const auto all_lengths = [&lengths, &strides]() {
      nd_index<top_ndim + 1> ret;
      ret[0] = 1;
      static_for_n<top_ndim>()([&](auto i) {
        ret[i + 1] = lengths[i];
        ret[0] += (lengths[i] - 1) * strides[i];
      });
      return ret;
    }();

    const auto polymorpher_tmp = poly::make_polymorpher<
        dim_pairs,
        alias_to_morpher,
        alias_to_morpher_local_dim>(mint::make_tuple(morpher), all_lengths);

    constexpr auto top_dim_aliases = array<DimAlias, top_ndim>{kDimAliases...};

    constexpr auto bot_dim_aliases = array<DimAlias, 1>{kOffsetDimAlias};

    return tensor_descriptor<
        remove_cvref_t<decltype(polymorpher_tmp)>,
        top_dim_aliases,
        bot_dim_aliases>{polymorpher_tmp};
  }
}

template <class TensorDesc>
MINT_HOST_DEVICE consteval bool is_naive_tensor_descriptor(const TensorDesc&) {
  if constexpr (
      remove_cvref_t<decltype(TensorDesc{}.polymorpher())>::num_morpher() != 1)
    return false;

  using morpher_type = decltype(TensorDesc{}.polymorpher().morphers()[0_ic]);

  if constexpr (
      (not ::mint::poly::is_merge_morpher_v<morpher_type>) and
      (not ::mint::poly::is_project_morpher_v<morpher_type>) and
      (not ::mint::poly::is_insert_length_one_morpher_v<morpher_type>))
    return false;

  return true;
}

template <class TensorDesc>
  requires(is_naive_tensor_descriptor<TensorDesc>(TensorDesc{}))
MINT_HOST_DEVICE constexpr auto get_naive_tensor_descriptor_strides(
    const TensorDesc& desc) {
  nd_index<TensorDesc::top_ndim()> ret;
  ret.fill(0);

  const auto& morpher = desc.polymorpher().morphers()[0_ic];
  using morpher_type = remove_cvref_t<decltype(morpher)>;

  if constexpr (::mint::poly::is_merge_morpher_v<morpher_type>) {
    std::copy(
        morpher.top_length_scan_.begin(),
        morpher.top_length_scan_.end(),
        ret.begin());
    ret[TensorDesc::top_ndim() - 1] = 1;
  } else if constexpr (::mint::poly::is_project_morpher_v<morpher_type>) {
    std::copy(
        morpher.coefficients_.begin(),
        morpher.coefficients_.end(),
        ret.begin());
  }

  return ret;
}

// ExtraMorpherInfos:
//   tuple<
//     Morpher,
//     sequence<DimAlias, kMorpherDimAliases...>>
//    reqiures: Morpher::all_ndim() == sizeof...(kMorpherDimAliases)
template <
    class OldTensorDesc,
    class DimAlias,
    DimAlias... kNewTopDimAliases,
    DimAlias... kNewBottomDimAliases,
    DimAlias... kExtraDimAliases,
    class ExtraDimLengths,
    class... ExtraMorpherInfos>
MINT_HOST_DEVICE constexpr auto morph_tensor_descriptor(
    const OldTensorDesc& old_tensor_desc,
    sequence<DimAlias, kNewTopDimAliases...>,
    sequence<DimAlias, kNewBottomDimAliases...>,
    sequence<DimAlias, kExtraDimAliases...>,
    const ExtraDimLengths& extra_dim_lengths,
    const ExtraMorpherInfos&... extra_morpher_infos_in) {
  constexpr index_t old_num_morpher =
      OldTensorDesc{}.polymorpher().num_morpher();
  constexpr index_t extra_num_morpher = sizeof...(ExtraMorpherInfos);

  constexpr auto extra_dim_aliases = sequence<DimAlias, kExtraDimAliases...>{};
  constexpr auto new_top_dim_aliases =
      array<DimAlias, sizeof...(kNewTopDimAliases)>{kNewTopDimAliases...};
  constexpr auto new_bot_dim_aliases =
      array<DimAlias, sizeof...(kNewBottomDimAliases)>{kNewBottomDimAliases...};

#if 0
  // FIXME: why doesn't compile?
  const auto extra_morpher_infos_tmp = mint::tie(extra_morpher_infos_in...);
#else
  const auto extra_morpher_infos_tmp =
      mint::make_tuple(extra_morpher_infos_in...);
#endif

  const auto extra_morpher_infos =
      generate_tuple<extra_num_morpher>([&](auto i) {
        return mint::make_tuple(
            extra_morpher_infos_tmp[i][0_ic],
            integral_constant<index_t, i + old_num_morpher>{},
            extra_morpher_infos_tmp[i][1_ic]);
      });

  const auto new_polymorpher =
      pack_arg<extra_num_morpher>([&](const auto&... xs) {
        return poly::add_to_polymorpher(
            old_tensor_desc.polymorpher(),
            extra_dim_aliases,
            extra_dim_lengths,
            xs...);
      })(extra_morpher_infos);

  using NewPolymorpher = remove_cvref_t<decltype(new_polymorpher)>;

  return tensor_descriptor<
      NewPolymorpher,
      new_top_dim_aliases,
      new_bot_dim_aliases>{new_polymorpher};
}

} // namespace tensor
} // namespace mint
