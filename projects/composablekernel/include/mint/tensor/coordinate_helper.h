#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/coordinate.h>

namespace mint {
namespace tensor {

template <class TensorDesc>
MINT_HOST_DEVICE constexpr auto move_coordinate_top_down(
    coordinate<TensorDesc>& coord,
    const TensorDesc& tensor_desc,
    const nd_index<TensorDesc::top_ndim()>& top_idx_delta) {
  coord.set_top_index(coord.get_top_index() + top_idx_delta);
  nd_index<TensorDesc::all_ndim()> all_idx_delta{};
  all_idx_delta
      .template set_subset<TensorDesc::top_ndim(), TensorDesc::top_dims()>(
          top_idx_delta);
  tensor_desc.polymorpher().propagate_index_and_delta_top_down(
      coord.all_index(), all_idx_delta);
  return all_idx_delta.template get_subset<
      TensorDesc::bottom_ndim(),
      TensorDesc::bottom_dims()>();
}

template <class TensorDesc>
MINT_HOST_DEVICE constexpr auto move_coordinate_bottom_up(
    coordinate<TensorDesc>& coord,
    const TensorDesc& tensor_desc,
    const nd_index<TensorDesc::bottom_ndim()>& bot_idx_delta) {
  coord.set_bottom_index(coord.get_bottom_index() + bot_idx_delta);
  nd_index<TensorDesc::all_ndim()> all_idx_delta{};
  all_idx_delta.template set_subset<
      TensorDesc::bottom_ndim(),
      TensorDesc::bottom_dims()>(bot_idx_delta);
  tensor_desc.polymorpher().propagate_index_and_delta_bottom_up(
      coord.all_index(), all_idx_delta);
  return all_idx_delta
      .template get_subset<TensorDesc::top_ndim(), TensorDesc::top_dims()>();
}

template <class TensorDesc>
MINT_HOST_DEVICE constexpr auto move_coordinate(
    coordinate<TensorDesc>& coord,
    const TensorDesc& tensor_desc,
    const nd_index<TensorDesc::top_ndim()>& top_idx_delta) {
  return move_coordinate_top_down(coord, tensor_desc, top_idx_delta);
}

// Masks:
//   tuple<
//     tuple<
//       sequence<dim_alias, kAliasedMaskedDims...>,
//       mask_function
//     >...
//   >
template <class TensorDesc, class Masks>
MINT_HOST_DEVICE constexpr bool is_unmasked(
    const coordinate<TensorDesc>& coord,
    const TensorDesc& /*tensor_desc*/,
    const Masks& masks) {
  bool flag = true;
  static_for_n<Masks::size()>()([&](auto i) {
    constexpr auto mask_dim_aliases = decltype(masks[i][0_ic]){};
    constexpr index_t kN = mask_dim_aliases.size();
    constexpr auto mask_dims = [&]() {
      nd_index<kN> ret;
      for (index_t j = 0; j < kN; j++)
        ret[j] = TensorDesc::alias_to_dim()[mask_dim_aliases[j]];
      return ret;
    }();
    flag &= masks[i][1_ic](
        coord.all_index().template get_subset<mask_dims.size(), mask_dims>());
  });
  return flag;
}

namespace experimental {

template <auto kIsFreezedDim, class TensorDesc>
MINT_HOST_DEVICE constexpr auto
move_coordinate_top_down_freezed_dim_conjectural(
    coordinate<TensorDesc>& coord,
    const TensorDesc& tensor_desc,
    const nd_index<TensorDesc::top_ndim()>& top_idx_delta) {
  coord.set_top_index(coord.get_top_index() + top_idx_delta);
  nd_index<TensorDesc::all_ndim()> all_idx_delta{};
  all_idx_delta
      .template set_subset<TensorDesc::top_ndim(), TensorDesc::top_dims()>(
          top_idx_delta);
  tensor_desc.polymorpher()
      .template propagate_index_and_delta_top_down_freezed_dim_conjectural<
          kIsFreezedDim>(coord.all_index(), all_idx_delta);
  return all_idx_delta.template get_subset<
      TensorDesc::bottom_ndim(),
      TensorDesc::bottom_dims()>();
}

} // namespace experimental
} // namespace tensor
} // namespace mint
