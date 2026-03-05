#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/coordinate.h>

namespace mint::tensor {

template <
    class Polymorpher,
    auto kTopDimAliases, // array<DimAlias, top_ndim>
    auto kBottomDimAliases // array<DimAlias, bottom_ndim>
    >
  requires(
      Polymorpher::top_ndim() == kTopDimAliases.size() &&
      Polymorpher::bottom_ndim() == kBottomDimAliases.size())
struct tensor_descriptor {
  const Polymorpher polymorpher_{};

  constexpr tensor_descriptor() = default;

  MINT_HOST_DEVICE constexpr tensor_descriptor(const Polymorpher& polymorpher)
      : polymorpher_{polymorpher} {}

  MINT_HOST_DEVICE static consteval bool can_bottom_up() {
    return Polymorpher::can_bottom_up();
  }

  MINT_HOST_DEVICE static consteval bool can_top_down() {
    return Polymorpher::can_top_down();
  }

  MINT_HOST_DEVICE static consteval index_t top_ndim() {
    return Polymorpher::top_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t bottom_ndim() {
    return Polymorpher::bottom_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t all_ndim() {
    return Polymorpher::all_ndim();
  }

  MINT_HOST_DEVICE static consteval auto top_dim_aliases() {
    return kTopDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto bottom_dim_aliases() {
    return kBottomDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto top_dims() {
    nd_index<top_ndim()> ret;
    for (index_t i = 0; i < top_ndim(); i++)
      ret[i] = Polymorpher::alias_to_dim()[kTopDimAliases[i]];
    return ret;
  }

  template <alias_t alias>
  MINT_HOST_DEVICE static consteval auto top_alias_to_dim() {
    constexpr auto res = []() {
      for (index_t i = 0; i < top_ndim(); i++) {
        if (top_dim_aliases()[i] == alias) {
          return top_dims()[i];
        }
      }
      return -1;
    }();
    static_assert(res != -1, "top dim_alias not found!");
    return res;
  }

  MINT_HOST_DEVICE static consteval auto bottom_dims() {
    nd_index<bottom_ndim()> ret;
    for (index_t i = 0; i < bottom_ndim(); i++)
      ret[i] = Polymorpher::alias_to_dim()[kBottomDimAliases[i]];
    return ret;
  }

  MINT_HOST_DEVICE constexpr const auto& all_lengths() const {
    return polymorpher_.all_lengths();
  }

  MINT_HOST_DEVICE constexpr auto top_lengths() const {
    return all_lengths().template get_subset<top_ndim(), top_dims()>();
  }

  MINT_HOST_DEVICE constexpr auto bottom_lengths() const {
    return all_lengths().template get_subset<bottom_ndim(), bottom_dims()>();
  }

  MINT_HOST_DEVICE constexpr const Polymorpher& polymorpher() const {
    return polymorpher_;
  }

  MINT_HOST_DEVICE static consteval auto alias_to_dim() {
    return Polymorpher::alias_to_dim();
  }

  MINT_HOST_DEVICE constexpr auto calculate_bottom_index(
      const nd_index<top_ndim()>& top_idx) const {
    nd_index<all_ndim()> all_idx{};
    all_idx.template set_subset<top_ndim(), top_dims()>(top_idx);
    polymorpher_.propagate_index_top_down(all_idx);
    return all_idx.template get_subset<bottom_ndim(), bottom_dims()>();
  }

  MINT_HOST_DEVICE constexpr auto calculate_top_index(
      const nd_index<bottom_ndim()>& bot_idx) const {
    nd_index<all_ndim()> all_idx{};
    all_idx.template set_subset<bottom_ndim(), bottom_dims()>(bot_idx);
    polymorpher_.propagate_index_bottom_up(all_idx);
    return all_idx.template get_subset<top_ndim(), top_dims()>();
  }

  MINT_HOST_DEVICE void print() const {
    printf("tensor_descriptor: {");

    printf("polymorpher_: ");
    polymorpher_.print();
    printf(", ");

    printf("top_dims(): ");
    top_dims().print();
    printf(", ");

    printf("bottom_dims(): ");
    bottom_dims().print();
    printf(", ");

    printf("top_dim_aliases(): ");
    top_dim_aliases().print();
    printf(", ");

    printf("bottom_dim_aliases(): ");
    bottom_dim_aliases().print();

    printf("}");
  }

  // sanity check
  static constexpr auto sorted_top_dims = []() {
    auto ret = top_dims();
    ::std::sort(ret.begin(), ret.end());
    return ret;
  }();
  static_assert(
      sorted_top_dims == Polymorpher::top_dims(),
      "wrong kTopDimAliases?");

  static constexpr auto sorted_bot_dims = []() {
    auto ret = bottom_dims();
    ::std::sort(ret.begin(), ret.end());
    return ret;
  }();
  static_assert(
      sorted_bot_dims == Polymorpher::bottom_dims(),
      "wrong kBottomDimAliases?");
};

// Masks:
//   tuple<
//     tuple<
//       sequence<DimAlias, kAliasedMaskedDims...>,
//       mask_function
//     >...
//   >
template <class TensorDesc, class Masks>
MINT_HOST_DEVICE constexpr bool is_unmasked(
    const nd_index<TensorDesc::top_ndim()>& top_idx,
    const TensorDesc& desc,
    const Masks& masks) {
  nd_index<TensorDesc::all_ndim()> all_idx;
  all_idx.template set_subset<TensorDesc::top_ndim(), TensorDesc::top_dims()>(
      top_idx);
  desc.polymorpher().propagate_index_top_down(all_idx);
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
        all_idx.template get_subset<mask_dims.size(), mask_dims>());
  });
  return flag;
}

} // namespace mint::tensor
