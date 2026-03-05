#pragma once
#include <mint/core.h>

namespace mint {
namespace poly {

template <typename Derived>
struct morpher {
  constexpr bool operator==(const morpher& other) const = default;

  MINT_HOST_DEVICE static consteval bool is_fundamental() {
    return Derived::is_fundamental_;
  }

  MINT_HOST_DEVICE static consteval bool can_top_down() {
    return Derived::can_top_down_;
  }

  MINT_HOST_DEVICE static consteval bool can_bottom_up() {
    return Derived::can_bottom_up_;
  }

  MINT_HOST_DEVICE static consteval bool is_bidirectional() {
    return can_top_down() && can_bottom_up();
  }

  MINT_HOST_DEVICE static consteval auto is_linear_top_down() {
    return Derived::is_linear_top_down_;
  }

  MINT_HOST_DEVICE static consteval auto is_linear_bottom_up() {
    return Derived::is_linear_bottom_up_;
  }

  MINT_HOST_DEVICE static consteval index_t all_ndim() {
    return Derived::all_ndim_;
  }

  MINT_HOST_DEVICE static consteval index_t top_ndim() {
    return Derived::top_ndim_;
  }

  MINT_HOST_DEVICE static consteval index_t bottom_ndim() {
    return Derived::bottom_ndim_;
  }

  MINT_HOST_DEVICE static consteval index_t paired_ndim() {
    return Derived::paired_ndim_;
  }

  MINT_HOST_DEVICE static consteval auto top_dims() {
    return Derived::top_dims_;
  }

  MINT_HOST_DEVICE static consteval auto bottom_dims() {
    return Derived::bottom_dims_;
  }

  MINT_HOST_DEVICE static consteval auto paired_dims() {
    return Derived::paired_dims_;
  }

  MINT_HOST_DEVICE static consteval auto is_top_dim() {
    array<bool, all_ndim()> ret;
    ret.fill(false);
    for (auto dim : top_dims())
      ret[dim] = true;
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto is_bottom_dim() {
    array<bool, all_ndim()> ret;
    ret.fill(false);
    for (auto dim : bottom_dims())
      ret[dim] = true;
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto is_paired_dim() {
    array<bool, all_ndim()> ret;
    ret.fill(false);
    for (auto dim : paired_dims())
      ret[dim] = true;
    return ret;
  }

  template <index_t kNDimTop>
  MINT_HOST_DEVICE constexpr auto calculate_bottom_index(
      const nd_index<kNDimTop>& top_idx) const {
    static_assert(kNDimTop == top_ndim(), "wrong!");
    nd_index<all_ndim()> idx{};
    idx.template set_subset<top_ndim(), top_dims()>(top_idx);
    static_cast<const Derived*>(this)->propagate_index_top_down(idx);
    return idx.template get_subset<bottom_ndim(), bottom_dims()>();
  }

  template <index_t kNDimBottom>
  MINT_HOST_DEVICE constexpr auto calculate_top_index(
      const nd_index<kNDimBottom>& bot_idx) const {
    static_assert(kNDimBottom == bottom_ndim(), "wrong!");
    nd_index<all_ndim()> idx{};
    idx.template set_subset<bottom_ndim(), bottom_dims()>(bot_idx);
    static_cast<const Derived*>(this)->propagate_index_bottom_up(idx);
    return idx.template get_subset<top_ndim(), top_dims()>();
  }

  // propagate idx and idx_delta with dim freezing
  // freezed elements in idx will not be updated
  // freezed elements in idx_delta will be set to 0
  template <auto kIsFreezedDim, class Index, class IndexDelta>
    requires(
        is_same_v<typename decltype(kIsFreezedDim)::value_type, bool> &&
        kIsFreezedDim.size() == all_ndim() &&
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim() &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim())
  MINT_HOST_DEVICE constexpr void
  propagate_index_and_delta_top_down_freezed_dim(
      Index& idx,
      IndexDelta& idx_delta) const {
    // init non-top_dims of idx_delta to 0
    static_for_n<all_ndim()>()([&](auto i) {
      if constexpr (not is_top_dim()[i])
        idx_delta[i] = 0;
    });

#if 1 // debug +0
    // init idx_freezed, idx_delta_freezed
    freezed_array<typename Index::value_type, all_ndim(), kIsFreezedDim>
        idx_freezed{idx}, idx_delta_freezed{idx_delta};

    // propagate idx_freezed and idx_delta_freezed
    static_cast<const Derived*>(this)->propagate_index_and_delta_top_down(
        idx_freezed, idx_delta_freezed);

    // retrieve idx and idx_delta from idx_freezed and idx_delta_freezed
    static_for_n<all_ndim()>()([&](auto i) {
      if constexpr (not is_top_dim()[i]) {
        idx[i] = idx_freezed[i];
        idx_delta[i] = idx_delta_freezed[i];
      }
    });
#elif 0 // +0
    // init idx_freezed, idx_delta_freezed
    freezed_array<
        typename Index::value_type,
        all_ndim(),
        array<bool, all_ndim()>{}.fill(false)>
        idx_freezed{idx}, idx_delta_freezed{idx_delta};

    // propagate idx_freezed and idx_delta_freezed
    static_cast<const Derived*>(this)->propagate_index_and_delta_top_down(
        idx_freezed, idx_delta_freezed);

    // retrieve idx and idx_delta from idx_freezed and idx_delta_freezed
    static_for_n<all_ndim()>()([&](auto i) {
      if constexpr (not is_top_dim()[i]) {
        idx[i] = idx_freezed[i];
        idx_delta[i] = idx_delta_freezed[i];
      }
    });
#else // +1
    static_cast<const Derived*>(this)->propagate_index_and_delta_top_down(
        idx, idx_delta);
#endif
  }

  // propagate idx and idx_delta with dim freezing, and with additional
  // conjecture. this is the default version with no conjecture, should be
  // overridden by the child morpher if it a has better conjecture
  template <auto kIsFreezedDim, class Index, class IndexDelta>
  MINT_HOST_DEVICE constexpr void
  propagate_index_and_delta_top_down_freezed_dim_conjectural(
      Index& idx,
      IndexDelta& idx_delta) const {
    propagate_index_and_delta_top_down_freezed_dim<kIsFreezedDim>(
        idx, idx_delta);
  }

  MINT_HOST_DEVICE void print() const {
#if 0
    printf("poly::morpher: {");
    printf("is_linear_top_down(): ");
    is_linear_top_down().print();
    printf(", ");
    printf("is_linear_bottom_up(): ");
    is_linear_bottom_up().print();
    printf(", ");
    printf("}");
#endif
  }
};

} // namespace poly
} // namespace mint
