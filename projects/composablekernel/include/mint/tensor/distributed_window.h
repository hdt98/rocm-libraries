#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/tensor_descriptor_helper.h>
#include <mint/tensor/tensor_view.h>

namespace mint {
namespace tensor {

namespace impl {

template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kDstrTensorDesc.element_ndim() == kElementTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
struct distributed_window_impl {
  using tensor_view_type = remove_reference_t<TensorView>;
  using tensor_desc_type = typename tensor_view_type::tensor_desc_type;
  using coord_type = coordinate<tensor_desc_type>;
  using dstr_tensor_desc_type = remove_cvref_t<decltype(kDstrTensorDesc)>;
  using element_tensor_desc_type = remove_cvref_t<decltype(kElementTensorDesc)>;
  using value_type = typename tensor_view_type::value_type;

#if 1 // debug
  const tensor_view_type tensor_view_;
#else
  // FIXME: dangerous to use reference version since the object it refers to may
  // go out of life span, but it use less registers. need a compiler fix so we
  // can use the non-reference version
  const tensor_view_type& tensor_view_;
#endif
  coord_type coord_;
  index_t coord_byte_offset_;

 public:
  MINT_HOST_DEVICE static consteval index_t ndim() {
    return dstr_tensor_desc_type::top_ndim();
  }

  MINT_HOST_DEVICE static consteval auto lengths() {
    return dstr_tensor_desc_type::top_lengths();
  }

  MINT_HOST_DEVICE constexpr distributed_window_impl(
      const tensor_view_type& tensor_view_in,
      const nd_index<ndim()>& idx)
      : tensor_view_{tensor_view_in}, coord_{}, coord_byte_offset_{} {
    // distributed coordinate
    using dstr_coord_type =
        coordinate<typename dstr_tensor_desc_type::tensor_desc_type>;
    dstr_coord_type dstr_coord;
    dstr_coord.all_index()
        .template subset_reference<
            dstr_tensor_desc_type::element_ndim(),
            dstr_tensor_desc_type::element_dims()>()
        .fill(0);
    dstr_coord.all_index()
        .template set_subset<
            dstr_tensor_desc_type::partition_ndim(),
            dstr_tensor_desc_type::partition_dims()>(
            kPartition.my_partition_idx());
    kDstrTensorDesc.tensor_desc().polymorpher().propagate_index_bottom_up(
        dstr_coord.all_index());

    // coordinate
    coord_ = coord_type{
        tensor_view_.tensor_desc(), idx + dstr_coord.get_top_index()};

    // byte_offset
    coord_byte_offset_ = coord_.get_bottom_index()[0] * sizeof(value_type);
  }

  MINT_HOST_DEVICE const tensor_view_type& tensor_view() const {
    return tensor_view_;
  }

  MINT_HOST_DEVICE static consteval auto dstr_tensor_desc() {
    return kDstrTensorDesc;
  }

  MINT_HOST_DEVICE static consteval auto element_tensor_desc() {
    return kElementTensorDesc;
  }

  template <alias_t alias>
  MINT_HOST_DEVICE index_t alias_to_index() const {
    constexpr auto index = tensor_desc_type::template top_alias_to_dim<alias>();
    return coord_.all_index()[index];
  }
};

} // namespace impl

template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(!TensorView::is_const_tensor_view())
struct distributed_window : impl::distributed_window_impl<
                                TensorView,
                                kDstrTensorDesc,
                                kElementTensorDesc,
                                kPartition> {
  using impl::distributed_window_impl<
      remove_cvref_t<TensorView>,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>::distributed_window_impl;
};

template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(TensorView::is_const_tensor_view())
struct const_distributed_window : impl::distributed_window_impl<
                                      TensorView,
                                      kDstrTensorDesc,
                                      kElementTensorDesc,
                                      kPartition> {
  using impl::distributed_window_impl<
      TensorView,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>::distributed_window_impl;
};

// make distributed window
template <
    class DstrTensorDesc,
    DstrTensorDesc kDstrTensorDesc,
    class ElementTensorDesc,
    ElementTensorDesc kElementTensorDesc,
    class TensorView,
    class Partition,
    Partition kPartition>
MINT_HOST_DEVICE constexpr auto make_distributed_window(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    integral_constant<DstrTensorDesc, kDstrTensorDesc>,
    integral_constant<ElementTensorDesc, kElementTensorDesc>,
    integral_constant<Partition, kPartition>) {
  if constexpr (TensorView::is_const_tensor_view()) {
    return const_distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>{view, idx};
  } else {
    return distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>{view, idx};
  }
}

// make distributed window (default element tensor descriptor)
template <
    class DstrTensorDesc,
    DstrTensorDesc kDstrTensorDesc,
    class TensorView,
    class Partition,
    Partition kPartition>
MINT_HOST_DEVICE constexpr auto make_distributed_window(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    integral_constant<DstrTensorDesc, kDstrTensorDesc> dstr_tensor_desc,
    integral_constant<Partition, kPartition> partition) {
  constexpr auto element_layout = make_aliased_naive_packed_tensor_descriptor(
      make_index_sequence<kDstrTensorDesc.element_ndim()>{},
      index_constant<-1>{},
      kDstrTensorDesc.element_lengths());
  return make_distributed_window(
      view, idx, dstr_tensor_desc, constant<element_layout>{}, partition);
}

// move window
template <
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
MINT_HOST_DEVICE void move_window(
    impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& win,
    const nd_index<TensorView::ndim()>& idx_delta) {
  using value_type = typename TensorView::value_type;
  const auto bot_idx_delta = move_coordinate_top_down(
      win.coord_, win.tensor_view().tensor_desc(), idx_delta);
  const index_t coord_byte_offset_delta = sizeof(value_type) * bot_idx_delta[0];
  win.coord_byte_offset_ += coord_byte_offset_delta;
}

// move window (variadic)
template <class... Tensors>
MINT_HOST_DEVICE void move_windows(
    const nd_index<
        std::tuple_element_t<0, std::tuple<Tensors...>>::dstr_tensor_desc()
            .top_ndim()>& idx_delta,
    Tensors&... wins) {
  static_assert(
      (... &&
       (Tensors::dstr_tensor_desc().top_ndim() ==
        std::tuple_element_t<0, std::tuple<Tensors...>>::dstr_tensor_desc()
            .top_ndim())),
      "All Tensors must have the same ndim");
  (move_window(wins, idx_delta), ...);
}

// move window_by_alias
template <
    alias_t alias,
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
MINT_HOST_DEVICE void move_window(
    impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& win,
    const index_t multiplier = 1) {
  static constexpr index_t top_alias_dim_index = []() {
    constexpr auto top_dim_aliases =
        TensorView::tensor_desc_type::top_dim_aliases();
    for (index_t i = 0; i < TensorView::ndim(); i++) {
      if (top_dim_aliases[i] == alias)
        return i;
    }
    return -1;
  }();
  static_assert(top_alias_dim_index != -1, "dim_alias not found!");
  nd_index<TensorView::ndim()> idx_delta;
  idx_delta.fill(0);
  idx_delta[top_alias_dim_index] =
      kDstrTensorDesc.template alias_length<alias>() * multiplier;
  move_window(win, idx_delta);
}

// move window_by_alias, default multiplier=1 (variadic)
template <alias_t alias, class... Tensors>
MINT_HOST_DEVICE void move_windows(Tensors&... wins) {
  (move_window<alias>(wins, 1), ...);
}

// move window_by_alias, multiplier specified (variadic)
template <alias_t alias, class... Tensors>
MINT_HOST_DEVICE void move_windows(const index_t multiplier, Tensors&... wins) {
  (move_window<alias>(wins, multiplier), ...);
}

namespace experimental {

namespace impl {

// impl for move window with freezed dims
template <
    auto kFreezedDimAliases, // freezed dimensions during index propagation
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
MINT_HOST_DEVICE void move_window_freezed_dim_conjectural_impl(
    ::mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& win,
    const nd_index<TensorView::ndim()>& idx_delta) {
  using tensor_desc_type =
      remove_cvref_t<decltype(win.tensor_view().tensor_desc())>;

  // sanity check kSrcFreezedDimAliases
  static_assert(
      ::std::all_of(
          kFreezedDimAliases.begin(),
          kFreezedDimAliases.end(),
          [&](auto alias) {
            return tensor_desc_type::alias_to_dim().contains(alias);
          }),
      "wrong! some alias in kFreezedDimAliases doesn't exist in tensor_desc_type");

  constexpr auto is_freezed_dims = [&]() {
    using dst_tensor_desc =
        remove_cvref_t<decltype(win.tensor_view().tensor_desc())>;
    array<bool, dst_tensor_desc::all_ndim()> ret{};
    ret.fill(false);
    for (index_t i = 0; i < kFreezedDimAliases.size(); i++)
      ret[dst_tensor_desc::alias_to_dim()[kFreezedDimAliases[i]]] = true;
    return ret;
  }();

  move_coordinate_top_down_freezed_dim_conjectural<is_freezed_dims>(
      win.coord_, win.tensor_view().tensor_desc(), idx_delta);
}

} // namespace impl

// move window with freezed dims
template <
    auto kFreezedDimAliases, // freezed dimensions during index propagation
    class TensorView,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1)
MINT_HOST_DEVICE void move_window_freezed_dim_conjectural(
    ::mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& win,
    const nd_index<TensorView::ndim()>& idx_delta) {
  impl::move_window_freezed_dim_conjectural_impl<kFreezedDimAliases>(
      win, idx_delta);
}

} // namespace experimental

#if defined(MINT_WORKAROUND_T224375980) && MINT_WORKAROUND_T224375980
namespace workaround {

// make distributed window, hack
template <
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class TensorView,
    auto kPartition>
MINT_HOST_DEVICE constexpr auto make_distributed_window_with_offset_adjustment(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    constant<kDstrTensorDesc> dstr_tensor_desc,
    constant<kElementTensorDesc> element_tensor_desc,
    constant<kPartition> partition,
    index_t offset_adjust) {
  auto win = make_distributed_window(
      view, idx, dstr_tensor_desc, element_tensor_desc, partition);
  win.coord_.bottom_index_reference()[0] += offset_adjust;
  return win;
}

} // namespace workaround
#endif

} // namespace tensor
} // namespace mint
