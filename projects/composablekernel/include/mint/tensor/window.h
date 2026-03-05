#pragma once
#include <mint/core.h>
#include <mint/poly.h>

namespace mint {
namespace tensor {
namespace impl {

template <class TensorView, auto kWindowLengths>
  requires(TensorView::ndim() == kWindowLengths.size())
struct window_impl {
  using tensor_view_type = remove_cvref_t<TensorView>;
  using tensor_desc_type =
      remove_cvref_t<typename TensorView::tensor_desc_type>;
  using coord_type = coordinate<tensor_desc_type>;
  using value_type = typename tensor_view_type::value_type;

 private:
#if 1 // debug
  const tensor_view_type tensor_view_;
#else
  // FIXME: dangerous to use reference version since the object it refers to may
  // go out of life span, but it use less registers. need a compiler fix so we
  // can use the non-reference version
  const tensor_view_type& tensor_view_;
#endif

 public:
  coord_type coord_;
  MINT_HOST_DEVICE static consteval index_t ndim() {
    return tensor_desc_type::top_ndim();
  }

  MINT_HOST_DEVICE static consteval auto lengths() {
    return kWindowLengths;
  }

  MINT_HOST_DEVICE constexpr window_impl(
      const tensor_view_type& view,
      const nd_index<ndim()>& idx)
      : tensor_view_{view}, coord_{view.tensor_desc(), idx} {}

  MINT_HOST_DEVICE constexpr window_impl(
      const tensor_view_type& view,
      const coord_type& coord)
      : tensor_view_{view}, coord_{coord} {}

  MINT_HOST_DEVICE const tensor_view_type& tensor_view() const {
    return tensor_view_;
  }

  MINT_HOST_DEVICE const coord_type& coordinate() const {
    return coord_;
  }
};

} // namespace impl

// window (non-const)
template <class TensorView, auto kWindowLengths>
struct window : impl::window_impl<remove_cvref_t<TensorView>, kWindowLengths> {
  using impl::window_impl<remove_cvref_t<TensorView>, kWindowLengths>::
      window_impl;
};

// window (const)
template <class TensorView, auto kWindowLengths>
struct const_window
    : impl::window_impl<const remove_cvref_t<TensorView>, kWindowLengths> {
  using impl::window_impl<const remove_cvref_t<TensorView>, kWindowLengths>::
      window_impl;
};

// make_window (non-const)
template <class TensorView, index_t... kWindowLengths>
  requires(!TensorView::is_const_tensor_view())
MINT_HOST_DEVICE constexpr auto make_window(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    index_sequence<kWindowLengths...>) {
  constexpr auto lengths =
      nd_index<sizeof...(kWindowLengths)>{kWindowLengths...};
  return window<remove_cvref_t<TensorView>, lengths>{view, idx};
}

// make_window (const)
template <class TensorView, index_t... kWindowLengths>
  requires(TensorView::is_const_tensor_view())
MINT_HOST_DEVICE constexpr auto make_window(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    index_sequence<kWindowLengths...>) {
  constexpr auto lengths =
      nd_index<sizeof...(kWindowLengths)>{kWindowLengths...};
  return const_window<remove_cvref_t<TensorView>, lengths>{view, idx};
}

// move_window (non-const)
template <class TensorView, auto kWindowLengths>
MINT_HOST_DEVICE constexpr void move_window(
    window<TensorView, kWindowLengths>& win,
    const nd_index<TensorView::ndim()>& idx_delta) {
  move_coordinate_top_down(
      win.coord_, win.tensor_view().tensor_desc(), idx_delta);
}

// move_window (const)
template <class TensorView, auto kWindowLengths>
MINT_HOST_DEVICE constexpr void move_window(
    const_window<TensorView, kWindowLengths>& win,
    const nd_index<TensorView::ndim()>& idx_delta) {
  move_coordinate_top_down(
      win.coord_, win.tensor_view().tensor_desc(), idx_delta);
}

#if defined(MINT_WORKAROUND_T224375980) && MINT_WORKAROUND_T224375980
namespace workaround {

// make_window (non-const), hack
template <class TensorView, index_t... kWindowLengths>
  requires(!TensorView::is_const_tensor_view())
MINT_HOST_DEVICE constexpr auto make_window_with_offset_adjustment(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    index_sequence<kWindowLengths...>,
    index_t offset_adjust) {
  constexpr auto lengths =
      nd_index<sizeof...(kWindowLengths)>{kWindowLengths...};
  using win_type = window<remove_cvref_t<TensorView>, lengths>;

  auto coord = typename win_type::coord_type{view.tensor_desc(), idx};
  coord.bottom_index_reference()[0] += offset_adjust;

  return win_type{view, coord};
}

// make_window (const), hack
template <class TensorView, index_t... kWindowLengths>
  requires(TensorView::is_const_tensor_view())
MINT_HOST_DEVICE constexpr auto make_window_with_offset_adjustment(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    index_sequence<kWindowLengths...>,
    index_t offset_adjust) {
  constexpr auto lengths =
      nd_index<sizeof...(kWindowLengths)>{kWindowLengths...};
  using win_type = const_window<remove_cvref_t<TensorView>, lengths>;

  auto coord = typename win_type::coord_type{view.tensor_desc(), idx};
  coord.bottom_index_reference()[0] += offset_adjust;

  return win_type{view, coord};
}

} // namespace workaround
#endif

} // namespace tensor
} // namespace mint
