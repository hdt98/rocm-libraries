#pragma once
#include <mint/config.h>
#include <mint/core.h>
#include <mint/poly/polymorpher_helper.h>
#include <mint/poly/z2_linear_morpher_helper.h>
#include <mint/tensor/coordinate.h>
#include <mint/tensor/tensor_descriptor.h>

namespace mint {
namespace tensor {

namespace impl {

template <class TensorDesc, class MemoryView>
  requires(TensorDesc::bottom_ndim() == 1)
struct tensor_view_impl {
  using tensor_desc_type = remove_cvref_t<TensorDesc>;
  using memory_view_type = remove_cvref_t<MemoryView>;
  using value_type = remove_cvref_t<typename MemoryView::value_type>;

  MINT_HOST_DEVICE constexpr tensor_view_impl(
      const tensor_desc_type& tensor_desc,
      const memory_view_type& mem_view)
      : tensor_desc_{tensor_desc}, mem_view_{mem_view} {}

  MINT_HOST_DEVICE static consteval index_t ndim() {
    return tensor_desc_type::top_ndim();
  }

  MINT_HOST_DEVICE constexpr auto lengths() const {
    return tensor_desc_.top_lengths();
  }

  MINT_HOST_DEVICE constexpr const tensor_desc_type& tensor_desc() const {
    return tensor_desc_;
  }

  MINT_HOST_DEVICE static consteval address_space address_space() {
    return memory_view_type::address_space();
  }

  MINT_HOST_DEVICE constexpr const memory_view_type& memory_view() const {
    return mem_view_;
  }

#if 1 // debug
  const tensor_desc_type tensor_desc_;
  const memory_view_type mem_view_;
#else
  // FIXME: dangerous to use reference version since the object it refers to may
  // go out of life span, but it use less registers. need a compiler fix so we
  // can use the non-reference version
  const tensor_desc_type& tensor_desc_;
  const memory_view_type& mem_view_;
#endif
};

} // namespace impl

template <class TensorDesc, class MemoryView>
  requires(TensorDesc::bottom_ndim() == 1)
struct tensor_view
    : impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>> {
  using base = impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>>;
  using tensor_desc_type = base::tensor_desc_type;
  using value_type = base::value_type;

  using impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>>::
      tensor_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_tensor_view() {
    return false;
  }

  MINT_HOST_DEVICE constexpr index_t calculate_offset(
      const nd_index<base::ndim()>& idx) const {
    return base::tensor_desc().calculate_bottom_index(idx)[0];
  }

  MINT_HOST_DEVICE value_type& element(
      const nd_index<base::ndim()>& idx) const {
    return base::memory_view()[calculate_offset(idx)];
  }

  MINT_HOST_DEVICE value_type& element(
      const coordinate<tensor_desc_type>& coord) const {
    return base::memory_view()[coord.get_bottom_index()[0]];
  }
};

template <class TensorDesc, class MemoryView>
  requires(TensorDesc::bottom_ndim() == 1)
struct const_tensor_view
    : impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>> {
  using base = impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>>;
  using tensor_desc_type = base::tensor_desc_type;
  using value_type = base::value_type;

  using impl::tensor_view_impl<TensorDesc, remove_cvref_t<MemoryView>>::
      tensor_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_tensor_view() {
    return true;
  }

  MINT_HOST_DEVICE const value_type& element(
      const nd_index<base::ndim()>& idx) const {
    return base::memory_view()[base::tensor_desc().calculate_bottom_index(
        idx)[0]];
  }

  MINT_HOST_DEVICE const value_type& element(
      const coordinate<tensor_desc_type>& coord) const {
    return base::memory_view()[coord.get_bottom_index()[0]];
  }
};

template <class TensorDesc, class MemoryView>
  requires(not MemoryView::is_const_memory_view())
MINT_HOST_DEVICE constexpr auto make_tensor_view(
    const TensorDesc& desc,
    const MemoryView& mem_view) {
  return tensor_view<TensorDesc, remove_cvref_t<MemoryView>>{desc, mem_view};
}

template <class TensorDesc, class MemoryView>
  requires(MemoryView::is_const_memory_view())
MINT_HOST_DEVICE constexpr auto make_tensor_view(
    const TensorDesc& desc,
    const MemoryView& mem_view) {
  return const_tensor_view<TensorDesc, remove_cvref_t<MemoryView>>{
      desc, mem_view};
}

// Type traits to check if a type is tensor_view or const_tensor_view
namespace impl {
template <typename T>
struct is_tensor_view_type : std::false_type {};

template <class TensorDesc, class MemoryView>
struct is_tensor_view_type<tensor_view<TensorDesc, MemoryView>>
    : std::true_type {};

template <class TensorDesc, class MemoryView>
struct is_tensor_view_type<const_tensor_view<TensorDesc, MemoryView>>
    : std::true_type {};
} // namespace impl

template <typename T>
concept is_tensor_view = impl::is_tensor_view_type<remove_cvref_t<T>>::value;

// Helper trait to check if a type has exactly one z2 morpher
namespace impl {
template <typename TensorView>
constexpr bool check_single_z2_morpher() {
  using tensor_desc_t = typename remove_cvref_t<TensorView>::tensor_desc_type;
  using polymorpher_t =
      remove_cvref_t<decltype(std::declval<tensor_desc_t>().polymorpher())>;

  if constexpr (polymorpher_t::num_morpher() == 1) {
    using morpher_t = remove_cvref_t<
        decltype(std::declval<polymorpher_t>().morphers()[0_ic])>;
    return poly::is_z2_linear_morpher<morpher_t>::value;
  }

  return false;
}

template <class TensorView, class TransformOp, class... Args>
MINT_HOST_DEVICE constexpr auto transform_tensor_view_z2(
    TensorView& tensor_view,
    Args&&... args) {
  auto new_polymorpher = transform_z2_polymorpher(
      tensor_view.tensor_desc_.polymorpher(),
      std::forward<TransformOp>(TransformOp{}),
      std::forward<Args>(args)...);
  auto desc = make_tensor_descriptor(new_polymorpher);
  const auto mem_view = tensor_view.memory_view();
  return make_tensor_view(desc, mem_view);
}
} // namespace impl

template <typename T>
concept is_single_z2_morpher_tensor_view = impl::check_single_z2_morpher<T>();
template <class TensorView, class... Args>
  requires is_tensor_view<TensorView> and
    is_single_z2_morpher_tensor_view<TensorView>
MINT_HOST_DEVICE constexpr auto reshape_logical(
    TensorView& tensor_view,
    Args&&... args) {
  return impl::
      transform_tensor_view_z2<TensorView, poly::reshape_top_z2, Args...>(
          tensor_view, std::forward<Args>(args)...);
}

template <class TensorView, class... Args>
  requires is_tensor_view<TensorView> and
    is_single_z2_morpher_tensor_view<TensorView>
MINT_HOST_DEVICE constexpr auto reorder_logical(
    TensorView& tensor_view,
    Args&&... args) {
  return impl::
      transform_tensor_view_z2<TensorView, poly::reorder_top_z2, Args...>(
          tensor_view, std::forward<Args>(args)...);
}

template <class TensorView, class... Args>
  requires is_tensor_view<TensorView> and
    is_single_z2_morpher_tensor_view<TensorView>
MINT_HOST_DEVICE constexpr auto swizzle_logical(
    TensorView& tensor_view,
    Args&&... args) {
  return impl::
      transform_tensor_view_z2<TensorView, poly::swizzle_top_z2, Args...>(
          tensor_view, std::forward<Args>(args)...);
}

} // namespace tensor
} // namespace mint
