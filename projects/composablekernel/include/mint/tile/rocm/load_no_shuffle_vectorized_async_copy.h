#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

#ifdef MINT_BACKEND_ROCM
#include <mint/tile/rocm/amd_buffer_load.h>
#endif

namespace mint {
namespace tile {
namespace generic {
namespace experimental {
namespace impl {

// generic scope, no shuffle
template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    auto kSrcFreezedDimAliases, // freezed dimensions during index propagation
    auto kDstFreezedDimAliases, // freezed dimensions during index propagation
    class SrcTensorView,
    class SrcTensorMask,
    class DstTensorView,
    class DstTensorMask,
    auto kSrcDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    auto kWindowLengths>
  requires(
      kSrcDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kSrcDstrTensorDesc.element_ndim() &&
      SrcTensorView::ndim() == kSrcDstrTensorDesc.top_ndim() &&
      DstTensorView::ndim() == kSrcDstrTensorDesc.top_ndim() &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size() &&
      kSrcFreezedDimAliases.size() == kSrcDstrTensorDesc.element_ndim() &&
      kDstFreezedDimAliases.size() == kSrcDstrTensorDesc.element_ndim())
MINT_HOST_DEVICE void masked_load_no_shuffle_vectorized_async_copy_impl(
    const SrcTensorView& src_tensor_view,
    const SrcTensorMask& src_tensor_mask,
    const typename mint::tensor::impl::distributed_window_impl<
        SrcTensorView,
        kSrcDstrTensorDesc,
        kElementTensorDesc,
        kPartition>::coord_type& src_coord_in,
    const DstTensorView& dst_tensor_view,
    const DstTensorMask& /*dst_tensor_mask*/,
    const typename mint::tensor::impl::
        window_impl<DstTensorView, kWindowLengths>::coord_type& dst_coord_in) {
  using namespace mint::tensor;
  static_assert(is_same_v<
                typename SrcTensorView::value_type,
                typename DstTensorView::value_type>);
  using value_type [[maybe_unused]] = typename SrcTensorView::value_type;

  using src_tensor_desc_type =
      remove_cvref_t<decltype(src_tensor_view.tensor_desc())>;

  using dst_tensor_desc_type =
      remove_cvref_t<decltype(dst_tensor_view.tensor_desc())>;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kSrcDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kSrcDstrTensorDesc");

  // sanity check kSrcFreezedDimAliases
  static_for_n<kSrcFreezedDimAliases.size()>()([&](auto i) {
    static_assert(
        ::std::all_of(
            kSrcFreezedDimAliases[i].begin(),
            kSrcFreezedDimAliases[i].end(),
            [&](auto alias) {
              return src_tensor_desc_type::alias_to_dim().contains(alias);
            }),
        "wrong! some alias in kSrcFreezedDimAliases doesn't exist in src_tensor_desc_type");
  });

  // sanity check kDstFreezedDimAliases
  static_for_n<kDstFreezedDimAliases.size()>()([&](auto i) {
    static_assert(
        ::std::all_of(
            kDstFreezedDimAliases[i].begin(),
            kDstFreezedDimAliases[i].end(),
            [&](auto alias) {
              return dst_tensor_desc_type::alias_to_dim().contains(alias);
            }),
        "wrong! some alias in kDstFreezedDimAliases doesn't exist in dst_tensor_desc_type");
  });

  auto src_coord = src_coord_in;
  auto dst_coord = dst_coord_in;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
    auto ret = idx;
    ret[ndim - 1]--;
    bool borrow = ret[ndim - 1] < 0;
    for (index_t i = ndim - 1; i > 0; i--) {
      if (borrow) {
        ret[i] = lengths[i] - 1;
        ret[i - 1]--;
        borrow = ret[i - 1] < 0;
      }
    }
    return ret;
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    nd_index<ndim> ret{};
    ret[0] = idx[0];
    index_t cnt = idx[0];
    for (index_t i = 1; i < ndim; i++) {
      ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
      cnt = cnt * lengths[i] + idx[i];
    }
    return ret;
  };

  constexpr index_t elem_top_ndim = kElementTensorDesc.top_ndim();
  constexpr index_t elem_top_vector_ndim = kElementVectorDimAliases.size();

  constexpr auto elem_top_vector_dims = [&]() {
    nd_index<elem_top_vector_ndim> ret;
    for (index_t i = 0; i < elem_top_vector_ndim; i++)
      ret[i] = kSrcDstrTensorDesc
                   .alias_to_element_dim()[kElementVectorDimAliases[i]];
    return ret;
  }();

  constexpr auto elem_top_inner_lengths = [&]() {
    nd_index<elem_top_ndim> ret;
    ret.fill(1);
    for (index_t i = 0; i < elem_top_vector_ndim; i++)
      ret[elem_top_vector_dims[i]] = kElementVectorLengths[i];
    return ret;
  }();

  constexpr auto elem_top_outter_lengths =
      kSrcDstrTensorDesc.element_lengths() / elem_top_inner_lengths;

  constexpr index_t vector_size = ::std::accumulate(
      elem_top_inner_lengths.begin(),
      elem_top_inner_lengths.end(),
      1,
      ::std::multiplies<index_t>{});

  static_for_nd2<elem_top_outter_lengths>()([&](auto... is) {
    constexpr auto outter_idx = nd_index<elem_top_ndim>{is...};
    constexpr auto old_outter_idx =
        get_old_idx(elem_top_outter_lengths, outter_idx);
    constexpr auto snake_outter_idx =
        get_snake_idx(elem_top_outter_lengths, outter_idx);
    constexpr auto snake_old_outter_idx =
        get_snake_idx(elem_top_outter_lengths, old_outter_idx);

    constexpr auto snake_idx = snake_outter_idx * elem_top_inner_lengths;
    constexpr auto snake_old_idx =
        snake_old_outter_idx * elem_top_inner_lengths;

    // move forward
    if constexpr (outter_idx != nd_index<elem_top_ndim>{}.fill(0)) {
      constexpr auto snake_idx_delta = snake_idx - snake_old_idx;

      constexpr index_t move_dim_tmp = [&]() {
        index_t ret = -1;
        for (index_t i = 0; i < kSrcDstrTensorDesc.element_ndim(); i++) {
          if (snake_idx_delta[i] != 0) {
            ret = i;
            continue;
          }
        }
        return ret;
      }();

      static_assert(move_dim_tmp >= 0);

      constexpr auto move_dim = constant<move_dim_tmp>{};

      constexpr auto is_freezed_dims_src = [&]() {
        array<bool, src_tensor_desc_type::all_ndim()> ret{};
        ret.fill(false);
        for (index_t i = 0; i < kSrcFreezedDimAliases[move_dim].size(); i++)
          ret[src_tensor_desc_type::alias_to_dim()
                  [kSrcFreezedDimAliases[move_dim][i]]] = true;
        return ret;
      }();

      constexpr auto is_freezed_dims_dst = [&]() {
        array<bool, dst_tensor_desc_type::all_ndim()> ret{};
        ret.fill(false);
        for (index_t i = 0; i < kDstFreezedDimAliases[move_dim].size(); i++)
          ret[dst_tensor_desc_type::alias_to_dim()
                  [kDstFreezedDimAliases[move_dim][i]]] = true;
        return ret;
      }();

      constexpr auto src_top_idx_delta =
          kSrcDstrTensorDesc.top_index_delta(snake_idx_delta);

      ::mint::tensor::experimental::
          move_coordinate_top_down_freezed_dim_conjectural<is_freezed_dims_src>(
              src_coord, src_tensor_view.tensor_desc(), src_top_idx_delta);

      ::mint::tensor::experimental::
          move_coordinate_top_down_freezed_dim_conjectural<is_freezed_dims_dst>(
              dst_coord, dst_tensor_view.tensor_desc(), src_top_idx_delta);
    }

    // FIXME: need to pass in customized value
    if (is_unmasked(
            src_coord, src_tensor_view.tensor_desc(), src_tensor_mask)) {
      {
#ifdef __gfx950__
        amd_buffer_load_async_copy<value_type, vector_size>(
            &src_tensor_view.memory_view()[0],
            &dst_tensor_view.memory_view()[0],
            src_coord.get_bottom_index()[0],
            0,
            dst_coord.get_bottom_index()[0],
            src_tensor_view.memory_view().size());
#else
        const index_t warp_id = threadIdx.x % MINT_WARP_SIZE;
        static_for_n<vector_size>()([&](auto i) {
          dst_tensor_view.memory_view()
              [dst_coord.get_bottom_index()[0] + warp_id * vector_size + i] =
              src_tensor_view
                  .memory_view()[src_coord.get_bottom_index()[0] + i];
        });
#endif
      }
    }
  });
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class SrcTensorView,
    class SrcTensorMask,
    class DstTensorView,
    class DstTensorMask,
    auto kSrcDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    auto kWindowLengths>
  requires(
      kSrcDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kSrcDstrTensorDesc.element_ndim() &&
      SrcTensorView::ndim() == kSrcDstrTensorDesc.top_ndim() &&
      DstTensorView::ndim() == kSrcDstrTensorDesc.top_ndim() &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size())
MINT_HOST_DEVICE void masked_load_no_shuffle_vectorized_async_copy_impl(
    const SrcTensorView& src_tensor_view,
    const SrcTensorMask& src_tensor_mask,
    const typename mint::tensor::impl::distributed_window_impl<
        SrcTensorView,
        kSrcDstrTensorDesc,
        kElementTensorDesc,
        kPartition>::coord_type& src_coord_in,
    const DstTensorView& dst_tensor_view,
    const DstTensorMask& dst_tensor_mask,
    const typename mint::tensor::impl::
        window_impl<DstTensorView, kWindowLengths>::coord_type& dst_coord_in) {
  using namespace mint::tensor;
  static_assert(is_same_v<
                typename SrcTensorView::value_type,
                typename DstTensorView::value_type>);
  using value_type [[maybe_unused]] = typename SrcTensorView::value_type;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kSrcDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kSrcDstrTensorDesc");

  auto src_coord = src_coord_in;
  auto dst_coord = dst_coord_in;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
    auto ret = idx;
    ret[ndim - 1]--;
    bool borrow = ret[ndim - 1] < 0;
    for (index_t i = ndim - 1; i > 0; i--) {
      if (borrow) {
        ret[i] = lengths[i] - 1;
        ret[i - 1]--;
        borrow = ret[i - 1] < 0;
      }
    }
    return ret;
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    nd_index<ndim> ret{};
    ret[0] = idx[0];
    index_t cnt = idx[0];
    for (index_t i = 1; i < ndim; i++) {
      ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
      cnt = cnt * lengths[i] + idx[i];
    }
    return ret;
  };

  constexpr index_t elem_top_ndim = kElementTensorDesc.top_ndim();
  constexpr index_t elem_top_vector_ndim = kElementVectorDimAliases.size();

  constexpr auto elem_top_vector_dims = [&]() {
    nd_index<elem_top_vector_ndim> ret;
    for (index_t i = 0; i < elem_top_vector_ndim; i++)
      ret[i] = kSrcDstrTensorDesc
                   .alias_to_element_dim()[kElementVectorDimAliases[i]];
    return ret;
  }();

  constexpr auto elem_top_inner_lengths = [&]() {
    nd_index<elem_top_ndim> ret;
    ret.fill(1);
    for (index_t i = 0; i < elem_top_vector_ndim; i++)
      ret[elem_top_vector_dims[i]] = kElementVectorLengths[i];
    return ret;
  }();

  constexpr auto elem_top_outter_lengths =
      kSrcDstrTensorDesc.element_lengths() / elem_top_inner_lengths;

  constexpr index_t vector_size = ::std::accumulate(
      elem_top_inner_lengths.begin(),
      elem_top_inner_lengths.end(),
      1,
      ::std::multiplies<index_t>{});

  static_for_nd2<elem_top_outter_lengths>()([&](auto... is) {
    constexpr auto outter_idx = nd_index<elem_top_ndim>{is...};
    constexpr auto old_outter_idx =
        get_old_idx(elem_top_outter_lengths, outter_idx);
    constexpr auto snake_outter_idx =
        get_snake_idx(elem_top_outter_lengths, outter_idx);
    constexpr auto snake_old_outter_idx =
        get_snake_idx(elem_top_outter_lengths, old_outter_idx);

    constexpr auto snake_idx = snake_outter_idx * elem_top_inner_lengths;
    constexpr auto snake_old_idx =
        snake_old_outter_idx * elem_top_inner_lengths;

    // move forward
    if constexpr (outter_idx != nd_index<elem_top_ndim>{}.fill(0)) {
      constexpr auto dstr_top_idx_delta =
          kSrcDstrTensorDesc.top_index_delta(snake_idx - snake_old_idx);
      move_coordinate(
          src_coord, src_tensor_view.tensor_desc(), dstr_top_idx_delta);
      move_coordinate(
          dst_coord, dst_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    // FIXME: need to pass in customized value
    if (is_unmasked(
            src_coord, src_tensor_view.tensor_desc(), src_tensor_mask)) {
      {
#ifdef __gfx950__
        amd_buffer_load_async_copy<value_type, vector_size>(
            &src_tensor_view.memory_view()[0],
            &dst_tensor_view.memory_view()[0],
            src_coord.get_bottom_index()[0],
            0,
            dst_coord.get_bottom_index()[0],
            src_tensor_view.memory_view().size());
#else
        const index_t warp_id = threadIdx.x % MINT_WARP_SIZE;
        static_for_n<vector_size>()([&](auto i) {
          dst_tensor_view.memory_view()
              [dst_coord.get_bottom_index()[0] + warp_id * vector_size + i] =
              src_tensor_view
                  .memory_view()[src_coord.get_bottom_index()[0] + i];
        });
#endif
      }
    }
  });
}

} // namespace impl

// generic scope, no shuffle, masked
template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    auto kSrcFreezedDimAliases, // src freezed dimensions during index
                                // propagation
    auto kDstFreezedDimAliases, // dst freezed dimensions during index
                                // propagation
    class SrcTensorView,
    class SrcTensorMask,
    class DstTensorView,
    class DstTensorMask,
    auto kSrcDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    auto kWindowLengths>
MINT_HOST_DEVICE auto masked_load_no_shuffle_vectorized_async_copy(
    const mint::tensor::impl::distributed_window_impl<
        SrcTensorView,
        kSrcDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& src_win,
    const SrcTensorMask& src_tensor_mask,
    const mint::tensor::window<DstTensorView, kWindowLengths>& dst_win,
    const DstTensorMask& dst_tensor_mask) {
  return impl::masked_load_no_shuffle_vectorized_async_copy_impl<
      kElementVectorDimAliases,
      kElementVectorLengths,
      kSrcFreezedDimAliases,
      kDstFreezedDimAliases,
      SrcTensorView,
      SrcTensorMask,
      DstTensorView,
      DstTensorMask,
      kSrcDstrTensorDesc,
      kElementTensorDesc,
      kPartition,
      kWindowLengths>(
      src_win.tensor_view(),
      src_tensor_mask,
      src_win.coord_,
      dst_win.tensor_view(),
      dst_tensor_mask,
      dst_win.coord_);
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class SrcTensorView,
    class SrcTensorMask = tuple<>,
    class DstTensorView,
    class DstTensorMask = tuple<>,
    auto kSrcDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition>
MINT_HOST_DEVICE auto masked_load_no_shuffle_vectorized_async_copy(
    const SrcTensorView& src_view,
    const nd_index<SrcTensorView::ndim()>& src_idx,
    mint::tensor::
        distributed_tensor<kSrcDstrTensorDesc, kElementTensorDesc, Memory>&
            dstr,
    const DstTensorView& dst_view,
    const nd_index<DstTensorView::ndim()>& dst_idx,
    constant<kPartition> partition,
    const SrcTensorMask& src_tensor_mask = tuple<>{},
    const DstTensorMask& dst_tensor_mask = tuple<>{}) {
  constexpr auto dst_lengths =
      typename remove_cvref_t<DstTensorView>::tensor_desc_type{}.top_lengths();
  const auto src_win = make_distributed_window(
      src_view,
      src_idx,
      constant<kSrcDstrTensorDesc>{},
      constant<kElementTensorDesc>{},
      partition);
  auto dst_win = make_window(dst_view, dst_idx, to_sequence<dst_lengths>());
  return impl::masked_load_no_shuffle_vectorized_async_copy_impl<
      kElementVectorDimAliases,
      kElementVectorLengths,
      SrcTensorView,
      SrcTensorMask,
      DstTensorView,
      DstTensorMask,
      kSrcDstrTensorDesc,
      kElementTensorDesc,
      kPartition,
      to_sequence<dst_lengths>()>(
      src_win.tensor_view(),
      src_tensor_mask,
      src_win.coord_,
      dst_win.tensor_view(),
      dst_tensor_mask,
      dst_win.coord_);
}

} // namespace experimental
} // namespace generic
} // namespace tile
} // namespace mint
