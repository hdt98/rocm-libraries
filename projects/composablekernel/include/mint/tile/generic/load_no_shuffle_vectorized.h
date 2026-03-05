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
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      kDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kDstrTensorDesc.element_ndim() &&
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size() &&
      kSrcFreezedDimAliases.size() == kDstrTensorDesc.element_ndim())
MINT_HOST_DEVICE auto masked_load_no_shuffle_vectorized_impl(
    const TensorView& src_tensor_view,
    const TensorMask& src_tensor_mask,
    const typename mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>::coord_type& src_coord_in) {
  using namespace mint::tensor;
  using value_type = typename TensorView::value_type;
  using mem_type =
      owned_vgpr_memory<value_type, kDstrTensorDesc.element_size()>;
  using dstr_coord_type [[maybe_unused]] =
      coordinate<decltype(kDstrTensorDesc.tensor_desc())>;
  using src_tensor_desc_type =
      remove_cvref_t<decltype(src_tensor_view.tensor_desc())>;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kDstrTensorDesc");

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

  distributed_tensor<kDstrTensorDesc, kElementTensorDesc, mem_type> dst{};

  auto src_coord = src_coord_in;

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
      ret[i] =
          kDstrTensorDesc.alias_to_element_dim()[kElementVectorDimAliases[i]];
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
      kDstrTensorDesc.element_lengths() / elem_top_inner_lengths;

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
        for (index_t i = 0; i < kDstrTensorDesc.element_ndim(); i++) {
          if (snake_idx_delta[i] != 0) {
            ret = i;
            continue;
          }
        }
        return ret;
      }();

      static_assert(move_dim_tmp >= 0);

      constexpr auto move_dim = constant<move_dim_tmp>{};

      constexpr auto is_freezed_dims = [&]() {
        array<bool, src_tensor_desc_type::all_ndim()> ret{};
        ret.fill(false);
        for (index_t i = 0; i < kSrcFreezedDimAliases[move_dim].size(); i++)
          ret[src_tensor_desc_type::alias_to_dim()
                  [kSrcFreezedDimAliases[move_dim][i]]] = true;
        return ret;
      }();

      constexpr auto src_top_idx_delta =
          kDstrTensorDesc.top_index_delta(snake_idx_delta);

      ::mint::tensor::experimental::
          move_coordinate_top_down_freezed_dim_conjectural<is_freezed_dims>(
              src_coord, src_tensor_view.tensor_desc(), src_top_idx_delta);
    }

    // vector load
    alignas(sizeof(value_type) * vector_size)
        vector_type<value_type, vector_size>
            v;

    // FIXME: need to pass in customized value
    v.fill(static_cast<value_type>(0.f));
    if (is_unmasked(
            src_coord, src_tensor_view.tensor_desc(), src_tensor_mask)) {
#ifdef MINT_BACKEND_ROCM
      if constexpr (TensorView::address_space() == address_space::global) {
        v = amd_buffer_load_invalid_element_return_zero<
            value_type,
            vector_size>(
            &src_tensor_view.memory_view()[0],
            src_coord.get_bottom_index()[0],
            src_tensor_view.memory_view().size());
      } else
#endif
      {
#if 0 // debug
        v = src_tensor_view.memory_view().template as_vectors<vector_size>()
                [src_coord.get_bottom_index()[0] / vector_size];
#elif 1
        __builtin_memcpy(
            __builtin_assume_aligned(&v, sizeof(v)),
            __builtin_assume_aligned(
                &(src_tensor_view
                      .memory_view()[src_coord.get_bottom_index()[0]]),
                sizeof(v)),
            sizeof(v));
#else
        static_for_n<vector_size>()([&](auto i) {
          v[i] = src_tensor_view
                     .memory_view()[src_coord.get_bottom_index()[0] + i];
        });
#endif
      }
    }

    static_for_nd2<elem_top_inner_lengths>()([&](auto... js) {
      constexpr auto inner_idx = nd_index<elem_top_ndim>{js...};
      constexpr auto idx = snake_idx + inner_idx;

      constexpr index_t elem_offset =
          kElementTensorDesc.calculate_bottom_index(idx)[0];

      constexpr auto v_idx = inner_idx.template get_subset<
          elem_top_vector_dims.size(),
          elem_top_vector_dims>();

      constexpr index_t v_offset = [&]() {
        index_t ret = v_idx[0];
        for (index_t i = 1; i < elem_top_vector_ndim; i++)
          ret = ret * kElementVectorLengths[i] + v_idx[i];
        return ret;
      }();

      // put data into dst distributed_tensor
      dst.memory().template at<elem_offset>() = v.template at<v_offset>();
    });
  });

  return dst;
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
  requires(
      kDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kDstrTensorDesc.element_ndim() &&
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size())
MINT_HOST_DEVICE auto masked_load_no_shuffle_vectorized_impl(
    const TensorView& src_tensor_view,
    const TensorMask& src_tensor_mask,
    const typename mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>::coord_type& src_coord_in) {
  using namespace mint::tensor;
  using value_type = typename TensorView::value_type;
  using mem_type =
      owned_vgpr_memory<value_type, kDstrTensorDesc.element_size()>;
  using dstr_coord_type [[maybe_unused]] =
      coordinate<decltype(kDstrTensorDesc.tensor_desc())>;
  using src_tensor_desc_type [[maybe_unused]] =
      remove_cvref_t<decltype(src_tensor_view.tensor_desc())>;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kDstrTensorDesc");

  distributed_tensor<kDstrTensorDesc, kElementTensorDesc, mem_type> dst{};

  auto src_coord = src_coord_in;

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
      ret[i] =
          kDstrTensorDesc.alias_to_element_dim()[kElementVectorDimAliases[i]];
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
      kDstrTensorDesc.element_lengths() / elem_top_inner_lengths;

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
          kDstrTensorDesc.top_index_delta(snake_idx - snake_old_idx);
      move_coordinate(
          src_coord, src_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    // vector load
    alignas(sizeof(value_type) * vector_size)
        vector_type<value_type, vector_size>
            v;

    // FIXME: need to pass in customized value
    v.fill((value_type)0.f);
    if (is_unmasked(
            src_coord, src_tensor_view.tensor_desc(), src_tensor_mask)) {
#ifdef MINT_BACKEND_ROCM
      if constexpr (TensorView::address_space() == address_space::global) {
        v = amd_buffer_load_invalid_element_return_zero<
            value_type,
            vector_size>(
            &src_tensor_view.memory_view()[0],
            src_coord.get_bottom_index()[0],
            src_tensor_view.memory_view().size());
      } else
#endif
      {
#if 0 // debug
        v = src_tensor_view.memory_view().template as_vectors<vector_size>()
                [src_coord.get_bottom_index()[0] / vector_size];
#elif 1
        __builtin_memcpy(
            __builtin_assume_aligned(&v, sizeof(v)),
            __builtin_assume_aligned(
                &(src_tensor_view
                      .memory_view()[src_coord.get_bottom_index()[0]]),
                sizeof(v)),
            sizeof(v));
#else
        static_for_n<vector_size>()([&](auto i) {
          v[i] = src_tensor_view
                     .memory_view()[src_coord.get_bottom_index()[0] + i];
        });
#endif
      }
    }

    static_for_nd2<elem_top_inner_lengths>()([&](auto... js) {
      constexpr auto inner_idx = nd_index<elem_top_ndim>{js...};
      constexpr auto idx = snake_idx + inner_idx;

      constexpr index_t elem_offset =
          kElementTensorDesc.calculate_bottom_index(idx)[0];

      constexpr auto v_idx = inner_idx.template get_subset<
          elem_top_vector_dims.size(),
          elem_top_vector_dims>();

      constexpr index_t v_offset = [&]() {
        index_t ret = v_idx[0];
        for (index_t i = 1; i < elem_top_vector_ndim; i++)
          ret = ret * kElementVectorLengths[i] + v_idx[i];
        return ret;
      }();

      // put data into dst distributed_tensor
      dst.memory().template at<elem_offset>() = v.template at<v_offset>();
    });
  });

  return dst;
}

} // namespace impl

// generic scope, no shuffle, masked
template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    auto kSrcFreezedDimAliases, // freezed dimensions during index propagation
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition>
MINT_HOST_DEVICE auto masked_load_no_shuffle_vectorized(
    const mint::tensor::impl::distributed_window_impl<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& src_win,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  return impl::masked_load_no_shuffle_vectorized_impl<
      kElementVectorDimAliases,
      kElementVectorLengths,
      kSrcFreezedDimAliases,
      TensorView,
      TensorMask,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>(src_win.tensor_view(), src_tensor_mask, src_win.coord_);
}

// generic scope, no shuffle, masked
template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    class TensorMask = tuple<>,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    class Memory,
    auto kPartition>
MINT_HOST_DEVICE void masked_load_no_shuffle_vectorized(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& dstr,
    constant<kPartition> partition,
    const TensorMask& src_tensor_mask = tuple<>{}) {
  const auto src_win = make_distributed_window(
      view,
      idx,
      constant<kDstrTensorDesc>{},
      constant<kElementTensorDesc>{},
      partition);
  dstr = impl::masked_load_no_shuffle_vectorized_impl<
      kElementVectorDimAliases,
      kElementVectorLengths,
      TensorView,
      TensorMask,
      kDstrTensorDesc,
      kElementTensorDesc,
      kPartition>(src_win.tensor_view(), src_tensor_mask, src_win.coord_);
}

} // namespace experimental
} // namespace generic
} // namespace tile
} // namespace mint
