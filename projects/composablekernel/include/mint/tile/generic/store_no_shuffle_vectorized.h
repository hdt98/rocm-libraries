#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {
namespace experimental {

// generic scope, no_shuffle, masked
template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    auto kDstFreezedDimAliases, // freezed dimensions during index propagation
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
  requires(
      (!TensorView::is_const_tensor_view()) &&
      kDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kDstrTensorDesc.element_ndim() &&
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      is_same_v<typename TensorView::value_type, typename Memory::value_type> &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size() &&
      kDstFreezedDimAliases.size() == kDstrTensorDesc.element_ndim())
MINT_HOST_DEVICE void masked_store_no_shuffle_vectorized(
    const mint::tensor::distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& dst_win,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  using value_type = typename TensorView::value_type;
  using dst_tensor_desc_type = typename TensorView::tensor_desc_type;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kDstrTensorDesc");

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

  const TensorView& dst_tensor_view = dst_win.tensor_view();

  auto dst_coord = dst_win.coord_;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
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
    }
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
      nd_index<ndim> ret{};
      ret[0] = idx[0];
      index_t cnt = idx[0];
      for (index_t i = 1; i < ndim; i++) {
        ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
        cnt = cnt * lengths[i] + idx[i];
      }
      return ret;
    }
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

  // loop
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
        array<bool, dst_tensor_desc_type::all_ndim()> ret{};
        ret.fill(false);
        for (index_t i = 0; i < kDstFreezedDimAliases[move_dim].size(); i++)
          ret[dst_tensor_desc_type::alias_to_dim()
                  [kDstFreezedDimAliases[move_dim][i]]] = true;
        return ret;
      }();

      constexpr auto dstr_top_idx_delta =
          kDstrTensorDesc.top_index_delta(snake_idx_delta);

      ::mint::tensor::experimental::
          move_coordinate_top_down_freezed_dim_conjectural<is_freezed_dims>(
              dst_coord, dst_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    // get data from src distributed_tensor
    alignas(sizeof(value_type) * vector_size)
        vector_type<value_type, vector_size>
            v;

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

      v.template at<v_offset>() = src.memory().template at<elem_offset>();
    });

    // vector store
    if (is_unmasked(
            dst_coord, dst_tensor_view.tensor_desc(), dst_tensor_mask)) {
#if 0 // debug
      dst_tensor_view.memory_view().template as_vectors<vector_size>()
          [dst_coord.get_bottom_index()[0] / vector_size] = v;
#elif 1
      __builtin_memcpy(
          __builtin_assume_aligned(
              &(dst_tensor_view.memory_view()[dst_coord.get_bottom_index()[0]]),
              sizeof(v)),
          __builtin_assume_aligned(&v, sizeof(v)),
          sizeof(v));
#else
      static_for_n<vector_size>()([&](auto i) {
        dst_tensor_view.memory_view()[dst_coord.get_bottom_index()[0] + i] =
            v[i];
      });
#endif
    }
  });
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
  requires(
      (!TensorView::is_const_tensor_view()) &&
      kDstrTensorDesc.can_bottom_up() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kElementTensorDesc.top_ndim() == kDstrTensorDesc.element_ndim() &&
      TensorView::ndim() == kDstrTensorDesc.top_ndim() &&
      is_same_v<typename TensorView::value_type, typename Memory::value_type> &&
      kElementVectorDimAliases.size() == kElementVectorLengths.size())
MINT_HOST_DEVICE void masked_store_no_shuffle_vectorized(
    const mint::tensor::distributed_window<
        TensorView,
        kDstrTensorDesc,
        kElementTensorDesc,
        kPartition>& dst_win,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  using value_type = typename TensorView::value_type;
  using dst_tensor_desc_type [[maybe_unused]] =
      typename TensorView::tensor_desc_type;

  // sanity check kElementVectorDimAliases
  static_assert(
      ::std::all_of(
          kElementVectorDimAliases.begin(),
          kElementVectorDimAliases.end(),
          [&](auto alias) {
            return kDstrTensorDesc.alias_to_element_dim().contains(alias);
          }),
      "wrong! some alias in kElementVectorDimAliases doesn't exist in kDstrTensorDesc");

  const TensorView& dst_tensor_view = dst_win.tensor_view();

  auto dst_coord = dst_win.coord_;

  //
  constexpr auto get_old_idx = []<index_t ndim>(
                                   const nd_index<ndim>& lengths,
                                   const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
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
    }
  };

  //
  constexpr auto get_snake_idx = []<index_t ndim>(
                                     const nd_index<ndim>& lengths,
                                     const nd_index<ndim>& idx) {
    if constexpr (ndim == 0) {
      return nd_index<0>{};
    } else {
      nd_index<ndim> ret{};
      ret[0] = idx[0];
      index_t cnt = idx[0];
      for (index_t i = 1; i < ndim; i++) {
        ret[i] = (cnt % 2 == 0) ? idx[i] : lengths[i] - 1 - idx[i];
        cnt = cnt * lengths[i] + idx[i];
      }
      return ret;
    }
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

  // loop
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
          dst_coord, dst_tensor_view.tensor_desc(), dstr_top_idx_delta);
    }

    // get data from src distributed_tensor
    alignas(sizeof(value_type) * vector_size)
        vector_type<value_type, vector_size>
            v;

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

      v.template at<v_offset>() = src.memory().template at<elem_offset>();
    });

    // vector store
    if (is_unmasked(
            dst_coord, dst_tensor_view.tensor_desc(), dst_tensor_mask)) {
#if 0 // debug
      dst_tensor_view.memory_view().template as_vectors<vector_size>()
          [dst_coord.get_bottom_index()[0] / vector_size] = v;
#elif 1
      __builtin_memcpy(
          __builtin_assume_aligned(
              &(dst_tensor_view.memory_view()[dst_coord.get_bottom_index()[0]]),
              sizeof(v)),
          __builtin_assume_aligned(&v, sizeof(v)),
          sizeof(v));
#else
      static_for_n<vector_size>()([&](auto i) {
        dst_tensor_view.memory_view()[dst_coord.get_bottom_index()[0] + i] =
            v[i];
      });
#endif
    }
  });
}

template <
    auto kElementVectorDimAliases, // dim_alias of element vector dims
    auto kElementVectorLengths, // vector_lengths of element vector dims
    class TensorView,
    class TensorMask,
    auto kDstrTensorDesc,
    auto kElementTensorDesc,
    auto kPartition,
    class Memory>
MINT_HOST_DEVICE void masked_store_no_shuffle_vectorized(
    const TensorView& view,
    const nd_index<TensorView::ndim()>& idx,
    constant<kPartition> partition,
    const TensorMask& dst_tensor_mask,
    const mint::tensor::
        distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>& src) {
  const auto dst_win = make_distributed_window(
      view,
      idx,
      constant<kDstrTensorDesc>{},
      constant<kElementTensorDesc>{},
      partition);

  masked_store_no_shuffle_vectorized<
      kElementVectorDimAliases,
      kElementVectorLengths>(dst_win, dst_tensor_mask, src);
}

} // namespace experimental
} // namespace generic
} // namespace tile
} // namespace mint
