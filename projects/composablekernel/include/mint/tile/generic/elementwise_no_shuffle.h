#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {

template <class... Tensors, class FElementwise>
MINT_HOST_DEVICE void elementwise_no_shuffle(
    const FElementwise& f_elementwise,
    Tensors&... tensors) {
  // FIXME : amolak : Enforce distribution is the same on all tensors
  using InTensorType = std::tuple_element_t<0, std::tuple<Tensors...>>;
  constexpr index_t ndim = InTensorType::dstr_tensor_desc().top_ndim();

  // FIXME : amolak : Hardcoded 1D and 2D  inputs/outputs, need to figure out
  // how to generalize an ND loop access.
  if constexpr (ndim == 1) {
    constexpr auto kM =
        InTensorType::dstr_tensor_desc().sharded_lengths()[0_ic];

    static_for_nd3<kM>()([&](auto iM) {
      constexpr auto idx = mint::make_tuple(iM);
      f_elementwise(tensors.sharded_element(idx)...);
    });
  } else if constexpr (ndim == 2) {
    constexpr auto kM =
        InTensorType::dstr_tensor_desc().sharded_lengths()[0_ic];
    constexpr auto kN =
        InTensorType::dstr_tensor_desc().sharded_lengths()[1_ic];

    static_for_nd3<kM>()([&](auto iM) {
      static_for_nd3<kN>()([&](auto iN) {
        constexpr auto idx = mint::make_tuple(iM, iN);
        f_elementwise(tensors.sharded_element(idx)...);
      });
    });
  } else {
    static_assert("Currently support only 1D and 2D tensors");
  }
}

} // namespace generic
} // namespace tile
} // namespace mint
