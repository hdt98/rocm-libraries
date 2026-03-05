#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {

template <
    class OutTensor,
    class InTensor,
    class DimAlias,
    DimAlias... kReduceDimAliases,
    class FReduce,
    bool kReduceWithOut>
MINT_HOST_DEVICE void reduce_no_shuffle(
    OutTensor& out,
    const InTensor& in,
    sequence<DimAlias, kReduceDimAliases...>,
    const FReduce& f_reduce,
    bool_constant<kReduceWithOut>) {
  using OutValue = OutTensor::value_type;
  using InValue = InTensor::value_type;

  constexpr auto reduce_dim_aliases =
      sequence<DimAlias, kReduceDimAliases...>{};

  constexpr auto reduce_dims = [&]() {
    nd_index<sizeof...(kReduceDimAliases)> ret;
    constexpr auto in_dstr = InTensor::dstr_tensor_desc();

    for (index_t i = 0; i < ret.size(); i++) {
      index_t dim = in_dstr.tensor_desc()
                        .polymorpher()
                        .alias_to_dim()[reduce_dim_aliases[i]];
      for (index_t j = 0; j < in_dstr.top_ndim(); j++)
        if (in_dstr.tensor_desc().top_dims()[j] == dim) {
          ret[i] = j;
          break;
        }
    }
    return ret;
  }();

  // FIXME: hardcoded for 2d-to-1d reduction
  // FIXME: not supporting reduce with OutTensor value
  static_assert(
      InTensor::dstr_tensor_desc().top_ndim() == 2 &&
          OutTensor::dstr_tensor_desc().top_ndim() == 1,
      "currently only support 2D-to-1D reduction");
  static_assert(reduce_dims[0] == 1, "currently only support reduce_dim = 1");

  constexpr auto in_sharded_lengths =
      InTensor::dstr_tensor_desc().sharded_lengths();
  constexpr auto kMs = in_sharded_lengths[0_ic];
  constexpr auto kNs = in_sharded_lengths[1_ic];

  if constexpr (kReduceWithOut) {
    static_for_nd3<kMs>()([&](auto iMs) {
      constexpr auto m_idx = mint::make_tuple(iMs);
      auto acc = out.sharded_element(m_idx);
      static_for_nd3<kNs>()([&](auto iNs) {
        constexpr auto m_n_idx = mint::make_tuple(iMs, iNs);
        acc = f_reduce(acc, in.sharded_element(m_n_idx));
      });
      out.sharded_element(m_idx) = acc;
    });
  } else {
    constexpr auto all_zero = [](auto seq) {
      bool ret = true;
      for (index_t i = 0; i < remove_cvref_t<decltype(seq)>::size(); i++)
        ret &= (seq[i] == 0);
      return ret;
    };

    static_for_nd3<kMs>()([&](auto iMs) {
      constexpr auto m_idx = mint::make_tuple(iMs);
      OutValue acc{};
      static_for_nd3<kNs>()([&](auto iNs) {
        constexpr auto m_n_idx = mint::make_tuple(iMs, iNs);
        const InValue val = in.sharded_element(m_n_idx);
        if constexpr (all_zero(iNs))
          acc = val;
        else
          acc = f_reduce(acc, val);
      });
      out.sharded_element(m_idx) = acc;
    });
  }
}

} // namespace generic
} // namespace tile
} // namespace mint
