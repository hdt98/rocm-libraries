#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {
namespace tile {
namespace generic {

// C[M, N] += A[M, K] * B[K, N]
// generic scope, no shuffle
template <class CTensor, class ATensor, class BTensor>
MINT_HOST_DEVICE void
matmul_mn_mk_kn_no_shuffle(CTensor& c, const ATensor& a, const BTensor& b) {
  // TODO: check a/c have same m-sharding
  // TODO: check b/c have same n-sharding
  // TODO: check a/b have same k-sharding
  // TODO: check a/b own full k
  //
  using c_value_type [[maybe_unused]] = typename CTensor::value_type;
  using a_value_type [[maybe_unused]] = typename ATensor::value_type;
  using b_value_type [[maybe_unused]] = typename BTensor::value_type;

  constexpr auto kMs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<0>();
  constexpr auto kNs =
      BTensor::dstr_tensor_desc().sharded_lengths().template at<1>();
  constexpr auto kKs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<1>();

  static_for_nd3<kMs>()([&](auto iMs) {
    static_for_nd3<kNs>()([&](auto iNs) {
      constexpr auto m_n_idx = mint::make_tuple(iMs, iNs);

      static_for_nd3<kKs>()([&](auto iKs) {
        constexpr auto m_k_idx = mint::make_tuple(iMs, iKs);
        constexpr auto k_n_idx = mint::make_tuple(iKs, iNs);

        c.template sharded_element<m_n_idx>() +=
            static_cast<c_value_type>(a.template sharded_element<m_k_idx>()) *
            static_cast<c_value_type>(b.template sharded_element<k_n_idx>());
      });
    });
  });
}

// C[M, N] += A[M, K] * B[N, K]
// generic scope, no shuffle
template <class CTensor, class ATensor, class BTensor>
MINT_HOST_DEVICE void
matmul_mn_mk_nk_no_shuffle(CTensor& c, const ATensor& a, const BTensor& b) {
  // TODO: check a/c have same m-sharding
  // TODO: check b/c have same n-sharding
  // TODO: check a/b have same k-sharding
  // TODO: check a/b own full k
  //
  using c_value_type [[maybe_unused]] = typename CTensor::value_type;
  using a_value_type [[maybe_unused]] = typename ATensor::value_type;
  using b_value_type [[maybe_unused]] = typename BTensor::value_type;

  constexpr auto kMs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<0>();
  constexpr auto kNs =
      BTensor::dstr_tensor_desc().sharded_lengths().template at<0>();
  constexpr auto kKs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<1>();

  static_for_nd3<kMs>()([&](auto iMs) {
    static_for_nd3<kNs>()([&](auto iNs) {
      constexpr auto m_n_idx = mint::make_tuple(iMs, iNs);

      static_for_nd3<kKs>()([&](auto iKs) {
        constexpr auto m_k_idx = mint::make_tuple(iMs, iKs);
        constexpr auto n_k_idx = mint::make_tuple(iNs, iKs);

        c.template sharded_element<m_n_idx>() +=
            static_cast<c_value_type>(a.template sharded_element<m_k_idx>()) *
            static_cast<c_value_type>(b.template sharded_element<n_k_idx>());
      });
    });
  });
}

} // namespace generic
} // namespace tile
} // namespace mint
