#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {

template <typename TA, typename TB, typename TC>
__device__ void inner_product(const TA& a, const TB& b, TC& c);

typedef fp16_t fp16x2_t __attribute__((ext_vector_type(2)));

template <>
__device__ void inner_product<fp16x2_t, fp16x2_t, float>(
    const fp16x2_t& a,
    const fp16x2_t& b,
    float& c) {
#if 1
  c = __builtin_amdgcn_fdot2(a, b, c, false);
#elif 1
  asm volatile(
      "\n \
            v_dot2_f32_f16 %0, %1, %2, %0\n \
            s_nop 2 \n \
            "
      : "=v"(c)
      : "v"(a), "v"(b), "0"(c));
#else
  using v2_t = vector_type<fp16_t, 2>;

  const auto a_vec = reinterpret_cast<const v2_t&>(a);
  const auto b_vec = reinterpret_cast<const v2_t&>(b);

  static_for_n<2>()([&](auto i) { c += a_vec[i] * b_vec[i]; });
#endif
}

namespace tile {
namespace rocm {

// C[M, N] += A[M, K] * B[K, N]
// generic scope, no shuffle
template <class CTensor, class ATensor, class BTensor>
MINT_HOST_DEVICE void
matmul_dl(CTensor& c, const ATensor& a, const BTensor& b) {
  // TODO: check a/c have same m-sharding
  // TODO: check b/c have same n-sharding
  // TODO: check a/b have same k-sharding
  // TODO: check a/b own full k

  constexpr auto kMs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<0>();
  constexpr auto kNs =
      BTensor::dstr_tensor_desc().sharded_lengths().template at<0>();
  constexpr auto kKs =
      ATensor::dstr_tensor_desc().sharded_lengths().template at<1>();

  using CType = CTensor::value_type;
  static_assert(is_same_v<CType, float>);

  using AType = ATensor::value_type;
  using BType = BTensor::value_type;

  static_assert(is_same_v<AType, BType>);

  constexpr index_t KPack = sizeof(float) / sizeof(AType);

  constexpr index_t kK0 = kKs[0];

  static_assert(kK0 % KPack == 0, "kKs cannot be divided by KPack");

  static_for_nd3<kMs>()([&](auto iMs) {
    static_for_nd3<kNs>()([&](auto iNs) {
      constexpr auto m_n_idx = mint::make_tuple(iMs, iNs);

      float Acc = 0;

      static_for_n<kK0 / KPack>()([&](auto iK0) {
        vector_type<fp16_t, KPack> a_vec, b_vec;

        static_for_n<KPack>()([&](auto i) {
          constexpr auto m_k_idx =
              mint::make_tuple(iMs, index_sequence<iK0 * KPack + i>{});
          constexpr auto n_k_idx =
              mint::make_tuple(iNs, index_sequence<iK0 * KPack + i>{});

          a_vec[i] = a.template sharded_element<m_k_idx>();
          b_vec[i] = b.template sharded_element<n_k_idx>();
        });

        inner_product(
            bit_cast<fp16x2_t>(a_vec.template as_vectors<KPack>()[0_ic]),
            bit_cast<fp16x2_t>(b_vec.template as_vectors<KPack>()[0_ic]),
            Acc);
      });

      c.template sharded_element<m_n_idx>() += Acc;
    });
  });
}

} // namespace rocm
} // namespace tile
} // namespace mint
