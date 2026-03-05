#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>

namespace mint {

typedef float floatx16_t __attribute__((ext_vector_type(16)));
typedef float floatx4_t __attribute__((ext_vector_type(4)));
typedef fp16_t fp16x4_t __attribute__((ext_vector_type(4)));

#ifdef __gfx950__
struct mfma_impl_f32_16x16x32f16 {
  using CType = floatx4_t;
  using AType = fp16x8_t;
  using BType = fp16x8_t;

  static constexpr index_t kMFMAKPack = 8;
  static constexpr index_t kMPerMFMA = 16;
  static constexpr index_t kNPerMFMA = 16;
  static constexpr index_t kKPerMFMA = 32;
  static constexpr index_t kCVecSize = 4;

  template <bool SwapAB = false>
  __device__ static void
  run(const fp16x8_t& a, const fp16x8_t& b, floatx4_t& acc) {
    if constexpr (SwapAB) {
      acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(b, a, acc, 0, 0, 0);
    } else {
      acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, acc, 0, 0, 0);
    }
  }
};

struct mfma_impl_f32_32x32x16f16 {
  using CType = floatx16_t;
  using AType = fp16x8_t;
  using BType = fp16x8_t;

  static constexpr index_t kMFMAKPack = 8;
  static constexpr index_t kMPerMFMA = 32;
  static constexpr index_t kNPerMFMA = 32;
  static constexpr index_t kKPerMFMA = 16;
  static constexpr index_t kCVecSize = 16;

  template <bool SwapAB = false>
  __device__ static void
  run(const fp16x8_t& a, const fp16x8_t& b, floatx16_t& acc) {
    if constexpr (SwapAB) {
      acc = __builtin_amdgcn_mfma_f32_32x32x16_f16(b, a, acc, 0, 0, 0);
    } else {
      acc = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, acc, 0, 0, 0);
    }
  }
};
#endif

struct mfma_impl_f32_16x16x16f16 {
  using CType = floatx4_t;
  using AType = fp16x4_t;
  using BType = fp16x4_t;

  static constexpr index_t kMFMAKPack = 4;
  static constexpr index_t kMPerMFMA = 16;
  static constexpr index_t kNPerMFMA = 16;
  static constexpr index_t kKPerMFMA = 16;
  static constexpr index_t kCVecSize = 4;

  template <bool SwapAB = false>
  __device__ static void
  run(const fp16x4_t& a, const fp16x4_t& b, floatx4_t& acc) {
    if constexpr (SwapAB) {
      acc = __builtin_amdgcn_mfma_f32_16x16x16f16(b, a, acc, 0, 0, 0);
    } else {
      acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
    }
  }
};

struct mfma_impl_f32_32x32x8f16 {
  using CType = floatx16_t;
  using AType = fp16x4_t;
  using BType = fp16x4_t;

  static constexpr index_t kMFMAKPack = 4;
  static constexpr index_t kMPerMFMA = 32;
  static constexpr index_t kNPerMFMA = 32;
  static constexpr index_t kKPerMFMA = 8;
  static constexpr index_t kCVecSize = 16;

  template <bool SwapAB = false>
  __device__ static void
  run(const fp16x4_t& a, const fp16x4_t& b, floatx16_t& acc) {
    if constexpr (SwapAB) {
      acc = __builtin_amdgcn_mfma_f32_32x32x8f16(b, a, acc, 0, 0, 0);
    } else {
      acc = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, acc, 0, 0, 0);
    }
  }
};

struct mfma_impl_f32_32x32x2f32 {
  using CType = floatx16_t;
  using AType = float;
  using BType = float;

  static constexpr index_t kMFMAKPack = 1;
  static constexpr index_t kMPerMFMA = 32;
  static constexpr index_t kNPerMFMA = 32;
  static constexpr index_t kKPerMFMA = 2;
  static constexpr index_t kCVecSize = 16;

  template <bool SwapAB = false>
  MINT_DEVICE static void run(const float& a, const float& b, floatx16_t& acc) {
    if constexpr (SwapAB) {
      acc = __builtin_amdgcn_mfma_f32_32x32x2f32(b, a, acc, 0, 0, 0);
    } else {
      acc = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, acc, 0, 0, 0);
    }
  }
};

namespace tile {
namespace rocm {

// C[M, N] += A[M, K] * B[K, N]
// AMD xdlops (i.e., mfma), shuffled output
template <
    bool SwapAB = false,
    index_t kMPerFMA = 16,
    index_t kNPerMFMA = 16,
    class CTensor,
    class ATensor,
    class BTensor>
MINT_DEVICE void matmul_xdl(CTensor& c, const ATensor& a, const BTensor& b) {
  // TODO: check a/c have same m-sharding
  // TODO: check b/c have same n-sharding
  // TODO: check a/b have same k-sharding
  // TODO: check a/b own full k
  constexpr auto a_elem_desc = ATensor::element_tensor_desc();

  constexpr index_t kAK0 = a_elem_desc.top_lengths()[0];
  constexpr index_t kAM0 = a_elem_desc.top_lengths()[1];
  constexpr index_t kAK1 = a_elem_desc.top_lengths()[2];

  constexpr auto b_elem_desc = BTensor::element_tensor_desc();

  constexpr index_t kBK0 = b_elem_desc.top_lengths()[0];
  constexpr index_t kBN0 = b_elem_desc.top_lengths()[1];
  constexpr index_t kBK1 = b_elem_desc.top_lengths()[2];

  static_assert(kAK0 == kBK0);
  static_assert(kAK1 == kBK1);

  using TypeA [[maybe_unused]] = typename ATensor::value_type;
  using TypeB [[maybe_unused]] = typename BTensor::value_type;
  using TypeC [[maybe_unused]] = typename CTensor::value_type;

  static_assert(kMPerFMA == kNPerMFMA);

  constexpr auto mfma_impl_attr_ = [&]() {
    if constexpr (
        kMPerFMA == 16 and
        (is_same_v<TypeA, fp16_t> or is_same_v<TypeA, fp16x8_t>))
#ifdef __gfx950__
      return mfma_impl_f32_16x16x32f16{};
#else
      return mfma_impl_f32_16x16x16f16{};
#endif

    if constexpr (kMPerFMA == 32 and is_same_v<TypeA, float>)
      return mfma_impl_f32_32x32x2f32{};

    if constexpr (
        kMPerFMA == 32 and
        (is_same_v<TypeA, fp16_t> or is_same_v<TypeA, fp16x8_t>))
      return mfma_impl_f32_32x32x8f16{};
  }();

  using mfma_impl_attr = decltype(mfma_impl_attr_);

  using MFMATypeA = typename mfma_impl_attr::AType;
  using MFMATypeB = typename mfma_impl_attr::BType;
  using MFMATypeC = typename mfma_impl_attr::CType;

  constexpr index_t kMFMAKPack = mfma_impl_attr::kMFMAKPack;
  constexpr index_t kCVecSize = mfma_impl_attr::kCVecSize;
  constexpr index_t kMRepeat = kAM0;
  constexpr index_t kNRepeat = kBN0;
  constexpr index_t kKRepeat = kAK0;
  constexpr index_t kKLoop = kAK1 / kMFMAKPack;

  static_for_n<kMRepeat>()([&](auto iM0) {
    static_for_n<kNRepeat>()([&](auto iN0) {
      static_for_n<kKRepeat>()([&](auto iKR) {
        if constexpr (
            (std::is_same_v<TypeA, fp16_t> or std::is_same_v<TypeA, float>) and
            std::is_same_v<TypeC, float>) {
          constexpr auto c_idx = nd_index<4>{iM0, iN0, 0, 0};
          // execute mfma c_vec = a_vec dot b_vec
          static_for_n<kKLoop>()([&](auto iKL) {
            constexpr auto a_idx = nd_index<3>{iKR, iM0, iKL * kMFMAKPack};
            constexpr auto b_idx = nd_index<3>{iKR, iN0, iKL * kMFMAKPack};

            const auto a_vec = bit_cast<MFMATypeA>(
                a.template element_vector<a_idx, kMFMAKPack>());
            const auto b_vec = bit_cast<MFMATypeB>(
                b.template element_vector<b_idx, kMFMAKPack>());

            auto c_tmp = bit_cast<MFMATypeC>(
                c.template element_vector<c_idx, kCVecSize>());

            mfma_impl_attr::template run<SwapAB>(a_vec, b_vec, c_tmp);

            c.template element_vector<c_idx, kCVecSize>() =
                bit_cast<vector_type<TypeC, kCVecSize>>(c_tmp);
          });
        } else if constexpr (
            std::is_same_v<TypeA, fp16_t> and !(std::is_same_v<TypeC, float>)) {
          constexpr auto c_idx = nd_index<2>{iM0, iN0};
          // execute mfma c_vec = a_vec dot b_vec
          static_for_n<kKLoop>()([&](auto iKL) {
            constexpr auto a_idx = nd_index<3>{iKR, iM0, iKL * kMFMAKPack};
            constexpr auto b_idx = nd_index<3>{iKR, iN0, iKL * kMFMAKPack};

            const auto a_vec = bit_cast<MFMATypeA>(
                a.template element_vector<a_idx, kMFMAKPack>());
            const auto b_vec = bit_cast<MFMATypeB>(
                b.template element_vector<b_idx, kMFMAKPack>());

            mfma_impl_attr::template run<SwapAB>(
                a_vec, b_vec, c.template element<c_idx>());
          });
        } else {
          constexpr auto a_idx = nd_index<3>{iKR, iM0, 0};
          constexpr auto b_idx = nd_index<3>{iKR, iN0, 0};

          constexpr auto c_idx = nd_index<2>{iM0, iN0};

          auto a_vec =
              bit_cast<vector_type<fp16_t, 8>>(a.template element<a_idx>());
          auto b_vec =
              bit_cast<vector_type<fp16_t, 8>>(b.template element<b_idx>());

          static_for_n<mfma_impl_attr::kMPerMFMA == 16 ? 1 : 2>()(
              [&](auto iKL) {
                mfma_impl_attr::template run<SwapAB>(
                    bit_cast<MFMATypeA>(
                        a_vec.template as_vectors<kMFMAKPack>()[iKL]),
                    bit_cast<MFMATypeB>(
                        b_vec.template as_vectors<kMFMAKPack>()[iKL]),
                    c.template element<c_idx>());
              });
        }
      });
    });
  });
}

} // namespace rocm
} // namespace tile
} // namespace mint
