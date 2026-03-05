#pragma once

#include <mint/core.h>
#include <mint/poly.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

template <
    index_t kENDim,
    nd_index<kENDim> kEDims,
    index_t kPNDim,
    nd_index<kPNDim> kPDims,
    class DstMemory,
    class SrcMemory,
    class SrcDstr,
    class DstDstr,
    class SrcElmLayout,
    class DstElmLayout>
MINT_DEVICE void shuffle_z2(
    DstMemory& p_dst,
    const SrcMemory& p_src,
    SrcDstr,
    DstDstr,
    integral_constant<nd_index<kENDim>, kEDims>,
    integral_constant<nd_index<kPNDim>, kPDims>,
    SrcElmLayout,
    DstElmLayout) {
  static_assert(SrcDstr::kBottomLengths == DstDstr::kBottomLengths);
  constexpr auto get_bot_bits =
      []<class Dstr, index_t kNDim, nd_index<kNDim> kDims>(Dstr) {
        constexpr auto bit_starts = []() {
          nd_index<kNDim> starts{};
          for (index_t i = 0; i < kNDim; i++) {
            starts[i] = Dstr{}.bot_bit_begins_[kDims[i]];
          }
          return starts;
        }();

        constexpr auto bit_ends = []() {
          nd_index<kNDim> ends{};
          for (index_t i = 0; i < kNDim; i++) {
            ends[i] = Dstr{}.bot_bit_ends_[kDims[i]];
          }
          return ends;
        }();

        constexpr auto bit_nums = [&]() {
          nd_index<kNDim> nums{};
          for (index_t i = 0; i < kNDim; i++) {
            nums[i] = bit_ends[i] - bit_starts[i];
          }
          return nums;
        }();

        constexpr index_t total_bits = [&]() {
          index_t total = 0;
          for (index_t i = 0; i < kNDim; i++) {
            total += bit_nums[i];
          }
          return total;
        }();

        nd_index<total_bits> ret;
        index_t cnt = 0;
        for (index_t i = 0; i < kNDim; i++) {
          for (index_t j = 0; j < bit_nums[i]; j++) {
            ret[cnt++] = j + bit_starts[i];
          }
        }
        return ret;
      };

  // Extract P-dimension matrices from source and destination distribution
  constexpr auto p_matrix_src = z2_matrix_extract_columns(
      SrcDstr{}.kTopDownZ2Matrix,
      get_bot_bits.template operator()<SrcDstr, kPNDim, kPDims>(SrcDstr{}));
  constexpr auto p_matrix_dst = z2_matrix_extract_columns(
      DstDstr{}.kTopDownZ2Matrix,
      get_bot_bits.template operator()<DstDstr, kPNDim, kPDims>(DstDstr{}));

  // matrix: {e_dst, p_dst} --> {e_src, p_src}
  constexpr auto dst2src_shuffle_m = chain(invert(DstDstr{}), SrcDstr{});

  // Extract separate morphers for E and P dimensions
  // E-morpher: {e_dst} -> {e_src}
  // P-morpher: {p_dst} -> {p_src}
  constexpr auto dst2src_shuffle_morpher_e =
      extract(dst2src_shuffle_m, constant<kEDims>{}, constant<kEDims>{});
  constexpr auto dst2src_shuffle_morpher_p =
      extract(dst2src_shuffle_m, constant<kPDims>{}, constant<kPDims>{});

  // Calculate number of elements per thread (2^num_bits)
  constexpr index_t kEPerThread = 1
      << dst2src_shuffle_morpher_e.bot_bit_total_num_;

  // chain: {e_dst} -> {e_src} -> {src_offset}
  constexpr auto src_element_access_m =
      chain(dst2src_shuffle_morpher_e, SrcElmLayout{});

  // STAGE 1: Intra-thread shuffle
  static_for_n<kEPerThread>()([&](auto e) {
    constexpr auto e_idx = nd_index<1>{e};
    constexpr index_t src_offset =
        src_element_access_m.calculate_bottom_index(e_idx)[0];
    constexpr index_t dst_offset =
        DstElmLayout{}.calculate_bottom_index(e_idx)[0];
    p_dst[index_constant<dst_offset>{}] = p_src[index_constant<src_offset>{}];
  });

  // Optimization: Check if P-matrices is identical between src and dst
  // If P-matrices are the same, no inter-thread communication is needed
  constexpr bool is_same_p_matrix = (p_matrix_src == p_matrix_dst);

  // STAGE 2: Inter-thread shuffle (if needed)
  if (!is_same_p_matrix) {
    // Calculate the number of bits needed to represent a warp lane ID
    constexpr index_t kNumBit = math::integer_log2_ceiling(MINT_WARP_SIZE);

    // Get current thread's lane ID as a bitset
    const auto lane_bits = bitset<kNumBit>{threadIdx.x};

    // Transform destination lane ID to source lane ID using the P-morpher
    // matrix
    const auto src_lane_bits = z2_matrix_dot_bitset(
        constant<dst2src_shuffle_morpher_p.kTopDownZ2Matrix>{}, lane_bits);

    // Exchange all elements with the source thread using warp shuffle
    // __shfl reads from the specified lane within the warp
    static_for_n<kEPerThread>()([&](auto e) {
      p_dst[e] = __shfl(p_dst[e], src_lane_bits.to_uint32(), MINT_WARP_SIZE);
    });
  }
}

namespace impl {
template <class T>
concept single_morpehr_tensor =
    T{}.dstr_tensor_desc().polymorpher().num_morpher() == 1;

template <class Src, class Dst>
concept compatible_shuffle_dimensions =
    Src{}.dstr_tensor_desc().element_dims() ==
        Dst{}.dstr_tensor_desc().element_dims() &&
    Src{}.dstr_tensor_desc().partition_dims() ==
        Dst{}.dstr_tensor_desc().partition_dims();

} // namespace impl

template <class DstDstrTensor, class SrcDstrTensor>
  requires(
      impl::single_morpehr_tensor<SrcDstrTensor> &&
      impl::single_morpehr_tensor<DstDstrTensor> &&
      impl::compatible_shuffle_dimensions<SrcDstrTensor, DstDstrTensor>)
MINT_DEVICE void shuffle_z2(
    DstDstrTensor& dst_dstr_tensor,
    const SrcDstrTensor& src_dstr_tensor) {
  // Extract descriptors once to avoid repeated temporary construction
  constexpr auto src_dstr_desc = SrcDstrTensor{}.dstr_tensor_desc();
  constexpr auto dst_dstr_desc = DstDstrTensor{}.dstr_tensor_desc();
  constexpr auto src_elem_desc = SrcDstrTensor{}.element_tensor_desc();
  constexpr auto dst_elem_desc = DstDstrTensor{}.element_tensor_desc();

  mint::tile::simt::warp::shuffle_z2(
      dst_dstr_tensor.memory(),
      src_dstr_tensor.memory(),
      src_dstr_desc.polymorpher().morphers()[0_ic],
      dst_dstr_desc.polymorpher().morphers()[0_ic],
      constant<src_dstr_desc.element_dims()>{},
      constant<src_dstr_desc.partition_dims()>{},
      src_elem_desc.polymorpher().morphers()[0_ic],
      dst_elem_desc.polymorpher().morphers()[0_ic]);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
