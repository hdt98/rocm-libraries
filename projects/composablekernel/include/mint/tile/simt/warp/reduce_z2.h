#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/reduce_no_shuffle.h>
#include <mint/tile/simt/warp/shuffle_z2.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

namespace impl {
template <class Morpher0, class Morpher1>
  requires(Morpher0::top_ndim() == Morpher1::top_ndim())
MINT_DEVICE constexpr auto fuse_bottom_dims(Morpher0, Morpher1) {
  constexpr index_t kTopNdim = Morpher0::top_ndim();

  // Create interleaved index pattern: [0, n, 1, n+1, 2, n+2, ...]
  constexpr auto interleaved_top = [&]() {
    nd_index<kTopNdim * 2> ret;
    static_for_n<kTopNdim>()([&](auto i) {
      ret[i * 2] = i;
      ret[i * 2 + 1] = i + kTopNdim;
    });
    return ret;
  }();

  // Create new top lengths
  constexpr auto fused_top_lengths = [&]() {
    nd_index<kTopNdim> ret;
    static_for_n<kTopNdim>()([&](auto i) {
      ret[i] = Morpher0::kTopLengths[i] * Morpher1::kTopLengths[i];
    });
    return ret;
  }();

  constexpr auto m0 = concat(Morpher0{}, Morpher1{});
  constexpr auto m1 = reorder_top(m0, constant<interleaved_top>{});
  constexpr auto m2 = reshape_top(m1, constant<fused_top_lengths>{});

  return m2;
}

template <index_t kBotNDim, nd_index<kBotNDim> kBotDims, class Morpher>
  requires(poly::is_z2_linear_morpher<Morpher>::value)
MINT_DEVICE constexpr auto extract_by_bottom_dims(
    Morpher,
    integral_constant<nd_index<kBotNDim>, kBotDims>) {
  constexpr auto bot_dims = kBotDims;
  constexpr index_t kNumDims = bot_dims.size();
  static_assert(kNumDims > 0, "kBotDims must contain at least one dimension");

  constexpr auto new_bot_bit_total_num = [&]() {
    index_t cnt = 0;
    for (index_t i = 0; i < kNumDims; i++) {
      cnt += Morpher::bot_bit_ends_[bot_dims[i]] -
          Morpher::bot_bit_begins_[bot_dims[i]];
    }
    return cnt;
  }();

  constexpr auto old_top_num_bits = Morpher::top_bit_total_num_;

  constexpr auto new_bot_bits = [&]() {
    nd_index<new_bot_bit_total_num> ret;
    index_t cnt = 0;
    for (index_t dim_idx = 0; dim_idx < kNumDims; dim_idx++) {
      auto bot_bit_begin = Morpher::bot_bit_begins_[bot_dims[dim_idx]];
      auto bot_bit_end = Morpher::bot_bit_ends_[bot_dims[dim_idx]];
      for (index_t i = bot_bit_begin; i < bot_bit_end; i++) {
        ret[cnt++] = i;
      }
    }
    return ret;
  }();

  constexpr auto new_top_bit_total_num = new_bot_bit_total_num;

  constexpr auto old_matrix = Morpher::kTopDownZ2Matrix;

  constexpr auto new_top_bits = [&]() {
    nd_index<new_top_bit_total_num> ret;
    index_t cnt = 0;
    for (index_t j = 0; j < old_top_num_bits; j++) {
      bool found = false;
      for (index_t i = 0; i < new_bot_bit_total_num; i++) {
        if (old_matrix(j, new_bot_bits[i])) {
          found = true;
          break;
        }
      }
      if (found) {
        ret[cnt++] = j;
      }
    }
    return ret;
  }();

  // Extract new_matrix
  constexpr auto new_matrix =
      z2_matrix_extract_sub_matrix(old_matrix, new_top_bits, new_bot_bits);

  // Calculate new top_lengths
  constexpr auto new_top_lengths = [&]() {
    nd_index<Morpher::top_ndim()> ret;
    static_for_n<Morpher::top_ndim()>()([&](auto i) {
      constexpr auto bit_begin = Morpher::top_bit_begins_[i];
      constexpr auto bit_end = Morpher::top_bit_ends_[i];
      // Count how many top_bits are in the range of top_dim i
      index_t cnt = 0;
      for (index_t j = 0; j < new_top_bit_total_num; j++) {
        if (new_top_bits[j] >= bit_begin and new_top_bits[j] < bit_end)
          cnt++;
      }
      // Convert bit_nums to lengths
      ret[i] = (1 << cnt);
    });
    return ret;
  }();

  // Calculate new bot_lengths
  constexpr auto new_bot_lengths = [&]() {
    nd_index<kNumDims> ret;
    for (index_t i = 0; i < kNumDims; i++) {
      ret[i] = Morpher::kBottomLengths[bot_dims[i]];
    }
    return ret;
  }();

  return poly::z2_linear<new_matrix, new_top_lengths, new_bot_lengths>{};
}

template <index_t kTopReducedDim, class Morpher>
MINT_HOST_DEVICE constexpr auto make_reduced_e(Morpher) {
  constexpr auto e_morpher = Morpher{};

  // reduced_e_morpher
  constexpr auto reduced_e_top_lengths = [&]() {
    auto ret = e_morpher.kTopLengths;
    ret[kTopReducedDim] = 1;
    return ret;
  }();

  constexpr auto reduced_e_size = [&]() {
    index_t size = 1;
    for (index_t i = 0; i < reduced_e_top_lengths.size(); i++) {
      size *= reduced_e_top_lengths[i];
    }
    return size;
  }();

  constexpr auto reduced_e_bit_nums =
      math::integer_log2_ceiling(reduced_e_size);

  // assume e_dstr is Packed
  constexpr auto reduced_e_morpher = poly::z2_linear<
      poly::z2::make_z2_unity_matrix<reduced_e_bit_nums>(),
      reduced_e_top_lengths,
      nd_index<1>{reduced_e_size}>{};

  return reduced_e_morpher;
}

template <index_t kTopReducedDim, class Morpher>
MINT_HOST_DEVICE constexpr auto make_reduced_p(Morpher) {
  constexpr auto p_morpher = Morpher{};
  constexpr auto p_matrix = p_morpher.kTopDownZ2Matrix;

  // Aggregate the non-reduced tops, and place the reduced tops as repeated tops
  // at the end
  constexpr auto new_matrix = [&]() {
    constexpr auto total_num_top_bits = p_morpher.top_bit_total_num_;
    constexpr auto total_num_bot_bits = p_morpher.bot_bit_total_num_;
    static_assert(total_num_top_bits == total_num_bot_bits);
    constexpr auto num_reduced_top_bits =
        p_morpher.top_bit_nums_[kTopReducedDim];
    constexpr auto num_free_top_bits =
        total_num_top_bits - num_reduced_top_bits;

    // Collect all bit indices NOT belonging to the reduced dimension
    constexpr auto free_top_bits = [&]() {
      nd_index<num_free_top_bits> ret{};
      index_t cnt = 0;
      static_for_n<p_morpher.top_ndim()>()([&](auto i) {
        constexpr auto top_bit_begin = p_morpher.top_bit_begins_[i];
        constexpr auto top_bit_num = p_morpher.top_bit_nums_[i];
        for (index_t j = 0; j < top_bit_num; j++) {
          if (i != kTopReducedDim)
            ret[cnt++] = j + top_bit_begin;
        }
      });
      return ret;
    }();

    // Collect all bit indices belonging to the reduced dimension
    constexpr auto reduced_top_bits = [&]() {
      nd_index<num_reduced_top_bits> ret{};
      index_t cnt = 0;
      constexpr auto top_bit_begin = p_morpher.top_bit_begins_[kTopReducedDim];
      for (index_t j = 0; j < num_reduced_top_bits; j++) {
        ret[cnt++] = j + top_bit_begin;
      }
      return ret;
    }();

    // Extract matrix rows corresponding to free bits
    constexpr auto free_rows = z2_matrix_extract_rows(p_matrix, free_top_bits);

    // Extract matrix rows corresponding to reduced bits
    constexpr auto reduced_rows =
        z2_matrix_extract_rows(p_matrix, reduced_top_bits);

    // Construct new matrix: free rows first, then reduced rows
    poly::z2::z2_matrix<total_num_top_bits, total_num_bot_bits> ret;

    z2_matrix_set_sub_matrix(ret, free_rows, 0, 0);
    z2_matrix_set_sub_matrix(ret, reduced_rows, num_free_top_bits, 0);

    return ret;
  }();

  // Reorder top dimension lengths: move reduced dimension to end
  constexpr auto reduced_p_top_lengths = [&]() {
    constexpr auto ndim = p_morpher.top_ndim();
    nd_index<ndim> ret = p_morpher.kTopLengths;
    // Move reduced dimension to the end
    if (kTopReducedDim != ndim - 1) {
      auto temp = ret[kTopReducedDim];
      for (index_t i = kTopReducedDim; i < ndim - 1; i++) {
        ret[i] = ret[i + 1];
      }
      ret[ndim - 1] = temp;
    }
    return ret;
  }();

  constexpr auto num_p_bits = p_morpher.bot_bit_total_num_;
  constexpr auto p_bot_size = 1 << num_p_bits;
  static_assert(p_bot_size == MINT_WARP_SIZE);

  constexpr auto reduced_p_morpher = poly::
      z2_linear<new_matrix, reduced_p_top_lengths, nd_index<1>{p_bot_size}>{};

  return reduced_p_morpher;
}

template <
    index_t kBotENDim,
    nd_index<kBotENDim> kBotEDims,
    index_t kBotPNDim,
    nd_index<kBotPNDim> kBotPDims,
    class Morpher>
  requires(poly::is_z2_linear_morpher<Morpher>::value)
MINT_HOST_DEVICE constexpr auto is_valid(Morpher) {
  constexpr auto src_matrix = Morpher{}.kTopDownZ2Matrix;
  constexpr auto bot_e_dims = kBotEDims;
  constexpr auto bot_p_dims = kBotPDims;

  // Collect bit begins for P dimensions
  constexpr auto p_bit_begins = [&]() {
    nd_index<bot_p_dims.size()> ret;
    for (index_t i = 0; i < bot_p_dims.size(); i++) {
      ret[i] = Morpher{}.bot_bit_begins_[bot_p_dims[i]];
    }
    return ret;
  }();

  // Collect bit ends for P dimensions
  constexpr auto p_bit_ends = [&]() {
    nd_index<bot_p_dims.size()> ret;
    for (index_t i = 0; i < bot_p_dims.size(); i++) {
      ret[i] = Morpher{}.bot_bit_ends_[bot_p_dims[i]];
    }
    return ret;
  }();

  // Collect bit begins for E dimensions
  constexpr auto e_bit_begins = [&]() {
    nd_index<bot_e_dims.size()> ret;
    for (index_t i = 0; i < bot_e_dims.size(); i++) {
      ret[i] = Morpher{}.bot_bit_begins_[bot_e_dims[i]];
    }
    return ret;
  }();

  // Collect bit ends for E dimensions
  constexpr auto e_bit_ends = [&]() {
    nd_index<bot_e_dims.size()> ret;
    for (index_t i = 0; i < bot_e_dims.size(); i++) {
      ret[i] = Morpher{}.bot_bit_ends_[bot_e_dims[i]];
    }
    return ret;
  }();

  // Check each row of the transformation matrix
  for (index_t i = 0; i < src_matrix.kM; i++) {
    bool has_p_bit = false, has_e_bit = false;
    for (index_t j = 0; j < src_matrix.kN; j++) {
      if (src_matrix(i, j)) {
        // Check if bit j is in any P dimension range
        for (index_t k = 0; k < bot_p_dims.size(); k++) {
          if (j >= p_bit_begins[k] && j < p_bit_ends[k]) {
            has_p_bit = true;
            break;
          }
        }

        // Check if bit j is in any E dimension range
        for (index_t k = 0; k < bot_e_dims.size(); k++) {
          if (j >= e_bit_begins[k] && j < e_bit_ends[k]) {
            has_e_bit = true;
            break;
          }
        }
      }

      // Invalid if row affects both P and E dimensions
      if (has_p_bit && has_e_bit)
        return false;
    }

    // Invalid if row affects neither P nor E dimension
    if (has_p_bit == has_e_bit)
      return false;
  }

  return true;
}

} // namespace impl

template <
    index_t kBotENDim,
    nd_index<kBotENDim> kBotEDims,
    index_t kBotPNDim,
    nd_index<kBotPNDim> kBotPDims,
    index_t kTopReducedDim,
    class SrcMemory,
    class SrcDstr,
    class SrcLayout,
    class FReduce>
  requires(
      poly::is_z2_linear_morpher<SrcDstr>::value &&
      poly::is_z2_linear_morpher<SrcLayout>::value &&
      impl::is_valid<kBotENDim, kBotEDims, kBotPNDim, kBotPDims>(SrcDstr{}))
MINT_DEVICE auto reduce_z2(
    const SrcMemory& p_src,
    SrcDstr,
    integral_constant<nd_index<kBotENDim>, kBotEDims> bot_e_dims,
    integral_constant<nd_index<kBotPNDim>, kBotPDims> bot_p_dims,
    integral_constant<index_t, kTopReducedDim>,
    SrcLayout,
    FReduce f_reduce) {
  constexpr auto p_morpher =
      impl::extract_by_bottom_dims(SrcDstr{}, bot_p_dims);
  constexpr auto e_morpher =
      impl::extract_by_bottom_dims(SrcDstr{}, bot_e_dims);

  constexpr auto reduced_p_morpher =
      impl::make_reduced_p<kTopReducedDim>(p_morpher);
  constexpr auto reduced_e_morpher =
      impl::make_reduced_e<kTopReducedDim>(e_morpher);

  constexpr auto tmp_dstr =
      impl::fuse_bottom_dims(reduced_p_morpher, reduced_e_morpher);

  constexpr auto reduced_e_size = 1 << reduced_e_morpher.bot_bit_total_num_;

  constexpr auto tmp_layout = poly::make_z2_pass_through_morpher(
      constant<nd_index<1>{reduced_e_size}>{});

  // e2src_m: {m, k} --> {e} --> {offset}
  constexpr auto e2src_m = chain(e_morpher, SrcLayout{});

  // e2dst_m: {m, k} --> {reduced_e} --> {reduced_offset}
  constexpr auto e2dst_m = chain(reduced_e_morpher, tmp_layout);

  owned_vgpr_memory<float, reduced_e_size> v_tmp;

  static_for_n<reduced_e_size>()([&](auto e) { v_tmp[e] = 0; });

  // intra-thread reduce
  static_for_nd2<e_morpher.kTopLengths>()([&](auto... e) {
    constexpr auto idx = nd_index<e_morpher.top_ndim()>{e...};
    constexpr auto src_offset = e2src_m.calculate_bottom_index(idx)[0];
    constexpr auto dst_offset = e2dst_m.calculate_bottom_index(idx)[0];

    v_tmp[constant<dst_offset>{}] =
        f_reduce(v_tmp[constant<dst_offset>{}], p_src[constant<src_offset>{}]);
  });

  constexpr index_t kNumStage = p_morpher.top_bit_nums_[kTopReducedDim];
  constexpr index_t kNumBit = math::integer_log2_ceiling(MINT_WARP_SIZE);

  constexpr auto p_matrix = p_morpher.kTopDownZ2Matrix;

  // shuffle and inter-thread reduce
  static_for_n<kNumStage>()([&](auto iStage) {
    // {src_lane_bits} = {lane_bits} + B * {zeros_for_m_bits, k_swizzle_bits},
    // in z2 space

    // swizzle for k_bits
    constexpr auto k_swizzle_bits = bitset<kNumBit>{(1u << iStage)};

    // swizzle for lane_bits
    constexpr auto lane_swizzle_bits =
        z2_matrix_dot_bitset(constant<p_matrix>{}, k_swizzle_bits);

    static_for_n<reduced_e_size>()([&](auto e) {
#if defined(MINT_BACKEND_CUDA)
      // FIXME: using 0xFFFFFFFF as mask assume warp size is 32
      static_assert(MINT_WARP_SIZE == 32, "current shuffle mask is 32 wide");
      const auto tmp =
          __shfl_xor_sync(0xFFFFFFFF, v_tmp[e], lane_swizzle_bits.to_uint32());
#elif defined(MINT_BACKEND_ROCM)
      static_assert(MINT_WARP_SIZE == 64, "current shuffle mask is 64 wide");
      const auto tmp = __shfl_xor(v_tmp[e], lane_swizzle_bits.to_uint32());
#endif
      v_tmp[e] = f_reduce(tmp, v_tmp[e]);
    });
  });

  return make_tuple(tmp_dstr, tmp_layout, v_tmp);
}

template <
    index_t kBotENDim,
    nd_index<kBotENDim> kBotEDims,
    index_t kBotPNDim,
    nd_index<kBotPNDim> kBotPDims,
    index_t kTopReducedDim,
    class DstMemory,
    class SrcMemory,
    class SrcDstr,
    class DstDstr,
    class SrcLayout,
    class DstLayout,
    class FReduce>
  requires(
      poly::is_z2_linear_morpher<SrcDstr>::value and
      poly::is_z2_linear_morpher<DstDstr>::value and
      poly::is_z2_linear_morpher<SrcLayout>::value and
      poly::is_z2_linear_morpher<DstLayout>::value and
      impl::is_valid<kBotENDim, kBotEDims, kBotPNDim, kBotPDims>(SrcDstr{}))
MINT_DEVICE void reduce_z2(
    DstMemory& p_dst,
    const SrcMemory& p_src,
    SrcDstr,
    DstDstr,
    integral_constant<nd_index<kBotENDim>, kBotEDims> /*bot_e_dims*/,
    integral_constant<nd_index<kBotPNDim>, kBotPDims> /*bot_p_dims*/,
    integral_constant<index_t, kTopReducedDim>,
    SrcLayout,
    DstLayout,
    FReduce f_reduce) {
  auto tmp_desc = reduce_z2(
      p_src,
      SrcDstr{},
      constant<kBotEDims>{},
      constant<kBotPDims>{},
      constant<kTopReducedDim>{},
      SrcLayout{},
      f_reduce);

  shuffle_z2(
      p_dst,
      tmp_desc[2_ic],
      tmp_desc[0_ic],
      DstDstr{},
      integral_constant<nd_index<kBotENDim>, kBotEDims>{},
      integral_constant<nd_index<kBotPNDim>, kBotPDims>{},
      tmp_desc[1_ic],
      DstLayout{});
}

template <
    class DstDstrTensor,
    class SrcDstrTensor,
    index_t kTopReducedDim,
    class FReduce>
  requires(
      SrcDstrTensor{}.dstr_tensor_desc().partition_dims() ==
      DstDstrTensor{}.dstr_tensor_desc().partition_dims())
MINT_DEVICE void reduce_z2(
    DstDstrTensor& dst_dstr_tensor,
    const SrcDstrTensor& src_dstr_tensor,
    integral_constant<index_t, kTopReducedDim> reduced_dim,
    FReduce f_reduce) {
  // Extract descriptors once to avoid repeated temporary construction
  constexpr auto src_dstr_desc = SrcDstrTensor{}.dstr_tensor_desc();
  constexpr auto dst_dstr_desc = DstDstrTensor{}.dstr_tensor_desc();
  constexpr auto src_elem_desc = SrcDstrTensor{}.element_tensor_desc();
  constexpr auto dst_elem_desc = DstDstrTensor{}.element_tensor_desc();

  reduce_z2(
      dst_dstr_tensor.memory(),
      src_dstr_tensor.memory(),
      src_dstr_desc.polymorpher().morphers()[0_ic],
      dst_dstr_desc.polymorpher().morphers()[0_ic],
      constant<src_dstr_desc.element_dims()>{},
      constant<src_dstr_desc.partition_dims()>{},
      reduced_dim,
      src_elem_desc.polymorpher().morphers()[0_ic],
      dst_elem_desc.polymorpher().morphers()[0_ic],
      f_reduce);
}

template <class SrcDstrTensor, index_t kTopReducedDim, class FReduce>
MINT_DEVICE constexpr auto reduce_z2(
    const SrcDstrTensor& src_dstr_tensor,
    integral_constant<index_t, kTopReducedDim> reduced_dim,
    FReduce f_reduce) {
  // Extract descriptors once to avoid repeated temporary construction
  constexpr auto src_dstr_desc = SrcDstrTensor{}.dstr_tensor_desc();
  constexpr auto src_elem_desc = SrcDstrTensor{}.element_tensor_desc();
  auto dst_desc = reduce_z2(
      src_dstr_tensor.memory(),
      src_dstr_desc.polymorpher().morphers()[0_ic],
      constant<src_dstr_desc.element_dims()>{},
      constant<src_dstr_desc.partition_dims()>{},
      reduced_dim,
      src_elem_desc.polymorpher().morphers()[0_ic],
      f_reduce);

  constexpr auto dstr_poly = make_z2_polymorpher_default_alias(dst_desc[0_ic]);
  constexpr auto dst_dstr_desc = tensor::make_distributed_tensor_descriptor(
      dstr_poly,
      constant<src_dstr_desc.partition_dims()>{},
      constant<src_dstr_desc.element_dims()>{});

  constexpr auto elm_poly = make_z2_polymorpher_default_alias(dst_desc[1_ic]);
  constexpr auto elm_layout = tensor::make_tensor_descriptor(elm_poly);

  return tensor::make_distributed_tensor(
      constant<dst_dstr_desc>{}, constant<elm_layout>{}, dst_desc[2_ic]);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
