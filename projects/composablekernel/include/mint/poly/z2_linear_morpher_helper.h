#pragma once
#include <mint/core.h>
#include <mint/poly/morpher.h>
#include <mint/poly/z2.h>

namespace mint::poly {

template <class T>
struct is_z2_linear_morpher : std::false_type {};

template <auto kTopDownZ2Matrix, auto kTopLengths, auto kBottomLengths>
struct is_z2_linear_morpher<
    z2_linear<kTopDownZ2Matrix, kTopLengths, kBottomLengths>> : std::true_type {
};

template <index_t kNDim, nd_index<kNDim> kLengths>
MINT_HOST_DEVICE constexpr auto make_z2_pass_through_morpher(
    integral_constant<nd_index<kNDim>, kLengths>) {
  constexpr index_t num_bit = [&]() {
    index_t cnt = 0;
    for (index_t i = 0; i < kNDim; i++)
      cnt += math::integer_log2_ceiling(kLengths[i]);
    return cnt;
  }();
  return z2_linear<z2::make_z2_unity_matrix<num_bit>(), kLengths, kLengths>{};
}

template <class OldMorpher>
  requires(is_z2_linear_morpher<OldMorpher>::value)
MINT_HOST_DEVICE constexpr auto invert(OldMorpher) {
  return z2_linear<
      z2::z2_matrix_invert(OldMorpher::kTopDownZ2Matrix),
      OldMorpher::kBottomLengths,
      OldMorpher::kTopLengths>{};
}

template <class TopMorpher, class BottomMorpher>
  requires(
      is_z2_linear_morpher<TopMorpher>::value &&
      is_z2_linear_morpher<BottomMorpher>::value &&
      TopMorpher::kBottomLengths == BottomMorpher::kTopLengths)
MINT_HOST_DEVICE constexpr auto chain(TopMorpher, BottomMorpher) {
  return z2_linear<
      z2::z2_matmul(
          BottomMorpher::kTopDownZ2Matrix, TopMorpher::kTopDownZ2Matrix),
      TopMorpher::kTopLengths,
      BottomMorpher::kBottomLengths>{};
}

template <
    class Morpher,
    index_t kNTop,
    index_t kNBot,
    nd_index<kNTop> kTopDims,
    nd_index<kNBot> kBotDims>
  requires(is_z2_linear_morpher<Morpher>::value)
MINT_HOST_DEVICE constexpr auto is_valid_extract() {
  constexpr auto compute_bit_set =
      []<index_t ndim, auto dims, auto bit_begins, auto bit_ends>() {
        array<bool, Morpher::top_bit_total_num_ + Morpher::bot_bit_total_num_>
            ret;
        ret.fill(false);
        for (index_t i = 0; i < dims.size(); i++) {
          index_t dim = dims[i];
          for (index_t j = bit_begins[dim]; j < bit_ends[dim]; j++) {
            ret[j] = true;
          }
        }
        return ret;
      };

  constexpr auto top_bit_set = compute_bit_set.template operator()<
      Morpher::top_ndim(),
      kTopDims,
      Morpher::top_bit_begins_,
      Morpher::top_bit_ends_>();

  constexpr auto bot_bit_set = compute_bit_set.template operator()<
      Morpher::bottom_ndim(),
      kBotDims,
      Morpher::bot_bit_begins_,
      Morpher::bot_bit_ends_>();

  constexpr auto matrix = Morpher::kTopDownZ2Matrix;

  constexpr bool is_off_diagonal_row_all_zero = [&]() {
    for (index_t i = 0; i < Morpher::top_bit_total_num_; i++) {
      if (top_bit_set[i])
        continue;
      for (index_t j = 0; j < Morpher::bot_bit_total_num_; j++) {
        if (bot_bit_set[j] and matrix(i, j))
          return false;
      }
    }
    return true;
  }();

  constexpr bool is_off_diagonal_column_all_zero = [&]() {
    for (index_t i = 0; i < Morpher::top_bit_total_num_; i++) {
      if (!top_bit_set[i])
        continue;
      for (index_t j = 0; j < Morpher::bot_bit_total_num_; j++) {
        if (!bot_bit_set[j] and matrix(i, j))
          return false;
      }
    }
    return true;
  }();

  return is_off_diagonal_row_all_zero and is_off_diagonal_column_all_zero;
}

template <
    class Morpher,
    index_t kNTop,
    index_t kNBot,
    nd_index<kNTop> kTopDims,
    nd_index<kNBot> kBotDims>
  requires(is_z2_linear_morpher<Morpher>::value)
MINT_HOST_DEVICE constexpr auto extract(
    Morpher,
    integral_constant<nd_index<kNTop>, kTopDims>,
    integral_constant<nd_index<kNBot>, kBotDims>) {
  constexpr nd_index<kNTop> top_lengths = [&]() {
    nd_index<kNTop> ret;
    for (index_t i = 0; i < kNTop; i++) {
      ret[i] = Morpher::kTopLengths[kTopDims[i]];
    }
    return ret;
  }();

  constexpr nd_index<kNBot> bot_lengths = [&]() {
    nd_index<kNBot> ret;
    for (index_t i = 0; i < kNBot; i++) {
      ret[i] = Morpher::kBottomLengths[kBotDims[i]];
    }
    return ret;
  }();

  constexpr index_t num_top_bits = [&]() {
    index_t cnt = 0;
    for (index_t i = 0; i < kNTop; i++) {
      index_t dim = kTopDims[i];
      cnt += Morpher::top_bit_ends_[dim] - Morpher::top_bit_begins_[dim];
    }
    return cnt;
  }();

  constexpr index_t num_bot_bits = [&]() {
    index_t cnt = 0;
    for (index_t i = 0; i < kNBot; i++) {
      index_t dim = kBotDims[i];
      cnt += Morpher::bot_bit_ends_[dim] - Morpher::bot_bit_begins_[dim];
    }
    return cnt;
  }();

  constexpr nd_index<num_top_bits> top_bits = [&]() {
    nd_index<num_top_bits> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < kNTop; i++) {
      index_t dim = kTopDims[i];
      for (index_t j = Morpher::top_bit_begins_[dim];
           j < Morpher::top_bit_ends_[dim];
           j++) {
        ret[cnt++] = j;
      }
    }
    return ret;
  }();

  constexpr nd_index<num_bot_bits> bot_bits = [&]() {
    nd_index<num_bot_bits> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < kNBot; i++) {
      index_t dim = kBotDims[i];
      for (index_t j = Morpher::bot_bit_begins_[dim];
           j < Morpher::bot_bit_ends_[dim];
           j++) {
        ret[cnt++] = j;
      }
    }
    return ret;
  }();

  constexpr auto sub_matrix = z2_matrix_extract_sub_matrix(
      Morpher::kTopDownZ2Matrix, top_bits, bot_bits);

  return z2_linear<sub_matrix, top_lengths, bot_lengths>{};
}

template <class Morpher0, class Morpher1>
  requires(
      is_z2_linear_morpher<Morpher0>::value &&
      is_z2_linear_morpher<Morpher1>::value)
MINT_HOST_DEVICE constexpr auto concat(Morpher0, Morpher1) {
  constexpr auto top_lengths = [&]() {
    nd_index<Morpher0::top_ndim() + Morpher1::top_ndim()> ret;
    std::copy(
        Morpher0::kTopLengths.begin(),
        Morpher0::kTopLengths.end(),
        ret.begin());
    std::copy(
        Morpher1::kTopLengths.begin(),
        Morpher1::kTopLengths.end(),
        ret.begin() + Morpher0::top_ndim());
    return ret;
  }();

  constexpr auto bot_lengths = [&]() {
    nd_index<Morpher0::bottom_ndim() + Morpher1::bottom_ndim()> ret;
    std::copy(
        Morpher0::kBottomLengths.begin(),
        Morpher0::kBottomLengths.end(),
        ret.begin());
    std::copy(
        Morpher1::kBottomLengths.begin(),
        Morpher1::kBottomLengths.end(),
        ret.begin() + Morpher0::bottom_ndim());
    return ret;
  }();

  // z2 matrix is ordered in reverse order of tensor dimensions
  constexpr auto mtx = [&]() {
    constexpr index_t kM0 = Morpher0::kTopDownZ2Matrix.kM;
    constexpr index_t kN0 = Morpher0::kTopDownZ2Matrix.kN;
    constexpr index_t kM1 = Morpher1::kTopDownZ2Matrix.kM;
    constexpr index_t kN1 = Morpher1::kTopDownZ2Matrix.kN;
    constexpr index_t kM = kM0 + kM1;
    constexpr index_t kN = kN0 + kN1;

    z2::z2_matrix<kM, kN> ret;
    ret.fill(0);

    // dims in morpher0 map to higher dims in concatenated morpher
    // dims in morpher1 map to higher dims in concatenated morpher
    z2::z2_matrix_set_sub_matrix(ret, Morpher0::kTopDownZ2Matrix, kM1, kN1);
    z2::z2_matrix_set_sub_matrix(ret, Morpher1::kTopDownZ2Matrix, 0, 0);

    return ret;
  }();

  return z2_linear<mtx, top_lengths, bot_lengths>{};
}

// pair some bottom dims of top morpher with some top dimes of bottom morpher
template <
    class TopMorpher,
    class BottomMorpher,
    class TopMorpherPairedBottomDims,
    TopMorpherPairedBottomDims kTopMorpherPairedBottomDims,
    class BottomMorpherPairedTopDims,
    BottomMorpherPairedTopDims kBottomMorpherPairedTopDims>
  requires(
      kTopMorpherPairedBottomDims.size() == kBottomMorpherPairedTopDims.size())
MINT_HOST_DEVICE constexpr auto fuse(
    TopMorpher,
    BottomMorpher,
    integral_constant<TopMorpherPairedBottomDims, kTopMorpherPairedBottomDims>,
    integral_constant<
        BottomMorpherPairedTopDims,
        kBottomMorpherPairedTopDims>) {
  auto f = []<auto ndim,
              auto bit_total_num,
              auto paired_dims,
              auto bit_begins,
              auto bit_ends>() {
    constexpr auto is_paired_dims = [&]() {
      array<bool, ndim> ret;
      ret.fill(false);
      for (auto i : paired_dims)
        ret[i] = true;
      return ret;
    }();

    constexpr auto is_paired_bits = [&]() {
      array<bool, bit_total_num> ret;
      ret.fill(false);
      for (index_t i = 0; i < ndim; i++)
        if (is_paired_dims[i])
          for (index_t j = bit_begins[i]; j < bit_ends[i]; j++)
            ret[j] = true;
      return ret;
    }();

    constexpr index_t num_paired_bit =
        std::count(is_paired_bits.begin(), is_paired_bits.end(), true);

    constexpr auto paired_bits = [&is_paired_bits]() {
      nd_index<num_paired_bit> ret;
      index_t cnt = 0;
      for (index_t i = 0; i < is_paired_bits.size(); i++) {
        if (is_paired_bits[i])
          ret[cnt++] = i;
      }
      return ret;
    }();

    constexpr index_t num_free_bit =
        std::count(is_paired_bits.begin(), is_paired_bits.end(), false);

    constexpr auto free_bits = [&]() {
      nd_index<num_free_bit> ret;
      index_t cnt = 0;
      for (index_t i = 0; i < is_paired_bits.size(); i++) {
        if (!is_paired_bits[i])
          ret[cnt++] = i;
      }
      return ret;
    }();

    return mint::make_tuple(
        is_paired_dims, is_paired_bits, paired_dims, paired_bits, free_bits);
  };

  constexpr auto top_tmp = f.template operator()<
      TopMorpher::bottom_ndim(),
      TopMorpher::bot_bit_total_num_,
      kTopMorpherPairedBottomDims,
      TopMorpher::bot_bit_begins_,
      TopMorpher::bot_bit_ends_>();

  constexpr auto bot_tmp = f.template operator()<
      BottomMorpher::top_ndim(),
      BottomMorpher::top_bit_total_num_,
      kBottomMorpherPairedTopDims,
      BottomMorpher::top_bit_begins_,
      BottomMorpher::top_bit_ends_>();

  constexpr auto is_top_morpher_paired_bot_dims = top_tmp[0_ic];
  constexpr auto is_bot_morpher_paired_top_dims = bot_tmp[0_ic];
  constexpr auto is_top_morpher_paired_bot_bits = top_tmp[1_ic];
  constexpr auto is_bot_morpher_paired_top_bits = bot_tmp[1_ic];
  constexpr auto top_morpher_paired_bot_dims = top_tmp[2_ic];
  constexpr auto bot_morpher_paired_top_dims = bot_tmp[2_ic];
  constexpr auto top_morpher_paired_bot_bits = top_tmp[3_ic];
  constexpr auto bot_morpher_paired_top_bits = bot_tmp[3_ic];
  constexpr auto top_morpher_free_bot_bits = top_tmp[4_ic];
  constexpr auto bot_morpher_free_top_bits = bot_tmp[4_ic];

  static_assert(
      top_morpher_paired_bot_dims.size() == bot_morpher_paired_top_dims.size());
  static_assert(
      top_morpher_paired_bot_bits.size() == bot_morpher_paired_top_bits.size());

  constexpr index_t ndim_paired = top_morpher_paired_bot_dims.size();
  constexpr index_t nbit_paired = top_morpher_paired_bot_bits.size();

  constexpr auto new_top_lengths = [&]() {
    constexpr index_t new_top_ndim =
        TopMorpher::top_ndim() + BottomMorpher::top_ndim() - ndim_paired;
    nd_index<new_top_ndim> ret;
    for (index_t i = 0; i < TopMorpher::top_ndim(); i++)
      ret[i] = TopMorpher::kTopLengths[i];
    index_t cnt = TopMorpher::top_ndim();
    for (index_t i = 0; i < BottomMorpher::top_ndim(); i++) {
      if (!is_bot_morpher_paired_top_dims[i])
        ret[cnt++] = BottomMorpher::kTopLengths[i];
    }
    return ret;
  }();

  constexpr auto new_bot_lengths = [&]() {
    constexpr index_t new_bot_ndim =
        TopMorpher::bottom_ndim() + BottomMorpher::bottom_ndim() - ndim_paired;
    nd_index<new_bot_ndim> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < TopMorpher::bottom_ndim(); i++) {
      if (!is_top_morpher_paired_bot_dims[i])
        ret[cnt++] = TopMorpher::kBottomLengths[i];
    }
    for (index_t i = 0; i < BottomMorpher::bottom_ndim(); i++)
      ret[cnt + i] = BottomMorpher::kBottomLengths[i];
    return ret;
  }();

  constexpr auto new_mtx = [&]() {
    constexpr auto a = TopMorpher::kTopDownZ2Matrix;
    constexpr auto b = BottomMorpher::kTopDownZ2Matrix;
    constexpr index_t kM = a.kM + b.kM - nbit_paired;
    constexpr index_t kN = a.kN + b.kN - nbit_paired;
    z2::z2_matrix<kM, kN> ret;
    ret.fill(0);

    // extract A0, A1, B0, B1
    constexpr auto a0 =
        z2::z2_matrix_extract_rows(a, top_morpher_free_bot_bits);
    constexpr auto a1 =
        z2::z2_matrix_extract_rows(a, top_morpher_paired_bot_bits);
    constexpr auto b0 =
        z2::z2_matrix_extract_columns(b, bot_morpher_free_top_bits);
    constexpr auto b1 =
        z2::z2_matrix_extract_columns(b, bot_morpher_paired_top_bits);

    // be care of sub matrix location:
    // top morher dims goes to higher dims in fused morher,
    // z2_matrix order is in reverse order of dimension order
    // top morher bits goes to lower bits
    z2::z2_matrix_set_sub_matrix(ret, b0, 0, 0);
    z2::z2_matrix_set_sub_matrix(ret, a0, b0.kM, b0.kN);
    z2::z2_matrix_set_sub_matrix(ret, z2::z2_matmul(b1, a1), 0, b0.kN);

    return ret;
  }();

  return z2_linear<new_mtx, new_top_lengths, new_bot_lengths>{};
}

template <class OldMorpher, index_t kN, nd_index<kN> kNewTopLengths>
  requires(is_z2_linear_morpher<OldMorpher>::value)
MINT_HOST_DEVICE constexpr auto reshape_top(
    OldMorpher,
    integral_constant<nd_index<kN>, kNewTopLengths>) {
  return z2_linear<
      OldMorpher::kTopDownZ2Matrix,
      kNewTopLengths,
      OldMorpher::kBottomLengths>{};
}

template <class OldMorpher, index_t kN, nd_index<kN> kNewBottomLengths>
  requires(is_z2_linear_morpher<OldMorpher>::value)
MINT_HOST_DEVICE constexpr auto reshape_bottom(
    OldMorpher,
    integral_constant<nd_index<kN>, kNewBottomLengths>) {
  return z2_linear<
      OldMorpher::kTopDownZ2Matrix,
      OldMorpher::kTopLengths,
      kNewBottomLengths>{};
}

template <class OldMorpher, index_t kN, nd_index<kN> kNewToOldDims>
  requires(
      is_z2_linear_morpher<OldMorpher>::value && OldMorpher::top_ndim() == kN)
MINT_HOST_DEVICE constexpr auto reorder_top(
    OldMorpher,
    integral_constant<nd_index<kN>, kNewToOldDims>) {
  constexpr index_t ndim = OldMorpher::top_ndim();
  constexpr index_t bit_num = OldMorpher::top_bit_total_num_;
  constexpr auto old_bit_nums = OldMorpher::top_bit_nums_;
  constexpr auto new_bit_nums = reorder<kNewToOldDims>(old_bit_nums);
  constexpr auto old_bit_begins = OldMorpher::top_bit_begins_;
  constexpr auto new_bit_begins = OldMorpher::f_idx_bit_begins(new_bit_nums);

  constexpr auto new_to_old_cols = [&]() {
    nd_index<bit_num> ret;
    for (index_t idim = 0; idim < ndim; idim++)
      for (index_t i = 0; i < new_bit_nums[idim]; i++)
        ret[new_bit_begins[idim] + i] = old_bit_begins[kNewToOldDims[idim]] + i;
    return ret;
  }();

  return z2_linear<
      z2::z2_matrix_reorder_columns(
          OldMorpher::kTopDownZ2Matrix, constant<new_to_old_cols>{}),
      generate_array<ndim>(
          [&](auto i) { return OldMorpher::kTopLengths[kNewToOldDims[i]]; }),
      OldMorpher::kBottomLengths>{};
}

template <class OldMorpher, index_t kN, nd_index<kN> kNewToOldDims>
  requires(
      is_z2_linear_morpher<OldMorpher>::value &&
      OldMorpher::bottom_ndim() == kN)
MINT_HOST_DEVICE constexpr auto reorder_bottom(
    OldMorpher,
    integral_constant<nd_index<kN>, kNewToOldDims>) {
  constexpr index_t ndim = OldMorpher::bottom_ndim();
  constexpr index_t bit_num = OldMorpher::bot_bit_total_num_;
  constexpr auto old_bit_nums = OldMorpher::bot_bit_nums_;
  constexpr auto new_bit_nums = reorder<kNewToOldDims>(old_bit_nums);
  constexpr auto old_bit_begins = OldMorpher::bot_bit_begins_;
  constexpr auto new_bit_begins = OldMorpher::f_idx_bit_begins(new_bit_nums);

  constexpr auto new_to_old_rows = [&]() {
    nd_index<bit_num> ret;
    for (index_t idim = 0; idim < ndim; idim++)
      for (index_t i = 0; i < new_bit_nums[idim]; i++)
        ret[new_bit_begins[idim] + i] = old_bit_begins[kNewToOldDims[idim]] + i;
    return ret;
  }();

  return z2_linear<
      z2::z2_matrix_reorder_rows(
          OldMorpher::kTopDownZ2Matrix, constant<new_to_old_rows>{}),
      OldMorpher::kTopLengths,
      generate_array<ndim>([&](auto i) {
        return OldMorpher::kBottomLengths[kNewToOldDims[i]];
      })>{};
}

// old_top_idx[j] =
//   1) new_top_idx[j] ^ new_top_idx[kI], if j = kJ
//   2) new_top_idx[j], if otherwise
template <class OldMorpher, index_t kI, index_t kJ>
  requires(is_z2_linear_morpher<OldMorpher>::value)
MINT_HOST_DEVICE constexpr auto
swizzle_top(OldMorpher, index_constant<kI>, index_constant<kJ>) {
  static_assert(OldMorpher::top_bit_nums_[kI] == OldMorpher::top_bit_nums_[kJ]);

  constexpr index_t num_bit = OldMorpher::top_bit_total_num_;
  constexpr index_t num_swizzle_bit = OldMorpher::top_bit_nums_[kI];

  constexpr auto i_bits = generate_array<num_swizzle_bit>(
      [](auto k) { return k + OldMorpher::top_bit_begins_[kI]; });
  constexpr auto j_bits = generate_array<num_swizzle_bit>(
      [](auto k) { return k + OldMorpher::top_bit_begins_[kJ]; });

  constexpr auto swizzle_mtx =
      z2::make_z2_bitxor_matrix<num_bit, i_bits, j_bits>();

  constexpr auto new_top_down_mtx =
      z2::z2_matmul(OldMorpher::kTopDownZ2Matrix, swizzle_mtx);

  return z2_linear<
      new_top_down_mtx,
      OldMorpher::kTopLengths,
      OldMorpher::kBottomLengths>{};
}

// new_bot_idx[j] =
//   1) old_bot_idx[j] ^ old_bot_idx[kI], if j = kJ
//   2) old_bot_idx[j], if otherwise
template <class OldMorpher, index_t kI, index_t kJ>
  requires(is_z2_linear_morpher<OldMorpher>::value)
MINT_HOST_DEVICE constexpr auto
swizzle_bottom(OldMorpher, index_constant<kI>, index_constant<kJ>) {
  static_assert(OldMorpher::top_bit_nums_[kI] == OldMorpher::top_bit_nums_[kJ]);

  constexpr index_t num_bit = OldMorpher::bot_bit_total_num_;
  constexpr index_t num_swizzle_bit = OldMorpher::bot_bit_nums_[kI];

  constexpr auto i_bits = generate_array<num_swizzle_bit>(
      [](auto k) { return k + OldMorpher::bot_bit_begins_[kI]; });
  constexpr auto j_bits = generate_array<num_swizzle_bit>(
      [](auto k) { return k + OldMorpher::bot_bit_begins_[kJ]; });

  constexpr auto swizzle_mtx =
      z2::make_z2_bitxor_matrix<num_bit, i_bits, j_bits>();

  constexpr auto new_top_down_mtx =
      z2::z2_matmul(swizzle_mtx, OldMorpher::kTopDownZ2Matrix);

  return z2_linear<
      new_top_down_mtx,
      OldMorpher::kTopLengths,
      OldMorpher::kBottomLengths>{};
}

} // namespace mint::poly
