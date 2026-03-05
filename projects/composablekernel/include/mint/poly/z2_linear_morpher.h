#pragma once
#include <mint/core.h>
#include <mint/poly/morpher.h>
#include <mint/poly/z2.h>

namespace mint::poly {

template <auto kTopDownZ2Matrix_, auto kTopLengths_, auto kBottomLengths_>
struct z2_linear
    : morpher<z2_linear<kTopDownZ2Matrix_, kTopLengths_, kBottomLengths_>> {
  static constexpr auto kTopDownZ2Matrix = kTopDownZ2Matrix_;
  static constexpr auto kTopLengths = kTopLengths_;
  static constexpr auto kBottomLengths = kBottomLengths_;
  using base_type =
      morpher<z2_linear<kTopDownZ2Matrix, kTopLengths, kBottomLengths>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ =
      is_z2_matrix_invertible(kTopDownZ2Matrix);
  static constexpr index_t all_ndim_ =
      kTopLengths.size() + kBottomLengths.size();
  static constexpr index_t top_ndim_ = kTopLengths.size();
  static constexpr index_t bottom_ndim_ = kBottomLengths.size();
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = []() {
    array<index_t, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i] = i + bottom_ndim_;
    return ret;
  }();
  static constexpr auto bottom_dims_ = []() {
    array<index_t, bottom_ndim_> ret;
    for (index_t i = 0; i < bottom_ndim_; i++)
      ret[i] = i;
    return ret;
  }();
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr nd_array<bool, top_ndim_, bottom_ndim_> is_linear_top_down_ =
      []() {
        nd_array<bool, top_ndim_, bottom_ndim_> ret;
        for (index_t i = 0; i < top_ndim_; i++)
          for (index_t j = 0; j < bottom_ndim_; j++)
            ret[i][j] = false;
        return ret;
      }();
  static constexpr nd_array<bool, bottom_ndim_, top_ndim_>
      is_linear_bottom_up_ = []() {
        nd_array<bool, bottom_ndim_, top_ndim_> ret;
        for (index_t i = 0; i < bottom_ndim_; i++)
          for (index_t j = 0; j < top_ndim_; j++)
            ret[i][j] = true;
        return ret;
      }();

  static constexpr auto f_idx_bit_nums = [](auto lengths) {
    constexpr index_t ndim = decltype(lengths)::size();
    nd_index<ndim> ret;
    for (index_t i = 0; i < ndim; i++)
      ret[i] = math::integer_log2_ceiling(lengths[i]);
    return ret;
  };

  static constexpr auto f_idx_bit_begins = [](auto bits_nums) {
    constexpr index_t ndim = decltype(bits_nums)::size();
    nd_index<ndim> ret;
    // bits is stored in reverse order in bitset (0-th bit is the smallest)
    ret[ndim - 1] = 0;
    for (index_t i = ndim - 2; i >= 0; i--)
      ret[i] = ret[i + 1] + bits_nums[i + 1];
    return ret;
  };

  static constexpr auto f_idx_bit_ends = [](auto bit_nums) {
    constexpr index_t ndim = decltype(bit_nums)::size();
    nd_index<ndim> ret;
    // bits is stored in reverse order in bitset (0-th bit is the smallest)
    ret[ndim - 1] = bit_nums[ndim - 1];
    for (index_t i = ndim - 2; i >= 0; i--)
      ret[i] = ret[i + 1] + bit_nums[i];
    return ret;
  };

  static constexpr auto top_bit_nums_ = f_idx_bit_nums(kTopLengths);
  static constexpr auto bot_bit_nums_ = f_idx_bit_nums(kBottomLengths);
  static constexpr auto top_bit_begins_ = f_idx_bit_begins(top_bit_nums_);
  static constexpr auto bot_bit_begins_ = f_idx_bit_begins(bot_bit_nums_);
  static constexpr auto top_bit_ends_ = f_idx_bit_ends(top_bit_nums_);
  static constexpr auto bot_bit_ends_ = f_idx_bit_ends(bot_bit_nums_);
  static constexpr index_t top_bit_total_num_ = top_bit_ends_[0];
  static constexpr index_t bot_bit_total_num_ = bot_bit_ends_[0];

  static constexpr auto f_remove_duplicate_impl =
      []<index_t kNa, index_t kNb>(nd_index<kNa> a, nd_index<kNb> b) {
        nd_index<kNa> ret;
        ret.fill(-1);
        index_t cnt = 0;
        for (index_t i = 0; i < kNa; i++)
          if (std::find(b.begin(), b.end(), a[i]) == b.end())
            ret[cnt++] = i;
        return mint::make_tuple(ret, cnt);
      };

  static constexpr auto f_remove_duplicate = []<class A, class B, A kA, B kB>(
                                                 integral_constant<A, kA>,
                                                 integral_constant<B, kB>) {
    constexpr auto arr_cnt = f_remove_duplicate_impl(kA, kB);
    constexpr auto arr = arr_cnt[0_ic];
    constexpr index_t cnt = arr_cnt[1_ic];
    nd_index<cnt> ret;
    for (index_t i = 0; i < cnt; i++)
      ret[i] = arr[i];
    return ret;
  };

  static constexpr auto f_generate_seq = [](auto kBegin, auto kEnd) {
    nd_index<kEnd - kBegin> ret;
    for (index_t i = 0; i < kEnd - kBegin; i++)
      ret[i] = i + kBegin;
    return ret;
  };

  static constexpr auto f_bit_mask = [](auto bit_num, auto bit_locs) {
    bitset<bit_num> ret;
    ret.reset();
    for (index_t i = 0; i < bit_locs.size(); i++)
      ret[bit_locs[i]] = 1;
    return ret;
  };

  constexpr bool operator==(const z2_linear&) const = default;

#if 0
  template <class Index>
    requires(Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index& idx) {
    constexpr index_t top_idx_bit_num = top_bit_ends_[0];
    constexpr index_t bot_idx_bit_num = bot_bit_ends_[0];

    // fill in top index bits
    bitset<top_idx_bit_num> top_bits;
    static_for_n<top_ndim_>()([&](auto i) {
      constexpr index_t shift = top_bit_begins_[i];
      top_bits ^= bitset<top_idx_bit_num>{
          (uint32_t)idx.template at<top_dims_[i]>() << shift};
    });

    const auto bot_bits =
        z2_matrix_dot_bitset(constant<kTopDownZ2Matrix>{}, top_bits);

    // extract bottom index bits
    static_for_n<bottom_ndim_>()([&](auto i) {
      constexpr auto mask = [&]() {
        bitset<bot_idx_bit_num> ret;
        for (index_t j = bot_bit_begins_[i]; j < bot_bit_ends_[i]; j++)
          ret[j] = 1;
        return ret;
      }();
      constexpr index_t shift = bot_bit_begins_[i];
      idx.template at<bottom_dims_[i]>() =
          (mask & bot_bits).to_uint32() >> shift;
    });
  }
#else
  template <class Index>
    requires(Index::size() == all_ndim_ and !is_same_v<Index, nd_index<0>>)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index& idx) {
    constexpr index_t top_bit_num = top_bit_ends_[0];
    constexpr index_t bot_bit_num = bot_bit_ends_[0];

    constexpr auto is_constant_idx = [&]() {
      array<bool, all_ndim_> ret;
      static_for_n<all_ndim_>()(
          [&](auto i) { ret[i] = is_constant_v<decltype(idx[i])>; });
      return ret;
    }();

    // "c": compile-time constant bits, "r": run-time variable bits
    constexpr auto r_c_top_bit_locs = [&]() {
      constexpr index_t c_top_bit_num = [&]() {
        index_t cnt = 0;
        static_for_n<top_ndim_>()([&](auto i) {
          constexpr bool is_const_idx = is_constant_idx[top_dims_[i]];
          if (is_const_idx)
            cnt += top_bit_ends_[i] - top_bit_begins_[i];
        });
        return cnt;
      }();
      constexpr index_t r_top_bit_num = top_bit_num - c_top_bit_num;

      nd_index<r_top_bit_num> r_top_bit_locs;
      nd_index<c_top_bit_num> c_top_bit_locs;

      index_t r_cnt = 0;
      index_t c_cnt = 0;
      static_for_n<top_ndim_>()([&](auto i) {
        if (is_constant_idx[top_dims_[i]]) {
          for (index_t j = top_bit_begins_[i]; j < top_bit_ends_[i]; j++)
            c_top_bit_locs[c_cnt++] = j;
        } else {
          for (index_t j = top_bit_begins_[i]; j < top_bit_ends_[i]; j++)
            r_top_bit_locs[r_cnt++] = j;
        }
      });
      return mint::make_tuple(r_top_bit_locs, c_top_bit_locs);
    }();

    constexpr auto r_top_bit_locs = r_c_top_bit_locs[0_ic];
    constexpr auto c_top_bit_locs = r_c_top_bit_locs[1_ic];

    constexpr auto c_add_top_bit_locs = z2::z2_matrix_orthogonal_columns(
        constant<kTopDownZ2Matrix>{}, constant<r_top_bit_locs>{});
    constexpr auto c_xor_top_bit_locs = f_remove_duplicate(
        constant<c_top_bit_locs>{}, constant<c_add_top_bit_locs>{});

    // c top bits
    constexpr auto c_top_bits = [&]() {
      bitset<top_bit_num> ret;
      static_for_n<top_ndim_>()([&](auto i) {
        constexpr auto idim = constant<top_dims_[i]>{};
        if (is_constant_idx[idim]) {
          constexpr index_t shift = top_bit_begins_[i];
          constexpr index_t value = remove_cvref_t<decltype(idx[idim])>{};
          ret ^= (value << shift);
        }
      });
      return ret;
    }();

    // bot_bits components wrt constant c xor bits
    constexpr auto c_xor_top_bit_mask =
        f_bit_mask(constant<top_bit_num>{}, c_xor_top_bit_locs);
    constexpr auto c_xor_bot_bits = z2::z2_matrix_dot_bitset(
        constant<kTopDownZ2Matrix>{}, c_top_bits & c_xor_top_bit_mask);

    // bot_bits components wrt constant c add bits
    constexpr auto c_add_top_bit_mask =
        f_bit_mask(constant<top_bit_num>{}, c_add_top_bit_locs);
    constexpr auto c_add_bot_bits = z2::z2_matrix_dot_bitset(
        constant<kTopDownZ2Matrix>{}, c_top_bits & c_add_top_bit_mask);

    // bot_bits
    bitset<bot_bit_num> bot_bits;
    bot_bits.reset();

    // bot_bits components wrt variable top bits
    static_for_n<top_ndim_>()([&](auto i) {
      constexpr auto idim = constant<top_dims_[i]>{};
      constexpr bool is_const_idx = is_constant_idx[idim];
      if constexpr (!is_const_idx) {
        constexpr auto top_bit_locs = f_generate_seq(
            constant<top_bit_begins_[i]>{}, constant<top_bit_ends_[i]>{});
        constexpr auto mtx =
            z2::z2_matrix_extract_columns(kTopDownZ2Matrix, top_bit_locs);
        bot_bits ^= z2::z2_matrix_dot_bitset(
            constant<mtx>{}, bitset<mtx.kN>{static_cast<uint32_t>(idx[idim])});
      }
    });

    // bot_bits components wrt c xor top bits
    bot_bits ^= c_xor_bot_bits;

    // bot_bits components wrt c add top bits
    bot_bits += c_add_bot_bits;

#if 0
    auto f_bot_bit_mask = [&](auto i) {
      bitset<bot_bit_num> ret;
      ret.reset();
      for (index_t j = bot_bit_begins_[i]; j < bot_bit_ends_[i]; j++)
        ret[j] = 1;
      return ret;
    };

    // extract bottom index from bot_bits
    uint32_t tmp = bot_bits.to_uint32();

    static_for_n<bottom_ndim_>()([&](auto is) {
      constexpr auto i = constant<bottom_ndim_ - 1 - is>{};
      constexpr auto mask = f_bot_bit_mask(i);
      constexpr auto shift = bot_bit_begins_[i];

      // tmp = left | right
      //   right becomes idx, left becomes new tmp
      const uint32_t right = (tmp & mask.to_uint32()) >> shift;
      idx.template at<i>() = right;
    });
#else
    auto f_bot_bit_mask = [&](auto i) {
      bitset<bot_bit_num> ret;
      ret.reset();
      for (index_t j = 0; j < bot_bit_nums_[i]; j++)
        ret[j] = 1;
      return ret;
    };

    // extract bottom index from bot_bits
    uint32_t tmp = bot_bits.to_uint32();

    static_for_n<bottom_ndim_ - 1>()([&](auto is) {
      constexpr auto i = constant<bottom_ndim_ - 1 - is>{};
      constexpr auto mask = f_bot_bit_mask(i);

      // tmp = left | right
      //   right becomes idx, left becomes new tmp
      const uint32_t right = tmp & mask.to_uint32();
      idx.template at<i>() = right;

      constexpr index_t shift = bot_bit_nums_[i];
      tmp = tmp >> shift;
    });
    idx.template at<0>() = tmp;
#endif
  }
#endif

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) {
    auto bot_idx_old =
        idx.template get_subset<bottom_dims_.size(), bottom_dims_>();

    propagate_index_top_down(idx);

    auto bot_idx_new =
        idx.template get_subset<bottom_dims_.size(), bottom_dims_>();
    auto bot_idx_delta = bot_idx_new - bot_idx_old;

    idx_delta.template set_subset<bot_idx_delta.size(), bottom_dims_>(
        bot_idx_delta);
  }

#if 0
  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index& idx) {
    constexpr auto kBottomUpZ2Matrix = z2_matrix_invert(kTopDownZ2Matrix);

    constexpr index_t top_idx_bit_num = top_bit_ends_[0];
    constexpr index_t bot_idx_bit_num = bot_bit_ends_[0];

    // fill in bottom index bits
    bitset<bot_idx_bit_num> bot_bits;
    static_for_n<bottom_ndim_>()([&](auto i) {
      constexpr index_t shift = bot_bit_begins_[i];
      bot_bits ^=
          bitset<bot_idx_bit_num>{(uint32_t)idx[bottom_dims_[i]] << shift};
    });

    auto top_bits =
        z2_matrix_dot_bitset(constant<kBottomUpZ2Matrix>{}, bot_bits);

    uint32_t tmp = top_bits.to_uint32();

    // extract top index bits
    static_for_n<top_ndim_>()([&](auto i) {
      constexpr auto mask = [&]() {
        bitset<top_idx_bit_num> ret;
        for (index_t j = top_bit_begins_[i]; j < top_bit_ends_[i]; j++)
          ret[j] = 1;
        return ret;
      }();
      constexpr index_t shift = top_bit_begins_[i];
      idx[top_dims_[i]] = (tmp & mask.to_uint32()) >> shift;
    });
  }
#else
  template <class Index>
    requires(Index::size() == all_ndim_ and !is_same_v<Index, nd_index<0>>)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index& idx) {
    constexpr auto kBottomUpZ2Matrix = z2_matrix_invert(kTopDownZ2Matrix);

    constexpr index_t top_bit_num = top_bit_ends_[0];
    constexpr index_t bot_bit_num = bot_bit_ends_[0];

    constexpr auto is_constant_idx = [&]() {
      array<bool, all_ndim_> ret;
      static_for_n<all_ndim_>()(
          [&](auto i) { ret[i] = is_constant_v<decltype(idx[i])>; });
      return ret;
    }();

    // "c": compile-time constant bits, "r": run-time variable bits
    constexpr auto r_c_bot_bit_locs = [&]() {
      constexpr index_t c_bot_bit_num = [&]() {
        index_t cnt = 0;
        static_for_n<bottom_ndim_>()([&](auto i) {
          constexpr bool is_const_idx = is_constant_idx[bottom_dims_[i]];
          if (is_const_idx)
            cnt += bot_bit_ends_[i] - bot_bit_begins_[i];
        });
        return cnt;
      }();
      constexpr index_t r_bot_bit_num = bot_bit_num - c_bot_bit_num;

      nd_index<r_bot_bit_num> r_bot_bit_locs;
      nd_index<c_bot_bit_num> c_bot_bit_locs;

      index_t r_cnt = 0;
      index_t c_cnt = 0;
      static_for_n<bottom_ndim_>()([&](auto i) {
        if (is_constant_idx[bottom_dims_[i]]) {
          for (index_t j = bot_bit_begins_[i]; j < bot_bit_ends_[i]; j++)
            c_bot_bit_locs[c_cnt++] = j;
        } else {
          for (index_t j = bot_bit_begins_[i]; j < bot_bit_ends_[i]; j++)
            r_bot_bit_locs[r_cnt++] = j;
        }
      });
      return mint::make_tuple(r_bot_bit_locs, c_bot_bit_locs);
    }();

    constexpr auto r_bot_bit_locs = r_c_bot_bit_locs[0_ic];
    constexpr auto c_bot_bit_locs = r_c_bot_bit_locs[1_ic];

    constexpr auto c_add_bot_bit_locs = z2::z2_matrix_orthogonal_columns(
        constant<kBottomUpZ2Matrix>{}, constant<r_bot_bit_locs>{});
    constexpr auto c_xor_bot_bit_locs = f_remove_duplicate(
        constant<c_bot_bit_locs>{}, constant<c_add_bot_bit_locs>{});

    // c bot bits
    constexpr auto c_bot_bits = [&]() {
      bitset<bot_bit_num> ret;
      static_for_n<bottom_ndim_>()([&](auto i) {
        constexpr auto idim = constant<bottom_dims_[i]>{};
        if (is_constant_idx[idim]) {
          constexpr index_t shift = bot_bit_begins_[i];
          constexpr index_t value = remove_cvref_t<decltype(idx[idim])>{};
          ret ^= (value << shift);
        }
      });
      return ret;
    }();

    // top_bits components wrt constant c xor bits
    constexpr auto c_xor_bot_bit_mask =
        f_bit_mask(constant<bot_bit_num>{}, c_xor_bot_bit_locs);
    constexpr auto top_bit_row_range_for_c_xor =
        f_generate_seq(constant<0>{}, constant<top_bit_num>{});
    constexpr auto kBottomUpZ2Matrix_truncated_for_c_xor =
        z2::z2_matrix_extract_rows(
            kBottomUpZ2Matrix, top_bit_row_range_for_c_xor);
    constexpr auto c_xor_top_bits = z2::z2_matrix_dot_bitset(
        constant<kBottomUpZ2Matrix_truncated_for_c_xor>{},
        c_bot_bits & c_xor_bot_bit_mask);

    // top_bits components wrt constant c add bits
    constexpr auto c_add_bot_bit_mask =
        f_bit_mask(constant<bot_bit_num>{}, c_add_bot_bit_locs);
    constexpr auto top_bit_row_range_for_c_add =
        f_generate_seq(constant<0>{}, constant<top_bit_num>{});
    constexpr auto kBottomUpZ2Matrix_truncated_for_c_add =
        z2::z2_matrix_extract_rows(
            kBottomUpZ2Matrix, top_bit_row_range_for_c_add);
    constexpr auto c_add_top_bits = z2::z2_matrix_dot_bitset(
        constant<kBottomUpZ2Matrix_truncated_for_c_add>{},
        c_bot_bits & c_add_bot_bit_mask);

    // top_bits
    bitset<top_bit_num> top_bits;
    top_bits.reset();

    // top_bits components wrt variable bot bits
    static_for_n<bottom_ndim_>()([&](auto i) {
      constexpr auto idim = constant<bottom_dims_[i]>{};
      constexpr bool is_const_idx = is_constant_idx[idim];
      if constexpr (!is_const_idx) {
        constexpr auto bot_bit_locs = f_generate_seq(
            constant<bot_bit_begins_[i]>{}, constant<bot_bit_ends_[i]>{});
        constexpr auto top_bit_row_range =
            f_generate_seq(constant<0>{}, constant<top_bit_num>{});
        constexpr auto mtx = z2::z2_matrix_extract_sub_matrix(
            kBottomUpZ2Matrix, top_bit_row_range, bot_bit_locs);
        top_bits ^= z2::z2_matrix_dot_bitset(
            constant<mtx>{}, bitset<mtx.kN>{static_cast<uint32_t>(idx[idim])});
      }
    });

    // top_bits components wrt c xor bot bits
    top_bits ^= c_xor_top_bits;

    // top_bits components wrt c add bot bits
    top_bits += c_add_top_bits;
#if 0
    auto f_top_bit_mask = [&](auto i) {
      bitset<top_bit_num> ret;
      ret.reset();
      for (index_t j = top_bit_begins_[i]; j < top_bit_ends_[i]; j++)
        ret[j] = 1;
      return ret;
    };

    // extract top index from top_bits
    uint32_t tmp = top_bits.to_uint32();

    static_for_n<top_ndim_>()([&](auto is) {
      constexpr auto i = constant<top_ndim_ - 1 - is>{};
      constexpr auto mask = f_top_bit_mask(i);
      constexpr auto shift = top_bit_begins_[i];

      // tmp = left | right
      //   right becomes idx, left becomes new tmp
      const uint32_t right = (tmp & mask.to_uint32()) >> shift;
      idx.template at<top_dims_[i]>() = right;
    });
#else
    auto f_top_bit_mask = [&](auto i) {
      bitset<top_bit_num> ret;
      ret.reset();
      for (index_t j = 0; j < top_bit_nums_[i]; j++)
        ret[j] = 1;
      return ret;
    };

    // extract bottom index from bot_bits
    uint32_t tmp = top_bits.to_uint32();

    static_for_n<top_ndim_ - 1>()([&](auto is) {
      constexpr auto i = constant<top_ndim_ - 1 - is>{};
      constexpr auto mask = f_top_bit_mask(i);

      // tmp = left | right
      //   right becomes idx, left becomes new tmp
      const uint32_t right = tmp & mask.to_uint32();
      idx.template at<top_dims_[i]>() = right;

      constexpr index_t shift = top_bit_nums_[i];
      tmp = tmp >> shift;
    });
    idx.template at<top_dims_[0]>() = tmp;
#endif
  }
#endif
  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) {
    auto top_idx_old = idx.template get_subset<top_dims_.size(), top_dims_>();

    propagate_index_bottom_up(idx);

    auto top_idx_new = idx.template get_subset<top_dims_.size(), top_dims_>();
    auto top_idx_delta = top_idx_new - top_idx_old;

    idx_delta.template set_subset<top_dims_.size(), top_dims_>(top_idx_delta);
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::z2_linear: {");
    base_type::print();
    printf("kTopDownZ2Matrix: ");
    kTopDownZ2Matrix_.print();
    printf("kTopLengths: ");
    kTopLengths_.print();
    printf(", ");
    printf("kBottomLengths: ");
    kBottomLengths_.print();
    printf(", ");
    printf("}");
  }
};

} // namespace mint::poly
