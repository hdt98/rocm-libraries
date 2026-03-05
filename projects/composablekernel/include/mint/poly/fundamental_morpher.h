#pragma once
#include <mint/core.h>
#include <mint/poly/morpher.h>
#include <mint/poly/z2_linear_morpher.h>

namespace mint::poly {

struct none : morpher<none> {
  using base_type = morpher<none>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = 1;
  static constexpr index_t top_ndim_ = 1;
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = array<index_t, 1>{0};
  static constexpr auto bottom_dims_ = array<index_t, 1>{0};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 1, 1>{{1}};
  static constexpr auto is_linear_bottom_up_ = nd_array<bool, 1, 1>{{1}};

  constexpr bool operator==(const none&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index&) {}

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      Index&,
      IndexDelta&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      Index&,
      IndexDelta&) {}

  MINT_HOST_DEVICE void print() const {
    printf("poly::none: {");
    base_type::print();
    printf("}");
  }
};

struct pass_through : morpher<pass_through> {
  using base_type = morpher<pass_through>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = 2;
  static constexpr index_t top_ndim_ = 1;
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = array<index_t, 1>{1};
  static constexpr auto bottom_dims_ = array<index_t, 1>{0};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 1, 1>{{1}};
  static constexpr auto is_linear_bottom_up_ = nd_array<bool, 1, 1>{{1}};

  constexpr bool operator==(const pass_through&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index& idx) {
    idx[0] = idx[1];
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index& idx) {
    idx[1] = idx[0];
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) {
    idx[0] = idx[1];
    idx_delta[0] = idx_delta[1];
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) {
    idx[1] = idx[0];
    idx_delta[1] = idx_delta[0];
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::pass_through: {");
    base_type::print();
    printf("}");
  }
};

template <class ShiftValue>
struct shift : morpher<shift<ShiftValue>> {
  using base_type = morpher<shift<ShiftValue>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = 2;
  static constexpr index_t top_ndim_ = 1;
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 1;
  static constexpr auto top_dims_ = array<index_t, 1>{1};
  static constexpr auto bottom_dims_ = array<index_t, 1>{0};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 1, 1>{{1}};
  static constexpr auto is_linear_bottom_up_ = nd_array<bool, 1, 1>{{1}};

  constexpr shift() = default;

  MINT_HOST_DEVICE constexpr shift(ShiftValue shift_value)
      : shift_value_{shift_value} {}

  constexpr bool operator==(const shift&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    idx[0] = idx[1] + static_cast<index_t>(shift_value_);
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_bottom_up(Index& idx) const {
    idx[1] = idx[0] - static_cast<index_t>(shift_value_);
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    idx[0] = idx[1] + static_cast<index_t>(shift_value_);
    idx_delta[0] = idx_delta[1];
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) const {
    idx[1] = idx[0] - static_cast<index_t>(shift_value_);
    idx_delta[1] = idx_delta[0];
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::shift: {");
    base_type::print();
    printf(", ");
    print_item(shift_value_);
    printf("}");
  }

  const ShiftValue shift_value_{};
};

template <class BottomLengths>
  requires(BottomLengths::size() > 0)
struct split : morpher<split<BottomLengths>> {
  using base_type = morpher<split<BottomLengths>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = BottomLengths::size() + 1;
  static constexpr index_t top_ndim_ = 1;
  static constexpr index_t bottom_ndim_ = BottomLengths::size();
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = array<index_t, 1>{bottom_ndim_};
  static constexpr auto bottom_dims_ = []() {
    array<index_t, bottom_ndim_> ret;
    for (index_t i = 0; i < bottom_ndim_; i++)
      ret[i] = i;
    return ret;
  }();
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr nd_array<bool, 1, bottom_ndim_> is_linear_top_down_ = []() {
    nd_array<bool, 1, bottom_ndim_> ret;
    for (index_t i = 0; i < bottom_ndim_; i++)
      ret[0][i] = false;
    return ret;
  }();
  static constexpr nd_array<bool, bottom_ndim_, 1> is_linear_bottom_up_ = []() {
    nd_array<bool, bottom_ndim_, 1> ret;
    for (index_t i = 0; i < bottom_ndim_; i++)
      ret[i][0] = true;
    return ret;
  }();

#if 0
  static constexpr auto cal_bottom_length_scan_ =
      [](const nd_index<bottom_ndim_>& bot_lengths) {
        nd_index<bottom_ndim_ - 1> ret;
        if constexpr (bottom_ndim_ > 1)
          ret[bottom_ndim_ - 2] = bot_lengths[bottom_ndim_ - 1];

        for (index_t i = bottom_ndim_ - 3; i >= 0; i--)
          ret[i] = bot_lengths[i + 1] * ret[i + 1];
        return ret;
      };
#else
  // nvcc doesn't allow static constexpr lambda to be used on device, yet
  MINT_HOST_DEVICE constexpr auto cal_bottom_length_scan(
      const nd_index<bottom_ndim_>& bot_lengths) {
    nd_index<bottom_ndim_ - 1> ret;
    if constexpr (bottom_ndim_ > 1)
      ret[bottom_ndim_ - 2] = bot_lengths[bottom_ndim_ - 1];

    for (index_t i = bottom_ndim_ - 3; i >= 0; i--)
      ret[i] = bot_lengths[i + 1] * ret[i + 1];
    return ret;
  }
#endif

  constexpr split() = default;

  MINT_HOST_DEVICE constexpr split(BottomLengths bottom_lengths_in)
      : bottom_length_scan_{cal_bottom_length_scan(bottom_lengths_in)},
        length_divisors_{elementwise_nd_containers_in<
            array<math::fast_div_mod, bottom_ndim_>>(
            static_for_n<bottom_ndim_>(),
            [](auto len) { return math::fast_div_mod{len}; },
            bottom_lengths_in)} {}

#if 0
  MINT_HOST_DEVICE constexpr bool operator==(const split&) const = default;
#else
  // workaround nvcc limitation: no default operator for device code
  MINT_HOST_DEVICE constexpr bool operator==(const split& other) const {
    return bottom_length_scan_ == other.bottom_length_scan_ &&
        length_divisors_ == other.length_divisors_;
  }
#endif

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    index_t cnt = idx[bottom_ndim_];
    static_for_n<bottom_ndim_ - 1>()([&](auto is) {
      constexpr index_t i = bottom_ndim_ - 1 - is;
      length_divisors_[i](cnt, idx[i], cnt);
    });
    idx[0] = cnt;
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_bottom_up(Index& idx) const {
    index_t cnt = idx[bottom_ndim_ - 1];
    static_for_n<bottom_ndim_ - 1>()(
        [&](auto i) { cnt += bottom_length_scan_[i] * idx[i]; });
    idx[bottom_ndim_] = cnt;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    index_t cnt = idx[bottom_ndim_];
    static_for_n<bottom_ndim_ - 1>()([&](auto is) {
      constexpr index_t i = bottom_ndim_ - 1 - is;
      index_t tmp;
      length_divisors_[i](cnt, tmp, cnt);
      idx_delta[i] = tmp - idx[i];
      idx[i] = tmp;
    });
    idx_delta[0] = cnt - idx[0];
    idx[0] = cnt;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) const {
    index_t cnt = idx_delta[bottom_ndim_ - 1];
    static_for_n<bottom_ndim_ - 1>()(
        [&](auto i) { cnt += bottom_length_scan_[i] * idx_delta[i]; });
    idx_delta[bottom_ndim_] = cnt;
    idx[bottom_ndim_] += idx_delta[bottom_ndim_];
  }

  template <auto kIsFreezedDim, class Index, class IndexDelta>
    requires(
        is_same_v<typename decltype(kIsFreezedDim)::value_type, bool> &&
        kIsFreezedDim.size() == all_ndim_ &&
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void
  propagate_index_and_delta_top_down_freezed_dim_conjectural(
      Index& idx,
      IndexDelta& idx_delta) const {
#if 0
    base_type::
        template propagate_index_and_delta_top_down_freezed_dim_conjectural<
            kIsFreezedDim>(idx, idx_delta);
#elif 1
    constexpr bool freeze_all_bottom_except_last =
        std::all_of(
            kIsFreezedDim.begin(),
            kIsFreezedDim.begin() + bottom_ndim_ - 1,
            [](bool v) { return v; }) &&
        (!kIsFreezedDim[bottom_ndim_ - 1]);

    // FIXME:
    constexpr bool hack_0_1_bottom_freeze_1 = (bottom_ndim_ == 2) &&
        (kIsFreezedDim[0] == false) && (kIsFreezedDim[1] == true);

    if constexpr (hack_0_1_bottom_freeze_1) {
      idx_delta[1] = 0;
      index_t tmp;
      length_divisors_[1](idx_delta[0], tmp, idx_delta[2]);
      idx[0] += idx_delta[0];
    } else if constexpr (freeze_all_bottom_except_last) {
      static_for_n<bottom_ndim_ - 1>()([&](auto i) { idx_delta[i] = 0; });
      idx_delta[bottom_ndim_ - 1] = idx_delta[bottom_ndim_];
    } else {
      base_type::template propagate_index_and_delta_top_down_freezed_dim<
          kIsFreezedDim>(idx, idx_delta);
    }
#else // good ISA
    static_for_n<bottom_ndim_ - 1>()([&](auto i) { idx_delta[i] = 0; });
    idx_delta[bottom_ndim_ - 1] = idx_delta[bottom_ndim_];
#endif
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::split: {");
    base_type::print();
    printf(", ");
    printf("bottom_ndim_: %d, ", bottom_ndim_);
    printf("bottom_length_scan_: {");
    print_item(bottom_length_scan_);
    printf("}");
    printf("}");
  }

  const nd_index<bottom_ndim_ - 1> bottom_length_scan_{};
  const array<math::fast_div_mod, bottom_ndim_> length_divisors_{};
};

template <class TopLengths>
  requires(TopLengths::size() > 0)
struct merge : morpher<merge<TopLengths>> {
  using base_type = morpher<merge<TopLengths>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = TopLengths::size() + 1;
  static constexpr index_t top_ndim_ = TopLengths::size();
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = []() {
    array<index_t, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i] = i + 1;
    return ret;
  }();
  static constexpr auto bottom_dims_ = array<index_t, 1>{0};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr nd_array<bool, top_ndim_, 1> is_linear_top_down_ = []() {
    nd_array<bool, top_ndim_, 1> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i][0] = true;
    return ret;
  }();
  static constexpr nd_array<bool, 1, top_ndim_> is_linear_bottom_up_ = []() {
    nd_array<bool, 1, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[0][i] = false;
    return ret;
  }();

  MINT_HOST_DEVICE constexpr auto cal_top_length_scan(
      const nd_index<top_ndim_>& top_lengths) {
    nd_index<top_ndim_ - 1> ret;
    if constexpr (top_ndim_ > 1)
      ret[top_ndim_ - 2] = top_lengths[top_ndim_ - 1];

    for (index_t i = top_ndim_ - 3; i >= 0; i--)
      ret[i] = top_lengths[i + 1] * ret[i + 1];
    return ret;
  }

  constexpr merge() = default;

  MINT_HOST_DEVICE constexpr merge(TopLengths top_lengths_in)
      : top_length_scan_{cal_top_length_scan(top_lengths_in)},
        length_divisors_{
            elementwise_nd_containers_in<array<math::fast_div_mod, top_ndim_>>(
                static_for_n<top_ndim_>(),
                [](auto len) { return math::fast_div_mod{len}; },
                top_lengths_in)} {}

  constexpr bool operator==(const merge&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    index_t cnt = idx[all_ndim_ - 1];
    static_for_n<top_ndim_ - 1>()(
        [&](auto i) { cnt += idx[1 + i] * top_length_scan_[i]; });
    idx[0] = cnt;
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_bottom_up(Index& idx) const {
    index_t cnt = idx[0];
    static_for_n<top_ndim_ - 1>()([&](auto is) {
      constexpr index_t i = top_ndim_ - is;
      length_divisors_[i - 1](cnt, idx[i], cnt);
    });
    idx[1] = cnt;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    index_t cnt = idx_delta[all_ndim_ - 1];
    static_for_n<top_ndim_ - 1>()(
        [&](auto i) { cnt += idx_delta[1 + i] * top_length_scan_[i]; });
    idx_delta[0] = cnt;
    idx[0] += idx_delta[0];
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) const {
    index_t cnt = idx[0];
    static_for_n<top_ndim_ - 1>()([&](auto is) {
      constexpr index_t i = top_ndim_ - is;
      index_t tmp;
      length_divisors_[i - 1](cnt, tmp, cnt);
      idx_delta[i] = tmp - idx[i];
      idx[i] = tmp;
    });
    idx_delta[1] = cnt - idx[1];
    idx[1] = cnt;
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::merge: {");
    base_type::print();
    printf(", ");
    printf("top_ndim_: %d, ", top_ndim_);
    printf("top_length_scan_: {");
    print_item(top_length_scan_);
    printf("}");
    printf("}");
  }

  const nd_index<top_ndim_ - 1> top_length_scan_{};
  const array<math::fast_div_mod, top_ndim_> length_divisors_{};
};

template <class Coefficients>
  requires(Coefficients::size() > 0)
struct project : morpher<project<Coefficients>> {
  using base_type = morpher<project<Coefficients>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = false;
  static constexpr index_t all_ndim_ = Coefficients::size() + 1;
  static constexpr index_t top_ndim_ = Coefficients::size();
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = []() {
    array<index_t, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i] = i + 1;
    return ret;
  }();
  static constexpr auto bottom_dims_ = array<index_t, 1>{0};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr nd_array<bool, top_ndim_, 1> is_linear_top_down_ = []() {
    nd_array<bool, top_ndim_, 1> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i][0] = true;
    return ret;
  }();
  static constexpr nd_array<bool, 1, top_ndim_> is_linear_bottom_up_ = []() {
    nd_array<bool, 1, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[0][i] = false;
    return ret;
  }();

  constexpr project() = default;

  MINT_HOST_DEVICE constexpr project(Coefficients coefficients)
      : coefficients_{coefficients} {}

  constexpr bool operator==(const project&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    index_t cnt = 0;
    static_for_n<top_ndim_>()(
        [&](auto i) { cnt += idx[i + 1] * coefficients_[i]; });
    idx[0] = cnt;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    index_t cnt = 0;
    static_for_n<top_ndim_>()(
        [&](auto i) { cnt += idx_delta[1 + i] * coefficients_[i]; });
    idx_delta[0] = cnt;
    idx[0] += idx_delta[0];
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::project: {");
    base_type::print();
    printf(", ");
    printf("top_ndim_: %d, ", top_ndim_);
    printf("coefficients_: {");
    print_item(coefficients_);
    printf("}");
    printf("}");
  }

  const nd_index<top_ndim_> coefficients_{};
};

template <index_t kN>
struct insert : morpher<insert<kN>> {
  using base_type = morpher<insert<kN>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = false;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = kN;
  static constexpr index_t top_ndim_ = 0;
  static constexpr index_t bottom_ndim_ = kN;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = array<index_t, 0>{};
  static constexpr auto bottom_dims_ = []() {
    array<index_t, bottom_ndim_> ret;
    for (index_t i = 0; i < bottom_ndim_; i++)
      ret[i] = i;
    return ret;
  }();
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 0, bottom_ndim_>{};
  static constexpr auto is_linear_bottom_up_ =
      nd_array<bool, bottom_ndim_, 0>{};

  constexpr bool operator==(const insert&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index&) {}

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      Index&,
      IndexDelta&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      Index&,
      IndexDelta&) {}

  MINT_HOST_DEVICE void print() const {
    printf("poly::insert: {");
    base_type::print();
    printf(", ");
    printf("bottom_ndim_ = %d", bottom_ndim_);
    printf("}");
  }
};

template <index_t kN>
struct remove : morpher<remove<kN>> {
  using base_type = morpher<remove<kN>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = false;
  static constexpr index_t all_ndim_ = kN;
  static constexpr index_t top_ndim_ = kN;
  static constexpr index_t bottom_ndim_ = 0;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = []() {
    array<index_t, top_ndim_> ret;
    for (index_t i = 0; i < top_ndim_; i++)
      ret[i] = i;
    return ret;
  }();
  static constexpr auto bottom_dims_ = array<index_t, 0>{};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 0, bottom_ndim_>{};
  static constexpr auto is_linear_bottom_up_ =
      nd_array<bool, bottom_ndim_, 0>{};

  constexpr bool operator==(const remove&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(Index&) {}

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(Index&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      Index&,
      IndexDelta&) {}

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      Index&,
      IndexDelta&) {}

  MINT_HOST_DEVICE void print() const {
    printf("poly::remove: {");
    base_type::print();
    printf(", ");
    printf("top_ndim_ = %d", top_ndim_);
    printf("}");
  }
};

struct insert_length_one : morpher<insert_length_one> {
  using base_type = morpher<insert_length_one>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = 1;
  static constexpr index_t top_ndim_ = 0;
  static constexpr index_t bottom_ndim_ = 1;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = nd_index<0>{};
  static constexpr auto bottom_dims_ = nd_index<1>{0};
  static constexpr auto paired_dims_ = nd_index<0>{};
  static constexpr auto is_linear_top_down_ = nd_array<bool, 0, 1>{};
  static constexpr auto is_linear_bottom_up_ = nd_array<bool, 1, 0>{};

  constexpr bool operator==(const insert_length_one&) const = default;

  MINT_HOST_DEVICE static constexpr void propagate_index_top_down(
      nd_index<1>& idx) {
    idx[0] = 0;
  }

  MINT_HOST_DEVICE static constexpr void propagate_index_bottom_up(
      nd_index<1>&) {}

  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_top_down(
      nd_index<1>& idx,
      nd_index<1>& idx_delta) {
    idx[0] = 0;
    idx_delta[0] = 0;
  }

  MINT_HOST_DEVICE static constexpr void propagate_index_and_delta_bottom_up(
      nd_index<1>&,
      nd_index<1>&) {}

  MINT_HOST_DEVICE void print() const {
    printf("poly::insert_length_one: {");
    base_type::print();
    printf("}");
  }
};

// rotate_bound_ needs to be power of 2 otherwise won't be efficient
template <class RotateBound, class RotateStep>
struct rotate2d : morpher<rotate2d<RotateBound, RotateStep>> {
  using base_type = morpher<rotate2d<RotateBound, RotateStep>>;
  static constexpr bool is_fundamental_ = true;
  static constexpr bool can_top_down_ = true;
  static constexpr bool can_bottom_up_ = true;
  static constexpr index_t all_ndim_ = 4;
  static constexpr index_t top_ndim_ = 2;
  static constexpr index_t bottom_ndim_ = 2;
  static constexpr index_t paired_ndim_ = 0;
  static constexpr auto top_dims_ = array<index_t, 2>{2, 3};
  static constexpr auto bottom_dims_ = array<index_t, 2>{0, 1};
  static constexpr auto paired_dims_ = array<index_t, 0>{};
  static constexpr auto is_linear_top_down_ =
      nd_array<bool, 2, 2>{{1, 0}, {0, 0}};
  static constexpr auto is_linear_bottom_up_ =
      nd_array<bool, 2, 2>{{1, 0}, {0, 0}};

  constexpr rotate2d() = default;

  MINT_HOST_DEVICE constexpr rotate2d(
      RotateBound rotate_bound,
      RotateStep rotate_step)
      : rotate_bound_{rotate_bound}, rotate_step_{rotate_step} {}

  constexpr bool operator==(const rotate2d&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    idx[0] = idx[2];
    idx[1] = (idx[3] - idx[2] * rotate_step_) % rotate_bound_;
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_bottom_up(Index& idx) const {
    idx[2] = idx[0];
    idx[3] = (idx[1] + idx[0] * rotate_step_) % rotate_bound_;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    idx[0] = idx[2];
    index_t tmp = (idx[3] - idx[2] * rotate_step_) % rotate_bound_;
    idx_delta[0] = idx_delta[2];
    idx_delta[1] = tmp - idx[1];
    idx[1] = tmp;
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) const {
    idx[2] = idx[0];
    index_t tmp = (idx[1] + idx[0] * rotate_step_) % rotate_bound_;
    idx_delta[2] = idx_delta[0];
    idx_delta[3] = tmp - idx[3];
    idx[3] = tmp;
  }

  MINT_HOST_DEVICE void print() const {
    printf("poly::rotate2d: {");
    base_type::print();
    printf("}");
  }

  const RotateBound rotate_bound_{};
  const RotateStep rotate_step_{};
};

} // namespace mint::poly
