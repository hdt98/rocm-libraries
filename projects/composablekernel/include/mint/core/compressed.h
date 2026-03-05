#pragma once
#include <mint/core/array.h>
#include <mint/core/math.h>
#include <mint/core/nd_array.h>
#include <mint/core/nd_index.h>
#include <mint/core/tuple.h>
#include <mint/core/type_traits.h>

namespace mint {

template <index_t kM, nd_index<kM> kNumNs>
struct compressed_row {
 private:
  static constexpr index_t kNumTotal_ = []() {
    index_t ret = 0;
    for (index_t m = 0; m < kM; m++)
      ret += kNumNs[m];
    return ret;
  }();
  static constexpr auto m2o_ = []() {
    array<index_t, kM + 1> ret{0};
    for (index_t m = 0; m < kM; m++)
      ret[m + 1] = ret[m] + kNumNs[m];
    return ret;
  }();
  static constexpr auto o2m_ = []() {
    array<index_t, kNumTotal_> ret;
    for (index_t m = 0; m < kM; m++)
      for (index_t n = 0; n < kNumNs[m]; n++)
        ret[m2o_[m] + n] = m;
    return ret;
  }();

 public:
  static constexpr index_t num_m() {
    return kM;
  }
  static constexpr auto num_ns() {
    return kNumNs;
  }
  static constexpr index_t num_total() {
    return kNumTotal_;
  }

  static constexpr auto m_to_o_begins() {
    return m2o_;
  }

  static constexpr index_t m_to_o_begin(index_t m) {
    return m2o_[m];
  }

  static constexpr index_t m_to_o_end(index_t m) {
    return m2o_[m + 1];
  }

  static constexpr index_t mn_to_o(index_t m, index_t n) {
    return m2o_[m] + n;
  }

  static constexpr nd_index<2> o_to_mn(index_t o) {
    index_t m = o2m_[o];
    index_t n = o - m2o_[m];
    return {m, n};
  }
};

template <index_t kNumOriginal, array<bool, kNumOriginal> kMasks>
struct compressed_mask {
 private:
  static constexpr auto tmp =
      []() -> tuple<index_t, array<index_t, kNumOriginal>> {
    index_t cnt = 0;
    array<index_t, kNumOriginal> original2compressed;
    original2compressed.fill(-1);
    for (index_t i = 0; i < kNumOriginal; i++) {
      if (kMasks[i])
        original2compressed[i] = cnt++;
    }
    return {cnt, original2compressed};
  }();
  static constexpr index_t num_compressed_ = tmp.template at<0>();
  static constexpr auto original2compressed_ = tmp.template at<1>();
  static constexpr auto compressed2original_ = []() {
    array<index_t, num_compressed_> ret;
    for (index_t i = 0; i < kNumOriginal; i++)
      if (kMasks[i])
        ret[original2compressed_[i]] = i;
    return ret;
  }();

 public:
  static constexpr index_t num_original() {
    return kNumOriginal;
  }

  static constexpr index_t num_compressed() {
    return num_compressed_;
  }

  static constexpr auto original_to_compressed() {
    return original2compressed_;
  }

  static constexpr auto compressed_to_original() {
    return compressed2original_;
  }
};

template <
    index_t kNOriginal,
    index_t kNPair,
    nd_array<index_t, kNPair, 2> kDuplicatedPairs>
struct compressed_unique {
 private:
  static constexpr auto original_to_unqiue_data_ =
      []() -> tuple<index_t, nd_index<kNOriginal>> {
    nd_index<kNOriginal> original_to_unique;

    for (index_t i = 0; i < kNOriginal; i++)
      original_to_unique[i] = i;

    for (index_t i = 0; i < kDuplicatedPairs.lengths()[0]; i++) {
      const index_t j0 =
          math::min(kDuplicatedPairs[i][0], kDuplicatedPairs[i][1]);
      const index_t j1 =
          math::max(kDuplicatedPairs[i][0], kDuplicatedPairs[i][1]);
      original_to_unique[j1] = j0;
    }

    index_t cnt = 0;
    for (index_t i = 0; i < kNOriginal; i++) {
      const index_t itmp = original_to_unique[i];
      if (itmp == i)
        original_to_unique[i] = cnt++;
      else
        original_to_unique[i] = original_to_unique[itmp];
    }

    return {cnt, original_to_unique};
  }();

 public:
  static constexpr index_t num_original() {
    return kNOriginal;
  }

  static constexpr index_t num_unique() {
    return original_to_unqiue_data_.template at<0>();
  }

  static constexpr auto original_to_unique() -> nd_index<num_original()> {
    return original_to_unqiue_data_.template at<1>();
  }

 private:
  static constexpr auto unique_to_original_data_ =
      []() -> tuple<
               nd_index<num_unique()>,
               nd_index<num_unique() + 1>,
               nd_index<num_original()>> {
    nd_index<num_unique()> nums{};
    nd_index<num_unique() + 1> ia{};
    nd_index<num_original()> data{};

    nums.fill(0);
    for (index_t j = 0; j < num_original(); j++)
      nums[original_to_unique()[j]]++;

    ia[0] = 0;
    for (index_t i = 0; i < num_unique(); i++)
      ia[i + 1] = ia[i] + nums[i];

    auto cnts = ia;
    for (index_t j = 0; j < num_original(); j++)
      data[cnts[original_to_unique()[j]]++] = j;

    return {nums, ia, data};
  }();

 public:
  static constexpr auto num_duplicates() -> nd_index<num_unique()> {
    return unique_to_original_data_.template at<0>();
  }

  static constexpr auto unique_to_original_begin()
      -> nd_index<num_unique() + 1> {
    return unique_to_original_data_.template at<1>();
  }

  static constexpr auto unique_to_original_data() -> nd_index<num_original()> {
    return unique_to_original_data_.template at<2>();
  }

  static constexpr index_t unique_to_original(
      index_t i_unique,
      index_t j_duplicate) {
    const index_t j_begin = unique_to_original_begin()[i_unique];
    return unique_to_original_data()[j_begin + j_duplicate];
  }
};

} // namespace mint
