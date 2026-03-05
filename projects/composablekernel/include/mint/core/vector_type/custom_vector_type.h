#pragma once
#include <mint/config.h>
#include <mint/core/arithmetic_type.h>
#include <mint/core/as_type.h>
#include <mint/core/index_t.h>
#include <mint/core/same_tuple.h>
#include <mint/core/tuple.h>
#include <mint/core/type_traits.h>

namespace mint {

// custom_vector_type
//   impl with same_tuple
template <class S, index_t kNS>
struct alignas(sizeof(S) * kNS) custom_vector_type {
  alignas(sizeof(S) * kNS) same_tuple<S, kNS> data_;

  constexpr custom_vector_type() = default;

  MINT_HOST_DEVICE constexpr custom_vector_type(initializer_list<S> init)
      : data_{init} {}

  MINT_HOST_DEVICE void fill(S s) {
    data_.fill(s);
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr const S& operator[](index_constant<kI>) const {
    return data_.template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& operator[](index_constant<kI>) {
    return data_.template at<kI>();
  }
  template <index_t kI>
  MINT_HOST_DEVICE constexpr const S& at() const {
    return data_.template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& at() {
    return data_.template at<kI>();
  }

  MINT_HOST_DEVICE constexpr const auto& as_scalars() const {
    return data_;
  }

  MINT_HOST_DEVICE constexpr auto& as_scalars() {
    return data_;
  }

  template <index_t kSPerX>
    requires(kNS % kSPerX == 0)
  MINT_HOST_DEVICE constexpr auto as_vectors() const
      -> const same_tuple<custom_vector_type<S, kSPerX>, kNS / kSPerX>& {
    return as_type<
        const same_tuple<custom_vector_type<S, kSPerX>, kNS / kSPerX>>(data_);
  }

  template <index_t kSPerX>
    requires(kNS % kSPerX == 0)
  MINT_HOST_DEVICE constexpr auto as_vectors()
      -> same_tuple<custom_vector_type<S, kSPerX>, kNS / kSPerX>& {
    return as_type<same_tuple<custom_vector_type<S, kSPerX>, kNS / kSPerX>>(
        data_);
  }
};

} // namespace mint
