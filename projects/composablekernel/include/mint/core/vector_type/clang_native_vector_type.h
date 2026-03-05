#pragma once
#include <mint/config.h>
#include <mint/core/as_type.h>
#include <mint/core/index_t.h>
#include <mint/core/same_tuple.h>
#include <mint/core/type_traits.h>

namespace mint {

// clang_native_vector_type: clang native vector type
template <class S, index_t kNS>
  requires(
      sizeof(S) * kNS == sizeof(__attribute__((__ext_vector_type__(kNS))) S))
struct alignas(sizeof(S) * kNS) clang_native_vector_type {
  using scalar_t = S;
  typedef scalar_t vector_t __attribute__((__ext_vector_type__(kNS)));

  alignas(sizeof(S) * kNS) vector_t data_;

  constexpr clang_native_vector_type() = default;

  MINT_HOST_DEVICE constexpr clang_native_vector_type(const vector_t& init)
      : data_{init} {}

  // FIXME: ugly
  MINT_HOST_DEVICE constexpr clang_native_vector_type(initializer_list<S> init)
      : data_{bit_cast<vector_t>(same_tuple<S, kNS>{init})} {}

#if 0
  // TODO: this produce wrong result for now
  // not implemented for compiler native vector type
  template <index_t kI>
  MINT_HOST_DEVICE constexpr const S& operator[](index_constant<kI>) const {
    return data_[kI];
  }

  // TODO: this does't compile for now, compiler complain non-const operator []
  // not implemented for compiler native vector type
  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& operator[](index_constant<kI>) {
    return data_[kI];
  }
#else
  template <index_t kI>
  MINT_HOST_DEVICE constexpr const S& operator[](index_constant<kI>) const {
    return as_type<same_tuple<S, kNS>>(data_).template at<kI>();
  }

  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& operator[](index_constant<kI>) {
    return as_type<same_tuple<S, kNS>>(data_).template at<kI>();
  }
#endif

  template <index_t kI>
  MINT_HOST_DEVICE constexpr const S& at() const {
    return data_[kI];
  }

#if 0
  // TODO: this does't compile for now, compiler complain non-const operator []
  // not implemented for compiler native vector type
  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& at() {
    return data_[kI];
  }
#else
  template <index_t kI>
  MINT_HOST_DEVICE constexpr S& at() {
    return as_type<same_tuple<S, kNS>>(data_).template at<kI>();
  }

#endif

  MINT_HOST_DEVICE constexpr const auto& as_scalars() const {
    return data_;
  }

#if 0
  // TODO: this does't compile for now, compiler complain non-const operator []
  // not implemented for compiler native vector type
  MINT_HOST_DEVICE constexpr auto& as_scalars() {
    return data_;
  }
#else
  MINT_HOST_DEVICE constexpr auto as_scalars() -> same_tuple<S, kNS>& {
    return as_type<same_tuple<S, kNS>>(data_);
  }
#endif

  MINT_HOST_DEVICE void fill(S s) {
    static_for_n<kNS>()([&](auto i) { this->template at<i.value>() = s; });
  }

  template <index_t kSPerX>
    requires(kNS % kSPerX == 0)
  MINT_HOST_DEVICE constexpr auto as_vectors() const
      -> const same_tuple<clang_native_vector_type<S, kSPerX>, kNS / kSPerX>& {
    constexpr index_t kNX = kNS / kSPerX;
    return as_type<const same_tuple<clang_native_vector_type<S, kSPerX>, kNX>>(
        data_);
  }

  template <index_t kSPerX>
    requires(kNS % kSPerX == 0)
  MINT_HOST_DEVICE constexpr auto as_vectors()
      -> same_tuple<clang_native_vector_type<S, kSPerX>, kNS / kSPerX>& {
    constexpr index_t kNX = kNS / kSPerX;
    return as_type<same_tuple<clang_native_vector_type<S, kSPerX>, kNX>>(data_);
  }
};

// operator +
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN> operator+(
    const clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  return clang_native_vector_type<S, kN>{lhs.data_ + rhs.data_};
}

// operator -
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN> operator-(
    const clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  return clang_native_vector_type<S, kN>{lhs.data_ - rhs.data_};
}

// operator *
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN> operator*(
    const clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  return clang_native_vector_type<S, kN>{lhs.data_ * rhs.data_};
}

// operator /
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN> operator/(
    const clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  return clang_native_vector_type<S, kN>{lhs.data_ / rhs.data_};
}

// operator +=
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN>& operator+=(
    clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  lhs.data_ += rhs.data_;
  return lhs;
}

// operator -=
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN>& operator-=(
    clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  lhs.data_ -= rhs.data_;
  return lhs;
}

// operator *=
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN>& operator*=(
    clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  lhs.data_ *= rhs.data_;
  return lhs;
}

// operator /=
template <class S, index_t kN>
MINT_HOST_DEVICE constexpr clang_native_vector_type<S, kN>& operator/=(
    clang_native_vector_type<S, kN>& lhs,
    const clang_native_vector_type<S, kN>& rhs) {
  lhs.data_ /= rhs.data_;
  return lhs;
}

} // namespace mint
