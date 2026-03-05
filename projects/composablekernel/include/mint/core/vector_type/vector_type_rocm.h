#pragma once
#include <mint/config.h>
#include <mint/core/vector_type/clang_native_vector_type.h>
#include <mint/core/vector_type/custom_vector_type.h>

namespace mint {

template <class T>
struct has_clang_native_vector_type : std::false_type {};

// fp32
template <>
struct has_clang_native_vector_type<float> : std::true_type {};

// i32
template <>
struct has_clang_native_vector_type<int32_t> : std::true_type {};

// fp16
template <>
struct has_clang_native_vector_type<_Float16> : std::true_type {};

namespace impl {

template <class S, index_t kNS, bool = has_clang_native_vector_type<S>::value>
struct vector_type_impl;

template <class S, index_t kNS>
struct vector_type_impl<S, kNS, true> {
  using type = clang_native_vector_type<S, kNS>;
};

template <class S, index_t kNS>
struct vector_type_impl<S, kNS, false> {
  using type = custom_vector_type<S, kNS>;
};

} // namespace impl

template <class S, index_t kNS>
using vector_type = typename impl::vector_type_impl<S, kNS>::type;

} // namespace mint
