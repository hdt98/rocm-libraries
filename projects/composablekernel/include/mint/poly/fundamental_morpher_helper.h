#pragma once
#include <mint/core.h>
#include <mint/poly/fundamental_morpher.h>
#include <mint/poly/morpher.h>

namespace mint {
namespace poly {

namespace impl {

// is_split_morpher_impl
template <class T>
struct is_split_morpher_impl : std::false_type {};

template <class BottomLengths>
  requires(BottomLengths::size() > 0)
struct is_split_morpher_impl<split<BottomLengths>> : std::true_type {};

// is_merge_morpher_impl
template <class T>
struct is_merge_morpher_impl : std::false_type {};

template <class TopLengths>
  requires(TopLengths::size() > 0)
struct is_merge_morpher_impl<merge<TopLengths>> : std::true_type {};

// is_project_morpher_impl
template <class T>
struct is_project_morpher_impl : std::false_type {};

template <class Coefficients>
  requires(Coefficients::size() > 0)
struct is_project_morpher_impl<project<Coefficients>> : std::true_type {};

// is_insert_length_one_morpher_impl
template <class T>
struct is_insert_length_one_morpher_impl : std::false_type {};

template <>
struct is_insert_length_one_morpher_impl<insert_length_one> : std::true_type {};

} // namespace impl

// is_split_morpher_v
template <class T>
static constexpr bool is_split_morpher_v =
    impl::is_split_morpher_impl<remove_cvref_t<T>>::value;

// is_merge_morpher_v
template <class T>
static constexpr bool is_merge_morpher_v =
    impl::is_merge_morpher_impl<remove_cvref_t<T>>::value;

// is_project_morpher_v
template <class T>
static constexpr bool is_project_morpher_v =
    impl::is_project_morpher_impl<remove_cvref_t<T>>::value;

// is_insert_length_one_morpher_v
template <class T>
static constexpr bool is_insert_length_one_morpher_v =
    impl::is_insert_length_one_morpher_impl<remove_cvref_t<T>>::value;

} // namespace poly
} // namespace mint
