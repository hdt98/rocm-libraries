#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/print.h>
#include <mint/core/sequence.h>
#include <mint/core/type_traits.h>

namespace mint {

template <class T, T... kIs>
  requires(is_integral_v<T>)
using integer_sequence = sequence<T, kIs...>;

#if defined(__CUDACC__)
namespace impl {

template <class Seq>
struct convert_to_sequence;

template <class T, T... kIs>
struct convert_to_sequence<std::integer_sequence<T, kIs...>> {
  using type = sequence<T, kIs...>;
};

} // namespace impl

template <class T, T kN>
using make_integer_sequence =
    typename impl::convert_to_sequence<std::make_integer_sequence<T, kN>>::type;
#else
// fast make_integer_sequence using LLVM builtin __make_integer_seq
// https://reviews.llvm.org/D13786
template <class T, T kN>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, kN>;
#endif

namespace impl {

template <class T, T kBegin, T kDiff, class Seq>
struct make_arithmetic_integer_sequence_impl;

template <class T, T kBegin, T kDiff, T... kIs>
struct make_arithmetic_integer_sequence_impl<
    T,
    kBegin,
    kDiff,
    integer_sequence<T, kIs...>> {
  using type = integer_sequence<T, kBegin + kIs * kDiff...>;
};

} // namespace impl

template <class T, T kN, T kBegin, T kDiff>
  requires(kN >= 0)
using make_arithmetic_integer_sequence =
    impl::make_arithmetic_integer_sequence_impl<
        T,
        kBegin,
        kDiff,
        make_integer_sequence<T, kN>>::type;

// index_sequence
template <index_t... kIs>
using index_sequence = integer_sequence<index_t, kIs...>;

template <index_t kN>
using make_index_sequence = make_integer_sequence<index_t, kN>;

template <index_t kN, index_t kBegin, index_t kDiff>
  requires(kN >= 0)
using make_arithmetic_index_sequence =
    make_arithmetic_integer_sequence<index_t, kN, kBegin, kDiff>;

// char_sequence
template <char... kChars>
using char_sequence = sequence<char, kChars...>;

template <char... kChars>
MINT_HOST_DEVICE consteval auto operator""_cs() {
  return char_sequence<kChars...>{};
}

} // namespace mint
