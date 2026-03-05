#pragma once
#include <mint/core/arithmetic_type.h>
#include <type_traits>

namespace mint {

template <class T>
struct is_mint_arithmetic : std::is_arithmetic<T> {};

template <class T>
struct is_mint_arithmetic<const T> : is_mint_arithmetic<T> {};

#if defined(MINT_BACKEND_CUDA)
// add CUDA supported arithmetic datatype
#elif defined(MINT_BACKEND_ROCM)
template <>
struct is_mint_arithmetic<fp16_t> : std::true_type {};

template <>
struct is_mint_arithmetic<floatx16_t> : std::true_type {};

template <>
struct is_mint_arithmetic<fp16x8_t> : std::true_type {};

template <>
struct is_mint_arithmetic<fp8_m4e3_t> : std::true_type {};
#endif

template <class T>
static constexpr bool is_mint_arithmetic_v = is_mint_arithmetic<T>::value;

} // namespace mint
