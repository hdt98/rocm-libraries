#pragma once
#include <mint/config.h>
#include <mint/core/arithmetic_type.h>
#include <limits>

namespace mint {

// intentionally dummy implementation as safe guard: to cause compiler complaint
//   for unsupported custom datatype
template <class T>
struct numeric_limits;

template <>
struct numeric_limits<float> : ::std::numeric_limits<float> {};

template <>
struct numeric_limits<fp16_t> : ::std::numeric_limits<fp16_t> {};

template <>
struct numeric_limits<int32_t> : ::std::numeric_limits<int32_t> {};

} // namespace mint
