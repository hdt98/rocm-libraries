#pragma once
#include <cmath>

namespace mint_test {

template <typename T>
bool is_equal_fp32(T a, T b, float atol) {
  float fp32_a = static_cast<float>(a);
  float fp32_b = static_cast<float>(b);
  if ((std::isnan(fp32_a) && std::isnan(fp32_b)) ||
      (std::isinf(fp32_a) && std::isinf(fp32_b))) {
    return true;
  } else {
    return std::abs(fp32_a - fp32_b) <= atol;
  }
}

template <typename T>
bool is_equal(T a, T b, float atol) {
  return is_equal_fp32<T>(a, b, atol);
}

} // namespace mint_test
