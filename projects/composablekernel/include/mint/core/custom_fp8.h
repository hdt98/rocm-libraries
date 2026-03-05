#pragma once
#include <cmath>

namespace mint {

struct alignas(1) fp8_m4e3_t {
  uint8_t data_;

  // convert to float
  // TODO: correctness not tested
  MINT_HOST_DEVICE constexpr operator float() const {
    int32_t sign = (data_ >> 7) & 0x01;
    int32_t exponent = (data_ >> 3) & 0x0F;
    float mantissa = (data_ & 0x07) / 8.0f;
    float num = 1.0f + mantissa;
    num *= pow(2.0f, exponent - 7);
    if (sign) {
      num = -num;
    }
    return num;
  }
};

} // namespace mint
