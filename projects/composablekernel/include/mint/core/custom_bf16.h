#pragma once

#include <mint/config.h>
#include <mint/core/bit_cast.h>

namespace mint {

struct alignas(2) custom_bf16 {
  uint16_t data_;

  MINT_HOST_DEVICE explicit custom_bf16(float x) {
#if defined(__CUDA_ARCH__)
    uint32_t bits = bit_cast<uint32_t>(x);

    // convert from float - round toward nearest
    // TODO: correctness not tested
    if ((bits & 0x7f800000) != 0x7f800000) {
      bool mantissa_bit = ((bits & (1 << 16)) != 0);
      bool round_bit = ((bits & (1 << 15)) != 0);
      bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        bits += uint32_t(1 << 16);
      }
    } else if (bits & ~0xff800000) {
      bits = 0x7fffffff;
    }

    data_ = uint16_t((bits >> 16) & 0xffff);
#else
    (void)x;
#endif
  }

  MINT_HOST_DEVICE explicit custom_bf16() : data_{0} {}

  // convert to float
  // TODO: correctness not tested
  MINT_HOST_DEVICE constexpr operator float() const {
    uint32_t bits = (uint32_t(data_) << 16);
    return bit_cast<float>(bits);
  }
};

} // namespace mint
