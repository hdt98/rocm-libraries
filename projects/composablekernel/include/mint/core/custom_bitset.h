#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>

namespace mint {

// custom_bitset is stored in LSB-first order
template <index_t kN>
  requires(kN <= 32)
class custom_bitset {
  static constexpr index_t kBitPerWord = 32;
  static constexpr index_t kNumWord = (kN + kBitPerWord - 1) / kBitPerWord;

  static_assert(kNumWord == 1 || kNumWord == 0);

  uint32_t data_[kNumWord] = {};

  // helper class for operator[]
  class reference {
    friend class custom_bitset;
    uint32_t& word_;
    uint32_t mask_;

    MINT_HOST_DEVICE constexpr reference(uint32_t& word, uint32_t mask)
        : word_(word), mask_(mask) {}

   public:
    MINT_HOST_DEVICE constexpr reference& operator=(bool x) {
      if (x)
        word_ |= mask_;
      else
        word_ &= ~mask_;
      return *this;
    }

    MINT_HOST_DEVICE constexpr reference& operator=(const reference& rhs) {
      return *this = bool(rhs);
    }

    MINT_HOST_DEVICE constexpr operator bool() const {
      return (word_ & mask_) != 0;
    }
  };

  MINT_HOST_DEVICE constexpr void check_index([[maybe_unused]] index_t pos) const {
    assert(pos < kN);
  }

  // Mask for the last word to clear unused bits
  MINT_HOST_DEVICE static constexpr uint32_t last_word_mask() {
    return (kN % kBitPerWord) ? ((uint32_t(1) << (kN % kBitPerWord)) - 1)
                              : uint32_t(0xFFFFFFFF);
  }

 public:
  // Default constructor
  constexpr custom_bitset() = default;

  MINT_HOST_DEVICE constexpr custom_bitset(uint32_t value) {
    // Set the first word to value, mask if kN < 32
    if constexpr (kNumWord > 0) {
      if constexpr (kN < kBitPerWord)
        data_[0] = value & last_word_mask();
      else
        data_[0] = value;
    }
    // All other words are zero-initialized by default
  }

  // Constructor from const char*
  MINT_HOST_DEVICE constexpr explicit custom_bitset(const char* str) {
    reset();
    if (!str)
      return;
    index_t str_len = 0;
    while (str[str_len] != '\0')
      str_len++;

    index_t nbits = (str_len < kN) ? str_len : kN;
    for (index_t i = 0; i < nbits; i++) {
      char c = str[str_len - 1 - i];
      if (c != '0')
        set(i);
    }
  }

  constexpr custom_bitset& operator=(const custom_bitset&) = default;

  MINT_HOST_DEVICE static constexpr index_t size() {
    return kN;
  }

  // Set bit at position pos to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set(index_t pos, bool value) {
    check_index(pos);
    uint32_t mask = uint32_t(1) << (pos % kBitPerWord);
    uint32_t& w = data_[pos / kBitPerWord];
    if (value) {
      w |= mask; // set bit
    } else {
      w &= ~mask; // clear bit
    }
    return *this;
  }

  // Set bit at position pos to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set(index_t pos) {
    check_index(pos);
    data_[pos / kBitPerWord] |= (uint32_t(1) << (pos % kBitPerWord));
    return *this;
  }

  // Set all bits to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set() {
    for (index_t i = 0; i < kNumWord; ++i)
      data_[i] = uint32_t(0xFFFFFFFF);
    data_[kNumWord - 1] &= last_word_mask();
    return *this;
  }

  // Set bit at position pos to 0
  MINT_HOST_DEVICE constexpr custom_bitset& reset(index_t pos) {
    check_index(pos);
    data_[pos / kBitPerWord] &= ~(uint32_t(1) << (pos % kBitPerWord));
    return *this;
  }

  // Set all bits to 0
  MINT_HOST_DEVICE constexpr custom_bitset& reset() {
    for (index_t i = 0; i < kNumWord; ++i)
      data_[i] = 0;
    return *this;
  }

  // Flip bit at position pos
  MINT_HOST_DEVICE constexpr custom_bitset& flip(index_t pos) {
    check_index(pos);
    data_[pos / kBitPerWord] ^= (uint32_t(1) << (pos % kBitPerWord));
    return *this;
  }

  // Flip all bits
  MINT_HOST_DEVICE constexpr custom_bitset& flip() {
    for (index_t i = 0; i < kNumWord; ++i)
      data_[i] = ~data_[i];
    data_[kNumWord - 1] &= last_word_mask();
    return *this;
  }

  // Get value of bit at position pos
  MINT_HOST_DEVICE constexpr bool test(index_t pos) const {
    check_index(pos);
    return (data_[pos / kBitPerWord] >> (pos % kBitPerWord)) & 1;
  }

  // Get value of bit at position pos (operator[])
  MINT_HOST_DEVICE constexpr bool operator[](index_t pos) const {
    return test(pos);
  }

  MINT_HOST_DEVICE constexpr reference operator[](index_t pos) {
    check_index(pos);
    return reference(
        data_[pos / kBitPerWord], uint32_t(1) << (pos % kBitPerWord));
  }

  MINT_HOST_DEVICE constexpr bool operator==(const custom_bitset& rhs) const {
    for (index_t i = 0; i < kNumWord; ++i) {
      if (data_[i] != rhs.data_[i]) {
        return false;
      }
    }
    return true;
  }

  MINT_HOST_DEVICE constexpr bool operator!=(const custom_bitset& rhs) const {
    return !(*this == rhs);
  }

  // Bitwise AND assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator&=(
      const custom_bitset& rhs) {
    for (index_t i = 0; i < kNumWord; ++i) {
      data_[i] &= rhs.data_[i];
    }
    // Mask the last word to clear unused bits
    data_[kNumWord - 1] &= last_word_mask();
    return *this;
  }

  // Bitwise OR assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator|=(
      const custom_bitset& rhs) {
    for (index_t i = 0; i < kNumWord; ++i) {
      data_[i] |= rhs.data_[i];
    }
    data_[kNumWord - 1] &= last_word_mask();
    return *this;
  }

  // Bitwise ADD assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator+=(
      const custom_bitset& rhs) {
    data_[0] += rhs.data();
    return *this;
  }

  // Bitwise XOR assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator^=(
      const custom_bitset& rhs) {
    for (index_t i = 0; i < kNumWord; ++i) {
      data_[i] ^= rhs.data_[i];
    }
    data_[kNumWord - 1] &= last_word_mask();
    return *this;
  }

  // Bitwise AND
  MINT_HOST_DEVICE constexpr custom_bitset operator&(
      const custom_bitset& rhs) const {
    custom_bitset ret = *this;
    ret &= rhs;
    return ret;
  }

  // Bitwise OR
  MINT_HOST_DEVICE constexpr custom_bitset operator|(
      const custom_bitset& rhs) const {
    custom_bitset ret = *this;
    ret |= rhs;
    return ret;
  }

  // Bitwise XOR
  MINT_HOST_DEVICE constexpr custom_bitset operator^(
      const custom_bitset& rhs) const {
    custom_bitset ret = *this;
    ret ^= rhs;
    return ret;
  }

  // Bitwise NOT (unary operator)
  MINT_HOST_DEVICE constexpr custom_bitset operator~() const {
    custom_bitset ret;
    for (index_t i = 0; i < kNumWord; ++i) {
      ret.data_[i] = ~data_[i];
    }
    ret.data_[kNumWord - 1] &= last_word_mask();
    return ret;
  }

  MINT_HOST_DEVICE constexpr const uint32_t& data() const {
    return data_[0];
  }

  MINT_HOST_DEVICE constexpr uint32_t& data() {
    return data_[0];
  }

  MINT_HOST_DEVICE constexpr uint32_t to_uint32() const {
    return data_[0];
  }

  MINT_HOST_DEVICE void print() const {
    printf("custom_bitset<%d>: \"", kN);
    for (index_t i = kN; i > 0; i--) {
      printf("%d", (*this)[i - 1]);
    }
    printf("\"");
  }
};

// custom_bitset is stored in LSB-first order
template <>
class custom_bitset<0> {
  static constexpr index_t kN = 0;
  static constexpr index_t kBitPerWord = 32;
  static constexpr index_t kNumWord = 0;

  uint32_t data_[1] = {};

  // helper class for operator[]
  class reference {
    friend class custom_bitset;
    uint32_t& word_;
    uint32_t mask_;

    MINT_HOST_DEVICE constexpr reference(uint32_t& word, uint32_t mask)
        : word_(word), mask_(mask) {}

   public:
    MINT_HOST_DEVICE constexpr reference& operator=(bool x) {
      if (x)
        word_ |= mask_;
      else
        word_ &= ~mask_;
      return *this;
    }

    MINT_HOST_DEVICE constexpr reference& operator=(const reference& rhs) {
      return *this = bool(rhs);
    }

    MINT_HOST_DEVICE constexpr operator bool() const {
      return (word_ & mask_) != 0;
    }
  };

  MINT_HOST_DEVICE constexpr void check_index([[maybe_unused]] index_t pos) const {
    assert(pos == 0);
  }

  // Mask for the last word to clear unused bits
  MINT_HOST_DEVICE static constexpr uint32_t last_word_mask() {
    return (kN % kBitPerWord) ? ((uint32_t(1) << (kN % kBitPerWord)) - 1)
                              : uint32_t(0xFFFFFFFF);
  }

 public:
  // Default constructor
  constexpr custom_bitset() = default;

  MINT_HOST_DEVICE constexpr custom_bitset(uint32_t /*value*/) {
    // All other words are zero-initialized by default
  }

  // Constructor from const char*
  MINT_HOST_DEVICE constexpr explicit custom_bitset(const char* /*str*/) {}

  constexpr custom_bitset& operator=(const custom_bitset&) = default;

  MINT_HOST_DEVICE static constexpr index_t size() {
    return kN;
  }

  // Set bit at position pos to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set(index_t /*pos*/, bool /*value*/) {
    return *this;
  }

  // Set bit at position pos to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set(index_t /*pos*/) {
    return *this;
  }

  // Set all bits to 1
  MINT_HOST_DEVICE constexpr custom_bitset& set() {
    return *this;
  }

  // Set bit at position pos to 0
  MINT_HOST_DEVICE constexpr custom_bitset& reset(index_t /*pos*/) {
    return *this;
  }

  // Set all bits to 0
  MINT_HOST_DEVICE constexpr custom_bitset& reset() {
    return *this;
  }

  // Get value of bit at position pos
  MINT_HOST_DEVICE constexpr bool test(index_t pos) const {
    check_index(pos);
    return 0;
  }

  // Get value of bit at position pos (operator[])
  MINT_HOST_DEVICE constexpr bool operator[](index_t pos) const {
    return test(pos);
  }

  MINT_HOST_DEVICE constexpr reference operator[](index_t pos) {
    check_index(pos);
    return reference(data_[0], 0);
  }

  MINT_HOST_DEVICE constexpr bool operator==(const custom_bitset& /*rhs*/) const {
    return true;
  }

  MINT_HOST_DEVICE constexpr bool operator!=(const custom_bitset& rhs) const {
    return !(*this == rhs);
  }

  // Bitwise AND assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator&=(
      const custom_bitset& /*rhs*/) {
    return *this;
  }

  // Bitwise OR assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator|=(
      const custom_bitset& /*rhs*/) {
    return *this;
  }

  // Bitwise ADD assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator+=(
      const custom_bitset& /*rhs*/) {
    return *this;
  }

  // Bitwise XOR assignment
  MINT_HOST_DEVICE constexpr custom_bitset& operator^=(
      const custom_bitset& /*rhs*/) {
    return *this;
  }

  // Bitwise AND
  MINT_HOST_DEVICE constexpr custom_bitset operator&(
      const custom_bitset&) const {
    return custom_bitset{};
  }

  // Bitwise OR
  MINT_HOST_DEVICE constexpr custom_bitset operator|(
      const custom_bitset&) const {
    return custom_bitset{};
  }

  // Bitwise XOR
  MINT_HOST_DEVICE constexpr custom_bitset operator^(
      const custom_bitset&) const {
    return custom_bitset{};
  }

  // Bitwise NOT (unary operator)
  MINT_HOST_DEVICE constexpr custom_bitset operator~() const {
    return custom_bitset{};
  }

  MINT_HOST_DEVICE constexpr uint32_t to_uint32() const {
    return 0;
  }

  MINT_HOST_DEVICE void print() const {
    printf("custom_bitset<%d>: \"", kN);
    for (index_t i = kN; i > 0; i--) {
      printf("%d", (*this)[i - 1]);
    }
    printf("\"");
  }
};
} // namespace mint
