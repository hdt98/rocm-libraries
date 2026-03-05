#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/print.h>

namespace mint {

template <index_t kMaxSize>
struct static_string {
  constexpr static_string() = default;

  MINT_HOST_DEVICE constexpr static_string(const char* in) {
    index_t i = 0;
    for (; i < kMaxSize; i++) {
      if (in[i] != '\0') {
        data_[i] = in[i];
      } else {
        break;
      }
    }
    for (; i < kMaxSize; i++)
      data_[i] = '\0';
  }

  constexpr bool operator==(const static_string&) const = default;

  MINT_HOST_DEVICE static consteval index_t max_size() {
    return kMaxSize;
  }

  MINT_HOST_DEVICE constexpr index_t size() const {
    index_t cnt = 0;
    while (data_[cnt] != '\0')
      cnt++;
    return cnt;
  }

  MINT_HOST_DEVICE constexpr const char* data() const {
    return data_;
  }

  MINT_HOST_DEVICE constexpr char* data() {
    return data_;
  }

  MINT_HOST_DEVICE constexpr auto append(const char* str) {
    index_t cnt = size();
    while (*str != '\0') {
      data_[cnt] = *str;
      cnt++;
      str++;
    }
    data_[cnt] = '\0';

    return *this;
  }

  MINT_HOST_DEVICE void print() const {
#if 0
    printf("static_string<%d>: \"", kMaxSize);
#else
    printf("\"");
#endif
    for (index_t i = 0; i < size(); i++) {
      if (data_[i] != '\0')
        printf("%c", data_[i]);
      else
        break;
    }
    printf("\"");
  }

  char data_[kMaxSize];
};

// TODO: put it somewhere else
MINT_HOST_DEVICE constexpr void integer_to_string(int num, char* str) {
  int i = 0;
  int isNegative = 0;
  // Handle 0 explicitly, otherwise an empty string is printed for 0
  if (num == 0) {
    str[i++] = '0';
    str[i] = '\0';
    return;
  }
  // Check if the number is negative
  if (num < 0) {
    isNegative = 1;
    num = -num;
  }
  // Process each digit of the number
  while (num != 0) {
    int digit = num % 10;
    str[i++] = digit + '0'; // Convert digit to character
    num /= 10;
  }
  // If the number was negative, add the negative sign
  if (isNegative) {
    str[i++] = '-';
  }
  // Null-terminate the string
  str[i] = '\0';
  // Reverse the string
  int start = 0;
  int end = i - 1;
  while (start < end) {
    char temp = str[start];
    str[start] = str[end];
    str[end] = temp;
    start++;
    end--;
  }
}

} // namespace mint
