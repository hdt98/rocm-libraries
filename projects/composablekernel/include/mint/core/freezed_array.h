#pragma once
#include <mint/config.h>
#include <mint/core/array.h>

namespace mint {

template <class T, index_t kN, auto kIsFreezedElement>
class freezed_array {
  using base_type = array<T, kN>;

 public:
  using value_type = base_type::value_type;

  constexpr freezed_array() = default;

  MINT_HOST_DEVICE constexpr freezed_array(initializer_list<T> init)
      : data_{init}, dummy_{} {}

  MINT_HOST_DEVICE constexpr freezed_array(const base_type& arr)
      : data_{arr}, dummy_{} {}

  MINT_HOST_DEVICE static consteval auto is_freezed_element() {
    return kIsFreezedElement;
  }

  MINT_HOST_DEVICE static consteval index_t size() {
    return base_type::size();
  }

  MINT_HOST_DEVICE constexpr const T& operator[](index_t i) const {
    return data_[i];
  }

  MINT_HOST_DEVICE constexpr T& operator[](index_t i) {
    if (kIsFreezedElement[i]) {
      dummy_[i] = data_[i];
      return dummy_[i];
    } else {
      return data_[i];
    }
  }

  template <auto kSubset>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() <= size())
  MINT_HOST_DEVICE constexpr auto get_subset() const {
    freezed_array<
        T,
        kSubset.size(),
        kIsFreezedElement.template get_subset<kSubset>()>
        ret{};

    unroll_for_n<kSubset.size()>()([this, &ret](auto i) {
      // force set regardless of freeze
      ret.data_[i] = (*this)[kSubset[i]];
    });
    return ret;
  }

  template <auto kSubset, class Arr>
    requires(
        is_same_v<typename decltype(kSubset)::value_type, index_t> &&
        kSubset.size() <= size() && is_same_v<typename Arr::value_type, T> &&
        Arr::size() == kSubset.size())
  MINT_HOST_DEVICE constexpr void set_subset(const Arr& in) {
    unroll_for_n<kSubset.size()>()(
        [this, &in](auto i) { (*this)[kSubset[i]] = in[i]; });
  }

  MINT_HOST_DEVICE void print() const {
    printf("freezed_array {");
    printf("is_freezed_element() ");
    is_freezed_element().print();
    printf("data_ ");
    data_.print();
    printf("}");
  }

  base_type data_;
  base_type dummy_;
};

} // namespace mint
