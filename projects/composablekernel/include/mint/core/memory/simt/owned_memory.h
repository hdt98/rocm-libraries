#pragma once
#include <mint/core/as_type.h>
#include <mint/core/integral_constant.h>
#include <mint/core/memory/simt/address_space.h>
#include <mint/core/tuple.h>
#include <mint/core/type_traits.h>
#include <mint/core/vector_type.h>

namespace mint {

struct owned_memory_base {};

// owned VGPR memory
template <class T, index_t kSize>
struct owned_vgpr_memory : owned_memory_base {
  using value_type = remove_cvref_t<T>;

  MINT_HOST_DEVICE static consteval auto address_space() {
    return address_space::vgpr;
  }

  MINT_HOST_DEVICE static consteval index_t size() {
    return kSize;
  }

  // scalar access (const)
  template <index_t kI>
  MINT_HOST_DEVICE const value_type& at() const {
    return data_.template at<kI>();
  }

  // scalar access (non-cast)
  template <index_t kI>
  MINT_HOST_DEVICE value_type& at() {
    return data_.template at<kI>();
  }

  // scalar access (const)
  template <index_t kI>
  MINT_HOST_DEVICE const value_type& operator[](index_constant<kI>) const {
    return data_.template at<kI>();
  }

  // scalar access (non-cast)
  template <index_t kI>
  MINT_HOST_DEVICE value_type& operator[](index_constant<kI>) {
    return data_.template at<kI>();
  }

  // cast for vector access (const)
  template <index_t kVectorSize>
    requires(kSize % kVectorSize == 0)
  MINT_HOST_DEVICE auto as_vectors() const -> const
      same_tuple<vector_type<value_type, kVectorSize>, kSize / kVectorSize>& {
    return as_type<const same_tuple<
        vector_type<value_type, kVectorSize>,
        kSize / kVectorSize>>(data_);
  }

  // cast for vector access (non-const)
  template <index_t kVectorSize>
    requires(kSize % kVectorSize == 0)
  MINT_HOST_DEVICE auto as_vectors() -> same_tuple<
      vector_type<value_type, kVectorSize>,
      kSize / kVectorSize>& {
    return as_type<
        same_tuple<vector_type<value_type, kVectorSize>, kSize / kVectorSize>>(
        data_);
  }

  // fill
  MINT_HOST_DEVICE void fill(const value_type& v) {
    data_.fill(v);
  }

  MINT_HOST_DEVICE void print() const {
    printf("owned_vgpr_memory: {");
    printf("data_: ");
    data_.print();
    printf("}");
  }

 private:
  same_tuple<value_type, kSize> data_;
};

//
template <class T, index_t kSize>
MINT_HOST_DEVICE auto make_owned_vgpr_memory()
    -> owned_vgpr_memory<remove_cvref_t<T>, kSize> {
  return {};
}

} // namespace mint
