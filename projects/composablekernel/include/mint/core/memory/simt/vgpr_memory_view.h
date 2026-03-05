#pragma once
#include <mint/config.h>
#include <mint/core/index_t.h>
#include <mint/core/memory/memory_view_base.h>
#include <mint/core/memory/simt/address_space.h>
#include <mint/core/memory/simt/owned_memory.h>
#include <mint/core/type_traits.h>
#include <mint/core/vector_type.h>

namespace mint {

namespace impl {

// vgpr memory view
template <class T, index_t kSize>
  requires is_mint_arithmetic_v<T>
struct vgpr_memory_view_impl : memory_view_base {
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  MINT_HOST_DEVICE static consteval auto address_space() {
    return address_space::vgpr;
  }

  MINT_HOST_DEVICE constexpr index_t size() const {
    return size_;
  }

  MINT_HOST_DEVICE vgpr_memory_view_impl(
      owned_vgpr_memory<remove_reference_t<T>, kSize>& data)
      : data_{data} {}

  owned_vgpr_memory<value_type, kSize>& data_;
  static constexpr index_t size_ = kSize;
};

} // namespace impl

// vgpr memory view (non-const)
template <class T, index_t kSize>
  requires is_mint_arithmetic_v<T>
struct vgpr_memory_view : impl::vgpr_memory_view_impl<T, kSize> {
  using base = impl::vgpr_memory_view_impl<remove_cvref_t<T>, kSize>;
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return false;
  }

  // scalar access (non-const)
  template <index_t kI>
  MINT_HOST_DEVICE T& operator[](index_constant<kI> i) const {
    return base::data_[i];
  }

  // cast for vector access (non-const)
  template <index_t kVectorSize>
  MINT_HOST_DEVICE auto& as_vectors() const {
    return base::data_.template as_vectors<kVectorSize>();
  }
};

// vgpr memory view (const)
template <class T, index_t kSize>
  requires is_mint_arithmetic_v<T>
struct const_vgpr_memory_view : impl::vgpr_memory_view_impl<T, kSize> {
  using base = impl::vgpr_memory_view_impl<remove_cvref_t<T>, kSize>;
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return true;
  }

  // scalar access (const)
  template <index_t kI>
  MINT_HOST_DEVICE const T& operator[](index_constant<kI> i) const {
    return base::data_[i];
  }

  // cast for vector access (const)
  template <index_t kVectorSize>
  MINT_HOST_DEVICE const auto& as_vectors() const {
    return base::data_.template as_vectors<kVectorSize>();
  }
};

// make vgpr memory view (non-const)
template <class T, index_t kSize>
MINT_HOST_DEVICE auto make_vgpr_memory_view(owned_vgpr_memory<T, kSize>& data)
    -> vgpr_memory_view<remove_cvref_t<T>, kSize> {
  return {data};
}

// make vgpr memory view (const)
template <class T, index_t kSize>
MINT_HOST_DEVICE auto make_vgpr_memory_view(
    const owned_vgpr_memory<T, kSize>& data)
    -> const_vgpr_memory_view<remove_cvref_t<T>, kSize> {
  return {data};
}

} // namespace mint
