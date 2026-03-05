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

// shared memory view impl
template <class T>
struct shared_memory_view_impl : memory_view_base {
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  MINT_DEVICE shared_memory_view_impl(T* p_data, index_t size)
      : p_data_{p_data}, size_{size} {}

  MINT_HOST_DEVICE static consteval auto address_space() {
    return address_space::shared;
  }

  MINT_DEVICE constexpr index_t size() const {
    return size_;
  }

  value_type* const p_data_;
  const index_t size_;
};

} // namespace impl

// shared memory view (non-const)
template <class T>
struct shared_memory_view : impl::shared_memory_view_impl<remove_cvref_t<T>> {
  using base = impl::shared_memory_view_impl<remove_cvref_t<T>>;
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  using impl::shared_memory_view_impl<
      remove_cvref_t<T>>::shared_memory_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return false;
  }

  // scalar access (non-const)
  MINT_DEVICE value_type& operator[](index_t i) const {
    return base::p_data_[i];
  }

  // cast for vector access (non-const)
  template <index_t kSPerX>
  MINT_DEVICE auto as_vectors() const -> vector_type<value_type, kSPerX>* {
    return reinterpret_cast<MINT_SHARED_MEM vector_type<value_type, kSPerX>*>(
        base::p_data_);
  }

  MINT_HOST_DEVICE MINT_GLOBAL_MEM value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// shared memory view (const)
template <class T>
struct const_shared_memory_view
    : impl::shared_memory_view_impl<remove_cvref_t<T>> {
  using base = impl::shared_memory_view_impl<remove_cvref_t<T>>;
  using value_type = remove_cvref_t<T>;
  using size_type = index_t;

  using impl::shared_memory_view_impl<
      remove_cvref_t<T>>::shared_memory_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return true;
  }

  // scalar access (const)
  MINT_DEVICE const value_type& operator[](index_t i) const {
    return base::p_data_[i];
  }

  // cast for vector access (const)
  template <index_t kSPerX>
  MINT_DEVICE auto as_vectors() const
      -> const vector_type<value_type, kSPerX>* {
    return reinterpret_cast<
        MINT_SHARED_MEM const vector_type<value_type, kSPerX>*>(base::p_data_);
  }

  MINT_HOST_DEVICE MINT_GLOBAL_MEM const value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// make shared memory view (non-const)
template <class T>
MINT_DEVICE auto make_shared_memory_view(MINT_SHARED_MEM T* p, index_t size)
    -> shared_memory_view<remove_cv_t<T>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

// make shared memory view (const)
template <class T>
MINT_DEVICE auto make_shared_memory_view(
    MINT_SHARED_MEM const T* p,
    index_t size) -> const_shared_memory_view<remove_cv_t<T>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

} // namespace mint
