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

// global memory view impl
template <class T, class SizeType>
  requires(is_same_v<SizeType, index_t> || is_same_v<SizeType, long_index_t>)
struct global_memory_view_impl : memory_view_base {
  using value_type = remove_cvref_t<T>;
  using size_type = remove_cvref_t<SizeType>;

  MINT_HOST_DEVICE global_memory_view_impl(
      MINT_GLOBAL_MEM T* p_data,
      SizeType size)
      : p_data_{p_data}, size_{size} {}

  MINT_HOST_DEVICE static consteval auto address_space() {
    return address_space::global;
  }

  MINT_HOST_DEVICE constexpr size_type size() const {
    return size_;
  }

  MINT_GLOBAL_MEM value_type* const p_data_;
  const size_type size_;
};

} // namespace impl

// global memory view (non-const)
template <class T, class SizeType>
struct global_memory_view
    : impl::global_memory_view_impl<remove_cvref_t<T>, SizeType> {
  using base = impl::global_memory_view_impl<remove_cvref_t<T>, SizeType>;
  using value_type = base::value_type;
  using size_type = base::size_type;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return false;
  }

  using impl::global_memory_view_impl<remove_cvref_t<T>, SizeType>::
      global_memory_view_impl;

  // scalar access (non-const)
  MINT_HOST_DEVICE value_type& operator[](size_type i) const {
    return base::p_data_[i];
  }

  // cast for vector access (non-const)
  template <index_t kSPerX>
  MINT_HOST_DEVICE auto as_vectors() const
      -> MINT_GLOBAL_MEM vector_type<value_type, kSPerX>* {
    return reinterpret_cast<MINT_GLOBAL_MEM vector_type<value_type, kSPerX>*>(
        base::p_data_);
  }

  MINT_HOST_DEVICE MINT_GLOBAL_MEM value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// global memory view (const)
template <class T, class SizeType>
struct const_global_memory_view
    : impl::global_memory_view_impl<remove_cvref_t<T>, SizeType> {
  using base = impl::global_memory_view_impl<remove_cvref_t<T>, SizeType>;
  using value_type = base::value_type;
  using size_type = base::size_type;

  using impl::global_memory_view_impl<remove_cvref_t<T>, SizeType>::
      global_memory_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return true;
  }

  // scalar access (const)
  MINT_HOST_DEVICE const value_type& operator[](size_type i) const {
    return base::p_data_[i];
  }

  // cast for vector access (const)
  template <index_t kSPerX>
  MINT_HOST_DEVICE auto as_vectors() const
      -> MINT_GLOBAL_MEM const vector_type<value_type, kSPerX>* {
    return reinterpret_cast<
        MINT_GLOBAL_MEM const vector_type<value_type, kSPerX>*>(base::p_data_);
  }

  MINT_HOST_DEVICE MINT_GLOBAL_MEM const value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// make global memory view (non-const)
template <class T, class SizeType>
MINT_HOST_DEVICE auto make_global_memory_view(
    MINT_GLOBAL_MEM T* p,
    SizeType size)
    -> global_memory_view<remove_cv_t<T>, remove_cvref_t<SizeType>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

// make global memory view (const)
template <class T, class SizeType>
MINT_HOST_DEVICE auto make_global_memory_view(
    MINT_GLOBAL_MEM const T* p,
    SizeType size)
    -> const_global_memory_view<remove_cv_t<T>, remove_cvref_t<SizeType>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

} // namespace mint
