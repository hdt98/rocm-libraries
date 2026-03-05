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

// cluster memory view impl
template <class T, class SizeType>
  requires(is_same_v<SizeType, index_t> || is_same_v<SizeType, long_index_t>)
struct cluster_memory_view_impl : memory_view_base {
  using value_type = remove_cvref_t<T>;
  using size_type = remove_cvref_t<SizeType>;

  MINT_DEVICE cluster_memory_view_impl(T* p_data, SizeType size)
      : p_data_{p_data}, size_{size} {}

  MINT_HOST_DEVICE static consteval auto address_space() {
    return address_space::cluster;
  }

  MINT_DEVICE constexpr index_t size() const {
    return size_;
  }

  value_type* const p_data_;
  const size_type size_;
};

} // namespace impl

// cluster memory view (non-const)
template <class T, class SizeType>
struct cluster_memory_view
    : impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType> {
  using base = impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType>;
  using value_type = base::value_type;
  using size_type = base::size_type;

  using impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType>::
      cluster_memory_view_impl;

  MINT_HOST_DEVICE static consteval bool is_const_memory_view() {
    return false;
  }

  // scalar access (non-const)
  MINT_DEVICE value_type& operator[](size_type i) const {
    return base::p_data_[i];
  }

  // cast for vector access (non-const)
  template <index_t kSPerX>
  MINT_DEVICE auto as_vectors() const
      -> vector_type<value_type, kSPerX>* const {
    return reinterpret_cast<vector_type<value_type, kSPerX>* const>(
        base::p_data_);
  }

  MINT_HOST_DEVICE const value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// cluster memory view (const)
template <class T, class SizeType>
struct const_cluster_memory_view
    : impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType> {
  using base = impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType>;
  using value_type = base::value_type;
  using size_type = base::size_type;

  using impl::cluster_memory_view_impl<remove_cvref_t<T>, SizeType>::
      cluster_memory_view_impl;

  MINT_DEVICE static consteval bool is_const_memory_view() {
    return true;
  }

  // scalar access (const)
  MINT_DEVICE const value_type& operator[](size_type i) const {
    return base::p_data_[i];
  }

  // cast for vector access (const)
  template <index_t kSPerX>
  MINT_DEVICE auto as_vectors() const
      -> const vector_type<value_type, kSPerX>* {
    return reinterpret_cast<const vector_type<value_type, kSPerX>*>(
        base::p_data_);
  }

  MINT_HOST_DEVICE const value_type* raw_pointer() const {
    return base::p_data_;
  }
};

// make cluster memory view (non-const)
template <class T, class SizeType>
MINT_DEVICE auto make_cluster_memory_view(T* p, SizeType size)
    -> cluster_memory_view<remove_cv_t<T>, remove_cvref_t<SizeType>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

// make cluster memory view (const)
template <class T, class SizeType>
MINT_HOST_DEVICE auto make_cluster_memory_view(const T* p, SizeType size)
    -> const_cluster_memory_view<remove_cv_t<T>, remove_cvref_t<SizeType>> {
  return {const_cast<remove_cv_t<T>*>(p), size};
}

} // namespace mint
