#pragma once
#include <mint/config.h>
#include <mint/core/array.h>
#include <mint/core/assert.h>
#include <mint/core/index_t.h>
#include <mint/core/tuple.h>

namespace mint {

template <class Key, class T, index_t kMaxSize>
struct static_map {
  using key_type = Key;
  using mapped_type = T;
  using value_type = tuple<Key, T>;

 private:
  using impl_type = array<value_type, kMaxSize>;

 public:
  impl_type impl_;
  index_t size_;

  struct iterator {
    impl_type& impl_;
    index_t pos_;

    MINT_HOST_DEVICE constexpr iterator(impl_type& impl, index_t pos)
        : impl_{impl}, pos_{pos} {}

    MINT_HOST_DEVICE constexpr iterator& operator++() {
      pos_++;
      return *this;
    }

    MINT_HOST_DEVICE constexpr bool operator==(const iterator& other) const {
      return other.pos_ == pos_;
    }

    MINT_HOST_DEVICE constexpr value_type& operator*() {
      return impl_.at(pos_);
    }
  };

  struct const_iterator {
    const impl_type& impl_;
    index_t pos_;

    MINT_HOST_DEVICE constexpr const_iterator(
        const impl_type& impl,
        index_t pos)
        : impl_{impl}, pos_{pos} {}

    MINT_HOST_DEVICE constexpr const_iterator& operator++() {
      pos_++;

      return *this;
    }

    MINT_HOST_DEVICE constexpr bool operator==(
        const const_iterator& other) const {
      return other.pos_ == pos_;
    }

    MINT_HOST_DEVICE constexpr const value_type& operator*() const {
      return impl_.at(pos_);
    }
  };

  MINT_HOST_DEVICE constexpr static_map() : impl_{}, size_{0} {}

  MINT_HOST_DEVICE static consteval index_t max_size() {
    return kMaxSize;
  }

  MINT_HOST_DEVICE constexpr index_t size() const {
    return size_;
  }

  MINT_HOST_DEVICE constexpr void clear() {
    size_ = 0;
  }

  MINT_HOST_DEVICE constexpr index_t find_position(const Key& k) const {
    for (index_t i = 0; i < size(); i++) {
      if (impl_[i].template at<0>() == k) {
        return i;
      }
    }

    return size_;
  }

  MINT_HOST_DEVICE constexpr const_iterator find(const Key& k) const {
    return const_iterator{impl_, find_position(k)};
  }

  MINT_HOST_DEVICE constexpr iterator find(const Key& k) {
    return iterator{impl_, find_position(k)};
  }

  MINT_HOST_DEVICE constexpr bool contains(const Key& k) const {
    return find(k) != end();
  }

  MINT_HOST_DEVICE constexpr const T& operator[](const Key& k) const {
    const auto it = find(k);

    mint_assert(it.pos_ < size());
    return impl_[it.pos_].template at<1>();
  }

  MINT_HOST_DEVICE constexpr T& operator[](const Key& k) {
    const auto it = find(k);

    // if entry not found
    if (it.pos_ == size()) {
      impl_[it.pos_].template at<0>() = k;
      size_++;
    }

    mint_assert(size_ <= kMaxSize);
    return impl_[it.pos_].template at<1>();
  }

  MINT_HOST_DEVICE tuple<iterator, bool> insert(const value_type& value) {
    const auto it = find(value.template at<0>());

    // if entry not found
    if (it.pos_ == size()) {
      impl_[it.pos_] = value;
      size_++;
      return {it, true};
    } else {
      return {it, false};
    }
  }

  // insert {key, data} into *this, if key not exist in *this
  template <index_t kSourceMaxSize>
  MINT_HOST_DEVICE constexpr void merge(
      const static_map<Key, T, kSourceMaxSize>& source) {
    for (const auto [key, data] : source)
      if (find(key) == end())
        (*this)[key] = data;
  }

  // WARNING: Don't use! For C++ range-based for loop only!
  MINT_HOST_DEVICE constexpr const_iterator begin() const {
    return const_iterator{impl_, 0};
  }

  // WARNING: Don't use! For C++ range-based for loop only!
  MINT_HOST_DEVICE constexpr const_iterator end() const {
    return const_iterator{impl_, size_};
  }

  // WARNING: Don't use! For C++ range-based for loop only!
  MINT_HOST_DEVICE constexpr iterator begin() {
    return iterator{impl_, 0};
  }

  // WARNING: Don't use! For C++ range-based for loop only!
  MINT_HOST_DEVICE constexpr iterator end() {
    return iterator{impl_, size_};
  }

  MINT_HOST_DEVICE void print() const {
    printf("static_map{size_: %d, ", size_);
    printf("impl_: {");
    for (const auto& [k, d] : *this) {
      printf("{Key: ");
      print_item(k);
      printf(", Mapped: ");
      print_item(d);
      printf("}, ");
    }
    //
    printf("}}");
  }
};

} // namespace mint
