#pragma once
#include <mint/core.h>
#include <mint/poly.h>

namespace mint {
namespace tensor {

template <class TensorDesc>
struct coordinate {
 private:
  static constexpr index_t all_ndim_ = TensorDesc::all_ndim();
  static constexpr auto top_dims_ = TensorDesc::top_dims();
  static constexpr auto bottom_dims_ = TensorDesc::bottom_dims();

 public:
  constexpr coordinate() = default;

  MINT_HOST_DEVICE constexpr coordinate(
      const TensorDesc& tensor_desc,
      const nd_index<top_dims_.size()>& top_idx)
      : all_idx_{} {
    all_idx_.template set_subset<top_dims_.size(), top_dims_>(top_idx);
    tensor_desc.polymorpher().propagate_index_top_down(all_idx_);
  }

  MINT_HOST_DEVICE static consteval index_t top_ndim() {
    return top_dims_.size();
  }

  MINT_HOST_DEVICE static consteval index_t bottom_ndim() {
    return bottom_dims_.size();
  }

#if 1
  MINT_HOST_DEVICE constexpr auto get_top_index() const {
    return all_idx_.template get_subset<top_dims_.size(), top_dims_>();
  }

  MINT_HOST_DEVICE constexpr void set_top_index(
      const nd_index<top_ndim()>& top_idx) {
    all_idx_.template set_subset<top_dims_.size(), top_dims_>(top_idx);
  }

  MINT_HOST_DEVICE constexpr void set_top_index_and_propagate_down(
      const TensorDesc& tensor_desc,
      const nd_index<top_ndim()>& top_idx) {
    all_idx_.template set_subset<top_dims_.size(), top_dims_>(top_idx);
    tensor_desc.polymorpher().propagate_index_top_down(all_idx_);
  }

  MINT_HOST_DEVICE constexpr auto get_bottom_index() const {
    return all_idx_.template get_subset<bottom_dims_.size(), bottom_dims_>();
  }

  MINT_HOST_DEVICE constexpr void set_bottom_index(
      const nd_index<bottom_ndim()>& bot_idx) {
    all_idx_.template set_subset<bottom_dims_.size(), bottom_dims_>(bot_idx);
  }
#endif

  MINT_HOST_DEVICE constexpr const auto& all_index() const {
    return all_idx_;
  }

  MINT_HOST_DEVICE constexpr auto& all_index() {
    return all_idx_;
  }

  MINT_HOST_DEVICE constexpr auto top_index_reference() const {
    return all_idx_.template subset_reference<top_dims_.size(), top_dims_>();
  }

  MINT_HOST_DEVICE constexpr auto top_index_reference() {
    return all_idx_.template subset_reference<top_dims_.size(), top_dims_>();
  }

  MINT_HOST_DEVICE constexpr auto bottom_index_reference() const {
    return all_idx_
        .template subset_reference<bottom_dims_.size(), bottom_dims_>();
  }

  MINT_HOST_DEVICE constexpr auto bottom_index_reference() {
    return all_idx_
        .template subset_reference<bottom_dims_.size(), bottom_dims_>();
  }

  nd_index<all_ndim_> all_idx_;
};

template <class TensorDesc>
MINT_HOST_DEVICE constexpr auto make_coordinate(
    const TensorDesc& tensor_desc,
    const nd_index<TensorDesc::top_ndim()>& top_idx) {
  return coordinate<TensorDesc>{tensor_desc, top_idx};
}

} // namespace tensor
} // namespace mint
