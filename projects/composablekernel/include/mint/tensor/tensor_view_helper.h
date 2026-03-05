#pragma once
#include <mint/config.h>
#include <mint/core.h>
#include <mint/tensor/tensor_view.h>

namespace mint {
namespace tensor {

template <index_t kTopNDim, class T>
MINT_HOST_DEVICE constexpr auto make_global_packed_tensor_view(
    T* p,
    const nd_index<kTopNDim>& top_lengths) {
  const index_t kBotSize = [&]() {
    index_t cnt = 1;
    for (index_t i = 0; i < kTopNDim; i++) {
      cnt *= top_lengths[i];
    }
    return cnt;
  }();
  const auto morpher = poly::merge<nd_index<kTopNDim>>{top_lengths};
  const auto poly = make_simple_polymorpher_default_alias(
      morpher, nd_index<1>{kBotSize}, top_lengths);
  const auto desc = make_tensor_descriptor(poly);
  const auto mem_view = make_global_memory_view(p, desc.bottom_lengths()[0]);
  return make_tensor_view(desc, mem_view);
}

template <index_t kTopNDim, nd_index<kTopNDim> kTopDims, class T>
MINT_HOST_DEVICE constexpr auto make_global_packed_tensor_view_z2(
    T* p,
    integral_constant<nd_index<kTopNDim>, kTopDims> top_dims) {
  constexpr auto desc = make_packed_tensor_descriptor_z2(top_dims);
  const auto mem_view = make_global_memory_view(p, desc.bottom_lengths()[0]);
  return make_tensor_view(desc, mem_view);
}

template <index_t kTopNDim, nd_index<kTopNDim> kTopDims, class T>
MINT_HOST_DEVICE constexpr auto make_shared_packed_tensor_view_z2(
    T* p,
    integral_constant<nd_index<kTopNDim>, kTopDims> top_dims) {
  constexpr auto desc = make_packed_tensor_descriptor_z2(top_dims);
  const auto mem_view = make_shared_memory_view(p, desc.bottom_lengths()[0]);
  return make_tensor_view(desc, mem_view);
}

} // namespace tensor
} // namespace mint
