#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/distributed_tensor_descriptor.h>
#include <mint/tensor/distributed_tensor_descriptor_helper.h>
#include <mint/tensor/tensor_descriptor_helper.h>

namespace mint {
namespace tensor {

template <auto kDstrTensorDesc, auto kElementTensorDesc, class Memory>
  requires(
      kDstrTensorDesc.element_ndim() == kElementTensorDesc.top_ndim() &&
      kElementTensorDesc.bottom_ndim() == 1 &&
      kDstrTensorDesc.element_size() == Memory::size())
struct distributed_tensor {
 private:
  using dstr_tensor_desc_type = remove_cvref_t<decltype(kDstrTensorDesc)>;
  using element_tensor_desc_type = remove_cvref_t<decltype(kElementTensorDesc)>;

 public:
  using value_type = Memory::value_type;

  constexpr distributed_tensor() = default;

  MINT_HOST_DEVICE constexpr distributed_tensor(const Memory& mem)
      : mem_(mem) {}

  MINT_HOST_DEVICE constexpr auto top_lengths() const {
    return kDstrTensorDesc.top_lengths();
  }

  MINT_HOST_DEVICE static constexpr auto dstr_tensor_desc() {
    return kDstrTensorDesc;
  }

  MINT_HOST_DEVICE static constexpr auto element_tensor_desc() {
    return kElementTensorDesc;
  }

  MINT_HOST_DEVICE const auto& memory() const {
    return mem_;
  }

  MINT_HOST_DEVICE auto& memory() {
    return mem_;
  }

  MINT_HOST_DEVICE void fill(const value_type& v) {
    mem_.fill(v);
  }

  template <auto kElementTopIndex>
    requires(kElementTopIndex.size() == kElementTensorDesc.top_ndim())
  MINT_HOST_DEVICE const value_type& element() const {
    constexpr index_t offset =
        element_tensor_desc().calculate_bottom_index(kElementTopIndex)[0];
    return memory().template at<offset>();
  }

  template <auto kElementTopIndex>
  MINT_HOST_DEVICE value_type& element() {
    constexpr index_t offset =
        element_tensor_desc().calculate_bottom_index(kElementTopIndex)[0];
    return memory().template at<offset>();
  }

  template <auto kElementTopIndex, index_t kVectorSize>
    requires(kElementTopIndex.size() == kElementTensorDesc.top_ndim())
  MINT_HOST_DEVICE const vector_type<value_type, kVectorSize>& element_vector()
      const {
    constexpr index_t offset =
        element_tensor_desc().calculate_bottom_index(kElementTopIndex)[0];
    return memory()
        .template as_vectors<kVectorSize>()
        .template at<offset / kVectorSize>();
  }

  template <auto kElementTopIndex, index_t kVectorSize>
  MINT_HOST_DEVICE vector_type<value_type, kVectorSize>& element_vector() {
    constexpr index_t offset =
        element_tensor_desc().calculate_bottom_index(kElementTopIndex)[0];
    return memory()
        .template as_vectors<kVectorSize>()
        .template at<offset / kVectorSize>();
  }

  // ShardedIndex: tuple<nd_index<...>...>
  // ShardedIndex: tuple<index_sequence<...>...>
  // ShardedIndex: tuple<tuple<...>...>
  template <auto kShardedIndex>
  MINT_HOST_DEVICE const value_type& sharded_element() const {
    constexpr auto element_top_idx =
        dstr_tensor_desc().convert_sharded_element_index_to_element_index(
            kShardedIndex);
    return element<element_top_idx>();
  }

  // ShardedIndex: tuple<nd_index<...>...>
  // ShardedIndex: tuple<index_sequence<...>...>
  // ShardedIndex: tuple<tuple<...>...>
  template <auto kShardedIndex>
  MINT_HOST_DEVICE value_type& sharded_element() {
    constexpr auto element_top_idx =
        dstr_tensor_desc().convert_sharded_element_index_to_element_index(
            kShardedIndex);
    return element<element_top_idx>();
  }

  // ShardedIndex: tuple<index_sequence<...>...>
  // TODO: sanity check SharedIndex is indeed tuple<index_sequence<...>...>,
  // otherwise result is wrong
  template <class ShardedIndex>
  MINT_HOST_DEVICE const value_type& sharded_element(ShardedIndex) const {
    constexpr auto element_top_idx =
        dstr_tensor_desc().convert_sharded_element_index_to_element_index(
            ShardedIndex{});
    return element<element_top_idx>();
  }

  // ShardedIndex: tuple<index_sequence<...>...>
  // TODO: sanity check SharedIndex is indeed tuple<index_sequence<...>...>,
  // otherwise result is wrong
  template <class ShardedIndex>
  MINT_HOST_DEVICE value_type& sharded_element(ShardedIndex) {
    constexpr auto element_top_idx =
        dstr_tensor_desc().convert_sharded_element_index_to_element_index(
            ShardedIndex{});
    return element<element_top_idx>();
  }

  MINT_HOST_DEVICE void print() const {
    printf("distributed_tensor: {");

    printf("dstr_tensor_desc(): ");
    dstr_tensor_desc().print();
    printf(", ");

    printf("mem_: ");
    mem_.print();
    printf(", ");

    printf("}");
  }

 private:
  Memory mem_;
};

template <
    class DstrTensorDesc,
    DstrTensorDesc kDstrTensorDesc,
    class ElementTensorDesc,
    ElementTensorDesc kElementTensorDesc,
    class Memory>
  requires(kDstrTensorDesc.element_size() == Memory::size())
MINT_HOST_DEVICE constexpr auto make_distributed_tensor(
    integral_constant<DstrTensorDesc, kDstrTensorDesc>,
    integral_constant<ElementTensorDesc, kElementTensorDesc>,
    const Memory& old_memory) {
  return distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>(
      old_memory);
}

template <class DstrTensorDesc, DstrTensorDesc kDstrTensorDesc, class Memory>
  requires(kDstrTensorDesc.element_size() == Memory::size())
MINT_HOST_DEVICE constexpr auto make_distributed_tensor(
    integral_constant<DstrTensorDesc, kDstrTensorDesc> /*dstr*/,
    const Memory& old_memory) {
  constexpr auto element_layout = make_aliased_naive_packed_tensor_descriptor(
      make_index_sequence<kDstrTensorDesc.element_ndim()>{},
      index_constant<-1>{},
      kDstrTensorDesc.element_lengths());
  return distributed_tensor<kDstrTensorDesc, element_layout, Memory>(
      old_memory);
}

template <
    class T,
    class DstrTensorDesc,
    DstrTensorDesc kDstrTensorDesc,
    class ElementTensorDesc,
    ElementTensorDesc kElementTensorDesc>
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_vgpr(
    integral_constant<DstrTensorDesc, kDstrTensorDesc> dstr_tensor,
    integral_constant<ElementTensorDesc, kElementTensorDesc> element_tensor) {
  using Memory = owned_vgpr_memory<T, kElementTensorDesc.bottom_lengths()[0]>;
  return make_distributed_tensor(dstr_tensor, element_tensor, Memory{});
}

template <class T, class DstrTensorDesc, DstrTensorDesc kDstrTensorDesc>
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_vgpr(
    integral_constant<DstrTensorDesc, kDstrTensorDesc> dstr) {
  constexpr auto element_layout = make_aliased_naive_packed_tensor_descriptor(
      make_index_sequence<kDstrTensorDesc.element_ndim()>{},
      index_constant<-1>{},
      kDstrTensorDesc.element_lengths());
  return make_distributed_tensor_vgpr<T>(dstr, constant<element_layout>{});
}

namespace impl {

template <class Lengths>
MINT_HOST_DEVICE constexpr auto scan_dimensions(const Lengths& lengths) {
  constexpr index_t kN = remove_cvref_t<Lengths>::size();
  nd_index<kN> ret;
  if constexpr (kN > 0) {
    ret[kN - 1] = lengths[kN - 1];
    for (index_t i = kN - 1; i > 0; --i) {
      ret[i - 1] = ret[i] * lengths[i - 1];
    }
  }
  return ret;
}

template <index_t NDims, index_t Offset>
MINT_HOST_DEVICE constexpr auto make_sequential_dims() {
  nd_index<NDims> ret;
  for (index_t i = 0; i < NDims; i++) {
    ret[i] = i + Offset;
  }
  return ret;
}

template <class kDstrTensor, class TransformOp, class... Args>
MINT_HOST_DEVICE constexpr auto transform_distributed_tensor_z2(
    kDstrTensor& old_dstr_tensor,
    Args&&... args) {
  constexpr auto old_dstr_desc = kDstrTensor::dstr_tensor_desc();
  auto new_polymorpher = transform_z2_polymorpher(
      old_dstr_desc.polymorpher(),
      std::forward<TransformOp>(TransformOp{}),
      std::forward<Args>(args)...);

  constexpr auto new_dstr_desc = make_distributed_tensor_descriptor(
      new_polymorpher,
      constant<old_dstr_desc.partition_dims()>{},
      constant<old_dstr_desc.element_dims()>{});

  return make_distributed_tensor(
      constant<new_dstr_desc>{},
      constant<kDstrTensor::element_tensor_desc()>{},
      old_dstr_tensor.memory());
}

} // namespace impl

template <
    class T,
    index_t TopNDim,
    nd_index<TopNDim> TopLengths,
    index_t TopPartNDim,
    nd_index<TopPartNDim> TopPartDims,
    index_t TopElemNDim,
    nd_index<TopElemNDim> TopElemDims,
    bool k1P>
  requires(TopNDim == TopPartNDim + TopElemNDim)
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_vgpr_z2(
    integral_constant<nd_index<TopNDim>, TopLengths> /*top_lengths*/,
    integral_constant<nd_index<TopPartNDim>, TopPartDims> /*top_part_dims*/,
    integral_constant<nd_index<TopElemNDim>, TopElemDims> /*top_elem_dims*/,
    integral_constant<bool, k1P>) {
  // {P0, P1, ..., E0, E1, ...}
  constexpr auto ReorderTopDims = []() {
    nd_index<TopNDim> ret;
    for (index_t i = 0; i < TopPartNDim; i++) {
      ret[i] = TopPartDims[i];
    }
    for (index_t i = 0; i < TopElemNDim; i++) {
      ret[i + TopPartNDim] = TopElemDims[i];
    }
    return ret;
  }();

  constexpr auto bot_config = []() {
    // Merge partition dims into one dim
    if constexpr (k1P) {
      // {P, E0, E1, ...}
      constexpr auto bot_lengths = []() {
        nd_index<TopElemNDim + 1> ret;
        index_t partSize = 1;
        for (index_t i = 0; i < TopPartNDim; i++) {
          partSize *= TopLengths[TopPartDims[i]];
        }
        ret[0] = partSize;
        for (index_t i = 0; i < TopElemNDim; i++) {
          ret[i + 1] = TopLengths[TopElemDims[i]];
        }
        return ret;
      }();

      // {P}
      constexpr auto bot_part_dims = nd_index<1>{0};

      // {E0, E1, ...}
      constexpr auto bot_elem_dims =
          impl::make_sequential_dims<TopElemNDim, 1>();

      return mint::make_tuple(bot_lengths, bot_part_dims, bot_elem_dims);
    } else {
      // {P0, P1, ..., E0, E1, ...}
      constexpr auto bot_lengths = []() {
        nd_index<TopNDim> ret;
        for (index_t i = 0; i < TopPartNDim; i++) {
          ret[i] = TopLengths[TopPartDims[i]];
        }
        for (index_t i = 0; i < TopElemNDim; i++) {
          ret[i + TopPartNDim] = TopLengths[TopElemDims[i]];
        }
        return ret;
      }();

      // {P0, P1, ...}
      constexpr auto bot_part_dims =
          impl::make_sequential_dims<TopPartNDim, 0>();

      // {E0, E1, ...}
      constexpr auto bot_elem_dims =
          impl::make_sequential_dims<TopElemNDim, TopPartNDim>();

      return mint::make_tuple(bot_lengths, bot_part_dims, bot_elem_dims);
    }
  }();

  constexpr auto kBotLengths = bot_config[0_ic];
  constexpr auto kBotPartDims = bot_config[1_ic];
  constexpr auto kBotElemDims = bot_config[2_ic];

  constexpr auto m0 =
      poly::make_z2_pass_through_morpher(constant<TopLengths>{});
  constexpr auto m1 = reorder_bottom(m0, constant<ReorderTopDims>{});
  constexpr auto m2 = reshape_bottom(m1, constant<kBotLengths>{});
  constexpr auto poly = make_z2_polymorpher_default_alias(m2);
  constexpr auto dstr_desc = make_distributed_tensor_descriptor(
      poly, constant<kBotPartDims>{}, constant<kBotElemDims>{});
  constexpr auto element_layout =
      make_packed_tensor_descriptor_z2(constant<dstr_desc.element_lengths()>{});
  return make_distributed_tensor_vgpr<T>(
      constant<dstr_desc>{}, constant<element_layout>{});
}

// Overload with default k1P = true
template <
    class T,
    index_t TopNDim,
    nd_index<TopNDim> TopLengths,
    index_t TopPartNDim,
    nd_index<TopPartNDim> TopPartDims,
    index_t TopElemNDim,
    nd_index<TopElemNDim> TopElemDims>
  requires(TopNDim == TopPartNDim + TopElemNDim)
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_vgpr_z2(
    integral_constant<nd_index<TopNDim>, TopLengths> top_lengths,
    integral_constant<nd_index<TopPartNDim>, TopPartDims> top_part_dims,
    integral_constant<nd_index<TopElemNDim>, TopElemDims> top_elem_dims) {
  return make_distributed_tensor_vgpr_z2<T>(
      top_lengths, top_part_dims, top_elem_dims, constant<true>{});
}

template <
    class T,
    index_t TopNDim,
    nd_index<TopNDim> TopDims,
    class PartitionInfo>
MINT_HOST_DEVICE constexpr auto make_simple_distributed_tensor_vgpr_z2(
    integral_constant<nd_index<TopNDim>, TopDims>,
    const PartitionInfo& /*partition_info*/) {
  constexpr auto top_size = impl::scan_dimensions(TopDims)[0];

  constexpr index_t part_num = PartitionInfo::partition_num();

  constexpr auto m0 = poly::make_z2_pass_through_morpher(constant<TopDims>{});

  constexpr auto m1 = reshape_bottom(
      m0, constant<nd_index<2>{part_num, top_size / part_num}>{});
  constexpr auto poly = make_z2_polymorpher_default_alias(m1);
  constexpr auto dstr_desc = make_distributed_tensor_descriptor(
      poly, constant<nd_index<1>{0}>{}, constant<nd_index<1>{1}>{});
  constexpr auto element_layout =
      make_packed_tensor_descriptor_z2(constant<dstr_desc.element_lengths()>{});
  return make_distributed_tensor_vgpr<T>(
      constant<dstr_desc>{}, constant<element_layout>{});
}

// Type trait to check if a type is distributed_tensor
namespace impl {
template <typename T>
struct is_distributed_tensor_type : std::false_type {};

template <auto kDstrTensorDesc, auto kElementTensorDesc, class Memory>
struct is_distributed_tensor_type<
    distributed_tensor<kDstrTensorDesc, kElementTensorDesc, Memory>>
    : std::true_type {};
} // namespace impl

template <typename T>
concept is_distributed_tensor =
    impl::is_distributed_tensor_type<remove_cvref_t<T>>::value;

// Helper trait to check if a distributed tensor has exactly one z2 morpher
namespace impl {
template <typename DistributedTensor>
constexpr bool check_single_z2_morpher_dstr() {
  using dstr_tensor_type = remove_cvref_t<DistributedTensor>;
  constexpr auto dstr_desc = dstr_tensor_type::dstr_tensor_desc();
  using polymorpher_t = remove_cvref_t<decltype(dstr_desc.polymorpher())>;

  if constexpr (polymorpher_t::num_morpher() == 1) {
    using morpher_t = remove_cvref_t<
        decltype(std::declval<polymorpher_t>().morphers()[0_ic])>;
    return poly::is_z2_linear_morpher<morpher_t>::value;
  }
  return false;
}
} // namespace impl

template <typename T>
concept is_single_z2_morpher_dstr = impl::check_single_z2_morpher_dstr<T>();

template <class kDstrTensor, class... Args>
  requires(
      is_distributed_tensor<kDstrTensor> and
      is_single_z2_morpher_dstr<kDstrTensor>)
MINT_HOST_DEVICE constexpr auto reshape_logical(
    kDstrTensor& dstr_tensor,
    Args&&... args) {
  return impl::transform_distributed_tensor_z2<
      kDstrTensor,
      poly::reshape_top_z2,
      Args...>(dstr_tensor, std::forward<Args>(args)...);
}

template <class kDstrTensor, class... Args>
  requires(
      is_distributed_tensor<kDstrTensor> and
      is_single_z2_morpher_dstr<kDstrTensor>)
MINT_HOST_DEVICE constexpr auto reorder_logical(
    kDstrTensor& dstr_tensor,
    Args&&... args) {
  return impl::transform_distributed_tensor_z2<
      kDstrTensor,
      poly::reorder_top_z2,
      Args...>(dstr_tensor, std::forward<Args>(args)...);
}

template <class kDstrTensor, class... Args>
  requires(
      is_distributed_tensor<kDstrTensor> and
      is_single_z2_morpher_dstr<kDstrTensor>)
MINT_HOST_DEVICE constexpr auto swizzle_logical(
    kDstrTensor& dstr_tensor,
    Args&&... args) {
  return impl::transform_distributed_tensor_z2<
      kDstrTensor,
      poly::swizzle_top_z2,
      Args...>(dstr_tensor, std::forward<Args>(args)...);
}

template <
    class kDstrTensor,
    index_t kNewBottomElmNDims,
    nd_index<kNewBottomElmNDims> kNewBottomElmLengths>
MINT_HOST_DEVICE constexpr auto reshape_element(
    kDstrTensor& dstr_tensor,
    integral_constant<nd_index<kNewBottomElmNDims>, kNewBottomElmLengths>) {
  auto old_dstr_desc = dstr_tensor.dstr_tensor_desc();

  static_assert(
      impl::scan_dimensions(old_dstr_desc.element_lengths())[0] ==
      impl::scan_dimensions(kNewBottomElmLengths)[0]);

  constexpr auto partition_ndim = old_dstr_desc.partition_ndim();

  constexpr auto kNewBotLengths = [old_dstr_desc, partition_ndim]() {
    constexpr auto new_bot_ndims = partition_ndim + kNewBottomElmNDims;
    nd_index<new_bot_ndims> ret;
    for (index_t i = 0; i < partition_ndim; i++) {
      ret[i] = old_dstr_desc.partition_lengths()[i];
    }
    for (index_t i = 0; i < kNewBottomElmNDims; i++) {
      ret[i + partition_ndim] = kNewBottomElmLengths[i];
    }
    return ret;
  }();

  auto new_polymorpher = transform_z2_polymorpher(
      old_dstr_desc.polymorpher(),
      poly::reshape_bottom_z2{},
      constant<kNewBotLengths>{});

  constexpr auto kNewBotElemDims =
      impl::make_sequential_dims<kNewBottomElmNDims, partition_ndim>();

  constexpr auto new_dstr_desc = make_distributed_tensor_descriptor(
      new_polymorpher,
      constant<old_dstr_desc.partition_dims()>{},
      constant<kNewBotElemDims>{});

  constexpr auto old_element_layout = kDstrTensor::element_tensor_desc();

  constexpr auto new_elem_polymorpher = transform_z2_polymorpher(
      old_element_layout.polymorpher(),
      poly::reshape_top_z2{},
      constant<kNewBottomElmLengths>{});

  constexpr auto new_element_layout =
      make_tensor_descriptor(new_elem_polymorpher);

  return make_distributed_tensor(
      constant<new_dstr_desc>{},
      constant<new_element_layout>{},
      dstr_tensor.memory());
}

template <
    class kDstrTensor,
    index_t kNewBottomPartNDims,
    nd_index<kNewBottomPartNDims> kNewBottomPartLengths>
MINT_HOST_DEVICE constexpr auto reshape_partition(
    kDstrTensor& dstr_tensor,
    integral_constant<nd_index<kNewBottomPartNDims>, kNewBottomPartLengths>) {
  auto old_dstr_desc = dstr_tensor.dstr_tensor_desc();

  static_assert(
      impl::scan_dimensions(old_dstr_desc.partition_lengths())[0] ==
      impl::scan_dimensions(kNewBottomPartLengths)[0]);

  constexpr auto element_ndim = old_dstr_desc.element_ndim();

  constexpr auto kNewBotLengths = [old_dstr_desc, element_ndim]() {
    constexpr auto new_bot_ndims = kNewBottomPartNDims + element_ndim;
    nd_index<new_bot_ndims> ret;
    for (index_t i = 0; i < kNewBottomPartNDims; i++) {
      ret[i] = kNewBottomPartLengths[i];
    }
    for (index_t i = 0; i < element_ndim; i++) {
      ret[i + kNewBottomPartNDims] = old_dstr_desc.element_lengths()[i];
    }
    return ret;
  }();

  auto new_polymorpher = transform_z2_polymorpher(
      old_dstr_desc.polymorpher(),
      poly::reshape_bottom_z2{},
      constant<kNewBotLengths>{});

  constexpr auto kNewBotPartDims =
      impl::make_sequential_dims<kNewBottomPartNDims, 0>();
  constexpr auto kNewBotElemDims =
      impl::make_sequential_dims<element_ndim, kNewBottomPartNDims>();

  constexpr auto new_dstr_desc = make_distributed_tensor_descriptor(
      new_polymorpher,
      constant<kNewBotPartDims>{},
      constant<kNewBotElemDims>{});

  return make_distributed_tensor(
      constant<new_dstr_desc>{},
      constant<kDstrTensor::element_tensor_desc()>{},
      dstr_tensor.memory());
}

} // namespace tensor
} // namespace mint
