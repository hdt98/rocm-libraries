#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/tensor_descriptor.h>

namespace mint {
namespace tensor {

template <
    auto kPolymorpher,
    auto kTopDimAliases, // array<DimAlias, xxx>
    auto kPartitionDimAliases, // array<DimAlias, xxxx>
    auto kElementDimAliases // array<DimAlias, xxxx>
    >
struct distributed_tensor_descriptor {
  // TODO: sanity check consistency of kPolymorpher with kTopDimAliases,
  //   kPartitionDimAliases, kElementDimAliases

  using polymorpher_type = remove_cvref_t<decltype(kPolymorpher)>;
  using morpher_alias_type = typename polymorpher_type::morpher_alias_type;
  using dim_alias_type = typename polymorpher_type::dim_alias_type;

  consteval distributed_tensor_descriptor() = default;

  MINT_HOST_DEVICE static consteval bool can_bottom_up() {
    return polymorpher_type::can_bottom_up();
  }

  MINT_HOST_DEVICE static consteval bool can_top_down() {
    return polymorpher_type::can_top_down();
  }

  MINT_HOST_DEVICE static consteval index_t top_ndim() {
    return polymorpher_type::top_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t bottom_ndim() {
    return polymorpher_type::bottom_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t all_ndim() {
    return polymorpher_type::all_ndim();
  }

  MINT_HOST_DEVICE static consteval index_t partition_ndim() {
    return kPartitionDimAliases.size();
  }

  MINT_HOST_DEVICE static consteval index_t element_ndim() {
    return kElementDimAliases.size();
  }

  MINT_HOST_DEVICE static consteval auto top_dim_aliases() {
    return kTopDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto partition_dim_aliases() {
    return kPartitionDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto element_dim_aliases() {
    return kElementDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto alias_to_partition_dim() {
    return array_to_static_map(kPartitionDimAliases);
  }

  MINT_HOST_DEVICE static consteval auto alias_to_element_dim() {
    return array_to_static_map(kElementDimAliases);
  }

  MINT_HOST_DEVICE static consteval auto bottom_dim_aliases() {
    array<dim_alias_type, bottom_ndim()> ret;
    for (index_t i = 0; i < partition_ndim(); i++)
      ret[i] = partition_dim_aliases()[i];
    for (index_t i = 0; i < element_ndim(); i++)
      ret[partition_ndim() + i] = element_dim_aliases()[i];
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto top_dims() {
    nd_index<top_ndim()> ret;
    for (index_t i = 0; i < top_ndim(); i++)
      ret[i] = polymorpher_type::alias_to_dim()[kTopDimAliases[i]];
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto top_alias_dim(auto alias) {
    return nd_index<1>{polymorpher_type::alias_to_dim()[alias]};
  }

  MINT_HOST_DEVICE static consteval auto partition_dims() {
    nd_index<partition_ndim()> ret;
    for (index_t i = 0; i < partition_ndim(); i++)
      ret[i] = polymorpher_type::alias_to_dim()[kPartitionDimAliases[i]];
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto element_dims() {
    nd_index<element_ndim()> ret;
    for (index_t i = 0; i < element_ndim(); i++)
      ret[i] = polymorpher_type::alias_to_dim()[kElementDimAliases[i]];
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto bottom_dims() {
    nd_index<bottom_ndim()> ret;
    for (index_t i = 0; i < partition_ndim(); i++)
      ret[i] = partition_dims()[i];
    for (index_t i = 0; i < element_ndim(); i++)
      ret[partition_ndim() + i] = element_dims()[i];
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto is_element_dim() {
    array<bool, all_ndim()> ret;
    ret.fill(false);
    for (auto dim : element_dims())
      ret[dim] = true;
    return ret;
  }

  // FIXME: hacky, and also assume top_dims are only tile_dims
  MINT_HOST_DEVICE static consteval auto tile_morphers() {
    nd_index<top_ndim()> ret;
    for (index_t i = 0; i < ret.size(); i++) {
      index_t dim = top_dims()[i];
      static_for_n<polymorpher_type::num_morpher()>()([&](auto imorpher) {
        constexpr auto morpher = kPolymorpher.morphers()[imorpher];
        if (polymorpher_type::morpher_local_to_unique_dim(
                imorpher, morpher.top_dims()[0]) == dim)
          ret[i] = imorpher;
      });
    }
    return ret;
  }

  // FIXME: hacky
  MINT_HOST_DEVICE static consteval auto partition_morphers() {
    nd_index<partition_ndim()> ret;
    for (index_t i = 0; i < ret.size(); i++) {
      index_t dim = partition_dims()[i];
      static_for_n<polymorpher_type::num_morpher()>()([&](auto imorpher) {
        constexpr auto morpher = kPolymorpher.morphers()[imorpher];
        if (polymorpher_type::morpher_local_to_unique_dim(
                imorpher, morpher.bottom_dims()[0]) == dim)
          ret[i] = imorpher;
      });
    }
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto top_lengths() {
    return kPolymorpher.all_lengths()
        .template get_subset<top_ndim(), top_dims()>();
  }

  MINT_HOST_DEVICE static consteval auto bottom_lengths() {
    return kPolymorpher.all_lengths()
        .template get_subset<bottom_ndim(), bottom_dims()>();
  }

  MINT_HOST_DEVICE static consteval auto partition_lengths() {
    return kPolymorpher.all_lengths()
        .template get_subset<partition_ndim(), partition_dims()>();
  }

  MINT_HOST_DEVICE static consteval auto element_lengths() {
    return kPolymorpher.all_lengths()
        .template get_subset<element_ndim(), element_dims()>();
  }

  template <alias_t alias>
  MINT_HOST_DEVICE static consteval auto alias_length() {
    return kPolymorpher.all_lengths()
        .template get_subset<
            top_alias_dim(alias).size(),
            top_alias_dim(alias)>()[0];
  }

  MINT_HOST_DEVICE static consteval index_t partition_size() {
    index_t ret = 1;
    for (const index_t v : partition_lengths())
      ret *= v;
    return ret;
  }

  MINT_HOST_DEVICE static consteval index_t element_size() {
    index_t ret = 1;
    for (const index_t v : element_lengths())
      ret *= v;
    return ret;
  }

  MINT_HOST_DEVICE static consteval polymorpher_type polymorpher() {
    return kPolymorpher;
  }

  // hacky to use tensor_desc
  MINT_HOST_DEVICE static consteval auto tensor_desc() {
    return tensor_descriptor<
        polymorpher_type,
        top_dim_aliases(),
        bottom_dim_aliases()>{kPolymorpher};
  }

  using tensor_desc_type = remove_cvref_t<decltype(tensor_desc())>;

  // FIXME: no guarantee can always bottom-to-top
  MINT_HOST_DEVICE static consteval auto top_over_element_derivative() {
    nd_array<index_t, top_ndim(), element_ndim()> ret;
    ret.fill(0);

    constexpr auto a_bot_idx = nd_index<bottom_ndim()>{}.fill(0);
    const auto a_top_idx = tensor_desc().calculate_top_index(a_bot_idx);

    for (index_t i = 0; i < element_ndim(); i++) {
      auto b_bot_idx = nd_index<bottom_ndim()>{}.fill(0);
      b_bot_idx[partition_ndim() + i] = 1;

      const auto b_top_idx = tensor_desc().calculate_top_index(b_bot_idx);

      for (index_t j = 0; j < top_ndim(); j++)
        ret[j][i] = b_top_idx[j] - a_top_idx[j];
    }

    return ret;
  }

  MINT_HOST_DEVICE static consteval auto top_to_element_ndim() {
    nd_index<top_ndim()> ret{};
    constexpr auto derivative = top_over_element_derivative();
    for (index_t i = 0; i < top_ndim(); i++)
      for (index_t j = 0; j < element_ndim(); j++)
        if (derivative[i][j] != 0)
          ret[i]++;
    return ret;
  }

  template <index_t... kNs>
  using sharded_index_type_impl = tuple<nd_index<kNs>...>;

  using sharded_index_type =
      pack_param<sharded_index_type_impl, top_to_element_ndim()>::type;

  // [idim_top] to [idim_elements...]
  MINT_HOST_DEVICE static consteval sharded_index_type top_to_element_dims() {
    sharded_index_type ret;
    constexpr auto derivative = top_over_element_derivative();

    static_for_n<top_ndim()>()([&ret, &derivative](auto i) {
      index_t cnt = 0;
      for (index_t j = 0; j < element_ndim(); j++) {
        if (derivative[i][j] != 0) {
          ret[i][cnt] = j;
          cnt++;
        }
      }
    });

    return ret;
  }

  // TODO: sanity-check: morpher linearity
  MINT_HOST_DEVICE static constexpr auto top_index_delta(
      const nd_index<element_ndim()>& element_idx_delta) {
    nd_index<top_ndim()> ret;
    ret.fill(0);
    constexpr auto derivative = top_over_element_derivative();

    for (index_t i = 0; i < top_ndim(); i++)
      for (index_t j = 0; j < element_ndim(); j++)
        ret[i] += element_idx_delta[j] * derivative[i][j];

    return ret;
  }

  MINT_HOST_DEVICE static consteval sharded_index_type sharded_lengths() {
    sharded_index_type ret;
    constexpr auto dims = top_to_element_dims();
    static_for_n<top_ndim()>()([&ret, &dims](auto i) {
      ret[i] = element_lengths().template get_subset<dims[i].size(), dims[i]>();
    });

    return ret;
  }

  // ShardedIndex: tuple<nd_index<>...>
  template <class ShardedIndex>
    requires(ShardedIndex::size() == top_ndim())
  MINT_HOST_DEVICE static constexpr auto
  convert_sharded_element_index_to_element_index(
      const ShardedIndex& sharded_index) {
    nd_index<element_ndim()> ret;
    constexpr auto ndim = top_to_element_ndim();
    constexpr auto dims = top_to_element_dims();

    static_for_n<top_ndim()>()([&ret, &sharded_index, &ndim, &dims](auto i) {
      static_assert(ShardedIndex{}[i].size() == ndim[i], "wrong");
      for (index_t j = 0; j < ndim[i]; j++)
        ret[dims[i][j]] = sharded_index[i][j];
    });

    return ret;
  }

  MINT_HOST_DEVICE static auto convert_partition_and_element_index_to_top_index(
      const nd_index<partition_ndim()>& part_idx,
      const nd_index<element_ndim()>& elem_idx) {
    nd_index<all_ndim()> all_idx;
    all_idx.template set_subset<partition_ndim(), partition_dims()>(part_idx);
    all_idx.template set_subset<element_ndim(), element_dims()>(elem_idx);
    polymorpher().propagate_index_bottom_up(all_idx);
    return all_idx.template get_subset<top_ndim(), top_dims()>();
  }

  template <class ShardedIndex>
    requires(ShardedIndex::size() == top_ndim())
  MINT_HOST_DEVICE static auto
  convert_partition_and_shared_element_index_to_top_index(
      const nd_index<partition_ndim()>& part_idx,
      const ShardedIndex& sharded_index) {
    return convert_partition_and_element_index_to_top_index(
        part_idx,
        convert_sharded_element_index_to_element_index(sharded_index));
  }

  MINT_HOST_DEVICE void print() const {
    printf("distributed_tensor_descriptor: {");
    printf("polymorpher(): ");
    polymorpher().print();
    printf("top_dim_aliase(): ");
    top_dim_aliases().print();
    printf("parition_dim_aliase(): ");
    partition_dim_aliases().print();
    printf("element_dim_aliase(): ");
    element_dim_aliases().print();
    printf("}");
  }

  static_assert(
      bottom_ndim() == partition_ndim() + element_ndim(),
      "Polymorpher bottom dimension must match combined partition+element ndim!");
};

} // namespace tensor
} // namespace mint
