#pragma once
#include <mint/core.h>
#include <mint/poly/fused_morpher.h>

namespace mint {
namespace poly {

// polymorpher
//   kDimPairs: nd_array<index_t, num_dim_pair, 2, 2>
//     kDimPairs[pair_id][bottom_or_top][morpher_local_dim_id]
//     bottom_or_top: [0]=bottom_or_top, [1]=top_or_bottom
//     morpher_local_dim_id: [0]=morpher_id, [1]=local_dim_id
//   kMorpherAliases: array<morpher_alias_type, num_of_morpher>
//   kDimAliases: array<dim_alias_type, num_of_all_dim>
template <
    class Morphers,
    auto kDimPairs,
    class AllLengths,
    auto kMorpherAliases,
    auto kDimAliases>
  requires(
      Morphers::size() == kMorpherAliases.size() &&
      AllLengths::size() == kDimAliases.size())
struct polymorpher : fused_morpher<Morphers, kDimPairs> {
  using FusedMorpher = fused_morpher<Morphers, kDimPairs>;
  using FusedMorpher::fused_morpher;

  static_assert(FusedMorpher::all_ndim() == AllLengths::size());

  using morpher_alias_type =
      remove_cvref_t<decltype(kMorpherAliases)>::value_type;
  using dim_alias_type = remove_cvref_t<decltype(kDimAliases)>::value_type;

  const AllLengths all_lengths_{};

  constexpr polymorpher() = default;

  MINT_HOST_DEVICE constexpr polymorpher(
      const Morphers& morphers,
      const AllLengths& all_lengths)
      : FusedMorpher::fused_morpher(morphers), all_lengths_(all_lengths) {}

  MINT_HOST_DEVICE static index_t consteval num_morpher() {
    return FusedMorpher::num_morpher();
  }

  MINT_HOST_DEVICE static consteval auto morpher_aliases() {
    return kMorpherAliases;
  }

  MINT_HOST_DEVICE static consteval auto dim_aliases() {
    return kDimAliases;
  }

  MINT_HOST_DEVICE static consteval auto top_dim_aliases() {
    return kDimAliases.template get_subset<
        FusedMorpher::top_ndim(),
        FusedMorpher::top_dims()>();
  }

  MINT_HOST_DEVICE static consteval auto bottom_dim_aliases() {
    return kDimAliases.template get_subset<
        FusedMorpher::bottom_ndim(),
        FusedMorpher::bottom_dims()>();
  }

  MINT_HOST_DEVICE constexpr const AllLengths& all_lengths() const {
    return all_lengths_;
  }

  MINT_HOST_DEVICE static consteval auto alias_to_morpher() {
    static_map<morpher_alias_type, index_t, FusedMorpher::num_morpher()> ret;
    for (index_t i = 0; i < FusedMorpher::num_morpher(); i++)
      ret[kMorpherAliases[i]] = i;
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto alias_to_dim() {
    static_map<dim_alias_type, index_t, FusedMorpher::all_ndim()> ret;
    for (index_t i = 0; i < FusedMorpher::all_ndim(); i++)
      ret[kDimAliases[i]] = i;
    return ret;
  }

  // given dim_alias, return the first (morpher, local_dim) copy
  MINT_HOST_DEVICE static consteval auto alias_to_morpher_local_dim() {
    static_map<dim_alias_type, nd_index<2>, FusedMorpher::all_ndim()> ret;
    for (index_t i = 0; i < FusedMorpher::all_ndim(); i++)
      ret[kDimAliases[i]] = FusedMorpher::unique_dim_to_morpher_local(i, 0);
    return ret;
  }

  MINT_HOST_DEVICE void print() const {
    printf("polymorpher: {");
    FusedMorpher::print();
    printf(", ");
    printf("all_lengths(): ");
    all_lengths().print();
    printf(", ");
    printf("morpher_aliases(): ");
    morpher_aliases().print();
    printf(", ");
    printf("dim_aliases(): ");
    dim_aliases().print();
    printf(", ");
    printf("top_dim_aliases(): ");
    top_dim_aliases().print();
    printf(", ");
    printf("bottom_dim_aliases(): ");
    bottom_dim_aliases().print();
    printf("}");
  }
};

// make polymorpher from fused_morpher
//   DimALiasOrderedLengths:
//     all_lengths of fused_morpher unique dims
//     in the order of kAliasToMorpherLocalDim
template <
    auto kDimPairs,
    auto kAliasToMorpher,
    auto kAliasToMorpherLocalDim,
    class Morphers,
    class DimAliasOrderedLengths>
MINT_HOST_DEVICE constexpr auto make_polymorpher(
    const Morphers& morphers,
    const DimAliasOrderedLengths& dim_alias_orderded_lengths) {
  using FusedMorpher = fused_morpher<Morphers, kDimPairs>;
  using morpher_alias_type =
      typename remove_cvref_t<decltype(kAliasToMorpher)>::key_type;
  using dim_alias_type =
      typename remove_cvref_t<decltype(kAliasToMorpherLocalDim)>::key_type;

  constexpr index_t num_morpher = FusedMorpher::num_morpher();
  constexpr index_t ndim = FusedMorpher::all_ndim();

  static_assert(
      kAliasToMorpher.size() == num_morpher &&
      kAliasToMorpherLocalDim.size() == ndim &&
      DimAliasOrderedLengths::size() == ndim);

  // sanity check: all morpher has alias
  constexpr auto has_morpher_aliases = [&]() {
    array<bool, num_morpher> ret;
    ret.fill(false);
    for (auto [alias, morpher] : kAliasToMorpher)
      ret[morpher] = true;
    return ret;
  }();

  static_assert(std::all_of(
      has_morpher_aliases.begin(), has_morpher_aliases.end(), [](index_t v) {
        return v;
      }));

  // sanity check: all dim has alias
  constexpr auto has_dim_aliases = [&]() {
    array<bool, ndim> ret;
    ret.fill(false);
    for (auto [alias, morpher_local] : kAliasToMorpherLocalDim) {
      index_t dim = FusedMorpher::morpher_local_to_unique_dim(
          morpher_local[0], morpher_local[1]);
      ret[dim] = true;
    }
    return ret;
  }();

  static_assert(std::all_of(
      has_dim_aliases.begin(), has_dim_aliases.end(), [](index_t v) {
        return v;
      }));

  constexpr auto morpher_aliases = [&]() {
    array<morpher_alias_type, num_morpher> ret;
    for (auto [alias, morpher] : kAliasToMorpher)
      ret[morpher] = alias;
    return ret;
  }();

  constexpr auto dim_aliases = [&]() {
    array<dim_alias_type, ndim> ret;
    for (auto [alias, morpher_local] : kAliasToMorpherLocalDim) {
      index_t dim = FusedMorpher::morpher_local_to_unique_dim(
          morpher_local[0], morpher_local[1]);
      ret[dim] = alias;
    }
    return ret;
  }();

  constexpr auto pos_to_dim_alias = [&]() {
    array<dim_alias_type, ndim> ret;
    index_t pos = 0;
    for (auto [alias, morpher_local] : kAliasToMorpherLocalDim)
      ret[pos++] = alias;
    return ret;
  }();

  // all_lengths in the order of unique dims
  nd_index<ndim> all_lengths;
  static_for_n<ndim>()([&](auto pos) {
    constexpr auto alias = pos_to_dim_alias[pos];
    constexpr auto morpher_local = kAliasToMorpherLocalDim[alias];
    constexpr index_t dim = FusedMorpher::morpher_local_to_unique_dim(
        morpher_local[0], morpher_local[1]);
    all_lengths[dim] = dim_alias_orderded_lengths[pos];
  });

  return polymorpher<
      Morphers,
      kDimPairs,
      remove_cvref_t<decltype(all_lengths)>,
      morpher_aliases,
      dim_aliases>{morphers, all_lengths};
}

} // namespace poly
} // namespace mint
