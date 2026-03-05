#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor/distributed_tensor_descriptor.h>

namespace mint {
namespace tensor {

template <
    index_t... kWindowLengths,
    class PartitionInfo,
    alias_t... kTileDimAliases,
    alias_t... kPartitionDimAliases>
MINT_HOST_DEVICE constexpr auto make_simple_distribution(
    index_sequence<kWindowLengths...>,
    PartitionInfo,
    sequence<alias_t, kTileDimAliases...>,
    sequence<alias_t, kPartitionDimAliases...>) {
  constexpr auto scan = [](const auto& lengths) {
    constexpr index_t kN = remove_cvref_t<decltype(lengths)>::size();
    nd_index<kN> ret;
    ret[kN - 1] = lengths[kN - 1];
    for (index_t i = kN - 2; i >= 0; i--)
      ret[i] = ret[i + 1] * lengths[i];
    return ret;
  };

  constexpr auto revert_scan = []<auto scans>() {
    constexpr index_t kN = remove_cvref_t<decltype(scans)>::size();
    nd_index<kN> ret;
    ret[kN - 1] = scans[kN - 1];
    static_for_n<kN - 1>()([&](auto i) {
      static_assert(
          scans[i] % scans[i + 1] == 0,
          "wrong! not dividable, invalid distribution");
      ret[i] = scans[i] / scans[i + 1];
    });
    return ret;
  };

  constexpr auto tile_dim_aliases = sequence<alias_t, kTileDimAliases...>{};
  constexpr auto part_dim_aliases =
      sequence<alias_t, kPartitionDimAliases...>{};

  constexpr index_t part_ndim = PartitionInfo::partition_ndim();
  constexpr auto part_lengths = PartitionInfo::partition_lengths();
  constexpr auto part_length_scans = scan(part_lengths);
  constexpr index_t part_num = part_length_scans[0];

  constexpr index_t tile_ndim = sizeof...(kWindowLengths);
  constexpr auto tile_lengths = nd_index<tile_ndim>{kWindowLengths...};
  constexpr auto tile_length_scans = scan(tile_lengths);
  constexpr index_t tile_size = tile_length_scans[0];

  static_assert(tile_size % part_num == 0);

  constexpr index_t element_num = tile_size / part_num;
  constexpr auto p_o_lengths = [&]() {
    nd_index<part_ndim + 1> ret;
    for (index_t i = 0; i < part_ndim; i++)
      ret[i] = part_lengths[i];
    ret[part_ndim] = element_num;
    return ret;
  }();
  constexpr auto p_o_length_scans = scan(p_o_lengths);

  constexpr auto merge_sorted = [](const auto& a, const auto& b) {
    constexpr index_t na = remove_cvref_t<decltype(a)>::size();
    constexpr index_t nb = remove_cvref_t<decltype(b)>::size();
    nd_index<na + 1> a_pos;
    nd_index<nb + 1> b_pos;

    index_t ia = 0;
    index_t ib = 0;
    index_t pos = 0;
    index_t va = a[0];
    index_t vb = b[0];
    while (ia < na or ib < nb) {
      if (ia < na)
        va = math::min(va, a[ia]);
      else
        va = numeric_limits<index_t>::lowest();

      if (ib < nb)
        vb = math::min(vb, b[ib]);
      else
        vb = numeric_limits<index_t>::lowest();

      if (va > vb) {
        a_pos[ia] = pos;
        ia++;
      } else if (va < vb) {
        b_pos[ib] = pos;
        ib++;
      } else {
        a_pos[ia] = pos;
        b_pos[ib] = pos;
        ia++;
        ib++;
      }

      pos++;
    }

    a_pos[na] = pos;
    b_pos[nb] = pos;

    return mint::make_tuple(a_pos, b_pos, pos);
  };

  constexpr auto tmp = merge_sorted(tile_length_scans, p_o_length_scans);
  constexpr auto tile_pos = tmp[0_ic];
  constexpr auto p_o_pos = tmp[1_ic];
  constexpr index_t mid_ndim = tmp[2_ic];

  constexpr auto mid_scans = [&]() {
    nd_index<mid_ndim> ret;
    for (index_t i = 0; i < tile_ndim; i++)
      ret[tile_pos[i]] = tile_length_scans[i];
    for (index_t i = 0; i < part_ndim + 1; i++)
      ret[p_o_pos[i]] = p_o_length_scans[i];
    return ret;
  }();

  constexpr auto mid_lengths = revert_scan.template operator()<mid_scans>();

  constexpr index_t mid_p_ndim = p_o_pos[part_ndim];
  constexpr index_t element_ndim = mid_ndim - mid_p_ndim;

  constexpr auto tile_morphers = generate_tuple<tile_ndim>([&](auto i) {
    constexpr index_t bot_ndim = tile_pos[i + 1] - tile_pos[i];
    nd_index<bot_ndim> bot_lengths;
    for (index_t j = 0; j < bot_ndim; j++)
      bot_lengths[j] = mid_lengths[tile_pos[i] + j];
    return mint::poly::split<nd_index<bot_ndim>>{bot_lengths};
  });
  constexpr auto part_morphers = generate_tuple<part_ndim>([&](auto i) {
    constexpr index_t top_ndim = p_o_pos[i + 1] - p_o_pos[i];
    nd_index<top_ndim> top_lengths;
    for (index_t j = 0; j < top_ndim; j++)
      top_lengths[j] = mid_lengths[p_o_pos[i] + j];
    return mint::poly::merge<nd_index<top_ndim>>{top_lengths};
  });
  constexpr auto morphers = tuple_cat(tile_morphers, part_morphers);
  constexpr auto dim_pairs = [&]() {
    nd_array<index_t, mid_p_ndim, 2, 2> ret;
    index_t tile_cnt = 0;
    index_t part_cnt = 0;
    for (index_t i = 0; i < mid_p_ndim; i++) {
      while (tile_pos[tile_cnt + 1] <= i)
        tile_cnt++;
      while (p_o_pos[part_cnt + 1] <= i)
        part_cnt++;
      ret[i][0] = {tile_cnt, i - tile_pos[tile_cnt]};
      ret[i][1] = {tile_ndim + part_cnt, 1 + i - p_o_pos[part_cnt]};
    }
    return ret;
  }();

  constexpr auto alias_to_morpher = [&]() {
    static_map<alias_t, index_t, tile_ndim + part_ndim> ret;
    // reuse tile_dim_alias for alias of morpher for tile dim
    for (index_t i = 0; i < tile_ndim; i++)
      ret[tile_dim_aliases[i]] = i;
    // reuse part_dim_alias for alias of morpher for partition
    for (index_t i = 0; i < part_ndim; i++)
      ret[part_dim_aliases[i]] = tile_ndim + i;
    return ret;
  }();

  constexpr index_t all_ndim = tile_ndim + part_ndim + mid_ndim;

  constexpr auto alias_to_morpher_local_dim = [&]() {
    static_map<alias_t, nd_index<2>, all_ndim> ret;
    ret.clear();
    // tile dim
    static_for_n<tile_ndim>()([&](auto i) {
      constexpr auto morpher = tile_morphers[i];
      ret[tile_dim_aliases[i]] = {i, morpher.top_dims()[0]};
    });
    // mid dim
    char tmp[16];
    index_t tile_cnt = 0;
    for (index_t i = 0; i < mid_ndim; i++) {
      while (tile_pos[tile_cnt + 1] <= i)
        tile_cnt++;
      integer_to_string(i - tile_pos[tile_cnt], tmp);
      auto dim_alias = tile_dim_aliases[tile_cnt].append("_").append(tmp);
      ret[dim_alias] = {tile_cnt, i - tile_pos[tile_cnt]};
    }
    // part dim
    static_for_n<part_ndim>()([&](auto i) {
      ret[part_dim_aliases[i]] = {
          tile_ndim + i, part_morphers[i].bottom_dims()[0]};
    });
    return ret;
  }();

  constexpr auto lengths = [&]() {
    nd_index<all_ndim> ret;
    index_t cnt = 0;
    // tile dim
    for (index_t i = 0; i < tile_ndim; i++)
      ret[cnt++] = tile_lengths[i];
    // mid dim
    for (index_t i = 0; i < mid_ndim; i++)
      ret[cnt++] = mid_lengths[i];
    // part dim
    for (index_t i = 0; i < part_ndim; i++)
      ret[cnt++] = part_lengths[i];
    return ret;
  }();

  constexpr auto poly = mint::poly::
      make_polymorpher<dim_pairs, alias_to_morpher, alias_to_morpher_local_dim>(
          morphers, lengths);

  constexpr auto element_dim_aliases = [&]() {
    array<alias_t, element_ndim> ret;
    index_t cnt = 0;
    char tmp[16];
    index_t tile_cnt = 0;
    for (index_t i = 0; i < mid_ndim; i++) {
      while (tile_pos[tile_cnt + 1] <= i)
        tile_cnt++;
      integer_to_string(i - tile_pos[tile_cnt], tmp);
      auto dim_alias = tile_dim_aliases[tile_cnt].append("_").append(tmp);
      index_t dim = poly.alias_to_dim()[dim_alias];
      if (poly.is_bottom_dim()[dim])
        ret[cnt++] = dim_alias;
    }
    return ret;
  }();

#if 0
  if (!std::is_constant_evaluated()) {
    printf("part_ndim:%d", part_ndim);
    printf("\npart_lengths:");
    part_lengths.print();
    printf("\npart_length_scans:");
    part_length_scans.print();
    printf("\npart_num:%d", part_num);
    printf("\ntile_ndim:%d", tile_ndim);
    printf("\ntile_lengths:");
    tile_lengths.print();
    printf("\ntile_length_scans:");
    tile_length_scans.print();
    printf("\ntile_size:%d", tile_size);
    printf("\nelem_num:%d", elem_num);
    printf("\np_o_lengths:");
    p_o_lengths.print();
    printf("\np_o_length_scans:");
    p_o_length_scans.print();
    printf("\ntmp:");
    tmp.print();
    printf("\ntile_pos:");
    tile_pos.print();
    printf("\np_o_pos:");
    p_o_pos.print();
    printf("\nmid_ndim:%d", mid_ndim);
    printf("\nmid_scans:");
    mid_scans.print();
    printf("\nmid_lengths:");
    mid_lengths.print();
    printf("\nmid_p_ndim:%d", mid_p_ndim);
    printf("\nelem_ndim:%d", elem_ndim);
    printf("\n");
  }
#endif

  return distributed_tensor_descriptor<
      poly,
      tile_dim_aliases,
      part_dim_aliases,
      element_dim_aliases>{};
}

// FIXME: assume alias of morpher of tile_dim is the same as tile_dim itself
template <class InDstr, class DimAlias, DimAlias... kExtractedDimAliases>
MINT_HOST_DEVICE constexpr auto extract_dimension_distribution(
    const InDstr& /*in_dstr*/,
    sequence<DimAlias, kExtractedDimAliases...>) {
  constexpr auto extracted_dim_aliases =
      array<DimAlias, sizeof...(kExtractedDimAliases)>{kExtractedDimAliases...};

  constexpr auto old_poly = InDstr::tensor_desc().polymorpher();

  constexpr index_t old_num_morpher = old_poly.num_morpher();

  // sanitiy check
  static_for_n<extracted_dim_aliases.size()>()([&](auto i) {
    static_assert(
        old_poly.alias_to_dim().contains(extracted_dim_aliases[i]),
        "wrong! this dim alias not exist");
  });

  // extract all morphers, except those tile_dim_morphers not specified in
  // kExtractedDimAliases
  constexpr auto should_extract_morpher = [&]() {
    array<bool, old_num_morpher> ret;
    ret.fill(true);
    for (auto imorpher : InDstr::tile_morphers())
      ret[imorpher] = false;
    for (auto dim_alias : extracted_dim_aliases) {
      auto morpher_alias = dim_alias;
      auto imorpher = old_poly.alias_to_morpher()[morpher_alias];
      ret[imorpher] = true;
    }
    return ret;
  }();

  constexpr index_t num_extracted_morpher = std::count(
      should_extract_morpher.begin(), should_extract_morpher.end(), true);

  constexpr auto extracted_morpher_aliases = [&]() {
    array<DimAlias, num_extracted_morpher> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < old_num_morpher; i++)
      if (should_extract_morpher[i])
        ret[cnt++] = old_poly.morpher_aliases()[i];
    return ret;
  }();

  constexpr auto extracted_poly = mint::poly::extract_polymorpher(
      old_poly, to_sequence<extracted_morpher_aliases>());

  constexpr auto remove_dims_and_cnt = [&]() {
    array<DimAlias, old_poly.all_ndim()> ret;
    ret.fill(DimAlias{});
    index_t cnt = 0;
    static_for_n<InDstr::tile_morphers().size()>()([&](auto i) {
      constexpr auto imorpher = InDstr::tile_morphers()[i];
      constexpr auto morpher = old_poly.morphers().template at<imorpher>();
      if (not should_extract_morpher[imorpher]) {
        for (auto local_dim : morpher.bottom_dims()) {
          auto dim = old_poly.morpher_local_to_unique_dim(imorpher, local_dim);
          if (not InDstr::is_element_dim()[dim])
            ret[cnt++] = old_poly.dim_aliases()[dim];
        }
      }
    });
    return mint::make_tuple(ret, cnt);
  }();

  constexpr auto remove_dims_tmp = remove_dims_and_cnt[0_ic];
  constexpr index_t num_remove_dim = remove_dims_and_cnt[1_ic];

  constexpr auto remove_morpher_infos =
      generate_tuple<num_remove_dim>([&](auto i) {
        constexpr auto dim_alias = remove_dims_tmp[i];
        // use same alias for poly::insert morpher as the dim it removes
        constexpr auto morpher_alias = dim_alias;
        return mint::make_tuple(
            poly::insert<1>{},
            constant<morpher_alias>{},
            sequence<DimAlias, dim_alias>{});
      });

  constexpr auto new_poly = pack_arg<num_remove_dim>([&](const auto&... args) {
    return add_to_polymorpher(
        extracted_poly, sequence<DimAlias>{}, nd_index<0>{}, args...);
  })(remove_morpher_infos);

  // new_part_dim_aliases
  constexpr auto new_part_dim_aliases_and_cnt = [&]() {
    constexpr index_t old_part_ndim = InDstr::partition_ndim();
    array<DimAlias, old_part_ndim> ret;
    ret.fill(DimAlias{});
    index_t cnt = 0;
    for (index_t dim : InDstr::partition_dims()) {
      auto dim_alias = old_poly.dim_aliases()[dim];
      if (new_poly.alias_to_dim().contains(dim_alias))
        ret[cnt++] = dim_alias;
    }
    return mint::make_tuple(ret, cnt);
  }();

  constexpr auto new_part_dim_aliases_tmp = new_part_dim_aliases_and_cnt[0_ic];
  constexpr index_t new_part_ndim = new_part_dim_aliases_and_cnt[1_ic];

  constexpr auto new_part_dim_aliases = [&]() {
    array<DimAlias, new_part_ndim> ret;
    std::copy(
        new_part_dim_aliases_tmp.begin(),
        new_part_dim_aliases_tmp.begin() + new_part_ndim,
        ret.begin());
    return ret;
  }();

  // new_elem_dim_aliases
  constexpr auto new_elem_dim_aliases_and_cnt = [&]() {
    constexpr index_t old_elem_ndim = InDstr::element_ndim();
    array<DimAlias, old_elem_ndim> ret;
    ret.fill(DimAlias{});
    index_t cnt = 0;
    for (index_t dim : InDstr::element_dims()) {
      auto dim_alias = old_poly.dim_aliases()[dim];
      if (new_poly.alias_to_dim().contains(dim_alias))
        ret[cnt++] = dim_alias;
    }
    return mint::make_tuple(ret, cnt);
  }();

  constexpr auto new_elem_dim_aliases_tmp = new_elem_dim_aliases_and_cnt[0_ic];
  constexpr index_t new_elem_ndim = new_elem_dim_aliases_and_cnt[1_ic];

  constexpr auto new_elem_dim_aliases = [&]() {
    array<DimAlias, new_elem_ndim> ret;
    std::copy(
        new_elem_dim_aliases_tmp.begin(),
        new_elem_dim_aliases_tmp.begin() + new_elem_ndim,
        ret.begin());
    return ret;
  }();

  constexpr auto new_dstr = distributed_tensor_descriptor<
      new_poly,
      extracted_dim_aliases,
      new_part_dim_aliases,
      new_elem_dim_aliases>{};

  return new_dstr;
}

template <
    class PolyMorpher,
    index_t kBotPartNDim,
    nd_index<kBotPartNDim> kBotPartDims,
    index_t kBotElemNDim,
    nd_index<kBotElemNDim> kBotElemDims>
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_descriptor_z2(
    PolyMorpher& poly,
    integral_constant<nd_index<kBotPartNDim>, kBotPartDims>,
    integral_constant<nd_index<kBotElemNDim>, kBotElemDims>) {
  /// Creates dimension name aliases (e.g., "bot0", "bot1", ...)
  constexpr auto make_dim_aliases =
      []<index_t NumDims, index_t Offset = 0>(const char* prefix) {
        array<alias_t, NumDims> aliases;
        char tmp[16];
        for (index_t i = 0; i < NumDims; i++) {
          integer_to_string(static_cast<int>(i + Offset), tmp);
          aliases[i] = alias_t(prefix).append(tmp);
        }
        return aliases;
      };

  // Verify poly contains exactly one morpher
  static_assert(
      PolyMorpher::num_morpher() == 1, "poly must contain exactly one morpher");

  auto z2_morpher = poly.morphers()[0_ic];
  using Morpher = remove_cvref_t<decltype(z2_morpher)>;

  // Verify the morpher is a z2 linear morpher
  static_assert(
      poly::is_z2_linear_morpher<Morpher>::value,
      "morpher must be a z2 linear morpher");

  constexpr auto new_poly = make_z2_polymorpher_default_alias(Morpher{});

  constexpr index_t bottom_ndim = PolyMorpher::FusedMorpher::bottom_ndim();
  constexpr index_t top_ndim = PolyMorpher::FusedMorpher::top_ndim();

  constexpr auto bot_dim_aliases =
      make_dim_aliases.template operator()<bottom_ndim>("bot");

  constexpr auto bot_part_aliases = [&]() {
    array<alias_t, kBotPartNDim> part_aliases;
    for (index_t i = 0; i < kBotPartNDim; i++) {
      part_aliases[i] = bot_dim_aliases[kBotPartDims[i]];
    }
    return part_aliases;
  }();

  constexpr auto bot_element_aliases = [&]() {
    array<alias_t, kBotElemNDim> element_aliases;
    for (index_t i = 0; i < kBotElemNDim; i++) {
      element_aliases[i] = bot_dim_aliases[kBotElemDims[i]];
    }
    return element_aliases;
  }();

  constexpr auto top_dim_aliases =
      make_dim_aliases.template operator()<top_ndim>("top");

  return distributed_tensor_descriptor<
      new_poly,
      top_dim_aliases,
      bot_part_aliases,
      bot_element_aliases>{};
}

template <
    class PolyMorpher,
    index_t kBotPartNDim,
    nd_index<kBotPartNDim> kBotPartDims,
    index_t kBotElemNDim,
    nd_index<kBotElemNDim> kBotElemDims>
MINT_HOST_DEVICE constexpr auto make_distributed_tensor_descriptor(
    PolyMorpher& poly,
    integral_constant<nd_index<kBotPartNDim>, kBotPartDims> bot_part_dims,
    integral_constant<nd_index<kBotElemNDim>, kBotElemDims> bot_elem_dims) {
  if constexpr (
      PolyMorpher::num_morpher() == 1 and
      poly::is_z2_linear_morpher<
          remove_cvref_t<decltype(poly.morphers()[0_ic])>>::value) {
    return make_distributed_tensor_descriptor_z2(
        poly, bot_part_dims, bot_elem_dims);
  } else {
    static_assert(dependent_false<PolyMorpher>{}, "not implemented");
  }
}

template <index_t TopNDim, nd_index<TopNDim> TopDims, class PartitionInfo>
MINT_HOST_DEVICE constexpr auto make_simple_distribution_z2(
    integral_constant<nd_index<TopNDim>, TopDims>,
    const PartitionInfo& /*partition_info*/) {
  constexpr auto scan = [](const auto& lengths) {
    constexpr index_t kN = remove_cvref_t<decltype(lengths)>::size();
    nd_index<kN> ret;
    ret[kN - 1] = lengths[kN - 1];
    for (index_t i = kN - 2; i >= 0; i--)
      ret[i] = ret[i + 1] * lengths[i];
    return ret;
  };

  constexpr auto top_size = scan(TopDims)[0];

  constexpr index_t part_num = PartitionInfo::partition_num();

  constexpr auto m0 = poly::make_z2_pass_through_morpher(constant<TopDims>{});

  constexpr auto m1 = reshape_bottom(
      m0, constant<nd_index<2>{part_num, top_size / part_num}>{});

  constexpr auto poly = make_z2_polymorpher_default_alias(m1);
  return make_distributed_tensor_descriptor(
      poly, constant<nd_index<1>{0}>{}, constant<nd_index<1>{1}>{});
}

} // namespace tensor
} // namespace mint
