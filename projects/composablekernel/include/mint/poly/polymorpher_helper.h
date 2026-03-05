#pragma once
#include <mint/core.h>
#include <mint/poly/polymorpher.h>
#include <mint/poly/z2_linear_morpher_helper.h>

namespace mint {
namespace poly {

// ExtraMorpherInfos:
//   tuple<
//     Morpher,
//     constant<kMorpherAlias>,
//     sequence<DimAlias, kMorpherDimAliases...>>
//    reqiures: Morpher::all_ndim() == sizeof...(kMorpherDimAliases)
template <
    class OldPolymorpher,
    class DimAlias,
    DimAlias... kExtraDimAliases,
    class ExtraDimLengths,
    class... ExtraMorpherInfos>
MINT_HOST_DEVICE constexpr auto add_to_polymorpher(
    const OldPolymorpher& old_polymorpher,
    const sequence<DimAlias, kExtraDimAliases...>,
    const ExtraDimLengths& extra_dim_lengths,
    const ExtraMorpherInfos&... extra_morpher_infos_in) {
  using morpher_alias_type =
      remove_cvref_t<decltype(OldPolymorpher::alias_to_morpher())>::key_type;
  using dim_alias_type =
      remove_cvref_t<decltype(OldPolymorpher::alias_to_dim())>::key_type;

  constexpr auto extra_dim_aliases = sequence<DimAlias, kExtraDimAliases...>{};

  // sanity check
  static_assert(sizeof...(kExtraDimAliases) == ExtraDimLengths::size());

  constexpr index_t old_num_morpher = OldPolymorpher{}.morphers().size();
  constexpr index_t extra_num_morpher = sizeof...(ExtraMorpherInfos);
  constexpr index_t new_num_morpher = old_num_morpher + extra_num_morpher;

  constexpr index_t old_ndim = OldPolymorpher::all_ndim();
  constexpr index_t extra_ndim = sizeof...(kExtraDimAliases);
  constexpr index_t new_ndim = old_ndim + extra_ndim;

  constexpr index_t old_num_dim_pair = OldPolymorpher::dim_pairs().lengths()[0];

  const auto extra_morphers = mint::make_tuple(extra_morpher_infos_in[0_ic]...);
  constexpr auto extra_morpher_aliases =
      array<morpher_alias_type, extra_num_morpher>{
          decltype(extra_morpher_infos_in[1_ic]){}.value...};
  constexpr auto extra_morpher_dim_aliases =
      mint::make_tuple(decltype(extra_morpher_infos_in[2_ic]){}...);

  // sanity check
  static_for_n<extra_ndim>()([&](auto i) {
    static_assert(
        !OldPolymorpher::alias_to_dim().contains(extra_dim_aliases[i]),
        "wrong! dim_alias already exists in old polymorpher");
  });

  // sanity check
  static_for_n<extra_num_morpher>()([&](auto i) {
    static_assert(
        !OldPolymorpher::alias_to_morpher().contains(extra_morpher_aliases[i]),
        "wrong! morpher_alias already exists in old polymorpher");
  });

  const auto new_morphers =
      tuple_cat(old_polymorpher.morphers(), extra_morphers);

  constexpr auto new_alias_to_morpher = [&]() {
    static_map<morpher_alias_type, index_t, new_num_morpher> ret;
    ret.clear();
    ret.merge(OldPolymorpher::alias_to_morpher());
    for (index_t i = 0; i < extra_num_morpher; i++)
      ret[extra_morpher_aliases[i]] = old_num_morpher + i;
    return ret;
  }();

  // extra_num_dim_pair is no larger than # of dimensions in extra morphers
  constexpr index_t max_extra_num_dim_pair = [&]() {
    index_t cnt = 0;
    static_for_n<extra_num_morpher>()(
        [&](auto i) { cnt += extra_morpher_dim_aliases[i].size(); });
    return cnt;
  }();

  constexpr auto tmp = [&]() {
    // new_alias_to_morpher_local_dim
    static_map<dim_alias_type, nd_index<2>, new_ndim> ret0;
    ret0.clear();
    ret0.merge(OldPolymorpher::alias_to_morpher_local_dim());

    // extra_dim_pairs
    nd_array<index_t, max_extra_num_dim_pair, 2, 2> ret1;
    ret1.fill(-1);

    // num_extra_dim_pair
    index_t cnt = 0;

    static_for_n<extra_num_morpher>()([&](auto i) {
      constexpr auto dim_aliases = extra_morpher_dim_aliases[i];
      for (index_t j = 0; j < dim_aliases.size(); j++) {
        auto alias = dim_aliases[j];
        if (ret0.find(alias) == ret0.end()) {
          ret0[alias] = {old_num_morpher + i, j};
        } else {
          ret1[cnt][0] = {ret0[alias][0], ret0[alias][1]};
          ret1[cnt][1] = {old_num_morpher + i, j};
          cnt++;
        }
      }
    });
    return mint::make_tuple(ret0, ret1, cnt);
  }();

  constexpr auto new_alias_to_morpher_local_dim = tmp[0_ic];
  constexpr auto extra_dim_pairs_tmp = tmp[1_ic];
  constexpr index_t extra_num_dim_pair = tmp[2_ic];

  constexpr auto new_dim_pairs = [&]() {
    nd_array<index_t, old_num_dim_pair + extra_num_dim_pair, 2, 2> ret;
    for (index_t i = 0; i < old_num_dim_pair; i++)
      ret[i] = OldPolymorpher::dim_pairs()[i];
    for (index_t i = 0; i < extra_num_dim_pair; i++)
      ret[old_num_dim_pair + i] = extra_dim_pairs_tmp[i];
    return ret;
  }();

  using NewFusedMorpher =
      fused_morpher<remove_cvref_t<decltype(new_morphers)>, new_dim_pairs>;

  static_assert(NewFusedMorpher::all_ndim() == new_ndim);

  // FIXME: this assume nd_index for all_lengths
  const auto new_all_lengths = [&]() {
    nd_index<new_ndim> ret;

    // copy all_lengths from old_polymorpher
    static_for_n<old_ndim>()([&](auto i) {
      constexpr auto alias = OldPolymorpher::dim_aliases()[i];
      constexpr index_t old_dim = OldPolymorpher::alias_to_dim()[alias];
      constexpr auto new_morpher_local = new_alias_to_morpher_local_dim[alias];
      constexpr index_t new_dim = NewFusedMorpher::morpher_local_to_unique_dim(
          new_morpher_local[0], new_morpher_local[1]);
      ret[new_dim] = old_polymorpher.all_lengths()[old_dim];
    });

    // copy all_lengths from extra_dim_lengths
    static_for_n<extra_ndim>()([&](auto i) {
      constexpr auto alias = extra_dim_aliases[i];
      constexpr auto new_morpher_local = new_alias_to_morpher_local_dim[alias];
      constexpr index_t new_dim = NewFusedMorpher::morpher_local_to_unique_dim(
          new_morpher_local[0], new_morpher_local[1]);
      ret[new_dim] = extra_dim_lengths[i];
    });

    return ret;
  }();

  return make_polymorpher<
      new_dim_pairs,
      new_alias_to_morpher,
      new_alias_to_morpher_local_dim>(new_morphers, new_all_lengths);
}

template <
    class OldPolymorpher,
    class MorpherAlias,
    MorpherAlias... kMorpherAliases>
  requires(is_same_v<
           typename remove_cvref_t<
               decltype(OldPolymorpher::alias_to_morpher())>::key_type,
           remove_cvref_t<MorpherAlias>>)
MINT_HOST_DEVICE constexpr auto extract_polymorpher(
    const OldPolymorpher& old_polymorpher,
    sequence<MorpherAlias, kMorpherAliases...>) {
  using morpher_alias_type =
      remove_cvref_t<decltype(old_polymorpher.alias_to_morpher())>::key_type;
  using dim_alias_type =
      remove_cvref_t<decltype(old_polymorpher.alias_to_dim())>::key_type;

  constexpr index_t old_ndim = OldPolymorpher::all_ndim();
  constexpr index_t old_num_morpher =
      remove_cvref_t<decltype(old_polymorpher.morphers())>::size();
  constexpr index_t new_num_morpher = sizeof...(kMorpherAliases);

  constexpr auto new_morpher_subset = nd_index<new_num_morpher>{
      OldPolymorpher::alias_to_morpher()[kMorpherAliases]...};

  const auto new_morphers =
      old_polymorpher.morphers()
          .template get_subset<new_morpher_subset.size(), new_morpher_subset>();

  constexpr auto old_to_new_morphers = [&]() {
    nd_index<old_num_morpher> ret;
    ret.fill(-1);
    for (index_t i = 0; i < new_num_morpher; i++)
      ret[new_morpher_subset[i]] = i;
    return ret;
  }();
  constexpr auto old_dim_pairs = OldPolymorpher::dim_pairs();
  constexpr index_t old_num_dim_pair = old_dim_pairs.lengths()[0];
  constexpr index_t new_num_dim_pair = [&]() {
    index_t cnt = 0;
    for (index_t i = 0; i < old_num_dim_pair; i++) {
      index_t new_morpher_bot = old_to_new_morphers[old_dim_pairs[i][0][0]];
      index_t new_morpher_top = old_to_new_morphers[old_dim_pairs[i][1][0]];
      if (new_morpher_bot >= 0 && new_morpher_top >= 0)
        cnt++;
    }
    return cnt;
  }();
  constexpr auto new_dim_pairs = [&]() {
    nd_array<index_t, new_num_dim_pair, 2, 2> ret;
    index_t cnt = 0;
    for (index_t i = 0; i < old_num_dim_pair; i++) {
      index_t new_morpher_bot = old_to_new_morphers[old_dim_pairs[i][0][0]];
      index_t new_morpher_top = old_to_new_morphers[old_dim_pairs[i][1][0]];
      if (new_morpher_bot >= 0 && new_morpher_top >= 0) {
        ret[cnt][0] = {new_morpher_bot, old_dim_pairs[i][0][1]};
        ret[cnt][1] = {new_morpher_top, old_dim_pairs[i][1][1]};
        cnt++;
      }
    }
    return ret;
  }();
  constexpr auto old_alias_to_morpher = OldPolymorpher::alias_to_morpher();
  constexpr auto old_alias_to_unique_dim = OldPolymorpher::alias_to_dim();

  constexpr index_t new_num_morpher_alias = [&]() {
    index_t cnt = 0;
    for (auto [alias, old_morpher] : old_alias_to_morpher) {
      index_t new_morpher = old_to_new_morphers[old_morpher];
      if (new_morpher >= 0)
        cnt++;
    }
    return cnt;
  }();

  constexpr auto new_alias_to_morpher = [&]() {
    static_map<morpher_alias_type, index_t, new_num_morpher_alias> ret;
    for (auto [alias, old_morpher] : old_alias_to_morpher) {
      index_t new_morpher = old_to_new_morphers[old_morpher];
      if (new_morpher >= 0)
        ret[alias] = new_morpher;
    }
    return ret;
  }();

  constexpr auto new_dim_aliases_and_cnt = [&]() {
    array<dim_alias_type, old_ndim> ret;
    ret.fill(dim_alias_type{});

    index_t cnt = 0;
    for (auto [alias, old_unique_dim] : old_alias_to_unique_dim) {
      bool flag = false;
      for (index_t i = 0;
           i < OldPolymorpher::num_duplicate_of_unique_dims()[old_unique_dim];
           i++) {
        auto old_morpher_local =
            OldPolymorpher::unique_dim_to_morpher_local(old_unique_dim, i);
        index_t old_morpher = old_morpher_local[0];
        index_t new_morpher = old_to_new_morphers[old_morpher];
        if (new_morpher >= 0) {
          flag = true;
          continue;
        }
      }
      if (flag)
        ret[cnt++] = alias;
    }
    return mint::make_tuple(ret, cnt);
  }();

  constexpr auto new_dim_aliases_tmp = new_dim_aliases_and_cnt[0_ic];
  constexpr index_t new_ndim = new_dim_aliases_and_cnt[1_ic];

  constexpr auto new_dim_aliases = [&]() {
    array<dim_alias_type, new_ndim> ret;
    std::copy(
        new_dim_aliases_tmp.begin(),
        new_dim_aliases_tmp.begin() + new_ndim,
        ret.begin());
    return ret;
  }();

  constexpr auto new_alias_to_morpher_local_dim = [&]() {
    static_map<dim_alias_type, nd_index<2>, new_ndim> ret;
    for (auto [alias, old_unique_dim] : old_alias_to_unique_dim) {
      for (index_t i = 0;
           i < OldPolymorpher::num_duplicate_of_unique_dims()[old_unique_dim];
           i++) {
        auto old_morpher_local =
            OldPolymorpher::unique_dim_to_morpher_local(old_unique_dim, i);
        index_t old_morpher = old_morpher_local[0];
        index_t old_local_dim = old_morpher_local[1];
        index_t new_morpher = old_to_new_morphers[old_morpher];
        if (new_morpher >= 0)
          ret[alias] = {new_morpher, old_local_dim};
      }
    }
    return ret;
  }();

#if 0
  using NewFusedMorpher =
      fused_morpher<remove_cvref_t<decltype(new_morphers)>, new_dim_pairs>;

  static_assert(new_ndim == NewFusedMorpher::all_ndim());
#endif

  // FIXME: this assume nd_index for all_lengths
  const auto new_all_lengths = [&]() {
    nd_index<new_ndim> ret;
    // copy all_lengths from old_polymorpher
    static_for_n<new_ndim>()([&](auto i) {
      constexpr auto alias = new_dim_aliases[i];
      constexpr index_t old_dim = OldPolymorpher::alias_to_dim()[alias];
      ret[i] = old_polymorpher.all_lengths()[old_dim];
    });
    return ret;
  }();

  return make_polymorpher<
      new_dim_pairs,
      new_alias_to_morpher,
      new_alias_to_morpher_local_dim>(new_morphers, new_all_lengths);
}

// take multiple polymorphers, remove duplicated component morphers, and merge
// them into a single polymorpher
template <class... OldPolymorphers>
MINT_HOST_DEVICE constexpr auto merge_polymorphers(
    const OldPolymorphers&... old_polymorphers_in) {
  constexpr index_t old_num_polymorpher = sizeof...(old_polymorphers_in);
  const auto old_polymorphers = mint::make_tuple(old_polymorphers_in...);

  using morpher_alias_type = remove_cvref_t<
      decltype(old_polymorphers[0_ic].alias_to_morpher())>::key_type;
  using dim_alias_type =
      remove_cvref_t<decltype(old_polymorphers[0_ic].alias_to_dim())>::key_type;

  const auto old_morphers = tuple_cat(old_polymorphers_in.morphers()...);
  constexpr index_t old_num_morpher = decltype(old_morphers)::size();

  constexpr auto tmp = [&]() {
    index_t new_num_morpher = 0;
    nd_index<old_num_morpher> old_to_new_morphers;
    nd_index<old_num_morpher> new_morpher_subset;
    static_map<morpher_alias_type, index_t, old_num_morpher>
        new_alias_to_morpher;
    old_to_new_morphers.fill(-1);
    new_morpher_subset.fill(-1);
    index_t cnt = 0;
    static_for_n<old_num_polymorpher>()([&](auto i) {
      constexpr auto old_alias_to_morpher =
          remove_cvref_t<decltype(old_polymorphers[i])>::alias_to_morpher();
      for (auto [alias, old_morpher] : old_alias_to_morpher) {
        if (new_alias_to_morpher.find(alias) == new_alias_to_morpher.end()) {
          old_to_new_morphers[old_morpher + cnt] = new_num_morpher;
          new_morpher_subset[new_num_morpher] = old_morpher + cnt;
          new_alias_to_morpher[alias] = new_num_morpher;
          new_num_morpher++;
        }
        old_to_new_morphers[old_morpher + cnt] = new_alias_to_morpher[alias];
      }
      cnt += remove_cvref_t<decltype(old_polymorphers[i].morphers())>::size();
    });

    return mint::make_tuple(
        new_num_morpher,
        old_to_new_morphers,
        new_morpher_subset,
        new_alias_to_morpher);
  }();

  constexpr index_t new_num_morpher = tmp[0_ic];
  constexpr auto old_to_new_morphers = tmp[1_ic];
  constexpr auto new_morpher_subset_tmp = tmp[2_ic];
  constexpr auto new_alias_to_morpher_tmp = tmp[3_ic];

  constexpr auto new_morpher_subset = [&]() {
    nd_index<new_num_morpher> ret;
    for (index_t i = 0; i < new_num_morpher; i++)
      ret[i] = new_morpher_subset_tmp[i];
    return ret;
  }();
  constexpr auto new_alias_to_morpher = [&]() {
    static_map<morpher_alias_type, index_t, new_num_morpher> ret;
    ret.clear();
    ret.merge(new_alias_to_morpher_tmp);
    return ret;
  }();

  const auto new_morphers =
      old_morphers
          .template get_subset<new_morpher_subset.size(), new_morpher_subset>();

  // FIXME: there are duplicate of dim pairs
  constexpr index_t new_num_dim_pair = []() {
    index_t cnt = 0;
    static_for_n<old_num_polymorpher>()([&](auto i) {
      constexpr auto old_dim_pairs =
          remove_cvref_t<decltype(old_polymorphers[i])>::dim_pairs();
      cnt += old_dim_pairs.lengths()[0];
    });
    return cnt;
  }();

  constexpr auto new_dim_pairs = [&]() {
    nd_array<index_t, new_num_dim_pair, 2, 2> ret;
    index_t cnt = 0;
    index_t old_morpher_begin = 0;
    static_for_n<old_num_polymorpher>()([&](auto i) {
      using OldPolymorpher = remove_cvref_t<decltype(old_polymorphers[i])>;
      constexpr auto old_dim_pairs = OldPolymorpher::dim_pairs();
      for (index_t j = 0; j < old_dim_pairs.lengths()[0]; j++) {
        index_t old_bot_morpher = old_dim_pairs[j][0][0];
        index_t old_top_morpher = old_dim_pairs[j][1][0];
        index_t new_bot_morpher =
            old_to_new_morphers[old_morpher_begin + old_bot_morpher];
        index_t new_top_morpher =
            old_to_new_morphers[old_morpher_begin + old_top_morpher];
        ret[cnt][0] = {new_bot_morpher, old_dim_pairs[j][0][1]};
        ret[cnt][1] = {new_top_morpher, old_dim_pairs[j][1][1]};
        cnt++;
      }
      old_morpher_begin += OldPolymorpher{}.morphers().size();
    });
    return ret;
  }();

  constexpr index_t new_num_dim_alias_tmp = []() {
    index_t cnt = 0;
    static_for_n<old_num_polymorpher>()([&](auto i) {
      using OldPolymorpher = remove_cvref_t<decltype(old_polymorphers[i])>;
      cnt += OldPolymorpher::alias_to_dim().size();
    });
    return cnt;
  }();

  constexpr auto new_alias_to_morpher_local_dim_tmp = [&]() {
    static_map<dim_alias_type, nd_index<2>, new_num_dim_alias_tmp> ret;
    index_t old_morpher_begin = 0;
    static_for_n<old_num_polymorpher>()([&](auto i) {
      using OldPolymorpher = remove_cvref_t<decltype(old_polymorphers[i])>;
      constexpr auto old_alias_to_unique_dim = OldPolymorpher::alias_to_dim();
      for (auto [alias, old_unique_dim] : old_alias_to_unique_dim) {
        for (index_t j = 0;
             j < OldPolymorpher::num_duplicate_of_unique_dims()[old_unique_dim];
             j++) {
          auto old_morpher_local =
              OldPolymorpher::unique_dim_to_morpher_local(old_unique_dim, j);
          index_t old_morpher = old_morpher_local[0];
          index_t old_local_dim = old_morpher_local[1];
          index_t new_morpher =
              old_to_new_morphers[old_morpher + old_morpher_begin];
          ret[alias] = {new_morpher, old_local_dim};
        }
      }
      old_morpher_begin += OldPolymorpher{}.morphers().size();
    });
    return ret;
  }();

  constexpr auto new_alias_to_morpher_local_dim = [&]() {
    static_map<
        dim_alias_type,
        nd_index<2>,
        new_alias_to_morpher_local_dim_tmp.size()>
        ret;
    ret.clear();
    ret.merge(new_alias_to_morpher_local_dim_tmp);
    return ret;
  }();

  using NewFusedMorpher =
      fused_morpher<remove_cvref_t<decltype(new_morphers)>, new_dim_pairs>;

  constexpr index_t new_ndim = NewFusedMorpher::all_ndim();

  // FIXME: this assume nd_index for all_lengths
  const auto new_all_lengths = [&]() {
    nd_index<new_ndim> ret;
    // copy all_lengths from old_polymorpher
    static_for_n<old_num_polymorpher>()([&](auto ipoly) {
      const auto& old_polymorpher = old_polymorphers[ipoly];
      using OldPolymorpher = remove_cvref_t<decltype(old_polymorpher)>;
      static_for_n<OldPolymorpher::all_ndim()>()([&](auto idim) {
        constexpr auto alias = OldPolymorpher::dim_aliases()[idim];
        static_assert(new_alias_to_morpher_local_dim.contains(alias));
        constexpr index_t old_dim = OldPolymorpher::alias_to_dim()[alias];
        constexpr auto new_morpher_local =
            new_alias_to_morpher_local_dim[alias];
        constexpr index_t new_dim =
            NewFusedMorpher::morpher_local_to_unique_dim(
                new_morpher_local[0], new_morpher_local[1]);
        ret[new_dim] = old_polymorpher.all_lengths()[old_dim];
      });
    });
    return ret;
  }();

  return make_polymorpher<
      new_dim_pairs,
      new_alias_to_morpher,
      new_alias_to_morpher_local_dim>(new_morphers, new_all_lengths);
}

// make a simple polymorpher from a single morpher with custom aliases
// and no dim_pairs
template <
    class Morpher,
    index_t kBotNDim,
    index_t kTopNDim,
    array<alias_t, kBotNDim> BottomAlias,
    array<alias_t, kTopNDim> TopAlias>
MINT_HOST_DEVICE constexpr auto make_simple_polymorpher(
    const Morpher& morpher,
    const nd_index<kBotNDim>& bottom_lengths,
    const nd_index<kTopNDim>& top_lengths,
    integral_constant<array<alias_t, kBotNDim>, BottomAlias>,
    integral_constant<array<alias_t, kTopNDim>, TopAlias>) {
  const auto morphers = mint::make_tuple(morpher);

  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 1> map;
    map["m0"] = 0;
    return map;
  }();

  constexpr index_t kBottomDims = kBotNDim;
  constexpr index_t kTopDims = kTopNDim;
  constexpr index_t kTotalDims = kBottomDims + kTopDims;

  constexpr auto alias_to_dim = [&]() {
    static_map<alias_t, nd_index<2>, kTotalDims> map;

    for (index_t i = 0; i < kBottomDims; ++i) {
      map[BottomAlias[i]] = nd_index<2>{0, i};
    }

    for (index_t i = 0; i < kTopDims; ++i) {
      map[TopAlias[i]] = nd_index<2>{0, kBottomDims + i};
    }

    return map;
  }();

  const auto lengths = [&]() {
    nd_index<kTotalDims> ret;

    for (index_t i = 0; i < kBottomDims; ++i) {
      ret[i] = bottom_lengths[i];
    }

    for (index_t i = 0; i < kTopDims; ++i) {
      ret[kBottomDims + i] = top_lengths[i];
    }

    return ret;
  }();

  constexpr auto empty_dim_pairs = nd_array<index_t, 0, 2, 2>{};

  return poly::
      make_polymorpher<empty_dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);
}

// make a simple polymorpher from a single morpher with default alias and no
// dim_pairs
template <class Morpher, index_t kBotNDim, index_t kTopNDim>
MINT_HOST_DEVICE constexpr auto make_simple_polymorpher_default_alias(
    const Morpher& morpher,
    const nd_index<kBotNDim>& bottom_lengths,
    const nd_index<kTopNDim>& top_lengths) {
  // Generate default aliases for bottom dimensions: "bot0", "bot1", ...
  constexpr auto bottom_aliases = [&]() {
    array<alias_t, kBotNDim> ret;
    char tmp[16];
    for (index_t i = 0; i < kBotNDim; i++) {
      integer_to_string(static_cast<int>(i), tmp);
      ret[i] = alias_t("bot").append(tmp);
    }
    return ret;
  }();

  // Generate default aliases for top dimensions: "top0", "top1", ...
  constexpr auto top_aliases = [&]() {
    array<alias_t, kTopNDim> ret;
    char tmp[16];
    for (index_t i = 0; i < kTopNDim; i++) {
      integer_to_string(static_cast<int>(i), tmp);
      ret[i] = alias_t("top").append(tmp);
    }
    return ret;
  }();

  // Reuse the custom alias version by passing in the generated default aliases
  return make_simple_polymorpher(
      morpher,
      bottom_lengths,
      top_lengths,
      constant<bottom_aliases>{},
      constant<top_aliases>{});
}

// make a simple polymorpher from a single z2 linear morpher
template <
    class Morpher,
    index_t kBotNDim,
    index_t kTopNDim,
    array<alias_t, kBotNDim> BottomAlias,
    array<alias_t, kTopNDim> TopAlias>
  requires(poly::is_z2_linear_morpher<Morpher>::value)
MINT_HOST_DEVICE constexpr auto make_z2_polymorpher(
    Morpher,
    integral_constant<array<alias_t, kBotNDim>, BottomAlias> bottom_alias,
    integral_constant<array<alias_t, kTopNDim>, TopAlias> top_alias) {
  return make_simple_polymorpher(
      Morpher{},
      Morpher::kBottomLengths,
      Morpher::kTopLengths,
      bottom_alias,
      top_alias);
}

// make a simple polymorpher from a single z2 linear morpher with default alias
template <class Morpher>
  requires(poly::is_z2_linear_morpher<Morpher>::value)
MINT_HOST_DEVICE constexpr auto make_z2_polymorpher_default_alias(Morpher) {
  return make_simple_polymorpher_default_alias(
      Morpher{}, Morpher::kBottomLengths, Morpher::kTopLengths);
}

struct reshape_top_z2 {
  template <class Morpher, class Dims>
  MINT_HOST_DEVICE constexpr auto operator()(Morpher morpher, Dims dims) const {
    return reshape_top(morpher, dims);
  }
};

struct reorder_top_z2 {
  template <class Morpher, class Dims>
  MINT_HOST_DEVICE constexpr auto operator()(Morpher morpher, Dims dims) const {
    return reorder_top(morpher, dims);
  }
};

struct swizzle_top_z2 {
  template <class Morpher, index_t kI, index_t kJ>
  MINT_HOST_DEVICE constexpr auto
  operator()(Morpher morpher, index_constant<kI>, index_constant<kJ>) const {
    return swizzle_top(morpher, index_constant<kI>{}, index_constant<kJ>{});
  }
};

struct reshape_bottom_z2 {
  template <class Morpher, class Dims>
  MINT_HOST_DEVICE constexpr auto operator()(Morpher morpher, Dims dims) const {
    return reshape_bottom(morpher, dims);
  }
};

template <class Polymorpher, class TransformOp, class... Args>
MINT_HOST_DEVICE constexpr auto transform_z2_polymorpher(
    const Polymorpher& poly,
    TransformOp&& /*op*/,
    Args&&... args) {
  static_assert(
      Polymorpher::num_morpher() == 1, "Only one z2 morpher per polymorpher");
  auto old_morpher = poly.morphers()[0_ic];
  static_assert(poly::is_z2_linear_morpher<decltype(old_morpher)>::value);
  auto new_morpher = TransformOp{}(old_morpher, std::forward<Args>(args)...);
  return make_z2_polymorpher_default_alias(new_morpher);
}

} // namespace poly
} // namespace mint
