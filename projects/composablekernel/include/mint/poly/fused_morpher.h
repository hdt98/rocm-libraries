#pragma once
#include <mint/core.h>
#include <mint/poly/morpher.h>

namespace mint::poly {

// fused_morpher
//   kDimPairs: nd_array<index_t, num_dim_pair, 2, 2>
//     kDimPairs[pair_id][bottom_or_top][morpher_local_dim_id]
//     bottom_or_top: [0]=bottom_or_top, [1]=top_or_bottom
//     morpher_local_dim_id: [0]=morpher_id, [1]=local_dim_id
template <class Morphers, auto kDimPairs>
struct fused_morpher : morpher<fused_morpher<Morphers, kDimPairs>> {
  using base_type = morpher<fused_morpher<Morphers, kDimPairs>>;
  //
  static constexpr index_t num_morpher_ = Morphers::size();
  static constexpr auto ndim_of_morpher_ =
      elementwise_nd_containers_in<array<index_t, num_morpher_>>(
          static_for_n<num_morpher_>(),
          [](const auto& morpher) { return morpher.all_ndim(); },
          Morphers{});

  //
  static constexpr index_t num_dim_pair = kDimPairs.lengths()[0];

  //
  static constexpr bool is_fundamental_ = false;
  static constexpr bool can_top_down_ = reduce_nd_containers<bool>(
      static_for_n<num_morpher_>(),
      [](bool& acc, const auto& morpher) { acc &= morpher.can_top_down(); },
      true,
      Morphers{});
  static constexpr bool can_bottom_up_ = reduce_nd_containers<bool>(
      static_for_n<num_morpher_>(),
      [](bool& acc, const auto& morpher) { acc &= morpher.can_bottom_up(); },
      true,
      Morphers{});

  MINT_HOST_DEVICE static index_t consteval num_morpher() {
    return num_morpher_;
  }

  // [morpher, local] to [original]
  //     bottom_or_top: [0]=bottom, [1]=top
  using cr_morpher_local_dims =
      compressed_row<ndim_of_morpher_.size(), ndim_of_morpher_>;

  // dim pairs, but using original dim id
  // original_dim_pairs_[pair_id][bottom_or_top]
  //   bottom_or_top: [0]=bottom, [1]=top
  static constexpr auto original_dim_pairs_ = []() {
    nd_array<index_t, num_dim_pair, 2> ret;
    for (index_t i = 0; i < num_dim_pair; i++) {
      ret[i][0] = cr_morpher_local_dims::mn_to_o(
          kDimPairs[i][0][0], kDimPairs[i][0][1]);
      ret[i][1] = cr_morpher_local_dims::mn_to_o(
          kDimPairs[i][1][0], kDimPairs[i][1][1]);
    }
    return ret;
  }();

  // [unique] to [original]
  using cu_unique_dims = compressed_unique<
      cr_morpher_local_dims::num_total(),
      original_dim_pairs_.lengths()[0],
      original_dim_pairs_>;

  // [morpher, local] to [unique]
  MINT_HOST_DEVICE static constexpr index_t morpher_local_to_unique_dim(
      index_t imorpher,
      index_t idim_local) {
    return cu_unique_dims::original_to_unique()[cr_morpher_local_dims::mn_to_o(
        imorpher, idim_local)];
  }

  template <index_t kIMorpher>
  MINT_HOST_DEVICE static consteval auto morpher_unique_dims() {
    constexpr index_t ndim_local = ndim_of_morpher_[kIMorpher];
    array<index_t, ndim_local> ret;
    for (index_t i = 0; i < ndim_local; i++)
      ret[i] = morpher_local_to_unique_dim(kIMorpher, i);
    return ret;
  }

  MINT_HOST_DEVICE static consteval auto num_duplicate_of_unique_dims() {
    return cu_unique_dims::num_duplicates();
  }

  MINT_HOST_DEVICE static constexpr nd_index<2> unique_dim_to_morpher_local(
      index_t i_unique_dim,
      index_t i_duplicate) {
    index_t original_dim =
        cu_unique_dims::unique_to_original(i_unique_dim, i_duplicate);
    return cr_morpher_local_dims::o_to_mn(original_dim);
  }

  static constexpr index_t all_ndim_ = cu_unique_dims::num_unique();

  static constexpr auto is_top_bottom_paired_dim_ = []() {
    struct {
      array<bool, cr_morpher_local_dims::num_total()> is_top;
      array<bool, cr_morpher_local_dims::num_total()> is_bottom;
      array<bool, cr_morpher_local_dims::num_total()> is_paired;
    } original_dims;

    original_dims.is_top.fill(false);
    original_dims.is_bottom.fill(false);
    original_dims.is_paired.fill(false);

    // if is top/bottom dim in component morpher
    static_for_n<num_morpher_>()([&original_dims](auto i) {
      constexpr auto morpher = Morphers{}[i];
      for (index_t j = 0; j < morpher.all_ndim(); j++) {
        const index_t idim = cr_morpher_local_dims::mn_to_o(i, j);
        original_dims.is_top[idim] = morpher.is_top_dim()[j];
        original_dims.is_bottom[idim] = morpher.is_bottom_dim()[j];
        original_dims.is_paired[idim] = morpher.is_paired_dim()[j];
      }
    });

    struct {
      array<bool, cu_unique_dims::num_unique()> is_top;
      array<bool, cu_unique_dims::num_unique()> is_bottom;
      array<bool, cu_unique_dims::num_unique()> is_paired;
    } unique_dims;

    unique_dims.is_top.fill(true);
    unique_dims.is_bottom.fill(true);
    unique_dims.is_paired.fill(false);

    for (index_t i = 0; i < cu_unique_dims::num_unique(); i++) {
      for (index_t j = 0; j < cu_unique_dims::num_duplicates()[i]; j++) {
        const index_t idim = cu_unique_dims::unique_to_original(i, j);
        // an unique dim is top dim, if all original dims are top dims within
        // the component morpher
        unique_dims.is_top[i] &= original_dims.is_top[idim];
        // an unique dim is bottom dim, if all original dims are bottom ims
        // within the component morpher
        unique_dims.is_bottom[i] &= original_dims.is_bottom[idim];
        // an unique dim is paired dim, if any original dim is a paired dim
        // within the component morpher
        unique_dims.is_paired[i] |= original_dims.is_paired[idim];
      }

      // an unique dim is paired dim, if there are more than 1 orignal dims
      unique_dims.is_paired[i] |= (cu_unique_dims::num_duplicates()[i] > 1);
    }

    return array<array<bool, all_ndim_>, 3>{
        unique_dims.is_top, unique_dims.is_bottom, unique_dims.is_paired};
  }();

  static constexpr auto is_top_dim_ = is_top_bottom_paired_dim_[0];
  static constexpr auto is_bottom_dim_ = is_top_bottom_paired_dim_[1];
  static constexpr auto is_paired_dim_ = is_top_bottom_paired_dim_[2];

  static constexpr index_t top_ndim_ = count_true_nd_containers(
      for_n(all_ndim_),
      [](bool v) { return v; },
      is_top_dim_);
  static constexpr index_t bottom_ndim_ = count_true_nd_containers(
      for_n(all_ndim_),
      [](bool v) { return v; },
      is_bottom_dim_);
  static constexpr index_t paired_ndim_ = count_true_nd_containers(
      for_n(all_ndim_),
      [](bool v) { return v; },
      is_paired_dim_);

  static constexpr auto top_dims_ =
      compressed_mask<is_top_dim_.size(), is_top_dim_>::
          compressed_to_original();
  static constexpr auto bottom_dims_ =
      compressed_mask<is_top_dim_.size(), is_bottom_dim_>::
          compressed_to_original();
  static constexpr auto paired_dims_ =
      compressed_mask<is_top_dim_.size(), is_paired_dim_>::
          compressed_to_original();

  static constexpr auto num_bottom_morphers_ = []() {
    nd_index<num_morpher_> ret;
    ret.fill(0);
    static_for_n<num_dim_pair>()([&](auto i) {
      constexpr index_t morpher0 = kDimPairs[i][0][0];
      constexpr index_t morpher1 = kDimPairs[i][1][0];
      constexpr index_t dim0 = kDimPairs[i][0][1];
      constexpr index_t dim1 = kDimPairs[i][1][1];

      constexpr bool is_bot_dim0 =
          Morphers{}.template at<morpher0>().is_bottom_dim()[dim0];
      constexpr bool is_top_dim0 =
          Morphers{}.template at<morpher0>().is_top_dim()[dim0];

      constexpr bool is_bot_dim1 =
          Morphers{}.template at<morpher1>().is_bottom_dim()[dim1];
      constexpr bool is_top_dim1 =
          Morphers{}.template at<morpher1>().is_top_dim()[dim1];

      // If dim0 is both bottom_dim and top_dim, morpher0 index propagation
      // has not effect on dim0, and vise versa, dim0 also has no effect on
      // morpher0 index propagation. Same for dim1 and morpher1
      // so it's ok to ignore this dim_pair when judging morpher_rank
      constexpr bool is_bot_and_top_dim0 = is_bot_dim0 and is_top_dim0;
      constexpr bool is_bot_and_top_dim1 = is_bot_dim1 and is_top_dim1;

      if constexpr (is_bot_and_top_dim0 or is_bot_and_top_dim1) {
        // ignore this dim_pair
      } else if constexpr (is_top_dim0 and is_bot_dim1) {
        // morpher1 is top_morpher
        ret[morpher1]++;
      } else if constexpr (is_top_dim1 and is_bot_dim0) {
        // morpher0 is top_morpher
        ret[morpher0]++;
      } else {
#if 0 // debug
        static_assert(
            dependent_false<Morphers>{},
            "wrong! fail sanity check, dim_pairs not specified correctly");
#endif
      }
    });

    return ret;
  }();

  using cr_bot_morphers_meta =
      compressed_row<num_bottom_morphers_.size(), num_bottom_morphers_>;
  static constexpr auto cr_bottom_morphers_data_ = []() {
    nd_index<cr_bot_morphers_meta::num_total()> ret;
    nd_index<num_morpher_> cnts;
    cnts.fill(0);
    static_for_n<num_dim_pair>()([&](auto i) {
      constexpr index_t morpher0 = kDimPairs[i][0][0];
      constexpr index_t morpher1 = kDimPairs[i][1][0];
      constexpr index_t dim0 = kDimPairs[i][0][1];
      constexpr index_t dim1 = kDimPairs[i][1][1];

      constexpr bool is_bot_dim0 =
          Morphers{}.template at<morpher0>().is_bottom_dim()[dim0];
      constexpr bool is_top_dim0 =
          Morphers{}.template at<morpher0>().is_top_dim()[dim0];

      constexpr bool is_bot_dim1 =
          Morphers{}.template at<morpher1>().is_bottom_dim()[dim1];
      constexpr bool is_top_dim1 =
          Morphers{}.template at<morpher1>().is_top_dim()[dim1];

      // If dim0 is both bottom_dim and top_dim, morpher0 index propagation
      // has not effect on dim0, and vise versa, dim0 also has no effect on
      // morpher0 index propagation. Same for dim1 and morpher1
      // so it's ok to ignore this dim_pair when judging morpher_rank
      constexpr bool is_bot_and_top_dim0 = is_bot_dim0 and is_top_dim0;
      constexpr bool is_bot_and_top_dim1 = is_bot_dim1 and is_top_dim1;

      if constexpr (is_bot_and_top_dim0 or is_bot_and_top_dim1) {
        // ignore this dim_pair
      } else if constexpr (is_top_dim0 and is_bot_dim1) {
        // morpher1 is top_morpher
        index_t bot_morpher = morpher0;
        index_t top_morpher = morpher1;
        ret[cr_bot_morphers_meta::mn_to_o(top_morpher, cnts[top_morpher])] =
            bot_morpher;
        cnts[top_morpher]++;
      } else if constexpr (is_top_dim1 and is_bot_dim0) {
        // morpher0 is top_morpher
        index_t bot_morpher = morpher1;
        index_t top_morpher = morpher0;
        ret[cr_bot_morphers_meta::mn_to_o(top_morpher, cnts[top_morpher])] =
            bot_morpher;
        cnts[top_morpher]++;
      } else {
#if 0 // debug
        static_assert(
            dependent_false<Morphers>{},
            "wrong! fail sanity check, dim_pairs not specified correctly");
#endif
      }
    });
    return ret;
  }();

  MINT_HOST_DEVICE static consteval auto top_down_sorted_morpher_ids() {
    constexpr auto graph = fixed_capacity_csr_graph<
        cr_bot_morphers_meta::num_m(),
        cr_bot_morphers_meta::num_total()>{
        cr_bot_morphers_meta::num_m(),
        cr_bot_morphers_meta::num_total(),
        cr_bottom_morphers_data_,
        cr_bot_morphers_meta::m_to_o_begins()};
    static_assert(is_acyclic_graph(graph), "morphers have circular dependency");
    return sort_graph(graph);
  }

  MINT_HOST_DEVICE static consteval auto bottom_up_sorted_morpher_ids() {
    nd_index<num_morpher_> ret;
    for (index_t i = 0; i < num_morpher_; i++)
      ret[i] = top_down_sorted_morpher_ids()[num_morpher_ - i - 1];
    return ret;
  }

  static constexpr auto is_linear_top_down_ = []() {
    nd_array<bool, top_ndim_, bottom_ndim_> ret;

    // loop over top_dims
    for (index_t i = 0; i < top_ndim_; i++) {
      nd_index<all_ndim_> flags;
      flags.fill(0);
      flags[top_dims_[i]] = 1;

      static_for_n<num_morpher_>()([&](auto j) {
        constexpr index_t imorpher = top_down_sorted_morpher_ids()[j];
        constexpr auto dims = morpher_unique_dims<imorpher>();
        auto local_flags = flags.template get_subset<dims.size(), dims>();
        using Morpher =
            remove_cvref_t<decltype(Morphers{}.template at<imorpher>())>;
        for (index_t k = 0; k < Morpher::top_ndim(); k++) {
          index_t local_top_dim = Morpher::top_dims()[k];
          for (index_t ks = 0; ks < Morpher::bottom_ndim(); ks++) {
            index_t local_bot_dim = Morpher::bottom_dims()[ks];
            if (local_flags[local_top_dim] == 1) {
              if (local_flags[local_bot_dim] != -1) {
                local_flags[local_bot_dim] =
                    Morpher::is_linear_top_down()[k][ks] ? 1 : -1;
              }
            } else if (local_flags[local_top_dim] == -1) {
              local_flags[local_bot_dim] = -1;
            }
          }
        }
        flags.template set_subset<dims.size(), dims>(local_flags);
      });

      //
      for (index_t j = 0; j < bottom_ndim_; j++)
        ret[i][j] = (flags[bottom_dims_[j]] == -1) ? false : true;
    }

    return ret;
  }();

  static constexpr auto is_linear_bottom_up_ = []() {
    nd_array<bool, bottom_ndim_, top_ndim_> ret;

    // loop over bottom_dims
    for (index_t i = 0; i < bottom_ndim_; i++) {
      nd_index<all_ndim_> flags;
      flags.fill(0);
      flags[bottom_dims_[i]] = 1;

      static_for_n<num_morpher_>()([&](auto j) {
        constexpr index_t imorpher = bottom_up_sorted_morpher_ids()[j];
        constexpr auto dims = morpher_unique_dims<imorpher>();
        auto local_flags = flags.template get_subset<dims.size(), dims>();
        using Morpher =
            remove_cvref_t<decltype(Morphers{}.template at<imorpher>())>;
        for (index_t k = 0; k < Morpher::bottom_ndim(); k++) {
          index_t local_bot_dim = Morpher::bottom_dims()[k];
          for (index_t ks = 0; ks < Morpher::top_ndim(); ks++) {
            index_t local_top_dim = Morpher::top_dims()[ks];
            if (local_flags[local_bot_dim] == 1) {
              if (local_flags[local_top_dim] != -1) {
                local_flags[local_top_dim] =
                    Morpher::is_linear_bottom_up()[k][ks] ? 1 : -1;
              }
            } else if (local_flags[local_bot_dim] == -1) {
              local_flags[local_top_dim] = -1;
            }
          }
        }
        flags.template set_subset<dims.size(), dims>(local_flags);
      });

      //
      for (index_t j = 0; j < top_ndim_; j++)
        ret[i][j] = (flags[top_dims_[j]] == -1) ? false : true;
    }

    return ret;
  }();

  constexpr fused_morpher() = default;

  MINT_HOST_DEVICE constexpr fused_morpher(const Morphers& morphers)
      : morphers_{morphers} {}

  constexpr bool operator==(const fused_morpher&) const = default;

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_top_down(Index& idx) const {
    static_for_n<num_morpher_>()([this, &idx](auto i) {
      constexpr index_t imorpher = top_down_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims.size(), dims>();
      this->morphers_.template at<imorpher>().propagate_index_top_down(
          local_idx);
      idx.template set_subset<dims.size(), dims>(local_idx);
    });
  }

  template <class Index>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_bottom_up(Index& idx) const {
    static_for_n<num_morpher_>()([this, &idx](auto i) {
      constexpr index_t imorpher = bottom_up_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims.size(), dims>();
      this->morphers_.template at<imorpher>().propagate_index_bottom_up(
          local_idx);
      idx.template set_subset<dims.size(), dims>(local_idx);
    });
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_top_down(
      Index& idx,
      IndexDelta& idx_delta) const {
    static_for_n<num_morpher_>()([this, &idx, &idx_delta](auto i) {
      constexpr index_t imorpher = top_down_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims.size(), dims>();
      auto local_idx_delta = idx_delta.template get_subset<dims.size(), dims>();
      this->morphers_.template at<imorpher>()
          .propagate_index_and_delta_top_down(local_idx, local_idx_delta);
      idx.template set_subset<dims.size(), dims>(local_idx);
      idx_delta.template set_subset<dims.size(), dims>(local_idx_delta);
    });
  }

  template <class Index, class IndexDelta>
    requires(
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void propagate_index_and_delta_bottom_up(
      Index& idx,
      IndexDelta& idx_delta) const {
    static_for_n<num_morpher_>()([this, &idx, &idx_delta](auto i) {
      constexpr index_t imorpher = bottom_up_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims.size(), dims>();
      auto local_idx_delta = idx_delta.template get_subset<dims>();
      this->morphers_.template at<imorpher>()
          .propagate_index_and_delta_bottom_up(local_idx, local_idx_delta);
      idx.template set_subset<dims.size(), dims>(local_idx);
    });
  }

  // override method in base_type morpher
  // propagate idx and idx_delta with dim freezing
  // freezed elements in idx will not be updated
  // freezed elements in idx_delta will be set to 0
  template <auto kIsFreezedDim, class Index, class IndexDelta>
    requires(
        is_same_v<typename decltype(kIsFreezedDim)::value_type, bool> &&
        kIsFreezedDim.size() == all_ndim_ &&
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void
  propagate_index_and_delta_top_down_freezed_dim(
      Index& idx,
      IndexDelta& idx_delta) const {
    static_for_n<num_morpher_>()([this, &idx, &idx_delta](auto i) {
      constexpr index_t imorpher = top_down_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims>();
      auto local_idx_delta = idx_delta.template get_subset<dims>();
      this->morphers_.template at<imorpher>()
          .template propagate_index_and_delta_top_down_freezed_dim<
              kIsFreezedDim.template get_subset<dims>()>(
              local_idx, local_idx_delta);
      idx.template set_subset<dims>(local_idx);
      idx_delta.template set_subset<dims>(local_idx_delta);
    });
  }

  // override method in base_type morpher
  // propagate idx and idx_delta with dim freezing, and with additional
  // conjecture.
  template <auto kIsFreezedDim, class Index, class IndexDelta>
    requires(
        is_same_v<typename decltype(kIsFreezedDim)::value_type, bool> &&
        kIsFreezedDim.size() == all_ndim_ &&
        is_same_v<typename Index::value_type, index_t> &&
        Index::size() == all_ndim_ &&
        is_same_v<typename IndexDelta::value_type, index_t> &&
        IndexDelta::size() == all_ndim_)
  MINT_HOST_DEVICE constexpr void
  propagate_index_and_delta_top_down_freezed_dim_conjectural(
      Index& idx,
      IndexDelta& idx_delta) const {
    static_for_n<num_morpher_>()([this, &idx, &idx_delta](auto i) {
      constexpr index_t imorpher = top_down_sorted_morpher_ids()[i];
      constexpr auto dims = morpher_unique_dims<imorpher>();
      auto local_idx = idx.template get_subset<dims.size(), dims>();
      auto local_idx_delta = idx_delta.template get_subset<dims.size(), dims>();
      this->morphers_.template at<imorpher>()
          .template propagate_index_and_delta_top_down_freezed_dim_conjectural<
              kIsFreezedDim.template get_subset<dims.size(), dims>()>(
              local_idx, local_idx_delta);
      idx.template set_subset<dims.size(), dims>(local_idx);
      idx_delta.template set_subset<dims.size(), dims>(local_idx_delta);
    });
  }

  MINT_HOST_DEVICE constexpr const auto& morphers() const {
    return morphers_;
  }

  MINT_HOST_DEVICE static consteval auto dim_pairs() {
    return kDimPairs;
  }

  MINT_HOST_DEVICE void print() const {
    printf("fused_morpher: {");
    base_type::print();
    printf(
        ", ndim: (all, top, bot, pair): (%d, %d, %d, %d), ",
        all_ndim_,
        top_ndim_,
        bottom_ndim_,
        paired_ndim_);
    printf("morphers_: {");
    morphers_.print();
    printf("}, ");
    printf("dim_pairs(): {");
    dim_pairs().print();
    printf("}, ");
    printf("cu_unique_dims::original_to_unique(): {");
    cu_unique_dims::original_to_unique().print();
    printf("}, ");
    printf("bottom_up_sorted_morpher_ids(): {");
    bottom_up_sorted_morpher_ids().print();
    printf("}");
    printf("}");
  }

  const Morphers morphers_;
};

//   kDimPairs: nd_array<index_t, num_dim_pair, 2, 2>
//     kDimPairs[pair_id][bottom_or_top][morpher_local_dim_id]
//     bottom_or_top: [0]=bottom_or_top, [1]=top_or_bottom
//     morpher_local_dim_id: [0]=morpher_id, [1]=local_dim_id
template <auto kDimPairs, class... Morphers>
MINT_HOST_DEVICE constexpr auto make_fused_morpher(
    const Morphers&... morphers_in) {
  const auto morphers = tuple<Morphers...>{morphers_in...};
  return fused_morpher<remove_cvref_t<decltype(morphers)>, kDimPairs>{morphers};
}

} // namespace mint::poly
