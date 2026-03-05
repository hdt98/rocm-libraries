#pragma once
#include <mint/core.h>
#include <mint/poly.h>
#include <mint/tensor.h>
#include <mint/tile/generic/reduce_no_shuffle.h>

namespace mint {
namespace tile {
namespace simt {
namespace warp {

// warp partial reduction: cross-lane only reduce
template <
    class OutTensor,
    class InDstr,
    class DimAlias,
    DimAlias... kReduceDimAliases,
    class FReduce>
MINT_HOST_DEVICE void reduce_sync(
    OutTensor& out,
    InDstr,
    sequence<DimAlias, kReduceDimAliases...>,
    const FReduce& f_reduce) {
  constexpr auto reduce_dim_aliases =
      array<DimAlias, sizeof...(kReduceDimAliases)>{kReduceDimAliases...};

  constexpr auto reduce_mid_dims_cnt = [&]() {
    constexpr index_t max_size = InDstr::tensor_desc().polymorpher().all_ndim();
    nd_index<max_size> ret;
    ret.fill(-1);
    index_t cnt = 0;
    static_for_n<reduce_dim_aliases.size()>()([&](auto i) {
      constexpr auto dim_alias = reduce_dim_aliases[i];
      // assume alias of morpher of tile_dim is the same as tile_dim itself
      constexpr auto morpher_alias = dim_alias;
      constexpr index_t imorpher =
          InDstr::tensor_desc().polymorpher().alias_to_morpher()[morpher_alias];
      constexpr auto morpher = InDstr::tensor_desc()
                                   .polymorpher()
                                   .morphers()
                                   .template at<imorpher>();
      char tmp[MINT_ALIAS_MAX_STRING_SIZE];
      for (index_t j = 0; j < morpher.bottom_ndim(); j++) {
        integer_to_string(j, tmp);
        auto alias_tmp = dim_alias;
        auto mid_dim_alias = alias_tmp.append("_").append(tmp);
        if (OutTensor::dstr_tensor_desc()
                .tensor_desc()
                .polymorpher()
                .alias_to_dim()
                .contains(mid_dim_alias))
          ret[cnt++] = OutTensor::dstr_tensor_desc()
                           .tensor_desc()
                           .polymorpher()
                           .alias_to_dim()[mid_dim_alias];
      }
    });
    return ::mint::make_tuple(ret, cnt);
  }();

  constexpr auto reduce_mid_dims_tmp = reduce_mid_dims_cnt[0_ic];
  constexpr index_t num_reduce_mid_dim = reduce_mid_dims_cnt[1_ic];

  constexpr auto reduce_mid_dims = [&]() {
    nd_index<num_reduce_mid_dim> ret;
    std::copy(
        reduce_mid_dims_tmp.begin(),
        reduce_mid_dims_tmp.begin() + num_reduce_mid_dim,
        ret.begin());
    return ret;
  }();

  constexpr auto reduce_mid_lengths =
      OutTensor::dstr_tensor_desc()
          .tensor_desc()
          .all_lengths()
          .template get_subset<reduce_mid_dims.size(), reduce_mid_dims>();

  constexpr auto elem_lengths = OutTensor::dstr_tensor_desc().element_lengths();

  // TODO: OK to hardcode lane_dim_alias as "Lane"?
  constexpr index_t lane_dim = OutTensor::dstr_tensor_desc()
                                   .tensor_desc()
                                   .polymorpher()
                                   .alias_to_dim()["Lane"];

  // TODO: OK to hardcode lane_morpher_alias as "Lane"?
  constexpr index_t lane_morpher_id = OutTensor::dstr_tensor_desc()
                                          .tensor_desc()
                                          .polymorpher()
                                          .alias_to_morpher()["Lane"];

  constexpr auto lane_morpher = OutTensor::dstr_tensor_desc()
                                    .tensor_desc()
                                    .polymorpher()
                                    .morphers()
                                    .template at<lane_morpher_id>();

  constexpr auto find_local_dim = [&](index_t dim) {
    index_t ret = -1;
    for (index_t i = 0; i < OutTensor::dstr_tensor_desc()
                                .tensor_desc()
                                .polymorpher()
                                .num_duplicate_of_unique_dims()[dim];
         i++) {
      auto morpher_local = OutTensor::dstr_tensor_desc()
                               .tensor_desc()
                               .polymorpher()
                               .unique_dim_to_morpher_local(dim, i);
      if (morpher_local[0] == lane_morpher_id)
        ret = morpher_local[1];
    }
    return ret;
  };

  constexpr index_t lane_dim_local = find_local_dim(lane_dim);

  static_assert(lane_dim_local >= 0);

  // FIXME: pass partition info, and remove hardcode for partition Id,
  /*const index_t lid = threadIdx.x % MINT_WARP_SIZE;*/

  // cross-lane reduction
  static_for_n<reduce_mid_lengths.size()>()([&](auto i) {
    constexpr index_t reduce_mid_dim = reduce_mid_dims[i];

#if 0
    // TODO: enable this after is_linear_top_down() support all_dims-to-all_dims
    // instead of top_dims-to-bottom_dims
    static_assert(OutTensor::dstr_tensor_desc()
                      .tensor_desc()
                      .polymorpher()
                      .is_linear_top_down()[lane_dim][reduce_mid_dim]);
#endif

    constexpr index_t reduce_mid_dim_local = find_local_dim(reduce_mid_dim);

    static_assert(reduce_mid_dim_local >= 0);

#if 0
    // nvcc cannot compile
    constexpr index_t lid_over_r_derivative = [&]() {
      nd_index<lane_morpher.all_ndim()> idx_old, idx_new;
      idx_old.fill(0);
      idx_new.fill(0);
      idx_new[reduce_mid_dim_local] = 1;
      lane_morpher.propagate_index_top_down(idx_old);
      lane_morpher.propagate_index_top_down(idx_new);
      return idx_new[lane_dim_local] - idx_old[lane_dim_local];
    }();
#else
    // nvcc can compile
    auto f_lid_over_r_derivative = []<auto kLaneMorpher,
                                      index_t kReduceMidDimLocal,
                                      index_t kLaneDimLocal>() {
      nd_index<kLaneMorpher.all_ndim()> idx_old, idx_new;
      idx_old.fill(0);
      idx_new.fill(0);
      idx_new[kReduceMidDimLocal] = 1;
      kLaneMorpher.propagate_index_top_down(idx_old);
      kLaneMorpher.propagate_index_top_down(idx_new);
      return idx_new[kLaneDimLocal] - idx_old[kLaneDimLocal];
    };

    constexpr index_t lid_over_r_derivative =
        f_lid_over_r_derivative.template
        operator()<lane_morpher, reduce_mid_dim_local, lane_dim_local>();
#endif

    // FIXME : amolak : T221905119, the min() is somewhat of a hack for
    // same-warp different-lane reductions when warps share dimensions that
    // *aren't* reduced on. Investigate for more thorough fix.
    constexpr index_t reduce_length =
        std::min(reduce_mid_lengths[i], MINT_WARP_SIZE);
    constexpr index_t nstage = math::integer_log2_ceiling(reduce_length);

    static_assert(
        math::is_power_of_2_integer(reduce_length) &&
            math::is_power_of_2_integer(lid_over_r_derivative),
        "wrong! not support power of 2 reduction length");

    static_for_nd3<elem_lengths>()([&](auto e_idx_seq) {
      // reduction sweep
      static_for_n<nstage>()([&](auto istage) {
        constexpr auto e_idx = to_array(e_idx_seq);

        // pull data from remote lane
        constexpr index_t lane_mask = 1
            << (math::integer_log2_floor(lid_over_r_derivative) + istage);
#if defined(MINT_BACKEND_CUDA)
        // FIXME: using 0xFFFFFFFF as mask assume warp size is 32
        static_assert(MINT_WARP_SIZE == 32, "current shuffle mask is 32 wide");
        const auto v_remote = __shfl_xor_sync(
            0xFFFFFFFF, out.template element<e_idx>(), lane_mask);
#elif defined(MINT_BACKEND_ROCM)
        static_assert(MINT_WARP_SIZE == 64, "current shuffle mask is 64 wide");
        const auto v_remote =
            __shfl_xor(out.template element<e_idx>(), lane_mask);
#else
        static_assert(false, "wrong! no implementation for this backend");
#endif

        // reduce
        out.template element<e_idx>() =
            f_reduce(out.template element<e_idx>(), v_remote);
      });
    });
  });
}

// warp full reduce: in-lane and cross-lane
template <
    class OutTensor,
    class InTensor,
    class DimAlias,
    DimAlias... kReduceDimAliases,
    class FReduce,
    bool kReduceWithOut>
MINT_HOST_DEVICE void reduce(
    OutTensor& out,
    const InTensor& in,
    sequence<DimAlias, kReduceDimAliases...> reduce_dim_aliases,
    const FReduce& f_reduce,
    bool_constant<kReduceWithOut> reduce_with_out) {
  // in-lane reduction
  ::mint::tile::generic::reduce_no_shuffle(
      out, in, reduce_dim_aliases, f_reduce, reduce_with_out);

  // cross-lane reduction
  ::mint::tile::simt::warp::reduce_sync(
      out, InTensor::dstr_tensor_desc(), reduce_dim_aliases, f_reduce);
}

} // namespace warp
} // namespace simt
} // namespace tile
} // namespace mint
