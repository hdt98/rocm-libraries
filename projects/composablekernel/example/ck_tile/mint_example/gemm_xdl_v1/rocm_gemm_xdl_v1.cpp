//#include <gtest/gtest.h>
#include <mint/mint.h>
#include <is_equal.hpp>
#include <random>

#include <iostream>


using namespace mint;
using namespace mint::tensor;
using namespace mint::tile::rocm;
using namespace mint_test;

void hip_check_error(hipError_t x) {
  if (x != hipSuccess) {
    std::ostringstream ss;
    ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__
       << ": " << __LINE__ << "in function: " << __func__;
    throw std::runtime_error(ss.str());
  }
}

__device__ void block_sync_lds() {
  __builtin_amdgcn_s_waitcnt(0xc07f);
  __builtin_amdgcn_s_barrier();
}

__device__ std::size_t integer_divide_ceil(std::size_t x, std::size_t y) {
  return (x + y - std::size_t{1}) / y;
}

namespace mint {

template <index_t MPerBlock, index_t NPerBlock>
__device__ auto GetBlockIdx(index_t block_1d_id, index_t M, index_t N)
    -> const tuple<index_t, index_t> {
  const auto M0 = integer_divide_ceil(M, MPerBlock);
  const auto N0 = integer_divide_ceil(N, NPerBlock);

  constexpr index_t GroupNum = 8;
  constexpr index_t M01 = 4;

  if (M0 == 1) {
    return make_tuple(0, block_1d_id);
  } else if (N0 == 1) {
    return make_tuple(block_1d_id, 0);
  }
  // block_1d_id = block_1d_id % (M0 * N0); // swallow batch index
  else {
    const auto group_size = integer_divide_ceil(M0 * N0, GroupNum);
    const auto big_group_num = GroupNum - (group_size * GroupNum - M0 * N0);
    const auto group_id_y = block_1d_id / GroupNum;
    const auto group_id_x = block_1d_id - group_id_y * GroupNum;
    const auto remap_block_1d_id = group_id_x <= big_group_num
        ? group_id_x * group_size + group_id_y
        : group_id_x * group_size + big_group_num - group_id_x + group_id_y;

    const index_t idx_M0 = remap_block_1d_id / N0;
    const index_t idx_N0 = remap_block_1d_id - idx_M0 * N0;

    const index_t M0_tmp = M0 / M01;
    const index_t M0_mod_M01 = M0 - M0_tmp * M01;

    const auto M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;

    const index_t idx_M00 = idx_M0 / M01;
    const index_t idx_M01 = idx_M0 - idx_M00 * M01;
    const index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

    /**
     *                        idxN0
     *
     *           |<               mtx   N                 >|
     *
     *             NPerBlock   NPerBlock   NPerBlock   NPerBlock
     *                N_0         N_1        N_2         N_3
     *       -   |-----------|-----------|-----------|-----|-----|-
     *       ^   | -   -  0  |/---->  2  |           |     |     |
     *           | |   |     /     |     |           |     |     |  M_0
     * MPerBlock | M   |    /|     |     |           |     |     |
     *           |-0---|---/-|-----|-----|-----------|-----|-----|-
     *           | 1   |  /  |     |     |  blockid  |     |     |
     * idxM0     | |   | /   |     V     |     5     |     |     |  M_1
     * MPerBlock | -   V   1 |     -  3  |           |     |     |
     *           |-----------|-----------|-----------|-----|-----|-
     *    mtx M  |           |           |           |     |     |
     *           |           |           |           |     |     |  M_2
     * MPerBlock |           |           |           |     |     |
     *           |-----------|-----------|-----------|-----|-----|-
     *           |           |           |           |     |     |
     *           |           |           |           |     |     |  M_3
     * MPerBlock |           |           |           |     |     |
     *           |-----------|-----------|-----------|-----|-----|-
     *       V   |           |           |           |     |     |
     *       -   |-----------|-----------|-----------|-----|-----|- M_4
     * MPerBlock |           |           |           |     |     |
     *           |-----------|-----------|-----------|-----|-----|-
     *  Example:
     *   assume:
     *      M0 = 5
     *      N0 = 4
     *      block_1d_id = 5
     *      M01 = 2
     *
     *   idx_N0 = 1
     *   idx_M0 = 1
     *   M01_adapt = 2
     *   idx_M00 = 0
     *   idx_M01 = 1
     *   idx_N0_M01_local = 5
     *   output {1, 2}
     */

    const index_t N_out = idx_N0_M01_local / M01_adapt;
    const index_t idx_loc_mod_M01 = idx_N0_M01_local - N_out * M01_adapt;

    return make_tuple(idx_loc_mod_M01 + idx_M00 * M01, N_out);
  }
}

__device__ index_t thread_id() {
  return threadIdx.x;
}

__device__ index_t lane_id() {
  return threadIdx.x % MINT_WARP_SIZE;
}

__device__ index_t warp_id() {
  return __builtin_amdgcn_readfirstlane(threadIdx.x / MINT_WARP_SIZE);
}

__device__ index_t block_id() {
  return __builtin_amdgcn_readfirstlane(blockIdx.x);
}

// C[M, N] = A[M, K] * B[K, N]
template <class T, class TAcc = float>
void cpu_gemm(
    T* p_c,
    const T* p_a,
    const T* p_b,
    index_t m_size,
    index_t n_size,
    index_t k_size) {
#pragma omp parallel for
  for (index_t m = 0; m < m_size; m++) {
    for (index_t n = 0; n < n_size; n++) {
      TAcc acc = 0;
      for (index_t k = 0; k < k_size; k++)
        acc += static_cast<float>(p_a[m * k_size + k]) *
            static_cast<float>(p_b[n * k_size + k]);
      p_c[m * n_size + n] = static_cast<T>(acc);
    }
  }
}

// block distribution for matmul
template <
    index_t BlockSize,
    index_t kMPerBlock,
    index_t kKPerBlock,
    index_t kKPack>
__device__ constexpr auto make_block_matmul_a_m_k_distribution() {
  constexpr auto K0 = kKPerBlock / kKPack;
  constexpr auto tid_k = K0;
  constexpr auto tid_m = BlockSize / tid_k;
  constexpr auto M1 = tid_m;
  constexpr auto M0 = kMPerBlock / M1;

  constexpr auto p_merge = poly::merge<nd_index<2>>{{tid_m, tid_k}};
  constexpr auto m_split = poly::split<nd_index<2>>{{M0, M1}};
  constexpr auto k_split = poly::split<nd_index<2>>{{K0, kKPack}};
  constexpr auto dim_pairs =
      nd_array<index_t, 2, 2, 2>{{{0, 1}, {1, 1}}, {{0, 2}, {2, 0}}};
  constexpr auto morphers = mint::make_tuple(p_merge, m_split, k_split);
  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 3> ret;
    ret["P"] = 0;
    ret["M"] = 1;
    ret["K"] = 2;
    return ret;
  }();
  constexpr auto alias_to_dim = []() {
    static_map<alias_t, nd_index<2>, 7> ret;
    ret["P"] = {0, 0};
    ret["M"] = {1, 2};
    ret["M_0"] = {1, 0};
    ret["M_1"] = {1, 1};
    ret["K"] = {2, 2};
    ret["K_0"] = {2, 0};
    ret["K_1"] = {2, 1};
    return ret;
  }();

  constexpr auto lengths =
      nd_index<7>{BlockSize, kMPerBlock, M0, M1, kKPerBlock, K0, kKPack};

  constexpr auto poly =
      poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);

  constexpr auto top_dim_aliases = array<alias_t, 2>{"M", "K"};
  constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
  constexpr auto element_dim_aliases = array<alias_t, 2>{"M_0", "K_1"};

  return distributed_tensor_descriptor<
      poly,
      top_dim_aliases,
      partition_dim_aliases,
      element_dim_aliases>{};
}

// block distribution for matmul
template <
    index_t BlockSize,
    index_t kNPerBlock,
    index_t kKPerBlock,
    index_t kKPack>
__device__ constexpr auto make_block_matmul_b_n_k_distribution() {
  constexpr auto K0 = kKPerBlock / kKPack;
  constexpr auto tid_k = K0;
  constexpr auto tid_n = BlockSize / tid_k;
  constexpr auto N1 = tid_n;
  constexpr auto N0 = kNPerBlock / N1;

  constexpr auto p_merge = poly::merge<nd_index<2>>{{tid_n, tid_k}};
  constexpr auto m_split = poly::split<nd_index<2>>{{N0, N1}};
  constexpr auto k_split = poly::split<nd_index<2>>{{K0, kKPack}};
  constexpr auto dim_pairs =
      nd_array<index_t, 2, 2, 2>{{{0, 1}, {1, 1}}, {{0, 2}, {2, 0}}};
  constexpr auto morphers = mint::make_tuple(p_merge, m_split, k_split);
  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 3> ret;
    ret["P"] = 0;
    ret["N"] = 1;
    ret["K"] = 2;
    return ret;
  }();
  constexpr auto alias_to_dim = []() {
    static_map<alias_t, nd_index<2>, 7> ret;
    ret["P"] = {0, 0};
    ret["N"] = {1, 2};
    ret["N_0"] = {1, 0};
    ret["N_1"] = {1, 1};
    ret["K"] = {2, 2};
    ret["K_0"] = {2, 0};
    ret["K_1"] = {2, 1};
    return ret;
  }();

  constexpr auto lengths =
      nd_index<7>{BlockSize, kNPerBlock, N0, N1, kKPerBlock, K0, kKPack};

  constexpr auto poly =
      poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);

  constexpr auto top_dim_aliases = array<alias_t, 2>{"N", "K"};
  constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
  constexpr auto element_dim_aliases = array<alias_t, 2>{"N_0", "K_1"};

  return distributed_tensor_descriptor<
      poly,
      top_dim_aliases,
      partition_dim_aliases,
      element_dim_aliases>{};
}

// warp distribution for matmul
template <
    index_t kMPerWarp,
    index_t kKPerWarp,
    index_t kKPack,
    index_t kMRepeat = 1,
    index_t kKRepeat = 1>
__device__ constexpr auto make_warp_matmul_a_m_k_distribution() {
  constexpr auto num_blk = MINT_WARP_SIZE / kMPerWarp;

  static_assert(num_blk * kKPack == kKPerWarp);

  constexpr auto p_merge = poly::merge<nd_index<2>>{{num_blk, kMPerWarp}};
  constexpr auto m_split = poly::split<nd_index<2>>{{kMRepeat, kMPerWarp}};
  constexpr auto k_split =
      poly::split<nd_index<3>>{{kKRepeat, num_blk, kKPack}};
  constexpr auto dim_pairs =
      nd_array<index_t, 2, 2, 2>{{{0, 1}, {2, 1}}, {{0, 2}, {1, 1}}};
  constexpr auto morphers = mint::make_tuple(p_merge, m_split, k_split);
  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 3> ret;
    ret["P"] = 0;
    ret["M"] = 1;
    ret["K"] = 2;
    return ret;
  }();
  constexpr auto alias_to_dim = []() {
    static_map<alias_t, nd_index<2>, 8> ret;
    ret["P"] = {0, 0};
    ret["M"] = {1, 2};
    ret["M_0"] = {1, 0};
    ret["M_1"] = {1, 1};
    ret["K"] = {2, 3};
    ret["K_0"] = {2, 0};
    ret["K_1"] = {2, 1};
    ret["K_2"] = {2, 2};
    return ret;
  }();

  constexpr auto lengths = nd_index<8>{
      MINT_WARP_SIZE,
      kMPerWarp * kMRepeat,
      kMRepeat,
      kMPerWarp,
      kKPerWarp * kKRepeat,
      kKRepeat,
      num_blk,
      kKPack};

  constexpr auto poly =
      poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);

  constexpr auto top_dim_aliases = array<alias_t, 2>{"M", "K"};
  constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
  constexpr auto element_dim_aliases = array<alias_t, 3>{"K_0", "M_0", "K_2"};

  return distributed_tensor_descriptor<
      poly,
      top_dim_aliases,
      partition_dim_aliases,
      element_dim_aliases>{};
}

// warp distribution for matmul
template <
    index_t kNPerWarp,
    index_t kKPerWarp,
    index_t kKPack,
    index_t kNRepeat = 1,
    index_t kKRepeat = 1>
__device__ constexpr auto make_warp_matmul_b_n_k_distribution() {
  constexpr auto num_blk = MINT_WARP_SIZE / kNPerWarp;

  static_assert(num_blk * kKPack == kKPerWarp);

  constexpr auto p_merge = poly::merge<nd_index<2>>{{num_blk, kNPerWarp}};
  constexpr auto n_split = poly::split<nd_index<2>>{{kNRepeat, kNPerWarp}};
  constexpr auto k_split =
      poly::split<nd_index<3>>{{kKRepeat, num_blk, kKPack}};
  constexpr auto dim_pairs =
      nd_array<index_t, 2, 2, 2>{{{0, 1}, {2, 1}}, {{0, 2}, {1, 1}}};
  constexpr auto morphers = mint::make_tuple(p_merge, n_split, k_split);
  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 3> ret;
    ret["P"] = 0;
    ret["N"] = 1;
    ret["K"] = 2;
    return ret;
  }();
  constexpr auto alias_to_dim = []() {
    static_map<alias_t, nd_index<2>, 8> ret;
    ret["P"] = {0, 0};
    ret["N"] = {1, 2};
    ret["N_0"] = {1, 0};
    ret["N_1"] = {1, 1};
    ret["K"] = {2, 3};
    ret["K_0"] = {2, 0};
    ret["K_1"] = {2, 1};
    ret["K_2"] = {2, 2};
    return ret;
  }();

  constexpr auto lengths = nd_index<8>{
      MINT_WARP_SIZE,
      kNPerWarp * kNRepeat,
      kNRepeat,
      kNPerWarp,
      kKPerWarp * kKRepeat,
      kKRepeat,
      num_blk,
      kKPack};

  constexpr auto poly =
      poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);

  constexpr auto top_dim_aliases = array<alias_t, 2>{"N", "K"};
  constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
  constexpr auto element_dim_aliases = array<alias_t, 3>{"K_0", "N_0", "K_2"};

  return distributed_tensor_descriptor<
      poly,
      top_dim_aliases,
      partition_dim_aliases,
      element_dim_aliases>{};
}

// warp distribution for matmul
template <
    index_t kMPerWarp,
    index_t kNPerWarp,
    index_t kMRepeat = 1,
    index_t kNRepeat = 1>
__device__ constexpr auto make_warp_matmul_c_m_n_distribution() {
  constexpr auto CPerThread = kMPerWarp * kNPerWarp / MINT_WARP_SIZE;
  constexpr auto CGroups = CPerThread / 4;

  constexpr auto num_blk = MINT_WARP_SIZE / kNPerWarp;

  constexpr auto p_merge = poly::merge<nd_index<2>>{{num_blk, kNPerWarp}};
  constexpr auto m_split =
      poly::split<nd_index<4>>{{kMRepeat, CGroups, num_blk, 4}};
  constexpr auto n_split = poly::split<nd_index<2>>{{kNRepeat, kNPerWarp}};
  constexpr auto dim_pairs =
      nd_array<index_t, 2, 2, 2>{{{0, 1}, {1, 2}}, {{0, 2}, {2, 1}}};
  constexpr auto morphers = mint::make_tuple(p_merge, m_split, n_split);
  constexpr auto alias_to_morpher = []() {
    static_map<alias_t, index_t, 3> ret;
    ret["P"] = 0;
    ret["M"] = 1;
    ret["N"] = 2;
    return ret;
  }();
  constexpr auto alias_to_dim = []() {
    static_map<alias_t, nd_index<2>, 9> ret;
    ret["P"] = {0, 0};
    ret["M"] = {1, 4};
    ret["M_0"] = {1, 0};
    ret["M_1"] = {1, 1};
    ret["M_2"] = {1, 2};
    ret["M_3"] = {1, 3};
    ret["N"] = {2, 2};
    ret["N_0"] = {2, 0};
    ret["N_1"] = {2, 1};
    return ret;
  }();

  constexpr auto lengths = nd_index<9>{
      MINT_WARP_SIZE,
      kMPerWarp * kMRepeat,
      kMRepeat,
      CGroups,
      num_blk,
      4,
      kNPerWarp * kNRepeat,
      kNRepeat,
      kNPerWarp};

  constexpr auto poly =
      poly::make_polymorpher<dim_pairs, alias_to_morpher, alias_to_dim>(
          morphers, lengths);

  constexpr auto top_dim_aliases = array<alias_t, 2>{"M", "N"};
  constexpr auto partition_dim_aliases = array<alias_t, 1>{"P"};
  constexpr auto element_dim_aliases =
      array<alias_t, 4>{"M_0", "N_0", "M_1", "M_3"};

  return distributed_tensor_descriptor<
      poly,
      top_dim_aliases,
      partition_dim_aliases,
      element_dim_aliases>{};
}

template <index_t kMPerBlock, index_t kKPerBlock, index_t kKPack>
constexpr auto make_a_smem_tensor_descriptor() {
  constexpr index_t K1 = kKPack;
  constexpr index_t K0 = kKPerBlock / K1;

#if 1
  constexpr index_t M1 = 8;
  constexpr index_t M0 = kMPerBlock / M1;

  constexpr auto a_smem_tensor_m0_m1_k0_k1_desc =
      make_aliased_naive_packed_tensor_descriptor(
          aliases<"M0", "M1", "K0", "K1">{},
          alias<"Offset">{},
          {M0, M1, K0, K1});

  constexpr auto a_smem_tensor_m0_m1r_k0r_k1_rotate_desc =
      morph_tensor_descriptor(
          a_smem_tensor_m0_m1_k0_k1_desc,
          aliases<"M0", "M1r", "K0r", "K1">{},
          aliases<"Offset">{},
          aliases<"M1r", "K0r">{},
          nd_index<2>{M1, K0},
          mint::make_tuple(
              poly::rotate2d<index_t, index_t>{K0, -1},
              aliases<"M1", "K0", "M1r", "K0r">{}));

  constexpr auto a_smem_tensor_desc = morph_tensor_descriptor(
      a_smem_tensor_m0_m1r_k0r_k1_rotate_desc,
      aliases<"M", "K">{},
      aliases<"Offset">{},
      aliases<"M", "K">{},
      nd_index<2>{kMPerBlock, kKPerBlock},
      mint::make_tuple(
          poly::split<nd_index<2>>{{K0, K1}}, aliases<"K0r", "K1", "K">{}),
      mint::make_tuple(
          poly::split<nd_index<2>>{{M0, M1}}, aliases<"M0", "M1r", "M">{}));
#else
  // a smem tensor_descriptor: naively packed [kMPerBlock, kKPerBlock]
  constexpr auto a_smem_tensor_k0_m_k1_desc =
      make_aliased_naive_packed_tensor_descriptor(
          aliases<"K0", "M", "K1">{},
          alias<"Offset">{},
          {kKPerBlock / kKPack, kMPerBlock, kKPack});

  constexpr auto a_smem_tensor_desc = morph_tensor_descriptor(
      a_smem_tensor_k0_m_k1_desc,
      aliases<"M", "K">{},
      aliases<"Offset">{},
      aliases<"K">{},
      nd_index<1>{kKPerBlock},
      mint::make_tuple(
          poly::split<nd_index<2>>{{kKPerBlock / kKPack, kKPack}},
          aliases<"K0", "K1", "K">{}),
      mint::make_tuple(poly::none{}, aliases<"M">{}));
#endif

  return a_smem_tensor_desc;
}

template <index_t kNPerBlock, index_t kKPerBlock, index_t kKPack>
constexpr auto make_b_smem_tensor_descriptor() {
  constexpr index_t K1 = kKPack;
  constexpr index_t K0 = kKPerBlock / K1;

#if 1
  constexpr index_t N1 = 8;
  constexpr index_t N0 = kNPerBlock / N1;

  constexpr auto b_smem_tensor_n0_n1_k0_k1_desc =
      make_aliased_naive_packed_tensor_descriptor(
          aliases<"N0", "N1", "K0", "K1">{},
          alias<"Offset">{},
          {N0, N1, K0, K1});

  constexpr auto b_smem_tensor_n0_n1r_k0r_k1_rotate_desc =
      morph_tensor_descriptor(
          b_smem_tensor_n0_n1_k0_k1_desc,
          aliases<"N0", "N1r", "K0r", "K1">{},
          aliases<"Offset">{},
          aliases<"N1r", "K0r">{},
          nd_index<2>{N1, K0},
          mint::make_tuple(
              poly::rotate2d<index_t, index_t>{K0, -1},
              aliases<"N1", "K0", "N1r", "K0r">{}));

  constexpr auto b_smem_tensor_desc = morph_tensor_descriptor(
      b_smem_tensor_n0_n1r_k0r_k1_rotate_desc,
      aliases<"N", "K">{},
      aliases<"Offset">{},
      aliases<"N", "K">{},
      nd_index<2>{kNPerBlock, kKPerBlock},
      mint::make_tuple(
          poly::split<nd_index<2>>{{K0, K1}}, aliases<"K0r", "K1", "K">{}),
      mint::make_tuple(
          poly::split<nd_index<2>>{{N0, N1}}, aliases<"N0", "N1r", "N">{}));
#else
  // b smem tensor_descriptor: naively packed [kKPerBlock, kNPerBlock]
  constexpr auto b_smem_tensor_k0_n_k1_desc =
      make_aliased_naive_packed_tensor_descriptor(
          aliases<"K0", "N", "K1">{},
          alias<"Offset">{},
          {kKPerBlock / kKPack, kNPerBlock, kKPack});

  constexpr auto b_smem_tensor_desc = morph_tensor_descriptor(
      b_smem_tensor_k0_n_k1_desc,
      aliases<"N", "K">{},
      aliases<"Offset">{},
      aliases<"K">{},
      nd_index<1>{kKPerBlock},
      mint::make_tuple(
          poly::split<nd_index<2>>{{kKPerBlock / kKPack, kKPack}},
          aliases<"K0", "K1", "K">{}),
      mint::make_tuple(poly::none{}, aliases<"N">{}));
#endif

  return b_smem_tensor_desc;
}

} // namespace mint

template <
    class T,
    class TAcc,
    index_t kThreadPerBlock,
    index_t kMPerBlock,
    index_t kNPerBlock,
    index_t kKPerBlock,
    index_t kMPerWarp,
    index_t kNPerWarp,
    index_t kKPerWarp,
    index_t kKPack,
    index_t kMRepeat = 1,
    index_t kNRepeat = 1,
    index_t kKRepeat = 1,
    index_t kMinOccupancy = 1>
__launch_bounds__(kThreadPerBlock, kMinOccupancy) __global__
    void gpu_gemm_kernel(
        T* p_c_gmem,
        const T* p_a_gmem,
        const T* p_b_gmem,
        index_t m_size,
        index_t n_size,
        index_t k_size) {
  static_assert(kKPerBlock % (kKPerWarp * kKRepeat) == 0);

  // this thread block
  using this_thread_block = thread_in_this_block<kThreadPerBlock>;

  // a gmem tensor_descriptor: [M, K]
  const auto a_gmem_tensor_desc = make_aliased_naive_packed_tensor_descriptor(
      aliases<"M", "K">{}, alias<"Offset">{}, {m_size, k_size});

  // b gmem tensor_descriptor: [K, N]
  const auto b_gmem_tensor_desc = make_aliased_naive_packed_tensor_descriptor(
      aliases<"N", "K">{}, alias<"Offset">{}, {n_size, k_size});

  // c gmem tensor_descriptor: [M, N]
  const auto c_gmem_tensor_desc = make_aliased_naive_packed_tensor_descriptor(
      aliases<"M", "N">{}, alias<"Offset">{}, {m_size, n_size});

  // a distributed_tensor_descriptor for block gmem-to-smem copy
  constexpr auto a_block_copy_dstr = make_block_matmul_a_m_k_distribution<
      kThreadPerBlock,
      kMPerBlock,
      kKPerBlock,
      kKPack>();

  // b distributed_tensor_descriptor for block gmem-to-smem copy
  constexpr auto b_block_copy_dstr = make_block_matmul_b_n_k_distribution<
      kThreadPerBlock,
      kNPerBlock,
      kKPerBlock,
      kKPack>();

  constexpr auto a_block_copy_element_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<a_block_copy_dstr.element_ndim()>{},
          index_constant<-1>{},
          a_block_copy_dstr.element_lengths());

  constexpr auto b_block_copy_element_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<b_block_copy_dstr.element_ndim()>{},
          index_constant<-1>{},
          b_block_copy_dstr.element_lengths());

  constexpr auto a_warp_matmul_dstr = make_warp_matmul_a_m_k_distribution<
      kMPerWarp,
      kKPerWarp,
      kKPack,
      kMRepeat,
      kKRepeat>();

  constexpr auto b_warp_matmul_dstr = make_warp_matmul_b_n_k_distribution<
      kNPerWarp,
      kKPerWarp,
      kKPack,
      kNRepeat,
      kKRepeat>();

  constexpr auto c_warp_matmul_dstr =
      make_warp_matmul_c_m_n_distribution<kMPerWarp, kNPerWarp, 1, 1>();

  constexpr auto acc_warp_matmul_dstr = make_warp_matmul_c_m_n_distribution<
      kMPerWarp,
      kNPerWarp,
      kMRepeat,
      kNRepeat>();

  constexpr auto a_warp_matmul_elem_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<a_warp_matmul_dstr.element_ndim()>{},
          index_constant<-1>{},
          a_warp_matmul_dstr.element_lengths());

  constexpr auto b_warp_matmul_elem_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<b_warp_matmul_dstr.element_ndim()>{},
          index_constant<-1>{},
          b_warp_matmul_dstr.element_lengths());

  constexpr auto c_warp_matmul_elem_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<c_warp_matmul_dstr.element_ndim()>{},
          index_constant<-1>{},
          c_warp_matmul_dstr.element_lengths());

  constexpr auto acc_warp_matmul_elem_layout =
      make_aliased_naive_packed_tensor_descriptor(
          make_index_sequence<acc_warp_matmul_dstr.element_ndim()>{},
          index_constant<-1>{},
          acc_warp_matmul_dstr.element_lengths());

  // a gmem tensor_view
  const auto a_gmem_tensor_view = make_tensor_view(
      a_gmem_tensor_desc,
      make_global_memory_view(
          p_a_gmem, a_gmem_tensor_desc.bottom_lengths()[0]));

  // a gmem mask
  const auto a_gmem_mask = tuple<>{};

  // b gmem tensor_view
  const auto b_gmem_tensor_view = make_tensor_view(
      b_gmem_tensor_desc,
      make_global_memory_view(
          p_b_gmem, b_gmem_tensor_desc.bottom_lengths()[0]));

  // b gmem mask
  const auto b_gmem_mask = tuple<>{};

  constexpr auto a_smem_tensor_desc =
      make_a_smem_tensor_descriptor<kMPerBlock, kKPerBlock, kKPack>();
  constexpr auto b_smem_tensor_desc =
      make_b_smem_tensor_descriptor<kNPerBlock, kKPerBlock, kKPack>();

  constexpr index_t a_smem_size =
      a_smem_tensor_desc.bottom_lengths()[0] * sizeof(T);
  constexpr index_t b_smem_size =
      b_smem_tensor_desc.bottom_lengths()[0] * sizeof(T);
  constexpr index_t smem_size = a_smem_size + b_smem_size;

  __shared__ char p_shared[smem_size];
  T* p_a_shared = reinterpret_cast<T*>(p_shared);
  T* p_b_shared = reinterpret_cast<T*>(p_shared + a_smem_size);

  // a smem tensor_view
  const auto a_smem_tensor_view = make_tensor_view(
      a_smem_tensor_desc,
      make_shared_memory_view(
          p_a_shared, a_smem_tensor_desc.bottom_lengths()[0]));

  // a smem mask
  const auto a_smem_mask = tuple<>{};

  // b smem tensor_view
  const auto b_smem_tensor_view = make_tensor_view(
      b_smem_tensor_desc,
      make_shared_memory_view(
          p_b_shared, b_smem_tensor_desc.bottom_lengths()[0]));

  // b smem mask
  const auto b_smem_mask = tuple<>{};

  const auto mn_block_idx =
      GetBlockIdx<kMPerBlock, kNPerBlock>(block_id(), m_size, n_size);

  const auto m_block = __builtin_amdgcn_readfirstlane(mn_block_idx[0_ic]);
  const auto n_block = __builtin_amdgcn_readfirstlane(mn_block_idx[1_ic]);

  // a block gmem window for block copy
  auto a_block_copy_gmem_window = make_distributed_window(
      a_gmem_tensor_view,
      {m_block * kMPerBlock, 0},
      constant<a_block_copy_dstr>{},
      constant<a_block_copy_element_layout>{},
      constant<this_thread_block{}>{});

  // b block gmem window for block copy
  auto b_block_copy_gmem_window = make_distributed_window(
      b_gmem_tensor_view,
      {n_block * kNPerBlock, 0},
      constant<b_block_copy_dstr>{},
      constant<b_block_copy_element_layout>{},
      constant<this_thread_block{}>{});

  // a block smem window for block copy
  auto a_block_copy_smem_window = make_distributed_window(
      a_smem_tensor_view,
      {0, 0},
      constant<a_block_copy_dstr>{},
      constant<a_block_copy_element_layout>{},
      constant<this_thread_block{}>{});

  // b block smem window for block copy
  auto b_block_copy_smem_window = make_distributed_window(
      b_smem_tensor_view,
      {0, 0},
      constant<b_block_copy_dstr>{},
      constant<b_block_copy_element_layout>{},
      constant<this_thread_block{}>{});

  // constexpr index_t kMWarpPerBlock = kMPerBlock / (kMPerWarp * kMRepeat);
  constexpr index_t kNWarpPerBlock = kNPerBlock / (kNPerWarp * kNRepeat);

  const index_t m_warp =
      __builtin_amdgcn_readfirstlane(warp_id() / kNWarpPerBlock);
  const index_t n_warp =
      __builtin_amdgcn_readfirstlane(warp_id() % kNWarpPerBlock);

  // a warp smem window for warp matmul
  auto a_warp_matmul_smem_window = make_distributed_window(
      a_smem_tensor_view,
      {m_warp * kMPerWarp * kMRepeat, 0},
      constant<a_warp_matmul_dstr>{},
      constant<a_warp_matmul_elem_layout>{},
      constant<thread_in_this_warp{}>{});

  // b warp smem window for warp matmul
  auto b_warp_matmul_smem_window = make_distributed_window(
      b_smem_tensor_view,
      {n_warp * kNPerWarp * kNRepeat, 0},
      constant<b_warp_matmul_dstr>{},
      constant<b_warp_matmul_elem_layout>{},
      constant<thread_in_this_warp{}>{});

  // acc warp tile
  auto acc_warp_tile = distributed_tensor<
      acc_warp_matmul_dstr,
      acc_warp_matmul_elem_layout,
      owned_vgpr_memory<
          TAcc,
          acc_warp_matmul_elem_layout.bottom_lengths()[0]>>{};

  constexpr auto a_block_vector_dims = array<alias_t, 1>{"K_1"};
  constexpr auto a_block_vector_lengths = array<index_t, 1>{kKPack};

  constexpr auto b_block_vector_dims = array<alias_t, 1>{"K_1"};
  constexpr auto b_block_vector_lengths = array<index_t, 1>{kKPack};

  constexpr auto a_warp_vector_dims = array<alias_t, 1>{"K_2"};
  constexpr auto a_warp_vector_lengths = array<index_t, 1>{kKPack};

  constexpr auto b_warp_vector_dims = array<alias_t, 1>{"K_2"};
  constexpr auto b_warp_vector_lengths = array<index_t, 1>{kKPack};

  constexpr auto a_block_load_freezed_dims =
      mint::make_tuple(array<alias_t, 1>{"K"}, array<alias_t, 1>{"M"});
  constexpr auto b_block_load_freezed_dims =
      mint::make_tuple(array<alias_t, 1>{"K"}, array<alias_t, 1>{"N"});

  constexpr auto a_block_store_freezed_dims = mint::make_tuple(
      array<alias_t, 6>{"M1r", "M1", "K", "K0r", "K1", "K0"},
      array<alias_t, 6>{"M", "M0", "M1r", "M1", "K0r", "K0"});
  constexpr auto b_block_store_freezed_dims = mint::make_tuple(
      array<alias_t, 6>{"N1r", "N1", "K", "K0r", "K1", "K0"},
      array<alias_t, 6>{"N", "N0", "N1r", "N1", "K0r", "K0"});

  constexpr auto a_warp_freezed_dims = mint::make_tuple(
      array<alias_t, 5>{"M", "M0", "M1r", "M1", "K1"},
      array<alias_t, 6>{"K", "K0r", "K1", "K0", "M1", "M1r"},
      array<alias_t, 6>{"M", "M0", "M1r", "M1", "K0", "K0r"});
  constexpr auto b_warp_freezed_dims = mint::make_tuple(
      array<alias_t, 5>{"N", "N0", "N1r", "N1", "K1"},
      array<alias_t, 6>{"K", "K0r", "K1", "K0", "N1", "N1r"},
      array<alias_t, 6>{"N", "N0", "N1r", "N1", "K0", "K0r"});

  // stage 0
  auto a_block_tile =
      mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
          a_block_vector_dims,
          a_block_vector_lengths,
          a_block_load_freezed_dims>(a_block_copy_gmem_window, a_gmem_mask);
  mint::tensor::experimental::move_window_freezed_dim_conjectural<
      array<alias_t, 1>{"M"}>(a_block_copy_gmem_window, {0, kKPerBlock});

  auto b_block_tile =
      mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
          b_block_vector_dims,
          b_block_vector_lengths,
          b_block_load_freezed_dims>(b_block_copy_gmem_window, b_gmem_mask);
  mint::tensor::experimental::move_window_freezed_dim_conjectural<
      array<alias_t, 1>{"N"}>(b_block_copy_gmem_window, {0, kKPerBlock});
    
      //store to smem
  mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
      a_block_vector_dims,
      a_block_vector_lengths,
      a_block_store_freezed_dims>(
      a_block_copy_smem_window, a_smem_mask, a_block_tile);

  mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
      b_block_vector_dims,
      b_block_vector_lengths,
      b_block_store_freezed_dims>(
      b_block_copy_smem_window, b_smem_mask, b_block_tile);

  acc_warp_tile.fill(0);

  index_t k = 0;
  do {
    a_block_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            a_block_vector_dims,
            a_block_vector_lengths,
            a_block_load_freezed_dims>(a_block_copy_gmem_window, a_gmem_mask);
    mint::tensor::experimental::move_window_freezed_dim_conjectural<
        array<alias_t, 1>{"M"}>(a_block_copy_gmem_window, {0, kKPerBlock});

    b_block_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            b_block_vector_dims,
            b_block_vector_lengths,
            b_block_load_freezed_dims>(b_block_copy_gmem_window, b_gmem_mask);
    mint::tensor::experimental::move_window_freezed_dim_conjectural<
        array<alias_t, 1>{"N"}>(b_block_copy_gmem_window, {0, kKPerBlock});

    block_sync_lds();
    //from lds to vgpr for MFMA
    const auto a_warp_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            a_warp_vector_dims,
            a_warp_vector_lengths,
            a_warp_freezed_dims>(a_warp_matmul_smem_window, a_smem_mask);

    const auto b_warp_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            b_warp_vector_dims,
            b_warp_vector_lengths,
            b_warp_freezed_dims>(b_warp_matmul_smem_window, b_smem_mask);

    matmul_xdl<false, kMPerWarp, kNPerWarp>(
        acc_warp_tile, a_warp_tile, b_warp_tile);

    block_sync_lds();

    mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
        a_block_vector_dims,
        a_block_vector_lengths,
        a_block_store_freezed_dims>(
        a_block_copy_smem_window, a_smem_mask, a_block_tile);

    mint::tile::generic::experimental::masked_store_no_shuffle_vectorized<
        b_block_vector_dims,
        b_block_vector_lengths,
        b_block_store_freezed_dims>(
        b_block_copy_smem_window, b_smem_mask, b_block_tile);

    k += kKPerBlock;
  } while (k < (k_size - kKPerBlock));

  // tail
  {
    block_sync_lds();

    const auto a_warp_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            a_warp_vector_dims,
            a_warp_vector_lengths,
            a_warp_freezed_dims>(a_warp_matmul_smem_window, a_smem_mask);

    const auto b_warp_tile =
        mint::tile::generic::experimental::masked_load_no_shuffle_vectorized<
            b_warp_vector_dims,
            b_warp_vector_lengths,
            b_warp_freezed_dims>(b_warp_matmul_smem_window, b_smem_mask);

    matmul_xdl<false, kMPerWarp, kNPerWarp>(
        acc_warp_tile, a_warp_tile, b_warp_tile);
  }

  // c gmem tensor_view
  const auto c_gmem_tensor_view = make_tensor_view(
      c_gmem_tensor_desc,
      make_global_memory_view(
          p_c_gmem, c_gmem_tensor_desc.bottom_lengths()[0]));

  // c tensor mask
  const auto c_gmem_mask = tuple<>{};

  // c warp gmem window
  auto c_warp_copy_gmem_window = make_distributed_window(
      c_gmem_tensor_view,
      {m_block * kMPerBlock + m_warp * kMPerWarp * kMRepeat,
       n_block * kNPerBlock + n_warp * kNPerWarp * kNRepeat},
      constant<c_warp_matmul_dstr>{},
      constant<c_warp_matmul_elem_layout>{},
      constant<thread_in_this_warp{}>{});

  // acc warp tile
  auto c_warp_tile = distributed_tensor<
      c_warp_matmul_dstr,
      c_warp_matmul_elem_layout,
      owned_vgpr_memory<T, c_warp_matmul_elem_layout.bottom_lengths()[0]>>{};

  constexpr auto m_n_sharded_lengths = c_warp_matmul_dstr.sharded_lengths();
  constexpr auto kMs = m_n_sharded_lengths[0_ic];
  // constexpr auto kNs = m_n_sharded_lengths[1_ic];

  static_for_n<kMRepeat>()([&](auto iMR) {
    static_for_n<kNRepeat>()([&](auto iNR) {
      static_for_n<kMs[1]>()([&](auto iM1s) {
        static_for_n<kMs[2]>()([&](auto iM3s) {
          constexpr auto acc_m_n_idx = nd_index<4>{iMR, iNR, iM1s, iM3s};
          constexpr auto c_m_n_idx = nd_index<4>{0, 0, iM1s, iM3s};
          c_warp_tile.template element<c_m_n_idx>() =
              static_cast<T>(acc_warp_tile.template element<acc_m_n_idx>());
        });
      });
      warp::masked_store(c_warp_copy_gmem_window, c_gmem_mask, c_warp_tile);
      move_window(c_warp_copy_gmem_window, {0, kNPerWarp});
    });

    move_window(c_warp_copy_gmem_window, {kMPerWarp, -kNPerWarp * kNRepeat});
  });
}

template <class T, class TAcc = float>
void gpu_gemm(
    T* p_c,
    const T* p_a,
    const T* p_b,
    index_t m_size,
    index_t n_size,
    index_t k_size) {
  constexpr index_t kThreadPerBlock = 256;

  constexpr index_t kMPerBlock = 128;
  constexpr index_t kNPerBlock = 256;
  constexpr index_t kKPerBlock = 64;

  constexpr index_t kMPerWarp = 16;
  constexpr index_t kNPerWarp = 16;
  constexpr index_t kKPerWarp = 32;

  constexpr index_t kKPack = 8;

  constexpr index_t kMRepeat = 4;
  constexpr index_t kNRepeat = 8;
  constexpr index_t kKRepeat = kKPerBlock / kKPerWarp;

  const int num_m_block = (m_size + kMPerBlock - 1) / kMPerBlock;
  const int num_n_block = (n_size + kNPerBlock - 1) / kNPerBlock;

  constexpr index_t kMinOccupancy = 2;

  const auto kernel = gpu_gemm_kernel<
      T,
      TAcc,
      kThreadPerBlock,
      kMPerBlock,
      kNPerBlock,
      kKPerBlock,
      kMPerWarp,
      kNPerWarp,
      kKPerWarp,
      kKPack,
      kMRepeat,
      kNRepeat,
      kKRepeat,
      kMinOccupancy>;

  // warm up
  for (int i = 0; i < 20; ++i) {
    kernel<<<num_m_block * num_n_block, kThreadPerBlock>>>(
        p_c, p_a, p_b, m_size, n_size, k_size);
    hip_check_error(hipGetLastError());
  }

  hipEvent_t start, stop;

  hip_check_error(hipEventCreate(&start));
  hip_check_error(hipEventCreate(&stop));

  hip_check_error(hipDeviceSynchronize());
  hip_check_error(hipEventRecord(start));

  for (int i = 0; i < 50; ++i) {
    kernel<<<num_m_block * num_n_block, kThreadPerBlock>>>(
        p_c, p_a, p_b, m_size, n_size, k_size);
    hip_check_error(hipGetLastError());
  }

  hip_check_error(hipEventRecord(stop));
  hip_check_error(hipEventSynchronize(stop));

  float total_time = 0;

  hip_check_error(hipEventElapsedTime(&total_time, start, stop));

  hip_check_error(hipEventDestroy(start));
  hip_check_error(hipEventDestroy(stop));

  const auto avg_time = total_time / 50;

  const std::size_t flop = std::size_t(2) * m_size * n_size * k_size;

  const float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

  std::cout << "kernel time (ms) = " << avg_time << " Tflops = " << tflops
            << std::endl;
}

template <class T>
bool test_gemm() {
  bool pass = true;

  const index_t m_size = 256;
  const index_t n_size = 256;
  const index_t k_size = 256;

  const index_t a_size = m_size * k_size;
  const index_t b_size = n_size * k_size;
  const index_t c_size = m_size * n_size;

  T* p_a_ref = new T[a_size];
  T* p_b_ref = new T[b_size];
  T* p_c_ref = new T[c_size];
  T* p_c_out = new T[c_size];

  T* p_a_dev;
  T* p_b_dev;
  T* p_c_dev;

  hip_check_error(hipMalloc(&p_a_dev, a_size * sizeof(T)));
  hip_check_error(hipMalloc(&p_b_dev, b_size * sizeof(T)));
  hip_check_error(hipMalloc(&p_c_dev, c_size * sizeof(T)));

  std::default_random_engine eng;
  int seed = 0;
  eng.seed(seed);
#if 1
  std::uniform_int_distribution<> dist_a(0, 2);
  std::uniform_int_distribution<> dist_b(-2, 2);
#else
  std::uniform_real_distribution<> dist_a(0, 2);
  std::uniform_real_distribution<> dist_b(-2, 2);
#endif

  for (index_t i = 0; i < a_size; i++)
    p_a_ref[i] = (T)dist_a(eng);

  for (index_t i = 0; i < b_size; i++)
    p_b_ref[i] = (T)dist_b(eng);

  cpu_gemm(p_c_ref, p_a_ref, p_b_ref, m_size, n_size, k_size);
printf("cpu done\n");
  hip_check_error(
      hipMemcpy(p_a_dev, p_a_ref, a_size * sizeof(T), hipMemcpyHostToDevice));
  hip_check_error(
      hipMemcpy(p_b_dev, p_b_ref, b_size * sizeof(T), hipMemcpyHostToDevice));
printf("going into gpu version \n");
  gpu_gemm(p_c_dev, p_a_dev, p_b_dev, m_size, n_size, k_size);
printf("came out of gpu version\n");
  hip_check_error(
      hipMemcpy(p_c_out, p_c_dev, c_size * sizeof(T), hipMemcpyDeviceToHost));

  int mismatch_count = 0;
  for (index_t i = 0; i < c_size; i++) {
    if (!is_equal(p_c_out[i], p_c_ref[i], 0.01)) {
      pass = false;
      if (mismatch_count < 20)
        printf(
            "mismatch: %d, %f %f\n", i, (float)p_c_out[i], (float)p_c_ref[i]);
      mismatch_count++;
    }
  }

  printf("error rate: %f\n", (float)mismatch_count / c_size);

  hip_check_error(hipFree(p_a_dev));
  hip_check_error(hipFree(p_b_dev));
  hip_check_error(hipFree(p_c_dev));

  delete[] p_a_ref;
  delete[] p_b_ref;
  delete[] p_c_ref;
  delete[] p_c_out;

  return pass;
}

// #if 0
// TEST(MintTestKernelROCM, rocm_gemm_xdl_v1) {
//   EXPECT_TRUE(test_gemm<fp16_t>());
// }
// #else
int main(int argc, char* argv[]) {
    printf("hello\n");
  if (test_gemm<fp16_t>())
    printf("pass\n");
  else
    printf("fail\n");
}
//#endif
