#pragma once
#include <mint/config.h>
#include <mint/core.h>

namespace mint {

struct thread_in_this_warp {
  static constexpr index_t kWarpSize = MINT_WARP_SIZE;

  static consteval index_t partition_ndim() {
    return 1;
  }

  static consteval nd_index<1> partition_lengths() {
    return {kWarpSize};
  }

  static consteval index_t partition_num() {
    return kWarpSize;
  }

  MINT_DEVICE static nd_index<1> my_partition_idx() {
    return {static_cast<index_t>(threadIdx.x % kWarpSize)};
  }
};

template <index_t kThreadPerBlock>
struct thread_in_this_block {
  static consteval index_t partition_ndim() {
    return 1;
  }

  static consteval nd_index<1> partition_lengths() {
    return {kThreadPerBlock};
  }

  static consteval index_t partition_num() {
    return kThreadPerBlock;
  }

  MINT_DEVICE static nd_index<1> my_partition_idx() {
    return {static_cast<index_t>(threadIdx.x)};
  }
};

} // namespace mint
