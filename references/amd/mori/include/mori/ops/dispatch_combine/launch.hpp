// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_runtime_api.h>

#include <string>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori {
namespace moe {

// -----------------------------------------------------------------------
// KernelRegistry — manages AOT-compiled .hsaco loading and function lookup
// -----------------------------------------------------------------------
class KernelRegistry {
 public:
  static KernelRegistry& Instance();

  // Load all .hsaco files from a directory and initialize shmem GPU states.
  void LoadFromDirectory(const std::string& dir);

  // Load a single .hsaco file.
  void LoadModule(const std::string& hsaco_path);

  // Auto-detect GPU arch + NIC, search for matching kernels.
  // With base_dir: search base_dir/<arch>_<nic>/, then base_dir/ flat.
  // Without args: search default paths (JIT cache ~/.mori/jit/<arch>_<nic>/latest/).
  // Called automatically on first LaunchDispatch/Combine if not already loaded.
  void AutoLoad(const std::string& base_dir);
  void AutoLoad();

  // Get a kernel function by name. Caches lookups.
  hipFunction_t GetFunction(const std::string& func_name);

  // Launch a kernel by name.
  void Launch(const std::string& func_name, unsigned int grid_x, unsigned int block_x,
              unsigned int shared_mem, hipStream_t stream, void* args, size_t args_size);

  bool IsLoaded() const;

 private:
  KernelRegistry() = default;
  struct Impl;
  static Impl& GetImpl();
};

// -----------------------------------------------------------------------
// C++ launch API — no Python, no JIT, no torch dependency
//
// All parameters with default value -1 use the config defaults.
// Requires BUILD_OPS_DEVICE=ON at CMake build time.
// -----------------------------------------------------------------------

// Full dispatch: prepare input + launch kernel
void LaunchDispatch(EpDispatchCombineHandle& handle, void* input, void* weights, void* scales,
                    void* indices, int64_t num_tokens, hipDataType dtype, int block_num = -1,
                    int rdma_block_num = -1, int warp_per_block = -1, hipStream_t stream = 0,
                    int hidden_dim = -1);

// Full combine: prepare input + launch kernel
void LaunchCombine(EpDispatchCombineHandle& handle, void* input, void* weights, void* indices,
                   int64_t num_tokens, hipDataType dtype, int block_num = -1,
                   int rdma_block_num = -1, int warp_per_block = -1, int use_external_inp_buf = -1,
                   hipStream_t stream = 0, int hidden_dim = -1);

// Async split: dispatch recv phase
void LaunchDispatchRecv(EpDispatchCombineHandle& handle, int block_num = -1,
                        int warp_per_block = -1, hipStream_t stream = 0);

// Async split: combine recv phase
void LaunchCombineRecv(EpDispatchCombineHandle& handle, int block_num = -1, int warp_per_block = -1,
                       hipStream_t stream = 0);

void LaunchLocalExpertCount(const EpDispatchCombineConfig& config, const index_t* indices,
                            const index_t* total_recv_token_num, int* local_expert_count,
                            int block_num = -1, int warp_per_block = -1, hipStream_t stream = 0);

void LaunchHotHelperPack(const EpDispatchCombineConfig& config, hipDataType dtype,
                         const void* input, const float* topk_weights,
                         const int64_t* hot_flat_positions, const int64_t* hot_owner_slots,
                         int64_t num_hot_routes, void* hot_packed_x,
                         float* hot_packed_scores, int64_t* hot_packed_src_info,
                         int* hot_packed_count, int64_t num_hot_slots,
                         int64_t max_hot_rows_per_slot, int block_num = -1,
                         int warp_per_block = -1, hipStream_t stream = 0,
                         int hidden_dim = -1);

void LaunchHotHelperCompactPack(const EpDispatchCombineConfig& config, hipDataType dtype,
                                const void* input, const float* topk_weights,
                                const int64_t* hot_flat_positions,
                                const int64_t* hot_owner_slots, int64_t num_hot_routes,
                                void* hot_packed_x, float* hot_packed_scores,
                                int64_t* hot_packed_src_info, int* hot_packed_count,
                                int64_t num_hot_slots, int block_num = -1,
                                int warp_per_block = -1, hipStream_t stream = 0,
                                int hidden_dim = -1);

void LaunchStandardEpCompactDispatch(EpDispatchCombineHandle& handle, hipDataType dtype,
                                     const void* local_rows,
                                     const int64_t* local_flat_positions,
                                     const int64_t* local_num_tokens_per_expert,
                                     const int64_t* recv_counts_rank_major,
                                     const int64_t* input_splits,
                                     const int64_t* output_splits, int64_t num_segments,
                                     int64_t num_input_rows, int64_t num_output_rows,
                                     void* rank_major_rows,
                                     int64_t* rank_major_flat_positions,
                                     int64_t flat_position_rank_stride,
                                     int block_num = -1, int warp_per_block = -1,
                                     hipStream_t stream = 0, int hidden_dim = -1);

void LaunchStandardEpCompactCombine(EpDispatchCombineHandle& handle, hipDataType dtype,
                                    const void* expert_major_rows,
                                    const int64_t* expert_major_flat_positions,
                                    const int64_t* expert_major_to_rank_major_indices,
                                    const int64_t* recv_counts_rank_major,
                                    const int64_t* input_splits,
                                    const int64_t* output_splits, int64_t num_segments,
                                    int64_t num_input_rows,
                                    int64_t num_output_rows, void* source_rank_rows,
                                    int64_t* source_rank_flat_positions,
                                    const float* top_scores_flat = nullptr,
                                    int64_t top_scores_flat_size = 0,
                                    int64_t top_k = 0,
                                    int64_t flat_position_offset = 0,
                                    void* token_output = nullptr,
                                    int64_t token_output_rows = 0,
                                    int block_num = -1, int warp_per_block = -1,
                                    hipStream_t stream = 0, int hidden_dim = -1);

void LaunchStandardEpCompactWeightedOutputBackward(
    EpDispatchCombineHandle& handle, hipDataType dtype, const void* source_rank_rows,
    const int64_t* source_rank_flat_positions, const float* top_scores_flat,
    int64_t top_scores_flat_size, int64_t top_k, int64_t flat_position_offset,
    const void* grad_token_output, int64_t token_output_rows, int64_t num_rows,
    void* grad_source_rank_rows, float* grad_top_scores_flat, int block_num = -1,
    int warp_per_block = -1, hipStream_t stream = 0, int hidden_dim = -1);

// Reset internal state between iterations
void LaunchReset(EpDispatchCombineHandle& handle, hipStream_t stream = 0);

}  // namespace moe
}  // namespace mori
