// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/instance_data/common.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

// Backward weight WMMA V3 algorithm descriptor.
// Uses 3D block transfers and BlockGemmPipeline (scheduler + pipeline_version).
// This satisfies BwdWmmaV3Algorithm which routes to ConvBwdWeightWmmaV3Factory
// and maps to DeviceGroupedConvBwdWeight_Wmma_CShuffleV3.
struct WmmaV3BwdWeightAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;

    ThreadBlock thread_block;

    struct GridwiseGemm
    {
        std::size_t k1;
        std::size_t m_per_wmma;
        std::size_t n_per_wmma;
        std::size_t m_wmma_per_wave;
        std::size_t n_wmma_per_wave;
    } gridwise_gemm;

    TransferABC transfer;

    ConvSpecial bwd_weight_specialization;

    struct BlockGemmPipeline
    {
        ckb::PipelineScheduler scheduler;
        ckb::PipelineVersion pipeline_version;
    } block_gemm_pipeline;

    std::size_t max_transpose_transfer_src_scalar_per_vector;
    std::size_t max_transpose_transfer_dst_scalar_per_vector;

    ElementwiseOps elementwise_ops;
};

static_assert(ckb::factory::BwdWmmaV3Algorithm<WmmaV3BwdWeightAlgorithm>);

template <std::size_t NumDTensor = 0>
using WmmaV3BwdWeightSignature = ConvSignature<NumDTensor>;

template <std::size_t NumDTensor = 0>
struct WmmaV3BwdWeightInstance
{
    WmmaV3BwdWeightSignature<NumDTensor> signature;
    WmmaV3BwdWeightAlgorithm algorithm;
};

// Helper to create WmmaV3BwdWeightInstance from
// DeviceGroupedConvBwdWeight_Wmma_CShuffleV3 template parameters.
template <std::size_t NumDTensor>
constexpr WmmaV3BwdWeightInstance<NumDTensor> DeviceGroupedConvBwdWeight_Wmma_CShuffleV3(
    // 1. NDimSpatial
    std::size_t spatial_dim,
    // 2-4. Layouts (A=In, B=Out, E=Wei)
    ckb::TensorLayout in_layout,
    ckb::TensorLayout wei_layout,
    ckb::TensorLayout out_layout,
    // 5-8. Data types
    ckb::DataType in_data_type,
    ckb::DataType wei_data_type,
    ckb::DataType out_data_type,
    ckb::DataType acc_data_type,
    // 9-11. Elementwise operations
    ckb::ElementwiseOperation in_elementwise_op,
    ckb::ElementwiseOperation wei_elementwise_op,
    ckb::ElementwiseOperation out_elementwise_op,
    // 12. Specialization
    ckb::ConvSpecialization conv_bwd_weight_specialization,
    // 13-16. Block dimensions
    std::size_t block_size,
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k0_per_block,
    // 17. ABK1
    std::size_t k1,
    // 18-21. WMMA parameters
    std::size_t m_per_wmma,
    std::size_t n_per_wmma,
    std::size_t m_wmma_per_wave,
    std::size_t n_wmma_per_wave,
    // 22-28. A block transfer parameters (3D: K0_M_K1)
    std::array<std::size_t, 3> a_thread_cluster_lengths,
    std::array<std::size_t, 3> a_thread_cluster_arrange_order,
    std::array<std::size_t, 3> a_block_transfer_src_access_order,
    std::size_t a_block_transfer_src_vector_dim,
    std::size_t a_block_transfer_src_scalar_per_vector,
    std::size_t a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    // 29-35. B block transfer parameters (3D: K0_N_K1)
    std::array<std::size_t, 3> b_thread_cluster_lengths,
    std::array<std::size_t, 3> b_thread_cluster_arrange_order,
    std::array<std::size_t, 3> b_block_transfer_src_access_order,
    std::size_t b_block_transfer_src_vector_dim,
    std::size_t b_block_transfer_src_scalar_per_vector,
    std::size_t b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    // 36-39. C shuffle parameters
    std::size_t c_shuffle_m_xdl_per_wave_per_shuffle,
    std::size_t c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    std::size_t c_block_transfer_scalar_per_vector,
    // 40. BlockGemmPipelineScheduler
    ckb::PipelineScheduler scheduler,
    // 41. BlockGemmPipelineVersion
    ckb::PipelineVersion pipeline_version)
{
    // clang-format off
    return WmmaV3BwdWeightInstance<NumDTensor>{
        .signature = {
            .spatial_dim            = spatial_dim,
            .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
            .input                  = {
                .config = {
                    .layout       = in_layout,
                    .data_type    = in_data_type,
                    .compute_type = in_data_type
                }
            },
            .weight = {
                .config = {
                    .layout       = wei_layout,
                    .data_type    = wei_data_type,
                    .compute_type = wei_data_type
                }
            },
            .output = {
                .config = {
                    .layout       = out_layout,
                    .data_type    = out_data_type,
                    .compute_type = out_data_type
                },
                .operation = {
                    .elementwise_operation    = out_elementwise_op,
                    .auxiliary_operand_configs = {}
                }
            },
            .data_type              = in_data_type,
            .accumulation_data_type = acc_data_type
        },
        .algorithm = {
            .thread_block = {
                .block_size = block_size,
                .tile_size  = {
                    .m = m_per_block,
                    .n = n_per_block,
                    .k = k0_per_block
                }
            },
            .gridwise_gemm = {
                .k1              = k1,
                .m_per_wmma      = m_per_wmma,
                .n_per_wmma      = n_per_wmma,
                .m_wmma_per_wave = m_wmma_per_wave,
                .n_wmma_per_wave = n_wmma_per_wave
            },
            .transfer = {
                .a = {
                    .block_transfer = {
                        .k0  = a_thread_cluster_lengths[0],
                        .m_n = a_thread_cluster_lengths[1],
                        .k1  = a_thread_cluster_lengths[2]
                    },
                    .lds_transfer = {
                        .src_vector_dim            = a_block_transfer_src_vector_dim,
                        .src_scalar_per_vector     = a_block_transfer_src_scalar_per_vector,
                        .lds_dst_scalar_per_vector = a_block_transfer_dst_scalar_per_vector_k1,
                        .is_direct_load            = false,
                        .lds_padding               = a_block_lds_extra_m
                    },
                    .thread_cluster_arrange_order = {
                        .order = a_thread_cluster_arrange_order
                    },
                    .src_access_order = {
                        .order = a_block_transfer_src_access_order
                    }
                },
                .b = {
                    .block_transfer = {
                        .k0  = b_thread_cluster_lengths[0],
                        .m_n = b_thread_cluster_lengths[1],
                        .k1  = b_thread_cluster_lengths[2]
                    },
                    .lds_transfer = {
                        .src_vector_dim            = b_block_transfer_src_vector_dim,
                        .src_scalar_per_vector     = b_block_transfer_src_scalar_per_vector,
                        .lds_dst_scalar_per_vector = b_block_transfer_dst_scalar_per_vector_k1,
                        .is_direct_load            = false,
                        .lds_padding               = b_block_lds_extra_n
                    },
                    .thread_cluster_arrange_order = {
                        .order = b_thread_cluster_arrange_order
                    },
                    .src_access_order = {
                        .order = b_block_transfer_src_access_order
                    }
                },
                .c = {
                    .thread_cluster_dims = {
                        .m_block        = c_thread_cluster_lengths[0],
                        .m_wave_per_xdl = c_thread_cluster_lengths[1],
                        .n_block        = c_thread_cluster_lengths[2],
                        .n_wave_per_xdl = c_thread_cluster_lengths[3]
                    },
                    .epilogue = {
                        .m_xdl_per_wave_per_shuffle = c_shuffle_m_xdl_per_wave_per_shuffle,
                        .n_per_wave_per_shuffle     = c_shuffle_n_xdl_per_wave_per_shuffle,
                        .scalar_per_vector          = c_block_transfer_scalar_per_vector
                    }
                }
            },
            .bwd_weight_specialization = conv_bwd_weight_specialization,
            .block_gemm_pipeline = {
                .scheduler        = scheduler,
                .pipeline_version = pipeline_version
            },
            .max_transpose_transfer_src_scalar_per_vector = 1,
            .max_transpose_transfer_dst_scalar_per_vector = 1,
            .elementwise_ops = {
                .input_op  = in_elementwise_op,
                .weight_op = wei_elementwise_op,
                .output_op = out_elementwise_op
            }
        }
    };
    // clang-format on
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
