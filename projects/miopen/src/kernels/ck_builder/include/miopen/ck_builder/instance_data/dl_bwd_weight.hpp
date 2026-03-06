// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/instance_data/common.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

// Backward weight DL algorithm descriptor.
// Key difference from forward DL: backward weight uses 5D block transfers
// (KBatch_K0_M0_M1_K1 and KBatch_K0_N0_N1_K1) instead of forward's 4D
// (K0_M0_M1_K1 and K0_N0_N1_K1).
struct DlBwdWeightAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;

    ThreadBlock thread_block;

    // DL-specific thread configuration
    struct ThreadConfig
    {
        std::size_t k0_per_block;
        std::size_t k1;
        std::size_t m1_per_thread;
        std::size_t n1_per_thread;
        std::size_t k_per_thread;
    } thread_config;

    static_assert(ckb::DlThreadConfigDescriptor<ThreadConfig>);

    // DL-specific thread cluster
    struct ThreadCluster
    {
        std::array<size_t, 2> m1_xs;
        std::array<size_t, 2> n1_xs;
    } thread_cluster;

    static_assert(ckb::DlThreadClusterDescriptor<ThreadCluster>);

    // DL-specific transfer descriptors for backward weight (5D block transfers)
    struct Transfer
    {
        // DL block transfer for A and B (5D: KBatch_K0_M0_M1_K1 / KBatch_K0_N0_N1_K1)
        struct DlBlockTransfer5D
        {
            std::array<size_t, 5> thread_slice_lengths;
            std::array<size_t, 5> thread_cluster_lengths;
            std::array<size_t, 5> thread_cluster_arrange_order;
            std::array<size_t, 5> src_access_order;
            std::array<size_t, 5> src_vector_tensor_lengths;
            std::array<size_t, 5> src_vector_tensor_contiguous_dim_order;
            std::array<size_t, 5> dst_vector_tensor_lengths;
        };

        static_assert(ckb::DlBlockTransferDescriptor5D<DlBlockTransfer5D>);

        DlBlockTransfer5D a;
        DlBlockTransfer5D b;

        // DL epilogue descriptor for C (6D) - same as forward
        struct DlEpilogue
        {
            std::array<size_t, 6> src_dst_access_order;
            std::size_t src_dst_vector_dim;
            std::size_t dst_scalar_per_vector;
        } c;

        static_assert(ckb::DlEpilogueDescriptor<DlEpilogue>);
    } transfer;

    // Specialization
    ConvSpecial bwd_weight_specialization;

    ElementwiseOps elementwise_ops;
};

static_assert(ckb::factory::BwdDlAlgorithm<DlBwdWeightAlgorithm>);

template <std::size_t NumDTensor = 0>
using DlBwdWeightSignature = ConvSignature<NumDTensor>;

template <std::size_t NumDTensor = 0>
struct DlBwdWeightInstance
{
    DlBwdWeightSignature<NumDTensor> signature;
    DlBwdWeightAlgorithm algorithm;
};

// Constexpr function to create DlBwdWeightInstance from
// DeviceGroupedConvBwdWeight_Dl template parameters.
// Parameters follow the same order as the CK template.
template <std::size_t NumDTensor>
constexpr DlBwdWeightInstance<NumDTensor> DeviceGroupedConvBwdWeightDl(
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
    // 13. Block size
    std::size_t block_size,
    // 14-16. Block dimensions
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k0_per_block,
    // 17. K1
    std::size_t k1,
    // 18-20. Thread parameters
    std::size_t m1_per_thread,
    std::size_t n1_per_thread,
    std::size_t k_per_thread,
    // 21-22. Thread cluster
    std::array<std::size_t, 2> m1n1_thread_cluster_m1_xs,
    std::array<std::size_t, 2> m1n1_thread_cluster_n1_xs,
    // 23-29. A Block Transfer (5D: KBatch_K0_M0_M1_K1)
    std::array<std::size_t, 5> a_block_transfer_thread_slice_lengths,
    std::array<std::size_t, 5> a_block_transfer_thread_cluster_lengths,
    std::array<std::size_t, 5> a_block_transfer_thread_cluster_arrange_order,
    std::array<std::size_t, 5> a_block_transfer_src_access_order,
    std::array<std::size_t, 5> a_block_transfer_src_vector_tensor_lengths,
    std::array<std::size_t, 5> a_block_transfer_src_vector_tensor_contiguous_dim_order,
    std::array<std::size_t, 5> a_block_transfer_dst_vector_tensor_lengths,
    // 30-36. B Block Transfer (5D: KBatch_K0_N0_N1_K1)
    std::array<std::size_t, 5> b_block_transfer_thread_slice_lengths,
    std::array<std::size_t, 5> b_block_transfer_thread_cluster_lengths,
    std::array<std::size_t, 5> b_block_transfer_thread_cluster_arrange_order,
    std::array<std::size_t, 5> b_block_transfer_src_access_order,
    std::array<std::size_t, 5> b_block_transfer_src_vector_tensor_lengths,
    std::array<std::size_t, 5> b_block_transfer_src_vector_tensor_contiguous_dim_order,
    std::array<std::size_t, 5> b_block_transfer_dst_vector_tensor_lengths,
    // 37-39. C Thread Transfer
    std::array<std::size_t, 6> c_thread_transfer_src_dst_access_order,
    std::size_t c_thread_transfer_src_dst_vector_dim,
    std::size_t c_thread_transfer_dst_scalar_per_vector)
{
    // Our project auto-formatting makes this initializer hard to read
    // clang-format off
    return DlBwdWeightInstance<NumDTensor>{
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
            .thread_config = {
                .k0_per_block  = k0_per_block,
                .k1            = k1,
                .m1_per_thread = m1_per_thread,
                .n1_per_thread = n1_per_thread,
                .k_per_thread  = k_per_thread
            },
            .thread_cluster = {
                .m1_xs = m1n1_thread_cluster_m1_xs,
                .n1_xs = m1n1_thread_cluster_n1_xs
            },
            .transfer = {
                .a = {
                    .thread_slice_lengths                    = a_block_transfer_thread_slice_lengths,
                    .thread_cluster_lengths                  = a_block_transfer_thread_cluster_lengths,
                    .thread_cluster_arrange_order            = a_block_transfer_thread_cluster_arrange_order,
                    .src_access_order                        = a_block_transfer_src_access_order,
                    .src_vector_tensor_lengths               = a_block_transfer_src_vector_tensor_lengths,
                    .src_vector_tensor_contiguous_dim_order  = a_block_transfer_src_vector_tensor_contiguous_dim_order,
                    .dst_vector_tensor_lengths               = a_block_transfer_dst_vector_tensor_lengths
                },
                .b = {
                    .thread_slice_lengths                    = b_block_transfer_thread_slice_lengths,
                    .thread_cluster_lengths                  = b_block_transfer_thread_cluster_lengths,
                    .thread_cluster_arrange_order            = b_block_transfer_thread_cluster_arrange_order,
                    .src_access_order                        = b_block_transfer_src_access_order,
                    .src_vector_tensor_lengths               = b_block_transfer_src_vector_tensor_lengths,
                    .src_vector_tensor_contiguous_dim_order  = b_block_transfer_src_vector_tensor_contiguous_dim_order,
                    .dst_vector_tensor_lengths               = b_block_transfer_dst_vector_tensor_lengths
                },
                .c = {
                    .src_dst_access_order     = c_thread_transfer_src_dst_access_order,
                    .src_dst_vector_dim       = c_thread_transfer_src_dst_vector_dim,
                    .dst_scalar_per_vector    = c_thread_transfer_dst_scalar_per_vector
                }
            },
            .bwd_weight_specialization = conv_bwd_weight_specialization,
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
