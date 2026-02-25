// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/instance_data/xdl_bwd_weight.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

// Multi-D XDL backward weight algorithm descriptor.
// Similar to XdlBwdWeightAlgorithm but satisfies BwdMultiDXdlAlgorithm instead of
// BwdXdlAlgorithm. The key difference is:
//   - Has static constexpr specialization = ConvAlgorithmSpecialization::MULTIPLE_D
//   - Does NOT have max_transpose_transfer_* fields
// This routes through ConvBwdWeightMultiDXdlFactory and creates
// DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle instances.
struct XdlMultiDBwdWeightAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;

    ThreadBlock thread_block;

    struct GridwiseGemm
    {
        std::size_t k1;
        struct XdlParams
        {
            std::size_t m_per_xdl      = 16;
            std::size_t n_per_xdl      = 16;
            std::size_t m_xdl_per_wave = 4;
            std::size_t n_xdl_per_wave = 1;
        } xdl_params;
        static_assert(ckb::GridwiseXdlGemmDescriptor<XdlParams>);
    } gridwise_gemm;

    static_assert(ckb::GridwiseBwdXdlGemmDescriptor<GridwiseGemm>);

    // Multi-D XDL uses 4D block transfers (same as regular XDL bwd weight)
    BwdWeightTransferABC transfer;

    ConvSpecial bwd_weight_specialization;

    static constexpr auto specialization = ckb::ConvAlgorithmSpecialization::MULTIPLE_D;

    ElementwiseOps elementwise_ops;
};

static_assert(ckb::factory::BwdMultiDXdlAlgorithm<XdlMultiDBwdWeightAlgorithm>);

// Multi-D backward weight signature: the weight tensor carries the Bilinear/Scale
// elementwise operation (since weight is the "output" of the backward GEMM), while
// the output tensor carries the auxiliary (D) tensor configs (since the CK Builder
// framework reads DsLayout/DsDataType from SIGNATURE.output.operation).
template <std::size_t NumDTensor = 0>
struct ConvMultiDBwdWeightSignature
{
    std::size_t spatial_dim;
    ckb::ConvDirection direction;
    TensorDescriptor input;
    // Weight with elementwise op (Bilinear/Scale) — maps to WeiElementwiseOp
    OutputTensorDescriptor<0> weight;
    // Output with aux D tensor configs — maps to DsLayout/DsDataType
    OutputTensorDescriptor<NumDTensor> output;
    ckb::DataType data_type;
    ckb::DataType accumulation_data_type;
};

template <std::size_t NumDTensor = 0>
using XdlMultiDBwdWeightSignature = ConvMultiDBwdWeightSignature<NumDTensor>;

template <std::size_t NumDTensor = 0>
struct XdlMultiDBwdWeightInstance
{
    XdlMultiDBwdWeightSignature<NumDTensor> signature;
    XdlMultiDBwdWeightAlgorithm algorithm;
};

// Helper to create an XdlMultiDBwdWeightInstance from the CK
// DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle template parameters.
// Parameters follow the same order as the regular XDL backward weight "Generic" helper.
template <std::size_t NumDTensor>
constexpr XdlMultiDBwdWeightInstance<NumDTensor> DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle(
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
    // 17. K1
    std::size_t k1,
    // 18-21. XDL parameters
    std::size_t m_per_xdl,
    std::size_t n_per_xdl,
    std::size_t m_xdl_per_wave,
    std::size_t n_xdl_per_wave,
    // 22-28. A block transfer (4D: K0_M_K1_KBatch)
    std::array<std::size_t, 4> a_thread_cluster_lengths,
    std::array<std::size_t, 4> a_thread_cluster_arrange_order,
    std::array<std::size_t, 4> a_block_transfer_src_access_order,
    std::size_t a_block_transfer_src_vector_dim,
    std::size_t a_block_transfer_src_scalar_per_vector,
    std::size_t a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    // 29-35. B block transfer (4D: K0_N_K1_KBatch)
    std::array<std::size_t, 4> b_thread_cluster_lengths,
    std::array<std::size_t, 4> b_thread_cluster_arrange_order,
    std::array<std::size_t, 4> b_block_transfer_src_access_order,
    std::size_t b_block_transfer_src_vector_dim,
    std::size_t b_block_transfer_src_scalar_per_vector,
    std::size_t b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    // 36-39. C shuffle
    std::size_t c_shuffle_m_xdl_per_wave_per_shuffle,
    std::size_t c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    std::size_t c_block_transfer_scalar_per_vector)
{
    // For bilinear (NumDTensor=1), the D tensor uses the weight layout and data type.
    // For scale (NumDTensor=0), this produces an empty array.
    constexpr auto make_d_configs = [](ckb::TensorLayout wl, ckb::DataType wdt) {
        std::array<AuxTensorConfig, NumDTensor> configs{};
        for(std::size_t i = 0; i < NumDTensor; ++i)
            configs[i] = AuxTensorConfig{wl, wdt};
        return configs;
    };
    // clang-format off
    return XdlMultiDBwdWeightInstance<NumDTensor>{
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
                },
                .operation = {
                    .elementwise_operation    = wei_elementwise_op,
                    .auxiliary_operand_configs = {}
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
                    .auxiliary_operand_configs = make_d_configs(wei_layout, wei_data_type)
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
                .k1         = k1,
                .xdl_params = {
                    .m_per_xdl      = m_per_xdl,
                    .n_per_xdl      = n_per_xdl,
                    .m_xdl_per_wave = m_xdl_per_wave,
                    .n_xdl_per_wave = n_xdl_per_wave
                }
            },
            .transfer = {
                .a = {
                    .block_transfer = {
                        .k0           = a_thread_cluster_lengths[1],
                        .m_n          = a_thread_cluster_lengths[2],
                        .k1           = a_thread_cluster_lengths[3],
                        .k_batch_size = a_thread_cluster_lengths[0]
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
                        .k0           = b_thread_cluster_lengths[1],
                        .m_n          = b_thread_cluster_lengths[2],
                        .k1           = b_thread_cluster_lengths[3],
                        .k_batch_size = b_thread_cluster_lengths[0]
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
