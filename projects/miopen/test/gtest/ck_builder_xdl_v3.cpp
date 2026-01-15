// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

namespace ckb      = ck_tile::builder;
using BaseOperator = ck::tensor_operation::device::BaseOperator;
struct DefaultAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeVers    = ckb::PipelineVersion;
    using PipeSched   = ckb::PipelineScheduler;

    struct ThreadBlock
    {
        unsigned int block_size = 128;
        struct TileSize
        {
            unsigned int m = 32;
            unsigned int n = 64;
            unsigned int k = 64;
        } tile_size;
    } thread_block;

    static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);
    struct GridwiseGemm
    {
        unsigned int ak1 = 8;
        unsigned int bk1 = 8;
        struct XdlParams
        {
            unsigned int m_per_xdl      = 32;
            unsigned int n_per_xdl      = 32;
            unsigned int m_xdl_per_wave = 1;
            unsigned int n_xdl_per_wave = 1;
        } xdl_params;

        static_assert(ckb::GridwiseXdlGemmDescriptor<XdlParams>);
    } gridwise_gemm;

    static_assert(ckb::GridwiseFwdXdlGemmDescriptor<GridwiseGemm>);
    struct TransferABC
    {
        struct TransferAB
        {
            struct BlockTransfer
            {
                unsigned int k0  = 8;
                unsigned int m_n = 16;
                unsigned int k1  = 1;
            } block_transfer;
            struct LdsTransfer
            {
                unsigned int src_vector_dim            = 2;
                unsigned int src_scalar_per_vector     = 4;
                unsigned int lds_dst_scalar_per_vector = 4;
                bool is_direct_load                    = false;
                bool lds_padding                       = false;
            } lds_transfer;
            struct BlockTransferAccessOrder
            {
                std::array<size_t, 3> order{1, 0, 2};
            } block_transfer_access_order;
            struct SrcAccessOrder
            {
                std::array<size_t, 3> order{1, 0, 2};
            } src_access_order;
        };
        TransferAB a;
        TransferAB b;
        struct TransferC
        {
            struct ThreadClusterDims
            {
                unsigned int m_block        = 1;
                unsigned int m_wave_per_xdl = 16;
                unsigned int n_block        = 1;
                unsigned int n_wave_per_xdl = 8;
            } thread_cluster_dims;
            struct Epilogue
            {
                unsigned int m_xdl_per_wave_per_shuffle = 1;
                unsigned int n_per_wave_per_shuffle     = 1;
                unsigned int scalar_per_vector          = 8;
            } epilogue;
        } c;
    } transfer;

    // TODO: Fix CK Builder schema to not require these defaults.
    ConvSpecial fwd_specialization  = ConvSpecial::DEFAULT;
    GemmSpecial gemm_specialization = GemmSpecial::MNKPadding;
    struct BlockGemm
    {
        PipeVers pipeline_version = PipeVers::V2;
        PipeSched scheduler       = PipeSched::INTRAWAVE;
    } block_gemm_pipeline;
};

struct Signature
{
    int spatial_dim              = 2;
    ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
    struct InputTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout   = ckb::TensorLayout::NGCHW;
            ckb::DataType data_type    = ckb::DataType::FP32;
            ckb::DataType compute_type = ckb::DataType::FP32;
        } config;
    } input;

    struct WeightTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout   = ckb::TensorLayout::GKCYX;
            ckb::DataType data_type    = ckb::DataType::FP32;
            ckb::DataType compute_type = ckb::DataType::FP32;
        } config;
    } weight;

    struct OutputTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout   = ckb::TensorLayout::NGKHW;
            ckb::DataType data_type    = ckb::DataType::FP32;
            ckb::DataType compute_type = ckb::DataType::FP32;
        } config;
    } output;
    ckb::DataType data_type              = ckb::DataType::FP32;
    ckb::DataType accumulation_data_type = ckb::DataType::FP32;
};

using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;
template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;
template <typename DataType>
using DeviceOpGFwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdDefault<DataType>>;

TEST(CKBuilderXdlV3, CreateExistingInstance)
{
    // Verify that the signature structure conforms to the signature concept.
    static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // Specify the signature in a constexpr value
    constexpr Signature kSignature{};
    // Verify the signature value is valid
    static_assert(ckb::ValidConvSignature<kSignature>);

    // Verify that the algorithm conforms to the algorithm concept
    static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);
    constexpr DefaultAlgorithm kAlgorithm{};

    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<kSignature, kAlgorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    static_assert(ckb::factory::FwdXdlV3Algorithm<DefaultAlgorithm>);

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);

    auto builderKernelInstance       = Builder::Instance{};
    auto builderKernelInstanceString = builderKernelInstance.GetInstanceString();

    // These are the instances that MIOpen currently gets from CK's static library
    auto factoryInstances = DeviceOpGFwdDefaultPtrs<float>::GetInstances();

    ASSERT_GT(factoryInstances.size(), 0);

    auto result =
        std::find_if(factoryInstances.begin(),
                     factoryInstances.end(),
                     [&builderKernelInstanceString](const auto& kernelPtr) {
                         return kernelPtr->GetInstanceString() == builderKernelInstanceString;
                     });

    ASSERT_NE(result, factoryInstances.end());
}
