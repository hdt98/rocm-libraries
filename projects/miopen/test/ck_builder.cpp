// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"

#include "utils/constexpr_data_processing.hpp"

template <typename OpNeedle, typename OpHaystack>
int FindInVector(const OpNeedle& opNeedle, const std::vector<OpHaystack>& haystack)
{
    int foundCount    = 0;
    auto needleString = opNeedle->GetInstanceString();
    for(auto&& h : haystack)
    {
        auto haystackString = h->GetInstanceString();
        if(needleString == haystackString)
        {
            foundCount++;
        }
    }

    return foundCount;
}

std::optional<std::size_t> GetFirstNonMatchingIndex(const auto& a, const auto& b)
{
    auto aSize = a.size();
    auto bSize = b.size();
    auto size  = min(aSize, bSize);

    for(std::size_t i = 0; i < size; i++)
    {
        if(a[i] != b[i])
        {
            return std::optional{i};
        }
    }

    if(aSize == bSize)
    {
        return {};
    }

    return std::optional{size};
}

void CompareAndPrint(const auto& a, const auto& b)
{
    auto comparison = GetFirstNonMatchingIndex(a, b);
    if(comparison.has_value())
    {
        auto value = *comparison;
        std::cout << a << std::endl;
        std::cout << b << std::endl;

        while(value > 0)
        {
            std::cout << " ";
            value--;
        }

        std::cout << "^" << std::endl;
    }
    else
    {
        std::cout << "!!! " << a << " matches!" << std::endl;
    }
}

namespace ckb      = ck_tile::builder;
using BaseOperator = ck::tensor_operation::device::BaseOperator;

constexpr std::array types = {
    ckb::DataType::FP32,
    // ckb::DataType::FP16,
    // ckb::DataType::BF16,
    // ckb::DataType::FP8,
    // ckb::DataType::I8,
};

using BaseOperator = ck::tensor_operation::device::BaseOperator;

template <auto KernelDescriptor>
constexpr void InstantiateKernel(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<KernelDescriptor.Signature, KernelDescriptor.Algorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);
    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

template <typename T, T... values>
constexpr void build_kernels_helper(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    std::array<std::unique_ptr<BaseOperator>, sizeof...(values)> result{};
    ((InstantiateKernel<values>(kernels)), ...);
}

template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<std::unique_ptr<BaseOperator>>& kernels,
                                  std::index_sequence<I...>)
{
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename T, std::size_t N, std::array<T, N> arr>
constexpr void build_kernels(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}

struct DefaultAlgorithm
{
    using ConvSpecial = ckb::ConvFwdSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeVers    = ckb::PipelineVersion;
    using PipeSched   = ckb::PipelineScheduler;

    struct ThreadBlock
    {
        int block_size = 128;
        struct TileSize
        {
            int m = 32;
            int n = 64;
            int k = 64;
        } tile_size;
    } thread_block;

    struct GridwiseGemm
    {
        int ak1            = 8;
        int bk1            = 8;
        int m_per_xdl      = 16;
        int n_per_xdl      = 16;
        int m_xdl_per_wave = 1;
        int n_xdl_per_wave = 4;
    } gridwise_gemm;

    struct TransferABC
    {
        struct TransferAB
        {
            struct BlockTransfer
            {
                int k0  = 8;
                int m_n = 16;
                int k1  = 1;
            } block_transfer;
            struct LdsTransfer
            {
                int src_vector_dim            = 2;
                int src_scalar_per_vector     = 8;
                int lds_dst_scalar_per_vector = 8;
                bool is_direct_load           = true;
                bool lds_padding              = false;
            } lds_transfer;
            struct BlockTransferAccessOrder
            {
                std::array<size_t, 3> order{0, 1, 2};
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
                int m_block        = 1;
                int m_wave_per_xdl = 32;
                int n_block        = 1;
                int n_wave_per_xdl = 8;
            } thread_cluster_dims;
            struct Epilogue
            {
                int m_per_wave_per_shuffle = 1;
                int n_per_wave_per_shuffle = 1;
                int scalar_per_vector      = 8;
            } epilogue;
        } c;
    } transfer;

    // TODO: Fix CK Builder schema to not require these defaults.
    ConvSpecial fwd_specialization  = ConvSpecial::DEFAULT;
    GemmSpecial gemm_specialization = GemmSpecial::MNKPadding;
    struct BlockGemm
    {
        PipeVers pipeline_version = PipeVers::V4;
        PipeSched scheduler       = PipeSched::INTRAWAVE;
    } block_gemm;
};

// Define a struct to specify the signature
struct Signature
{
    int spatial_dim = 2;
    // TODO: This direction should be OK as default, but the factory fails.
    ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout2D::NGCHW_GKCYX_NGKHW;
    ckb::DataType data_type      = ckb::DataType::FP32;
};

struct KernelArguments
{
    Signature Signature;
    DefaultAlgorithm Algorithm;
};

using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
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

int main()
{
    // Verify that the signature structure conforms to the signature concept.
    static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // Specify the signature in a constexpr value
    constexpr Signature kSignature{};
    // Verify the signature value is valid
    static_assert(ckb::ValidConvSignature<kSignature>);

    // Define a struct to specify the algorithm.
    // TODO improve CK Builder schema to reduce duplication and simplify.

    //     // Verify that the signature conforms to the expected descriptor
    // static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // // Specify the signature in a constexpr value
    // constexpr Signature kSignature{};
    // // Verify the signature value is valid
    // static_assert(ckb::ValidConvSignature<kSignature>);
    // Verify that the algorithm conforms to the algorithm concept
    static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

    // TODO: Verify the algorithm value is valid.

    constexpr std::array sizes = {
        256,
        /* 128, 64, 32 */
    };
    constexpr std::array xdl_per_wave = {1};

    constexpr auto algorithms = map_array(sizes, [](int size) {
        DefaultAlgorithm algorithm;
        algorithm.thread_block.block_size = size;
        return algorithm;
    });

    constexpr auto algorithms2 =
        multiplex_array(algorithms, sizes, [](DefaultAlgorithm d, int size) {
            d.thread_block.tile_size.n = size;
            return d;
        });

    constexpr auto algorithms3 =
        multiplex_array(algorithms2, xdl_per_wave, [](DefaultAlgorithm d, int size) {
            d.gridwise_gemm.m_xdl_per_wave = d.gridwise_gemm.n_xdl_per_wave = size;
            return d;
        });

    struct BuilderParameters
    {
        DefaultAlgorithm Algorithm;
        Signature Signature;
    };

    static constexpr auto arr = std::array<Signature, 1>{kSignature};
    static constexpr auto typesAdded =
        multiplex_array(arr, types, [](Signature s, ckb::DataType dt) {
            s.data_type = dt;
            return s;
        });

    static constexpr auto parameters = multiplex_array(
        algorithms3, typesAdded, [](DefaultAlgorithm algorithm, Signature signature) {
            BuilderParameters params{
                .Algorithm = algorithm,
                .Signature = signature,
            };

            return params;
        });

    std::cout << std::endl;

    // /*
    std::vector<std::unique_ptr<BaseOperator>> kernels{};
    build_kernels<BuilderParameters, parameters.size(), parameters>(kernels);

    for(auto&& kernel : kernels)
    {
        std::cout << kernel->GetInstanceString() << std::endl;
    }

    std::cout << std::endl << "Kernel count: " << kernels.size() << std::endl;
    // */

    // /*
    auto instances = DeviceOpGFwdDefaultPtrs<float>::GetInstances();
    std::cout << std::endl << "Pre-built instance count: " << instances.size() << std::endl;
    // */

    /*
    using Empty_Tuple = ck::Tuple<>;
    using GNHWC       = ck::tensor_layout::convolution::GNHWC;
    using GKYXC       = ck::tensor_layout::convolution::GKYXC;
    using GNHWK       = ck::tensor_layout::convolution::GNHWK;
    using F32         = float;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    std::vector<
        std::unique_ptr<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<2,
                                                                                      GNHWC,
                                                                                      GKYXC,
                                                                                      Empty_Tuple,
                                                                                      GNHWK,
                                                                                      F32,
                                                                                      F32,
                                                                                      Empty_Tuple,
                                                                                      F32,
                                                                                      PassThrough,
                                                                                      PassThrough,
                                                                                      PassThrough>>>
        instances{};

    ck::tensor_operation::device::instance::
        add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instances(instances);
    std::cout << std::endl << "Precompiled instance count: " << instances.size() << std::endl;

    // */

    /*
    for(auto&& myInstance : kernels)
    {
        auto count = FindInVector(myInstance, instances);
        std::cout << myInstance->GetInstanceString() << std::endl
                  << "\toccurs: " << count << std::endl;
    }
                  */

    auto firstKernelInstanceString = kernels[0]->GetInstanceString();

    for(auto&& instance : instances)
    {
        auto instanceString = instance->GetInstanceString();
        if(!instanceString.starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"))
        {
            continue;
        }

        CompareAndPrint(firstKernelInstanceString, instanceString);
    }
    return 0;
}
