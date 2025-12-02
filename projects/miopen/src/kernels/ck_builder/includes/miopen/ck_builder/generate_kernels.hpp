// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

namespace ckb      = ck_tile::builder;
namespace ckc      = ck::tensor_layout::convolution;
using BaseOperator = ck::tensor_operation::device::BaseOperator;

template <typename T>
concept ConvLayout2DConcept = std::is_convertible_v<T, ckc::BaseConvolutionLayout>;

template <typename InLayout, typename WeiLayout, typename OutLayout>
requires ConvLayout2DConcept<InLayout> && ConvLayout2DConcept<WeiLayout> &&
    ConvLayout2DConcept<OutLayout>
constexpr auto GroupConvLayout2DHelper()
{
    static_assert(false, "Invalid combination of layouts.");
}

template <>
constexpr auto GroupConvLayout2DHelper<ckc::GNHWC, ckc::GKYXC, ckc::GNHWK>()
{
    return ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
}

template <>
constexpr auto GroupConvLayout2DHelper<ckc::NHWGC, ckc::GKYXC, ckc::NHWGK>()
{
    return ckb::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
}

template <>
constexpr auto GroupConvLayout2DHelper<ckc::NGCHW, ckc::GKYXC, ckc::NGKHW>()
{
    return ckb::GroupConvLayout2D::NGCHW_GKYXC_NGKHW;
}

template <>
constexpr auto GroupConvLayout2DHelper<ckc::NGCHW, ckc::GKCYX, ckc::NGKHW>()
{
    return ckb::GroupConvLayout2D::NGCHW_GKCYX_NGKHW;
}

template <typename T, std::size_t N, std::size_t ShardCount, std::size_t ShardIndex>
constexpr std::array<T, ((N - ShardIndex - 1) / ShardCount) + 1>
shard_array(const std::array<T, N>& arr)
{
    std::array<T, ((N - ShardIndex - 1) / ShardCount) + 1> shard{};
    for(auto i = 0; i < shard.size(); i++)
    {
        auto index = i * ShardCount + ShardIndex;
        shard[i]   = arr[index];
    }

    return shard;
}

/**
 * This function transforms an input array into an output array by applying func to each element of
 * the array. output[i] = func(input[i])
 */
template <typename T, std::size_t N, typename F>
constexpr auto map_array(const std::array<T, N>& input, F&& func)
{
    using U = std::invoke_result_t<F, T>;
    std::array<U, N> result{};
    for(auto i = 0; i < N; i++)
    {
        result[i] = func(input[i]);
    }

    return result;
}

/**
 * For every pair of elements taken from input1 and input2, apply func, yielding a new element. All
 * of these elements are returned in the resulting array. This array will be N1 * N2 elements in
 * size.
 */
template <typename T1, std::size_t N1, typename T2, std::size_t N2, typename F>
constexpr auto
multiplex_array(const std::array<T1, N1>& input1, const std::array<T2, N2>& input2, F&& func)
{
    using U = std::invoke_result_t<F, T1, T2>;
    std::array<U, N1 * N2> result{};
    for(auto i1 = 0; i1 < N1; i1++)
    {
        auto arg1 = input1[i1];
        for(auto i2 = 0; i2 < N2; i2++)
        {
            auto arg2            = input2[i2];
            auto retval          = func(arg1, arg2);
            result[i1 * N2 + i2] = retval;
        }
    }

    return result;
}

/**
 * Container concept for a Convolution Kernel Descriptor
 */
template <typename T>
concept ConvKernelDescriptor = requires(T t)
{
    {
        t.Signature
        } -> ckb::ConvSignatureDescriptor;
    {
        t.Algorithm
        } -> ckb::ConvAlgorithmDescriptor;
};

/**
 * Instantiates the kernel specified by the non-type template parameter KernelDescriptor
 */
template <auto K>
requires ConvKernelDescriptor<decltype(K)>
constexpr void instantiate_kernel(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = typename ckb::ConvBuilder<K.Signature, K.Algorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);
    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

template <typename T, T... values>
requires ConvKernelDescriptor<T>
constexpr void build_kernels_helper(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    std::array<std::unique_ptr<BaseOperator>, sizeof...(values)> result{};
    ((InstantiateKernel<values>(kernels)), ...);
}

template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
requires ConvKernelDescriptor<T>
constexpr void build_kernels_impl(std::vector<std::unique_ptr<BaseOperator>>& kernels,
                                  std::index_sequence<I...>)
{
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename T, std::size_t N, std::array<T, N> arr>
requires ConvKernelDescriptor<T>
constexpr void build_kernels(std::vector<std::unique_ptr<BaseOperator>>& kernels)
{
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}

namespace miopen {
namespace kernels {
namespace ck_builder {

template <typename DeviceOp>
struct DeviceOperationInstanceFactory
{
    static auto GetInstances()
    {
        static_assert(false, "DeviceOperationInstanceFactory is not valid for the given DeviceOp.");
    }
};

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename InLayout     = ck::tensor_layout::convolution::NHWGC,
          typename WeiLayout    = ck::tensor_layout::convolution::GKYXC,
          typename OutLayout    = ck::tensor_layout::convolution::NHWGK>
using DeviceOpGFwdAct =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>, // diff
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  ck::Tuple<>, // diff
                                                                  OutDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  AComputeType,
                                                                  BComputeType>;

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType,
          typename BComputeType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
using DeviceOpGFwdAct =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>, // diff
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  ck::Tuple<>, // diff
                                                                  OutDataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  AComputeType,
                                                                  BComputeType>;

template <ck::index_t NumDimSpatial,
          typename DataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
struct DeviceOperationInstanceFactory<DeviceOpGFwdAct<NumDimSpatial,
                                                      DataType,
                                                      DataType,
                                                      DataType,
                                                      DataType,
                                                      DataType,
                                                      InLayout,
                                                      WeiLayout,
                                                      OutLayout>>
{
    using DeviceOp = DeviceOpGFwdAct<NumDimSpatial,
                                     DataType,
                                     DataType,
                                     DataType,
                                     DataType,
                                     DataType,
                                     InLayout,
                                     WeiLayout,
                                     OutLayout>;
    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        constexpr auto layout = GroupConvLayout2DHelper<InLayout, WeiLayout, OutLayout>();
        std::cout << layout << std::endl;
        return op_ptrs;
    }
};

} // namespace ck_builder
} // namespace kernels
} // namespace miopen
