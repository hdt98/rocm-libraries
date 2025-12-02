// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

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