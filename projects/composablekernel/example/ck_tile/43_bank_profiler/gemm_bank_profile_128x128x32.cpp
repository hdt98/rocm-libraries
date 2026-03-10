// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Tile config: 128x128x32

#include <hip/hip_runtime.h>
#include <iostream>
#include <string>

#include "ck_tile/host.hpp"
#include "gemm_bank_profile.hpp"

using ADataType   = ck_tile::half_t;
using BDataType   = ck_tile::half_t;
using AccDataType = float;
using CDataType   = ck_tile::half_t;
using GemmConfig  = Config_128x128x32;

template <typename ALayout, typename BLayout>
int run_with_layouts(ck_tile::ArgParser& arg_parser)
{
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    int n_warmup = arg_parser.get_int("warmup");
    int n_repeat = arg_parser.get_int("repeat");

    auto f_default_stride = [](std::size_t row, std::size_t col, std::size_t stride, auto layout) -> std::size_t {
        if(stride == 0)
        {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
                return col;
            else
                return row;
        }
        return stride;
    };

    stride_A = f_default_stride(M, K, stride_A, ALayout{});
    stride_B = f_default_stride(K, N, stride_B, BLayout{});
    stride_C = f_default_stride(M, N, stride_C, CLayout{});

    auto f_desc = [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
        if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            return ck_tile::HostTensorDescriptor({row, col}, {stride, std::size_t{1}});
        else
            return ck_tile::HostTensorDescriptor({row, col}, {std::size_t{1}, stride});
    };

    ck_tile::HostTensor<ADataType> a_m_k(f_desc(M, K, stride_A, ALayout{}));
    ck_tile::HostTensor<BDataType> b_k_n(f_desc(K, N, stride_B, BLayout{}));
    ck_tile::HostTensor<CDataType> c_m_n(f_desc(M, N, stride_C, CLayout{}));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

    ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_buf(c_m_n.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_m_k.data());
    b_buf.ToDevice(b_k_n.data());
    c_buf.SetZero();

    ck_tile::GemmHostArgs args;
    args.a_ptr    = a_buf.GetDeviceBuffer();
    args.b_ptr    = b_buf.GetDeviceBuffer();
    args.e_ptr    = c_buf.GetDeviceBuffer();
    args.M        = M;
    args.N        = N;
    args.K        = K;
    args.stride_A = stride_A;
    args.stride_B = stride_B;
    args.stride_C = stride_C;
    args.k_batch  = 1;

    ck_tile::stream_config s{nullptr, true, 1, n_warmup, n_repeat};

    float ave_time = BankProfileInvoker::run<GemmConfig, ADataType, BDataType, AccDataType,
                                             ALayout, BLayout, CLayout>(args, s);

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * N * K +
                           sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "M=" << M << " N=" << N << " K=" << K << " : " << ave_time << " ms, " << tflops
              << " TFlops, " << gb_per_sec << " GB/s" << std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A layout (R/C)")
        .insert("b_layout", "C", "B layout (R/C)")
        .insert("stride_a", "0", "A stride")
        .insert("stride_b", "0", "B stride")
        .insert("stride_c", "0", "C stride")
        .insert("warmup", "1", "warmup iterations")
        .insert("repeat", "1", "benchmark iterations");

    if(!arg_parser.parse(argc, argv))
        return 1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "R")
        return run_with_layouts<Row, Row>(arg_parser);
    else if(a_layout == "R" && b_layout == "C")
        return run_with_layouts<Row, Col>(arg_parser);
    else if(a_layout == "C" && b_layout == "R")
        return run_with_layouts<Col, Row>(arg_parser);
    else if(a_layout == "C" && b_layout == "C")
        return run_with_layouts<Col, Col>(arg_parser);
    else
        throw std::runtime_error("Unsupported layout!");
}
