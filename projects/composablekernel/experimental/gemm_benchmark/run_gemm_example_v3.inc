// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

static example_gemm_xdl_v3_benchmark_instances op_ptrs;

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1 || stride == 0)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 1:
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    }

    Tensor<float> a_m_k_f32({M, K});
    Tensor<float> b_k_n_f32({K, N});

    if constexpr(ck::is_same_v<ADataType, ck::pk_i4_t>)
    {
        permute_a_pk_i4(a_m_k, M, K, a_m_k_f32, config.use_gfx9_i4);
    }

    if constexpr(ck::is_same_v<BDataType, ck::pk_i4_t>)
    {
        permute_b_pk_i4(b_k_n, N, K, b_k_n_f32, config.use_gfx9_i4);
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem a_m_k_device_buf(sizeof(KernelADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    const Tensor<KernelADataType> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    bool pass = true;
    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        if constexpr(ck::is_same_v<ADataType, ck::pk_i4_t> && ck::is_same_v<BDataType, ck::pk_i4_t>)
        {
            using PkInt4ReferenceGemmInstance =
                ck::tensor_operation::host::ReferenceGemm<float,
                                                          float,
                                                          CDataType,
                                                          AccDataType,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>;
            auto ref_gemm    = PkInt4ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_m_k_f32,
                                                      b_k_n_f32,
                                                      c_m_n_host_result,
                                                      PassThrough{},
                                                      PassThrough{},
                                                      PassThrough{});

            ref_invoker.Run(ref_argument);
        }
        else
        {
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

            ref_invoker.Run(ref_argument);
        }
    }
    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;
    // do GEMM
    std::string best_op_name;
    std::optional<std::string> best_op_object_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;
    size_t best_index     = 0;
    for(size_t i = 0; i < op_ptrs.size(); i++)
    {
        auto& gemm       = op_ptrs[i];
        auto invoker_ptr = gemm->MakeInvokerPointer();
        float ave_time   = 0;

        auto argument_ptr = gemm->MakeArgumentPointer(
#ifdef BUILD_INT4_EXAMPLE
            static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
            static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            KBatch,
            a_element_op,
            b_element_op,
            c_element_op);

        if(config.instance_index != -1 && config.instance_index != static_cast<int>(i))
        {
            continue;
        }

        if(!gemm->IsSupportedArgument(argument_ptr.get()))
        {
            std::cerr << gemm->GetTypeString() << " does not support this problem" << std::endl;
            continue;
        }

        if(config.do_verification)
        {
            ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 1});
#ifdef BUILD_INT4_EXAMPLE
            Tensor<CDataType> c_m_n_device_result_converted(c_m_n_host_result.mDesc);

            c_m_n_device_buf.FromDevice(c_m_n_device_result_converted.mData.data());

            c_m_n_device_result = c_m_n_device_result_converted.CopyAsType<CDataType>();

            return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
#else
            c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

            bool ret = ck::utils::check_err(c_m_n_device_result,
                                            c_m_n_host_result,
                                            "Error: Incorrect results!",
                                            get_rtol<CDataType>(),
                                            get_atol<CDataType>());
            pass &= ret;
            if(!ret)
            {
                std::cout << "Error: [" << i << "]: " << gemm->GetTypeString()
                          << " results incorrect!" << std::endl;
            }
#endif
        }

        if(config.time_kernel)
        {
            ave_time = invoker_ptr->Run(argument_ptr.get(),
                                        StreamConfig{nullptr,
                                                     config.time_kernel,
                                                     0,
                                                     config.cold_niters,
                                                     config.nrepeat,
                                                     true,
                                                     config.rotating_count});

            std::size_t flop = 2_uz * M * N * K;
            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << " [" << i << "] " << gemm->GetTypeString() << std::endl;
            if(tflops > best_tflops && ave_time > 1e-10)
            {
                std::string op_name                    = gemm->GetTypeString();
                std::optional<std::string> op_obj_name = gemm->GetObjectName();

                best_op_name        = op_name;
                best_op_object_name = op_obj_name;
                best_tflops         = tflops;
                best_ave_time       = ave_time;
                best_gb_per_sec     = gb_per_sec;
                best_kbatch         = KBatch;
                best_index          = i;
            }
        }
    }

    if(config.time_kernel)
    {
        std::cout << "Best Perf for M = " << M << " N = " << N << " K = " << K
                  << " StrideA = " << StrideA << " StrideB = " << StrideB
                  << " StrideC = " << StrideC << " KBatch = " << best_kbatch << " : "
                  << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
                  << " GB/s, " << " [" << best_index << "] " << best_op_name << std::endl;

        if(best_op_object_name)
            std::cout << best_op_object_name.value() << std::endl;
    }

    return pass;
}

namespace ck::tensor_operation::device::instance {

extern void
add_example_gemm_xdl_v3_benchmark_instances(example_gemm_xdl_v3_benchmark_instances& instances);

bool init_opt_ptrs()
{
    add_example_gemm_xdl_v3_benchmark_instances(op_ptrs);
    return true;
}

} // namespace ck::tensor_operation::device::instance

bool run_gemm_splitk_example(int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !ck::tensor_operation::device::instance::init_opt_ptrs() ||
           !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
