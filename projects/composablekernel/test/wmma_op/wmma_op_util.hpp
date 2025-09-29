// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/utility/amd_wmma.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace wmma_op_util {

#if defined(__gfx125__)

#define CK_WMMA_CALL_INTRIN(dst, src0) \
    intrin_wmma_##dst##_16x16x32_##src0<16, 16>::Run(reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}))

template <typename T, index_t kMultiplier, typename = void>
struct WMMAVecType
{
    static_assert(sizeof(T) == 0, "VecType is not specialized for this type");
};

// fp16 specialization 
template <typename T, index_t kMultiplier>
struct WMMAVecType<T,
                   kMultiplier,
                   ck::enable_if_t<ck::is_same_v<T, ck::half_t> || ck::is_same_v<T, ck::bhalf_t>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 2;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<T, kMultiplier * 16>;
    using ViewT = vector_type<T, 2>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<T,
                   kMultiplier,
                   ck::enable_if_t<ck::is_same_v<T, float>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 1;

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    using VecT  = vector_type<T, kMultiplier * 8>;
    using ViewT = vector_type<T, 1>;
};

template <typename T, index_t kMultiplier>
struct WMMAVecType<T, kMultiplier, ck::enable_if_t<ck::is_same_v<T, ck::f8_t> || ck::is_same_v<T, ck::bf8_t>>>
{
    static constexpr bool layoutTransform = false;
    static constexpr int ToIntDim         = 4; // adjust as needed

    template <typename D>
    constexpr static bool is_compatible()
    {
        return ck::is_same_v<T, D>;
    }

    // For FP8 input, hardware expects kMultiplier * 32 elements per fragment
    using VecT  = vector_type<T, kMultiplier * 32>;
    using ViewT = vector_type<T, 4>;
};

// gfx125 builtin_wmma_naive_selector 
template <typename srcAType, typename srcBType, typename dstType, index_t kMultiplier>
__device__ void builtin_wmma_naive_selector(
    const typename WMMAVecType<srcAType, kMultiplier>::VecT::type& reg_a,
    const typename WMMAVecType<srcBType, kMultiplier>::VecT::type& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, dstType, 1, 8/* * WMMAVecType<dstType, kMultiplier>::ToIntDim*/, true>& reg_c)
{
    // if constexpr(std::is_same_v<srcAType, float>)
    // {
    //     printf("----- srcAType is float \n");
    // }
    // else if constexpr(std::is_same_v<srcAType, ck::half_t>)
    // {
    //     printf("----- srcAType is half \n");
    // }
    //     if constexpr(std::is_same_v<srcBType, float>)
    // {
    //     printf("----- srcBType is float \n");
    // }
    // else if constexpr(std::is_same_v<srcBType, ck::half_t>)
    // {
    //     printf("----- srcBType is half \n");
    // }    
    // if constexpr(std::is_same_v<dstType, float>)
    // {
    //     printf("----- dstType is float \n");
    // }
    // else if constexpr(std::is_same_v<dstType, ck::half_t>)
    // {
    //     printf("----- dstType is half \n");
    // }
    if constexpr(std::is_same_v<srcAType, ck::half_t> && std::is_same_v<srcBType, ck::half_t> && std::is_same_v<dstType, ck::half_t>)
    {        
        // printf("--------- Calling f16, f16 ---------- \n");
        CK_WMMA_CALL_INTRIN(f16, f16);
    } 
    else if constexpr(std::is_same_v<srcAType, ck::half_t> && std::is_same_v<srcBType, ck::half_t> && std::is_same_v<dstType, float>)    {
        // printf("--------- Calling f32, f16 UNIMPLEMENTED ---------- \n");
        CK_WMMA_CALL_INTRIN(f32, f16);
    } else {
        printf("---------- No builtin_wmma_naive_selector implementation for these types ----------\n");
    }
}

template<typename srcA_t, typename srcB_t, typename dst_t, ck::index_t kMultiplier>
__global__ void matmul(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    //printf("---------- Running gfx125 matmul - INCORRECT INDEXING ----------\n");
    static_assert(WMMAVecType<srcA_t, kMultiplier>::template is_compatible<srcB_t>(),
                  "the data format for srcA and srcB is unsupported in gfx1250");
    using srcA_cast_T    = WMMAVecType<srcA_t, kMultiplier>::ViewT;
    using srcB_cast_T    = WMMAVecType<srcB_t, kMultiplier>::ViewT;
    using srcA_cast_type = typename srcA_cast_T::type;
    using srcB_cast_type = typename srcB_cast_T::type;

    //KO TODO:: revisit this. 
    using srcA_vec      = typename WMMAVecType<srcA_t, kMultiplier>::VecT;
    using srcB_vec      = typename WMMAVecType<srcB_t, kMultiplier>::VecT;
    using srcA_vec_type = srcA_vec::type;
    using srcB_vec_type = srcB_vec::type;

    const srcA_cast_type* a_ptr = reinterpret_cast<const srcA_cast_type*>(a);
    const srcB_cast_type* b_ptr = reinterpret_cast<const srcB_cast_type*>(b);

    srcA_vec a_frag = {};
    srcB_vec b_frag = {};

    srcA_vec a_temp = {};
    srcB_vec b_temp = {};

    using dst_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                              dst_t,
                                              1,
                                              8/** WMMAVecType<dst_t, kMultiplier>::ToIntDim*/,
                                              true>;
    dst_vec dst_thread_buf_;

    constexpr int ToIntDim    = WMMAVecType<srcA_t, kMultiplier>::ToIntDim;
    constexpr int SRC_DIM     = 2;//8 * kMultiplier; // KO TODO:: Fix... Number of packed elements per thread
    constexpr int ROW_SIZE    = SRC_DIM * 2;     // Two threads per block
    constexpr int LDS_DIM     = ROW_SIZE * 32 * 2 / ToIntDim; // A and B, packed
    constexpr int LDS_B_START = LDS_DIM / 2;
    //printf("LDS_DIM = %d\n", LDS_DIM);

    __shared__ srcA_cast_type p_shared[LDS_DIM];

    // strongly-type compile time index value of 0 for template containers
    static constexpr auto I0          = Number<0>{};
    
    // use a directly, as lds_shared allocated with A; A better be bigger than B if mixed
    const srcA_cast_type* local_a_ptr = p_shared;

    // get pointer as B type given LDS allocated with A
    const srcB_cast_type* local_b_ptr = reinterpret_cast<const srcB_cast_type*>(p_shared + LDS_B_START);
    //printf("[LDS_B_START] LDS_B_START = %d\n", LDS_B_START);
    //printf("[local_b_ptr] start address: %p\n", static_cast<const void*>(local_b_ptr));
    //printf("[local_b_ptr] address at offset 0: %p\n", static_cast<const void*>(&local_b_ptr[0]));
    //printf("[local_b_ptr] address at offset 256: %p\n", static_cast<const void*>(&local_b_ptr[256]));
    //printf("[local_b_ptr] address at offset 512: %p\n", static_cast<const void*>(&local_b_ptr[512]));

    const int lIdx = threadIdx.x;
    const int lane = lIdx % 32; // wave size

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        // Blocked M, K pattern:
        // For lanes 0-15: K = ele + 0
        // For lanes 16-31: K = ele + SRC_DIM
        int m = lane % 16;
        int k = (lane < 16) ? ele : ele + SRC_DIM;
        int a_idx = m * (SRC_DIM * 2) + k; // M x K-major
        a_temp.template AsType<srcA_cast_type>()(ele) = a_ptr[a_idx];
    });

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        int n = lane % 16;
        int kb = (lane < 16) ? ele : ele + SRC_DIM;
        int b_idx = kb * (SRC_DIM * 2) + n; // K x N-major
        b_temp.template AsType<srcB_cast_type>()(ele) = b_ptr[b_idx];
    });

    // __syncthreads(); // KO TODO:: Needed?

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        int lds_a_idx = SRC_DIM * lIdx + ele;
        //printf("[LDS WRITE] p_shared[%d] = \n", lds_a_idx);
        p_shared[lds_a_idx] = a_temp.template AsType<srcA_cast_type>()(ele);
    });

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        int lds_b_idx = SRC_DIM * lIdx + ele + LDS_B_START;
        //printf("[LDS WRITE] p_shared[%d] = \n", lds_b_idx);
        p_shared[lds_b_idx] = b_temp.template AsType<srcB_cast_type>()(ele);
    });

    __syncthreads(); //KO TODO:: move to inline asm

    static_for<0, SRC_DIM, 1>{}([&](auto ele) {
        int a_idx = SRC_DIM * lIdx + ele;
        //printf("[LDS READ] local_a_ptr[%d] (p_shared[%d]) \n", a_idx, a_idx);
        a_frag.template AsType<srcA_cast_type>()(ele) = local_a_ptr[a_idx];

        int b_idx = SRC_DIM * lIdx + ele + LDS_B_START;
           //printf("[LDS RANGE] p_shared start: %p, end: %p\n", 
        //    static_cast<void*>(p_shared), 
        //    static_cast<void*>(p_shared + LDS_DIM - 1));
        //printf("[LDS READ] local_b_ptr[%d] (p_shared[%d]) \n", b_idx, b_idx);
        //printf("[LDS ACCESS] local_b_ptr[%d] address: %p\n", 
        //    b_idx, 
        //    static_cast<const void*>(&local_b_ptr[b_idx]));
        b_frag.template AsType<srcB_cast_type>()(ele) = local_b_ptr[0];//local_b_ptr[b_idx];
    });

    __syncthreads(); //KO TODO:: move to inline asm

    // Call the WMMA intrinsic selector
    //printf("------- calling builtin_naive_wmma_selector ------- \n");
    builtin_wmma_naive_selector<srcA_t, srcB_t, dst_t, kMultiplier>(
        a_frag.template AsType<srcA_vec_type>()(I0),
        b_frag.template AsType<srcB_vec_type>()(I0),
        dst_thread_buf_);

    //printf("------- FINISHED builtin_naive_wmma_selector ------- \n");

    // Store results to global memory (row-major, adjust as needed)
    static_for<0, 8, 1>{}([&](auto ele) {
        int col = lIdx >> 1;
        int row = ((ele & 6) << 1) + (ele & 1) + ((lIdx & 1) << 1);
        c[16 * row + col] = dst_thread_buf_[Number<ele>{}];
    });
}

template <typename srcA_t, typename srcB_t, typename dst_t, ck::index_t KMultiplier>
__global__ void matmul_swizzle_a(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
    //printf("---------- Running gfx125 matmul_swizzle_a - NOT IMPLEMENTED YET ----------\n");
}

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
    //printf("---------- Running gfx1250 matmul - DISABLED original matmul ----------\n");
}

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
    //printf("---------- Running gfx1250 matmul - DISABLED original matmul_swizzle_a ----------\n");
}

#else 

template <typename AccType>
struct WMMA_ACCNumber_traits
{
    static constexpr index_t ACC_NUMBER = 8;
};

template <>
struct WMMA_ACCNumber_traits<ck::half_t>
{
    static constexpr index_t ACC_NUMBER = 16;
};

template <>
struct WMMA_ACCNumber_traits<ck::bhalf_t>
{
    static constexpr index_t ACC_NUMBER = 16;
};

template <typename src_vec, typename acc_vec>
__device__ void builtin_wmma_naive_selector(const src_vec&, const src_vec&, acc_vec&)
{
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<bhalf16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true>& reg_c)
{
    intrin_wmma_f32_16x16x16_bf16_w32<16, 16>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<half16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>>(
    const half16_t& reg_a,
    const half16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, half_t, 1, 16, true>& reg_c)
{
    intrin_wmma_f16_16x16x16_f16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void builtin_wmma_naive_selector<
    bhalf16_t,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>>(
    const bhalf16_t& reg_a,
    const bhalf16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, bhalf_t, 1, 16, true>& reg_c)
{
    intrin_wmma_bf16_16x16x16_bf16_w32<16, 16, 0>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

template <>
__device__ void
builtin_wmma_naive_selector<int8x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int8x16_t& reg_a,
    const int8x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu8_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
__device__ void
builtin_wmma_naive_selector<int4x16_t,
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>>(
    const int4x16_t& reg_a,
    const int4x16_t& reg_b,
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, int32_t, 1, 8, true>& reg_c)
{
    intrin_wmma_i32_16x16x16_iu4_w32<16, 16, true, true, false>::Run(
        reg_a, reg_b, reg_c.GetVectorTypeReference(Number<0>{}));
}
#endif


template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul(const src_t* a, const src_t* b, dst_t* c)
{
    //printf("------ USING LEGACY MATMUL ------ ");
    __shared__ src_t p_shared[16 * 16 * 2];
    const int lIdx = threadIdx.x;
    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and
    // b a_frag will store one column of the 16x16 matrix tile b_frag will store one row of the
    // 16x16 matrix tile
    using src_vec  = typename vector_type<src_t, 16>::type;
    constexpr index_t acc_num = WMMA_ACCNumber_traits<acc_t>::ACC_NUMBER;
    src_vec a_frag = {};
    src_vec b_frag = {};

    src_vec a_temp = {};
    src_vec b_temp = {};
    // initialize c fragment to 0
    using acc_vec = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in gfx11
    // see https://atlvsp3.amd.com/sp3_gfx11_5_instructions.pdf page 482
    // TODO: remove this dependency in gfx12 https://ontrack-internal.amd.com/browse/DEGFXSP3-101
    const int lane    = lIdx % 16;
    const int lane_lo = lIdx / 2;
    const int lane_hi = lIdx % 2;
    for(int ele = 0; ele < 8; ++ele)
    {
        a_temp[ele] = a[8 * lane_hi + 16 * lane_lo + ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        b_temp[ele] = b[8 * lane_hi + 16 * lane_lo + ele];
    }

    __syncthreads();

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele] = a_temp[ele];
    }

    for(int ele = 0; ele < 8; ++ele)
    {
        p_shared[8 * 16 * lane_hi + 8 * lane_lo + ele + 16 * 16] = b_temp[ele];
    }

#if defined(__gfx120__) || defined(__gfx125__)
    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);
#else
    asm volatile("\
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8 + 16 * 16];
    }
    // follow origin design
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = p_shared[(ele / 8) * 16 * 8 + 8 * lane + ele % 8];
    }

#if defined(__gfx120__) || defined(__gfx125__)
    asm volatile("\
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);
#else
    asm volatile("\
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif

    // sync threads, similar to mma_sync
    // __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    // since only fp16_fp32 asm wmma implemented for experiment purpose, restrict test case to fp16
    // when enable this ck::amd_assembly_wmma_f32_16x16x16_f16_w32(a_frag, b_frag,
    // c_thread_buf_.GetVectorTypeReference(Number<0>{}).template AsType<float8_t>()(Number<0>{}));
    __syncthreads();
    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

template <typename src_t, typename dst_t, typename acc_t>
__global__ void matmul_swizzle_a(const src_t* a, const src_t* b, dst_t* c)
{
    //printf("------ USING LEGACY MATMUL_SWIZZLE_A ------ ");
    const int lIdx = threadIdx.x;

    using src_vec  = typename vector_type<src_t, 16>::type;
    constexpr index_t acc_num = WMMA_ACCNumber_traits<acc_t>::ACC_NUMBER;
    src_vec a_frag = {};
    src_vec b_frag = {};
    using acc_vec  = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, acc_t, 1, acc_num, true>;
    acc_vec c_thread_buf_;

    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }

    const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * offset_m + ele];
    }

    __syncthreads();
    builtin_wmma_naive_selector<src_vec, acc_vec>(a_frag, b_frag, c_thread_buf_);
    __syncthreads();

    static_for<0, 8, 1>{}([&](auto ele) {
        const int blk = lIdx / 16;
        const int r   = ele;
        c[16 * 8 * blk + 16 * r + lane] =
            ck::type_convert<dst_t>(c_thread_buf_[Number<ele * acc_num / 8>{}]);
    });
}

template<typename srcA_t, typename srcB_t, typename dst_t, ck::index_t kMultiplier>
__global__ void matmul(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
}

template <typename srcA_t, typename srcB_t, typename dst_t, ck::index_t KMultiplier>
__global__ void matmul_swizzle_a(const srcA_t* a, const srcB_t* b, dst_t* c)
{
    ignore = a;
    ignore = b;
    ignore = c;
}
#endif




struct GemmParams
{
    GemmParams() : M(16), N(16), K(16), StrideA(16), StrideB(16), StrideC(16), alpha(1), beta(0) {}

    ck::index_t M;
    ck::index_t N;
    ck::index_t K;

    ck::index_t StrideA;
    ck::index_t StrideB;
    ck::index_t StrideC;

    float alpha;
    float beta;
};

template <typename GemmInstance,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void RunHostGEMM(const Tensor<ADataType>& A,
                 const Tensor<BDataType>& B,
                 Tensor<CDataType>& C,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
{
    auto ref_gemm     = GemmInstance{};
    auto ref_invoker  = ref_gemm.MakeInvoker();
    auto ref_argument = ref_gemm.MakeArgument(A, B, C, a_element_op, b_element_op, c_element_op);

    ref_invoker.Run(ref_argument);
}

template <typename KernelType, typename ADataType, typename BDataType, typename CDataType>
bool RunDeviceGEMM(KernelType kernel,
                   const Tensor<ADataType>& A,
                   const Tensor<BDataType>& B,
                   Tensor<CDataType>& C)
{
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * A.mDesc.GetElementSpaceSize());
    DeviceMem b_n_k_device_buf(sizeof(BDataType) * B.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * C.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(A.mData.data());
    b_n_k_device_buf.ToDevice(B.mData.data());
    kernel<<<1, 32>>>(static_cast<const ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                      static_cast<const BDataType*>(b_n_k_device_buf.GetDeviceBuffer()),
                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()));
    c_m_n_device_buf.FromDevice(C.mData.data());

    return true;
}

template <typename DeviceWmma,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GPUAccDataType,
          typename CPUAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t KMultiplier>
struct TestWmma
{
    auto PrepareGemmTensor(const ck::wmma_op_util::GemmParams& params)
    {
        auto f_host_tensor_descriptor =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({stride, 1}));
                }
                else
                {
                    return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                                std::vector<std::size_t>({1, stride}));
                }
            };

        Tensor<ADataType> a_m_k(
            f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
        Tensor<BDataType> b_n_k(
            f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
        Tensor<CDataType> c_m_n_host_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
        Tensor<CDataType> c_m_n_device_result(
            f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

        auto f_generate_tensor_value = [](auto& tensor, auto type) {
            using dataType = decltype(type);

            tensor.GenerateTensorValue(GeneratorTensor_2<dataType>{-5, 5});
        };

        f_generate_tensor_value(a_m_k, ADataType{});
        f_generate_tensor_value(b_n_k, BDataType{});

        return std::make_tuple(a_m_k, b_n_k, c_m_n_host_result, c_m_n_device_result);
    }

    auto operator()(const DeviceWmma& wmma_kernel)
    {
        // Arrange
        ck::wmma_op_util::GemmParams params;
        params.M       = 16;
        params.N       = 16;
        params.K       = 16 * KMultiplier;
        params.StrideA = 16;
        params.StrideB = 16;
        params.StrideC = 16;

        auto host_tensors = PrepareGemmTensor(params);

        const Tensor<ADataType>& a  = std::get<0>(host_tensors);
        const Tensor<BDataType>& b  = std::get<1>(host_tensors);
        Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
        Tensor<CDataType>& c_device = std::get<3>(host_tensors);

        auto a_element_op = AElementwiseOperation{};
        auto b_element_op = BElementwiseOperation{};
        auto c_element_op = CElementwiseOperation{};

        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      CPUAccDataType,
                                                      AElementwiseOperation,
                                                      BElementwiseOperation,
                                                      CElementwiseOperation>;
        ck::wmma_op_util::RunHostGEMM<ReferenceGemmInstance>(
            a, b, c_host, a_element_op, b_element_op, c_element_op);

        // Act
        bool is_supported = (ck::is_gfx11_supported() || ck::is_gfx12_supported()) &&
                            ck::wmma_op_util::RunDeviceGEMM(wmma_kernel, a, b, c_device);

        if(is_supported)
        {
            // Assert
            bool res = false;
            if(std::is_same<CDataType, float>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::half_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, ck::bhalf_t>::value)
            {
                // 0.5 Pixel Error Tolerance is introduced by Accumulator difference.
                // BF16 WMMA Accumulator is in BF16 Type while On Host-side Accumulator is Float.
                res = ck::utils::check_err(
                    c_device.mData, c_host.mData, "Error: Incorrect results!", 0, 1.0);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, int8_t>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else if(std::is_same<CDataType, double>::value)
            {
                res = ck::utils::check_err(c_device.mData, c_host.mData);
                std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;
            }
            else
            {
                std::cout << "UNSUPPORTED CDataType" << std::endl;
            }

            return res;
        }
        else
        {
            return true;
        }
    }
};

} // namespace wmma_op_util
} // namespace ck
