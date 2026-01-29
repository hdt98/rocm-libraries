#include <hip/hip_runtime.h>

#include "aiter_kernels.hpp"

AiterKernelPtr get_mxfp4_mxfp4_bf16_256_256_gemm_custom()
{
    return aiter::f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256;
}

void runAiterKernel(AiterKernelPtr                 kernel,
                    dim3                           grid,
                    dim3                           block,
                    hipStream_t                    stream,
                    AiterKernelLaunchParams const& params)
{
    hipLaunchKernelGGL(kernel,
                       grid,
                       block,
                       0,
                       stream,
                       params.D,
                       params.C,
                       params.A,
                       params.B,
                       params.alpha,
                       params.beta,
                       params.stride_D0,
                       params.stride_D1,
                       params.stride_C0,
                       params.stride_C1,
                       params.stride_A0,
                       params.stride_A1,
                       params.stride_B0,
                       params.stride_B1,
                       params.m,
                       params.n,
                       params.k,
                       params.scaleA,
                       params.scaleB,
                       params.stride_scaleA0,
                       params.stride_scaleA1,
                       params.stride_scaleB0,
                       params.stride_scaleB1,
                       params.log2_split_k);
}
