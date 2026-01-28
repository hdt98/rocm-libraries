#include "custom_kernels.hpp"

#include <hip/hip_runtime.h>

// List of available custom kernels
extern "C"
{
    __global__ void mxfp4_mxfp4_bf16_256_256_gemm_custom(void* D, void* C, void* A, void* B, float alpha, float beta,
        unsigned int stride_D0, unsigned int stride_D1,
        unsigned int stride_C0, unsigned int stride_C1,
        unsigned int stride_A0, unsigned int stride_A1,
        unsigned int stride_B0, unsigned int stride_B1,
        unsigned int m, unsigned int n, unsigned int k,
        void* scaleA, void* scaleB,
        unsigned int stride_scaleA0, unsigned int stride_scaleA1,
        unsigned int stride_scaleB0, unsigned int stride_scaleB1,
        int log2_split_k);
}

std::shared_ptr<GemmKernel> createCustomGemmKernel(CustomKernelPtr customKernelPtr, const KernelType& kernelType, const WorkGroupTileSize& wgt)
{
    auto gemmKernel = std::make_shared<GemmKernel>();

    gemmKernel->params = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType = kernelType;
    gemmKernel->params->workgroupTile = wgt;
    
    gemmKernel->customKernel = customKernelPtr;

    return gemmKernel;
}

void preloadCustomKernels(SolutionCache& cache)
{
    KernelType mxfp4Kernel;
    mxfp4Kernel.typeA = rocRoller::DataType::FP4;
    mxfp4Kernel.typeB = rocRoller::DataType::FP4;
    mxfp4Kernel.typeC = rocRoller::DataType::BFloat16;
    mxfp4Kernel.typeD = rocRoller::DataType::BFloat16;
    mxfp4Kernel.transA = true;
    mxfp4Kernel.transB = false;
    mxfp4Kernel.scaleTypeA.mode = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeA.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeA.preTile = {32, 8};
    mxfp4Kernel.scaleTypeB.mode = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeB.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeB.preTile = {8, 32};

    SolutionIndexParameters params;
    params.workgroupTile = {256, 256, 256};
    params.workgroupMapping = false;
    params.streamK = false;

    cache.addKernel(mxfp4Kernel, params, createCustomGemmKernel(mxfp4_mxfp4_bf16_256_256_gemm_custom, mxfp4Kernel, params.workgroupTile));
}

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel> gemm, const RocblasltContractionProblem& prob)
{
    dim3 grid;
    grid.x = (prob.n + gemm->params->workgroupTile.n - 1) / gemm->params->workgroupTile.n;
    grid.y = (prob.m + gemm->params->workgroupTile.m - 1) / gemm->params->workgroupTile.m;
    grid.z = 1;

    dim3 block{256, 1, 1};

    hipLaunchKernelGGL(gemm->customKernel, grid, block, 0, prob.stream,
        prob.D, prob.C, prob.A, prob.B, prob.alpha, prob.beta,
        prob.col_stride_d, prob.row_stride_d,
        prob.col_stride_c, prob.row_stride_c,
        prob.col_stride_a, prob.row_stride_a,
        prob.col_stride_b, prob.row_stride_b,
        prob.scaleA, prob.scaleB,
        prob.m, 1,
        prob.n, 1);
}
