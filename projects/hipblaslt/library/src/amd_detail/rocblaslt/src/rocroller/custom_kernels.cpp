#include <hip/hip_runtime.h>

#include "aiter_kernels.hpp"
#include "custom_kernels.hpp"

std::shared_ptr<GemmKernel> createCustomGemmKernel(AiterKernelPtr           AiterKernelPtr,
                                                   const KernelType&        kernelType,
                                                   const WorkGroupTileSize& wgt)
{
    auto gemmKernel = std::make_shared<GemmKernel>();

    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;

    gemmKernel->customKernel = AiterKernelPtr;

    return gemmKernel;
}

void preloadCustomKernels(SolutionCache& cache)
{
    KernelType mxfp4Kernel;
    mxfp4Kernel.typeA                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeB                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeC                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.typeD                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.transA                    = true;
    mxfp4Kernel.transB                    = false;
    mxfp4Kernel.scaleTypeA.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeA.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeA.preTile        = {32, 8};
    mxfp4Kernel.scaleTypeB.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeB.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeB.preTile        = {8, 32};

    SolutionIndexParameters params;
    params.workgroupTile    = {256, 256, 256};
    params.workgroupMapping = false;
    params.streamK          = false;

    cache.addKernel(mxfp4Kernel,
                    params,
                    createCustomGemmKernel(get_mxfp4_mxfp4_bf16_256_256_gemm_custom(),
                                           mxfp4Kernel,
                                           params.workgroupTile));
}

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel>        gemm,
                                 const RocblasltContractionProblem& prob)
{
    dim3 grid;
    grid.x = (prob.n + gemm->params->workgroupTile.n - 1) / gemm->params->workgroupTile.n;
    grid.y = (prob.m + gemm->params->workgroupTile.m - 1) / gemm->params->workgroupTile.m;
    grid.z = 1;

    dim3 block{256, 1, 1};

    AiterKernelLaunchParams params{prob.D,
                                   prob.C,
                                   prob.A,
                                   prob.B,
                                   *static_cast<const float*>(prob.alpha),
                                   *static_cast<const float*>(prob.beta),
                                   static_cast<uint32_t>(prob.col_stride_d),
                                   static_cast<uint32_t>(prob.row_stride_d),
                                   static_cast<uint32_t>(prob.col_stride_c),
                                   static_cast<uint32_t>(prob.row_stride_c),
                                   static_cast<uint32_t>(prob.col_stride_a),
                                   static_cast<uint32_t>(prob.row_stride_a),
                                   static_cast<uint32_t>(prob.col_stride_b),
                                   static_cast<uint32_t>(prob.row_stride_b),
                                   static_cast<uint32_t>(prob.m),
                                   static_cast<uint32_t>(prob.n),
                                   static_cast<uint32_t>(prob.k),
                                   prob.scaleA,
                                   prob.scaleB,
                                   static_cast<uint32_t>(prob.m),
                                   1,
                                   static_cast<uint32_t>(prob.n),
                                   1,
                                   1};

    runAiterKernel(gemm->customKernel, grid, block, prob.stream, params);

    return rocblaslt_status_success;
}
