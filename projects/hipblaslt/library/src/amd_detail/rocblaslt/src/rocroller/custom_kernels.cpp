#include "custom_kernels.hpp"

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&       customKernelName,
                                                   const KernelType&        kernelType,
                                                   const WorkGroupTileSize& wgt,
                                                   const std::string&       path)
{
    auto gemmKernel = std::make_shared<GemmKernel>();

    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;

    gemmKernel->customKernelName = customKernelName;

    // TODO: Error handling on failed module load
    auto error = gemmKernel->loadModule(path);

    return gemmKernel;
}

std::string getCoPath(const std::string& filename)
{
    const char* env = std::getenv("HIPBLASLT_ROCROLLER_CUSTOM_KERNEL_DIR");
    if(env)
    {
        std::string path = env;
        // Ensure trailing slash
        if(!path.empty() && path.back() != '/')
        {
            path += "/";
        }
        return path + filename;
    }

    // TODO: Fallback should be where the kernels are packaged
    return "./" + filename;
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

    cache.addKernel(
        mxfp4Kernel,
        params,
        createCustomGemmKernel("_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E",
                               mxfp4Kernel,
                               params.workgroupTile,
                               getCoPath("f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co")));
}

// F4 GEMM Kernel Args

struct p3
{
    uint32_t _p0, _p1, _p2;
};
struct p2
{
    uint32_t _p0, _p1;
};
struct __attribute__((packed)) F4GemmKernelArgs
{
    void*       ptr_D;
    p2          _p0;
    const void* ptr_C;
    p2          _p1;
    const void* ptr_A;
    p2          _p2;
    const void* ptr_B;
    p2          _p3;
    float       alpha;
    p3          _p4;
    float       beta;
    p3          _p5;
    uint32_t    stride_D0;
    p3          _p6;
    uint32_t    stride_D1;
    p3          _p7;
    uint32_t    stride_C0;
    p3          _p8;
    uint32_t    stride_C1;
    p3          _p9;
    uint32_t    stride_A0;
    p3          _p10;
    uint32_t    stride_A1;
    p3          _p11;
    uint32_t    stride_B0;
    p3          _p12;
    uint32_t    stride_B1;
    p3          _p13;
    uint32_t    M;
    p3          _p14;
    uint32_t    N;
    p3          _p15;
    uint32_t    K;
    p3          _p16;
    const void* ptr_ScaleA;
    p2          _p17;
    const void* ptr_ScaleB;
    p2          _p18;
    uint32_t    stride_ScaleA0;
    p3          _p19;
    uint32_t    stride_ScaleA1;
    p3          _p20;
    uint32_t    stride_ScaleB0;
    p3          _p21;
    uint32_t    stride_ScaleB1;
    p3          _p22;
    int         log2_k_split;

    F4GemmKernelArgs(const RocblasltContractionProblem& prob)
        : ptr_D(prob.D)
        , ptr_C(const_cast<void*>(prob.C))
        , ptr_A(const_cast<void*>(prob.A))
        , ptr_B(const_cast<void*>(prob.B))
        , alpha(*static_cast<const float*>(prob.alpha))
        , beta(*static_cast<const float*>(prob.beta))
        , stride_D0(static_cast<uint32_t>(prob.col_stride_d))
        , stride_D1(static_cast<uint32_t>(prob.row_stride_d))
        , stride_C0(static_cast<uint32_t>(prob.col_stride_c))
        , stride_C1(static_cast<uint32_t>(prob.row_stride_c))
        , stride_A0(static_cast<uint32_t>(prob.col_stride_a))
        , stride_A1(static_cast<uint32_t>(prob.row_stride_a))
        , stride_B0(static_cast<uint32_t>(prob.col_stride_b))
        , stride_B1(static_cast<uint32_t>(prob.row_stride_b))
        , M(static_cast<uint32_t>(prob.m))
        , N(static_cast<uint32_t>(prob.n))
        , K(static_cast<uint32_t>(prob.k))
        , ptr_ScaleA(prob.scaleA)
        , ptr_ScaleB(prob.scaleB)
        , stride_ScaleA0(static_cast<uint32_t>(prob.m) / 32)
        , stride_ScaleA1(1)
        , stride_ScaleB0(static_cast<uint32_t>(prob.n) / 32)
        , stride_ScaleB1(1)
        , log2_k_split(0)
    {
    }
};

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel>        gemm,
                                 const RocblasltContractionProblem& prob)
{
    dim3 grid;
    grid.x = (prob.n + gemm->params->workgroupTile.n - 1) / gemm->params->workgroupTile.n;
    grid.y = (prob.m + gemm->params->workgroupTile.m - 1) / gemm->params->workgroupTile.m;
    grid.z = 1;

    dim3 block{256, 1, 1};

    auto   args     = F4GemmKernelArgs(prob);
    void*  argsPtr  = &args;
    size_t argsSize = sizeof(args);

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               argsPtr,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    hipFunction_t function;
    if(hipError_t error = gemm->getHipFunction(function))
    {
        std::cerr << "GemmKernel::getHipFunction failed: " << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    if(hipError_t error = hipExtModuleLaunchKernel(function,
                                                   grid.x,
                                                   grid.y,
                                                   grid.z,
                                                   block.x,
                                                   block.y,
                                                   block.z,
                                                   0, // sharedMem
                                                   prob.stream, // stream
                                                   nullptr,
                                                   (void**)&hipLaunchParams,
                                                   nullptr, // event
                                                   nullptr // event
                                                   ))
    {
        std::cerr << "hipExtModuleLaunchKernel failed: " << gemm->customKernelName
                  << std::endl
                  //   << " with workgroup size: " << block
                  //   << std::endl
                  //   << " with numWorkGroups : " << kernel.numWorkGroups << std::endl
                  //   << " with numWorkItems : " << grid << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    return rocblaslt_status_success;
}
