#include "custom_kernels.hpp"

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <iostream>

std::shared_ptr<GemmKernel> createCustomGemmKernel(const std::string&       customKernelName,
                                                   const KernelType&        kernelType,
                                                   const WorkGroupTileSize& wgt,
                                                   const std::string&       path)
{
    auto gemmKernel = std::make_shared<GemmKernel>();

    gemmKernel->params                = std::make_shared<SolutionParameters>();
    gemmKernel->params->kernelType    = kernelType;
    gemmKernel->params->workgroupTile = wgt;

    gemmKernel->module = GemmHipModuleWrapper(customKernelName, path);

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
    mxfp4Kernel.typeA                   = rocRoller::DataType::FP4;
    mxfp4Kernel.typeB                   = rocRoller::DataType::FP4;
    mxfp4Kernel.typeC                   = rocRoller::DataType::BFloat16;
    mxfp4Kernel.typeD                   = rocRoller::DataType::BFloat16;
    mxfp4Kernel.transA                  = true;
    mxfp4Kernel.transB                  = false;
    mxfp4Kernel.scaleTypeA.mode         = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeA.blockRowSize = 32;
    mxfp4Kernel.scaleTypeA.blockColSize = 1;
    // Note: AITER kernel uses its own scale swizzle pattern.
    // Use --scaleA 1002 --scaleB 1002 to enable AITER scale swizzle in the data generator
    // instead of the rocRoller preSwizzle pattern defined here.
    mxfp4Kernel.scaleTypeA.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeA.preTile        = {32, 8};
    mxfp4Kernel.scaleTypeB.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeB.blockRowSize   = 1;
    mxfp4Kernel.scaleTypeB.blockColSize   = 32;
    mxfp4Kernel.scaleTypeB.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeB.preTile        = {8, 32};

    for(bool workgroupMapping : {false, true})
    {
        for(bool streamK : {false, true})
        {
            for(bool tailLoops : {false, true})
            {
                SolutionIndexParameters params;
                params.workgroupTile    = {256, 256, 256};
                params.workgroupMapping = workgroupMapping;
                params.streamK          = streamK;
                params.tailLoops        = tailLoops;

                cache.addKernel(mxfp4Kernel,
                                params,
                                createCustomGemmKernel(
                                    "_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E",
                                    mxfp4Kernel,
                                    params.workgroupTile,
                                    getCoPath("f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co")));
            }
        }
    }
}

// F4 GEMM Kernel Args

struct __attribute__((packed)) p3
{
    uint32_t _p0 = 0;
    uint32_t _p1 = 0;
    uint32_t _p2 = 0;
};
struct __attribute__((packed)) p2
{
    uint32_t _p0 = 0;
    uint32_t _p1 = 0;
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

    // AITER kernel computes D[N,M] = B^T * A instead of C[M,N] = A^T * B
    // So we swap A<->B pointers/scales and M<->N dimensions
    F4GemmKernelArgs(const RocblasltContractionProblem& prob)
        : ptr_D(prob.D)
        , ptr_C(nullptr)
        , ptr_A(const_cast<void*>(prob.B)) // Swapped: kernel's A = hipBLASLt's B
        , ptr_B(const_cast<void*>(prob.A)) // Swapped: kernel's B = hipBLASLt's A
        , alpha(*static_cast<const float*>(prob.alpha))
        , beta(*static_cast<const float*>(prob.beta))
        , stride_D0(0)
        , stride_D1(0)
        , stride_C0(static_cast<uint32_t>(prob.col_stride_c))
        , stride_C1(0)
        , stride_A0(static_cast<uint32_t>(prob.col_stride_b)) // Swapped
        , stride_A1(0)
        , stride_B0(static_cast<uint32_t>(prob.col_stride_a)) // Swapped
        , stride_B1(0)
        , M(static_cast<uint32_t>(prob.n)) // Swapped: kernel's M = hipBLASLt's N
        , N(static_cast<uint32_t>(prob.m)) // Swapped: kernel's N = hipBLASLt's M
        , K(static_cast<uint32_t>(prob.k))
        , ptr_ScaleA(prob.scaleB) // Swapped
        , ptr_ScaleB(prob.scaleA) // Swapped
        , stride_ScaleA0(static_cast<uint32_t>(prob.k / 32))
        , stride_ScaleA1(0)
        , stride_ScaleB0(static_cast<uint32_t>(prob.k / 32))
        , stride_ScaleB1(0)
        , log2_k_split(0)
    {
    }
};

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel>        gemm,
                                 const RocblasltContractionProblem& prob)
{
    if(!gemm->module.has_value())
    {
        std::cerr << "runCustomKernel failed: Module not loadable" << std::endl;
        return rocblaslt_status_internal_error;
    }

    F4GemmKernelArgs args(prob);

    // The AITER kernel processes tiles of size 256x256
    // Note: args.M and args.N are already swapped in the constructor
    const uint32_t tileM     = 256;
    const uint32_t tileN     = 256;
    const uint32_t blockSize = 256; // Threads per workgroup

    // Number of tiles in each dimension
    uint32_t tilesM = (args.M + tileM - 1) / tileM;
    uint32_t tilesN = (args.N + tileN - 1) / tileN;

    dim3 grid;
    grid.x = tilesN * blockSize; // Total threads in X
    grid.y = tilesM; // Workgroups in Y
    grid.z = 1;

    dim3 block{blockSize, 1, 1};

    void*  argsPtr  = &args;
    size_t argsSize = sizeof(args);

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               argsPtr,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    hipFunction_t function;
    if(hipError_t error = gemm->module->getHipFunction(function))
    {
        std::cerr << "GemmHipModuleWrapper::getHipFunction failed: " << std::endl
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
        std::cerr << "hipExtModuleLaunchKernel in runCustomKernel failed: "
                  << gemm->module->getKernelName() << std::endl
                  << " error: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    // Synchronize to ensure kernel completes before verification
    if(hipError_t error = hipStreamSynchronize(prob.stream))
    {
        std::cerr << "hipStreamSynchronize failed: " << hipGetErrorString(error) << std::endl;
        return rocblaslt_status_internal_error;
    }

    return rocblaslt_status_success;
}
