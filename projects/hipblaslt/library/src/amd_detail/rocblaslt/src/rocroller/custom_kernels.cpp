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
    mxfp4Kernel.typeA                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeB                     = rocRoller::DataType::FP4;
    mxfp4Kernel.typeC                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.typeD                     = rocRoller::DataType::BFloat16;
    mxfp4Kernel.transA                    = true;
    mxfp4Kernel.transB                    = false;
    mxfp4Kernel.scaleTypeA.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeA.blockRowSize   = 32;
    mxfp4Kernel.scaleTypeA.blockColSize   = 1;
    mxfp4Kernel.scaleTypeA.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeA.preTile        = {32, 8};
    mxfp4Kernel.scaleTypeB.mode           = rocRoller::Operations::ScaleMode::Separate;
    mxfp4Kernel.scaleTypeB.blockRowSize   = 1;
    mxfp4Kernel.scaleTypeB.blockColSize   = 32;
    mxfp4Kernel.scaleTypeB.preSwizzleTile = {32, 8, 4};
    mxfp4Kernel.scaleTypeB.preTile        = {8, 32};

    SolutionIndexParameters params;
    params.workgroupTile    = {256, 256, 256};
    params.workgroupMapping = true;
    params.streamK          = false;
    params.tailLoops        = true;

    cache.addKernel(
        mxfp4Kernel,
        params,
        createCustomGemmKernel("_ZN5aiter44f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256E",
                               mxfp4Kernel,
                               params.workgroupTile,
                               getCoPath("f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co")));
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

    F4GemmKernelArgs(const RocblasltContractionProblem& prob)
        : ptr_D(prob.D)
        , ptr_C(nullptr) // , ptr_C(const_cast<void*>(prob.C))
        , ptr_A(const_cast<void*>(prob.A))
        , ptr_B(const_cast<void*>(prob.B))
        , alpha(*static_cast<const float*>(prob.alpha))
        , beta(*static_cast<const float*>(prob.beta))
        , stride_D0(0) //, stride_D0(static_cast<uint32_t>(prob.col_stride_d))
        , stride_D1(0) //, stride_D1(static_cast<uint32_t>(prob.row_stride_d))
        , stride_C0(static_cast<uint32_t>(prob.col_stride_c))
        , stride_C1(0) //, stride_C1(static_cast<uint32_t>(prob.row_stride_c))
        , stride_A0(static_cast<uint32_t>(prob.col_stride_a))
        , stride_A1(0) //, stride_A1(static_cast<uint32_t>(prob.row_stride_a))
        , stride_B0(static_cast<uint32_t>(prob.col_stride_b))
        , stride_B1(0) //, stride_B1(static_cast<uint32_t>(prob.row_stride_b))
        , M(static_cast<uint32_t>(prob.m))
        , N(static_cast<uint32_t>(prob.n))
        , K(static_cast<uint32_t>(prob.k))
        , ptr_ScaleA(prob.scaleA)
        , ptr_ScaleB(prob.scaleB)
        , stride_ScaleA0(static_cast<uint32_t>(
              prob.col_stride_a / 32)) //, stride_ScaleA0(static_cast<uint32_t>(prob.m))
        , stride_ScaleA1(0)
        , stride_ScaleB0(static_cast<uint32_t>(
              prob.col_stride_b / 32)) //, stride_ScaleB0(static_cast<uint32_t>(prob.n))
        , stride_ScaleB1(0)
        , log2_k_split(0)
    {
    }
};


void printKernelArgs(const F4GemmKernelArgs& args)
    {

        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&args);
        size_t               size  = sizeof(args);

        std::cout << "Offset | 00 01 02 03 04 05 06 07 | 08 09 0A 0B 0C 0D 0E 0F" << std::endl;
        std::cout << "-------|-------------------------|-------------------------" << std::endl;

        for(size_t i = 0; i < size; i += 16)
        {
            std::cout << std::setw(4) << std::setfill('0') << std::dec << i << "   | ";
            for(size_t j = 0; j < 16; ++j)
            {
                if(i + j < size)
                {
                    std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i + j]
                              << " ";
                }
                else
                {
                    std::cout << "   ";
                }
                if(j == 7)
                    std::cout << "| ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================================\n" << std::endl;

        std::cout << "\n========== [DIRECT ASSEMBLY REPLICATION DATA] ==========" << std::endl;

        std::cout << "\n--- KernelArgs Struct Content ---" << std::endl;
        std::cout << std::hex;
        std::cout << "ptr_D        : " << args.ptr_D << std::endl;
        std::cout << "ptr_C        : " << args.ptr_C << std::endl;
        std::cout << "ptr_A        : " << args.ptr_A << std::endl;
        std::cout << "ptr_B        : " << args.ptr_B << std::endl;
        std::cout << "ptr_ScaleA   : " << args.ptr_ScaleA << std::endl;
        std::cout << "ptr_ScaleB   : " << args.ptr_ScaleB << std::endl;
        std::cout << std::dec;
        std::cout << "alpha        : " << args.alpha << std::endl;
        std::cout << "beta         : " << args.beta << std::endl;
        std::cout << "M            : " << args.M << std::endl;
        std::cout << "N            : " << args.N << std::endl;
        std::cout << "K            : " << args.K << std::endl;
        std::cout << "\n[Strides]" << std::endl;
        std::cout << "stride_D0 (Out)   : " << args.stride_D0 << std::endl;
        std::cout << "stride_D1         : " << args.stride_D1 << std::endl;
        std::cout << "stride_C0 (Bias)  : " << args.stride_C0 << std::endl;
        std::cout << "stride_C1         : " << args.stride_C1 << std::endl;
        std::cout << "stride_A0 (In A)  : " << args.stride_A0 << std::endl;
        std::cout << "stride_A1         : " << args.stride_A1 << std::endl;
        std::cout << "stride_B0 (In B)  : " << args.stride_B0 << std::endl;
        std::cout << "stride_B1         : " << args.stride_B1 << std::endl;
        std::cout << "stride_ScaleA0    : " << args.stride_ScaleA0 << std::endl;
        std::cout << "stride_ScaleA1    : " << args.stride_ScaleA1 << std::endl;
        std::cout << "stride_ScaleB0    : " << args.stride_ScaleB0 << std::endl;
        std::cout << "stride_ScaleB1    : " << args.stride_ScaleB1 << std::endl;
        std::cout << "========== [DIRECT ASSEMBLY DATA END] ==========\n" << std::endl;
    }

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel>        gemm,
                                 const RocblasltContractionProblem& prob)
{
    if(!gemm->module.has_value())
    {
        std::cerr << "runCustomKernel failed: Module not loadable" << std::endl;
        return rocblaslt_status_internal_error;
    }

    dim3 grid;
    grid.x = (prob.n + gemm->params->workgroupTile.n - 1) / gemm->params->workgroupTile.n;
    grid.y = (prob.m + gemm->params->workgroupTile.m - 1) / gemm->params->workgroupTile.m;
    grid.z = 1;

    dim3 block{256, 1, 1};

    auto   args     = F4GemmKernelArgs(prob);
    printKernelArgs(args);
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

    return rocblaslt_status_success;
}
