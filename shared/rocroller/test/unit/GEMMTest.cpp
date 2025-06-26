// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <regex>

#include "GEMMF8F6F4.hpp"
#include "GEMMTestBase.hpp"

#include "common/SourceMatcher.hpp"

namespace GEMMTests
{
    using namespace rocRoller;
    namespace SolutionParams = rocRoller::Parameters::Solution;

    std::set<int> NonZeroDSReadOffsets(std::string const& instruction, std::string const& s)
    {
        std::regex ds_read_offset(instruction + ".*offset:(\\d+)");

        auto begin = std::sregex_iterator(s.begin(), s.end(), ds_read_offset);
        auto end   = std::sregex_iterator();

        std::set<int> rv;
        for(auto i = begin; i != end; ++i)
        {
            auto m = (*i)[1].str();
            rv.insert(std::stoi(m));
        }
        return rv;
    }

    std::set<int> Direct2LDSWriteStrides(std::string const& s)
    {
        std::regex m0_stride_pattern("s_add_u32 m0, m0, (\\d+)");

        auto begin = std::sregex_iterator(s.begin(), s.end(), m0_stride_pattern);
        auto end   = std::sregex_iterator();

        std::set<int> rv;
        for(auto i = begin; i != end; ++i)
        {
            auto m = (*i)[1].str();
            rv.insert(std::stoi(m));
        }
        return rv;
    }

        template <typename TA,
                  typename TB  = TA,
                  typename TC  = TA,
                  typename TD  = TC,
                  typename ACC = float>
        void basicGEMM(const GEMMProblem&      gemm,
                       bool                    debuggable  = false,
                       bool                    setIdentity = false,
                       int                     numIters    = 1,
                       bool                    notSetC     = false,
                       std::optional<uint32_t> srCvtSeed   = std::nullopt)
        {
            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_fp8,
                                        GPUCapability::HasWMMA_f32_16x16x16_f8,
                                        GPUCapability::HasWMMA_f32_16x16x64_f8);
            }

            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4,
                                        GPUCapability::HasWMMA_f8f6f4);
            }

            if((isF8<TA> || isF8<TB>)&&(gemm.waveK >= 64))
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4,
                                        GPUCapability::HasWMMA_f8f6f4);
            }

            if(gemm.scaleAMode != Operations::ScaleMode::None
               || gemm.scaleBMode != Operations::ScaleMode::None)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4,
                                        GPUCapability::HasWMMA_scale_f8f6f4);
                const auto  scaleType = gemm.scaleAMode != Operations::ScaleMode::None
                                            ? gemm.scaleTypeA
                                            : gemm.scaleTypeB;
                const auto& arch      = m_context->targetArchitecture();
                AssertFatal(gemm.scaleAMode == Operations::ScaleMode::None
                                || arch.isSupportedScaleType(gemm.scaleTypeA),
                            fmt::format("Scale mode for A set but architecture {} does not "
                                        "support scale type {}.",
                                        arch.target().toString(),
                                        toString(gemm.scaleTypeA)));
                AssertFatal(gemm.scaleBMode == Operations::ScaleMode::None
                                || arch.isSupportedScaleType(gemm.scaleTypeB),
                            fmt::format("Scale mode for B set but architecture {} does not "
                                        "support scale type {}.",
                                        arch.target().toString(),
                                        toString(gemm.scaleTypeB)));
            }

            AssertFatal(gemm.scaleAMode == Operations::ScaleMode::None
                            || gemm.scaleAMode == Operations::ScaleMode::SingleScale
                            || gemm.scaleAMode == Operations::ScaleMode::Separate,
                        "Scale mode not supported!",
                        ShowValue(gemm.scaleAMode));
            AssertFatal(gemm.scaleBMode == Operations::ScaleMode::None
                            || gemm.scaleBMode == Operations::ScaleMode::SingleScale
                            || gemm.scaleBMode == Operations::ScaleMode::Separate,
                        "Scale mode not supported!",
                        ShowValue(gemm.scaleBMode));

            auto dataTypeA   = TypeInfo<TA>::Var.dataType;
            auto dataTypeB   = TypeInfo<TB>::Var.dataType;
            auto dataTypeC   = TypeInfo<TC>::Var.dataType;
            auto dataTypeD   = TypeInfo<TD>::Var.dataType;
            auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

            // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
            int   M     = gemm.m;
            int   N     = gemm.n;
            int   K     = gemm.k;
            float alpha = gemm.alpha;
            float beta  = gemm.beta;

            AssertFatal(M % gemm.macM == 0,
                        "MacroTile size mismatch (M)",
                        ShowValue(M),
                        ShowValue(gemm.macM));
            AssertFatal(N % gemm.macN == 0,
                        "MacroTile size mismatch (N)",
                        ShowValue(N),
                        ShowValue(gemm.macN));

            if(gemm.scaleAMode == Operations::ScaleMode::Separate
               || gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(
                    m_context->targetArchitecture().isSupportedScaleBlockSize(gemm.scaleBlockSize),
                    fmt::format("Architecture {} does not support block scaling (size: {}).",
                                m_context->targetArchitecture().target().toString(),
                                gemm.scaleBlockSize));
            }

            if(gemm.unrollK > 0 && !gemm.tailLoops)
            {
                AssertFatal(K % (gemm.macK * gemm.unrollK) == 0,
                            "MacroTile size mismatch (K unroll)");
            }

            auto bpeA = DataTypeInfo::Get(dataTypeA).elementBytes;
            auto bpeB = DataTypeInfo::Get(dataTypeB).elementBytes;
            AssertFatal(gemm.macM * gemm.macK * bpeA >= gemm.waveM * gemm.waveK,
                        "Not enough elements (A).");
            AssertFatal(gemm.macN * gemm.macK * bpeB >= gemm.waveN * gemm.waveK,
                        "Not enough elements (B).");

            AssertFatal(gemm.workgroupSizeX % gemm.wavefrontSize == 0,
                        "Workgroup Size X must be multiply of wave front size");

            uint wavetilePerWavefrontM
                = gemm.wavefrontSize * gemm.macM / gemm.waveM / gemm.workgroupSizeX;
            uint wavetilePerWavefrontN = gemm.macN / gemm.waveN / gemm.workgroupSizeY;

            AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch (M).");
            AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch (N).");

            AssertFatal(gemm.macM % (gemm.waveM * wavetilePerWavefrontM) == 0,
                        "WaveTile size mismatch (M)");
            AssertFatal(gemm.macN % (gemm.waveN * wavetilePerWavefrontN) == 0,
                        "WaveTile size mismatch (N)");

            Log::debug("GEMMTest jamming: {}x{}", wavetilePerWavefrontM, wavetilePerWavefrontN);

            uint workgroupSizeX = gemm.workgroupSizeX * gemm.workgroupSizeY;
            uint workgroupSizeY = 1;

            uint numWorkgroupX;
            uint numWorkgroupY;

            if(gemm.loopOverTiles > 0)
            {
                // multiple output macro tiles per workgroup
                numWorkgroupX = M * N / gemm.macM / gemm.macN / 2;
                numWorkgroupY = 1;
            }
            else if(gemm.streamK)
            {
                numWorkgroupX = gemm.numWGs;
                numWorkgroupY = 1;
            }
            else
            {
                // one output macro tile per workgroup
                numWorkgroupX = M / gemm.macM;
                numWorkgroupY = N / gemm.macN;
            }

            // Host data
            using PackedTypeA = typename PackedTypeOf<TA>::type;
            using PackedTypeB = typename PackedTypeOf<TB>::type;
            std::vector<PackedTypeA> hostA;
            std::vector<PackedTypeB> hostB;
            std::vector<TC>          hostC;

            std::vector<uint8_t> hostScaleA, hostScaleB;

            TensorDescriptor descA(dataTypeA, {size_t(M), size_t(K)}, gemm.transA);
            TensorDescriptor descB(dataTypeB, {size_t(K), size_t(N)}, gemm.transB);
            TensorDescriptor descC(dataTypeD, {size_t(M), size_t(N)}, "N");
            TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

            auto seed = 31415u;
            if(gemm.scaleAMode == Operations::ScaleMode::Separate
               || gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                auto const& arch = m_context->targetArchitecture();

                auto scaleBlockSize = gemm.scaleBlockSize;
                AssertFatal(scaleBlockSize > 0, "scaleBlockSize must be set to scale A or B.");
                AssertFatal(
                    arch.isSupportedScaleBlockSize(scaleBlockSize),
                    fmt::format("Architecture {} does not support block scaling (size: {}).",
                                arch.target().toString(),
                                scaleBlockSize));
                AssertFatal(gemm.k % scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        gemm.k,
                                        scaleBlockSize));
                DGenInput(seed,
                          hostA,
                          descA,
                          hostB,
                          descB,
                          hostC,
                          descC,
                          hostScaleA,
                          hostScaleB,
                          gemm.scaleAMode == Operations::ScaleMode::Separate,
                          gemm.scaleBMode == Operations::ScaleMode::Separate,
                          -1.f,
                          1.f,
                          static_cast<uint>(scaleBlockSize));
            }
            else
            {
                DGenInput(seed, hostA, descA, hostB, descB, hostC, descC);
            }

            if(setIdentity)
            {
                SetIdentityMatrix(hostA, K, M);
                SetIdentityMatrix(hostB, N, K);

                std::fill(hostC.begin(), hostC.end(), static_cast<TD>(0.0));
            }

            auto deviceA = make_shared_device<TA>(hostA);
            auto deviceB = make_shared_device<TB>(hostB);

            std::shared_ptr<TC> deviceC = (notSetC) ? nullptr : make_shared_device(hostC);
            std::shared_ptr<TD> deviceD = make_shared_device<TD>(M * N, TD{});

            std::shared_ptr<uint8_t> deviceScaleA, deviceScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
                deviceScaleA = make_shared_device(hostScaleA);
            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
                deviceScaleB = make_shared_device(hostScaleB);

            // In SingleScale mode, don't need to copy to device
            if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
                hostScaleA = std::vector<uint8_t>{rotatingSingleScaleValue(gemm.scaleTypeA)};

            if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
                hostScaleB = std::vector<uint8_t>{rotatingSingleScaleValue(gemm.scaleTypeB)};

            auto command = std::make_shared<Command>();

            std::vector<size_t> oneStridesN
                = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

            std::vector<size_t> oneStridesT = gemm.literalStrides
                                                  ? std::vector<size_t>({(size_t)0, (size_t)1})
                                                  : std::vector<size_t>({});

            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeA, gemm.transA == "N" ? oneStridesN : oneStridesT)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeB, gemm.transB == "N" ? oneStridesN : oneStridesT)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto mulInputA = tagLoadA;
            auto mulInputB = tagLoadB;

            std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
                tagTensorScaleB, tagLoadScaleB, tagBlockScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                tagTensorScaleA = command->addOperation(rocRoller::Operations::Tensor(
                    2, gemm.scaleTypeA, gemm.transA == "N" ? oneStridesN : oneStridesT));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleA));

                tagBlockScaleA = mulInputA
                    = command->addOperation(rocRoller::Operations::BlockScale(
                        tagLoadA,
                        2,
                        tagLoadScaleA,
                        {1, static_cast<unsigned int>(gemm.scaleBlockSize)}));
            }
            else if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
            {
                tagTensorScaleA
                    = command->addOperation(rocRoller::Operations::Scalar(gemm.scaleTypeA));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Scalar(*tagTensorScaleA));
                tagBlockScaleA = mulInputA = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadA, 0, tagLoadScaleA));
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                tagTensorScaleB = command->addOperation(rocRoller::Operations::Tensor(
                    2, gemm.scaleTypeB, gemm.transB == "N" ? oneStridesN : oneStridesT));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleB));

                tagBlockScaleB = mulInputB
                    = command->addOperation(rocRoller::Operations::BlockScale(
                        tagLoadB,
                        2,
                        tagLoadScaleB,
                        {static_cast<unsigned int>(gemm.scaleBlockSize), 1}));
            }
            else if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
            {
                tagTensorScaleB
                    = command->addOperation(rocRoller::Operations::Scalar(gemm.scaleTypeB));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Scalar(*tagTensorScaleB));
                tagBlockScaleB = mulInputB = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadB, 0, tagLoadScaleB));
            }

            auto tagTensorC = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeC, oneStridesN)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagScalarAlpha
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
            auto tagLoadAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

            auto tagScalarBeta
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
            auto tagLoadBeta
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

            auto tagAB = command->addOperation(
                rocRoller::Operations::T_Mul(mulInputA, mulInputB, dataTypeAcc)); // A * B

            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto                             tagBetaC
                = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, tagLoadC)); // beta * C

            auto tagAlphaAB = execute.addXOp(
                rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)

            rocRoller::Operations::OperationTag tagStoreD;
            if(gemm.betaInFma)
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
            }
            else
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
            }

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            auto tagTensorD = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeD, oneStridesN)); // D
            Operations::OperationTag tagScalarSeed;
            if constexpr(std::is_same_v<TC, TD>)
            {
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));
            }
            else
            {
                Operations::OperationTag tagLoadSeed;
                // If Matrix C and D are of different types, an explicit type conversion is required
                if(srCvtSeed.has_value())
                {
                    tagScalarSeed = command->addOperation(
                        rocRoller::Operations::Scalar(DataType::UInt32)); // alpha
                    tagLoadSeed = command->addOperation(
                        rocRoller::Operations::T_Load_Scalar(tagScalarSeed));
                }

                auto cvtOp = rocRoller::Operations::T_Execute(command->getNextTag());
                // (SR)Convert( alpha * (A * B) + beta * C )
                auto tagCvt
                    = srCvtSeed.has_value()
                          ? cvtOp.addXOp(rocRoller::Operations::E_StochasticRoundingCvt(
                              tagStoreD, tagLoadSeed, dataTypeD))
                          : cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagStoreD, dataTypeD));
                tagStoreD = command->addOperation(std::move(cvtOp));
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));
            }

            auto tagScratch = command->allocateTag();
            command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                      tagScratch,
                                      ArgumentType::Value,
                                      DataDirection::ReadWrite,
                                      rocRoller::SCRATCH);

            Operations::OperationTag tagNumWGs;
            if(gemm.streamK)
            {
                tagNumWGs      = command->allocateTag();
                auto numWGsArg = command->allocateArgument(DataType::UInt32,
                                                           tagNumWGs,
                                                           ArgumentType::Value,
                                                           DataDirection::ReadOnly,
                                                           rocRoller::NUMWGS);
            }

            Operations::OperationTag tagWGM;
            if(gemm.workgroupMappingDim != -1)
            {
                tagWGM      = command->allocateTag();
                auto wgmArg = command->allocateArgument(DataType::Int32,
                                                        tagWGM,
                                                        ArgumentType::Value,
                                                        DataDirection::ReadOnly,
                                                        rocRoller::WGM);
            }

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});

            // TODO: Calculate these values internally based on workgroup sizes.
            params->setManualWavefrontCount(
                {static_cast<uint>(gemm.macM / gemm.waveM / wavetilePerWavefrontM),
                 static_cast<uint>(gemm.macN / gemm.waveN / wavetilePerWavefrontN)});
            params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);
            params->setSplitStoreTileIntoWaveBlocks(gemm.splitStoreTileIntoWaveBlocks);

            params->swizzleScale                  = gemm.swizzleScale;
            params->prefetchScale                 = gemm.prefetchScale;
            params->fuseLoops                     = gemm.fuseLoops;
            params->tailLoops                     = gemm.tailLoops;
            params->allowAmbiguousMemoryNodes     = gemm.allowAmbiguousMemoryNodes;
            params->unrollX                       = gemm.unrollX;
            params->unrollY                       = gemm.unrollY;
            params->unrollK                       = gemm.unrollK;
            params->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
            params->prefetch                      = gemm.prefetch;
            params->prefetchInFlight              = gemm.prefetchInFlight;
            params->prefetchLDSFactor             = gemm.prefetchLDSFactor;
            params->prefetchMixMemOps             = gemm.prefetchMixMemOps;
            params->transposeMemoryAccess.set(LayoutType::MATRIX_A, gemm.transA == "T");
            params->transposeMemoryAccess.set(LayoutType::MATRIX_B, gemm.transB == "T");

            if(gemm.workgroupMappingDim != -1)
            {
                params->workgroupMappingDim = gemm.workgroupMappingDim;
            }

            if(gemm.workgroupRemapXCC)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
                params->workgroupRemapXCC = m_context->targetArchitecture().GetCapability(
                    GPUCapability::DefaultRemapXCCValue);
            }

            if(gemm.loopOverTiles > 0)
            {
                params->loopOverOutputTilesDimensions = {0, 1};
                params->loopOverOutputTilesCoordSizes
                    = {static_cast<uint>(M / gemm.macM), static_cast<uint>(N / gemm.macN)};
                params->loopOverOutputTilesIteratedTiles = 2;
            }

            if(gemm.streamK)
            {
                REQUIRE_ARCH_CAP(GPUCapability::ArchAccUnifiedRegs);

                AssertFatal(
                    numWorkgroupY == 1,
                    "Current scratch space implementation assumes that the kernel is launched "
                    "with numWorkgroupY == 1");

                params->loopOverOutputTilesDimensions = {0, 1};
                params->streamK                       = true;
                params->streamKTwoTile                = gemm.streamKTwoTile;
            }

            auto memoryTypeA = MemoryType::WAVE;
            auto memoryTypeB = MemoryType::WAVE;
            if(gemm.direct2LDSA)
            {
                memoryTypeA = MemoryType::WAVE_Direct2LDS;
            }
            else if(gemm.loadLDSA)
            {
                memoryTypeA = MemoryType::LDS;
            }
            if(gemm.direct2LDSB)
            {
                memoryTypeB = MemoryType::WAVE_Direct2LDS;
            }
            else if(gemm.loadLDSB)
            {
                memoryTypeB = MemoryType::LDS;
            }

            {
                auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    memoryTypeA);
                params->setDimensionInfo(tagLoadA, macTileA);
            }

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.waveK % gemm.scaleBlockSize == 0,
                            fmt::format("waveK: {} must be a multiple of the scale block size: {}",
                                        gemm.waveK,
                                        gemm.scaleBlockSize));
                auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK / gemm.scaleBlockSize},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    gemm.loadLDSScaleA ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(*tagLoadScaleA, macTileAScale);
            }

            {
                auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    memoryTypeB);
                params->setDimensionInfo(tagLoadB, macTileB);
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.waveK % gemm.scaleBlockSize == 0,
                            fmt::format("waveK: {} must be a multiple of the scale block size: {}",
                                        gemm.waveK,
                                        gemm.scaleBlockSize));
                auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK / gemm.scaleBlockSize, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK / gemm.scaleBlockSize, gemm.waveB},
                    gemm.loadLDSScaleB ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(*tagLoadScaleB, macTileBScale);
            }

            {
                auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB});
                params->setDimensionInfo(tagLoadC, macTileC);
            }

            {
                auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    gemm.storeLDSD ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(tagStoreD, macTileD);
            }

            CommandKernel commandKernel(command, testKernelName());

            // TODO Some test have loops, we need to reset the context.
            m_context = createContext();

            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            CommandArguments commandArgs = command->createArguments();

            setCommandTensorArg(commandArgs, tagTensorA, descA, deviceA.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, deviceB.get());
            setCommandTensorArg(commandArgs, tagTensorC, descC, deviceC.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, deviceD.get());

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(K % gemm.scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        K,
                                        gemm.scaleBlockSize));
                TensorDescriptor descAScale(
                    dataTypeA, {size_t(M), size_t(K / gemm.scaleBlockSize)}, gemm.transA);
                setCommandTensorArg(
                    commandArgs, tagTensorScaleA.value(), descAScale, deviceScaleA.get());
            }
            else if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
            {
                commandArgs.setArgument(*tagTensorScaleA, ArgumentType::Value, hostScaleA[0]);
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(K % gemm.scaleBlockSize == 0,
                            fmt::format("K: {} must be a multiple of the scale block size: {}",
                                        K,
                                        gemm.scaleBlockSize));
                TensorDescriptor descBScale(
                    dataTypeB, {size_t(K / gemm.scaleBlockSize), size_t(N)}, gemm.transB);
                setCommandTensorArg(
                    commandArgs, tagTensorScaleB.value(), descBScale, deviceScaleB.get());
            }
            else if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
            {
                commandArgs.setArgument(*tagTensorScaleB, ArgumentType::Value, hostScaleB[0]);
            }

            commandArgs.setArgument(tagScalarAlpha, ArgumentType::Value, alpha);
            commandArgs.setArgument(tagScalarBeta, ArgumentType::Value, beta);
            if(srCvtSeed.has_value())
                commandArgs.setArgument(tagScalarSeed, ArgumentType::Value, srCvtSeed.value());

            // Create scratch space
            if(gemm.streamK)
            {
                commandArgs.setArgument(tagNumWGs, ArgumentType::Value, gemm.numWGs);
            }

            auto scratchSpaceRequired
                = commandKernel.scratchSpaceRequired(commandArgs.runtimeArguments());
            auto deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            commandArgs.setArgument(tagScratch, ArgumentType::Value, deviceScratch.get());

            if(gemm.workgroupMappingDim != -1)
            {
                commandArgs.setArgument(tagWGM, ArgumentType::Value, gemm.workgroupMappingValue);
            }

            // Host result
            std::vector<TD> h_result(M * N, TD{});
            if(gemm.scaleAMode != Operations::ScaleMode::None
               || gemm.scaleBMode != Operations::ScaleMode::None)
            {
                rocRoller::ScaledCPUMM(h_result,
                                       hostC,
                                       hostA,
                                       hostB,
                                       hostScaleA,
                                       hostScaleB,
                                       M,
                                       N,
                                       K,
                                       alpha,
                                       beta,
                                       gemm.transA == "T",
                                       gemm.transB == "T",
                                       gemm.scaleBlockSize,
                                       gemm.scaleTypeA,
                                       gemm.scaleTypeB);
            }
            else if constexpr(std::is_same_v<TC, TD>)
            {
                rocRoller::CPUMM(h_result,
                                 hostC,
                                 hostA,
                                 hostB,
                                 M,
                                 N,
                                 K,
                                 alpha,
                                 beta,
                                 gemm.transA == "T",
                                 gemm.transB == "T");
            }
            else
            {
                std::vector<TC> hostD(M * N, TC{});
                rocRoller::CPUMM(hostD,
                                 hostC,
                                 hostA,
                                 hostB,
                                 M,
                                 N,
                                 K,
                                 alpha,
                                 beta,
                                 gemm.transA == "T",
                                 gemm.transB == "T");
                ASSERT_EQ(hostD.size(), h_result.size());
                bool const isSRConversion = srCvtSeed.has_value();
                for(size_t i = 0; i < hostD.size(); i++)
                {
                    if(isSRConversion)
                    {
                        // SR conversion currently only supports F32 to FP8/BF8
                        AssertFatal((std::is_same_v<TC, float>),
                                    "Source type of SR conversion only accepts float");
                        AssertFatal((std::is_same_v<TD, FP8>) || (std::is_same_v<TD, BF8>),
                                    "Destionation type of SR conversion can only be FP8/BF8");

                        int constexpr exp_width      = std::is_same_v<TD, FP8> ? 4 : 5;
                        int constexpr mantissa_width = 7 - exp_width;
                        bool constexpr is_bf8        = std::is_same_v<TD, BF8>;

                        auto const f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);

                        if(f8Mode == rocRoller::F8Mode::NaNoo)
                        {
                            h_result[i].data = DataTypes::cast_to_f8<mantissa_width,
                                                                     exp_width,
                                                                     float,
                                                                     false /* is_ocp */,
                                                                     is_bf8,
                                                                     true /*negative_zero_nan*/,
                                                                     true /*clip*/>(
                                hostD[i],
                                true /* is stochastic rounding? */,
                                srCvtSeed.value() /* seed for stochastic rounding */);
                        }
                        else
                        {
                            h_result[i].data = DataTypes::cast_to_f8<mantissa_width,
                                                                     exp_width,
                                                                     float,
                                                                     true /* is_ocp */,
                                                                     is_bf8,
                                                                     true /*negative_zero_nan*/,
                                                                     true /*clip*/>(
                                hostD[i],
                                true /* is stochastic rounding? */,
                                srCvtSeed.value() /* seed for stochastic rounding */);
                        }
                    }
                    else
                        h_result[i] = TD(hostD[i]);
                }
            }

            // Device result
            std::vector<TD> d_result(M * N);

            for(int iteration = 0; iteration < numIters; ++iteration)
            {
                ASSERT_THAT(hipMemset(deviceD.get(), 0, M * N * sizeof(TD)), HasHipSuccess(0));
                ASSERT_THAT(hipMemset(deviceScratch.get(), 0, scratchSpaceRequired),
                            HasHipSuccess(0));

                commandKernel.launchKernel(commandArgs.runtimeArguments());

                ASSERT_THAT(
                    hipMemcpy(
                        d_result.data(), deviceD.get(), M * N * sizeof(TD), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

                auto tol = gemmAcceptableError<TA, TB, TD>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(d_result, h_result, tol);
                Log::info("RNorm is {} (acceptable {}, iteration {})",
                          res.relativeNormL2,
                          res.acceptableError.relativeL2Tolerance,
                          iteration);

                if(debuggable && !res.ok)
                {
                    for(size_t i = 0; i < M; i++)
                    {
                        for(size_t j = 0; j < N; j++)
                        {
                            auto a = d_result[i * N + j];
                            auto b = h_result[i * N + j];
                            if((a - b) * (a - b) / (b * b)
                               > res.acceptableError.relativeL2Tolerance)
                            {
                                std::cout << std::setw(8) << i << std::setw(8) << j //
                                          << std::setw(16) << std::scientific << a //
                                          << std::setw(16) << std::scientific << b //
                                          << std::setw(16) << std::scientific << a - b //
                                          << std::endl;
                            }
                        }
                    }
                }
                EXPECT_TRUE(res.ok) << res.message();
            }
        }

        template <typename TA>
        void basicGEMMMixed(rocRoller::DataType typeB, GEMMProblem const& problem)
        {
            if(typeB == rocRoller::DataType::FP8)
                basicGEMM<TA, FP8, float>(problem);
            else if(typeB == rocRoller::DataType::BF8)
                basicGEMM<TA, BF8, float>(problem);
            else if(typeB == rocRoller::DataType::FP6)
                basicGEMM<TA, FP6, float>(problem);
            else if(typeB == rocRoller::DataType::BF6)
                basicGEMM<TA, BF6, float>(problem);
            else if(typeB == rocRoller::DataType::FP4)
                basicGEMM<TA, FP4, float>(problem);
            else
                Throw<FatalError>("Invalid type.");
        }

        void basicGEMMMixed(rocRoller::DataType typeA,
                            rocRoller::DataType typeB,
                            GEMMProblem const&  problem)
        {
            if(typeA == rocRoller::DataType::FP8)
                basicGEMMMixed<FP8>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF8)
                basicGEMMMixed<BF8>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP6)
                basicGEMMMixed<FP6>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF6)
                basicGEMMMixed<BF6>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP4)
                basicGEMMMixed<FP4>(typeB, problem);
            else
                Throw<FatalError>("Invalid type.");
        }
    };

    class GEMMTestGPU : public BaseGEMMContextFixture<>
    {
    };

    class GEMMJammedTestGPU : public BaseGEMMContextFixture<>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMF16TestGPU
        : public BaseGEMMContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A & B type, M tile size, (transA, transB), DirectLDS A & B
    class GEMMDirectLDSTestGPU
        : public BaseGEMMContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>, bool, bool>>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMF8F6F4TestGPU
        : public BaseGEMMContextFixture<
              std::tuple<rocRoller::DataType, int, std::pair<std::string, std::string>>>
    {
    };

    class GEMMF8TestGPU : public BaseGEMMContextFixture<>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MixedGEMMF8F6F4TestGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, Load A scale though LDS, Load B scale through LDS, (transA, transB)
    class ScaledMixedGEMMF8F6F4TestGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   rocRoller::Operations::ScaleMode,
                                                   rocRoller::Operations::ScaleMode,
                                                   bool,
                                                   bool,
                                                   std::pair<std::string, std::string>>>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMTestWMMAGPU
        : public BaseGEMMContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A & B type, K tile size, (transA, transB)
    class GEMMTestWMMAF16AccumGPU
        : public BaseGEMMContextFixture<
              std::tuple<std::pair<rocRoller::DataType, int>, std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MixedGEMMTestWMMAGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, K tile size, (transA, transB)
    class MixedGEMMTestF8F6F4WMMAGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType,
                                                   rocRoller::DataType,
                                                   int,
                                                   std::pair<std::string, std::string>>>
    {
    };

    // Params are: A type, B type, scaleTypeA, scaleTypeB, K tile size, (scaleAMode, scaleBMode, scaleBlockSize), (transA, transB)
    class ScaledMixedGEMMTestF8F6F4WMMAGPU
        : public BaseGEMMContextFixture<std::tuple<
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              rocRoller::DataType,
              int,
              std::tuple<rocRoller::Operations::ScaleMode, rocRoller::Operations::ScaleMode, int>,
              std::pair<std::string, std::string>>>
    {
    };

    class GEMMTestLargeMacroTileGPU : public BaseGEMMContextFixture<>
    {
    };

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_P(GEMMTestSuite, GPU_GEMM_Schedulers)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.macK = 8;

        // TODO: Re-enable LDS once LDS deallocations are fixed
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;

        auto settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Sequential);
        basicGEMM<float>(gemm);
        std::string seq = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::RoundRobin);
        basicGEMM<float>(gemm);
        std::string rr = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
        basicGEMM<float>(gemm);
        std::string coop_nop = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Priority);
        basicGEMM<float>(gemm);
        std::string priority_nop = m_context->instructions()->toString();

        EXPECT_NE(NormalizedSource(seq), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(coop_nop), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(priority_nop), NormalizedSource(rr));
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_LDS_Padded)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.transA    = "T";
        gemm.transB    = "N";

        // This kernel uses 128bit buffer_load and ds_write
        // instructions; so each wave loads 16 * 64 bytes per
        // instruction.
        gemm.padA = {32 * 64, 4};
        gemm.padB = {16 * 64, 4};
        basicGEMM<float>(gemm);

        auto instructions = m_context->instructions()->toString();
        auto ldsOffsets   = NonZeroDSReadOffsets("ds_read_b32", instructions);

        // With no padding in A, the LDS buffer for B would start at
        //
        //   B[0] = WGTS M * WGTS K * sizeof(float)
        //
        // so make sure this isn't in the read offsets!
        EXPECT_FALSE(ldsOffsets.contains(gemm.macM * gemm.macK * sizeof(float)));
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Workgroup_Mapping)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Workgroup_Mapping_XCC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
        GEMMProblem gemm;
        gemm.workgroupMappingDim   = 0;
        gemm.workgroupMappingValue = 6;
        gemm.workgroupRemapXCC     = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_LDS_Larger)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.macM             = 128;
        gemm.macN             = 256;
        gemm.loadPathA        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath        = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;
        gemm.prefetchInFlight = 1;
        auto maxLDS = m_context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
        auto bytesPerElement = sizeof(float);
        auto ldsA            = gemm.macM * gemm.macK * bytesPerElement * gemm.prefetchInFlight;
        auto ldsB            = gemm.macK * gemm.macN * bytesPerElement * gemm.prefetchInFlight;
        auto ldsD            = gemm.waveM * gemm.waveN * bytesPerElement;

        if(ldsA + ldsB + ldsD <= maxLDS)
        {
            basicGEMM<float>(gemm);
        }
        else
        {
            GTEST_SKIP() << "LDS usage exceeds maxLDS.";
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_BetaZero)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP32_NotSetC)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm, false, false, 1, true);
    }

    TEST_P(GEMMTestSuite, DISABLED_GPU_GEMM_DataType_FP32_MultipleOutputTiles)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.storePath     = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.loopOverTiles = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_A)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_B)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_NoLDS_AB)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToVGPR;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_16x16x32_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        basicGEMM<BF8, BF8, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_Conversion_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (FP8) = Convert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_Conversion_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (BF8) = Convert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_StochasticRounding_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (FP8) = StochasticRoundingConvert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm,
                                        /* debuggable  */ false,
                                        /* setIdentity */ false,
                                        /* numIters    */ 1,
                                        /* notSetC     */ false,
                                        /* seed        */ 56789u);

        // Check stochastic rounding instruction has be generated
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string const generatedCode = m_context->instructions()->toString();
            EXPECT_NE(generatedCode.find("v_cvt_sr_fp8_f32"), std::string::npos);
            EXPECT_EQ(generatedCode.find("v_cvt_pk_fp8_f32"), std::string::npos);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_StochasticRounding_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8NT{};
        // D (BF8) = StochasticRoundingConvert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm,
                                        /* debuggable  */ false,
                                        /* setIdentity */ false,
                                        /* numIters    */ 1,
                                        /* notSetC     */ false,
                                        /* seed        */ 56789u);

        // Check stochastic rounding instruction has be generated
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string const generatedCode = m_context->instructions()->toString();
            EXPECT_NE(generatedCode.find("v_cvt_sr_bf8_f32"), std::string::npos);
            EXPECT_EQ(generatedCode.find("v_cvt_pk_bf8_f32"), std::string::npos);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Scaled_Prefetch_MX_F8_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToVGPR;

        gemm.unrollK           = 2;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 2;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        basicGEMM<FP8, FP8, float>(gemm);
    }

    void CheckGEMMF8TN(rocRoller::ContextPtr m_context)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string generatedCode = m_context->instructions()->toString();

            EXPECT_EQ(countSubstring(generatedCode, "buffer_load"), 3);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "), 2);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), 2);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), 1);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b32 "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_read"), 4);
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64 "), 4);
        }
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<FP8, FP8, float>(gemm);
        CheckGEMMF8TN(m_context);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_BF8_16x16x32_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        auto gemm = GEMMProblemF8TN{};
        basicGEMM<BF8, BF8, float>(gemm);
        CheckGEMMF8TN(m_context);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP8_LargerLDS_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm             = GEMMProblemF8F6F4{32, 32, 64};
        gemm.macM             = 128;
        gemm.macN             = 128;
        gemm.macK             = 256;
        gemm.loadPathA        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB        = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.prefetchInFlight = 2;

        basicGEMM<FP8, FP8, float>(gemm);
    }

    static void CheckNumDwordx4(std::string generatedCode,
                                const int   numBitsPerElementAB,
                                const int   macM,
                                const int   macN,
                                const int   macK,
                                const int   workgroupSizeTotal)
    {
        auto const numBitsPerDwordx4   = 4 * 4 * 8;
        auto const numBitsPerElementC  = 32;
        auto const numBufferLoadsForAB = ((macM * macK + macN * macK) * numBitsPerElementAB)
                                         / workgroupSizeTotal / numBitsPerDwordx4;
        auto const numBufferLoadsForC
            = ((macM * macN) * numBitsPerElementC) / workgroupSizeTotal / numBitsPerDwordx4;

        EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4"),
                  numBufferLoadsForAB + numBufferLoadsForC);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_FP8_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP8, FP8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_BF8_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<BF8, BF8, float>(gemm);

        auto const  numBitsPerElementAB = 8;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_LoadPath_Direct2LDS_FP4_MT256x256x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm      = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m         = 512;
        gemm.n         = 256;
        gemm.k         = 512;
        gemm.macM      = 256;
        gemm.macN      = 256;
        gemm.macK      = 128;
        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA    = "T";
        gemm.transB    = "N";

        basicGEMM<FP4, FP4, float>(gemm);

        auto const  numBitsPerElementAB = 4;
        std::string generatedCode       = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        CheckNumDwordx4(generatedCode,
                        numBitsPerElementAB,
                        gemm.macM,
                        gemm.macN,
                        gemm.macK,
                        gemm.workgroupSizeX * gemm.workgroupSizeY);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MT256x256x128_LDSSwizzle)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{32, 32, 64};
        gemm.m              = 512;
        gemm.n              = 256;
        gemm.k              = 512;
        gemm.macM           = 256;
        gemm.macN           = 256;
        gemm.macK           = 128;
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;
        gemm.storePath      = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
        gemm.transA         = "T";
        gemm.transB         = "N";
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);

        // LDS swizzle uses XOR-based permutation; expect exactly 12 v_xor_b32.
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "v_xor_b32"), 12);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MI16x16x128_MT64x64x256_LDSSwizzle)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{16, 16, 128};
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDS;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDS;
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);

        // LDS swizzle uses XOR-based permutation; expect exactly 12 v_xor_b32.
        std::string generatedCode = m_context->instructions()->toString();
        EXPECT_EQ(countSubstring(generatedCode, "ds_write"), 0);
        EXPECT_EQ(countSubstring(generatedCode, "v_xor_b32"), 12);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_FP4_MI16x16x128_MT64x64x256_LDSSwizzle_ViaVGPR)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm           = GEMMProblemF8F6F4{16, 16, 128};
        gemm.loadPathA      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.ldsSwizzleMode = LDSBankSwizzleMode::Swizzle;

        basicGEMM<FP4, FP4, float>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_AllLDS)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_96x256)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 192;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 96;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_StoreDWave)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        gemm.splitStoreTileIntoWaveBlocks = true;
        basicGEMM<Half>(gemm);
        auto instructions0 = output();
        EXPECT_EQ(NonZeroDSReadOffsets("ds_read_b128", instructions0), std::set<int>{1024});

        gemm.splitStoreTileIntoWaveBlocks = false;
        basicGEMM<Half>(gemm);
        auto instructions1 = output();
        EXPECT_EQ(NonZeroDSReadOffsets("ds_read_b128", instructions1), std::set<int>{64});
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_DataType_FP16_AllLDS_Debug)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.storePath = SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestSuite, GPU_GEMM_Scaled_LDS_MX_F8_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
        REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
        auto gemm = GEMMProblemF8F6F4{32, 32, 64};

        gemm.macM = 64;
        gemm.macN = 256;
        gemm.macK = 128;
        gemm.m    = 2 * gemm.macM;
        gemm.n    = 3 * gemm.macN;
        gemm.k    = 4 * gemm.macK;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadPathA      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadPathB      = SolutionParams::LoadPath::BufferToVGPR;
        gemm.loadScalePathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        gemm.loadScalePathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.scaleTypeA = DataType::E8M0;
        gemm.scaleTypeB = DataType::E8M0;

        gemm.scaleBlockSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultScaleBlockSize);

        basicGEMM<FP8, FP8, float>(gemm);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, GEMMTestSuite, currentGPUISA());

    // ========================================================================
    // GEMMSchedulerRandomTestSuite
    // ========================================================================

    // Params are: random seed value
    class GEMMSchedulerRandomTestSuite : public BaseGEMMContextFixture<std::tuple<int>>
    {
    };

    // Test to verify different random seeds produce different instruction sequences
    TEST_P(GEMMSchedulerRandomTestSuite, GPU_GEMM_Schedulers_Random)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = setup_GEMMF8F6F4(waveM, waveN, waveK);

        std::tie(problem.transA, problem.transB) = transOp;

        // TODO: enable non-TN F6 tests
        auto const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        auto const elementBitsB = DataTypeInfo::Get(typeB).elementBits;

        basicGEMMMixed(typeA, typeB, problem);

        auto const mfma{fmt::format("v_mfma_f32_{}x{}x{}_f8f6f4", waveM, waveN, waveK)};

        std::string modifierA = "defaultModiferA";
        std::string modifierB = "defaultModiferB";

        if(typeA == rocRoller::DataType::FP8)
            modifierA = "cbsz:0b000";
        else if(typeA == rocRoller::DataType::BF8)
            modifierA = "cbsz:0b001";
        else if(typeA == rocRoller::DataType::FP6)
            modifierA = "cbsz:0b010";
        else if(typeA == rocRoller::DataType::BF6)
            modifierA = "cbsz:0b011";
        else if(typeA == rocRoller::DataType::FP4)
            modifierA = "cbsz:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeA));

        if(typeB == rocRoller::DataType::FP8)
            modifierB = "blgp:0b000";
        else if(typeB == rocRoller::DataType::BF8)
            modifierB = "blgp:0b001";
        else if(typeB == rocRoller::DataType::FP6)
            modifierB = "blgp:0b010";
        else if(typeB == rocRoller::DataType::BF6)
            modifierB = "blgp:0b011";
        else if(typeB == rocRoller::DataType::FP4)
            modifierB = "blgp:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeB));

        check_mfma_f8f6f4(m_context, mfma, modifierA + " " + modifierB);
    }

    TEST_P(ScaledMixedGEMMF8F6F4TestGPU, GPU_ScaledMixedBasicGEMMF8F6F4)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);

        auto [typeA, typeB, MFMAK, scaleAMode, scaleBMode, loadLDSScaleA, loadLDSScaleB, transOp]
            = std::get<1>(GetParam());

        int waveM = (MFMAK == 128) ? 16 : 32;
        int waveN = (MFMAK == 128) ? 16 : 32;
        int waveK = MFMAK;

        auto problem = setup_GEMMF8F6F4(waveM, waveN, waveK);

        std::tie(problem.transA, problem.transB) = transOp;

        // TODO: enable non-TN F6 tests
        auto const elementBitsA = DataTypeInfo::Get(typeA).elementBits;
        auto const elementBitsB = DataTypeInfo::Get(typeB).elementBits;

        problem.scaleAMode = scaleAMode;
        problem.scaleBMode = scaleBMode;

        problem.scaleTypeA = DataType::E8M0;
        problem.scaleTypeB = DataType::E8M0;

        problem.loadLDSScaleA = loadLDSScaleA;
        problem.loadLDSScaleB = loadLDSScaleB;

        if(loadLDSScaleA
           && (scaleAMode == rocRoller::Operations::ScaleMode::None
               || scaleAMode == rocRoller::Operations::ScaleMode::SingleScale))
            GTEST_SKIP() << "Meaningless combination of LoadLDSScaleA and ScaleA";
        if(loadLDSScaleB
           && (scaleBMode == rocRoller::Operations::ScaleMode::None
               || scaleBMode == rocRoller::Operations::ScaleMode::SingleScale))
            GTEST_SKIP() << "Meaningless combination of LoadLDSScaleB and ScaleB";

        if(scaleAMode == rocRoller::Operations::ScaleMode::Separate
           || scaleBMode == rocRoller::Operations::ScaleMode::Separate)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasBlockScaling32);
            problem.scaleBlockSize = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultScaleBlockSize);
        }

        basicGEMMMixed(typeA, typeB, problem);
    }

    TEST_P(GEMMTestWMMAGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeABAndWaveK, transOp] = std::get<1>(GetParam());
        auto [typeAB, waveK]           = typeABAndWaveK;

        switch(waveK)
        {
        case 4:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x4_f32);
            break;
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f16);
            break;
        case 32:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x32_f16);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        AssertFatal((waveK == 4) || (waveK == 16) || (waveK == 32),
                    "Invalid waveK value.",
                    ShowValue(waveK));

        GEMMProblem gemm;
        gemm.macK = 8;

        if(typeAB == DataType::Half)
        {
            basicGEMM<Half, Half, float>(gemm);
        }
        else if(typeAB == DataType::BFloat16)
        {
            basicGEMM<BFloat16, BFloat16, float>(gemm);
        }
        else if(typeAB == DataType::Float)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x4_f32);
            basicGEMM<float, float, float>(gemm);
        }
        else
        {
            Throw<FatalError>("Invalid type.", ShowValue(typeAB));
        }
    }

    TEST_P(GEMMTestWMMAF16AccumGPU, GPU_BasicGEMMF16Accum)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_F16_ACC);
        auto [dataTypeAndWaveK, transOp] = std::get<1>(GetParam());
        auto [dataType, waveK]           = dataTypeAndWaveK;

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x16_f16);
            break;
        case 32:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f16_16x16x32_f16);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = waveK;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        std::tie(gemm.transA, gemm.transB) = transOp;

        if(dataType == DataType::Half)
        {
            basicGEMM<Half, Half, Half, Half, Half>(gemm);
        }
        else if(dataType == DataType::BFloat16)
        {
            basicGEMM<BFloat16, BFloat16, BFloat16, BFloat16, BFloat16>(gemm);
        }
        else
        {
            Throw<FatalError>("Invalid type.", ShowValue(dataType));
        }
    }

    TEST_P(MixedGEMMTestWMMAGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA);
        auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());

        switch(waveK)
        {
        case 16:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x16_f8);
            break;
        case 64:
            REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f32_16x16x64_f8);
            break;
        default:
            Throw<FatalError>("Invalid waveK value.", ShowValue(waveK));
        }

        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = waveK;
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        std::tie(gemm.transA, gemm.transB) = transOp;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    TEST_P(MixedGEMMTestF8F6F4WMMAGPU, GPU_BasicGEMM)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasWMMA_f8f6f4);
        auto [typeA, typeB, waveK, transOp] = std::get<1>(GetParam());
        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        GEMMProblem gemm = setup_GEMMF8F6F4(16, 16, waveK);
        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        // TODO: change setup_GEMMF8F6F4 to query wavefrontSize
        gemm.workgroupSizeX                = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY                = 2;
        std::tie(gemm.transA, gemm.transB) = transOp;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    TEST_P(ScaledMixedGEMMTestF8F6F4WMMAGPU, GPU_ScaledMixedBasicGEMMF8F6F4)
    {
        REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasWMMA_scale_f8f6f4);
        auto [typeA, typeB, scaleTypeA, scaleTypeB, waveK, scaleModesAndSize, transOp]
            = std::get<1>(GetParam());

        AssertFatal(waveK == 128, "Invalid waveK value.", ShowValue(waveK));

        auto gemm = setup_GEMMF8F6F4(16, 16, waveK);

        std::tie(gemm.transA, gemm.transB) = transOp;

        gemm.wavefrontSize
            = m_context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;
        gemm.scaleTypeA     = scaleTypeA;
        gemm.scaleTypeB     = scaleTypeB;

        std::tie(gemm.scaleAMode, gemm.scaleBMode, gemm.scaleBlockSize) = scaleModesAndSize;

        basicGEMMMixed(typeA, typeB, gemm);
    }

    TEST_P(GEMMTestLargeMacroTileGPU, DISABLED_GPU_BasicGEMM)
    {
        // NOTE: This test takes hours to finish
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        GEMMProblem problem{.m = 1024, .n = 1024, .k = 256, .macM = 256, .macN = 256};
        basicGEMM<float>(problem);

        // To see runtime of kernel generation, compile code with timer enabled and
        // std::cout << TimerPool::summary() << std::endl;
    }

    TEST_P(GEMMTestLargeMacroTileGPU, GPU_BasicGEMMF8F6F4)
    {
        // NOTE: This test takes about 13 seconds (without enabling Unroll) to
        // finish when FuseLoops orders all pairs of memory nodes one by one.
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto gemm = setup_GEMMF8F6F4(32, 32, 64);
        gemm.m    = 512;
        gemm.n    = 256;
        gemm.k    = 512;

        gemm.macM = 256;
        gemm.macN = 256;
        gemm.macK = 128;

        gemm.loadLDSA = true;
        gemm.loadLDSB = true;

        // Use unrollK will significantly increase the kernel generation time.
        // To enable unrollK, maxVGPR has to be increased as well.
        //
        //gemm.unrollK = 2;
        //setKernelOptions({.maxVGPRs = 1024});

        basicGEMM<FP8, FP8, float>(gemm);

        // To see runtime of kernel generation, compile code with timer enabled and
        // std::cout << TimerPool::summary() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, GEMMTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMTestLargeMacroTile, GEMMTestLargeMacroTileGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(
        GEMMTest,
        GEMMSchedulerRandomTestSuite,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(2, 4, 8, 314, 1729))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMDirectLDSTest,
        GEMMDirectLDSTestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::Float),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")),
                               ::testing::Values(true, false),
                               ::testing::Values(true, false))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMF8F6F4Test,
        GEMMF8F6F4TestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(GEMMF8Test, GEMMF8TestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMJammedTest, GEMMJammedTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(
        GEMMMixedF8F6F4Test,
        MixedGEMMF8F6F4TestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        ScaledMixedGEMMTest,
        ScaledMixedGEMMF8F6F4TestGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(64, 128),
                               ::testing::Values(rocRoller::Operations::ScaleMode::SingleScale,
                                                 rocRoller::Operations::ScaleMode::Separate),
                               ::testing::Values(rocRoller::Operations::ScaleMode::SingleScale,
                                                 rocRoller::Operations::ScaleMode::Separate),
                               ::testing::Values(false, true),
                               ::testing::Values(false, true),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA,
        GEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA1250,
        GEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::Float, /*waveK*/ 4)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA,
        GEMMTestWMMAF16AccumGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 16),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 16)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMTestWMMA1250,
        GEMMTestWMMAF16AccumGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(std::make_pair(rocRoller::DataType::Half, /*waveK*/ 32),
                                  std::make_pair(rocRoller::DataType::BFloat16, /*waveK*/ 32)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MixedGEMMTestWMMA,
        MixedGEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(16),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));
    INSTANTIATE_TEST_SUITE_P(
        MixedGEMMTestWMMA1250,
        MixedGEMMTestWMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(rocRoller::DataType::FP8, rocRoller::DataType::BF8),
                ::testing::Values(/*waveK*/ 64),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        MixedGEMMTestWMMA1250,
        MixedGEMMTestF8F6F4WMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(rocRoller::DataType::FP8,
                                                 rocRoller::DataType::BF8,
                                                 rocRoller::DataType::FP6,
                                                 rocRoller::DataType::BF6,
                                                 rocRoller::DataType::FP4),
                               ::testing::Values(/*waveK*/ 128),
                               ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                                 std::pair<std::string, std::string>("N", "T"),
                                                 std::pair<std::string, std::string>("T", "N"),
                                                 std::pair<std::string, std::string>("T", "T")))));

    INSTANTIATE_TEST_SUITE_P(
        ScaledMixedGEMMTestWMMA1250,
        ScaledMixedGEMMTestF8F6F4WMMAGPU,
        ::testing::Combine(
            currentGPUISA(),
            ::testing::Combine(
                ::testing::Values(rocRoller::DataType::FP8,
                                  rocRoller::DataType::BF8,
                                  rocRoller::DataType::FP6,
                                  rocRoller::DataType::BF6,
                                  rocRoller::DataType::FP4),
                ::testing::Values(rocRoller::DataType::FP8,
                                  rocRoller::DataType::BF8,
                                  rocRoller::DataType::FP6,
                                  rocRoller::DataType::BF6,
                                  rocRoller::DataType::FP4),
                ::testing::Values(rocRoller::DataType::E8M0),
                ::testing::Values(rocRoller::DataType::E8M0),
                ::testing::Values(/*waveK*/ 128),
                ::testing::Values(
                    std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                        Operations::ScaleMode::Separate, Operations::ScaleMode::Separate, 32),
                    std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                        Operations::ScaleMode::Separate, Operations::ScaleMode::Separate, 16),
                    std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                        Operations::ScaleMode::Separate, Operations::ScaleMode::SingleScale, 32),
                    std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                        Operations::ScaleMode::SingleScale, Operations::ScaleMode::Separate, 32),
                    std::tuple<Operations::ScaleMode, Operations::ScaleMode, int>(
                        Operations::ScaleMode::SingleScale,
                        Operations::ScaleMode::SingleScale,
                        32)),
                ::testing::Values(std::pair<std::string, std::string>("N", "N"),
                                  std::pair<std::string, std::string>("N", "T"),
                                  std::pair<std::string, std::string>("T", "N"),
                                  std::pair<std::string, std::string>("T", "T")))));
}
