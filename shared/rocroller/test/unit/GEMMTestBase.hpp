// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include "GPUContextFixture.hpp"

#include <common/CommonGraphs.hpp>
#include <common/GEMMProblem.hpp>
#include <common/mxDataGen.hpp>

namespace GEMMTests
{
    std::set<int> NonZeroDSReadOffsets(std::string const& instruction, std::string const& s);
    std::set<int> Direct2LDSWriteStrides(std::string const& s);

    template <typename T>
    concept isF8 = std::is_same_v<T, rocRoller::FP8> || std::is_same_v<T, rocRoller::BF8>;

    template <typename T>
    concept isF6F4 = std::is_same_v<T, rocRoller::FP6> || std::is_same_v<T, rocRoller::BF6> || std::
        is_same_v<T, rocRoller::FP4>;

    template <typename... Ts>
    class BaseGEMMContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<rocRoller::GPUArchitectureTarget, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            auto device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

        int m_scaleValueIndex = 0;

    public:
        uint8_t rotatingSingleScaleValue(rocRoller::DataType scaleType)
        {
            using namespace rocRoller;
            AssertFatal(isScaleType(scaleType));
            const std::vector<float> scaleValues{1.0, 2.0, 4.0, 8.0};
            m_scaleValueIndex = (++m_scaleValueIndex) % scaleValues.size();
            return floatToScale(scaleType, scaleValues[m_scaleValueIndex]);
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
            using namespace rocRoller;
            REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA, GPUCapability::HasWMMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ANY_OF_ARCH_CAP(GPUCapability::HasMFMA_fp8,
                                        GPUCapability::HasWMMA_f32_16x16x16_f8);
            }

            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            if((isF8<TA> || isF8<TB>)&&(gemm.waveK >= 64))
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            if(gemm.scaleAMode != Operations::ScaleMode::None
               || gemm.scaleBMode != Operations::ScaleMode::None)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_scale_f8f6f4);
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

            // CommonGraphs::GEMM builds the command graph, kernel parameters, and arguments.
            auto problem = gemm;
            if constexpr(!std::is_same_v<TC, TD>)
                problem.useSrCvt = srCvtSeed.has_value();

            rocRollerTest::Graphs::GEMM gemmGraph(dataTypeA, dataTypeB, dataTypeC, dataTypeD);
            gemmGraph.setProblem(problem);
            auto command = gemmGraph.getCommand();

            auto params = gemmGraph.getCommandParameters();
            if(gemm.workgroupRemapXCC)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasXCC);
                params->workgroupRemapXCC = m_context->targetArchitecture().GetCapability(
                    GPUCapability::DefaultRemapXCCValue);
            }

            CommandKernel commandKernel(command, testKernelName());

            // TODO Some tests have loops, we need to reset the context.
            m_context = createContext();

            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            auto [commandArgs,
                  deviceA,
                  deviceB,
                  deviceC,
                  deviceD,
                  hostA,
                  hostB,
                  hostC,
                  hostScaleA,
                  hostScaleB]
                = gemmGraph.getCommandArguments<TA, TB, TC, TD, ACC>(
                    setIdentity, notSetC, srCvtSeed);

            // SingleScale arg values require fixture-specific rotating scale value generation.
            if(gemm.scaleAMode == Operations::ScaleMode::SingleScale)
            {
                hostScaleA                  = {rotatingSingleScaleValue(gemm.scaleTypeA)};
                auto [scaleTagA, scaleTagB] = gemmGraph.getABScaleTags();
                commandArgs.setArgument(*scaleTagA, ArgumentType::Value, hostScaleA[0]);
            }
            if(gemm.scaleBMode == Operations::ScaleMode::SingleScale)
            {
                hostScaleB                  = {rotatingSingleScaleValue(gemm.scaleTypeB)};
                auto [scaleTagA, scaleTagB] = gemmGraph.getABScaleTags();
                commandArgs.setArgument(*scaleTagB, ArgumentType::Value, hostScaleB[0]);
            }

            // Create scratch space
            size_t scratchSpaceRequired[static_cast<int>(Operations::ScratchPolicy::Count)];
            std::shared_ptr<uint8_t>
                deviceScratch[static_cast<int>(Operations::ScratchPolicy::Count)];
            std::fill(std::begin(scratchSpaceRequired), std::end(scratchSpaceRequired), 0);
            std::fill(std::begin(deviceScratch), std::end(deviceScratch), nullptr);
            if(gemm.streamK)
            {
                commandArgs.setArgument(gemmGraph.getNumWGsTag(), ArgumentType::Value, gemm.numWGs);
                auto scratchTags = gemmGraph.getScratchTags();
                for(int i = 0; i < static_cast<int>(Operations::ScratchPolicy::Count); ++i)
                {
                    auto policy             = static_cast<Operations::ScratchPolicy>(i);
                    scratchSpaceRequired[i] = commandKernel.scratchSpaceRequired(
                        policy, commandArgs.runtimeArguments());
                    if(scratchSpaceRequired[i] > 0)
                    {
                        deviceScratch[i] = make_shared_device<uint8_t>(scratchSpaceRequired[i], 0);
                        commandArgs.setArgument(
                            scratchTags.at(policy), ArgumentType::Value, deviceScratch[i].get());
                    }
                }
            }

            if(gemm.workgroupMappingDim != -1)
            {
                commandArgs.setArgument(
                    gemmGraph.getWGMTag(), ArgumentType::Value, gemm.workgroupMappingValue);
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
                if(iteration == 0)
                {
                    for(int i = 0; i < static_cast<int>(Operations::ScratchPolicy::Count); ++i)
                    {
                        if(scratchSpaceRequired[i] > 0)
                        {
                            ASSERT_THAT(
                                hipMemset(deviceScratch[i].get(), 0, scratchSpaceRequired[i]),
                                HasHipSuccess(0));
                        }
                    }
                }

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

                // Verify ZeroedBeforeAndAfter scratch is all zeros after kernel execution
                auto zeroedIdx
                    = static_cast<size_t>(Operations::ScratchPolicy::ZeroedBeforeAndAfter);
                if(scratchSpaceRequired[zeroedIdx] > 0)
                {
                    std::vector<uint8_t> zeroedResult(scratchSpaceRequired[zeroedIdx]);
                    ASSERT_THAT(hipMemcpy(zeroedResult.data(),
                                          deviceScratch[zeroedIdx].get(),
                                          scratchSpaceRequired[zeroedIdx],
                                          hipMemcpyDeviceToHost),
                                HasHipSuccess(0));

                    bool allZeros = true;
                    for(size_t i = 0; i < zeroedResult.size(); ++i)
                    {
                        if(zeroedResult[i] != 0)
                        {
                            allZeros = false;
                            // Print as uint32 since flags are UInt32
                            size_t flagIndex = i / sizeof(uint32_t);
                            std::cerr << "Non-zero at byte " << i << " (flag index " << flagIndex
                                      << "): " << static_cast<int>(zeroedResult[i]) << std::endl;
                        }
                    }
                    EXPECT_TRUE(allZeros)
                        << "ZeroedBeforeAndAfter scratch should be all zeros after kernel "
                           "execution (size="
                        << scratchSpaceRequired[zeroedIdx] << " bytes)";
                }

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
            using namespace rocRoller;
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
            using namespace rocRoller;
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
}
