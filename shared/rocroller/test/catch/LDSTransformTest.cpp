// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Catch2 tests that exercise and demonstrate coordinate transforms to/from LDS
 * by:
 *
 * 1. Running small matrix-multiply problems. Using a real GEMM
 *    ensures that tests honour MFMA layouts on the hardware.
 *
 * 2. Comparing the load pattern from a BufferToVGPR reference to
 *    other load paths.  This compares the I/J elements loaded into
 *    workitems and VGPRs to that reference.
 */

#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Policy.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/TensorDescriptor.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

using namespace rocRoller;

namespace LDSTransformTest
{
    /** Matrix-instruction and tile dimensions (M, N, K). */
    struct MNKTuple
    {
        int m;
        int n;
        int k;
    };

    /**
     * Builds a minimal GEMM kernel graph (Load A, Load B, Mul, Store D) and
     * evaluates coordinate transforms for a chosen matrix layout.
     */
    class LDSCoordinateMappingFixture
    {
    public:
        using KernelGraphPtr = rocRoller::KernelGraph::KernelGraphPtr;

        using HWToIJMap = std::map<std::tuple<unsigned int, unsigned int>,
                                   std::pair<unsigned int, unsigned int>>;

        using LDSToIJMap = std::map<unsigned int, std::pair<unsigned int, unsigned int>>;

        /**
         * Builds a minimal GEMM kernel graph using \p loadPath for both
         * A and B tiles. \p layout selects which matrix's coordinate transform is examined
         * (A or B), and A is transposed when \p transposeA is true.
         */
        LDSCoordinateMappingFixture(rocRoller::ContextPtr                     context,
                                    bool                                      transposeA,
                                    rocRoller::LayoutType                     layout,
                                    rocRoller::Parameters::Solution::LoadPath loadPath);

        /** Mutable view of the generated kernel graph. */
        rocRoller::KernelGraph::KernelGraph& kernelGraph();
        /** Const view of the generated kernel graph. */
        rocRoller::KernelGraph::KernelGraph const& kernelGraph() const;

        /** Shared pointer to the same graph as kernelGraph(). */
        KernelGraphPtr const& kernelGraphPtr() const;

        /**
         * Control-graph tag for the LoadTiled of \p m_layout, and its User coordinate tag.
         */
        std::pair<int, int> getLoadAndUserTag() const;
        /**
         * Tags for the StoreLDSTile on that load path and its connected LDS coordinate.
         */
        std::pair<int, int> getStoreAndLDSTag() const;
        /**
         * Tags for LoadLDSTile on that LDS tile and the LDS coordinate (same LDS tag as getStoreAndLDSTag).
         */
        std::pair<int, int> getLoadAndLDSTag() const;

        /**
         * BufferToVGPR only: for each (workitem, VGPR), reverse-evaluate split dims to (row, col).
         */
        HWToIJMap getHWToIJ();
        /**
         * For the LDS load path: map each LDS word index to (i, j) by composing load and store transforms.
         */
        LDSToIJMap getLDSToIJ();
        /**
         * LoadLDSTile path: for each (workitem, VGPR), map through LDS using \p ldsToIJ to (i, j).
         */
        HWToIJMap getHWToIJFromLDS(LDSToIJMap const& ldsToIJ);

    private:
        rocRoller::ContextPtr                     m_context;
        bool                                      m_transposeA;
        rocRoller::LayoutType                     m_layout;
        rocRoller::Parameters::Solution::LoadPath m_loadPath;

        KernelGraphPtr m_kgraph;

        uint m_workgroupSize;
    };

    LDSCoordinateMappingFixture::LDSCoordinateMappingFixture(
        rocRoller::ContextPtr                     context,
        bool                                      transposeA,
        rocRoller::LayoutType                     layout,
        rocRoller::Parameters::Solution::LoadPath loadPath)
        : m_context(std::move(context))
        , m_transposeA(transposeA)
        , m_layout(layout)
        , m_loadPath(loadPath)
    {
        auto const& arch = m_context->targetArchitecture();
        auto        wfs  = arch.GetCapability(rocRoller::GPUCapability::DefaultWavefrontSize);

        const int wavefrontCountX = 2, wavefrontCountY = 2;
        // MFMA wave tile (32x32x2) and 2x2 wavefront grid — keep in sync with matrixMultiplyAB.
        MNKTuple const mi{32, 32, 2};
        MNKTuple const tile{wavefrontCountX * mi.m, wavefrontCountY * mi.n, 2 * mi.k};

        auto                dataType     = rocRoller::DataType::Float;
        std::vector<size_t> unitStridesN = {1, 0};
        std::vector<size_t> unitStridesT = {0, 1};

        auto command    = std::make_shared<rocRoller::Command>();
        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
            2, dataType, {}, m_transposeA ? unitStridesT : unitStridesN));
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));
        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, dataType, {}, unitStridesN));
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));
        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB, dataType));
        auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataType));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        auto params     = std::make_shared<rocRoller::CommandParameters>();
        m_workgroupSize = wavefrontCountX * wavefrontCountY * static_cast<uint>(wfs);
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({m_workgroupSize, 1u, 1u});
        params->setManualWavefrontCount(
            {static_cast<uint>(wavefrontCountX), static_cast<uint>(wavefrontCountY)});
        params->transposeMemoryAccess.set(rocRoller::LayoutType::MATRIX_A, m_transposeA);
        params->transposeMemoryAccess.set(rocRoller::LayoutType::MATRIX_B, false);

        using rocRoller::KernelGraph::CoordinateGraph::MacroTile;
        auto macTileA = MacroTile({tile.m, tile.k},
                                  rocRoller::LayoutType::MATRIX_A,
                                  {mi.m, mi.n, mi.k, 1},
                                  GetMemoryType(m_loadPath));
        auto macTileB = MacroTile({tile.k, tile.n},
                                  rocRoller::LayoutType::MATRIX_B,
                                  {mi.m, mi.n, mi.k, 1},
                                  GetMemoryType(m_loadPath));
        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);
        auto macTileD = MacroTile(
            {tile.m, tile.n}, rocRoller::LayoutType::MATRIX_ACCUMULATOR, {mi.m, mi.n, mi.k, 1});
        params->setDimensionInfo(tagStoreD, macTileD);

        rocRoller::CommandKernel commandKernel(command, "DirectToVGPRMapping");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        m_kgraph = commandKernel.getKernelGraph();
    }

    rocRoller::KernelGraph::KernelGraph& LDSCoordinateMappingFixture::kernelGraph()
    {
        return *m_kgraph;
    }

    rocRoller::KernelGraph::KernelGraph const& LDSCoordinateMappingFixture::kernelGraph() const
    {
        return *m_kgraph;
    }

    LDSCoordinateMappingFixture::KernelGraphPtr const&
        LDSCoordinateMappingFixture::kernelGraphPtr() const
    {
        return m_kgraph;
    }

    std::pair<int, int> LDSCoordinateMappingFixture::getLoadAndUserTag() const
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        auto const& kgraph = kernelGraph();

        int loadTag = -1;
        for(int op : kgraph.control.getNodes())
        {
            if(!kgraph.control.get<LoadTiled>(op))
                continue;
            int macTag = kgraph.mapper.get<CoordinateGraph::MacroTile>(op);
            if(macTag < 0)
                continue;
            auto macOpt = kgraph.coordinates.get<CoordinateGraph::MacroTile>(macTag);
            if(macOpt && macOpt->layoutType == m_layout)
            {
                loadTag = op;
                break;
            }
        }
        INFO("No LoadTiled with layout " << toString(m_layout));
        REQUIRE(loadTag >= 0);

        int userTag = kgraph.mapper.get<CoordinateGraph::User>(loadTag);
        INFO("Load op has no User tag (loadTag=" << loadTag << ")");
        REQUIRE(userTag >= 0);

        return {loadTag, userTag};
    }

    std::pair<int, int> LDSCoordinateMappingFixture::getStoreAndLDSTag() const
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        auto const& kgraph = kernelGraph();

        int storeTag = -1;
        int ldsTag   = -1;

        auto [loadTag, userTag] = getLoadAndUserTag();

        // From the User coordinate, look through connections to find a StoreLDSTile operation
        for(auto conn : kgraph.mapper.getCoordinateConnections(userTag))
        {
            if(kgraph.control.get<StoreLDSTile>(conn.control))
            {
                storeTag = conn.control;
                break;
            }
        }

        // From the StoreLDSTile operation, look through connections to find a LDS coordinate
        if(storeTag >= 0)
        {
            for(auto conn : kgraph.mapper.getConnections(storeTag))
            {
                auto maybeLDS = kgraph.coordinates.get<CoordinateGraph::LDS>(conn.coordinate);
                if(maybeLDS)
                {
                    ldsTag = conn.coordinate;
                    break;
                }
            }
        }

        INFO("No StoreLDSTile connected to User for layout " << toString(m_layout));
        REQUIRE(storeTag >= 0);
        INFO("No LDS coordinate connected to StoreLDSTile for layout " << toString(m_layout));
        REQUIRE(ldsTag >= 0);

        return {storeTag, ldsTag};
    }

    std::pair<int, int> LDSCoordinateMappingFixture::getLoadAndLDSTag() const
    {
        using namespace rocRoller::KernelGraph::ControlGraph;
        auto const& kgraph = kernelGraph();

        auto [storeTag, ldsTag] = getStoreAndLDSTag();

        int loadTag = -1;

        // From the LDS coordinate, look through connections to find a LoadLDSTile operation
        for(auto conn : kgraph.mapper.getCoordinateConnections(ldsTag))
        {
            if(kgraph.control.get<LoadLDSTile>(conn.control))
            {
                loadTag = conn.control;
                break;
            }
        }

        INFO("No LoadLDSTile connected to LDS for layout " << toString(m_layout));
        REQUIRE(loadTag >= 0);

        return {loadTag, ldsTag};
    }

    LDSCoordinateMappingFixture::HWToIJMap LDSCoordinateMappingFixture::getHWToIJ()
    {
        REQUIRE(m_loadPath == Parameters::Solution::LoadPath::BufferToVGPR);

        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::Expression;
        namespace KG = rocRoller::KernelGraph;

        HWToIJMap hwToIJ;

        auto& kgraph            = kernelGraph();
        auto [loadTag, userTag] = getLoadAndUserTag();

        auto sdims
            = kgraph.coordinates.getOutputNodeIndices(userTag, isEdge<Split>).to<std::vector>();
        REQUIRE(sdims.size() >= 2u);
        Log::debug("getHWToIJ: loadTag={}, userTag={}, sdims={}", loadTag, userTag, sdims);

        auto transformer = kgraph.buildTransformer(loadTag, rocRoller::IgnoreCache);

        auto required
            = KG::findRequiredCoordinates(userTag, Graph::Direction::Downstream, kgraph).first;

        std::vector<int> remainingRequired;
        for(int tag : required)
        {
            if(KG::isHardwareCoordinate(tag, kgraph) || KG::isLoopishCoordinate(tag, kgraph))
                continue;
            remainingRequired.push_back(tag);
        }
        // BufferToVGPR path: only hardware + loopish coords are required for the split dims.
        REQUIRE(remainingRequired.empty());

        for(auto tag : required)
        {
            auto maybeForLoop   = kgraph.coordinates.get<ForLoop>(tag);
            auto maybeWorkgroup = kgraph.coordinates.get<Workgroup>(tag);
            if(maybeForLoop || maybeWorkgroup)
                transformer.setCoordinate(tag, literal(0));
        }

        auto workitemTags = KG::filterCoordinates<Workitem>(required, kgraph);
        auto vgprTags     = KG::filterCoordinates<VGPR>(required, kgraph);

        REQUIRE(workitemTags.size() == 1);
        REQUIRE(vgprTags.size() == 1);

        auto workitemTag = *workitemTags.begin();
        auto vgprTag     = *vgprTags.begin();
        auto vgprSize    = getUnsignedInt(evaluate(kgraph.coordinates.get<VGPR>(vgprTag)->size));

        for(int wi = 0; wi < static_cast<int>(m_workgroupSize); wi++)
        {
            for(int vgpr = 0; vgpr < vgprSize; vgpr++)
            {
                transformer.setCoordinate(workitemTag, literal(wi));
                transformer.setCoordinate(vgprTag, literal(vgpr));
                auto indexExprs = transformer.reverse(sdims);
                REQUIRE(indexExprs.size() == 2);
                auto i = tryEvaluate(indexExprs[0]);
                auto j = tryEvaluate(indexExprs[1]);
                REQUIRE(i);
                REQUIRE(j);
                hwToIJ[{static_cast<unsigned int>(wi), static_cast<unsigned int>(vgpr)}]
                    = {getUnsignedInt(*i), getUnsignedInt(*j)};
            }
        }

        return hwToIJ;
    }

    LDSCoordinateMappingFixture::LDSToIJMap LDSCoordinateMappingFixture::getLDSToIJ()
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::Expression;
        namespace KG = rocRoller::KernelGraph;

        LDSToIJMap ldsToIJ;

        auto& kgraph               = kernelGraph();
        auto [loadTag, userTag]    = getLoadAndUserTag();
        auto [storeLDSTag, ldsTag] = getStoreAndLDSTag();

        auto sdims
            = kgraph.coordinates.getOutputNodeIndices(userTag, isEdge<Split>).to<std::vector>();
        REQUIRE(sdims.size() >= 2u);
        Log::debug("getLDSToIJ: loadTag={}, userTag={}, sdims={}", loadTag, userTag, sdims);
        Log::debug("getLDSToIJ: storeLDSTag={}, ldsTag={}", storeLDSTag, ldsTag);

        auto loadTransformer  = kgraph.buildTransformer(loadTag, rocRoller::IgnoreCache);
        auto storeTransformer = kgraph.buildTransformer(storeLDSTag, rocRoller::IgnoreCache);

        auto loadRequired
            = KG::findRequiredCoordinates(userTag, Graph::Direction::Downstream, kgraph).first;
        auto storeRequired
            = KG::findRequiredCoordinates(ldsTag, Graph::Direction::Upstream, kgraph).first;

        std::vector<int> remainingLoadRequired;
        for(int tag : loadRequired)
        {
            if(KG::isHardwareCoordinate(tag, kgraph) || KG::isLoopishCoordinate(tag, kgraph))
                continue;
            remainingLoadRequired.push_back(tag);
        }
        // Shape of the LoadTiled subgraph: two ElementNumber coords besides loopish/hardware.
        std::sort(remainingLoadRequired.begin(), remainingLoadRequired.end());

        REQUIRE(remainingLoadRequired.size() == 2);
        REQUIRE(kgraph.coordinates.get<ElementNumber>(remainingLoadRequired[0]).has_value());
        REQUIRE(kgraph.coordinates.get<ElementNumber>(remainingLoadRequired[1]).has_value());

        for(auto tag : loadRequired)
        {
            auto maybeForLoop   = kgraph.coordinates.get<ForLoop>(tag);
            auto maybeWorkgroup = kgraph.coordinates.get<Workgroup>(tag);
            if(maybeForLoop || maybeWorkgroup)
                loadTransformer.setCoordinate(tag, literal(0));
        }

        Log::debug("getLDSToIJ: loadRequired={}, storeRequired={}", loadRequired, storeRequired);

        // Store path around LDS: Workitem + two ElementNumbers (update if the graph changes).
        REQUIRE(storeRequired.size() == 3);

        auto loadWorkitemTags      = KG::filterCoordinates<Workitem>(loadRequired, kgraph);
        auto loadElementNumberTags = KG::filterCoordinates<ElementNumber>(loadRequired, kgraph);

        REQUIRE(loadWorkitemTags.size() == 1);
        REQUIRE(loadElementNumberTags.size() == 2);

        auto storeWorkitemTags      = KG::filterCoordinates<Workitem>(storeRequired, kgraph);
        auto storeElementNumberTags = KG::filterCoordinates<ElementNumber>(storeRequired, kgraph);

        REQUIRE(storeWorkitemTags.size() == 1);
        REQUIRE(storeElementNumberTags.size() == 2);

        std::vector<int> loadElemSorted(loadElementNumberTags.begin(), loadElementNumberTags.end());
        std::vector<int> storeElemSorted(storeElementNumberTags.begin(),
                                         storeElementNumberTags.end());
        std::sort(loadElemSorted.begin(), loadElemSorted.end());
        std::sort(storeElemSorted.begin(), storeElemSorted.end());

        auto loadWorkitemTag        = *loadWorkitemTags.begin();
        auto storeWorkitemTag       = *storeWorkitemTags.begin();
        auto loadElementNumberTag0  = loadElemSorted[0];
        auto loadElementNumberTag1  = loadElemSorted[1];
        auto storeElementNumberTag0 = storeElemSorted[0];
        auto storeElementNumberTag1 = storeElemSorted[1];
        auto loadElemSize0          = getUnsignedInt(
            evaluate(kgraph.coordinates.get<ElementNumber>(loadElementNumberTag0).value().size));
        auto loadElemSize1 = getUnsignedInt(
            evaluate(kgraph.coordinates.get<ElementNumber>(loadElementNumberTag1).value().size));
        auto storeElemSize0 = getUnsignedInt(
            evaluate(kgraph.coordinates.get<ElementNumber>(storeElementNumberTag0).value().size));
        auto storeElemSize1 = getUnsignedInt(
            evaluate(kgraph.coordinates.get<ElementNumber>(storeElementNumberTag1).value().size));

        for(int wi = 0; wi < static_cast<int>(m_workgroupSize); wi++)
        {
            for(int e0 = 0; e0 < loadElemSize0; e0++)
            {
                for(int e1 = 0; e1 < loadElemSize1; e1++)
                {
                    loadTransformer.setCoordinate(loadWorkitemTag, literal(wi));
                    loadTransformer.setCoordinate(loadElementNumberTag0, literal(e0));
                    loadTransformer.setCoordinate(loadElementNumberTag1, literal(e1));
                    storeTransformer.setCoordinate(storeWorkitemTag, literal(wi));
                    storeTransformer.setCoordinate(storeElementNumberTag0, literal(e0));
                    storeTransformer.setCoordinate(storeElementNumberTag1, literal(e1));

                    auto indexExprs = loadTransformer.reverse(sdims);
                    REQUIRE(indexExprs.size() == 2);
                    auto i = tryEvaluate(indexExprs[0]);
                    auto j = tryEvaluate(indexExprs[1]);

                    indexExprs = storeTransformer.forward({ldsTag});
                    REQUIRE(indexExprs.size() == 1);
                    auto lds = tryEvaluate(indexExprs[0]);

                    REQUIRE(i);
                    REQUIRE(j);
                    REQUIRE(lds);
                    ldsToIJ[getUnsignedInt(*lds)] = {getUnsignedInt(*i), getUnsignedInt(*j)};
                }
            }
        }

        // Bijection between (workitem, e0, e1) triples and LDS indices: no collisions, full range.
        auto const expectedCount = static_cast<size_t>(m_workgroupSize)
                                   * static_cast<size_t>(loadElemSize0)
                                   * static_cast<size_t>(loadElemSize1);
        INFO("ldsToIJ.size()=" << ldsToIJ.size() << " expected unique LDS slots=" << expectedCount);
        REQUIRE(ldsToIJ.size() == expectedCount);

        return ldsToIJ;
    }

    LDSCoordinateMappingFixture::HWToIJMap
        LDSCoordinateMappingFixture::getHWToIJFromLDS(LDSToIJMap const& ldsToIJ)
    {
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::Expression;
        namespace KG = rocRoller::KernelGraph;

        auto& kgraph = kernelGraph();

        HWToIJMap hwToIJ;

        auto [loadTag, ldsTag] = getLoadAndLDSTag();
        Log::debug("getHWToIJFromLDS: loadLDSTag={}, ldsTag={}", loadTag, ldsTag);

        auto loadTransformer = kgraph.buildTransformer(loadTag, rocRoller::IgnoreCache);

        auto required
            = KG::findRequiredCoordinates(ldsTag, Graph::Direction::Downstream, kgraph).first;

        // Filter out loopish coordinates (LoadLDSTile path: Workitem + VGPR for LDS index).
        std::vector<int> remainingRequired;
        for(int tag : required)
        {
            if(KG::isLoopishCoordinate(tag, kgraph))
                continue;
            remainingRequired.push_back(tag);
        }
        std::sort(remainingRequired.begin(), remainingRequired.end());

        Log::debug("getHWToIJFromLDS: remainingRequired={}", remainingRequired);

        REQUIRE(remainingRequired.size() == 2);

        // Set loopish (unroll) coordinates to 0
        for(int tag : required)
        {
            auto maybeUnroll = kgraph.coordinates.get<Unroll>(tag);
            if(maybeUnroll)
                loadTransformer.setCoordinate(tag, literal(0));
        }

        auto workitemTags = KG::filterCoordinates<Workitem>(remainingRequired, kgraph);
        auto vgprTags     = KG::filterCoordinates<VGPR>(remainingRequired, kgraph);

        REQUIRE(workitemTags.size() == 1);
        REQUIRE(vgprTags.size() == 1);

        auto workitemTag = *workitemTags.begin();
        auto vgprTag     = *vgprTags.begin();
        auto vgprSize
            = getUnsignedInt(evaluate(kgraph.coordinates.get<VGPR>(vgprTag).value().size));

        for(int wi = 0; wi < static_cast<int>(m_workgroupSize); wi++)
        {
            for(int vgpr = 0; vgpr < vgprSize; vgpr++)
            {
                loadTransformer.setCoordinate(workitemTag, literal(wi));
                loadTransformer.setCoordinate(vgprTag, literal(vgpr));
                auto indexExprs = loadTransformer.reverse({ldsTag});
                REQUIRE(indexExprs.size() == 1);
                auto lds = tryEvaluate(indexExprs[0]);
                REQUIRE(lds);

                auto ldsInt = getUnsignedInt(*lds);

                auto ldsIt = ldsToIJ.find(ldsInt);
                REQUIRE(ldsIt != ldsToIJ.end());
                auto [i, j] = ldsIt->second;

                hwToIJ[{static_cast<unsigned int>(wi), static_cast<unsigned int>(vgpr)}] = {i, j};
            }
        }

        return hwToIJ;
    }

    /**
     * Run a small matrix multiply D = A*B with given MFMA wave tile and load path.
     * When loadPathAB is BufferToLDSViaVGPR (or other LDS path), the kernel uses
     * LDS for A and/or B, exercising coordinate transforms to/from LDS.
     *
     * If runOnDevice is false, only the kernel is generated (no allocation/launch);
     * use this to test codegen without a HIP device.
     *
     * Template params: TA, TB = input element types; TD = output type; ACC = accumulator.
     */
    template <typename TA, typename TB, typename TD, typename ACC = float>
    void matrixMultiplyAB(rocRoller::ContextPtr const&   context,
                          int                            wave_m,
                          int                            wave_n,
                          int                            wave_k,
                          int                            wave_b,
                          Parameters::Solution::LoadPath loadPathAB
                          = Parameters::Solution::LoadPath::BufferToLDSViaVGPR,
                          bool transA      = false,
                          bool transB      = false,
                          int  M           = 128,
                          int  N           = 128,
                          int  K           = 64,
                          bool runOnDevice = true)
    {
        auto dataTypeA   = TypeInfo<TA>::Var.dataType;
        auto dataTypeB   = TypeInfo<TB>::Var.dataType;
        auto dataTypeD   = TypeInfo<TD>::Var.dataType;
        auto dataTypeAcc = TypeInfo<ACC>::Var.dataType;

        const auto wavefrontCountX = 2;
        const auto wavefrontCountY = 2;

        int mac_m = wavefrontCountX * wave_m;
        int mac_n = wavefrontCountY * wave_n;
        int mac_k = 2 * wave_k;

        REQUIRE(M % mac_m == 0);
        REQUIRE(N % mac_n == 0);
        REQUIRE(K % mac_k == 0);

        auto const& arch = context->targetArchitecture();
        auto        wfs  = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

        uint workgroup_size_x = wavefrontCountX * wavefrontCountY * wfs;
        uint workgroup_size_y = 1;

        TensorDescriptor descA(
            dataTypeA, {static_cast<size_t>(M), static_cast<size_t>(K)}, transA ? "T" : "N");
        TensorDescriptor descB(
            dataTypeB, {static_cast<size_t>(K), static_cast<size_t>(N)}, transB ? "T" : "N");
        TensorDescriptor descD(dataTypeD,
                               {static_cast<size_t>(M), static_cast<size_t>(N)},
                               {1u, static_cast<size_t>(M)});

        uint32_t seed = 9861u;
        auto     A    = DGenVector<TA>(descA, -1.f, 1.f, seed + 1);
        auto     B    = DGenVector<TB>(descB, -1.f, 1.f, seed + 2);

        std::shared_ptr<TA> d_A;
        std::shared_ptr<TB> d_B;
        std::shared_ptr<TD> d_D;
        if(runOnDevice)
        {
#ifdef ROCROLLER_USE_HIP
            d_A = make_shared_device(A);
            d_B = make_shared_device(B);
            d_D = make_shared_device<TD>(M * N);
#else
            SKIP("HIP not available");
#endif
        }

        auto command = std::make_shared<Command>();

        std::vector<size_t> unitStridesN = {1, 0};
        std::vector<size_t> unitStridesT = {0, 1};

        auto tagTensorA = command->addOperation(
            rocRoller::Operations::Tensor(2, dataTypeA, {}, transA ? unitStridesT : unitStridesN));
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(
            rocRoller::Operations::Tensor(2, dataTypeB, {}, transB ? unitStridesT : unitStridesN));
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB, dataTypeAcc));

        auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD));
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        if(runOnDevice)
        {
#ifdef ROCROLLER_USE_HIP
            setCommandTensorArg(commandArgs, tagTensorA, descA, (TA*)d_A.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, (TB*)d_B.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, d_D.get());
#endif
        }

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

        auto macTileA = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                                LayoutType::MATRIX_A,
                                                                {wave_m, wave_n, wave_k, wave_b},
                                                                GetMemoryType(loadPathAB));
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile({mac_k, mac_n},
                                                                LayoutType::MATRIX_B,
                                                                {wave_m, wave_n, wave_k, wave_b},
                                                                GetMemoryType(loadPathAB));

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);
        params->setManualWavefrontCount({wavefrontCountX, wavefrontCountY});
        params->transposeMemoryAccess.set(LayoutType::MATRIX_A, transA);
        params->transposeMemoryAccess.set(LayoutType::MATRIX_B, transB);

        CommandKernel commandKernel(command, "MatrixMultiplyAB");
        commandKernel.setContext(context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        if(runOnDevice)
        {
#ifdef ROCROLLER_USE_HIP
            commandKernel.launchKernel(commandArgs.runtimeArguments());

            std::vector<TD> D(M * N);
            CHECK_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(TD), hipMemcpyDefault),
                       HasHipSuccess(0));

            std::vector<TD> c_D(M * N, TD{});
            std::vector<TD> c_C(M * N, TD{});

            CPUMM(c_D, c_C, A, B, M, N, K, 1.0f, 0.0, transA, transB);

            auto tol
                = gemmAcceptableError<TA, TB, TD>(M, N, K, context->targetArchitecture().target());
            auto res = compare(D, c_D, tol);

            Log::debug("RNorm is {}", res.relativeNormL2);
            CHECK(res.ok);
            if(!res.ok)
                FAIL(res.message());
#endif
        }
    }

    // Smoke test: float GEMM with LDS load path; other types are left to matrixMultiplyAB callers.
    TEST_CASE("LDSTransform matrixMultiplyAB codegen with LDS path",
              "[kernel-graph][lds-transform][matrix-multiply][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        if(!context->targetArchitecture().HasCapability(GPUCapability::HasMFMA))
        {
            SKIP("Target has no MFMA");
        }

        matrixMultiplyAB<float, float, float>(context.get(),
                                              32,
                                              32,
                                              2,
                                              1,
                                              Parameters::Solution::LoadPath::BufferToLDSViaVGPR,
                                              false,
                                              false,
                                              128,
                                              128,
                                              64,
                                              true);
    }

    /**
     * Parameterized test: build coordinate transform for a "direct to VGPR" load
     * (BufferToVGPR), create a fake matrix where each element is (i, j), and
     * build a mapping from (workitem, vgpr item) to (i, j) by evaluating the
     * transform.
     *
     * Uses a minimal GEMM (A*B -> D) so the kernel graph is valid; we only
     * exercise the coordinate mapping for the chosen matrix (A or B).
     */
    TEST_CASE("LDSTransform direct-to-VGPR (workitem, vgpr) -> (i,j) mapping",
              "[kernel-graph][lds-transform][coordinate-transform]")
    {
        auto context = TestContext::ForDefaultTarget();

        auto transpose = GENERATE("T", "N");
        auto layout    = GENERATE(LayoutType::MATRIX_A, LayoutType::MATRIX_B);
        bool trans     = (transpose == std::string("T"));

        // Reference uses BufferToVGPR
        LDSCoordinateMappingFixture reference(
            context.get(), trans, layout, Parameters::Solution::LoadPath::BufferToVGPR);

        LDSCoordinateMappingFixture simulator(
            context.get(), trans, layout, Parameters::Solution::LoadPath::BufferToLDSViaVGPR);

        auto hwToIJ = reference.getHWToIJ();
        CHECK(not hwToIJ.empty());

        auto ldsToIJ = simulator.getLDSToIJ();
        CHECK(not ldsToIJ.empty());

        auto hwToIJFromLDS = simulator.getHWToIJFromLDS(ldsToIJ);
        CHECK(not hwToIJFromLDS.empty());

        CHECK(hwToIJ == hwToIJFromLDS);
    }
}
