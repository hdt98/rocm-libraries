// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
namespace rocRoller
{
    namespace KernelGraph
    {
        namespace RegisterUsageAnalysis_Detail
        {
            using Record = ControlFlowRWTracer::ReadWriteRecord;
            using Edge   = ControlGraph::ControlEdge;
            enum class CoordLiveness
            {
                Dead,
                Alive,
                MaybeAlive
            };
            struct CoordExtent
            {
                int                                  coordTag = -1;
                AliasDataFlowTagsDetail::GraphExtent extent;
                std::set<int>                        allAccessNodes;
                bool                                 empty() const;
            };
            struct CoordTypeInfo
            {
                VariableType   varType    = DataType::None;
                Register::Type regType    = Register::Type::Count;
                int            valueCount = 0;
                bool           resolved   = false;
            };
            struct CoordRegisterEstimate
            {
                int  registerCount = 0;
                bool resolved      = false;
            };
            struct OperationRegisterUsage
            {
                int operationTag = -1;
                // Per-work-item GPR count by register type.
                // Vector and Accumulator are per-work-item.
                // Scalar is per-wave.
                std::map<Register::Type, int> regCountByType;
                // Sum of all register types.
                int totalRegisters = 0;
            };
            std::vector<Record> deduplicateRecords(std::vector<Record> const& records);
            AliasDataFlowTagsDetail::TagRWGraph
                                       getOrderingWithBodies(KernelGraph const&         kgraph,
                                                             std::vector<Record> const& records);
            CoordExtent                getCoordExtent(KernelGraph const&         kgraph,
                                                      int                        coordTag,
                                                      std::vector<Record> const& records);
            std::map<int, CoordExtent> buildAllCoordExtents(KernelGraph const& kgraph);
            CoordLiveness              getCoordLiveness(KernelGraph const& kgraph,
                                                        int                operationTag,
                                                        CoordExtent const& coordExtent);
            std::map<int, CoordLiveness>
                          getLivenessAtOperation(KernelGraph const&                kgraph,
                                                 int                               operationTag,
                                                 std::map<int, CoordExtent> const& allExtents);
            CoordTypeInfo resolveCoordTypeFromOperations(KernelGraph const&         kgraph,
                                                         int                        coordTag,
                                                         std::vector<Record> const& records);
            std::map<int, CoordTypeInfo>
                                  resolveAllCoordTypes(KernelGraph const&                        kgraph,
                                                       std::map<int, std::vector<Record>> const& recordsByCoord);
            CoordRegisterEstimate estimateCoordRegisterCount(KernelGraph const&   kgraph,
                                                             int                  coordTag,
                                                             CoordTypeInfo const& typeInfo,
                                                             int                  wavefrontSize);
            std::string           toString(CoordLiveness liveness);
            // Top-level API.
            // wavefrontSize: obtain from context->kernel()->wavefront_size().
            std::map<int, OperationRegisterUsage> estimateRegisterUsage(KernelGraph const& kgraph,
                                                                        int wavefrontSize);

            // ============================================================
            // ASCII / text visualization
            // ============================================================
            // Dump a text table showing, for each control operation,
            // which coordinates are Alive or MaybeAlive and how many
            // registers (per work-item) each contributes.
            //
            // Format:
            //   === Operation [tag] OpName ===
            //     VGPR: N   AGPR: N   SGPR: N   Total: N
            //     Alive:
            //       coord [tag] DimName  regType  regs=N
            //     MaybeAlive:
            //       coord [tag] DimName  regType  regs=N
            //
            // Returns the full text as a string.
            std::string
                dumpControlGraphLiveness(KernelGraph const&                           kgraph,
                                         std::map<int, CoordExtent> const&            allExtents,
                                         std::map<int, CoordTypeInfo> const&          allTypes,
                                         std::map<int, CoordRegisterEstimate> const&  regEstimates,
                                         std::map<int, OperationRegisterUsage> const& allUsage);
            // Dump a text table showing, for each tracked coordinate,
            // which operations access it, with R/W type.
            //
            // Format:
            //   === Coordinate [tag] DimName  type=...  regs=N ===
            //     Extent: begin={...}  end={...}
            //     Accesses:
            //       op [tag] OpName  R/W
            //
            // Returns the full text as a string.
            std::string
                dumpCoordinateAccessMap(KernelGraph const&                          kgraph,
                                        std::map<int, CoordExtent> const&           allExtents,
                                        std::map<int, CoordTypeInfo> const&         allTypes,
                                        std::map<int, CoordRegisterEstimate> const& regEstimates,
                                        std::map<int, std::vector<Record>> const&   recordsByCoord);
            // ============================================================
            // Graphviz DOT export
            // ============================================================
            // Export the control graph as a DOT file.
            // Each operation node shows its tag and name.
            // Each node is annotated with a table of live coordinates
            // and their register costs.
            // Edges are labeled with edge type (Sequence, Body, etc.).
            //
            // Write to the given output stream.
            void exportControlGraphLivenessDot(
                std::ostream&                                out,
                KernelGraph const&                           kgraph,
                std::map<int, CoordExtent> const&            allExtents,
                std::map<int, CoordTypeInfo> const&          allTypes,
                std::map<int, CoordRegisterEstimate> const&  regEstimates,
                std::map<int, OperationRegisterUsage> const& allUsage);
            // Export a coordinate-centric DOT graph.
            // Coordinate nodes are shown with their type/register info.
            // Operation nodes are shown with their name.
            // Edges from operations to coordinates indicate access (R/W).
            //
            // Write to the given output stream.
            void
                exportCoordinateAccessDot(std::ostream&                               out,
                                          KernelGraph const&                          kgraph,
                                          std::map<int, CoordExtent> const&           allExtents,
                                          std::map<int, CoordTypeInfo> const&         allTypes,
                                          std::map<int, CoordRegisterEstimate> const& regEstimates,
                                          std::map<int, std::vector<Record>> const& recordsByCoord);
        } // namespace RegisterUsageAnalysis

        class RegisterUsageAnalysis : public GraphTransform
        {
        public:
            RegisterUsageAnalysis() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "RegisterUsageAnalysis";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }
        };
    }
}
