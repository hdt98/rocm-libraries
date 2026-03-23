// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class AllocatingObserver
        {
        public:
            AllocatingObserver() {}
            AllocatingObserver(ContextPtr context)
                : m_context(context)
            {
            }

            InstructionStatus peek(Instruction const& inst) const
            {
                auto              ctx = m_context.lock();
                InstructionStatus rv;

                // Remaining registers after this instruction:
                // currently remaining - newly allocated
                for(int i = 0; i < static_cast<int>(Register::Type::Count); i++)
                {
                    auto regType   = static_cast<Register::Type>(i);
                    auto allocator = ctx->allocator(regType);

                    if(allocator)
                        rv.remainingRegisters[i] = allocator->currentlyFree();
                }

                for(auto alloc : inst.allocations())
                {
                    if(alloc)
                    {
                        // Determine amount of new registers that will be allocated

                        auto regIdx = static_cast<size_t>(alloc->regType());
                        AssertFatal(regIdx < rv.allocatedRegisters.size(),
                                    ShowValue(regIdx),
                                    ShowValue(rv.allocatedRegisters.size()));
                        rv.allocatedRegisters.at(regIdx) += alloc->registerCount();

                        // Determine new highwater mark by simulating the allocation

                        auto allocator = ctx->allocator(alloc->regType());
                        auto newRegs
                            = allocator->findFree(alloc->registerCount(), alloc->options());
                        if(newRegs.size() > 0)
                        {
                            int myHWM = std::max(0, newRegs.back() - allocator->maxUsed());

                            AssertFatal(regIdx < rv.highWaterMarkRegistersDelta.size(),
                                        ShowValue(regIdx),
                                        ShowValue(rv.highWaterMarkRegistersDelta.size()));
                            rv.highWaterMarkRegistersDelta.at(regIdx)
                                = std::max(myHWM, rv.highWaterMarkRegistersDelta.at(regIdx));
                        }

                        rv.outOfRegisters.set(alloc->regType(),
                                              rv.outOfRegisters[alloc->regType()]
                                                  || alloc->registerCount() > 0 && newRegs.empty());
                        rv.remainingRegisters[regIdx] -= alloc->registerCount();
                    }
                }
                return rv;
            }

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            void modify(Instruction& inst) const
            {
                inst.allocateNow();
            }

            //> This instruction _will_ be scheduled now, record any side effects.
            void observe(Instruction const& inst) {}

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return true;
            }

        private:
            std::weak_ptr<Context> m_context;
        };

        static_assert(CObserverConst<AllocatingObserver>);

        //class RegisterCountObserver
        //{
        //    public:
        //        static constexpr size_t NUM_TRACKED_TYPES = 3;
        //        static constexpr std::array<Register::Type, NUM_TRACKED_TYPES> TRACKED_TYPES{
        //            Register::Type::Scalar,
        //                Register::Type::Vector,
        //                Register::Type::Accumulator};
        //        // One deferred allocation event, recorded chronologically
        //        struct AllocEvent
        //        {
        //            int            seqNum; // Instruction scheduling sequence number
        //            int            opTag;  // Control-op tag (from innerControlOp())
        //            Register::Type regType;
        //            int            count;
        //            std::vector<int> indices; // Actual physical register indices
        //                                      // Global allocator snapshot immediately after this allocation
        //            std::map<Register::Type, int> globalInUse; // size - currentlyFree
        //            std::map<Register::Type, int> globalHWM;   // maxUsed + 1
        //        };
        //        // Per-operation summary
        //        struct OpSummary
        //        {
        //            int instructionCount = 0;
        //            // regType -> total deferred allocation count
        //            std::map<Register::Type, int> allocCount;
        //            // regType -> sorted list of all allocated indices
        //            std::map<Register::Type, std::vector<int>> allocIndices;
        //        };
        //        RegisterCountObserver() {}
        //        RegisterCountObserver(ContextPtr context)
        //            : m_context(context)
        //        {
        //        }
        //        InstructionStatus peek(Instruction const& inst) const
        //        {
        //            return {};
        //        }
        //        void modify(Instruction& inst) const {}
        //        /**
        //         * Called after all observers' modify() (including
        //         * AllocatingObserver which commits deferred allocations).
        //         *
        //         * We record:
        //         *   1. Per-operation: deferred allocation counts and indices
        //         *   2. Chronological: each allocation event with global state
        //         *   3. On FinalInstruction: write the report
        //         */
        //        void observe(Instruction const& inst)
        //        {
        //            auto ctx = m_context.lock();
        //            if(!ctx)
        //                return;
        //            m_seqNum++;
        //            auto ctrlOp = inst.innerControlOp();
        //            // Examine each deferred allocation in this instruction
        //            for(auto const& alloc : inst.allocations())
        //            {
        //                if(!alloc)
        //                    continue;
        //                // Only track Scalar, Vector, Accumulator
        //                auto rt = alloc->regType();
        //                if(!isTracked(rt))
        //                    continue;
        //                // After allocateNow(), state should be Allocated
        //                // (non-register types are skipped by allocateNow)
        //                if(alloc->allocationState() != Register::AllocationState::Allocated)
        //                    continue;
        //                auto const& indices = alloc->registerIndices();
        //                int         count   = static_cast<int>(indices.size());
        //                // Record per-operation summary
        //                if(ctrlOp.has_value())
        //                {
        //                    auto& summary = m_opSummaries[ctrlOp.value()];
        //                    summary.allocCount[rt] += count;
        //                    auto& vec = summary.allocIndices[rt];
        //                    vec.insert(vec.end(), indices.begin(), indices.end());
        //                }
        //                // Record chronological event with global snapshot
        //                AllocEvent evt;
        //                evt.seqNum  = m_seqNum;
        //                evt.opTag   = ctrlOp.value_or(-1);
        //                evt.regType = rt;
        //                evt.count   = count;
        //                evt.indices = indices;
        //                // Snapshot global allocator state (post-allocation)
        //                for(auto t : TRACKED_TYPES)
        //                {
        //                    auto a = ctx->allocator(t);
        //                    if(a && a->size() > 0)
        //                    {
        //                        evt.globalInUse[t]
        //                            = static_cast<int>(a->size()) - a->currentlyFree();
        //                        evt.globalHWM[t] = a->maxUsed() + 1;
        //                    }
        //                    else
        //                    {
        //                        evt.globalInUse[t] = 0;
        //                        evt.globalHWM[t]   = 0;
        //                    }
        //                }
        //                m_events.push_back(std::move(evt));
        //            }
        //            // Count instructions per operation
        //            if(ctrlOp.has_value())
        //                m_opSummaries[ctrlOp.value()].instructionCount++;
        //            // Detect FinalInstruction to trigger report
        //            auto const& architecture = ctx->targetArchitecture();
        //            GPUInstructionInfo info
        //                = architecture.GetInstructionInfo(inst.getOpCode());
        //            auto wq = info.getWaitQueues();
        //            if(std::find(wq.begin(), wq.end(),
        //                        GPUWaitQueueType::FinalInstruction)
        //                    != wq.end())
        //            {
        //                writeReport(ctx);
        //            }
        //        }
        //        static bool runtimeRequired()
        //        {
        //            return true;
        //            //return Settings::getInstance()->get(Settings::KernelAnalysis);
        //        }
        //    private:
        //        std::weak_ptr<Context>       m_context;
        //        int                          m_seqNum = 0;
        //        std::map<int, OpSummary>     m_opSummaries;
        //        std::vector<AllocEvent>      m_events;
        //        static bool isTracked(Register::Type rt)
        //        {
        //            return std::find(TRACKED_TYPES.begin(), TRACKED_TYPES.end(), rt)
        //                != TRACKED_TYPES.end();
        //        }
        //        static std::string rtName(Register::Type rt)
        //        {
        //            switch(rt)
        //            {
        //                case Register::Type::Scalar:      return "SGPR";
        //                case Register::Type::Vector:      return "VGPR";
        //                case Register::Type::Accumulator: return "ACCVGPR";
        //                default:                          return "???";
        //            }
        //        }
        //        /**
        //         * Format a sorted list of register indices into a compact
        //         * range representation. E.g., {0,1,2,5,6,9} -> "[0-2, 5-6, 9]"
        //         */
        //        static std::string formatIndices(std::vector<int> indices)
        //        {
        //            if(indices.empty())
        //                return "[]";
        //            std::sort(indices.begin(), indices.end());
        //            std::ostringstream oss;
        //            oss << "[";
        //            size_t i     = 0;
        //            bool   first = true;
        //            while(i < indices.size())
        //            {
        //                if(!first)
        //                    oss << ", ";
        //                first = false;
        //                // Find end of contiguous range starting at i
        //                size_t j = i;
        //                while(j + 1 < indices.size()
        //                        && indices[j + 1] == indices[j] + 1)
        //                    j++;
        //                if(j == i)
        //                    oss << indices[i]; // Single index
        //                else
        //                    oss << indices[i] << "-" << indices[j]; // Range
        //                i = j + 1;
        //            }
        //            oss << "]";
        //            return oss.str();
        //        }
        //        /**
        //         * Compute free ranges within [0, limit) by scanning the
        //         * allocator. Returns a string like "3-5, 12, 20-25".
        //         */
        //        static std::string computeFreeRanges(
        //                std::shared_ptr<Register::Allocator> const& allocator)
        //        {
        //            int limit = allocator->maxUsed() + 1;
        //            if(limit <= 0)
        //                return "(none)";
        //            std::vector<int> freeIndices;
        //            for(int i = 0; i < limit; ++i)
        //            {
        //                if(allocator->isFree(i))
        //                    freeIndices.push_back(i);
        //            }
        //            if(freeIndices.empty())
        //                return "(none)";
        //            return formatIndices(freeIndices);
        //        }
        //        static int getVal(std::map<Register::Type, int> const& m,
        //                Register::Type                       rt)
        //        {
        //            auto it = m.find(rt);
        //            return (it != m.end()) ? it->second : 0;
        //        }
        //        void writeReport(ContextPtr ctx) const
        //        {
        //            std::string   filename = ctx->assemblyFileName() + ".regcount";
        //            std::ofstream file(filename, std::ios_base::out);
        //            if(!file.is_open())
        //                return;
        //            file << generateReport(ctx);
        //            file.flush();
        //        }
        //        std::string generateReport(ContextPtr ctx) const
        //        {
        //            std::ostringstream oss;
        //            // Collect pool sizes
        //            std::map<Register::Type, int> poolSz;
        //            for(auto rt : TRACKED_TYPES)
        //            {
        //                auto a   = ctx->allocator(rt);
        //                poolSz[rt] = a ? static_cast<int>(a->size()) : 0;
        //            }
        //            // ============================================================
        //            // Header
        //            // ============================================================
        //            oss << std::string(72, '=') << "\n";
        //            oss << "  rocRoller Register Allocation Report\n";
        //            oss << std::string(72, '=') << "\n\n";
        //            oss << "  IMPORTANT NOTES:\n";
        //            oss << "  - Only DEFERRED register allocations (those going\n"
        //                "    through the scheduling pipeline) are tracked.\n"
        //                "    Eager allocations (e.g., argument loading) are\n"
        //                "    not visible to this observer.\n";
        //            oss << "  - The scheduler interleaves instructions from\n"
        //                "    different operations. Per-operation HWM, in-use\n"
        //                "    counts, and fragmentation are NOT reported\n"
        //                "    because they cannot be correctly attributed to\n"
        //                "    individual operations.\n\n";
        //            // ============================================================
        //            // SECTION 1: Per-Operation Deferred Allocation Counts
        //            // ============================================================
        //            oss << std::string(72, '=') << "\n";
        //            oss << "  SECTION 1: Per-Operation Deferred Register "
        //                "Allocation Counts\n";
        //            oss << std::string(72, '=') << "\n";
        //            oss << std::setw(8) << "OpTag" << " | "
        //                << std::setw(6) << "#Inst" << " | "
        //                << std::setw(6) << "SGPR" << " | "
        //                << std::setw(6) << "VGPR" << " | "
        //                << std::setw(8) << "ACCVGPR" << "\n";
        //            oss << std::string(8, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(8, '-') << "\n";
        //            int tS = 0, tV = 0, tA = 0, tI = 0;
        //            for(auto const& [opTag, sm] : m_opSummaries)
        //            {
        //                int s = getVal(sm.allocCount, Register::Type::Scalar);
        //                int v = getVal(sm.allocCount, Register::Type::Vector);
        //                int a = getVal(sm.allocCount, Register::Type::Accumulator);
        //                tS += s; tV += v; tA += a; tI += sm.instructionCount;
        //                oss << std::setw(8) << opTag << " | "
        //                    << std::setw(6) << sm.instructionCount << " | "
        //                    << std::setw(6) << s << " | "
        //                    << std::setw(6) << v << " | "
        //                    << std::setw(8) << a << "\n";
        //            }
        //            oss << std::string(8, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(8, '-') << "\n";
        //            oss << std::setw(8) << "TOTAL" << " | "
        //                << std::setw(6) << tI << " | "
        //                << std::setw(6) << tS << " | "
        //                << std::setw(6) << tV << " | "
        //                << std::setw(8) << tA << "\n\n";
        //            // ============================================================
        //            // SECTION 2: Per-Operation Allocated Register Indices
        //            // ============================================================
        //            oss << std::string(72, '=') << "\n";
        //            oss << "  SECTION 2: Per-Operation Allocated Register "
        //                "Indices\n";
        //            oss << std::string(72, '=') << "\n\n";
        //            for(auto const& [opTag, sm] : m_opSummaries)
        //            {
        //                bool anyAlloc = false;
        //                for(auto rt : TRACKED_TYPES)
        //                {
        //                    auto it = sm.allocIndices.find(rt);
        //                    if(it != sm.allocIndices.end() && !it->second.empty())
        //                        anyAlloc = true;
        //                }
        //                if(!anyAlloc)
        //                    continue;
        //                oss << "  Op " << opTag << " ("
        //                    << sm.instructionCount << " instructions):\n";
        //                for(auto rt : TRACKED_TYPES)
        //                {
        //                    auto it = sm.allocIndices.find(rt);
        //                    if(it == sm.allocIndices.end() || it->second.empty())
        //                        continue;
        //                    oss << "    " << std::setw(7) << rtName(rt)
        //                        << ": " << formatIndices(it->second) << "\n";
        //                }
        //                oss << "\n";
        //            }
        //            // ============================================================
        //            // SECTION 3: Allocation Event Log (chronological)
        //            // ============================================================
        //            oss << std::string(72, '=') << "\n";
        //            oss << "  SECTION 3: Allocation Event Log (chronological)\n";
        //            oss << "  Each row = one deferred allocation. Global state\n"
        //                "  is the allocator snapshot AFTER this allocation.\n";
        //            oss << std::string(72, '=') << "\n\n";
        //            oss << std::setw(6) << "Seq" << " | "
        //                << std::setw(6) << "OpTag" << " | "
        //                << std::setw(7) << "Type" << " | "
        //                << std::setw(5) << "Count" << " | "
        //                << std::setw(22) << "Indices" << " | "
        //                << "Global InUse(S/V/A) | HWM(S/V/A)\n";
        //            oss << std::string(6, '-') << "-+-"
        //                << std::string(6, '-') << "-+-"
        //                << std::string(7, '-') << "-+-"
        //                << std::string(5, '-') << "-+-"
        //                << std::string(22, '-') << "-+-"
        //                << std::string(19, '-') << "-+-"
        //                << std::string(11, '-') << "\n";
        //            for(auto const& evt : m_events)
        //            {
        //                std::string idxStr = formatIndices(evt.indices);
        //                // Truncate very long index strings for readability
        //                if(idxStr.size() > 22)
        //                    idxStr = idxStr.substr(0, 19) + "...";
        //                oss << std::setw(6) << evt.seqNum << " | "
        //                    << std::setw(6) << evt.opTag << " | "
        //                    << std::setw(7) << rtName(evt.regType) << " | "
        //                    << std::setw(5) << evt.count << " | "
        //                    << std::setw(22) << idxStr << " | "
        //                    << std::setw(4)
        //                    << getVal(evt.globalInUse, Register::Type::Scalar) << "/"
        //                    << std::setw(4)
        //                    << getVal(evt.globalInUse, Register::Type::Vector) << "/"
        //                    << std::setw(4)
        //                    << getVal(evt.globalInUse, Register::Type::Accumulator)
        //                    << "       | "
        //                    << std::setw(3)
        //                    << getVal(evt.globalHWM, Register::Type::Scalar) << "/"
        //                    << std::setw(3)
        //                    << getVal(evt.globalHWM, Register::Type::Vector) << "/"
        //                    << std::setw(3)
        //                    << getVal(evt.globalHWM, Register::Type::Accumulator)
        //                    << "\n";
        //            }
        //            oss << "\n";
        //            // ============================================================
        //            // SECTION 4: Global Summary (at end of program)
        //            // ============================================================
        //            oss << std::string(72, '=') << "\n";
        //            oss << "  SECTION 4: Global Summary (at end of program)\n";
        //            oss << std::string(72, '=') << "\n\n";
        //            oss << "  Register Pools:\n";
        //            for(auto rt : TRACKED_TYPES)
        //            {
        //                auto a = ctx->allocator(rt);
        //                if(!a || a->size() == 0)
        //                    continue;
        //                int pool  = static_cast<int>(a->size());
        //                int inUse = pool - a->currentlyFree();
        //                int hwm   = a->maxUsed() + 1;
        //                int free  = pool - inUse;
        //                oss << "    " << std::setw(7) << rtName(rt) << ": "
        //                    << pool << " total, "
        //                    << inUse << " in-use, "
        //                    << free << " free, "
        //                    << "HWM = " << hwm << "\n";
        //            }
        //            oss << "\n";
        //            oss << "  Fragmentation (free ranges within [0, HWM)):\n";
        //            for(auto rt : TRACKED_TYPES)
        //            {
        //                auto a = ctx->allocator(rt);
        //                if(!a || a->size() == 0)
        //                    continue;
        //                oss << "    " << std::setw(7) << rtName(rt) << ": "
        //                    << computeFreeRanges(a) << "\n";
        //            }
        //            oss << "\n";
        //            return oss.str();
        //        }
        //};

        class RegisterCountObserver
        {
        public:
            static constexpr size_t                                        NUM_TRACKED_TYPES = 3;
            static constexpr std::array<Register::Type, NUM_TRACKED_TYPES> TRACKED_TYPES{
                Register::Type::Scalar, Register::Type::Vector, Register::Type::Accumulator};
            // One deferred allocation event
            struct AllocEvent
            {
                int              seqNum;
                int              opTag;
                Register::Type   regType;
                int              count;
                std::vector<int> indices;
                // Global allocator snapshot after this allocation
                std::map<Register::Type, int> globalInUse;
                std::map<Register::Type, int> globalHWM;
            };
            // Per-operation summary
            struct OpSummary
            {
                int instructionCount = 0;
                // Deferred allocations: type -> count
                std::map<Register::Type, int> allocCount;
                // Deferred allocations: type -> sorted list of indices
                std::map<Register::Type, std::vector<int>> allocIndices;
                // Peak global register usage observed at any instruction
                // of this operation. This is the max of
                // (allocator->size() - allocator->currentlyFree())
                // across all instructions that belong to this operation.
                // It reflects total system register pressure at the
                // moments when this operation's instructions ran.
                std::map<Register::Type, int> peakGlobalInUse;
            };
            RegisterCountObserver() {}
            RegisterCountObserver(ContextPtr context)
                : m_context(context)
            {
            }
            InstructionStatus peek(Instruction const& inst) const
            {
                return {};
            }
            void modify(Instruction& inst) const {}
            void observe(Instruction const& inst)
            {
                auto ctx = m_context.lock();
                if(!ctx)
                    return;
                m_seqNum++;
                auto ctrlOp = inst.innerControlOp();
                // -------------------------------------------------------
                // 1. For every instruction with an op tag, snapshot the
                //    global in-use count and update peak for that op.
                //    This runs on EVERY instruction of the operation,
                //    not just ones with allocations.
                // -------------------------------------------------------
                if(ctrlOp.has_value())
                {
                    int   opTag   = ctrlOp.value();
                    auto& summary = m_opSummaries[opTag];
                    summary.instructionCount++;
                    for(auto rt : TRACKED_TYPES)
                    {
                        auto a = ctx->allocator(rt);
                        if(!a || a->size() == 0)
                            continue;
                        // Exact global in-use count right now
                        int inUse = static_cast<int>(a->size()) - a->currentlyFree();
                        // Track peak across all instructions of this op
                        auto it = summary.peakGlobalInUse.find(rt);
                        if(it == summary.peakGlobalInUse.end())
                            summary.peakGlobalInUse[rt] = inUse;
                        else
                            it->second = std::max(it->second, inUse);
                    }
                }
                // -------------------------------------------------------
                // 2. Record deferred allocations from this instruction.
                // -------------------------------------------------------
                for(auto const& alloc : inst.allocations())
                {
                    if(!alloc)
                        continue;
                    auto rt = alloc->regType();
                    if(!isTracked(rt))
                        continue;
                    if(alloc->allocationState() != Register::AllocationState::Allocated)
                        continue;
                    auto const& indices = alloc->registerIndices();
                    int         count   = static_cast<int>(indices.size());
                    // Per-operation summary
                    if(ctrlOp.has_value())
                    {
                        auto& summary = m_opSummaries[ctrlOp.value()];
                        summary.allocCount[rt] += count;
                        auto& vec = summary.allocIndices[rt];
                        vec.insert(vec.end(), indices.begin(), indices.end());
                    }
                    // Chronological event log
                    AllocEvent evt;
                    evt.seqNum  = m_seqNum;
                    evt.opTag   = ctrlOp.value_or(-1);
                    evt.regType = rt;
                    evt.count   = count;
                    evt.indices = indices;
                    for(auto t : TRACKED_TYPES)
                    {
                        auto a = ctx->allocator(t);
                        if(a && a->size() > 0)
                        {
                            evt.globalInUse[t] = static_cast<int>(a->size()) - a->currentlyFree();
                            evt.globalHWM[t]   = a->maxUsed() + 1;
                        }
                        else
                        {
                            evt.globalInUse[t] = 0;
                            evt.globalHWM[t]   = 0;
                        }
                    }
                    m_events.push_back(std::move(evt));
                }
                // -------------------------------------------------------
                // 3. Detect FinalInstruction to dump report.
                // -------------------------------------------------------
                auto const&        architecture = ctx->targetArchitecture();
                GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());
                auto               wq           = info.getWaitQueues();
                if(std::find(wq.begin(), wq.end(), GPUWaitQueueType::FinalInstruction) != wq.end())
                {
                    writeReport(ctx);
                }
            }
            static bool runtimeRequired()
            {
                return true;
                //return Settings::getInstance()->get(
                //    Settings::KernelAnalysis);
            }

        private:
            std::weak_ptr<Context>   m_context;
            int                      m_seqNum = 0;
            std::map<int, OpSummary> m_opSummaries;
            std::vector<AllocEvent>  m_events;
            static bool              isTracked(Register::Type rt)
            {
                return std::find(TRACKED_TYPES.begin(), TRACKED_TYPES.end(), rt)
                       != TRACKED_TYPES.end();
            }
            static std::string rtName(Register::Type rt)
            {
                switch(rt)
                {
                case Register::Type::Scalar:
                    return "SGPR";
                case Register::Type::Vector:
                    return "VGPR";
                case Register::Type::Accumulator:
                    return "ACCVGPR";
                default:
                    return "???";
                }
            }
            /**
               * Format sorted indices into compact ranges.
               * E.g. {0,1,2,5,6,9} -> "[0-2, 5-6, 9]"
               */
            static std::string formatIndices(std::vector<int> indices)
            {
                if(indices.empty())
                    return "[]";
                std::sort(indices.begin(), indices.end());
                std::ostringstream oss;
                oss << "[";
                size_t i     = 0;
                bool   first = true;
                while(i < indices.size())
                {
                    if(!first)
                        oss << ", ";
                    first    = false;
                    size_t j = i;
                    while(j + 1 < indices.size() && indices[j + 1] == indices[j] + 1)
                        j++;
                    if(j == i)
                        oss << indices[i];
                    else
                        oss << indices[i] << "-" << indices[j];
                    i = j + 1;
                }
                oss << "]";
                return oss.str();
            }
            /**
               * Free ranges within [0, maxUsed) for fragmentation.
               */
            static std::string
                computeFreeRanges(std::shared_ptr<Register::Allocator> const& allocator)
            {
                int limit = allocator->maxUsed() + 1;
                if(limit <= 0)
                    return "(none)";
                std::vector<int> freeIdx;
                for(int i = 0; i < limit; ++i)
                    if(allocator->isFree(i))
                        freeIdx.push_back(i);
                if(freeIdx.empty())
                    return "(none)";
                return formatIndices(freeIdx);
            }
            static int getVal(std::map<Register::Type, int> const& m, Register::Type rt)
            {
                auto it = m.find(rt);
                return (it != m.end()) ? it->second : 0;
            }
            void writeReport(ContextPtr ctx) const
            {
                std::string   fn = ctx->assemblyFileName() + ".regcount";
                std::ofstream file(fn, std::ios_base::out);
                if(!file.is_open())
                    return;
                file << generateReport(ctx);
                file.flush();
            }
            std::string generateReport(ContextPtr ctx) const
            {
                std::ostringstream            oss;
                std::map<Register::Type, int> poolSz;
                for(auto rt : TRACKED_TYPES)
                {
                    auto a     = ctx->allocator(rt);
                    poolSz[rt] = a ? static_cast<int>(a->size()) : 0;
                }
                // ========================================================
                // Header
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  rocRoller Register Allocation Report\n";
                oss << std::string(72, '=') << "\n\n";
                oss << "  NOTES:\n"
                    << "  - Only DEFERRED allocations (via scheduling\n"
                    << "    pipeline) are tracked. Eager allocations\n"
                    << "    (e.g. argument loading) are not visible.\n"
                    << "  - 'PeakGlobalInUse' is the maximum of\n"
                    << "    (pool_size - currently_free) observed at\n"
                    << "    any instruction of the operation. This is\n"
                    << "    the peak GLOBAL usage (all operations\n"
                    << "    combined) at the moments when this\n"
                    << "    operation's instructions were scheduled.\n"
                    << "    It reflects register pressure during this\n"
                    << "    operation, not registers used solely by it.\n"
                    << "\n";
                // ========================================================
                // SECTION 1: Per-Op Deferred Allocation Counts
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  SECTION 1: Per-Operation Deferred Register "
                       "Allocation Counts\n";
                oss << std::string(72, '=') << "\n";
                oss << std::setw(8) << "OpTag"
                    << " | " << std::setw(6) << "#Inst"
                    << " | " << std::setw(6) << "SGPR"
                    << " | " << std::setw(6) << "VGPR"
                    << " | " << std::setw(8) << "ACCVGPR"
                    << "\n";
                oss << std::string(8, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(6, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(8, '-') << "\n";
                int tS = 0, tV = 0, tA = 0, tI = 0;
                for(auto const& [opTag, sm] : m_opSummaries)
                {
                    int s = getVal(sm.allocCount, Register::Type::Scalar);
                    int v = getVal(sm.allocCount, Register::Type::Vector);
                    int a = getVal(sm.allocCount, Register::Type::Accumulator);
                    tS += s;
                    tV += v;
                    tA += a;
                    tI += sm.instructionCount;
                    oss << std::setw(8) << opTag << " | " << std::setw(6) << sm.instructionCount
                        << " | " << std::setw(6) << s << " | " << std::setw(6) << v << " | "
                        << std::setw(8) << a << "\n";
                }
                oss << std::string(8, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(6, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(8, '-') << "\n";
                oss << std::setw(8) << "TOTAL"
                    << " | " << std::setw(6) << tI << " | " << std::setw(6) << tS << " | "
                    << std::setw(6) << tV << " | " << std::setw(8) << tA << "\n\n";
                // ========================================================
                // SECTION 2: Per-Op Peak Global Register Usage
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  SECTION 2: Per-Operation Peak Global Register"
                       " Usage\n";
                oss << "  (max of all-operations-combined in-use count\n"
                    << "   observed at each instruction of the operation)\n";
                oss << std::string(72, '=') << "\n";
                oss << std::setw(8) << "OpTag"
                    << " | " << std::setw(6) << "#Inst"
                    << " | " << std::setw(10) << "SGPR_Peak"
                    << " | " << std::setw(10) << "VGPR_Peak"
                    << " | " << std::setw(12) << "ACCVGPR_Peak"
                    << "\n";
                oss << std::string(8, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(10, '-') << "-+-" << std::string(10, '-') << "-+-"
                    << std::string(12, '-') << "\n";
                for(auto const& [opTag, sm] : m_opSummaries)
                {
                    oss << std::setw(8) << opTag << " | " << std::setw(6) << sm.instructionCount
                        << " | " << std::setw(10)
                        << getVal(sm.peakGlobalInUse, Register::Type::Scalar) << " | "
                        << std::setw(10) << getVal(sm.peakGlobalInUse, Register::Type::Vector)
                        << " | " << std::setw(12)
                        << getVal(sm.peakGlobalInUse, Register::Type::Accumulator) << "\n";
                }
                oss << std::string(8, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(10, '-') << "-+-" << std::string(10, '-') << "-+-"
                    << std::string(12, '-') << "\n";
                // Show pool sizes for context
                oss << std::setw(8) << "(pool)"
                    << " | " << std::setw(6) << ""
                    << " | " << std::setw(10) << poolSz[Register::Type::Scalar] << " | "
                    << std::setw(10) << poolSz[Register::Type::Vector] << " | " << std::setw(12)
                    << poolSz[Register::Type::Accumulator] << "\n\n";
                // ========================================================
                // SECTION 3: Per-Op Allocated Register Indices
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  SECTION 3: Per-Operation Allocated Register "
                       "Indices\n";
                oss << std::string(72, '=') << "\n\n";
                for(auto const& [opTag, sm] : m_opSummaries)
                {
                    bool any = false;
                    for(auto rt : TRACKED_TYPES)
                    {
                        auto it = sm.allocIndices.find(rt);
                        if(it != sm.allocIndices.end() && !it->second.empty())
                            any = true;
                    }
                    if(!any)
                        continue;
                    oss << "  Op " << opTag << " (" << sm.instructionCount << " instructions):\n";
                    for(auto rt : TRACKED_TYPES)
                    {
                        auto it = sm.allocIndices.find(rt);
                        if(it == sm.allocIndices.end() || it->second.empty())
                            continue;
                        oss << "    " << std::setw(7) << rtName(rt) << ": "
                            << formatIndices(it->second) << "\n";
                    }
                    oss << "\n";
                }
                // ========================================================
                // SECTION 4: Allocation Event Log
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  SECTION 4: Allocation Event Log "
                       "(chronological)\n";
                oss << "  Each row = one deferred allocation. Global "
                       "state\n"
                    << "  is the snapshot AFTER this allocation.\n";
                oss << std::string(72, '=') << "\n\n";
                oss << std::setw(6) << "Seq"
                    << " | " << std::setw(6) << "OpTag"
                    << " | " << std::setw(7) << "Type"
                    << " | " << std::setw(5) << "Count"
                    << " | " << std::setw(22) << "Indices"
                    << " | "
                    << "InUse(S/V/A) | HWM(S/V/A)\n";
                oss << std::string(6, '-') << "-+-" << std::string(6, '-') << "-+-"
                    << std::string(7, '-') << "-+-" << std::string(5, '-') << "-+-"
                    << std::string(22, '-') << "-+-" << std::string(12, '-') << "-+-"
                    << std::string(11, '-') << "\n";
                for(auto const& evt : m_events)
                {
                    std::string idxStr = formatIndices(evt.indices);
                    if(idxStr.size() > 22)
                        idxStr = idxStr.substr(0, 19) + "...";
                    oss << std::setw(6) << evt.seqNum << " | " << std::setw(6) << evt.opTag << " | "
                        << std::setw(7) << rtName(evt.regType) << " | " << std::setw(5) << evt.count
                        << " | " << std::setw(22) << idxStr << " | " << std::setw(3)
                        << getVal(evt.globalInUse, Register::Type::Scalar) << "/" << std::setw(3)
                        << getVal(evt.globalInUse, Register::Type::Vector) << "/" << std::setw(3)
                        << getVal(evt.globalInUse, Register::Type::Accumulator) << " | "
                        << std::setw(3) << getVal(evt.globalHWM, Register::Type::Scalar) << "/"
                        << std::setw(3) << getVal(evt.globalHWM, Register::Type::Vector) << "/"
                        << std::setw(3) << getVal(evt.globalHWM, Register::Type::Accumulator)
                        << "\n";
                }
                oss << "\n";
                // ========================================================
                // SECTION 5: Global Summary
                // ========================================================
                oss << std::string(72, '=') << "\n";
                oss << "  SECTION 5: Global Summary "
                       "(at end of program)\n";
                oss << std::string(72, '=') << "\n\n";
                oss << "  Register Pools:\n";
                for(auto rt : TRACKED_TYPES)
                {
                    auto a = ctx->allocator(rt);
                    if(!a || a->size() == 0)
                        continue;
                    int pool  = static_cast<int>(a->size());
                    int inUse = pool - a->currentlyFree();
                    int hwm   = a->maxUsed() + 1;
                    oss << "    " << std::setw(7) << rtName(rt) << ": " << pool << " total, "
                        << inUse << " in-use, " << (pool - inUse) << " free, "
                        << "HWM = " << hwm << "\n";
                }
                oss << "\n";
                oss << "  Fragmentation (free ranges within "
                       "[0, HWM)):\n";
                for(auto rt : TRACKED_TYPES)
                {
                    auto a = ctx->allocator(rt);
                    if(!a || a->size() == 0)
                        continue;
                    oss << "    " << std::setw(7) << rtName(rt) << ": " << computeFreeRanges(a)
                        << "\n";
                }
                oss << "\n";
                return oss.str();
            }
        };

        static_assert(CObserverRuntime<RegisterCountObserver>);

    }
}
