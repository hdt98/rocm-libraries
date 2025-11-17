/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <concepts>
#include <optional>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/LDSBankModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        std::optional<std::pair<LDSBankModel::LdsDirection, int>>
            getLdsInfoFromOpcode(const std::string& opCode)
        {
            // Model does not support sub-dword or special opcodes
            // e.g. ds_read_u8, ds_read2st64_b32

            LDSBankModel::LdsDirection direction;
            if(opCode.find("ds_write_") != std::string::npos)
                direction = LDSBankModel::LdsDirection::Write;
            else if(opCode.find("ds_read_") != std::string::npos)
                direction = LDSBankModel::LdsDirection::Read;
            else
                return std::nullopt;

            int dwords;
            if(opCode.find("_b32") != std::string::npos)
                dwords = 1;
            else if(opCode.find("_b64") != std::string::npos)
                dwords = 2;
            else if(opCode.find("_b96") != std::string::npos)
                dwords = 3;
            else if(opCode.find("_b128") != std::string::npos)
                dwords = 4;
            else
                return std::nullopt;

            return std::make_optional(std::make_pair(direction, dwords));
        }

        bool useWeightlessObserver(Instruction const& inst)
        {
            return inst.getAddresses().has_value()
                   && getLdsInfoFromOpcode(inst.getOpCode()).has_value();
        }

        VMEMObserver::VMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx,
                          "VMEM",
                          MEMObserver::getWeights(ctx).vmemCycles,
                          MEMObserver::getWeights(ctx).vmemQueueSize)
        {
        }

        bool VMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVMEM(inst.getOpCode());
        }

        int VMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().vmcnt();
        }

        DSMEMObserver::DSMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx,
                          "DSMEM",
                          MEMObserver::getWeights(ctx).dsmemCycles,
                          MEMObserver::getWeights(ctx).dsmemQueueSize)
        {
        }

        bool DSMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isLDS(inst.getOpCode()) && !useWeightlessObserver(inst);
        }

        int DSMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().dscnt();
        }

        int queueSlots(LDSBankModel::LdsDirection direction, int dwords)
        {
            if(direction == LDSBankModel::LdsDirection::Write)
                return dwords + 1;
            return 1;
        }

        const int WeightlessDSMemObserver::queueSize = 10;

        WeightlessDSMemObserver::WeightlessDSMemObserver(ContextPtr ctx)
            : m_context(ctx)
            , m_programCycle(0)
        {
        }

        InstructionStatus WeightlessDSMemObserver::peek(Instruction const& inst) const
        {
            InstructionStatus status;
            if(GPUInstructionInfo::isLDS(inst.getOpCode()) && useWeightlessObserver(inst))
            {
                auto ldsInfo = getLdsInfoFromOpcode(inst.getOpCode());

                auto [direction, dwords] = ldsInfo.value();
                int requiredSlots        = queueSlots(direction, dwords);
                int remainingSlots       = calculateRemainingSlots();

                if(requiredSlots > remainingSlots)
                {
                    int cyclesNeeded = m_programCycle;
                    int slotsToFree  = requiredSlots - remainingSlots;
                    int slotsFreed   = 0;

                    for(const auto& entry : m_queue)
                    {
                        if(slotsFreed >= slotsToFree)
                            break;
                        cyclesNeeded = std::max(cyclesNeeded, entry.queueFreedCycle);
                        slotsFreed += entry.slotsUsed;
                    }
                    status.stallCycles = cyclesNeeded - m_programCycle;
                }

                LDSBankModel::MemoryOpLDS memOp{direction};
                int issueCycles = LDSBankModel::getInstructionIssueCycles(memOp, dwords) / 4 - 1;
                status.additionalCycles = issueCycles;
            }
            return status;
        }

        void WeightlessDSMemObserver::modify(Instruction& inst) const
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()) && useWeightlessObserver(inst))
            {
                const auto status = peek(inst);
                inst.addComment(fmt::format("WeightlessDSMemObserver {}: stall cycles {}",
                                            m_programCycle,
                                            status.stallCycles));
            }
        }

        void WeightlessDSMemObserver::observe(Instruction const& inst)
        {
            int wait = inst.getWaitCount().dscnt();
            // TODO: handle waitcount

            m_programCycle += inst.totalCycles();

            while(!m_queue.empty() && m_programCycle >= m_queue.front().completionCycle)
            {
                auto& front = m_queue.front();
                const_cast<Instruction&>(inst).addComment(
                    fmt::format("WeightlessDSMemObserver {}: freed {} slots, remaining slots {}",
                                m_programCycle,
                                front.slotsUsed,
                                calculateRemainingSlots()));
                m_queue.pop_front();
            }

            if(GPUInstructionInfo::isLDS(inst.getOpCode()) && useWeightlessObserver(inst))
            {
                auto ldsInfo = getLdsInfoFromOpcode(inst.getOpCode());

                auto [direction, dwords] = ldsInfo.value();
                int requiredSlots        = queueSlots(direction, dwords);

                LDSBankModel::MemoryOpLDS memOp{direction};

                std::vector<size_t> addresses = inst.getAddresses().value();

                LDSBankModel::RuntimeLDSInstruction ldsInst{memOp, dwords, addresses};

                auto ctx = m_context.lock();
                auto gfx = ctx->targetArchitecture().target().gfx;

                // Here a cycle means an issue cycle (a quadcycle)
                int cycles = LDSBankModel::getInstructionDataCycles(ldsInst, gfx) / 4;

                const auto completionCycle
                    = cycles + (m_queue.empty() ? m_programCycle : m_queue.back().completionCycle);

                // Adjustment for read instructions: queue slots appear to be freed a bit later
                int queueFreedCycle;
                if(direction == LDSBankModel::LdsDirection::Read)
                    queueFreedCycle = completionCycle + 1;
                else
                    queueFreedCycle = completionCycle;

                LDSQueueEntry entry{completionCycle, queueFreedCycle, requiredSlots};
                m_queue.push_back(entry);

                std::stringstream queueStatus;
                for(const auto& e : m_queue)
                {
                    queueStatus << fmt::format("[complete {}, freed {}, slots {}] ",
                                               e.completionCycle,
                                               e.queueFreedCycle,
                                               e.slotsUsed);
                }
                const_cast<Instruction&>(inst).addComment(fmt::format(
                    "WeightlessDSMemObserver {}: queued {} taking {} cycles "
                    "using {} slots, freed at cycle {}, remaining slots {}, queue status: {}",
                    m_programCycle,
                    direction == LDSBankModel::LdsDirection::Read ? "read" : "write",
                    cycles,
                    requiredSlots,
                    queueFreedCycle,
                    calculateRemainingSlots(),
                    queueStatus.str()));
            }
        }

        int WeightlessDSMemObserver::calculateRemainingSlots() const
        {
            int usedSlots = 0;
            for(const auto& entry : m_queue)
            {
                if(entry.queueFreedCycle > m_programCycle)
                {
                    usedSlots += entry.slotsUsed;
                }
            }
            return queueSize - usedSlots;
        }
    }
}
