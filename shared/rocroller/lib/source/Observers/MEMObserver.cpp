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
        bool useWeightlessObserver(Instruction const& inst)
        {
            return inst.getAddresses().has_value();
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

        std::pair<LDSBankModel::LdsDirection, int> getLdsInfoFromOpcode(const std::string& opCode)
        {
            AssertFatal(opCode.find("ds_write") != std::string::npos
                            || opCode.find("ds_read") != std::string::npos,
                        "WeightlessDSMemObserver: Opcode is not an LDS operation: " + opCode);
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
                Throw<FatalError>(
                    "WeightlessDSMemObserver: Unable to determine LDS data size from opcode: "
                    + opCode);

            LDSBankModel::LdsDirection direction = opCode.find("ds_write") != std::string::npos
                                                       ? LDSBankModel::LdsDirection::Write
                                                       : LDSBankModel::LdsDirection::Read;

            return {direction, dwords};
        }

        int queueSlots(LDSBankModel::LdsDirection direction, int dwords)
        {
            // 1 slot for addresses
            if(direction == LDSBankModel::LdsDirection::Write)
                return dwords + 1;
            return 1;
        }

        WeightlessDSMemObserver::WeightlessDSMemObserver(ContextPtr ctx)
            : m_context(ctx)
            , m_remainingSlots(WeightlessDSMemObserver::queueSize)
            , m_programCycle(0)
        {
        }

        InstructionStatus WeightlessDSMemObserver::peek(Instruction const& inst) const
        {
            InstructionStatus status;

            if(GPUInstructionInfo::isLDS(inst.getOpCode()) && useWeightlessObserver(inst))
            {
                auto [direction, dwords] = getLdsInfoFromOpcode(inst.getOpCode());
                int requiredSlots        = queueSlots(direction, dwords);

                if(requiredSlots > m_remainingSlots)
                {
                    // Calculate how many cycles until we have enough slots
                    int cyclesNeeded = 0;
                    int slotsToFree  = requiredSlots - m_remainingSlots;
                    int slotsFreed   = 0;

                    for(const auto& entry : m_queue)
                    {
                        if(slotsFreed >= slotsToFree)
                            break;
                        cyclesNeeded = std::max(cyclesNeeded, entry.completionCycle);
                        slotsFreed += entry.slotsUsed;
                    }

                    status.stallCycles = cyclesNeeded - m_programCycle;
                }
            }
            return status;
        }

        void WeightlessDSMemObserver::modify(Instruction& inst) const {}

        void WeightlessDSMemObserver::observe(Instruction const& inst)
        {
            int wait = inst.getWaitCount().dscnt();
            if(wait >= 0)
            {
                while(m_queue.size() > wait)
                {
                    auto& front    = m_queue.front();
                    m_programCycle = std::max(m_programCycle, front.completionCycle);
                    m_remainingSlots += front.slotsUsed;
                    m_queue.pop_front();
                    const_cast<Instruction&>(inst).addComment(
                        fmt::format("WeightlessDSMemObserver: waited for wait count, remaining "
                                    "slots {}",
                                    m_remainingSlots));
                }
            }

            int instCycles = inst.numExecutedInstructions();

            if(GPUInstructionInfo::isLDS(inst.getOpCode()) && useWeightlessObserver(inst))
            {
                auto [direction, dwords] = getLdsInfoFromOpcode(inst.getOpCode());
                int requiredSlots        = queueSlots(direction, dwords);

                // Wait for enough slots if needed
                while(requiredSlots > m_remainingSlots && !m_queue.empty())
                {
                    auto& front    = m_queue.front();
                    m_programCycle = std::max(m_programCycle, front.completionCycle);
                    m_remainingSlots += front.slotsUsed;
                    m_queue.pop_front();
                    const_cast<Instruction&>(inst).addComment(
                        fmt::format("WeightlessDSMemObserver: waited for slots, remaining slots {}",
                                    m_remainingSlots));
                }

                LDSBankModel::MemoryOpLDS memOp{direction};

                AssertFatal(inst.getAddresses().has_value(),
                            "WeightlessDSMemObserver: LDS instruction missing addresses for cycle "
                            "calculation.");
                std::vector<size_t> addresses = inst.getAddresses().value();

                LDSBankModel::RuntimeLDSInstruction ldsInst{memOp, dwords, addresses};

                auto ctx = m_context.lock();
                auto gfx = ctx->targetArchitecture().target().gfx;

                // Here a cycle means an issue cycle (so a quadcycle)
                int cycles = LDSBankModel::getInstructionCycles(ldsInst, gfx) / 4;

                m_remainingSlots -= requiredSlots;
                m_programCycle += instCycles;

                LDSQueueEntry entry{m_programCycle + cycles, requiredSlots};
                m_queue.push_back(entry);

                std::stringstream queueStatus;
                queueStatus << fmt::format("this takes {} cycles", cycles);
                for(const auto& e : m_queue)
                {
                    queueStatus << fmt::format(
                        "[cycle {}, slots {}] ", e.completionCycle, e.slotsUsed);
                }
                const_cast<Instruction&>(inst).addComment(
                    fmt::format("WeightlessDSMemObserver: queued LDS instruction, remaining slots "
                                "{}, program cycle {}, stall cycles {}, queue: {}",
                                m_remainingSlots,
                                m_programCycle,
                                inst.peekedStatus().stallCycles,
                                queueStatus.str()));
            }
            else
            {
                m_programCycle += instCycles + inst.peekedStatus().stallCycles;
                while(!m_queue.empty() && m_programCycle >= m_queue.front().completionCycle)
                {
                    auto& front = m_queue.front();
                    m_remainingSlots += front.slotsUsed;
                    m_queue.pop_front();
                    const_cast<Instruction&>(inst).addComment(
                        fmt::format("WeightlessDSMemObserver: freed slots, remaining slots {}",
                                    m_remainingSlots));
                }
            }
        }
    }
}
