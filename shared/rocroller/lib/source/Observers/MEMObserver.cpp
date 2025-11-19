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

        bool useWeightlessObserver(Instruction const& inst, ContextPtr context)
        {
            AssertFatal(context != nullptr);
            const auto target = context->targetArchitecture().target();
            return inst.getAddresses().has_value()
                   && getLdsInfoFromOpcode(inst.getOpCode()).has_value() && target.isCDNAGPU();
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
            return GPUInstructionInfo::isLDS(inst.getOpCode())
                   && !useWeightlessObserver(inst, m_context.lock());
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

        const int WeightlessDSMemObserver::dataQueueSize    = 10;
        const int WeightlessDSMemObserver::commandQueueSize = 8;

        WeightlessDSMemObserver::WeightlessDSMemObserver(ContextPtr ctx)
            : m_context(ctx)
            , m_programCycle(0)
        {
        }

        InstructionStatus WeightlessDSMemObserver::peek(Instruction const& inst) const
        {
            InstructionStatus status;
            if(GPUInstructionInfo::isLDS(inst.getOpCode())
               && useWeightlessObserver(inst, m_context.lock()))
            {
                const auto ldsInfo = getLdsInfoFromOpcode(inst.getOpCode());

                const auto [direction, dwords] = ldsInfo.value();
                const auto requiredSlots       = queueSlots(direction, dwords);
                const auto remainingSlots      = calculateRemainingSlots();

                if(m_commandQueue.size() >= commandQueueSize)
                {
                    const auto completionCycle = m_commandQueue.front();
                    status.stallCycles         = completionCycle - m_programCycle;
                }

                if(requiredSlots > remainingSlots)
                {
                    const auto completionCycle = m_dataQueue[requiredSlots - remainingSlots - 1];
                    status.stallCycles
                        = std::max(status.stallCycles, completionCycle - m_programCycle);
                }

                LDSBankModel::MemoryOpLDS memOp{direction};
                status.additionalCycles
                    = LDSBankModel::getInstructionIssueCycles(memOp, dwords) / 4 - 1;
            }
            return status;
        }

        void WeightlessDSMemObserver::modify(Instruction& inst) const
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode())
               && useWeightlessObserver(inst, m_context.lock()))
            {
                const auto status = peek(inst);
                inst.addComment(fmt::format("WeightlessDSMemObserver {}: stall {}, additional {}",
                                            m_programCycle,
                                            status.stallCycles,
                                            status.additionalCycles));
            }
        }

        void WeightlessDSMemObserver::observe(Instruction const& inst)
        {
            int wait = inst.getWaitCount().dscnt();
            // TODO: handle waitcount

            m_programCycle += inst.totalCycles();

            while(!m_commandQueue.empty() && m_programCycle >= m_commandQueue.front())
            {
                m_commandQueue.pop_front();
            }
            while(!m_dataQueue.empty() && m_programCycle >= m_dataQueue.front())
            {
                m_dataQueue.pop_front();
            }

            if(GPUInstructionInfo::isLDS(inst.getOpCode())
               && useWeightlessObserver(inst, m_context.lock()))
            {
                auto ldsInfo = getLdsInfoFromOpcode(inst.getOpCode());

                auto [direction, dwords] = ldsInfo.value();
                int requiredSlots        = queueSlots(direction, dwords);

                LDSBankModel::MemoryOpLDS memOp{direction};

                std::vector<size_t> addresses = inst.getAddresses().value();

                LDSBankModel::RuntimeLDSInstruction ldsInst{memOp, dwords, addresses};

                auto ctx = m_context.lock();
                auto gfx = ctx->targetArchitecture().target().gfx;

                int cycles = LDSBankModel::getInstructionDataCycles(ldsInst, gfx) / 4;

                AssertFatal(calculateRemainingSlots() >= requiredSlots
                                && m_commandQueue.size() < commandQueueSize,
                            "Expected queue space to be accounted for in peek function and passed "
                            "through to total cycles calculation.");

                const auto base = m_commandQueue.empty() ? m_programCycle : m_commandQueue.back();

                m_commandQueue.push_back(base + cycles);

                if(direction == LDSBankModel::LdsDirection::Write)
                {
                    const auto cyclesPerSlot = cycles / requiredSlots;
                    for(int i = 0; i < requiredSlots; ++i)
                    {
                        const auto queueFreedCycle = base + i * cyclesPerSlot;
                        m_dataQueue.push_back(queueFreedCycle);
                    }
                }
                else if(direction == LDSBankModel::LdsDirection::Read)
                {
                    AssertFatal(requiredSlots == 1);
                    const auto queueFreedCycle = base;
                    m_dataQueue.push_back(queueFreedCycle);
                }

                const_cast<Instruction&>(inst).addComment(
                    fmt::format("WeightlessDSMemObserver {}: {} cycles, "
                                "{} slots, remaining {} slots, "
                                "command queue size {}, data queue size {}",
                                m_programCycle,
                                cycles,
                                requiredSlots,
                                calculateRemainingSlots(),
                                m_commandQueue.size(),
                                m_dataQueue.size()));
            }
        }

        int WeightlessDSMemObserver::calculateRemainingSlots() const
        {
            int usedSlots = 0;
            for(const auto& slotFreedCycle : m_dataQueue)
            {
                if(slotFreedCycle > m_programCycle)
                {
                    usedSlots++;
                }
            }
            return dataQueueSize - usedSlots;
        }
    }
}
