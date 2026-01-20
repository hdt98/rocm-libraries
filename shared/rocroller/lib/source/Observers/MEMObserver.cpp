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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        bool useWeightlessObserver(Instruction const& inst, ContextPtr context)
        {
            AssertFatal(context != nullptr);

            const DSObserverType observerType = context->kernelOptions()->dsObserver;

            if(observerType == DSObserverType::WeightlessDSMemObserver)
            {
                const auto addrs = inst.getModelledAddresses();
                AssertFatal(addrs.has_value(), ShowValue(inst.toString(LogLevel::Terse)));
                AssertFatal(addrs->size() % 64 == 0, ShowValue(addrs->size()));
                AssertFatal(LDSModel::getLdsInfoFromOpcodeIfSupported(inst.getOpCode()).has_value(),
                            ShowValue(inst.toString(LogLevel::Terse)));
                AssertFatal(context->targetArchitecture().target().isGFX9GPU(),
                            ShowValue(context->targetArchitecture().target().toString()));
                return true;
            }

            return false;
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

        WeightlessDSMemObserver::WeightlessDSMemObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        InstructionStatus WeightlessDSMemObserver::peek(Instruction const& inst) const
        {
            if(!m_scheduler.has_value())
            {
                // Observers get created before workgroupSize is set in m_context
                auto context = m_context.lock();
                AssertFatal(context != nullptr);

                const auto& gpuArch = context->targetArchitecture().target();
                m_scheduler.emplace(gpuArch.gfx, product(context->kernel()->workgroupSize()) / 64);
            }

            InstructionStatus status;

            if(GPUInstructionInfo::isLDS(inst.getOpCode())
               && useWeightlessObserver(inst, m_context.lock()))
            {
                const auto ldsInfo = LDSModel::getLdsInfoFromOpcodeIfSupported(inst.getOpCode());
                if(ldsInfo.has_value())
                {
                    const auto [direction, dwords] = *ldsInfo;

                    auto ctx = m_context.lock();
                    AssertFatal(ctx != nullptr);

                    std::vector<size_t> addresses = inst.getModelledAddresses().value();
                    auto [stallCycles, additionalCycles]
                        = m_scheduler.value().predictCycles({{direction}, dwords, addresses});

                    status.stallCycles      = stallCycles / 4;
                    status.additionalCycles = additionalCycles / 4;
                }
            }

            const auto waitcnt = inst.getWaitCount().dscnt();
            if(waitcnt >= 0)
            {
                auto ctx = m_context.lock();
                AssertFatal(ctx != nullptr);
                if(ctx->kernelOptions()->dsObserver == DSObserverType::WeightlessDSMemObserver)
                {
                    auto stallCycles   = m_scheduler.value().predictWaitcntStallCycles(waitcnt);
                    status.stallCycles = stallCycles / 4;
                }
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
                                            m_scheduler.value().getProgramCycle(),
                                            status.stallCycles,
                                            status.additionalCycles));
            }
        }

        void WeightlessDSMemObserver::observe(Instruction const& inst)
        {
            m_scheduler.value().incrementProgramCycleBy(inst.totalCycles() * 4);
            m_scheduler.value().updateQueues();

            if(GPUInstructionInfo::isLDS(inst.getOpCode())
               && useWeightlessObserver(inst, m_context.lock()))
            {
                auto ldsInfo = LDSModel::getLdsInfoFromOpcodeIfSupported(inst.getOpCode());
                if(ldsInfo.has_value())
                {
                    auto [direction, dwords] = *ldsInfo;

                    std::vector<size_t> addresses = inst.getModelledAddresses().value();
                    m_scheduler.value().scheduleInstruction({{direction}, dwords, addresses});
                }
            }
        }
    }
}
