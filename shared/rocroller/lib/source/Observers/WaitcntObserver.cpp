// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        WaitcntState::WaitcntState() = default;

        WaitcntState::WaitcntState(WaitQueueMap<bool> const&             needsWaitZero,
                                   WaitQueueMap<GPUWaitQueueType> const& typeInQueue,
                                   WaitCntQueues const& instruction_queues_with_alloc)
            : m_needsWaitZero(needsWaitZero)
            , m_typeInQueue(typeInQueue)
        {
            // Here we're iterating through all of the Register::ValuePtrs and
            // converting them to RegisterIDs
            for(auto& queue : instruction_queues_with_alloc)
            {
                if(m_instructionQueues.find(queue.first) == m_instructionQueues.end())
                {
                    m_instructionQueues[queue.first] = {};
                }
                for(auto& dsts : queue.second)
                {
                    m_instructionQueues[queue.first].emplace_back(
                        std::vector<Register::RegisterId>{});
                    for(auto& dst : dsts)
                    {
                        if(dst)
                        {
                            for(auto& regid : dst->getRegisterIds())
                            {
                                m_instructionQueues[queue.first]
                                                   [m_instructionQueues[queue.first].size() - 1]
                                                       .emplace_back(regid);
                            }
                        }
                    }
                }
            }
        }

        void WaitcntState::assertSafeToBranchTo(const WaitcntState& branchState,
                                                std::string const&  label,
                                                bool                strict) const
        {
            if(*this == branchState)
                return;

            bool fail = false;

            // In Debug mode, defer throwing the exception until we have
            // captured a more complete error message.
            bool longErrMsg = Settings::Get(Settings::LogLvl) >= LogLevel::Debug;

            std::string msg
                = "Branching to label '" + label + "' with a different waitcnt state.\n";

            if(strict)
            {
                // Strict mode (loops / backward branches): require both sides
                // to have empty queues if they don't match exactly.
                for(auto const& [queue, instructions] : m_instructionQueues)
                {
                    if(m_needsWaitZero.at(queue) || branchState.m_needsWaitZero.at(queue))
                    {
                        fail = true;
                        msg += concatenate(" Wait zero: ",
                                           ShowValue(m_needsWaitZero.at(queue)),
                                           ShowValue(branchState.m_needsWaitZero.at(queue)),
                                           ShowValue(queue),
                                           "\n");

                        if(!longErrMsg)
                            AssertFatal(!fail, msg);
                    }

                    for(auto const& instruction : instructions)
                    {
                        if(!instruction.empty())
                        {
                            fail = true;
                            msg += concatenate(" Extra register at label: ",
                                               ShowValue(instruction),
                                               ShowValue(queue),
                                               "\n");

                            if(!longErrMsg)
                                AssertFatal(!fail, msg);
                        }
                    }

                    for(auto const& instruction : branchState.m_instructionQueues.at(queue))
                    {
                        if(!instruction.empty())
                        {
                            fail = true;
                            msg += concatenate(" Extra register at branch: ",
                                               ShowValue(instruction),
                                               ShowValue(queue),
                                               "\n");

                            if(!longErrMsg)
                                AssertFatal(!fail, msg);
                        }
                    }
                }
            }
            else
            {
                // Relaxed mode (conditionals / forward branches): the label
                // state may be more conservative (superset) than the branch
                // state. Only fail if the branch has entries or flags that the
                // label doesn't know about.
                for(auto const& [queue, labelInstructions] : m_instructionQueues)
                {
                    auto const& branchInstructions = branchState.m_instructionQueues.at(queue);

                    if(branchState.m_needsWaitZero.at(queue) && !m_needsWaitZero.at(queue))
                    {
                        fail = true;
                        msg += concatenate(
                            " Branch needs waitZero but label does not: ", ShowValue(queue), "\n");

                        if(!longErrMsg)
                            AssertFatal(!fail, msg);
                    }

                    if(branchInstructions.size() > labelInstructions.size())
                    {
                        fail = true;
                        msg += concatenate(" Branch queue larger than label queue: ",
                                           " branch=",
                                           branchInstructions.size(),
                                           " label=",
                                           labelInstructions.size(),
                                           ShowValue(queue),
                                           "\n");

                        if(!longErrMsg)
                            AssertFatal(!fail, msg);
                    }
                }
            }
            AssertFatal(!fail, msg);
        }

        WaitcntObserver::WaitcntObserver() = default;

        WaitcntObserver::WaitcntObserver(ContextPtr context)
            : m_context(context)
        {
            m_includeExplanation
                = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Verbose;
            m_displayState = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Debug;

            {
                auto const& architecture = context->targetArchitecture();

                auto hasBarrier       = architecture.HasCapability(GPUCapability::s_barrier);
                auto hasBarrierSignal = architecture.HasCapability(GPUCapability::s_barrier_signal);

                AssertFatal(hasBarrier || hasBarrierSignal,
                            "Either s_barrier or s_barrier_signal must be supported.",
                            ShowValue(architecture.target()));

                m_barrierOpcode = hasBarrierSignal ? "s_barrier_signal" : "s_barrier";
            }

            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
            {
                GPUWaitQueue waitQueue         = static_cast<GPUWaitQueue>(i);
                m_instructionQueues[waitQueue] = {};
                m_needsWaitZero[waitQueue]     = false;
                m_typeInQueue[waitQueue]       = GPUWaitQueueType::None;
            }

            AssertFatal(m_instructionQueues.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_instructionQueues.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
            AssertFatal(m_needsWaitZero.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_needsWaitZero.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
            AssertFatal(m_typeInQueue.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_typeInQueue.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
        };

        void WaitcntObserver::observe(Instruction const& inst)
        {
            auto               context      = m_context.lock();
            auto const&        architecture = context->targetArchitecture();
            GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());

            if(context->kernelOptions()->assertWaitCntState)
            {
                if(info.isBranch())
                {
                    AssertFatal(inst.getSrcs()[0],
                                "Branch without a label\n",
                                ShowValue(inst.toString(LogLevel::Debug)));
                    addBranchState(inst.getSrcs()[0]->toString());

                    // After an unconditional branch (e.g. end of the if-body),
                    // flag so that addLabelState resets the observer to the
                    // saved branch state instead of continuing with the
                    // state accumulated by the previous block.
                    if(inst.getOpCode() == "s_branch")
                        m_afterUnconditionalBranch = true;
                }
                else if(inst.isLabel())
                {
                    addLabelState(inst.getLabel());
                }
            }

            auto instWaitQueues = info.getWaitQueues();

            WaitCount waiting = inst.getWaitCount();

            if(std::find(
                   instWaitQueues.begin(), instWaitQueues.end(), GPUWaitQueueType::FinalInstruction)
               != instWaitQueues.end())
            {
                waiting = WaitCount::Zero(context->targetArchitecture());

                if(context->kernelOptions()->assertWaitCntState)
                {
                    assertLabelConsistency();
                }
            }

            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
            {
                applyWaitToQueue(waiting.getCount(static_cast<GPUWaitQueue>(i)),
                                 static_cast<GPUWaitQueue>(i));
            }

            for(GPUWaitQueueType queueType : instWaitQueues)
            {
                GPUWaitQueue waitQueue(queueType);
                if(queueType != GPUWaitQueueType::None
                   && m_instructionQueues.find(waitQueue) != m_instructionQueues.end())
                {
                    int instWaitCnt = info.getWaitCount();
                    if(instWaitCnt >= 0)
                    {
                        if(instWaitCnt == 0)
                        {
                            m_needsWaitZero[waitQueue] = true;
                            instWaitCnt                = 1;
                        }
                        else if(m_typeInQueue[waitQueue] != GPUWaitQueueType::None
                                && m_typeInQueue[waitQueue] != queueType)
                        {
                            m_needsWaitZero[waitQueue] = true;
                        }
                        for(int i = 0; i < instWaitCnt; i++)
                        {
                            WaitQueueRegisters queueRegisters;
                            append(queueRegisters, inst.getAllDsts());
                            // track LDS access to avoid write-after-read races.
                            auto isLDSReg = [](Register::ValuePtr const reg) -> bool {
                                return reg->regType() == Register::Type::LocalData;
                            };
                            append(queueRegisters, filter(isLDSReg, inst.getAllSrcs()));

                            m_instructionQueues[waitQueue].push_back(std::move(queueRegisters));
                        }
                        m_typeInQueue[waitQueue] = queueType;
                    }
                }
            }
        }

        std::string WaitcntObserver::getWaitQueueState() const
        {
            std::stringstream retval;
            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
            {
                GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);

                // Only include state information for wait queues in a non-default state.
                if(m_needsWaitZero.at(waitQueue)
                   || m_typeInQueue.at(waitQueue) != GPUWaitQueueType::None
                   || m_instructionQueues.at(waitQueue).size() > 0)
                {
                    if(retval.rdbuf()->in_avail() == 0)
                    {
                        retval << "\nWait Queue State:";
                    }
                    retval << "\n--Queue: " << waitQueue.toString();
                    retval << "\n----Needs Wait Zero: "
                           << (m_needsWaitZero.at(waitQueue) ? "True" : "False");
                    retval << "\n----Type In Queue  : " << m_typeInQueue.at(waitQueue).toString();
                    retval << "\n----Registers      : ";

                    for(int queue_i = 0; queue_i < m_instructionQueues.at(waitQueue).size();
                        queue_i++)
                    {
                        retval << "\n------Dst: {";
                        for(auto& reg : m_instructionQueues.at(waitQueue)[queue_i])
                        {
                            if(reg)
                            {
                                retval << reg->toString() << ", ";
                            }
                        }
                        retval << "}";
                    }
                }
            }
            return retval.str();
        }

        WaitCount WaitcntObserver::computeImplicitWaitCount(Instruction const& inst,
                                                            std::string*       explanation) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            WaitCount rv;

            AssertFatal(architecture.HasCapability(GPUCapability::s_barrier)
                            || architecture.HasCapability(GPUCapability::s_barrier_signal),
                        "Either s_barrier or s_barrier_signal must be supported");
            if(inst.getOpCode() == m_barrierOpcode)
            {
                if(context->kernelOptions()->alwaysWaitZeroBeforeBarrier)
                {
                    if(explanation != nullptr)
                    {
                        *explanation += "WaitCnt Needed: alwaysWaitZeroBeforeBarrier is set.\n";
                    }
                    rv.combine(WaitCount::Zero(architecture));
                }
            }

            return rv;
        }

        WaitCount WaitcntObserver::computeWaitCount(Instruction const& inst,
                                                    std::string*       explanation) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            WaitCount retval = computeImplicitWaitCount(inst, explanation);

            // No wait required before LDS reads as the wait happens before LDS barriers
            if(GPUInstructionInfo::isLDSRead(inst.getOpCode()))
                return retval.getAsSaturatedWaitCount(architecture);

            if(inst.getOpCode().size() > 0 && inst.hasRegisters())
            {
                for(int i = 0; i < static_cast<int>(GPUWaitQueue::Count); i++)
                {
                    GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);
                    for(int queue_i = m_instructionQueues.at(waitQueue).size() - 1; queue_i >= 0;
                        queue_i--)
                    {
                        if(inst.isAfterWriteDependency(m_instructionQueues.at(waitQueue)[queue_i]))
                        {
                            if(m_needsWaitZero.at(waitQueue))
                            {
                                retval.combine(WaitCount(architecture, waitQueue, 0));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + waitQueue.toString()
                                                    + "', which needs a wait zero.\n";
                                }
                            }
                            else
                            {
                                int waitval
                                    = m_instructionQueues.at(waitQueue).size() - (queue_i + 1);
                                retval.combine(WaitCount(architecture, waitQueue, waitval));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + waitQueue.toString() + "', at "
                                                    + std::to_string(queue_i)
                                                    + " and the queue size is "
                                                    + std::to_string(
                                                        m_instructionQueues.at(waitQueue).size())
                                                    + ", so a waitcnt of " + std::to_string(waitval)
                                                    + " is required.\n";
                                }
                            }
                            break;
                        }
                    }
                }
            }
            return retval.getAsSaturatedWaitCount(architecture);
        }
    }
}
