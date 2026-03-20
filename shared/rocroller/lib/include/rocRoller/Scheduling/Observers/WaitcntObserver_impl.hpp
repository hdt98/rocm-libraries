// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        constexpr inline bool WaitcntObserver::required(GPUArchitectureTarget const& target)
        {
            return true;
        }

        inline InstructionStatus WaitcntObserver::peek(Instruction const& inst) const
        {
            auto rv = InstructionStatus::Wait(computeWaitCount(inst));

            // The new length of each queue is:
            // - The current length of the queue
            // - With the new WaitCount applied.
            // - Plus the contribution from this instruction

            // Get current length of queue
            for(auto const& pair : m_instructionQueues)
            {
                if(pair.second.size() > 0)
                {
                    auto wqType = m_typeInQueue.at(pair.first);
                    auto idx    = static_cast<size_t>(wqType);

                    AssertFatal(idx < rv.waitLengths.size(),
                                ShowValue(static_cast<size_t>(wqType)),
                                ShowValue(rv.waitLengths.size()),
                                ShowValue(pair.second.size()));

                    rv.waitLengths.at(idx) = pair.second.size();
                }
            }

            // Apply the waitcount from this instruction.
            for(int i = 0; i < static_cast<int>(GPUWaitQueueType::Count); i++)
            {
                auto wqType = static_cast<GPUWaitQueueType>(i);
                auto wq     = fromWaitQueueType(wqType);

                auto count = rv.waitCount.getCount(wq);

                if(count >= 0)
                    rv.waitLengths.at(wqType) = std::min(rv.waitLengths.at(wqType), count);
            }

            // Add contribution from this instruction
            GPUInstructionInfo info
                = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());
            auto whichQueues = info.getWaitQueues();
            for(auto qt : whichQueues)
            {
                auto idx = static_cast<size_t>(qt);
                AssertFatal(
                    idx < rv.waitLengths.size(), ShowValue(qt), ShowValue(rv.waitLengths.size()));
                auto waitCount = info.getWaitCount();
                rv.waitLengths.at(qt) += waitCount == 0 ? 1 : waitCount;
            }

            return rv;
        };

        inline void WaitcntObserver::modify(Instruction& inst) const
        {
            auto        context      = m_context.lock();
            auto const& architecture = context->targetArchitecture();

            // Handle if manually specified waitcnts are over the sat limits.
            inst.addWaitCount(inst.getWaitCount().getAsSaturatedWaitCount(architecture));

            std::string  explanation;
            std::string* pExplanation = nullptr;
            if(m_includeExplanation)
                pExplanation = &explanation;

            inst.addWaitCount(computeWaitCount(inst, pExplanation));
            if(m_includeExplanation)
                inst.addComment(explanation);

            if(m_displayState && !inst.isCommentOnly())
            {
                inst.addComment(getWaitQueueState());
            }
        }

        inline void WaitcntObserver::applyWaitToQueue(int waitCnt, GPUWaitQueue queue)
        {
            if(waitCnt >= 0 && m_instructionQueues[queue].size() > (size_t)waitCnt)
            {
                // Do not partially clear the queue if a waitcnt zero is needed.
                if(!(m_needsWaitZero[queue] && waitCnt > 0))
                {
                    m_instructionQueues[queue].erase(m_instructionQueues[queue].begin(),
                                                     m_instructionQueues[queue].begin()
                                                         + m_instructionQueues[queue].size()
                                                         - waitCnt);
                }
                if(m_instructionQueues[queue].size() == 0)
                {
                    m_needsWaitZero[queue] = false;
                    m_typeInQueue[queue]   = GPUWaitQueueType::None;
                }
            }
        }

        inline void WaitcntObserver::addLabelState(std::string const& label)
        {
            if(m_afterUnconditionalBranch)
            {
                // After an unconditional branch the linear-traversal state is stale
                // Restore the live state from the branch that targets this label so
                // that subsequent instructions get correct wait-count computation.
                if(m_liveBranchStates.contains(label))
                {
                    auto& bs = m_liveBranchStates.at(label);
                    m_instructionQueues = bs.instructionQueues;
                    m_needsWaitZero     = bs.needsWaitZero;
                    m_typeInQueue       = bs.typeInQueue;
                }
                m_afterUnconditionalBranch = false;
            }
            else
            {
                // At join points (e.g. ConditionalBottom) the label is reached
                // both by fall-through and by a forward branch whose path may
                // have outstanding operations the fall-through path doesn't.
                // Both paths diverged from the same pre-conditional state.
                //
                // We perform a per-position union of entries (back-aligned)
                // so that every register that might be in-flight on either
                // path is present in the merged queue at a position that
                // yields a sufficient waitcnt. Simply taking the
                // larger queue may lose track of registers that only
                // appear in the shorter queue.
                auto it = m_liveBranchStates.find(label);
                if(m_liveBranchStates.contains(label))
                {
                    auto& bs = m_liveBranchStates.at(label);
                    for(auto& [queue, branchEntries] : bs.instructionQueues)
                    {
                        auto& labelEntries = m_instructionQueues[queue];

                        // When the two queues have different sizes, we must
                        // align entries by their distance from the end so
                        // the computed waitcnt is ≤ the correct value on
                        // both paths. We pad the shorter queue from the
                        // front with empty entries.
                        size_t mergedSize = std::max(labelEntries.size(), branchEntries.size());

                        if(labelEntries.size() < mergedSize)
                        {
                            size_t padCount = mergedSize - labelEntries.size();
                            labelEntries.insert(
                                labelEntries.begin(), padCount, WaitQueueRegisters{});
                        }

                        // Offset so branch position i maps to merged
                        // position i + branchOffset (back-aligned).
                        size_t branchOffset = mergedSize - branchEntries.size();

                        // Per-position union: merge branch registers into
                        // the label entries at the back-aligned position.
                        for(size_t i = 0; i < branchEntries.size(); i++)
                        {
                            auto& mergedEntry = labelEntries[i + branchOffset];

                            for(auto const& branchReg : branchEntries[i])
                            {
                                if(!branchReg)
                                    continue;

                                // Skip if this register is already tracked
                                // at this position.
                                bool alreadyPresent = std::any_of(
                                    mergedEntry.begin(), mergedEntry.end(), [&](auto const& r) {
                                        return r && r->intersects(branchReg);
                                    });

                                if(alreadyPresent)
                                    continue;

                                // Find an empty slot in the entry.
                                auto slotIt = std::find_if(mergedEntry.begin(),
                                                           mergedEntry.end(),
                                                           [](auto const& r) { return !r; });

                                if(slotIt != mergedEntry.end())
                                    *slotIt = branchReg;
                                else // No room, fall back to wait-zero
                                    m_needsWaitZero[queue] = true;
                            }
                        }

                        m_needsWaitZero[queue] = m_needsWaitZero[queue] || bs.needsWaitZero[queue];
                        if(m_typeInQueue[queue] == GPUWaitQueueType::None)
                        {
                            m_typeInQueue[queue] = bs.typeInQueue[queue];
                        }
                        else if(bs.typeInQueue[queue] != GPUWaitQueueType::None
                                && bs.typeInQueue[queue] != m_typeInQueue[queue])
                        {
                            m_needsWaitZero[queue] = true;
                        }
                    }
                }
            }

            m_labelStates[label]
                = WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues);
        }

        inline void WaitcntObserver::addBranchState(std::string const& label)
        {
            if(!m_branchStates.contains(label))
            {
                m_branchStates[label] = {};
            }

            m_branchStates[label].emplace_back(
                WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues));

            // If the label hasn't been encountered yet, this is a forward
            // branch (conditional pattern). Backward branches (loops) target
            // labels that have already been recorded.
            bool isForward = !m_labelStates.contains(label);
            if(isForward)
            {
                m_forwardBranchLabels.insert(label);

                // Save the native state so we can restore/merge it at the label.
                if(!m_liveBranchStates.contains(label))
                {
                    m_liveBranchStates[label]
                        = {m_instructionQueues, m_needsWaitZero, m_typeInQueue};
                }
            }
        }

        inline void WaitcntObserver::assertLabelConsistency()
        {
            for(auto const& label_state : m_labelStates)
            {
                if(m_branchStates.contains(label_state.first))
                {
                    // Forward branches (conditionals): relaxed check, label
                    // state may be a superset of branch state.
                    // Backward branches (loops): strict check, states must
                    // match exactly.
                    bool isForward = m_forwardBranchLabels.contains(label_state.first);
                    bool strict    = !isForward;

                    for(auto const& branch_state : m_branchStates.at(label_state.first))
                    {
                        label_state.second.assertSafeToBranchTo(
                            branch_state, label_state.first, strict);
                    }
                }
            }
        }

    };

}
