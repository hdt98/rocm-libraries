/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <fmt/format.h>

#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/LDSBankModel.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller::Scheduling::LDSBankModel
{
    std::optional<std::pair<LdsDirection, int>> getLdsInfoFromOpcode(const std::string& opCode)
    {
        // Model does not support sub-dword or special opcodes
        // e.g. ds_read_u8, ds_read2st64_b32

        LdsDirection direction;
        if(opCode.find("ds_write_") != std::string::npos)
            direction = LdsDirection::Write;
        else if(opCode.find("ds_read_") != std::string::npos)
            direction = LdsDirection::Read;
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

    int getQueueSlotsRequired(LdsDirection direction, int dwords)
    {
        if(direction == LdsDirection::Write)
            return dwords + 1;
        return 1;
    }

    LDSScheduler::LDSScheduler(GPUArchitectureGFX gfx, int waveCount)
        : m_gfx(gfx)
        , m_programCycle(0)
        // Both SPs share the same LDS, so if more than one wave is active,
        // conflicts double (assuming waves perfectly interleave)
        , m_interWaveMultiplier(std::min(2, waveCount))
        // With 3 or more waves, two SIMDs will be active on at least one SP
        // so two SIMDs share the same LDS queues
        , m_intraSPMultiplier(waveCount > 2 ? 2 : 1)
    {
        AssertFatal(waveCount >= 1, ShowValue(waveCount));
        AssertFatal(waveCount != 3, "wave count of 3 is untested");
    }

    // Advance program cycle by delta cycles
    void LDSScheduler::incrementProgramCycle(int cycles)
    {
        m_programCycle += cycles;
    }

    void LDSScheduler::reset()
    {
        m_programCycle = 0;
        m_commandQueue.clear();
        m_waitcntQueue.clear();
        m_dataQueue.clear();
    }

    int LDSScheduler::getRemainingDataSlots() const
    {
        int usedSlots = 0;
        for(const auto& slotFreedCycle : m_dataQueue)
        {
            if(slotFreedCycle > static_cast<unsigned int>(m_programCycle))
            {
                usedSlots++;
            }
        }
        return dataQueueSize - usedSlots;
    }

    std::tuple<int, int> LDSScheduler::predictStallCycles(const RuntimeLDSInstruction& instr) const
    {
        int stallCycles = 0;

        const auto multiplier            = getIntraSPConflictMultiplier();
        const auto [direction, dwords]   = std::make_pair(instr.memoryOp.direction, instr.dwords);
        const auto requiredDataSlots     = getQueueSlotsRequired(direction, dwords) * multiplier;
        const auto requiredCommandSlots  = multiplier;
        const auto remainingDataSlots    = getRemainingDataSlots();
        const auto remainingCommandSlots = commandQueueSize - m_commandQueue.size();

        if(requiredCommandSlots > remainingCommandSlots)
        {
            const auto completionCycle
                = m_commandQueue[requiredCommandSlots - remainingCommandSlots - 1];
            stallCycles = std::max(stallCycles, static_cast<int>(completionCycle) - m_programCycle);
        }

        if(requiredDataSlots > remainingDataSlots)
        {
            const auto completionCycle = m_dataQueue[requiredDataSlots - remainingDataSlots - 1];
            stallCycles = std::max(stallCycles, static_cast<int>(completionCycle) - m_programCycle);
        }

        MemoryOpLDS memOp{direction};
        int         additionalCycles = (getInstructionIssueCycles(memOp, dwords) * multiplier) - 4;

        return std::make_tuple(stallCycles, additionalCycles);
    }

    int LDSScheduler::predictWaitcntStall(int waitcnt) const
    {
        int stallCycles = 0;
        AssertFatal(m_waitcntQueue.size() <= std::numeric_limits<int>::max(),
                    "Waitcnt queue size exceeds int max");
        const auto size = static_cast<int>(m_waitcntQueue.size());
        if(waitcnt >= 0 && size > waitcnt)
        {
            const auto commandsToWaitFor = size - waitcnt - 1;
            AssertFatal(commandsToWaitFor >= 0 && commandsToWaitFor < size,
                        ShowValue(commandsToWaitFor),
                        ShowValue(size),
                        ShowValue(waitcnt));
            const auto waitCompletionCycle = m_waitcntQueue[commandsToWaitFor];
            stallCycles                    = waitCompletionCycle - m_programCycle;
        }

        return stallCycles;
    }

    // TODO: this is a copypaste from test/common
    std::vector<size_t>
        generateLDSAddresses(size_t count, size_t strideMultiplier, size_t instrDwords)
    {
        std::vector<size_t> addresses;
        for(size_t workitemId = 0; workitemId < count; ++workitemId)
        {
            size_t address = workitemId * (4 * strideMultiplier * instrDwords);
            addresses.push_back(address);
        }
        return addresses;
    }

    void LDSScheduler::scheduleInstruction(const RuntimeLDSInstruction& instr)
    {
        updateQueues();

        int requiredSlots = getQueueSlotsRequired(instr.memoryOp.direction, instr.dwords);
        int dataCycles = getInstructionDataCycles(instr, m_gfx) * getInterWaveConflictMultiplier();

        for(int i = 0; i < getIntraSPConflictMultiplier(); ++i)
        {
            AssertFatal(getRemainingDataSlots() >= requiredSlots
                            && m_commandQueue.size() < static_cast<size_t>(commandQueueSize),
                        "Expected queue space to be accounted for in predict function and passed "
                        "through to total cycles calculation",
                        ShowValue(getRemainingDataSlots()),
                        ShowValue(requiredSlots));
            ;

            if(instr.memoryOp.direction == LdsDirection::Write)
            {
                auto cmdBase = m_programCycle + dataCycles;
                if(!m_commandQueue.empty())
                {
                    cmdBase = m_commandQueue.back() + dataCycles;
                }

                auto waitcntBase = m_programCycle + 40 + dataCycles;
                if(!m_waitcntQueue.empty())
                {
                    waitcntBase = std::max(waitcntBase, m_waitcntQueue.back() + dataCycles);
                }

                m_commandQueue.push_back(cmdBase);
                m_waitcntQueue.push_back(waitcntBase);
                {
                    auto const base
                        = m_commandQueue.empty() ? (m_programCycle) : (m_commandQueue.back());
                    const auto fraction = dataCycles / requiredSlots;
                    for(int j = 0; j < requiredSlots; ++j)
                    {
                        m_dataQueue.push_back(base + (j * 1) * fraction);
                    }
                }
            }
            else if(instr.memoryOp.direction == LdsDirection::Read)
            {
                auto cmdBase = m_programCycle + dataCycles;
                if(!m_commandQueue.empty())
                {
                    cmdBase = m_commandQueue.back() + dataCycles;
                }
                auto waitcntBase = m_programCycle + 40 + dataCycles;
                if(!m_waitcntQueue.empty())
                {
                    waitcntBase = std::max(waitcntBase, m_waitcntQueue.back() + dataCycles);
                }

                AssertFatal(requiredSlots == 1,
                            ShowValue(requiredSlots)); // read should only need 1 slot

                m_commandQueue.push_back(cmdBase);
                m_waitcntQueue.push_back(waitcntBase);
                m_dataQueue.push_back(cmdBase);
            }
            else
            {
                Throw<FatalError>("Unsupported LDS direction", ShowValue(instr.toString()));
            }
        }

        const auto cmdQueueStr
            = fmt::format("{}", fmt::join(m_commandQueue.begin(), m_commandQueue.end(), ", "));
        const auto waitcntQueueStr
            = fmt::format("{}", fmt::join(m_waitcntQueue.begin(), m_waitcntQueue.end(), ", "));
        const auto dataQueueStr
            = fmt::format("{}", fmt::join(m_dataQueue.begin(), m_dataQueue.end(), ", "));
        Log::debug("LdsScheduler {}: scheduled ds_{}_b{}, dataCycles {}, "
                   "issue cycles {}, "
                   "command queue cycles [{}], "
                   "data queue [{}], "
                   "waitcnt queue [{}]",
                   m_programCycle,
                   instr.memoryOp.direction == LdsDirection::Read ? "read" : "write",
                   instr.dwords * 32,
                   dataCycles,
                   getInstructionIssueCycles(instr.memoryOp, instr.dwords),
                   cmdQueueStr,
                   dataQueueStr,
                   waitcntQueueStr);
    }

    void LDSScheduler::updateQueues()
    {
        while(!m_commandQueue.empty() && static_cast<int>(m_commandQueue.front()) <= m_programCycle)
        {
            m_commandQueue.pop_front();
        }
        while(!m_waitcntQueue.empty() && static_cast<int>(m_waitcntQueue.front()) <= m_programCycle)
        {
            m_waitcntQueue.pop_front();
        }
        while(!m_dataQueue.empty() && static_cast<int>(m_dataQueue.front()) <= m_programCycle)
        {
            m_dataQueue.pop_front();
        }
        // Assert invariant that all queues are in increasing order
        AssertFatal(
            std::is_sorted(m_commandQueue.begin(), m_commandQueue.end()),
            "Command queue not sorted: ",
            fmt::format("{}", fmt::join(m_commandQueue.begin(), m_commandQueue.end(), ", ")));
        AssertFatal(
            std::is_sorted(m_waitcntQueue.begin(), m_waitcntQueue.end()),
            "Waitcnt queue not sorted: ",
            fmt::format("{}", fmt::join(m_waitcntQueue.begin(), m_waitcntQueue.end(), ", ")));
        AssertFatal(std::is_sorted(m_dataQueue.begin(), m_dataQueue.end()),
                    "Data queue not sorted: ",
                    fmt::format("{}", fmt::join(m_dataQueue.begin(), m_dataQueue.end(), ", ")));
    }

    std::string RuntimeLDSInstruction::toString() const
    {
        std::stringstream ss;
        ss << fmt::format("ds_{}_b{}, baseAddresses: [",
                          memoryOp.direction == LdsDirection::Read ? "read" : "write",
                          dwords * 32);
        rocRoller::streamJoin(ss, baseAddresses, ", ");
        ss << "]";
        return ss.str();
    }

    uint getThreadsPerClock(const MemoryOpLDS& memoryOp, uint dwords, GPUArchitectureGFX gfx)
    {
        // Assumes aligned accesses (e.g. b128 is 16-byte aligned)
        // In future, when linked to codegen, update interface to check alignment of allocation
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == LdsDirection::Read)
        {
            switch(dwords)
            {
            case 1:
                return 16;
            case 2:
                return 16;
            case 3:
                // ds_read_b96 on gfx950 retains the throughput of gfx942
                return 4;
            case 4:
                return 8;
            }
        }
        else
        {
            switch(dwords)
            {
            case 1:
                return 16;
            case 2:
                return 8;
            case 3:
            case 4:
                return 4;
            }
        }
        Throw<FatalError>("Unsupported dword count: ", dwords);
    }

    uint getNumLDSBanks(GPUArchitectureGFX gfx, const MemoryOpLDS& memoryOp, uint dwords)
    {
        // Non ds_read_b128 and ds_read_b64 on gfx950 act as-if there are still only 32 banks for conflict resolution
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == LdsDirection::Read
           && (dwords == 2 || dwords == 4))
        {
            return 64;
        }
        return 32;
    }

    std::vector<std::vector<size_t>> divideIntoThreadGroups(const std::vector<size_t>& addresses,
                                                            uint threadsPerClock)
    {
        AssertFatal(addresses.size() % threadsPerClock == 0,
                    fmt::format("Number of addresses {} is not a multiple of threads per clock {}",
                                addresses.size(),
                                threadsPerClock));

        std::vector<std::vector<size_t>> threadGroups;

        for(size_t groupStart = 0; groupStart < addresses.size(); groupStart += threadsPerClock)
        {
            size_t              groupEnd = std::min(groupStart + threadsPerClock, addresses.size());
            std::vector<size_t> group(addresses.begin() + groupStart, addresses.begin() + groupEnd);
            threadGroups.push_back(group);
        }

        return threadGroups;
    }

    std::map<uint, uint> createBankToAddressCounts(const std::vector<size_t>& baseAddresses,
                                                   uint                       dwords,
                                                   GPUArchitectureGFX         gfx,
                                                   const MemoryOpLDS&         memoryOp)
    {
        std::map<uint, uint> bankToAddressCounts;
        uint                 numBanks = getNumLDSBanks(gfx, memoryOp, dwords);

        for(size_t i = 0; i < baseAddresses.size(); ++i)
        {
            AssertFatal(
                baseAddresses[i] % 4 == 0, "Base address is not dword aligned ", baseAddresses[i]);
            uint baseAddr = baseAddresses[i] / 4; // in dwords

            // Note: using dword as the unit here
            for(uint offset = 0; offset < dwords; offset += 1)
            {
                uint currentAddr = baseAddr + offset;
                uint bankIndex   = currentAddr % numBanks;
                bankToAddressCounts[bankIndex]++;
            }
        }

        return bankToAddressCounts;
    }

    uint calculateBankConflictCycles(const std::map<uint, uint>& bankToAddressCounts)
    {
        if(bankToAddressCounts.empty())
        {
            return 0;
        }

        // The number of clock cycles is determined by the bank accessed by the most addresses,
        // since only one address per bank can be serviced per cycle
        uint maxAddressesPerBank = 0;
        for(const auto& [bankIndex, count] : bankToAddressCounts)
        {
            maxAddressesPerBank = std::max(maxAddressesPerBank, count);
        }

        return maxAddressesPerBank;
    }

    std::vector<std::map<uint, uint>>
        computeThreadGroupBankMappings(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        std::vector<std::map<uint, uint>> threadGroupBankMappings;

        const auto threadGroupAddresses = divideIntoThreadGroups(
            instr.baseAddresses, getThreadsPerClock(instr.memoryOp, instr.dwords, gfx));

        for(const auto& groupAddresses : threadGroupAddresses)
        {
            threadGroupBankMappings.push_back(
                createBankToAddressCounts(groupAddresses, instr.dwords, gfx, instr.memoryOp));
        }

        return threadGroupBankMappings;
    }

    uint calculateTotalCyclesFromBankMappings(
        const std::vector<std::map<uint, uint>>& threadGroupBankMappings)
    {
        uint cycles = 0;
        for(const auto& bankMapping : threadGroupBankMappings)
        {
            cycles += calculateBankConflictCycles(bankMapping);
        }
        return cycles;
    }

    uint getInstructionDataCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        AssertFatal(instr.baseAddresses.size() == 64,
                    "Expected 64 for a wave, got ",
                    instr.baseAddresses.size());

        const auto threadGroupBankMappings = computeThreadGroupBankMappings(instr, gfx);

        for(const auto& mapping : threadGroupBankMappings)
        {
            for(const auto& [bankIndex, count] : mapping)
            {
                Log::trace("Bank {} accessed {} times", bankIndex, count);
            }
        }

        uint cycles = calculateTotalCyclesFromBankMappings(threadGroupBankMappings);

        return cycles;
    }

    uint getInstructionIssueCycles(const MemoryOpLDS& memoryOp, uint dwords)
    {
        // 4 cycles for addresses, additional cycles for write
        uint cycles = 4;
        if(memoryOp.direction == LdsDirection::Write)
        {
            cycles += 4 * dwords;
        }
        return cycles;
    }

    uint getInstructionCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        uint issueCycles = getInstructionIssueCycles(instr.memoryOp, instr.dwords);
        uint dataCycles  = getInstructionDataCycles(instr, gfx);
        return std::max(issueCycles, dataCycles);
    }

    std::string Summary::toString() const
    {
        std::stringstream ss;
        for(auto const& [tag, access] : this->tagToAccess)
        {
            auto const& [ldsTag, accessedBanks, banksToWorkitems] = access;
            ss << fmt::format("Operation tag {} accesses LDS {}:\n", tag, ldsTag);
            for(auto const& [bankIndex, workitemsAccessed, imbalanced] : accessedBanks)
            {
                ss << fmt::format("  Bank {}: {} workitems {}\n",
                                  bankIndex,
                                  workitemsAccessed,
                                  imbalanced ? "(imbalanced)" : "");
            }
        }
        ss << fmt::format("  Imbalanced tags: {}\n", this->imbalancedTags);
        return ss.str();
    }

    std::pair<std::string, uint> stringifyInstructionAnalysis(const RuntimeLDSInstruction& instr,
                                                              GPUArchitectureGFX           gfx)
    {
        std::stringstream ss;

        std::string instructionName;
        if(instr.memoryOp.direction == LdsDirection::Read)
        {
            instructionName = fmt::format("ds_read_b{}", instr.dwords * 32);
        }
        else
        {
            instructionName = fmt::format("ds_write_b{}", instr.dwords * 32);
        }
        ss << fmt::format("  Instruction: {}\n", instructionName);

        // Follows getClockCount (checked against it later for consistency)
        uint cycles = 0;
        {
            const auto threadsPerClock = getThreadsPerClock(instr.memoryOp, instr.dwords, gfx);
            uint       i               = 0;
            for(const auto& groupAddresses :
                divideIntoThreadGroups(instr.baseAddresses, threadsPerClock))
            {
                const auto bankToAddressCounts
                    = createBankToAddressCounts(groupAddresses, instr.dwords, gfx, instr.memoryOp);
                uint groupCycles = calculateBankConflictCycles(bankToAddressCounts);
                ss << fmt::format("    Group {}: threads {}-{}\n",
                                  i,
                                  i * threadsPerClock,
                                  (i + 1) * threadsPerClock - 1);

                uint maxCount = 0;
                for(const auto& [bankIndex, count] : bankToAddressCounts)
                {
                    maxCount = std::max(maxCount, count);
                }
                std::vector<uint> maxBanks;
                for(const auto& [bankIndex, count] : bankToAddressCounts)
                {
                    if(count == maxCount)
                    {
                        maxBanks.push_back(bankIndex);
                    }
                }

                if(!maxBanks.empty())
                {
                    ss << "      Max bank contention: " << maxCount
                       << " addresses/bank for bank(s) ";
                    rocRoller::streamJoin(ss, maxBanks, ", ");
                    ss << "\n";
                }
                ss << fmt::format("      Group cycles: {}\n", groupCycles);
                cycles += groupCycles;
                i++;
            }
        }

        const auto instructionTotalClocks = getInstructionDataCycles(instr, gfx);

        AssertFatal(
            cycles == instructionTotalClocks,
            "Cycle count mismatch between stringify and getInstructionDataCycles, {} and {}",
            cycles,
            instructionTotalClocks);

        ss << fmt::format("    Instruction cycles: {}\n", instructionTotalClocks);

        return std::make_pair(ss.str(), instructionTotalClocks);
    }

    std::string OperationsAnalysis::toString() const
    {
        std::stringstream ss;

        ss << rocRoller::toString(gfx) << "\n";

        for(const auto& [operationTag, opAccesses] : tagToAccess)
        {
            ss << fmt::format("Operation Tag: {}, LDS Tag: {}\n", operationTag, opAccesses.ldsTag);

            uint operationTotalClocks = 0;

            for(const auto& instr : opAccesses.instructions)
            {
                auto [instrStr, instructionClocks] = stringifyInstructionAnalysis(instr, gfx);
                ss << instrStr;
                operationTotalClocks += instructionClocks;
            }
            ss << fmt::format("  Operation cycles: {}\n", operationTotalClocks);
            ss << "\n";
        }

        return ss.str();
    }

    std::ostream& operator<<(std::ostream& stream, OperationsAnalysis const& operationAnalysis)
    {
        return stream << operationAnalysis.toString();
    }

    std::ostream& operator<<(std::ostream& stream, Summary const& summary)
    {
        return stream << summary.toString();
    }

    std::ostream& operator<<(std::ostream& stream, RuntimeLDSInstruction const& instruction)
    {
        return stream << instruction.toString();
    }
}
