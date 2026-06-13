// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

template <typename T>
__device__ __forceinline__ float StandardEpCompactToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T StandardEpCompactFromFloat(float value) {
  return static_cast<T>(value);
}

__device__ __forceinline__ uint16_t StandardEpCompactBf16Bits(hip_bfloat16 value) {
  return *reinterpret_cast<uint16_t*>(&value);
}

__device__ __forceinline__ hip_bfloat16 StandardEpCompactBf16FromBits(uint16_t bits) {
  hip_bfloat16 value;
  *reinterpret_cast<uint16_t*>(&value) = bits;
  return value;
}

__device__ __forceinline__ void StandardEpCompactAtomicAddBf16(hip_bfloat16* address,
                                                              hip_bfloat16 value) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(address);
  auto* base = reinterpret_cast<unsigned int*>(addr & ~uintptr_t{0x3});
  const bool high_half = (addr & uintptr_t{0x2}) != 0;
  unsigned int old_word = *base;
  unsigned int assumed = 0;
  do {
    assumed = old_word;
    const uint16_t old_bits =
        high_half ? static_cast<uint16_t>(assumed >> 16) : static_cast<uint16_t>(assumed);
    const float old_value = static_cast<float>(StandardEpCompactBf16FromBits(old_bits));
    const hip_bfloat16 new_value =
        StandardEpCompactFromFloat<hip_bfloat16>(old_value + static_cast<float>(value));
    const uint16_t new_bits = StandardEpCompactBf16Bits(new_value);
    const unsigned int new_word =
        high_half ? ((assumed & 0x0000FFFFu) | (static_cast<unsigned int>(new_bits) << 16))
                  : ((assumed & 0xFFFF0000u) | static_cast<unsigned int>(new_bits));
    old_word = atomicCAS(base, assumed, new_word);
  } while (old_word != assumed);
}

template <typename T>
__device__ __forceinline__ void StandardEpCompactWeightedAccumulateTokenOutput(
    const StandardEpCompactCombineArgs& args, const T* srcRow, int64_t flatPosition,
    size_t dimOffset, size_t dimChunk, int laneId) {
  if (args.tokenOutput == nullptr || args.topScoresFlat == nullptr || args.topK <= 0) {
    return;
  }
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, hip_bfloat16>) {
    const int64_t localFlat = flatPosition - args.flatPositionOffset;
    if (localFlat < 0 || localFlat >= args.topScoresFlatSize) {
      return;
    }
    const int64_t tokenRow = localFlat / args.topK;
    if (tokenRow < 0 || tokenRow >= args.tokenOutputRows) {
      return;
    }
    const float score = args.topScoresFlat[localFlat];
    const size_t hiddenDim = args.config.HiddenDimSz();
    T* tokenOut = reinterpret_cast<T*>(args.tokenOutput) + static_cast<size_t>(tokenRow) * hiddenDim;
    for (size_t d = laneId; d < dimChunk; d += warpSize) {
      const float weighted = StandardEpCompactToFloat(srcRow[dimOffset + d]) * score;
      if constexpr (std::is_same_v<T, float>) {
        core::AtomicAddRelaxed(tokenOut + dimOffset + d,
                               StandardEpCompactFromFloat<T>(weighted));
      } else {
        StandardEpCompactAtomicAddBf16(tokenOut + dimOffset + d,
                                       StandardEpCompactFromFloat<T>(weighted));
      }
    }
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint64_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  __syncthreads();
  if (thdId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
    __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp compute token offset on destinition PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      bool routeActive = (args.routeMask == nullptr) || (args.routeMask[i] != 0);
      if (!routeActive) {
        if (laneId == 0) args.dispDestTokIdMap[i] = NullFlatTokenIndex(config);
        continue;
      }
      index_t destExpert = args.tokenIndices[i];
      index_t destPe = destExpert / config.numExpertPerRank;
      index_t destTokId = 0;

      // Deduplicate
      assert(config.numExpertPerToken < warpSize);
      int condition = 0;
      if (laneId < (i % config.numExpertPerToken)) {
        const int prevSlot = srcTokId * config.numExpertPerToken + laneId;
        const bool prevRouteActive =
            (args.routeMask == nullptr) || (args.routeMask[prevSlot] != 0);
        condition =
            prevRouteActive &&
            destPe == (args.tokenIndices[prevSlot] / config.numExpertPerRank);
      }
      if (__any(condition)) {
        // Indicate that this token is already sent to the destination PE by setting an overflow
        // token index
        if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        continue;
      }

      if (laneId == 0) {
        // decide token id in dest pe
        destTokId = atomicAdd(args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1);
        assert(destTokId < config.MaxNumTokensToRecv() &&
               "Total recv token overflow: increase maxTotalRecvTokens");
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        // In dispDestTokIdMap, record the destination slot for this token-expert pair (flat index
        // into the dest PE's recv buffer) In dispTokIdToSrcTokIdMemObj on the dest PE, record which
        // global source token occupies this slot (for combine-phase routing)
        args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
        args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
            FlatTokenIndex(config, myPe, srcTokId);
      }
      destTokId = __shfl(destTokId, 0);

      // Write weights and indices
      if (laneId < config.numExpertPerToken) {
        const int slot = srcTokId * config.numExpertPerToken + laneId;
        const bool slotActive = (args.routeMask == nullptr) || (args.routeMask[slot] != 0);
        const index_t maskedExpert =
            (config.worldSize > 1)
                ? (((destPe + 1) % config.worldSize) * config.numExpertPerRank)
                : args.tokenIndices[slot];
        if (args.weightsBuf) {
          args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              slotActive ? args.weightsBuf[slot] : 0.0f;
        }
        args.shmemOutIndicesMemObj->template GetAs<index_t*>(
            destPe)[destTokId * config.numExpertPerToken + laneId] =
            slotActive ? args.tokenIndices[slot] : maskedExpert;
      }

      // Write scales
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        size_t destScaleOffset = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
        size_t srcScaleOffset = (size_t)srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }

      size_t srcTokOffset = srcTokId * hiddenDim;
      size_t destTokOffset = destTokId * hiddenDim;

      core::WarpCopy(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
                     args.inpTokenBuf + srcTokOffset, hiddenDim);
    }
  }
  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                         StandardEpCompactDispatchIntraNodeKernel                               */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void StandardEpCompactGridWait(uint32_t* barrier) {
  __shared__ uint32_t target;
  __syncthreads();
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t old = atomicAdd(barrier, 1);
    target = ((old / static_cast<uint32_t>(gridDim.x)) + 1u) *
             static_cast<uint32_t>(gridDim.x);
  }
  __syncthreads();
  while (core::AtomicLoadRelaxed(barrier) < target) {
  }
  __syncthreads();
}

template <typename T>
__device__ void StandardEpCompactDispatchIntraNodeKernel_body(StandardEpCompactDispatchArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;
  const int myPe = config.rank;
  const int npes = config.worldSize;
  const size_t hiddenDim = config.HiddenDimSz();

  (void)args.localNumTokensPerExpert;
  (void)args.recvCountsRankMajor;
  (void)args.numSegments;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchStageRows);

  const T* localRows = reinterpret_cast<const T*>(args.localRows);
  T* rankMajorRows = reinterpret_cast<T*>(args.rankMajorRows);
  T* localStageRows = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>();
  index_t* localStageFlat = args.shmemOutIndicesMemObj->template GetAs<index_t*>();
  const bool emitFlatPositions =
      args.localFlatPositions != nullptr && args.rankMajorFlatPositions != nullptr;

  int64_t inputPrefix = 0;
  for (int destPe = 0; destPe < npes; ++destPe) {
    const int64_t count = args.inputSplits[destPe];
    if (destPe == myPe) {
      inputPrefix += count;
      continue;
    }
    assert(count <= config.MaxNumTokensToSendPerRank() &&
           "standard-EP compact dispatch remote input split exceeds MORI staging stride");
    for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
      const int64_t srcRow = inputPrefix + slot;
      const int64_t stageSlot = SendBufSlotOffset(config, destPe, static_cast<int>(slot));
      assert(srcRow < args.numInputRows);
      core::WarpCopy(localStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                     localRows + static_cast<size_t>(srcRow) * hiddenDim, hiddenDim);

      if (emitFlatPositions && laneId == 0) {
        const int64_t localFlat = args.localFlatPositions[srcRow];
        const int64_t encodedFlat =
            (args.flatPositionRankStride > 0)
                ? static_cast<int64_t>(myPe) * args.flatPositionRankStride + localFlat
                : localFlat;
        localStageFlat[stageSlot] = static_cast<index_t>(encodedFlat);
      }
    }
    inputPrefix += count;
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchGridWaitSend);
  StandardEpCompactGridWait(args.dispatchGridBarrier);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchSignalCounts);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      const index_t sendCount = static_cast<index_t>(args.inputSplits[destPe]);
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, sendCount + 1);
    }
  }

  __syncthreads();

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchWaitCounts);
  if (globalWarpId == 0) {
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      const index_t expected = static_cast<index_t>(args.outputSplits[srcPe]);
      const index_t observed = shmem::ShmemInt32WaitUntilGreaterThan(recvTokenNums + srcPe, 0) - 1;
      assert(observed == expected &&
             "standard-EP compact dispatch split mismatch against MORI signal");
      (void)observed;
    }
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchGridWaitRecv);
  StandardEpCompactGridWait(args.combineGridBarrier);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchReceiveCopy);
  const int64_t localExpertNum =
      (args.recvCountsRankMajor != nullptr && args.numSegments > 0 &&
       args.numSegments % npes == 0)
          ? args.numSegments / npes
          : 0;
  if (localExpertNum > 0) {
    int64_t selfInputPrefix = 0;
    for (int prevDestPe = 0; prevDestPe < myPe; ++prevDestPe) {
      selfInputPrefix += args.inputSplits[prevDestPe];
    }

    if (globalWarpId == 0) {
      for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
        int64_t rankCount = 0;
        for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
          rankCount += args.recvCountsRankMajor[srcPe * localExpertNum + localExpert];
        }
        assert(rankCount == args.outputSplits[srcPe] &&
               "standard-EP compact dispatch recv-count metadata does not match output split");
      }
    }

    int64_t expertMajorBase = 0;
    for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
      int64_t expertSrcPrefix = 0;
      for (int srcPe = 0; srcPe < npes; ++srcPe) {
        int64_t rankMajorSlotPrefix = 0;
        for (int64_t prevExpert = 0; prevExpert < localExpert; ++prevExpert) {
          rankMajorSlotPrefix += args.recvCountsRankMajor[srcPe * localExpertNum + prevExpert];
        }
        const int64_t count =
            args.recvCountsRankMajor[srcPe * localExpertNum + localExpert];
        assert(count >= 0);
        assert(rankMajorSlotPrefix + count <= args.outputSplits[srcPe] &&
               "standard-EP compact dispatch recv-count metadata exceeds output split");
        const T* peerStageRows = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(srcPe);
        const index_t* peerStageFlat =
            args.shmemOutIndicesMemObj->template GetAs<index_t*>(srcPe);
        if (srcPe != myPe) {
          assert(args.outputSplits[srcPe] <= config.MaxNumTokensToSendPerRank() &&
                 "standard-EP compact dispatch remote output split exceeds MORI staging stride");
        }
        if (count == 0) {
          expertSrcPrefix += count;
          continue;
        }
        if (count >= globalWarpNum) {
          for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
            const int64_t rankMajorSlot = rankMajorSlotPrefix + slot;
            const int64_t dstRow = expertMajorBase + expertSrcPrefix + slot;
            assert(rankMajorSlot < args.outputSplits[srcPe]);
            assert(dstRow < args.numOutputRows);
            if (srcPe == myPe) {
              const int64_t srcRow = selfInputPrefix + rankMajorSlot;
              assert(srcRow < args.numInputRows);
              core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim,
                             localRows + static_cast<size_t>(srcRow) * hiddenDim, hiddenDim);
              if (emitFlatPositions && laneId == 0) {
                const int64_t localFlat = args.localFlatPositions[srcRow];
                args.rankMajorFlatPositions[dstRow] =
                    (args.flatPositionRankStride > 0)
                        ? static_cast<int64_t>(myPe) * args.flatPositionRankStride + localFlat
                        : localFlat;
              }
            } else {
              const int64_t stageSlot =
                  SendBufSlotOffset(config, myPe, static_cast<int>(rankMajorSlot));
              core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim,
                             peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                             hiddenDim);
              if (emitFlatPositions && laneId == 0) {
                args.rankMajorFlatPositions[dstRow] =
                    static_cast<int64_t>(peerStageFlat[stageSlot]);
              }
            }
          }
        } else {
          MultiWarpIter mwIter(globalWarpNum, static_cast<int>(count), hiddenDim);
          const int64_t copyWorkItems =
              static_cast<int64_t>(count) * static_cast<int64_t>(mwIter.warpsPerItem);
          for (int64_t work = globalWarpId; work < copyWorkItems; work += globalWarpNum) {
            int itemId = 0;
            int inItemPartId = 0;
            size_t dimOffset = 0;
            size_t dimChunk = 0;
            mwIter.Decode(static_cast<int>(work), itemId, inItemPartId, dimOffset, dimChunk);
            if (dimChunk == 0) {
              continue;
            }
            const int64_t slot = static_cast<int64_t>(itemId);
            const int64_t rankMajorSlot = rankMajorSlotPrefix + slot;
            const int64_t dstRow = expertMajorBase + expertSrcPrefix + slot;
            assert(rankMajorSlot < args.outputSplits[srcPe]);
            assert(dstRow < args.numOutputRows);
            if (srcPe == myPe) {
              const int64_t srcRow = selfInputPrefix + rankMajorSlot;
              assert(srcRow < args.numInputRows);
              core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim + dimOffset,
                             localRows + static_cast<size_t>(srcRow) * hiddenDim + dimOffset,
                             dimChunk);
              if (emitFlatPositions && inItemPartId == 0 && laneId == 0) {
                const int64_t localFlat = args.localFlatPositions[srcRow];
                args.rankMajorFlatPositions[dstRow] =
                    (args.flatPositionRankStride > 0)
                        ? static_cast<int64_t>(myPe) * args.flatPositionRankStride + localFlat
                        : localFlat;
              }
            } else {
              const int64_t stageSlot =
                  SendBufSlotOffset(config, myPe, static_cast<int>(rankMajorSlot));
              core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim + dimOffset,
                             peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim + dimOffset,
                             dimChunk);
              if (emitFlatPositions && inItemPartId == 0 && laneId == 0) {
                args.rankMajorFlatPositions[dstRow] =
                    static_cast<int64_t>(peerStageFlat[stageSlot]);
              }
            }
          }
        }
        expertSrcPrefix += count;
      }
      expertMajorBase += expertSrcPrefix;
    }
    assert(expertMajorBase == args.numOutputRows &&
           "standard-EP compact dispatch recv-count metadata does not match output rows");
  } else {
    int64_t outputPrefix = 0;
    int64_t selfInputPrefix = 0;
    for (int prevDestPe = 0; prevDestPe < myPe; ++prevDestPe) {
      selfInputPrefix += args.inputSplits[prevDestPe];
    }
    for (int srcPe = 0; srcPe < npes; ++srcPe) {
      const int64_t count = args.outputSplits[srcPe];
      const T* peerStageRows = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(srcPe);
      const index_t* peerStageFlat =
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(srcPe);
      for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
        const int64_t dstRow = outputPrefix + slot;
        assert(dstRow < args.numOutputRows);
        if (srcPe == myPe) {
          const int64_t srcRow = selfInputPrefix + slot;
          assert(srcRow < args.numInputRows);
          core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim,
                         localRows + static_cast<size_t>(srcRow) * hiddenDim, hiddenDim);
          if (emitFlatPositions && laneId == 0) {
            const int64_t localFlat = args.localFlatPositions[srcRow];
            args.rankMajorFlatPositions[dstRow] =
                (args.flatPositionRankStride > 0)
                    ? static_cast<int64_t>(myPe) * args.flatPositionRankStride + localFlat
                    : localFlat;
          }
        } else {
          assert(count <= config.MaxNumTokensToSendPerRank() &&
                 "standard-EP compact dispatch remote output split exceeds MORI staging stride");
          const int64_t stageSlot = SendBufSlotOffset(config, myPe, static_cast<int>(slot));
          core::WarpCopy(rankMajorRows + static_cast<size_t>(dstRow) * hiddenDim,
                         peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                         hiddenDim);
          if (emitFlatPositions && laneId == 0) {
            args.rankMajorFlatPositions[dstRow] =
                static_cast<int64_t>(peerStageFlat[stageSlot]);
          }
        }
      }
      outputPrefix += count;
    }
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchGridWaitReset);
  StandardEpCompactGridWait(args.dispatchGridBarrier + 1);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactDispatchResetCounters);
  if (globalWarpId == 0) {
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      core::AtomicStoreRelaxedSystem(recvTokenNums + srcPe, 0);
    }
  }

  __syncthreads();
  __threadfence_system();

  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
    }
  }

  if (thdId == 0 && args.totalRecvTokenNum != nullptr) {
    *args.totalRecvTokenNum = static_cast<index_t>(args.numOutputRows);
  }
}

template <typename T>
__global__ void StandardEpCompactDispatchIntraNodeKernel(StandardEpCompactDispatchArgs args) {
  StandardEpCompactDispatchIntraNodeKernel_body<T>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                         StandardEpCompactCombineIntraNodeKernel                                */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__device__ void StandardEpCompactCombineIntraNodeKernel_body(StandardEpCompactCombineArgs args) {
  const EpDispatchCombineConfig& config = args.config;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = thdId / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;
  const int myPe = config.rank;
  const int npes = config.worldSize;
  const size_t hiddenDim = config.HiddenDimSz();

  const T* inputRows = reinterpret_cast<const T*>(args.expertMajorRows);
  T* outputRows = reinterpret_cast<T*>(args.sourceRankRows);
  T* localStageRows = args.intraNodeTokBufs.combineInp->template GetAs<T*>();
  index_t* localStageFlat = args.shmemOutIndicesMemObj->template GetAs<index_t*>();
  const bool emitFlatPositions =
      args.expertMajorFlatPositions != nullptr && args.sourceRankFlatPositions != nullptr;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineStageRows);

  const int64_t localExpertNum =
      (args.recvCountsRankMajor != nullptr && args.numSegments > 0 &&
       args.numSegments % npes == 0)
          ? args.numSegments / npes
          : 0;
  if (localExpertNum > 0) {
    int64_t selfOutputPrefix = 0;
    for (int prevSrcPe = 0; prevSrcPe < myPe; ++prevSrcPe) {
      selfOutputPrefix += args.outputSplits[prevSrcPe];
    }

    if (globalWarpId == 0) {
      for (int destPe = laneId; destPe < npes; destPe += warpSize) {
        int64_t rankCount = 0;
        for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
          rankCount += args.recvCountsRankMajor[destPe * localExpertNum + localExpert];
        }
        assert(rankCount == args.inputSplits[destPe] &&
               "standard-EP compact combine recv-count metadata does not match input split");
      }
    }

    int64_t expertMajorBase = 0;
    for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
      int64_t expertSrcPrefix = 0;
      for (int destPe = 0; destPe < npes; ++destPe) {
        const int64_t count = args.recvCountsRankMajor[destPe * localExpertNum + localExpert];
        assert(count >= 0);
        assert(count <= args.inputSplits[destPe] &&
               "standard-EP compact combine recv-count metadata exceeds input split");

        int64_t rankMajorSlotPrefix = 0;
        for (int64_t prevExpert = 0; prevExpert < localExpert; ++prevExpert) {
          rankMajorSlotPrefix += args.recvCountsRankMajor[destPe * localExpertNum + prevExpert];
        }
        assert(rankMajorSlotPrefix + count <= args.inputSplits[destPe] &&
               "standard-EP compact combine recv-count metadata exceeds source-rank split");

        if (count == 0) {
          expertSrcPrefix += count;
          continue;
        }
        if (count >= globalWarpNum) {
          for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
            const int64_t expertRow = expertMajorBase + expertSrcPrefix + slot;
            const int64_t rankMajorSlot = rankMajorSlotPrefix + slot;
            assert(expertRow < args.numInputRows);
            if (destPe == myPe) {
	              const int64_t dstRow = selfOutputPrefix + rankMajorSlot;
	              assert(dstRow < args.numOutputRows);
	              core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim,
	                             inputRows + static_cast<size_t>(expertRow) * hiddenDim, hiddenDim);
	              StandardEpCompactWeightedAccumulateTokenOutput(
	                  args, inputRows + static_cast<size_t>(expertRow) * hiddenDim,
	                  emitFlatPositions ? args.expertMajorFlatPositions[expertRow] : -1, 0,
	                  hiddenDim, laneId);
	              if (emitFlatPositions && laneId == 0) {
	                args.sourceRankFlatPositions[dstRow] =
	                    static_cast<int64_t>(args.expertMajorFlatPositions[expertRow]);
              }
            } else {
              const int64_t stageSlot =
                  SendBufSlotOffset(config, destPe, static_cast<int>(rankMajorSlot));
              core::WarpCopy(localStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                             inputRows + static_cast<size_t>(expertRow) * hiddenDim, hiddenDim);
              if (emitFlatPositions && laneId == 0) {
                localStageFlat[stageSlot] =
                    static_cast<index_t>(args.expertMajorFlatPositions[expertRow]);
              }
            }
          }
        } else {
          MultiWarpIter mwIter(globalWarpNum, static_cast<int>(count), hiddenDim);
          const int64_t copyWorkItems =
              static_cast<int64_t>(count) * static_cast<int64_t>(mwIter.warpsPerItem);
          for (int64_t work = globalWarpId; work < copyWorkItems; work += globalWarpNum) {
            int itemId = 0;
            int inItemPartId = 0;
            size_t dimOffset = 0;
            size_t dimChunk = 0;
            mwIter.Decode(static_cast<int>(work), itemId, inItemPartId, dimOffset, dimChunk);
            if (dimChunk == 0) {
              continue;
            }
            const int64_t slot = static_cast<int64_t>(itemId);
            const int64_t expertRow = expertMajorBase + expertSrcPrefix + slot;
            const int64_t rankMajorSlot = rankMajorSlotPrefix + slot;
            assert(expertRow < args.numInputRows);
            if (destPe == myPe) {
              const int64_t dstRow = selfOutputPrefix + rankMajorSlot;
              assert(dstRow < args.numOutputRows);
	              core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim + dimOffset,
	                             inputRows + static_cast<size_t>(expertRow) * hiddenDim + dimOffset,
	                             dimChunk);
	              StandardEpCompactWeightedAccumulateTokenOutput(
	                  args, inputRows + static_cast<size_t>(expertRow) * hiddenDim,
	                  emitFlatPositions ? args.expertMajorFlatPositions[expertRow] : -1,
	                  dimOffset, dimChunk, laneId);
	              if (emitFlatPositions && inItemPartId == 0 && laneId == 0) {
	                args.sourceRankFlatPositions[dstRow] =
	                    static_cast<int64_t>(args.expertMajorFlatPositions[expertRow]);
              }
            } else {
              const int64_t stageSlot =
                  SendBufSlotOffset(config, destPe, static_cast<int>(rankMajorSlot));
              core::WarpCopy(localStageRows + static_cast<size_t>(stageSlot) * hiddenDim + dimOffset,
                             inputRows + static_cast<size_t>(expertRow) * hiddenDim + dimOffset,
                             dimChunk);
              if (emitFlatPositions && inItemPartId == 0 && laneId == 0) {
                localStageFlat[stageSlot] =
                    static_cast<index_t>(args.expertMajorFlatPositions[expertRow]);
              }
            }
          }
        }
        expertSrcPrefix += count;
      }
      expertMajorBase += expertSrcPrefix;
    }
  } else if (args.expertMajorToRankMajorIndices != nullptr) {
    int64_t selfOutputPrefix = 0;
    for (int prevSrcPe = 0; prevSrcPe < myPe; ++prevSrcPe) {
      selfOutputPrefix += args.outputSplits[prevSrcPe];
    }
    for (int64_t expertRow = globalWarpId; expertRow < args.numInputRows;
         expertRow += globalWarpNum) {
      const int64_t rankMajorRow = args.expertMajorToRankMajorIndices[expertRow];
      assert(rankMajorRow >= 0 && rankMajorRow < args.numInputRows);
      int destPe = 0;
      int64_t slot = rankMajorRow;
      for (; destPe < npes; ++destPe) {
        const int64_t count = args.inputSplits[destPe];
        if (slot < count) {
          break;
        }
        slot -= count;
      }
      assert(destPe < npes);
      if (destPe == myPe) {
	        const int64_t dstRow = selfOutputPrefix + slot;
	        assert(dstRow < args.numOutputRows);
	        core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim,
	                       inputRows + static_cast<size_t>(expertRow) * hiddenDim, hiddenDim);
	        StandardEpCompactWeightedAccumulateTokenOutput(
	            args, inputRows + static_cast<size_t>(expertRow) * hiddenDim,
	            emitFlatPositions ? args.expertMajorFlatPositions[expertRow] : -1, 0,
	            hiddenDim, laneId);
	        if (emitFlatPositions && laneId == 0) {
	          args.sourceRankFlatPositions[dstRow] =
	              static_cast<int64_t>(args.expertMajorFlatPositions[expertRow]);
        }
        continue;
      }
      assert(args.inputSplits[destPe] <= config.MaxNumTokensToSendPerRank() &&
             "standard-EP compact combine remote input split exceeds MORI staging stride");
      const int64_t stageSlot = SendBufSlotOffset(config, destPe, static_cast<int>(slot));
      core::WarpCopy(localStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                     inputRows + static_cast<size_t>(expertRow) * hiddenDim, hiddenDim);
      if (emitFlatPositions && laneId == 0) {
        localStageFlat[stageSlot] =
            static_cast<index_t>(args.expertMajorFlatPositions[expertRow]);
      }
    }
  } else {
    int64_t inputPrefix = 0;
    int64_t selfOutputPrefix = 0;
    for (int prevSrcPe = 0; prevSrcPe < myPe; ++prevSrcPe) {
      selfOutputPrefix += args.outputSplits[prevSrcPe];
    }
    for (int destPe = 0; destPe < npes; ++destPe) {
      const int64_t count = args.inputSplits[destPe];
      if (destPe == myPe) {
        for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
          const int64_t srcRow = inputPrefix + slot;
	          const int64_t dstRow = selfOutputPrefix + slot;
	          assert(srcRow < args.numInputRows);
	          assert(dstRow < args.numOutputRows);
	          core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim,
	                         inputRows + static_cast<size_t>(srcRow) * hiddenDim, hiddenDim);
	          StandardEpCompactWeightedAccumulateTokenOutput(
	              args, inputRows + static_cast<size_t>(srcRow) * hiddenDim,
	              emitFlatPositions ? args.expertMajorFlatPositions[srcRow] : -1, 0,
	              hiddenDim, laneId);
	          if (emitFlatPositions && laneId == 0) {
	            args.sourceRankFlatPositions[dstRow] =
	                static_cast<int64_t>(args.expertMajorFlatPositions[srcRow]);
          }
        }
        inputPrefix += count;
        continue;
      }
      assert(count <= config.MaxNumTokensToSendPerRank() &&
             "standard-EP compact combine remote input split exceeds MORI staging stride");
      for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
        const int64_t srcRow = inputPrefix + slot;
        const int64_t stageSlot = SendBufSlotOffset(config, destPe, static_cast<int>(slot));
        assert(srcRow < args.numInputRows);
        core::WarpCopy(localStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
                       inputRows + static_cast<size_t>(srcRow) * hiddenDim, hiddenDim);
        if (emitFlatPositions && laneId == 0) {
          localStageFlat[stageSlot] =
              static_cast<index_t>(args.expertMajorFlatPositions[srcRow]);
        }
      }
      inputPrefix += count;
    }
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineGridWaitSend);
  StandardEpCompactGridWait(args.dispatchGridBarrier);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineSignalCounts);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      const index_t sendCount = static_cast<index_t>(args.inputSplits[destPe]);
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      core::AtomicStoreRelaxedSystem(signal, sendCount + 1);
    }
  }

  __syncthreads();

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineWaitCounts);
  if (globalWarpId == 0) {
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      const index_t expected = static_cast<index_t>(args.outputSplits[srcPe]);
      const index_t observed = shmem::ShmemInt32WaitUntilGreaterThan(recvTokenNums + srcPe, 0) - 1;
      assert(observed == expected &&
             "standard-EP compact combine split mismatch against MORI signal");
      (void)observed;
    }
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineGridWaitRecv);
  StandardEpCompactGridWait(args.combineGridBarrier);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineReceiveCopy);
  int64_t outputPrefix = 0;
  for (int srcPe = 0; srcPe < npes; ++srcPe) {
    const int64_t count = args.outputSplits[srcPe];
    const T* peerStageRows = args.intraNodeTokBufs.combineInp->template GetAs<T*>(srcPe);
    const index_t* peerStageFlat = args.shmemOutIndicesMemObj->template GetAs<index_t*>(srcPe);
    if (srcPe == myPe) {
      outputPrefix += count;
      continue;
    }
    assert(count <= config.MaxNumTokensToSendPerRank() &&
           "standard-EP compact combine remote output split exceeds MORI staging stride");
    if (count >= globalWarpNum) {
      for (int64_t slot = globalWarpId; slot < count; slot += globalWarpNum) {
        const int64_t dstRow = outputPrefix + slot;
	        const int64_t stageSlot = SendBufSlotOffset(config, myPe, static_cast<int>(slot));
	        assert(dstRow < args.numOutputRows);
	        core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim,
	                       peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim, hiddenDim);
	        StandardEpCompactWeightedAccumulateTokenOutput(
	            args, peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
	            emitFlatPositions ? static_cast<int64_t>(peerStageFlat[stageSlot]) : -1, 0,
	            hiddenDim, laneId);
	        if (emitFlatPositions && laneId == 0) {
	          args.sourceRankFlatPositions[dstRow] = static_cast<int64_t>(peerStageFlat[stageSlot]);
	        }
      }
    } else {
      MultiWarpIter mwIter(globalWarpNum, static_cast<int>(count), hiddenDim);
      const int64_t copyWorkItems =
          static_cast<int64_t>(count) * static_cast<int64_t>(mwIter.warpsPerItem);
      for (int64_t work = globalWarpId; work < copyWorkItems; work += globalWarpNum) {
        int itemId = 0;
        int inItemPartId = 0;
        size_t dimOffset = 0;
        size_t dimChunk = 0;
        mwIter.Decode(static_cast<int>(work), itemId, inItemPartId, dimOffset, dimChunk);
        if (dimChunk == 0) {
          continue;
        }
        const int64_t slot = static_cast<int64_t>(itemId);
        const int64_t dstRow = outputPrefix + slot;
        const int64_t stageSlot = SendBufSlotOffset(config, myPe, static_cast<int>(slot));
	        assert(dstRow < args.numOutputRows);
	        core::WarpCopy(outputRows + static_cast<size_t>(dstRow) * hiddenDim + dimOffset,
	                       peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim + dimOffset,
	                       dimChunk);
	        StandardEpCompactWeightedAccumulateTokenOutput(
	            args, peerStageRows + static_cast<size_t>(stageSlot) * hiddenDim,
	            emitFlatPositions ? static_cast<int64_t>(peerStageFlat[stageSlot]) : -1,
	            dimOffset, dimChunk, laneId);
	        if (emitFlatPositions && inItemPartId == 0 && laneId == 0) {
	          args.sourceRankFlatPositions[dstRow] = static_cast<int64_t>(peerStageFlat[stageSlot]);
	        }
      }
    }
    outputPrefix += count;
  }

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineGridWaitReset);
  StandardEpCompactGridWait(args.dispatchGridBarrier + 1);

  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineResetCounters);
  if (globalWarpId == 0) {
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
      core::AtomicStoreRelaxedSystem(recvTokenNums + srcPe, 0);
    }
  }

  __syncthreads();
  __threadfence_system();

  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
    }
  }

  if (thdId == 0 && args.totalRecvTokenNum != nullptr) {
    *args.totalRecvTokenNum = static_cast<index_t>(args.numOutputRows);
  }
}

template <typename T>
__global__ void StandardEpCompactCombineIntraNodeKernel(StandardEpCompactCombineArgs args) {
  StandardEpCompactCombineIntraNodeKernel_body<T>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                 StandardEpCompactWeightedOutputBackwardIntraNodeKernel                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
__device__ void StandardEpCompactWeightedOutputBackwardIntraNodeKernel_body(
    StandardEpCompactWeightedOutputBackwardArgs args) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int warpId = threadIdx.x / warpSize;
  const int warpNum = blockDim.x / warpSize;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;
  const size_t hiddenDim =
      args.hiddenDim > 0 ? static_cast<size_t>(args.hiddenDim) : args.config.HiddenDimSz();

  const T* sourceRows = reinterpret_cast<const T*>(args.sourceRankRows);
  const T* gradTokenOutput = reinterpret_cast<const T*>(args.gradTokenOutput);
  T* gradSourceRows = reinterpret_cast<T*>(args.gradSourceRankRows);

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::StandardCompactCombineReceiveCopy);

  for (int64_t row = globalWarpId; row < args.numRows; row += globalWarpNum) {
    const int64_t localFlat = args.sourceRankFlatPositions[row] - args.flatPositionOffset;
    const bool validFlat =
        localFlat >= 0 && localFlat < args.topScoresFlatSize && args.topK > 0;
    const int64_t tokenRow = validFlat ? (localFlat / args.topK) : 0;
    const bool validToken = validFlat && tokenRow >= 0 && tokenRow < args.tokenOutputRows;
    const float score = validToken ? args.topScoresFlat[localFlat] : 0.0f;
    float dot = 0.0f;

    for (size_t d = laneId; d < hiddenDim; d += warpSize) {
      const size_t offset = static_cast<size_t>(row) * hiddenDim + d;
      const float gradValue =
          validToken
              ? StandardEpCompactToFloat(
                    gradTokenOutput[static_cast<size_t>(tokenRow) * hiddenDim + d])
              : 0.0f;
      const float srcValue = validToken ? StandardEpCompactToFloat(sourceRows[offset]) : 0.0f;
      gradSourceRows[offset] = StandardEpCompactFromFloat<T>(gradValue * score);
      dot += gradValue * srcValue;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      dot += __shfl_down(dot, offset);
    }
    if (validToken && laneId == 0) {
      core::AtomicAddRelaxed(args.gradTopScoresFlat + localFlat, dot);
    }
  }
}

template <typename T>
__global__ void StandardEpCompactWeightedOutputBackwardIntraNodeKernel(
    StandardEpCompactWeightedOutputBackwardArgs args) {
  StandardEpCompactWeightedOutputBackwardIntraNodeKernel_body<T>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0>
__device__ __forceinline__ void EpCombineIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  using TokT =
      std::conditional_t<UseFp8DirectCast || UseFp8BlockwiseQuant, core::CombineInternalFp8, T>;
  static_assert(!(UseFp8DirectCast && UseFp8BlockwiseQuant),
                "Fp8 direct cast and blockwise quant are mutually exclusive");
  static_assert((!UseFp8DirectCast && !UseFp8BlockwiseQuant) || std::is_same_v<T, hip_bfloat16>,
                "Fp8 combine quant currently only supports bf16 input");
  static_assert((Vec8Top8BlockElems & (Vec8Top8BlockElems - 1)) == 0,
                "Vec8Top8BlockElems must be 0 or a power of two");
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::CombineStageInput);

  const uint64_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
  // When TokT != T (e.g. fp8 combine), staging layout uses TokT-sized tokens
  const size_t hiddenDim = config.HiddenDimSz();
  const size_t hiddenBytes = hiddenDim * sizeof(TokT);
  const size_t weightBytes =
      (UseWeights && args.weightsBuf != nullptr) ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t scaleBytes =
      UseFp8BlockwiseQuant ? static_cast<size_t>(args.fp8BlockwiseCombineScaleDim) * sizeof(float)
                           : 0;
  const size_t combXferBytes = hiddenBytes + scaleBytes + weightBytes;

  if constexpr (EnableStdMoE) {
#ifdef ENABLE_STANDARD_MOE_ADAPT
    InvokeConvertCombineInput<T, UseP2PRead>(args, myPe);
#endif
  } else if constexpr (UseP2PRead) {
    if (args.config.useExternalInpBuffer) {
      for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
        if constexpr (UseFp8BlockwiseQuant) {
          core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.shmemInpScalesMemObj->template GetAs<float*>() +
                  i * args.fp8BlockwiseCombineScaleDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        } else if constexpr (!std::is_same_v<T, TokT> &&
                             std::is_same_v<TokT, core::CombineInternalFp8>) {
          core::WarpCastBf16ToCombineInternalFp8<T>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, laneId);
        } else {
          core::WarpCopy(args.intraNodeTokBufs.combineInp->template GetAs<T*>() + i * hiddenDim,
                         args.inpTokenBuf + i * hiddenDim, hiddenDim);
        }
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
          core::WarpCopy(
              args.shmemInpWeightsMemObj->template GetAs<float*>() + i * config.numExpertPerToken,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken);
        }
      }
    }
  } else {
#ifdef ENABLE_PROFILER
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
          index_t destTokId =
              args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
          index_t destPe = PeFromFlatTokenIndex(config, destTokId);
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          uint8_t* destStagingPtr =
              args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
              SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#else
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe)[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
      if constexpr (UseWeights) {
        if (args.weightsBuf) {
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#endif
  }

  // Make sure copy on all GPUs are finished
  MORI_TRACE_NEXT(seq, Slot::CombineBarrier);
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
  *args.totalRecvTokenNum = 0;
  if (args.curRankNumToken == 0) return;

  MORI_TRACE_NEXT(seq, Slot::CombineAccumSetup);
  extern __shared__ char sharedMem[];
  // Layout: [srcPtrs] [srcWeightsPtr if UseWeights] [srcScalePtrs if UseFp8BlockwiseQuant];
  // host-side combine_shared_mem() must use the same flags.
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = nullptr;
  if constexpr (UseWeights) {
    srcWeightsPtr = reinterpret_cast<float**>(sharedMem) + warpNum * config.numExpertPerToken +
                    warpId * config.numExpertPerToken;
  }
  float** srcScalePtrs = nullptr;
  if constexpr (UseFp8BlockwiseQuant) {
    constexpr int scalePtrArrayOffset = UseWeights ? 2 : 1;
    srcScalePtrs = reinterpret_cast<float**>(sharedMem) +
                   scalePtrArrayOffset * warpNum * config.numExpertPerToken +
                   warpId * config.numExpertPerToken;
  }

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    // Prepare data pointers on different GPUs
    MORI_TRACE_NEXT(seq, Slot::CombinePreparePtrs);
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);

      if (destPe < config.worldSize) {
        if constexpr (UseP2PRead) {
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          srcPtrs[j] = args.intraNodeTokBufs.combineInp->template GetAs<TokT*>(destPe) +
                       destLocalTokId * hiddenDim + hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                               destLocalTokId * config.numExpertPerToken;
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = args.shmemInpScalesMemObj->template GetAs<float*>(destPe) +
                              destLocalTokId * args.fp8BlockwiseCombineScaleDim;
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        } else {
          srcPtrs[j] = reinterpret_cast<TokT*>(
                           args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                           SendBufSlotOffset(config, destPe, tokenId) * combXferBytes) +
                       hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes +
                scaleBytes);
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes);
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (UseWeights) {
          srcWeightsPtr[j] = nullptr;
        }
        if constexpr (UseFp8BlockwiseQuant) {
          srcScalePtrs[j] = nullptr;
        }
      }
    }

    T* outPtr = args.intraNodeTokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
                hiddenDimOffset;

    int validAccumCount = config.numExpertPerToken;
    if (config.worldSize <= 4) {
      {
        int isValid = 0;
        TokT* myTokPtr = nullptr;
        float* myScalePtr = nullptr;
        if (laneId < config.numExpertPerToken) {
          myTokPtr = srcPtrs[laneId];
          if constexpr (UseFp8BlockwiseQuant) {
            myScalePtr = srcScalePtrs[laneId];
          }
          isValid = (myTokPtr != nullptr) ? 1 : 0;
        }
        unsigned long long validMask = __ballot(isValid);
        validAccumCount = __popcll(validMask);
        if (validAccumCount < config.numExpertPerToken && isValid) {
          int myPos = __popcll(validMask & ((1ULL << laneId) - 1));
          srcPtrs[myPos] = myTokPtr;
          if constexpr (UseFp8BlockwiseQuant) {
            srcScalePtrs[myPos] = myScalePtr;
          }
        }
      }
    }

    if constexpr (UseFp8BlockwiseQuant) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      if constexpr (Vec8Top8BlockElems != 0) {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFullBlockVec8Top8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDim);
        } else if ((hiddenDimOffset & 0x7) == 0 && (hiddenDimSize & 0x7) == 0) {
          core::WarpAccumFp8DequantSegmentBlockVec8Top8<T, core::CombineInternalFp8,
                                                        Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        } else {
          // Misaligned segment: vec8 helper would fault on the load. Tiny scalar fallback.
          core::WarpAccumFp8DequantSegmentScalarTop8<T, core::CombineInternalFp8,
                                                     Vec8Top8BlockElems>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        }
      } else {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumFp8DequantFull<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDim,
              args.fp8BlockwiseCombineScaleDim);
        } else {
          core::WarpAccumFp8DequantSegment<T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDimOffset,
              hiddenDimSize, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        }
      }
    } else if constexpr (!std::is_same_v<T, TokT> &&
                         std::is_same_v<TokT, core::CombineInternalFp8>) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccumCombineInternalFp8ToBf16(outPtr, reinterpret_cast<const TokT* const*>(srcPtrs),
                                              validAccumCount, laneId, hiddenDimSize);
    } else {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccum<T, 4>(outPtr, srcPtrs, nullptr, validAccumCount, hiddenDimSize);
    }

    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineAccumWeights);
      if (args.weightsBuf && inTokenPartId == mwIter.warpsPerItem - 1) {
        core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                      tokenId * config.numExpertPerToken,
                                  srcWeightsPtr, nullptr, config.numExpertPerToken,
                                  config.numExpertPerToken);
      }
    }
  }
}

template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems>(args);
}

}  // namespace moe
}  // namespace mori
