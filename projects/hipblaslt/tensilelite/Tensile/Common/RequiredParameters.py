################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the
# Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from functools import lru_cache
from .ValidParameters import validParameters


@lru_cache
def getRequiredParametersFull() -> set:
    return frozenset(validParameters.keys())


@lru_cache
def getRequiredParametersMin() -> set:
    """Parameters that contribute to the kernel-name-min hash.

    A parameter belongs here iff two solutions differing only in it compile
    to different GPU assembly. See also getParamsNotAffectingKernelName().
    """
    return frozenset({
        '1LDSBuffer',
        'ActivationFuncCall',
        'AdaptiveGemm',
        'AssertFree0ElementMultiple',
        'AssertFree1ElementMultiple',
        'AssertSummationElementMultiple',
        'BAddrInterleave',
        'KRingShift',
        'ClusterLocalRead',
        'ConvertAfterDS',
        'DirectToVgprA',
        'DirectToVgprB',
        'DirectToVgprSparseMetadata',
        'DirectToLdsA',
        'DirectToLdsB',
        'DirectToVgprMXSA',
        'DirectToVgprMXSB',
        'ExpandPointerSwap',
        'ExtraLatencyForLR',
        'ExtraMiLatencyLeft',
        'ForceDisableShadowInit',
        'GlobalReadPerMfma',
        'GlobalReadVectorWidthA',
        'GlobalReadVectorWidthB',
        'GlobalSplitUAlgorithm',
        'GroupLoadStore',
        'ISA',
        'InnerUnroll',
        'Kernel',
        'LdsBlockSizePerPadA',
        'LdsBlockSizePerPadB',
        'LdsBlockSizePerPadMetadata',
        'LdsPadA',
        'LdsPadB',
        'LdsPadMetadata',
        'LDSTrInst',
        'LocalReadVectorWidth',
        'LocalWritePerMfma',
        'MIArchVgpr',
        'MaxOccupancy',
        'NonTemporal',
        'NonTemporalA',
        'NonTemporalB',
        'NonTemporalC',
        'NonTemporalD',
        'NonTemporalMetadata',
        'NumElementsPerBatchStore',
        'NumLoadsCoalescedA',
        'NumLoadsCoalescedB',
        'OptNoLoadLoop',
        'PrefetchGlobalRead',
        'PrefetchLocalRead',
        'PreloadKernArgs',
        'ScheduleIterAlg',
        'ScheduleGROverBarrier',
        'SourceSwap',
        'SpaceFillingAlgo',
        'StorePriorityOpt',
        'StoreRemapVectorWidth',
        'StoreSyncOpt',
        'StoreVectorWidth',
        'StreamK',
        'StreamKXCCMapping',
        'StreamKFixupTreeReduction',
        'SwapGlobalReadOrder',
        'TailloopInNll',
        'TransposeLDS',
        'TransposeLDSMetadata',
        'TDMInst',
        "TDMSplit",
        'UnrollLoopSwapGlobalReadOrder',
        'Use64bShadowLimit',
        'Use64bShadowLimitMX',
        'UseInstOffsetForGRO',
        'UseSgprForGRO',
        'VectorStore',
        'VectorWidthA',
        'VectorWidthB',
        'WaveSeparateGlobalReadA',
        'WaveSeparateGlobalReadB',
        'WavefrontSize',
        'WorkGroup',
        'DtlPlusLdsBuf',
        'MinGRIncPerMfma',
        'UsePLRPack',
        'UseSubtileImpl'
    })


@lru_cache
def getParamsNotAffectingKernelName() -> frozenset:
    """Parameters in validParameters that do NOT contribute to the
    kernel-name-min hash (kernels differing only in these are byte-identical).

    Categories below:
      1. Internal dispatch args (mirrors Naming.py::_INTERNAL_ARGS).
      2. Special-cased in _getName (emitted directly, not via the for-loop).
      3. Host-side problem-predicate values.
      4. Grandfathered: currently unclassified; candidates for promotion to
         getRequiredParametersMin() if found to alter generated assembly.
         New parameters must NOT be added to category 4 without proof that
         differing values produce byte-identical .co files.
    """
    return frozenset({
        # ---- 1. internal dispatch args (mirror Naming.py::_INTERNAL_ARGS) ----
        "WorkGroupMapping",
        "WorkGroupMappingXCC",
        "WorkGroupMappingXCCGroup",
        "StaggerU",
        "StaggerUStride",
        "StaggerUMapping",
        "GlobalSplitUCoalesced",
        "GlobalSplitUWorkGroupMappingRoundRobin",
        "SFCWGM",
        # GlobalSplitU: handled via the splitGSU branch in _getName.
        "GlobalSplitU",

        # ---- 2. emitted directly by _getName ----
        "MacroTile",                  # MT{MT0}x{MT1}x{DU} prefix
        "DepthU",                     # MT{MT0}x{MT1}x{DU} prefix
        "MatrixInstruction",          # MI{M}x{N}x{B} prefix
        "ThreadTile",                 # added when no MatrixInstruction
        "UseCustomMainLoopSchedule",  # 'CMS' tag when True
        "CustomKernelName",           # returned directly

        # ---- 3. host-side problem-predicate values ----
        "AssertAIGreaterThanEqual",
        "AssertAILessThanEqual",
        "AssertFree1DivByMT1LowbitGT1",
        "AssertKRingShiftTailWrapOnly",

        # ---- 4. grandfathered (audit candidates) ----
        "ActivationAlt",
        "ActivationFused",
        "AdaptiveGemmGSUA",
        "BufferLoad",
        "BufferStore",
        "DebugStreamK",
        "DirectToLds",                # legacy; superseded by DirectToLds[A|B]
        "InterleaveAlpha",
        "KernelLanguage",
        "LdsBlockSizePerPadMXSA",
        "LdsBlockSizePerPadMXSB",
        "LdsPadMXSA",
        "LdsPadMXSB",
        "LocalReadVectorWidthA",
        "LocalReadVectorWidthB",
        "MagicDivAlg",
        "MaxLDS",
        "MbskPrefetchMethod",
        "NoReject",
        "NonTemporalE",
        "NonTemporalMXSA",
        "NonTemporalMXSB",
        "NonTemporalWS",
        "ScheduleGlobalRead",
        "ScheduleLocalWrite",
        "Sparse",
        "StreamKAtomic",
        "SuppressNoLoadLoop",
        "WaveSplitK",
        "WorkGroupReduction",
    })


def validateParametersClassification() -> list:
    """Return human-readable error strings; empty iff every key in
    validParameters is in exactly one of getRequiredParametersMin() or
    getParamsNotAffectingKernelName()."""
    valid = set(validParameters.keys())
    inMin = set(getRequiredParametersMin())
    notInName = set(getParamsNotAffectingKernelName())

    errors = []

    unclassified = valid - inMin - notInName
    if unclassified:
        errors.append(
            "These solution parameters are in validParameters but not "
            "classified. Add each EITHER to getRequiredParametersMin() (if "
            "it changes the generated kernel assembly) OR to "
            "getParamsNotAffectingKernelName() (if kernels with different "
            "values are byte-identical):\n  - "
            + "\n  - ".join(sorted(unclassified))
        )

    inBoth = inMin & notInName
    if inBoth:
        errors.append(
            "These parameters are in both getRequiredParametersMin() and "
            "getParamsNotAffectingKernelName(); each must be in exactly "
            "one:\n  - " + "\n  - ".join(sorted(inBoth))
        )

    return errors
