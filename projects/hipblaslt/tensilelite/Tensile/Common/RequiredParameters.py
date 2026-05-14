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
    """Solution parameters that must contribute to the *kernel-name-min* hash.

    A parameter belongs in this set iff two solutions that differ only in this
    parameter compile to *different* GPU assembly. The kernel filename is
    derived from this set (see SolutionStructs/Naming.py::getKernelNameMin),
    so any codegen-affecting parameter that is missing from this set will
    cause two solutions with different `.s`/`.co` outputs to share a single
    on-disk binary. The runtime predicate dispatcher will then load the wrong
    binary for the selected solution and produce a "Memory access fault by
    GPU" segfault. See also: getParamsNotAffectingKernelName().

    Regression history:
      - PR #3679 ("codegen: handle unaligned K for TN") added BAddrInterleave
        and KRingShift to validParameters but forgot to add them here. PR #7349
        ([AIHPBLAS-3494]) restored both entries after a rare TN bf16 segfault
        was traced back to this collision.
    """
    return frozenset({
        '1LDSBuffer',
        'ActivationFuncCall',
        'AdaptiveGemm',
        'AssertFree0ElementMultiple',
        'AssertFree1ElementMultiple',
        'AssertSummationElementMultiple',
        # BAddrInterleave / KRingShift change tail-loop SGPR setup, address
        # interleave, and ring-shift control flow. Two solutions differing
        # only in these flags MUST get distinct kernel filenames.
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
    """Solution parameters in `validParameters` that legitimately do NOT
    contribute to the kernel-name-min hash.

    A parameter belongs in this set iff two solutions that differ only in
    this parameter compile to *byte-identical* GPU assembly. Such parameters
    fall into a few categories:

      1. Internal dispatch args (mirrors SolutionStructs/Naming.py::_INTERNAL_ARGS).
         Runtime-dispatch-only knobs that are masked out of the kernel name.
      2. Special-cased in `_getName` (emitted directly, not via the for-loop
         over `requiredParametersTemp`): MacroTile family, MatrixInstruction
         family, ThreadTile, UseCustomMainLoopSchedule's CMS tag,
         CustomKernelName, problem-type fields.
      3. Host-side problem-predicate values (encode runtime restrictions,
         not codegen).
      4. Grandfathered (currently in validParameters but not classified).
         Each entry here is a *candidate* for promotion to
         getRequiredParametersMin() if subsequent audit shows it changes the
         generated assembly. New parameters MUST NOT be added here without
         explicit reasoning.

    The unit test in `Tensile/Tests/unit/test_RequiredParameters.py` enforces
    that `validParameters.keys()` is partitioned exactly between this set and
    getRequiredParametersMin(). New parameters added to `validParameters` must
    be classified into one or the other, which is the safeguard against the
    regression mode in PR #3679.
    """
    return frozenset({
        # ---- Category 1: internal dispatch args ----
        # Must mirror SolutionStructs/Naming.py::_INTERNAL_ARGS. The unit
        # test asserts the two stay in sync.
        "WorkGroupMapping",
        "WorkGroupMappingXCC",
        "WorkGroupMappingXCCGroup",
        "StaggerU",
        "StaggerUStride",
        "StaggerUMapping",
        "GlobalSplitUCoalesced",
        "GlobalSplitUWorkGroupMappingRoundRobin",
        "SFCWGM",
        # GlobalSplitU is handled specially via the `splitGSU` branch in
        # `_getName`; positive / -1 values are masked to "M" in the kernel
        # name and re-emitted as a runtime arg.
        "GlobalSplitU",

        # ---- Category 2: emitted directly by `_getName` ----
        "MacroTile",                  # MT{MT0}x{MT1}x{DU} prefix
        "DepthU",                     # MT{MT0}x{MT1}x{DU} prefix
        "MatrixInstruction",          # MI{M}x{N}x{B} prefix
        "ThreadTile",                 # added to set when no MatrixInstruction
        "UseCustomMainLoopSchedule",  # 'CMS' tag when True
        "CustomKernelName",           # returned directly (skips _getName)

        # ---- Category 3: host-side problem-predicate values ----
        # These encode runtime restrictions consumed by predicate dispatch
        # (see ContractionProblemPredicates.hpp); they do not change asm.
        "AssertAIGreaterThanEqual",
        "AssertAILessThanEqual",
        "AssertFree1DivByMT1LowbitGT1",
        "AssertKRingShiftTailWrapOnly",

        # ---- Category 4: grandfathered (audit candidate) ----
        # These existed in `validParameters` prior to this guard being
        # introduced and are not currently in getRequiredParametersMin().
        # Each is a candidate for promotion if it is found to alter the
        # generated assembly. Do NOT add new parameters here without proof
        # that two solutions differing only in this param produce
        # byte-identical .co files.
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
    """Return a list of human-readable error strings, empty iff every entry
    in `validParameters` is classified into exactly one of
    `getRequiredParametersMin()` or `getParamsNotAffectingKernelName()`.

    This invariant is what prevents the PR #3679 regression mode: a new
    parameter added to `validParameters` that materially changes the kernel
    assembly but is missing from `RequiredParametersMin`. When that happens,
    two distinct solutions can hash to the same kernel filename, the second
    compile overwrites the first, and the runtime predicate dispatcher loads
    the wrong `.co` binary -> "Memory access fault by GPU".
    """
    valid = set(validParameters.keys())
    inMin = set(getRequiredParametersMin())
    notInName = set(getParamsNotAffectingKernelName())

    errors = []

    unclassified = valid - inMin - notInName
    if unclassified:
        errors.append(
            "The following solution parameters are in validParameters but are "
            "NOT classified for kernel-name-min handling. Each must be added "
            "EITHER to getRequiredParametersMin() (if it affects the generated "
            "kernel assembly, so kernels with different values get different "
            "kernel filenames / .co files) OR to "
            "getParamsNotAffectingKernelName() (if it is a runtime/dispatch "
            "parameter and kernels with different values produce byte-identical "
            "assembly):\n  - " + "\n  - ".join(sorted(unclassified)) +
            "\n\nSee Common/RequiredParameters.py for the rationale, and PR "
            "#3679 / PR #7349 for the regression mode this guard protects "
            "against."
        )

    inBoth = inMin & notInName
    if inBoth:
        errors.append(
            "The following solution parameters are in BOTH "
            "getRequiredParametersMin() and getParamsNotAffectingKernelName(); "
            "each must be in exactly one:\n  - " + "\n  - ".join(sorted(inBoth))
        )

    return errors
