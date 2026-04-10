################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
from __future__ import annotations

from typing import Any, Mapping


def is_rdna4_swizzle_slab_isa(isa: Any) -> bool:
    """True for gfx1200 / gfx1201 (IsaVersion 12,0,0 and 12,0,1)."""
    isaVersion = tuple(isa)
    return isaVersion == (12, 0, 0) or isaVersion == (12, 0, 1)


def cal_swizzle_pack_k_tensor(state_or_kernel: Mapping[str, Any], tc: str, isa: Any) -> int:
    """
    Effective PackK for swizzled global loads (128 bytes).

    Parameters
    ----------
    state_or_kernel :
        Solution 'state' during tuning, or finalized 'kernel' dict; must have
        'ProblemType', 'MIInputPerThread{tc}', 'DataType{tc}'.
    tc :
        'A' or 'B'.
    isa :
        'kernel['ISA']' / 'state['ISA']' ('IsaVersion' or 3-tuple).

    Returns
    -------
    int
        **RDNA4** + ``SwizzleTensorA`` on **A** → ``16 // MIInputPerThread // bpe``
        (Half/BF16 → 1, FP8/Int8 → 2).

        Otherwise **CDNA3** formula: ``16 // MIInputPerThread // elementBytes``.
    """
    pt = state_or_kernel["ProblemType"]
    mi_pt = state_or_kernel[f"MIInputPerThread{tc}"]
    dt = pt[f"DataType{tc}"]
    bpe = dt.numBytes()

    # RDNA4 WMMA slab: explicit path for supported types on tensor A.
    # FP16/BF16 → PackK=1 (two b64 → one b128 via 8-element lanes)
    # FP8/Int8  → PackK=2 (single b128 via 16-element lanes)
    # The values match the generic formula below but are kept explicit for clarity.
    if is_rdna4_swizzle_slab_isa(isa) and (dt.isHalf() or dt.isBFloat16() or bpe == 1):
        if tc == "A" and pt.get("SwizzleTensorA", False):
            return 16 // mi_pt // bpe

    return 16 // mi_pt // bpe


def cal_swizzle_lane_size_elements(state_or_kernel: Mapping[str, Any], tc: str, isa: Any) -> int:
    """
    Elements along one swizzle **lane** for 'graUnrollOffsets' / 'F(row,col)'.

    Legacy formula: '(MatrixInstK // 4) * swizzlePackK' (same 'kPack' as stride).

    RDNA4 SwizzleTensorA: each lane loads exactly 128 bits (16 bytes) contiguously.
    The element count per lane is therefore ``16 // bpe``:
    - FP16/BF16 (bpe=2) → 8 elements per lane
    - FP8/Int8  (bpe=1) → 16 elements per lane
    The generic formula would give incorrect values for RDNA4 because MatrixInstK
    differs from CDNA3 (16 vs 32 for FP8).
    """
    mik = int(state_or_kernel["MatrixInstK"])
    pk = cal_swizzle_pack_k_tensor(state_or_kernel, tc, isa)
    pt = state_or_kernel["ProblemType"]
    dt = pt[f"DataType{tc}"]
    bpe = dt.numBytes()
    if (
        is_rdna4_swizzle_slab_isa(isa)
        and tc == "A"
        and pt.get("SwizzleTensorA", False)
        and (dt.isHalf() or dt.isBFloat16() or bpe == 1)
    ):
        return 16 // bpe
    return (mik // 4) * pk