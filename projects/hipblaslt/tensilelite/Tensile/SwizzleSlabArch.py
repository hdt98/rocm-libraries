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
        Solution state during tuning, or finalized kernel dict; must have:
        ProblemType, MIInputPerThread{tc}, DataType{tc}.
    tc :
        A or B.
    isa :
        kernel['ISA'] / state['ISA'] (IsaVersion or 3-tuple).

    Returns
    -------
    int
        RDNA4 + Half/BF16 + SwizzleTensorA on A → 1 (RDNA4 WMMA slab).

        Otherwise CDNA3 formula: 16 // MIInputPerThread // elementBytes.
    """
    pt = state_or_kernel["ProblemType"]
    mi_pt = state_or_kernel[f"MIInputPerThread{tc}"]
    dt = pt[f"DataType{tc}"]
    bpe = dt.numBytes()

    # TODO: generalize this for rdna4 and support for tensor B
    if is_rdna4_swizzle_slab_isa(isa) and (dt.isHalf() or dt.isBFloat16()):
        if tc == "A" and pt.get("SwizzleTensorA", False):
            return 1

    return 16 // mi_pt // bpe


def cal_swizzle_lane_size_elements(state_or_kernel: Mapping[str, Any], tc: str, isa: Any) -> int:
    """
    Elements along one swizzle lane for graUnrollOffsets / F(row,col).

    Legacy formula: (MatrixInstK // 4) * swizzlePackK (same kPack as stride).

    RDNA4 FP16/BF16 SwizzleTensorA: host slab uses MiKv*PackK = 8 half per lane strip
    (see client reshape); global loads use 8 coalesced half per lane (128 bytes).
    With swizzlePackK == 1 the stride formula alone would give lane size 4, which
    does not match the pre-shuffle layout — force 8.
    """
    mik = int(state_or_kernel["MatrixInstK"])
    pk = cal_swizzle_pack_k_tensor(state_or_kernel, tc, isa)
    pt = state_or_kernel["ProblemType"]
    dt = pt[f"DataType{tc}"]
    if (
        is_rdna4_swizzle_slab_isa(isa)
        and tc == "A"
        and pt.get("SwizzleTensorA", False)
        and (dt.isHalf() or dt.isBFloat16())
    ):
        return 8
    return (mik // 4) * pk
