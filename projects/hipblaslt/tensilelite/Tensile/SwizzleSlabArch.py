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


def _is_rdna4_swizzle_a(pt: Mapping[str, Any], tc: str) -> bool:
    return tc == "A" and pt.get("SwizzleTensorA", False)


def cal_swizzle_pack_k_tensor(state_or_kernel: Mapping[str, Any], tc: str, isa: Any) -> int:
    """
    Effective PackK for swizzled global loads.

    RDNA4 SwizzleTensorA (FP16/BF16/FP8/BF8): always PackK=1.
      - FP16/BF16: MI_K=16, GRVW = 8*1 = 8 half = 16B → buffer_load_b128.
      - FP8/BF8:   MI_K=16, GRVW = 8*1 = 8 FP8  = 8B  → buffer_load_b64.
        PackK=2 would merge two K-halves into one load (GRVW=16), but for
        MIWaveTile>1 the kernel splits the 4 loaded VGPRs across M-tiles,
        creating a host-layout mismatch.  PackK=1 keeps each load aligned
        to exactly one WMMA A operand and one M-tile.

    Otherwise falls back to the CDNA3 formula: 16 // MIInputPerThread // bpe.
    """
    pt = state_or_kernel["ProblemType"]
    mi_pt = state_or_kernel[f"MIInputPerThread{tc}"]
    dt = pt[f"DataType{tc}"]
    bpe = dt.numBytes()

    if is_rdna4_swizzle_slab_isa(isa) and _is_rdna4_swizzle_a(pt, tc):
        return 1

    return 16 // mi_pt // bpe


def cal_swizzle_lane_size_elements(state_or_kernel: Mapping[str, Any], tc: str, isa: Any) -> int:
    """
    Elements along one swizzle lane for graUnrollOffsets / F(row,col).

    Legacy formula: (MatrixInstK // 4) * swizzlePackK.

    RDNA4 SwizzleTensorA: always 8 (= GRVW for all datatypes with PackK=1).
      - FP16/BF16: 8 half  = 16 bytes (buffer_load_b128)
      - FP8/BF8:   8 FP8   = 8 bytes  (buffer_load_b64)
    """
    mik = int(state_or_kernel["MatrixInstK"])
    pk = cal_swizzle_pack_k_tensor(state_or_kernel, tc, isa)
    pt = state_or_kernel["ProblemType"]
    if is_rdna4_swizzle_slab_isa(isa) and _is_rdna4_swizzle_a(pt, tc):
        return 8
    return (mik // 4) * pk
