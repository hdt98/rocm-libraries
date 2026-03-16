#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Self-contained FMHA pipeline selection and compatibility rules.

Reproduces the exact filtering logic from CK Tile's codegen/ops/fmha_fwd.py
without importing from the CK example folder. All rules are encoded here.

This file is the authoritative source for which (dtype, hdim, pipeline, features)
combinations produce valid FMHA kernels. Use validate_arch_specs_parity.py to
verify parity with the CK upstream.
"""

import itertools
from dataclasses import dataclass
from typing import List, Tuple

# Supported mask types for 'generic' mask_impl (default in CK)
MASKS = ["no", "causal", "generic"]
BIASES = ["no", "bias", "alibi"]
BOOLS = ["t", "f"]


@dataclass(frozen=True)
class PipelineSpec:
    """One pipeline variant with its feature flags and padding."""

    tag: str
    mask: str
    bias: str
    lse: str
    dropout: str
    logits: str
    skip: str
    sink: str
    qscale: str = "no"
    spad: str = "f"
    skpad: str = "f"
    dpad: str = "f"
    dvpad: str = "f"


def _feature_product_fp16bf16(
    pipeline_tag: str,
    hdim: int,
    hdim_v: int,
    receipt: int,
) -> List[PipelineSpec]:
    """Pipeline specs for fp16/bf16 on gfx9/gfx950 (matches KernelComponentFactoryGfx9.get_pipelines)."""
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        if hdim == 256 and hdim_v == 256:
            # hdim=256: only qr, 3 pad variants
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
        else:
            if bias == "bias":
                # bias="bias" forces qr (rocm compiler workaround)
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="f",
                        skpad="f",
                        dpad="f",
                        dvpad="f",
                    )
                )
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )
            else:
                # Default: qr_async, 2 pad variants
                specs.append(
                    PipelineSpec(
                        "qr_async",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="f",
                        dpad="t",
                        dvpad="t",
                    )
                )
                specs.append(
                    PipelineSpec(
                        "qr_async",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )
            if receipt == 1 and bias != "bias":
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )

    return specs


def _feature_product_fp16bf16_gfx950_extra(
    hdim: int,
    hdim_v: int,
) -> List[PipelineSpec]:
    """Additional trload/v3 pipelines for gfx950 fp16/bf16 (matches KernelComponentFactoryGfx950.get_pipelines)."""
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        if (
            (hdim, hdim_v) in [(64, 64), (128, 128)]
            and logits == "f"
            and bias == "no"
            and dropout == "f"
            and skip == "f"
        ):
            specs.append(
                PipelineSpec(
                    "qr_async_trload",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_async_trload",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="t",
                    dvpad="t",
                )
            )

    # v3 only for (128,128)
    if (hdim, hdim_v) == (128, 128):
        for logits, mask in itertools.product(BOOLS, ["no", "causal"]):
            specs.append(
                PipelineSpec(
                    "qr_async_trload_v3",
                    mask,
                    "no",
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    return specs


def _feature_product_fp8(
    pipeline_tag_base: str,
    hdim: int,
    hdim_v: int,
) -> List[PipelineSpec]:
    """Pipeline specs for fp8bf16/fp8fp32 (matches KernelComponentFactoryGfx9.get_pipelines fp8 path)."""
    specs: List[PipelineSpec] = []

    for logits, qscale, mask, bias, sink in itertools.product(
        BOOLS,
        ["no", "pertensor", "blockscale"],
        MASKS,
        ["no"],
        BOOLS,
    ):
        if hdim == 64:
            tag = "qr"
        else:
            tag = "qr_async"
        specs.append(
            PipelineSpec(
                tag,
                mask,
                bias,
                "f",
                "f",
                logits,
                "f",
                sink,
                qscale=qscale,
                spad="t",
                skpad="f",
                dpad="t",
                dvpad="t",
            )
        )
        specs.append(
            PipelineSpec(
                tag,
                mask,
                bias,
                "f",
                "f",
                logits,
                "f",
                sink,
                qscale=qscale,
                spad="t",
                skpad="t",
                dpad="t",
                dvpad="t",
            )
        )

    return specs


def _feature_product_fp32(
    hdim: int,
    hdim_v: int,
) -> List[PipelineSpec]:
    """Pipeline specs for fp32 (matches KernelComponentFactoryGfx9.get_pipelines fp32 path)."""
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="f",
                skpad="f",
                dpad="f",
                dvpad="f",
            )
        )
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="f",
                skpad="t",
                dpad="f",
                dvpad="f",
            )
        )
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="t",
                skpad="t",
                dpad="t",
                dvpad="t",
            )
        )

    return specs


# ===== Compatibility Rules (matches CompatibilityRuleFactory hierarchy) =====


def _check_mode(mode: str, spec: PipelineSpec) -> bool:
    """Group mode requires spad=t and skpad=t."""
    if mode == "group":
        return spec.spad == "t" and spec.skpad == "t"
    return True


def _check_feature(spec: PipelineSpec) -> bool:
    """logits_soft_cap requires no bias."""
    if spec.logits == "t" and spec.bias != "no":
        return False
    return True


def _check_hdim_tile_gfx9(
    dtype: str,
    hdim: int,
    hdim_v: int,
    pipeline_tag: str,
    tile_bm0: int,
    tile_bn0: int,
    tile_bk0: int,
) -> bool:
    """Gfx9 tile constraints (matches CompatibilityRuleFactoryGfx9.check_hdim_tile).

    IMPORTANT: This rule uses the GFX9 _AVAILABLE_PIPELINES set {qr, qr_async, qs},
    NOT the gfx950 expanded set. In CK, the closure captures cls=CompatibilityRuleFactoryGfx9
    because gfx950's get_rules() calls CompatibilityRuleFactoryGfx9.get_rules() directly
    (not super().get_rules()). So trload/v3 pipelines bypass this rule entirely and are
    handled by check_tile_pipeline_gfx950 instead.
    """
    if dtype == "fp32":
        return True
    gfx9_pipelines = {"qr", "qr_async", "qs"}
    if pipeline_tag not in gfx9_pipelines:
        return True
    if (hdim, hdim_v) == (128, 128) and tile_bn0 != 128:
        return False
    if (hdim, hdim_v) == (128, 128) and pipeline_tag == "qr_async" and tile_bm0 != 128:
        return False
    if (hdim, hdim_v) != (128, 128) and tile_bm0 != 128:
        return False
    if (hdim, hdim_v) == (128, 128) and pipeline_tag != "qr_async" and tile_bk0 == 64:
        return False
    return True


def _check_tile_pipeline_gfx950(
    hdim: int,
    hdim_v: int,
    pipeline_tag: str,
    tile_bm0: int,
    tile_bn0: int,
) -> bool:
    """Gfx950 trload/v3 tile constraints (matches CompatibilityRuleFactoryGfx950.check_tile_pipeline).

    The CK rule also checks warp counts (rm0*rn0*rk0==8) for v3, but since bm0=256 is
    the ONLY tile with 8 warps in the tile table, bm0==256 is a sufficient discriminant.
    """
    if pipeline_tag == "qr_async_trload":
        if (hdim, hdim_v) == (128, 128) and tile_bn0 == 128:
            return False
        if (hdim, hdim_v) not in [(64, 64), (128, 128)]:
            return False
    is_v3_dedicated_tile = tile_bm0 == 256
    is_v3_pipeline = pipeline_tag == "qr_async_trload_v3"
    if is_v3_dedicated_tile != is_v3_pipeline:
        return False
    return True


# ===== Receipt / Product filters =====

RECEIPT_FILTERS = {
    0: lambda dtype, spec: dtype != "fp32",
    2: lambda dtype, spec: (
        dtype in ("fp16", "bf16")
        and spec.bias in ("no", "alibi")
        and spec.qscale == "no"
        and spec.skip == "f"
        and spec.sink == "f"
    ),
    4: lambda dtype, spec: (
        dtype in ("fp16", "bf16")
        and spec.bias in ("no", "bias")
        and spec.qscale == "no"
        and spec.skip == "f"
        and spec.logits == "f"
    ),
    100: lambda dtype, spec: (dtype in ("fp16", "bf16", "fp8bf16")),
    200: lambda dtype, spec: (dtype in ("fp16", "bf16", "fp8bf16")),
    600: lambda dtype, spec: (dtype in ("fp16", "bf16", "fp8bf16")),
    888: lambda dtype, spec: (dtype in ("fp8bf16", "fp8fp32")),
    800: lambda dtype, spec: (
        dtype == "fp32" and spec.skip == "f" and spec.logits == "f"
    ),
}


def receipt_filter(receipt: int, dtype: str, spec: PipelineSpec) -> bool:
    """Apply receipt-level filter. Returns True if the kernel should be kept."""
    fn = RECEIPT_FILTERS.get(receipt)
    if fn is None:
        return dtype != "fp32"
    return fn(dtype, spec)


# ===== Main enumeration =====

# Dtype groups matching CK's _DT_ constants
_DT_FP16_BF16 = {"fp16", "bf16"}
_DT_FP8BF16 = {"fp8bf16", "fp8", "bf8"}
_DT_FP8FP32 = {"fp8fp32"}
_DT_FP32 = {"fp32"}

# Supported dtypes per arch family
ARCH_DTYPES = {
    "gfx90a": ["fp16", "bf16", "fp32"],
    "gfx942": ["fp16", "bf16", "fp32", "fp8bf16", "fp8fp32", "fp8", "bf8"],
    "gfx950": ["fp16", "bf16", "fp32", "fp8bf16", "fp8fp32", "fp8", "bf8"],
    "gfx1100": ["fp16", "bf16"],
    "gfx1201": ["fp16", "bf16"],
}


def get_pipelines_for_config(
    arch: str,
    dtype: str,
    hdim: int,
    hdim_v: int,
    receipt: int = 0,
) -> List[PipelineSpec]:
    """Get all valid pipeline specs for a given (arch, dtype, hdim, hdim_v, receipt).

    This is the self-contained equivalent of CK's get_pipelines() factory method.
    """
    specs: List[PipelineSpec] = []

    if dtype in _DT_FP32:
        specs = _feature_product_fp32(hdim, hdim_v)
    elif dtype in _DT_FP16_BF16:
        specs = _feature_product_fp16bf16("qr_async", hdim, hdim_v, receipt)
        if arch in ("gfx950",):
            specs.extend(_feature_product_fp16bf16_gfx950_extra(hdim, hdim_v))
    elif dtype in _DT_FP8BF16 or dtype in _DT_FP8FP32:
        specs = _feature_product_fp8("qr", hdim, hdim_v)
    else:
        return []

    # Apply compatibility rules
    result = []
    for spec in specs:
        if not _check_feature(spec):
            continue
        if not receipt_filter(receipt, dtype, spec):
            continue
        result.append(spec)

    return result


# ===== Variant-specific tile tables =====
# These are separate from the fwd hdim_tile_combos in fmha_arch_specs.json.
# Each variant has its own (typically smaller) tile set per hdim.

SPLITKV_TILES_FP16 = {
    (32, 32): (32, 64, 16, 32, 32, 32),
    (64, 64): (64, 64, 32, 64, 32, 64),
    (96, 128): (64, 128, 32, 128, 32, 96),
    (128, 128): (64, 128, 32, 128, 32, 128),
    (256, 256): (64, 128, 32, 256, 32, 256),
}

SPLITKV_TILES_FP8 = {
    (64, 64): (128, 64, 32, 64, 32, 64),
    (128, 128): (128, 128, 32, 128, 32, 128),
}

SPLITKV_COMBINE_HDIMS_FP16 = [32, 64, 96, 128, 256]
SPLITKV_COMBINE_HDIMS_FP8 = [64, 128, 256]

PAGEDKV_TILES_FP16 = {
    (128, 128): (64, 128, 32, 128, 32, 128),
}

PAGEDKV_TILES_FP8 = {
    (64, 64): (128, 64, 32, 64, 32, 64),
    (128, 128): (128, 128, 32, 128, 32, 128),
    (256, 256): (64, 128, 32, 256, 32, 256),
}

# Append-KV tiles: (bs, bsk, bd, bdv)
APPENDKV_TILES_FP16 = {
    32: (64, 64, 32, 32),
    64: (64, 64, 64, 64),
    128: (64, 64, 128, 128),
    256: (64, 64, 256, 256),
}

APPENDKV_TILES_FP8 = {
    64: (64, 64, 64, 64),
    128: (64, 64, 128, 128),
    256: (64, 64, 256, 256),
}

# Batch prefill tiles (hdim -> tile, same as fwd for the hdims that exist)
BATCH_PREFILL_TILES_FP16 = {
    (128, 128): [
        (128, 128, 32, 128, 32, 128),
        (64, 128, 64, 128, 64, 128),  # CustomFactory extra tile
    ],
    (256, 256): [
        (128, 128, 32, 256, 32, 256),
    ],
}

BATCH_PREFILL_TILES_FP8 = {
    (128, 128): [
        (128, 128, 32, 128, 32, 128),
    ],
}

# BWD dq_dk_dv: simple single tile per hdim (the "main" tile).
# Multiple tiles per hdim exist in CK (trload, small, bn192 variants) but
# we only enumerate the main tile for now. The feature product per tile is
# 3 masks x 4 (bias,dbias) x 3 dropout x 2 deterministic x 7 pads = 504.
BWD_DQ_DK_DV_TILES_FP16 = {
    (32, 32): (32, 128, 32, 32, 32, 32, 64, 32, 32),
    (64, 64): (32, 128, 64, 32, 64, 32, 32, 64, 64),
    (96, 128): (32, 128, 96, 32, 96, 32, 32, 96, 96),
    (128, 128): (16, 128, 128, 16, 128, 16, 32, 128, 128),
    (256, 256): (16, 64, 256, 16, 256, 16, 32, 256, 256),
}

# Additional tiles for h64 (2 extra) and h128 (3 extra).
# Each entry: (tile_tuple, tag, is_batch_only)
BWD_DQ_DK_DV_EXTRA_TILES = {
    (64, 64): [
        ((32, 128, 64, 32, 64, 32, 32, 64, 64), "trload", False),
        ((32, 16, 64, 32, 64, 32, 16, 64, 64), "small", True),
    ],
    (128, 128): [
        ((16, 16, 128, 16, 128, 16, 16, 128, 128), "small", True),
        ((16, 192, 128, 16, 128, 16, 32, 128, 128), "bn192", False),
        ((32, 128, 128, 32, 128, 32, 32, 128, 128), "trload", False),
    ],
}

# Extra tiles use reduced pad combos
BWD_EXTRA_PAD_COMBOS = [
    ("f", "f"),  # dpad=0, dvpad=0
    ("8", "8"),  # dpad=8, dvpad=8
]

BWD_SMALL_DROPOUTS = ["no"]  # small tiles: no dropout

BWD_DOT_DO_O_HDIMS = [32, 64, 96, 128, 256]
BWD_CONVERT_DQ_HDIMS = [32, 64, 96, 128, 256]

# Per-hdim number of associated dq_dk_dv tile groups for convert_dq.
# h128 has 3 tile groups (main, bn192, trload) that produce convert_dq kernels.
# Others have 1 tile group (main only). Small tiles don't produce convert_dq.
# h128 has extra convert_dq variants for the bn192 and trload tiles.
# These are captured via extra (spad, dpad) combos, not via tile_groups.
BWD_CONVERT_DQ_TILE_GROUPS = {32: 1, 64: 1, 96: 1, 128: 1, 256: 1}


# ===== Split-KV pipeline rules (matches fmha_fwd_splitkv.py) =====


@dataclass(frozen=True)
class SplitKVPipelineSpec:
    """Split-KV main kernel pipeline variant."""

    tag: str  # "qr" always for split-KV
    mask: str
    bias: str
    logits: str
    sink: str
    pagedkv: str = "f"
    squant: str = "f"
    spad: str = "f"
    skpad: str = "f"
    dpad: str = "f"
    dvpad: str = "f"
    lse: str = "t"  # split-KV always has lse


@dataclass(frozen=True)
class SplitKVCombineSpec:
    """Split-KV combine kernel pipeline variant."""

    spad: str
    dvpad: str
    lse: str
    squant: str = "f"


def get_splitkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[SplitKVPipelineSpec]:
    """Split-KV main kernel pipelines (matches KernelComponentFactoryBase.get_pipelines)."""
    specs: List[SplitKVPipelineSpec] = []

    if dtype in _DT_FP16_BF16:
        for logits, mask, bias, pagedkv, sink in itertools.product(
            BOOLS,
            MASKS,
            BIASES,
            BOOLS,
            BOOLS,
        ):
            if logits == "t" and bias != "no":
                continue
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
    elif dtype in ("fp8", "bf8"):
        for logits, mask, bias in itertools.product(BOOLS, MASKS, BIASES):
            if logits == "t" and bias != "no":
                continue
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    squant="t",
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    squant="t",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    if receipt != 0:
        specs = [s for s in specs if _splitkv_receipt_filter(receipt, dtype, s)]

    return specs


def _splitkv_receipt_filter(
    receipt: int, dtype: str, spec: SplitKVPipelineSpec
) -> bool:
    if receipt == 2:
        return (
            dtype in ("fp16", "bf16")
            and spec.bias in ("no", "alibi")
            and spec.squant == "f"
            and spec.sink == "f"
        )
    if receipt == 4:
        return (
            dtype in ("fp16", "bf16")
            and spec.bias in ("no", "bias")
            and spec.squant == "f"
            and spec.sink == "f"
        )
    if receipt == 200:
        return dtype in ("fp16", "bf16") and spec.squant == "f"
    if receipt == 600:
        return dtype in ("fp16", "bf16") and spec.squant == "f"
    if receipt in (800, 801):
        return dtype == "fp32"
    return True


def get_splitkv_combine_pipelines(
    dtype: str, receipt: int = 0
) -> List[SplitKVCombineSpec]:
    """Split-KV combine kernel pipelines (matches KernelComponentFactoryBase.get_combine_pipelines)."""
    specs: List[SplitKVCombineSpec] = []
    squant = "t" if dtype in ("fp8", "bf8") else "f"

    if dtype in _DT_FP16_BF16:
        for spad, dvpad, lse in itertools.product(BOOLS, BOOLS, BOOLS):
            specs.append(SplitKVCombineSpec(spad, dvpad, lse, squant))
    elif dtype in ("fp8", "bf8"):
        for spad, dvpad in itertools.product(BOOLS, BOOLS):
            specs.append(SplitKVCombineSpec(spad, dvpad, "f", squant))

    return specs


# ===== PagedKV pipeline rules (matches fmha_pagedkv_prefill.py) =====


def get_pagedkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[PipelineSpec]:
    """PagedKV prefill pipelines (matches fmha_pagedkv_prefill.py KernelComponentFactoryBase.get_pipelines)."""
    specs: List[PipelineSpec] = []

    if dtype in _DT_FP16_BF16:
        for logits, mask, bias, sink in itertools.product(
            BOOLS,
            MASKS,
            BIASES,
            BOOLS,
        ):
            if logits == "t" and bias != "no":
                continue
            # pagedkv=t, skip=f always; 2 pad variants (skpad varies)
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    sink,
                    spad="t",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
    elif dtype in ("fp8", "bf8"):
        for logits, mask, bias in itertools.product(BOOLS, MASKS, BIASES):
            if logits == "t" and bias != "no":
                continue
            # fp8: pagedkv=t, skip=f, sink=f; 2 pad variants
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    if receipt != 0:
        specs = [s for s in specs if receipt_filter(receipt, dtype, s)]

    return specs


# ===== Append-KV pipeline rules (matches fmha_fwd_appendkv.py) =====


@dataclass(frozen=True)
class AppendKVPipelineSpec:
    """Append-KV pipeline variant."""

    rope: str = "none"  # none, interleaved, half_rotated
    pagedkv: str = "f"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


def get_appendkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[AppendKVPipelineSpec]:
    """Append-KV pipelines (matches KernelComponentFactoryBase.get_pipelines for appendkv)."""
    specs: List[AppendKVPipelineSpec] = []

    if dtype in _DT_FP16_BF16:
        for pagedkv in ["t", "f"]:
            # rope=no: 2 pad variants
            specs.append(
                AppendKVPipelineSpec(
                    rope="none",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="none",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
            # rope=interleaved: 2 pad variants (dpad=t always for rope)
            specs.append(
                AppendKVPipelineSpec(
                    rope="interleaved",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="t",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="interleaved",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
            # rope=half_rotated: 2 pad variants
            specs.append(
                AppendKVPipelineSpec(
                    rope="half_rotated",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="t",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="half_rotated",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
    elif dtype in ("fp8", "bf8"):
        specs.append(
            AppendKVPipelineSpec(
                rope="none", pagedkv="f", spad="t", skpad="t", dpad="t", dvpad="t"
            )
        )

    return specs


# ===== Batch Prefill pipeline rules (matches fmha_batch_prefill.py) =====


@dataclass(frozen=True)
class BatchPrefillPipelineSpec:
    """Batch prefill pipeline variant -- extends PipelineSpec with paged-KV fields."""

    mask: str
    bias: str
    logits: str
    sink: str
    lse: str = "f"
    dropout: str = "f"
    skip: str = "f"
    qscale: str = "no"
    page_size: int = 1
    kv_memory_layout: str = "vectorized"
    kv_lookup_table: str = "sglang"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


def get_batch_prefill_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[BatchPrefillPipelineSpec]:
    """Batch prefill pipelines (matches fmha_batch_prefill.py KernelComponentFactory.get_pipelines).

    Note: page_size is NOT part of the pipeline -- it's iterated at kernel level.
    This function returns pipeline specs without page_size; the builder adds page_size.
    """
    specs: List[BatchPrefillPipelineSpec] = []

    if dtype in _DT_FP16_BF16:
        for logits, mask, bias, lse, dropout, kvl, kvt in itertools.product(
            BOOLS,
            MASKS,
            BIASES,
            BOOLS,
            BOOLS,
            ["vectorized", "linear"],
            ["vllm", "sglang"],
        ):
            if logits == "t" and bias != "no":
                continue
            # Single pad variant: all t
            specs.append(
                BatchPrefillPipelineSpec(
                    mask,
                    bias,
                    logits,
                    "f",
                    lse,
                    dropout,
                    "f",
                    page_size=0,
                    kv_memory_layout=kvl,
                    kv_lookup_table=kvt,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
    elif dtype == "fp8bf16":
        for logits, qscale, mask, bias, kvl, kvt in itertools.product(
            BOOLS,
            ["pertensor", "kv_blockscale"],
            MASKS,
            ["no"],
            ["vectorized", "linear"],
            ["vllm", "sglang"],
        ):
            if logits == "t" and bias != "no":
                continue
            specs.append(
                BatchPrefillPipelineSpec(
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    "f",
                    "f",
                    qscale=qscale,
                    page_size=0,
                    kv_memory_layout=kvl,
                    kv_lookup_table=kvt,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )

    return specs


# ===== BWD pipeline rules (matches fmha_bwd.py) =====


@dataclass(frozen=True)
class BwdPipelineSpec:
    """BWD pipeline variant."""

    family: str  # "bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"
    mask: str = "no"
    bias: str = "no"
    dbias: str = "f"
    dropout: str = "f"
    deterministic: str = "f"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


BWD_DROPOUTS = ["no", "dropout_wg16", "dropout_wg16_storerandval"]
BWD_PAD_COMBOS = [
    ("f", "f"),  # dpad=0, dvpad=0
    ("f", "t"),  # dpad=0, dvpad=1
    ("f", "8"),  # dpad=0, dvpad=8
    ("t", "f"),  # dpad=1, dvpad=0
    ("t", "t"),  # dpad=1, dvpad=1
    ("t", "8"),  # dpad=1, dvpad=8
    ("8", "8"),  # dpad=8, dvpad=8
]


def get_bwd_dq_dk_dv_pipelines(dtype: str, receipt: int = 0) -> List[BwdPipelineSpec]:
    """BWD dq_dk_dv feature product (matches fmha_bwd.py iteration).

    72 features x 7 pad combos = 504 per (hdim, tile, mode).
    Features: 3 masks x 4 (bias,dbias) x 3 dropout x 2 deterministic = 72.
    """
    if dtype not in _DT_FP16_BF16:
        return []
    specs: List[BwdPipelineSpec] = []
    for mask, bias, dbias, dropout, deterministic in itertools.product(
        MASKS,
        BIASES,
        BOOLS,
        BWD_DROPOUTS,
        BOOLS,
    ):
        if bias != "bias" and dbias == "t":
            continue
        for dpad, dvpad in BWD_PAD_COMBOS:
            specs.append(
                BwdPipelineSpec(
                    "bwd_dq_dk_dv",
                    mask,
                    bias,
                    dbias,
                    dropout,
                    deterministic,
                    dpad=dpad,
                    dvpad=dvpad,
                )
            )
    return specs


def get_bwd_dot_do_o_pipelines(dtype: str) -> List[BwdPipelineSpec]:
    """BWD dot_do_o: spad x dvpad variants only."""
    if dtype not in _DT_FP16_BF16:
        return []
    specs: List[BwdPipelineSpec] = []
    for spad, dvpad in itertools.product(BOOLS, BOOLS):
        specs.append(BwdPipelineSpec("bwd_dot_do_o", spad=spad, dvpad=dvpad))
    return specs


def get_bwd_convert_dq_pipelines(dtype: str, hdim: int = 0) -> List[BwdPipelineSpec]:
    """BWD convert_dq: spad x deterministic x dpad variants.
    h128 has dpad in {f, t, 8} (3 values); others have {f, t} (2 values).
    """
    if dtype not in _DT_FP16_BF16:
        return []
    dpads = ["f", "t", "8"] if hdim == 128 else BOOLS
    specs: List[BwdPipelineSpec] = []
    for spad, deterministic, dpad in itertools.product(BOOLS, BOOLS, dpads):
        specs.append(
            BwdPipelineSpec(
                "bwd_convert_dq", spad=spad, deterministic=deterministic, dpad=dpad
            )
        )
    return specs


def get_bwd_pipelines(dtype: str, hdim: int, receipt: int = 0) -> List[BwdPipelineSpec]:
    """All BWD pipelines combined."""
    return (
        get_bwd_dot_do_o_pipelines(dtype)
        + get_bwd_dq_dk_dv_pipelines(dtype, receipt)
        + get_bwd_convert_dq_pipelines(dtype)
    )


def get_bwd_dq_dk_dv_extra_pipelines(
    dtype: str, is_small: bool = False, receipt: int = 0
) -> List[BwdPipelineSpec]:
    """Reduced feature product for BWD extra tiles.
    trload/bn192: 72 features x 2 pads = 144 per mode.
    small: 24 features (no dropout) x 2 pads = 48, batch-only.
    """
    if dtype not in _DT_FP16_BF16:
        return []
    dropouts = BWD_SMALL_DROPOUTS if is_small else BWD_DROPOUTS
    specs: List[BwdPipelineSpec] = []
    for mask, bias, dbias, dropout, deterministic in itertools.product(
        MASKS,
        BIASES,
        BOOLS,
        dropouts,
        BOOLS,
    ):
        if bias != "bias" and dbias == "t":
            continue
        for dpad, dvpad in BWD_EXTRA_PAD_COMBOS:
            specs.append(
                BwdPipelineSpec(
                    "bwd_dq_dk_dv",
                    mask,
                    bias,
                    dbias,
                    dropout,
                    deterministic,
                    dpad=dpad,
                    dvpad=dvpad,
                )
            )
    return specs


def tile_compatible(
    arch: str,
    dtype: str,
    hdim: int,
    hdim_v: int,
    pipeline_tag: str,
    tile: Tuple[int, ...],
) -> bool:
    """Check if a tile is compatible with the given config.

    tile is (bm0, bn0, bk0, bn1, bk1, bk0max) from fmha_arch_specs.json.
    """
    bm0, bn0, bk0 = tile[0], tile[1], tile[2]

    if not _check_hdim_tile_gfx9(dtype, hdim, hdim_v, pipeline_tag, bm0, bn0, bk0):
        return False

    if arch in ("gfx950",):
        if not _check_tile_pipeline_gfx950(hdim, hdim_v, pipeline_tag, bm0, bn0):
            return False

    return True
