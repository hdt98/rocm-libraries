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
    "gfx942": ["fp16", "bf16", "fp32", "fp8bf16", "fp8fp32"],
    "gfx950": ["fp16", "bf16", "fp32", "fp8bf16", "fp8fp32"],
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
