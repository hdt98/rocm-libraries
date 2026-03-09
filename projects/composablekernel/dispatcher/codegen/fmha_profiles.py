#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from fmha_symbol_map import canonical_bias, canonical_qscale


PROFILE_ALIASES: Dict[str, str] = {
    "0": "ck_default",
    "1": "ck_extended",
    "2": "flash_fwd",
    "3": "flash_bwd",
    "4": "pytorch",
    "100": "aiter_batch",
    "200": "aiter_group",
    "300": "aiter_bwd_batch",
    "400": "aiter_bwd_group",
    "600": "aiter_cpp",
    "800": "fp32_all",
    "801": "fp32_min",
    "888": "fp8_test",
}


def normalize_profile(
    profile: Optional[str] = None, receipt: Optional[str] = None
) -> str:
    if profile:
        return PROFILE_ALIASES.get(str(profile), str(profile))
    if receipt is not None:
        return PROFILE_ALIASES.get(str(receipt), str(receipt))
    return "ck_default"


@dataclass(frozen=True)
class FmhaProfile:
    name: str
    predicate: Callable[[dict], bool]

    def allows(self, config: dict) -> bool:
        return self.predicate(config)


def _dtype_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["data_type"] in set(allowed)


def _mode_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["mode"] in set(allowed)


def _family_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["family"] in set(allowed)


def _common_row_major_filter(config: dict) -> bool:
    return config["signature"]["vlayout"] == "r"


def _bias_is(config: dict, allowed: Iterable[str]) -> bool:
    return canonical_bias(config["signature"]["bias"]) in set(allowed)


def _qscale_is(config: dict, allowed: Iterable[str]) -> bool:
    return canonical_qscale(config["signature"]["qscale"]) in set(allowed)


def _no_skip_or_logits(config: dict) -> bool:
    return (not config["signature"]["skip_min_seqlen_q"]) and (
        not config["signature"]["logits"]
    )


def _allow_all(_: dict) -> bool:
    return True


PROFILES: Dict[str, FmhaProfile] = {
    "ck_default": FmhaProfile(
        "ck_default", lambda c: c["signature"]["data_type"] != "fp32"
    ),
    "ck_extended": FmhaProfile(
        "ck_extended",
        lambda c: c["signature"]["data_type"] != "fp32",
    ),
    "flash_fwd": FmhaProfile(
        "flash_fwd",
        lambda c: _family_is(c, {"fwd", "fwd_splitkv", "fwd_appendkv", "fwd_pagedkv"})
        and _dtype_is(c, {"fp16", "bf16"})
        and _common_row_major_filter(c)
        and _bias_is(c, {"no", "alibi"})
        and _qscale_is(c, {"no"})
        and not c["signature"]["skip_min_seqlen_q"],
    ),
    "flash_bwd": FmhaProfile(
        "flash_bwd",
        lambda c: _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
        and _dtype_is(c, {"fp16", "bf16"}),
    ),
    "pytorch": FmhaProfile(
        "pytorch",
        lambda c: _dtype_is(c, {"fp16", "bf16"})
        and _common_row_major_filter(c)
        and _bias_is(c, {"no", "bias"})
        and _qscale_is(c, {"no"})
        and _no_skip_or_logits(c)
        and not c["signature"].get("sink", False),
    ),
    "aiter_batch": FmhaProfile(
        "aiter_batch",
        lambda c: _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
        and _mode_is(c, {"batch"})
        and _common_row_major_filter(c)
        and (
            c["signature"]["data_type"] != "fp8bf16"
            or c["signature"]["hdim_q"] in {128, 192}
        ),
    ),
    "aiter_group": FmhaProfile(
        "aiter_group",
        lambda c: _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
        and _mode_is(c, {"group"})
        and _common_row_major_filter(c),
    ),
    "aiter_bwd_batch": FmhaProfile(
        "aiter_bwd_batch",
        lambda c: _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
        and _dtype_is(c, {"fp16", "bf16"})
        and _mode_is(c, {"batch"}),
    ),
    "aiter_bwd_group": FmhaProfile(
        "aiter_bwd_group",
        lambda c: _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
        and _dtype_is(c, {"fp16", "bf16"})
        and _mode_is(c, {"group"}),
    ),
    "aiter_cpp": FmhaProfile(
        "aiter_cpp",
        lambda c: _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
        and _common_row_major_filter(c)
        and (
            c["signature"]["data_type"] != "fp8bf16"
            or c["signature"]["hdim_q"] in {128, 192}
        ),
    ),
    "fp32_all": FmhaProfile(
        "fp32_all", lambda c: _dtype_is(c, {"fp32"}) and _no_skip_or_logits(c)
    ),
    "fp32_min": FmhaProfile(
        "fp32_min",
        lambda c: _dtype_is(c, {"fp32"})
        and _mode_is(c, {"batch"})
        and c["signature"]["hdim_q"] in {48, 128}
        and c["signature"]["hdim_v"] in {48, 128}
        and canonical_bias(c["signature"]["bias"]) == "no"
        and not c["signature"]["lse"]
        and not c["signature"]["dropout"]
        and canonical_qscale(c["signature"]["qscale"]) == "no",
    ),
    "fp8_test": FmhaProfile(
        "fp8_test",
        lambda c: _dtype_is(c, {"fp8bf16", "fp8fp32"})
        and c["signature"]["hdim_q"] in {128, 192}
        and _common_row_major_filter(c),
    ),
    "all": FmhaProfile("all", _allow_all),
}


def get_profile(
    profile: Optional[str] = None, receipt: Optional[str] = None
) -> FmhaProfile:
    normalized = normalize_profile(profile=profile, receipt=receipt)
    if normalized not in PROFILES:
        raise KeyError(f"Unknown FMHA profile: {normalized}")
    return PROFILES[normalized]


def profile_allows(
    config: dict, profile: Optional[str] = None, receipt: Optional[str] = None
) -> bool:
    return get_profile(profile=profile, receipt=receipt).allows(config)
