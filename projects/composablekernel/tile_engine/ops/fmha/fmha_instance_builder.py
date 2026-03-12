#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA kernel sweep builder for the tile engine.

Expands JSON sweep configs via cartesian product, then filters through
CK-compatible validation rules. The JSON defines the superset of all
possible values; the builder trims to valid-only configs using arch_specs.

Usage:
    python fmha_instance_builder.py configs/receipt0_fwd.json --arch gfx950
    python fmha_instance_builder.py configs/fwd.json --arch gfx950 --list
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from fmha_utils import FmhaKernelConfig  # noqa: E402

VARIANT_TO_FAMILY = {
    "fwd": "fwd",
    "bwd": "bwd_dq_dk_dv",
    "splitkv": "fwd_splitkv",
    "appendkv": "fwd_appendkv",
    "pagedkv": "fwd_pagedkv",
    "batch_prefill": "batch_prefill",
}


def _load_arch_specs() -> dict:
    specs_path = _DISPATCHER_ROOT / "codegen" / "fmha_arch_specs.json"
    with open(specs_path) as f:
        return json.load(f)


def _build_tile_lookup(arch_specs: dict, arch: str) -> Dict[str, List[Tuple]]:
    """Build {dtype -> {(hdim_q, hdim_v) -> [full_6_tile, ...]}} from arch_specs."""
    arch_info = None
    for a, info in arch_specs.get("architectures", {}).items():
        if a == arch:
            arch_info = info
            break
    if arch_info is None:
        for a, info in arch_specs.get("architectures", {}).items():
            if arch.startswith(a[:5]):
                arch_info = info
                break
    if arch_info is None:
        return {}

    combos = arch_info.get("hdim_tile_combos", {})
    lookup = {}
    for dtype, hdim_map in combos.items():
        if dtype not in lookup:
            lookup[dtype] = {}
        for hdim_key, tiles in hdim_map.items():
            parts = hdim_key.split("_")
            hq, hv = int(parts[0]), int(parts[1])
            lookup[dtype][(hq, hv)] = [tuple(t) for t in tiles]
    return lookup


def _pipeline_ok(dtype: str, pipe: str, arch_info: dict) -> bool:
    if "trload" in pipe and not arch_info.get("supports_trload", False):
        return False
    if "v3" in pipe and not arch_info.get("supports_v3", False):
        return False
    if "fp8" in dtype and not arch_info.get("supports_fp8", False):
        return False
    return True


def expand_sweep(config_path: str, arch: str) -> List[FmhaKernelConfig]:
    """Expand JSON sweep via cartesian product + arch_specs-based filtering."""
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    family = VARIANT_TO_FAMILY[variant]

    arch_specs = _load_arch_specs()
    tile_lookup = _build_tile_lookup(arch_specs, arch)

    arch_info = {}
    for a, info in arch_specs.get("architectures", {}).items():
        if a == arch or arch.startswith(a[:5]):
            arch_info = info
            break

    trait_cfg = config.get("trait_config", {})
    dtypes = trait_cfg.get("data_type", {}).get("values", ["fp16"])
    pipelines = trait_cfg.get("pipeline", {}).get("values", ["qr_async"])
    masks = trait_cfg.get("mask", {}).get("values", ["no"])
    biases = trait_cfg.get("bias", {}).get("values", ["no"])
    modes = trait_cfg.get("mode", {}).get("values", ["batch"])
    lses = trait_cfg.get("lse", {}).get("values", [False])
    dropouts = trait_cfg.get("dropout", {}).get("values", [False])
    logits_vals = trait_cfg.get("logits", {}).get("values", [False])
    sinks = trait_cfg.get("sink", {}).get("values", [False])

    configs = []

    for dtype in dtypes:
        dtype_tiles = tile_lookup.get(dtype, {})
        if not dtype_tiles:
            continue

        for pipe in pipelines:
            if not _pipeline_ok(dtype, pipe, arch_info):
                continue

            is_fp8 = "fp8" in dtype
            warp_k = 32 if is_fp8 else 16
            wave_m = 2 if is_fp8 else 4

            for (hq, hv), tiles in dtype_tiles.items():
                for tile in tiles:
                    m0, n0, k0, n1, k1, k0max = tile

                    for mask, bias, mode, lse, drop, log_sc, sink in itertools.product(
                        masks, biases, modes, lses, dropouts, logits_vals, sinks
                    ):
                        if log_sc and bias != "no":
                            continue

                        configs.append(
                            FmhaKernelConfig(
                                family=family,
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline=pipe,
                                tile_m0=m0,
                                tile_n0=n0,
                                tile_k0=k0,
                                tile_n1=n1,
                                tile_k1=k1,
                                tile_k0max=k0max,
                                wave_m0=wave_m,
                                wave_n0=1,
                                wave_k0=1,
                                wave_m1=wave_m,
                                wave_n1=1,
                                wave_k1=1,
                                warp_k0=warp_k,
                                warp_k1=warp_k,
                                mask=mask,
                                bias=bias,
                                lse=lse,
                                dropout=drop,
                                logits=log_sc,
                                sink=sink,
                                gfx_arch=arch,
                            )
                        )

    return configs


def main():
    parser = argparse.ArgumentParser(description="FMHA tile engine sweep builder")
    parser.add_argument("config", help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch)
    print(f"Expanded {args.config} -> {len(configs)} valid kernel configs")

    if args.count_only:
        return

    if args.list:
        for i, c in enumerate(configs):
            print(
                f"  [{i}] {c.name}  {c.data_type} h{c.hdim_q} {c.pipeline}"
                f" mask={c.mask} bias={c.bias} lse={c.lse} drop={c.dropout}"
            )


if __name__ == "__main__":
    main()
