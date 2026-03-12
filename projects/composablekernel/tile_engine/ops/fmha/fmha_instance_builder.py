#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA kernel sweep builder for the tile engine.

Expands JSON sweep configs via the self-contained pipeline rules in
fmha_pipeline_rules.py, achieving exact parity with CK's get_fwd_blobs().

Usage:
    python fmha_instance_builder.py configs/receipt0_fwd.json --arch gfx950
    python fmha_instance_builder.py configs/fwd_ci.json --arch gfx950 --list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from fmha_utils import FmhaKernelConfig, get_dispatcher_root  # noqa: E402
from fmha_pipeline_rules import (  # noqa: E402
    ARCH_DTYPES,
    get_pipelines_for_config,
    tile_compatible,
    _check_mode,
)

VARIANT_TO_FAMILY = {
    "fwd": "fwd",
    "bwd": "bwd_dq_dk_dv",
    "splitkv": "fwd_splitkv",
    "appendkv": "fwd_appendkv",
    "pagedkv": "fwd_pagedkv",
    "batch_prefill": "batch_prefill",
}

MODES = ["batch", "group"]

# Maps from PipelineSpec feature flags to FmhaKernelConfig field values
_MASK_MAP = {"no": "no", "causal": "top_left", "generic": "generic"}
_BIAS_MAP = {"no": "no", "bias": "bias", "alibi": "alibi"}


def _load_arch_specs() -> dict:
    specs_path = get_dispatcher_root() / "codegen" / "fmha_arch_specs.json"
    with open(specs_path) as f:
        return json.load(f)


def _get_tile_lookup(
    arch_specs: dict, arch: str
) -> Dict[str, Dict[Tuple[int, int], List[Tuple]]]:
    """Build {dtype -> {(hdim_q, hdim_v) -> [full_tile_tuple, ...]}} from arch_specs."""
    arch_info = None
    for a, info in arch_specs.get("architectures", {}).items():
        if a == arch or arch.startswith(a[:5]):
            arch_info = info
            break
    if arch_info is None:
        return {}

    combos = arch_info.get("hdim_tile_combos", {})
    lookup: Dict[str, Dict[Tuple[int, int], List[Tuple]]] = {}
    for dtype, hdim_map in combos.items():
        lookup[dtype] = {}
        for hdim_key, tiles in hdim_map.items():
            parts = hdim_key.split("_")
            hq, hv = int(parts[0]), int(parts[1])
            lookup[dtype][(hq, hv)] = [tuple(t) for t in tiles]
    return lookup


def expand_sweep(
    config_path: str, arch: str, receipt: int = 0
) -> List[FmhaKernelConfig]:
    """Expand JSON sweep config using self-contained pipeline rules.

    Pipeline rules (fmha_pipeline_rules.py) generate ALL valid kernels for the
    receipt. The JSON trait_config acts as an allow-list filter: if a trait key
    is present, only the listed values survive. If absent, all values pass.

    This means:
      - receipt0_fwd.json (no trait_config) -> full 11,980 kernels
      - fwd_ci.json (fp16, qr_async, no mask, no bias) -> small subset
    """
    with open(config_path) as f:
        config = json.load(f)

    variant = config["variant"]
    family = VARIANT_TO_FAMILY[variant]

    arch_specs = _load_arch_specs()
    tile_lookup = _get_tile_lookup(arch_specs, arch)

    # Build allow-list filters from JSON trait_config
    trait_cfg = config.get("trait_config", {})

    def _allow(key: str, default=None):
        entry = trait_cfg.get(key)
        if entry is None:
            return default
        return set(entry.get("values", []))

    allowed_dtypes = _allow("data_type")
    allowed_pipes = _allow("pipeline")
    allowed_masks = _allow("mask")
    allowed_biases = _allow("bias")
    allowed_modes = _allow("mode")
    allowed_lse = _allow("lse")
    allowed_dropout = _allow("dropout")
    allowed_logits = _allow("logits")
    allowed_sink = _allow("sink")

    # Intersect requested dtypes with arch support
    arch_dtypes = set(ARCH_DTYPES.get(arch, ARCH_DTYPES.get("gfx950", [])))
    if allowed_dtypes is not None:
        dtypes = sorted(allowed_dtypes & arch_dtypes)
    else:
        dtypes = sorted(arch_dtypes)

    configs: List[FmhaKernelConfig] = []

    for dtype in dtypes:
        dtype_tiles = tile_lookup.get(dtype, {})
        if not dtype_tiles:
            for alias in ("fp16", "bf16"):
                if alias in tile_lookup:
                    dtype_tiles = tile_lookup[alias]
                    break
        if not dtype_tiles:
            continue

        for (hq, hv), tiles in sorted(dtype_tiles.items()):
            pipeline_specs = get_pipelines_for_config(arch, dtype, hq, hv, receipt)

            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue

                for spec in pipeline_specs:
                    if not _check_mode(mode, spec):
                        continue
                    if allowed_pipes is not None and spec.tag not in allowed_pipes:
                        continue

                    mapped_mask = _MASK_MAP.get(spec.mask, spec.mask)
                    mapped_bias = _BIAS_MAP.get(spec.bias, spec.bias)
                    lse_val = spec.lse == "t"
                    drop_val = spec.dropout == "t"
                    logits_val = spec.logits == "t"
                    sink_val = spec.sink == "t"

                    if allowed_masks is not None and mapped_mask not in allowed_masks:
                        continue
                    if allowed_biases is not None and mapped_bias not in allowed_biases:
                        continue
                    if allowed_lse is not None and lse_val not in allowed_lse:
                        continue
                    if allowed_dropout is not None and drop_val not in allowed_dropout:
                        continue
                    if allowed_logits is not None and logits_val not in allowed_logits:
                        continue
                    if allowed_sink is not None and sink_val not in allowed_sink:
                        continue

                    for tile in tiles:
                        if not tile_compatible(arch, dtype, hq, hv, spec.tag, tile):
                            continue

                        m0, n0, k0 = tile[0], tile[1], tile[2]
                        n1 = tile[3] if len(tile) > 3 else hv
                        k1 = tile[4] if len(tile) > 4 else k0
                        k0max = tile[5] if len(tile) > 5 else hq

                        is_fp8 = "fp8" in dtype
                        warp_k = 32 if is_fp8 else 16
                        wave_m = tile[6] if len(tile) > 6 else (2 if is_fp8 else 4)

                        configs.append(
                            FmhaKernelConfig(
                                family=family,
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline=spec.tag,
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
                                pad_s=(spec.spad == "t"),
                                pad_sk=(spec.skpad == "t"),
                                pad_d=(spec.dpad == "t"),
                                pad_dv=(spec.dvpad == "t"),
                                mask=mapped_mask,
                                bias=mapped_bias,
                                lse=lse_val,
                                dropout=drop_val,
                                logits=logits_val,
                                sink=sink_val,
                                skip_min_seqlen_q=(spec.skip == "t"),
                                qscale=spec.qscale,
                                gfx_arch=arch,
                            )
                        )

    # Dedup truly identical configs (same name = same compiled kernel)
    seen: set = set()
    unique: List[FmhaKernelConfig] = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    return unique


def apply_filter(
    configs: List[FmhaKernelConfig], expr: str = "", filter_file: str = ""
) -> List[FmhaKernelConfig]:
    """Apply user-defined filters to a config list.

    Args:
        expr: Python expression evaluated per config with 'c' as the config.
              Example: "c.hdim_q == 128 and c.pipeline == 'qr_async'"
        filter_file: Path to a .py file defining filter_config(c) -> bool.

    Both can be combined (AND logic).
    """
    result = configs

    if filter_file:
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_filter", filter_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "filter_config")
        result = [c for c in result if fn(c)]

    if expr:
        result = [c for c in result if eval(expr, {"c": c})]  # noqa: S307

    return result


# --- Sample filter file (save as e.g. my_filter.py) ---
#
# def filter_config(c):
#     """Keep only h128 kernels with 128x128 tiles, no dropout."""
#     if c.hdim_q != 128:
#         return False
#     if c.tile_m0 != 128 or c.tile_n0 != 128:
#         return False
#     if c.dropout:
#         return False
#     return True


def main():
    parser = argparse.ArgumentParser(description="FMHA tile engine sweep builder")
    parser.add_argument("config", help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--receipt", type=int, default=0)
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expression per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file",
        default="",
        help="Path to .py file with filter_config(c) -> bool",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch, args.receipt)
    before = len(configs)
    configs = apply_filter(configs, args.filter_expr, args.filter_file)
    filtered = before - len(configs)

    print(
        f"Expanded {args.config} -> {before} configs"
        f"{f' (filtered {filtered}, kept {len(configs)})' if filtered else ''}"
    )

    if args.count_only:
        return

    if args.list:
        for i, c in enumerate(configs):
            print(f"  [{i}] {c.name}")


if __name__ == "__main__":
    main()
