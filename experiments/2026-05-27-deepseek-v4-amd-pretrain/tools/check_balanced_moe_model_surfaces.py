#!/usr/bin/env python3
"""Check that retained balanced-MoE perf gates have real TorchTitan surfaces.

This is intentionally source-inspection only.  Route-shape parity can prove
that MORI and Primus-Turbo agree on top-k layout contracts, but a retained
throughput gate also needs an actual TorchTitan training module/config.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TORCHTITAN_MODELS = (
    EXPERIMENT_ROOT / "sources" / "wip" / "torchtitan" / "torchtitan" / "models"
)


@dataclasses.dataclass(frozen=True)
class RequiredSurface:
    name: str
    module: str
    config_function: str
    flavor: str | None
    topology: str
    perf_gate_required: bool = True


REQUIRED_SURFACES = (
    RequiredSurface(
        name="deepseek_v4_flash_12layer",
        module="deepseek_v4",
        config_function="deepseek_v4_flash_12layer_ceiling_probe_bf16_canary",
        flavor="flash_12layer_ceiling_probe",
        topology="12 layers, 256 experts, top-k 6, DSv4 Flash MLA",
    ),
    RequiredSurface(
        name="qwen3_5_397b_a17b_reduced_8layer",
        module="qwen3",
        config_function="qwen3_5_397b_a17b_reduced_8layer_hothelper_gate",
        flavor="397B-A17B-reduced-8layer",
        topology="8 layers, 512 experts, top-k 10 plus one shared expert",
    ),
    RequiredSurface(
        name="kimi_k2_6_reduced_6layer",
        module="kimi_k2",
        config_function="kimi_k2_6_reduced_6layer_hothelper_gate",
        flavor="kimi-k2-6-reduced-6layer",
        topology=(
            "6 layers, 384 experts, top-k 8 plus one shared expert, MLA, "
            "native hidden 7168 with 4096 fallback"
        ),
    ),
)


def _parse_python(path: Path) -> ast.Module | None:
    if not path.is_file():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function_names(path: Path) -> set[str]:
    tree = _parse_python(path)
    if tree is None:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _string_keys_assigned_to(path: Path, variable_name: str) -> set[str]:
    tree = _parse_python(path)
    if tree is None:
        return set()
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def _supported_model_names(models_root: Path) -> set[str]:
    tree = _parse_python(models_root / "__init__.py")
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _surface_status(models_root: Path, surface: RequiredSurface) -> dict[str, Any]:
    module_dir = models_root / surface.module
    config_registry = module_dir / "config_registry.py"
    module_init = module_dir / "__init__.py"
    supported_models = _supported_model_names(models_root)
    config_functions = _function_names(config_registry)
    config_dict_name = f"{surface.module.replace('_v4', 'v4')}_configs"
    if surface.module == "deepseek_v4":
        config_dict_name = "deepseekv4_configs"
    elif surface.module == "qwen3":
        config_dict_name = "qwen3_configs"
    elif surface.module == "kimi_k2":
        config_dict_name = "kimi_k2_configs"
    flavors = _string_keys_assigned_to(module_init, config_dict_name)

    checks = {
        "module_dir_exists": module_dir.is_dir(),
        "module_listed_in_torchtitan_models": surface.module in supported_models,
        "config_registry_exists": config_registry.is_file(),
        "config_function_exists": surface.config_function in config_functions,
        "flavor_exists": surface.flavor is None or surface.flavor in flavors,
    }
    present = all(checks.values())
    return {
        "name": surface.name,
        "module": surface.module,
        "config_function": surface.config_function,
        "flavor": surface.flavor,
        "topology": surface.topology,
        "present": present,
        "checks": checks,
        "paths": {
            "module_dir": str(module_dir),
            "config_registry": str(config_registry),
            "module_init": str(module_init),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torchtitan-models",
        type=Path,
        default=DEFAULT_TORCHTITAN_MODELS,
        help="Path to torchtitan/torchtitan/models",
    )
    parser.add_argument(
        "--allow-missing",
        action="append",
        default=[],
        help="Surface name that is allowed to be missing for current partial gate runs.",
    )
    args = parser.parse_args()

    statuses = [_surface_status(args.torchtitan_models, s) for s in REQUIRED_SURFACES]
    allowed_missing = set(args.allow_missing)
    missing = [s["name"] for s in statuses if not s["present"]]
    unallowed_missing = [name for name in missing if name not in allowed_missing]

    payload = {
        "torchtitan_models": str(args.torchtitan_models),
        "surfaces": statuses,
        "missing": missing,
        "allowed_missing": sorted(allowed_missing),
        "unallowed_missing": unallowed_missing,
        "read": (
            "All required balanced-MoE retained perf model surfaces are present."
            if not missing
            else "Some retained perf model surfaces are missing; route-shape ABI parity is not a substitute for throughput coverage."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if unallowed_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
