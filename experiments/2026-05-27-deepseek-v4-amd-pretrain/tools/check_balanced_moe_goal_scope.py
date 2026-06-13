#!/usr/bin/env python3
"""Check that local diffs stay inside the native balanced-MoE goal scope.

The final branch should be small enough to review as backend-native
balanced-MoE work:

* MORI owns a native balanced_moe feature.
* Primus-Turbo owns an equivalent native balanced_moe feature.
* TorchTitan owns only policy/layer integration.
* tools/ owns only small validation harnesses.

This tool is intentionally conservative. It reports paths outside that contract
so they can be removed, split out, or explicitly justified before promotion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[4]
EXP = "sdks/mlops/playground/experiments/2026-05-27-deepseek-v4-amd-pretrain"
NESTED_REPOS = (
    f"{EXP}/sources/wip/primus-turbo",
)

ALLOWED_EXACT = {
    f"{EXP}/sources/references/mori/python/mori/ops/balanced_moe.py",
    f"{EXP}/sources/references/mori/tests/python/ops/test_balanced_moe.py",
    f"{EXP}/sources/wip/primus-turbo/primus_turbo/pytorch/ops/moe/balanced_moe.py",
    f"{EXP}/sources/wip/primus-turbo/primus_turbo/pytorch/ops/moe/__init__.py",
    f"{EXP}/sources/wip/primus-turbo/tests/pytorch/ops/test_balanced_moe.py",
    f"{EXP}/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/common/moe.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/common/token_dispatcher.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/__init__.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/kimi_k2/__init__.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/kimi_k2/attention.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/kimi_k2/config_registry.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/qwen3/__init__.py",
    f"{EXP}/sources/wip/torchtitan/torchtitan/models/qwen3/config_registry.py",
    f"{EXP}/tools/check_balanced_moe_backend_parity.py",
    f"{EXP}/tools/check_balanced_moe_goal_scope.py",
    f"{EXP}/tools/check_balanced_moe_model_shapes.py",
    f"{EXP}/tools/check_balanced_moe_model_surfaces.py",
    f"{EXP}/tools/mori_hot_helper_pack_smoke.py",
    f"{EXP}/tools/run_balanced_moe_table_sweep.py",
    f"{EXP}/tools/smoke_mori_owner_compact_exchange_gloo.py",
    f"{EXP}/tools/README.md",
}

ALLOWED_CHANGESETS = {
    f"{EXP}/changesets/balanced_moe_backend_abi_parity_20260612.md",
    f"{EXP}/changesets/balanced_moe_backend_parity_harness_20260612.md",
    f"{EXP}/changesets/balanced_moe_cross_model_retained_perf_gates_20260612.md",
    f"{EXP}/changesets/balanced_moe_runtime_layout_abi_20260612.md",
    f"{EXP}/changesets/mori_native_balanced_moe_owner_compact_exchange_autograd_20260612.md",
    f"{EXP}/changesets/mori_native_balanced_moe_owner_compact_exchange_tensor_abi_20260612.md",
    f"{EXP}/changesets/mori_native_balanced_moe_planner_api_20260611.md",
    f"{EXP}/changesets/mori_native_balanced_moe_source_partition_api_20260611.md",
    f"{EXP}/changesets/mori_primus_balanced_moe_raw_topk_backend_api_20260612.md",
    f"{EXP}/changesets/primus_turbo_native_balanced_moe_planner_abi_20260612.md",
    f"{EXP}/changesets/torchtitan_balanced_moe_backend_runtime_handoff_20260612.md",
    f"{EXP}/changesets/torchtitan_balanced_moe_backend_runtime_layout_20260612.md",
    f"{EXP}/changesets/torchtitan_balanced_moe_backend_topk_planner_handoff_20260612.md",
    f"{EXP}/changesets/torchtitan_balanced_moe_planner_backend_selector_20260612.md",
    f"{EXP}/changesets/torchtitan_balanced_moe_runtime_layout_handoff_20260612.md",
    f"{EXP}/changesets/torchtitan_mori_balanced_runtime_layout_gate_20260612.md",
    f"{EXP}/changesets/torchtitan_mori_native_balanced_needed_exchange_plan_20260612.md",
    f"{EXP}/changesets/torchtitan_mori_native_needed_tensor_abi_consumer_20260612.md",
    f"{EXP}/changesets/torchtitan_mori_native_weighted_combine_abi_gap_20260612.md",
    f"{EXP}/changesets/torchtitan_primus_turbo_deepep_viability_20260612.md",
    f"{EXP}/changesets/torchtitan_standard_ep_hothelper_mbs1_mbs2_ce8_ladder_20260612.md",
}


def _git(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args],
        text=True,
    )


def _git_optional(args: list[str], *, cwd: Path = REPO_ROOT) -> str:
    try:
        return _git(args, cwd=cwd)
    except subprocess.CalledProcessError:
        return ""


def _prefixed_status_paths(prefix: str, out: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for line in out.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths[f"{prefix}/{path}"] = status
    return paths


def _nested_repo_path(prefix: str) -> Path:
    return REPO_ROOT / prefix


def _branch_diff_paths(base: str) -> set[str]:
    out = _git(["diff", "--name-only", f"{base}...HEAD"])
    paths = {line.strip() for line in out.splitlines() if line.strip()}
    for prefix in NESTED_REPOS:
        nested_root = _nested_repo_path(prefix)
        if not (nested_root / ".git").exists():
            continue
        nested_out = _git_optional(
            ["diff", "--name-only", f"{base}...HEAD"],
            cwd=nested_root,
        )
        paths.update(
            f"{prefix}/{line.strip()}"
            for line in nested_out.splitlines()
            if line.strip()
        )
    return paths


def _dirty_status_paths() -> dict[str, str]:
    out = _git(["status", "--short", "-uall"])
    paths = _prefixed_status_paths("", out)
    paths = {path.removeprefix("/"): status for path, status in paths.items()}
    for prefix in NESTED_REPOS:
        # The parent repo may report the nested checkout as one dirty directory.
        # Drop that duplicate marker because the nested repo status below gives
        # the reviewable file-level scope result.
        paths.pop(prefix, None)
        paths.pop(f"{prefix}/", None)
        nested_root = _nested_repo_path(prefix)
        if not (nested_root / ".git").exists():
            continue
        nested_out = _git(["status", "--short", "-uall"], cwd=nested_root)
        paths.update(_prefixed_status_paths(prefix, nested_out))
    return paths


def _is_allowed(path: str, *, allow_changesets: bool) -> bool:
    if path in ALLOWED_EXACT:
        return True
    if allow_changesets and path in ALLOWED_CHANGESETS:
        return True
    return False


def _partition(paths: Iterable[str], *, allow_changesets: bool) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    out_of_scope: list[str] = []
    for path in sorted(paths):
        if _is_allowed(path, allow_changesets=allow_changesets):
            allowed.append(path)
        else:
            out_of_scope.append(path)
    return allowed, out_of_scope


def _build_report(
    *,
    base: str,
    include_dirty: bool,
    allow_changesets: bool,
    dirty_status: dict[str, str],
) -> dict[str, object]:
    committed_paths = _branch_diff_paths(base)
    dirty_paths = set(dirty_status) if include_dirty else set()
    combined_paths = committed_paths | dirty_paths

    committed_allowed, committed_out = _partition(
        committed_paths,
        allow_changesets=allow_changesets,
    )
    dirty_allowed, dirty_out = _partition(
        dirty_paths,
        allow_changesets=allow_changesets,
    )
    combined_allowed, combined_out = _partition(
        combined_paths,
        allow_changesets=allow_changesets,
    )

    return {
        "base": base,
        "include_dirty": include_dirty,
        "allow_changesets": allow_changesets,
        "summary": {
            "committed": {
                "allowed": len(committed_allowed),
                "out_of_scope": len(committed_out),
                "total": len(committed_paths),
            },
            "dirty": {
                "allowed": len(dirty_allowed),
                "out_of_scope": len(dirty_out),
                "total": len(dirty_paths),
            },
            "combined": {
                "allowed": len(combined_allowed),
                "out_of_scope": len(combined_out),
                "total": len(combined_paths),
            },
        },
        "committed": {
            "allowed": committed_allowed,
            "out_of_scope": committed_out,
        },
        "dirty": {
            "allowed": dirty_allowed,
            "out_of_scope": dirty_out,
            "status": {path: dirty_status[path] for path in sorted(dirty_paths)},
        },
        "combined": {
            "allowed": combined_allowed,
            "out_of_scope": combined_out,
        },
    }


def _print_section(title: str, section: dict[str, object], *, max_print: int) -> None:
    allowed = list(section["allowed"])  # type: ignore[index]
    out_of_scope = list(section["out_of_scope"])  # type: ignore[index]
    print(f"{title}: allowed={len(allowed)} out_of_scope={len(out_of_scope)}")
    for path in allowed[:max_print]:
        print(f"  OK {path}")
    if len(allowed) > max_print:
        print(f"  ... {len(allowed) - max_print} more allowed")
    for path in out_of_scope[:max_print]:
        print(f"  OUT {path}")
    if len(out_of_scope) > max_print:
        print(f"  ... {len(out_of_scope) - max_print} more out-of-scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="main",
        help="Local base branch used for committed branch diff checks.",
    )
    parser.add_argument(
        "--no-dirty",
        action="store_true",
        help="Skip working-tree dirty/untracked files and check only base...HEAD.",
    )
    parser.add_argument(
        "--strict-code-only",
        action="store_true",
        help="Reject changesets too; useful for final upstream code PR prep.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=80,
        help="Maximum out-of-scope paths to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Write the full report to this JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allow_changesets = not args.strict_code_only
    dirty_status = _dirty_status_paths()
    report = _build_report(
        base=args.base,
        include_dirty=not args.no_dirty,
        allow_changesets=allow_changesets,
        dirty_status=dirty_status,
    )

    if args.write_json is not None:
        args.write_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "balanced-MoE goal scope report "
            f"(base={args.base}, include_dirty={not args.no_dirty}, "
            f"allow_changesets={allow_changesets})"
        )
        _print_section(
            "committed main...HEAD",
            report["committed"],  # type: ignore[arg-type]
            max_print=args.max_print,
        )
        if not args.no_dirty:
            _print_section(
                "dirty working tree",
                report["dirty"],  # type: ignore[arg-type]
                max_print=args.max_print,
            )
        _print_section(
            "combined",
            report["combined"],  # type: ignore[arg-type]
            max_print=args.max_print,
        )

    combined = report["combined"]  # type: ignore[assignment]
    if combined["out_of_scope"]:  # type: ignore[index]
        raise SystemExit(1)

    if not args.json:
        print("balanced-MoE goal scope OK")


if __name__ == "__main__":
    main()
