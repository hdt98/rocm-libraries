#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
"""Generate appendix statistics for docs/testing/rocblas_hipblaslt_test_parity_catalog.md

Counts top-level entries under YAML ``Tests:`` in each included file (templates before
cartesian expansion). This is a lightweight proxy for suite size, not exact gtest case count.

Usage (from anywhere):
  python3 generate_hipblaslt_parity_appendix.py
  python3 generate_hipblaslt_parity_appendix.py --json stats.json --no-appendix-stdout
  python3 generate_hipblaslt_parity_appendix.py --parity-table-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

INCLUDE_RE = re.compile(r"^\s*include\s*:\s*([-.\w/]+)\s*$", re.MULTILINE)


def _repo_rocblas_gtest_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "clients" / "gtest"


def _repo_hipblaslt_data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "hipblaslt" / "clients" / "tests" / "data"


def collect_includes(entry: Path, base: Path, seen: set[Path]) -> list[Path]:
    """Depth-first list of YAML files reachable from entry via ``include:`` lines."""
    entry = entry.resolve()
    if entry in seen:
        return []
    seen.add(entry)
    out: list[Path] = [entry]
    text = entry.read_text(encoding="utf-8", errors="replace")
    for m in INCLUDE_RE.finditer(text):
        name = m.group(1)
        if not name.endswith(".yaml"):
            name += ".yaml"
        child = base / name
        if child.is_file():
            out.extend(collect_includes(child, base, seen))
        else:
            # hipBLASlt uses includes without path; all live in same data dir
            alt = entry.parent / name
            if alt.is_file():
                out.extend(collect_includes(alt, alt.parent, seen))
    return out


TEST_START_RE = re.compile(
    r"^\s*-\s+(?:name:|\{\s*name:)", flags=re.MULTILINE
)


def count_tests_in_file(path: Path) -> int:
    """Count test *templates* after ``Tests:`` (rocBLAS YAML uses cross-file anchors)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^Tests:\s*(?:$|#)", line):
            start = i + 1
            break
    if start is None:
        return 0
    body = "\n".join(lines[start:])
    return len(TEST_START_RE.findall(body))


def appendix_for_rocblas() -> tuple[str, dict[str, Any]]:
    gtest = _repo_rocblas_gtest_dir()
    root = gtest / "rocblas_gtest.yaml"
    seen: set[Path] = set()
    files = collect_includes(root, gtest, seen)
    rows = []
    total = 0
    per_file: dict[str, int] = {}
    for f in sorted(set(files), key=lambda p: str(p)):
        c = count_tests_in_file(f)
        rel = f.relative_to(gtest)
        per_file[str(rel)] = c
        total += c
        rows.append((str(rel), c))

    lines = [
        "## Appendix: rocBLAS YAML test templates (rocblas_gtest.yaml closure)",
        "",
        "Each number counts lines after `Tests:` that start a test template: `- name:` or "
        "`- {name: ...` (before `rocblas_gentest.py` expands cross-products). "
        "YAML anchor merge is not evaluated; per-file counts are intrinsic to each file.",
        "",
        f"| YAML file | Test templates |",
        f"|-----------|----------------:|",
    ]
    for rel, c in rows:
        lines.append(f"| `{rel}` | {c} |")
    lines.append(f"| **Sum** | **{total}** |")
    lines.append("")

    meta = {"rocblas_gtest_yaml_closure": per_file, "rocblas_tests_entry_sum": total}
    return "\n".join(lines), meta


def appendix_for_hipblaslt() -> tuple[str, dict[str, Any]]:
    data = _repo_hipblaslt_data_dir()
    if not data.is_dir():
        return (
            "## Appendix: hipBLASLt YAML (hipblaslt tree not found next to rocblas)\n",
            {},
        )
    root = data / "hipblaslt_gtest.yaml"
    seen: set[Path] = set()
    files = collect_includes(root, data, seen)
    rows = []
    total = 0
    per_file: dict[str, int] = {}
    for f in sorted(set(files), key=lambda p: str(p)):
        c = count_tests_in_file(f)
        rel = f.relative_to(data)
        per_file[str(rel)] = c
        total += c
        rows.append((str(rel), c))

    lines = [
        "## Appendix: hipBLASLt YAML test templates (hipblaslt_gtest.yaml closure)",
        "",
        "Same counting convention as rocBLAS appendix.",
        "",
        f"| YAML file | Test templates |",
        f"|-----------|----------------:|",
    ]
    for rel, c in rows:
        lines.append(f"| `{rel}` | {c} |")
    lines.append(f"| **Sum** | **{total}** |")
    lines.append("")

    meta = {"hipblaslt_gtest_yaml_closure": per_file, "hipblaslt_tests_entry_sum": total}
    return "\n".join(lines), meta


CMAKE_GTEST = Path(__file__).resolve().parent.parent / "clients" / "gtest" / "CMakeLists.txt"

# Stems (``foo`` from ``foo_gtest.cpp``) whose rocBLAS tests exercise the Tensile contraction
# entry point ``runContractionProblem`` / ``gemm_tensile.hpp`` (hipBLASLt may run first).
_GEMM_CONTRACTION_STEMS = frozenset(
    {
        "gemm",
        "gemm_ex",
        "get_solutions",
        "multiheaded",
        "atomics_mode",
    }
)

_TENSILE_ONLY_CPP = frozenset(
    {
        "multiheaded_gtest.cpp",
        "atomics_mode_gtest.cpp",
        "get_solutions_gtest.cpp",
    }
)

_NO_YAML_CPP = frozenset(
    {
        "rocblas_gtest_main.cpp",
        "rocblas_test.cpp",
        "asan_helpers_gtest.cpp",
    }
)

_YAML_STEM_OVERRIDES = {
    "iamaxmin": "iamax_iamin",
    "geru": "geruc",
    "gerc": "geruc",
}

# ``blas_ex/*_ex_gtest.cpp`` sources often share the Level-1 YAML driver for the base routine.
_PRIMARY_YAML_BY_CPP: dict[str, str] = {
    "blas_ex/axpy_ex_gtest.cpp": "axpy_gtest.yaml",
    "blas_ex/dot_ex_gtest.cpp": "dot_gtest.yaml",
    "blas_ex/nrm2_ex_gtest.cpp": "nrm2_gtest.yaml",
    "blas_ex/rot_ex_gtest.cpp": "rot_gtest.yaml",
    "blas_ex/scal_ex_gtest.cpp": "scal_gtest.yaml",
}


def _parse_gtest_cpp_from_cmake() -> list[str]:
    text = CMAKE_GTEST.read_text(encoding="utf-8")
    # Strip CMake `#` comments so we do not match `foo_gtest.cpp` inside comment text.
    stripped = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    found = re.findall(r"([\w./]+_gtest\.cpp)", stripped)
    seen: set[str] = set()
    ordered: list[str] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    # `rocblas_gtest_main.cpp` / `rocblas_test.cpp` are not `*_gtest.cpp`; mirror CMake order
    # (after Tensile-only sources, before `asan_helpers_gtest.cpp`).
    harness = ["rocblas_gtest_main.cpp", "rocblas_test.cpp"]
    for i, f in enumerate(ordered):
        if f == "asan_helpers_gtest.cpp":
            return ordered[:i] + harness + ordered[i:]
    return ordered + harness


def _stem_from_cpp(cpp: str) -> str:
    stem = Path(cpp).stem
    if stem.endswith("_gtest"):
        return stem[: -len("_gtest")]
    return stem


def _yaml_name_for_cpp(cpp: str) -> str | None:
    if cpp in _NO_YAML_CPP:
        return None
    if cpp in _PRIMARY_YAML_BY_CPP:
        return _PRIMARY_YAML_BY_CPP[cpp]
    st = _stem_from_cpp(cpp)
    yst = _YAML_STEM_OVERRIDES.get(st, st)
    return f"{yst}_gtest.yaml"


def _category_for_cpp(cpp: str) -> str:
    if cpp in _TENSILE_ONLY_CPP:
        return "Tensile / GEMM host"
    if cpp in _NO_YAML_CPP:
        return "Client harness / infra"
    if cpp.startswith("blas1/"):
        return "BLAS 1"
    if cpp.startswith("blas2/"):
        return "BLAS 2"
    if cpp.startswith("blas3/"):
        return "BLAS 3"
    if cpp.startswith("blas_ex/"):
        return "BLAS extension"
    return "Client / infra"


def _apis_hint(cpp: str) -> str:
    st = _stem_from_cpp(cpp)
    if cpp in _NO_YAML_CPP:
        if cpp == "rocblas_gtest_main.cpp":
            return "gtest main"
        if cpp == "rocblas_test.cpp":
            return "shared test fixtures"
        return "ASan helpers"
    if st == "multiheaded":
        return "Multi-handle / Tensile host init (GEMM-related)"
    if st == "atomics_mode":
        return "Atomics mode + GEMM path"
    if st == "get_solutions":
        return "`rocblas_get_*_solutions` / solution indices"
    return f"`rocblas_*{st}*` (generated)"


def _hipblaslt_today(stem: str, cpp: str) -> str:
    if stem in _GEMM_CONTRACTION_STEMS:
        return (
            "Yes* — hipBLASLt tried in `runContractionProblem` when "
            "`BUILD_WITH_HIPBLASLT`, not Tensile-only solution index, "
            "`useHipBLASLt` / `tryHipBLASLt` (see legend)"
        )
    if stem == "logging_mode":
        return "Partial — `logging_mode_internal` targets hipBLASLt backend logging on matching arch"
    return "No — not `gemm_tensile` / contraction dispatch"


def _hypothetical(stem: str) -> str:
    if stem in _GEMM_CONTRACTION_STEMS:
        return "In-scope — GEMM-family / contraction"
    return "Out-of-scope — not hipBLASLt matmul unless rocBLAS routes here"


def _hipblaslt_analog(stem: str, cpp: str) -> str:
    if stem in _GEMM_CONTRACTION_STEMS:
        if stem == "get_solutions":
            return "`matmul_gtest.yaml` (heuristic / algo) — partial parity"
        if stem in ("multiheaded", "atomics_mode"):
            return "Gap — multi-handle / atomics + Tensile host"
        return "`matmul_gtest.yaml` core matmul; mixed precision & epilogues vary"
    if cpp in _NO_YAML_CPP and stem in ("general", "set_get_pointer_mode", "set_get_atomics_mode"):
        return "`auxiliary_gtest.yaml` — partial"
    return "Gap — no direct matmul analog"


def emit_parity_suite_table() -> str:
    gtest_dir = _repo_rocblas_gtest_dir()
    lines = [
        "| rocBLAS module | Primary YAML | Category | APIs (hint) | YAML-backed | Tensile-only build | hipBLASLt today | Hypothetical LT delegation | hipBLASLt analog | Notes |",
        "|---|---|---|---|:---:|:---:|---|---|---|---|",
    ]
    for cpp in _parse_gtest_cpp_from_cmake():
        yname = _yaml_name_for_cpp(cpp)
        ycell = f"`{yname}`" if yname else "—"
        yaml_backed = "Y" if yname and (gtest_dir / yname).is_file() else ("N" if yname else "—")
        tensile_only = "Y" if cpp in _TENSILE_ONLY_CPP else "—"
        stem = _stem_from_cpp(cpp)
        notes = ""
        if cpp in _TENSILE_ONLY_CPP:
            notes = "Linked only when `BUILD_WITH_TENSILE`"
        elif cpp in _PRIMARY_YAML_BY_CPP:
            notes = "`*_ex` cases live in the Level-1 driver YAML"
        elif stem == "gemm_ex":
            notes = "Includes mixed precision / `_ex` paths"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{cpp}`",
                    ycell,
                    _category_for_cpp(cpp),
                    _apis_hint(cpp),
                    yaml_backed,
                    tensile_only,
                    _hipblaslt_today(stem, cpp),
                    _hypothetical(stem),
                    _hipblaslt_analog(stem, cpp),
                    notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="Write combined statistics JSON for tooling / agents.",
    )
    parser.add_argument(
        "--parity-table-only",
        action="store_true",
        help="Print only the suite-level markdown table (Part B rows).",
    )
    parser.add_argument(
        "--no-appendix-stdout",
        action="store_true",
        help="With --json, do not print appendix tables to stdout.",
    )
    args = parser.parse_args()

    if args.parity_table_only:
        print(emit_parity_suite_table())
        return 0

    a1, m1 = appendix_for_rocblas()
    a2, m2 = appendix_for_hipblaslt()
    if not (args.json and args.no_appendix_stdout):
        print(a1)
        print()
        print(a2)

    if args.json:
        out = {**m1, **m2}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
