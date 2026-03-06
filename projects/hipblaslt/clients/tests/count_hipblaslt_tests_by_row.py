#!/usr/bin/env python3
"""
Count hipBLASLt tests by crosstab row from the test generator (no C++ binary).

Run the generator with --list to get category<tab>name lines, then map names to
the operation-group rows in hipblaslt_test_coverage_crosstab.md and count.

Usage:
  # From clients/tests, run generator and count (full suite):
  python count_hipblaslt_tests_by_row.py

  # Filter by category (quick or smoke):
  python count_hipblaslt_tests_by_row.py --filter quick
  python count_hipblaslt_tests_by_row.py --filter smoke

  # Read pre-generated list from file:
  python hipblaslt_gentest.py -t data/hipblaslt_common.yaml data/hipblaslt_gtest.yaml --list > /tmp/list.txt
  python count_hipblaslt_tests_by_row.py --list-file /tmp/list.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Rows in the crosstab (matmul only). Each row has a list of patterns:
# - string: exact match on test name
# - compiled regex: match name with regex
# - (str, bool): (pattern, is_prefix) for startswith
ROW_PATTERNS = [
    (
        "core / bad-arg / NaN",
        [
            "matmul_bad_arg",
            "alpha_beta_zero_NaN",
            "matmul_one",
            "matmul_small",
            "matmul_conj_small",
            "matmul_medium",
            "matmul_batch_medium",
            "matmul_medium_HMM",
            "matmul_chunk",
        ],
    ),
    (
        "sizes / fixed shapes",
        [
            "matmul_8",
            "matmul_16",
            "matmul_24",
            "matmul_32_8_128",
            "matmul_48_8_128",
            "matmul_64_8_128",
            "matmul_64_8",
            "matmul_8_64",
            "matmul_96",
            "matmul_128",
            "matmul_128_streamk",
            "matmul_256",
            "matmul_256_8_16",
            "matmul_16_256_8",
            "matmul_8_16_256",
            "matmul_512",
            "matmul_1024",
            "matmul_small2",
            "matmul_k0",
        ],
    ),
    (
        "algo / heuristic / MX",
        [
            ("matmul_algo", True),
            ("matmul_heuristic_", True),
            ("matmul_mx_", True),
        ],
    ),
    (
        "stress / nightly shapes",
        [
            ("matmul_grid_limit_", True),
            "matmul_deepbench",
            ("resnet50_", True),
            ("inception4_", True),
            ("ctest_", True),
            ("matmul_large_nt_", True),
            ("matmul_conv3d_", True),
            "matmul_stride_lt_ld",
        ],
    ),
    (
        "bias (relu, gelu, swish, …)",
        [
            "matmul_bias_relu",
            "matmul_bias_sigmoid",
            "matmul_bias_gelu",
            "matmul_bias_swish",
            "matmul_bias_clamp",
            "matmul_bias_only",
            "matmul_bias_type",
        ],
    ),
    (
        "gradients (dgelu, drelu, bgradb)",
        [
            ("matmul_dgelu_", True),
            ("matmul_drelu_", True),
            ("matmul_bgradb_", True),
        ],
    ),
    (
        "bias_gelu_aux / equality",
        [
            ("matmul_bias_gelu_aux_", True),
            ("matmul_equality_NN_bias_", True),
            "matmul_relu_clamp_useE",
        ],
    ),
    (
        "grouped gemm",
        [
            "matmul_groupedgemm",
            ("matmul_groupedgemm_specific_", True),
            ("matmul_groupedgemm_f8_", True),
            "matmul_extapi_groupedgemm",
        ],
    ),
    (
        "extended API (algo, swizzle)",
        [
            "matmul_extapi_algo_method_gemm",
            "matmul_extapi_gemm",
            ("matmul_extapi_algo_method_tuning_", True),
            ("matmul_extapi_swizzle", True),
            ("matmul_swizzle", True),
        ],
    ),
]


def name_matches_row(name: str, patterns: list) -> bool:
    for p in patterns:
        if isinstance(p, tuple):
            prefix, _ = p
            if name.startswith(prefix):
                return True
        elif isinstance(p, str):
            # Exact match or name with suffix (e.g. matmul_one_bf16_...)
            if name == p or name.startswith(p + "_"):
                return True
    return False


def assign_row(name: str) -> str | None:
    """Return the crosstab row label for this test name, or None if not matmul."""
    for row_label, patterns in ROW_PATTERNS:
        if name_matches_row(name, patterns):
            return row_label
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count hipBLASLt tests by crosstab row from generator --list output",
    )
    ap.add_argument(
        "--list-file",
        type=argparse.FileType("r"),
        default=None,
        help="Read category<tab>name from this file (default: run generator)",
    )
    ap.add_argument(
        "--filter",
        choices=("quick", "smoke"),
        default=None,
        help="Only count tests with this category (for quick/smoke tables)",
    )
    ap.add_argument(
        "--gentest-dir",
        type=Path,
        default=None,
        help="Path to clients/tests (default: script dir)",
    )
    args = ap.parse_args()

    if args.list_file is not None:
        lines = args.list_file.read().splitlines()
        args.list_file.close()
    else:
        script_dir = Path(__file__).resolve().parent
        gentest_dir = args.gentest_dir or script_dir
        data_dir = gentest_dir / "data"
        gentest_py = gentest_dir / "hipblaslt_gentest.py"
        common_yaml = data_dir / "hipblaslt_common.yaml"
        main_yaml = data_dir / "hipblaslt_gtest.yaml"
        if not gentest_py.exists() or not main_yaml.exists():
            print(
                "Run from repo with clients/tests layout, or pass --list-file.",
                file=sys.stderr,
            )
            return 1
        result = subprocess.run(
            [
                sys.executable,
                str(gentest_py),
                "-t",
                str(common_yaml),
                str(main_yaml),
                "--list",
            ],
            cwd=str(gentest_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            return 1
        lines = result.stdout.splitlines()

    # Parse category\tname
    counts: dict[str, int] = {row: 0 for row, _ in ROW_PATTERNS}
    other = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        category = parts[0] if len(parts) > 0 else ""
        name = parts[1] if len(parts) > 1 else ""
        if args.filter == "quick" and "quick" not in category:
            continue
        if args.filter == "smoke" and "smoke" not in category:
            continue
        row = assign_row(name)
        if row is not None:
            counts[row] += 1
        else:
            other += 1

    # Print table
    print(f"Filter: {args.filter or 'full'}")
    print()
    for row_label, _ in ROW_PATTERNS:
        print(f"  {row_label}: {counts[row_label]}")
    print(f"  (other / aux / rocroller): {other}")
    print()
    print(f"  Total (matmul rows): {sum(counts.values())}")
    print(f"  Total (all): {sum(counts.values()) + other}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
