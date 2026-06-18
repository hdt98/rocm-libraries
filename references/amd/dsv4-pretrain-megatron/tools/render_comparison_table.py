#!/usr/bin/env python3
"""Render correctness/perf comparison measurements as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROWS = [
    {
        "implementation": "megatron_miles_pr1300_async_rl_train",
        "recipe": "references/amd/miles-deepseek-v4-pr1300/scripts/amd/run_deepseek_v4.py",
        "max_abs_logprob_diff": None,
        "mean_abs_logprob_diff": None,
        "tflops_per_gpu": None,
        "status": "not_run",
        "notes": "Reference Miles async RL recipe; use actor/train logprob export for correctness.",
    },
    {
        "implementation": "megatron_pretrain_dsv4_reduced_12l",
        "recipe": "references/amd/dsv4-pretrain-megatron/run_pretrain_dsv4_megatron.sh",
        "max_abs_logprob_diff": None,
        "mean_abs_logprob_diff": None,
        "tflops_per_gpu": None,
        "status": "not_run",
        "notes": "Pretraining-only Megatron recipe using PR1300 DSV4 model args.",
    },
    {
        "implementation": "torchtitan_graph_trainer_dsv4_reduced_12l",
        "recipe": "projects/torchtitan/examples/dsv4_pretrain/run_dsv4_reduced_12l.sh",
        "max_abs_logprob_diff": None,
        "mean_abs_logprob_diff": None,
        "tflops_per_gpu": None,
        "status": "not_run",
        "notes": "TorchTitan DeepSeek MLA/MoE baseline with sqrtsoftplus routing.",
    },
]


def _fmt(value: Any) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    return [payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    if args.measurements:
        for path in args.measurements:
            rows.extend(_load_json(path))
    else:
        rows = DEFAULT_ROWS

    header = (
        "| Implementation | Recipe | Max logprob diff | Mean logprob diff | "
        "TFLOP/s/GPU | Status | Notes |"
    )
    sep = "|---|---|---:|---:|---:|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {implementation} | `{recipe}` | {max_diff} | {mean_diff} | {tflops} | {status} | {notes} |".format(
                implementation=row.get("implementation", ""),
                recipe=row.get("recipe", row.get("candidate", "")),
                max_diff=_fmt(row.get("max_abs_logprob_diff")),
                mean_diff=_fmt(row.get("mean_abs_logprob_diff")),
                tflops=_fmt(row.get("tflops_per_gpu")),
                status=row.get("status", ""),
                notes=row.get("notes", ""),
            )
        )
    table = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
