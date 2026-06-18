#!/usr/bin/env python3
"""Compare implementation logprobs against the PyTorch oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_logprobs(path: Path) -> torch.Tensor:
    payload: Any = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, dict):
        for key in ("logprobs", "log_probs", "per_token_logprobs"):
            if key in payload:
                return payload[key].float()
    raise KeyError(f"{path} does not contain a logprobs tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tflops-per-gpu", type=float)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle = _load_logprobs(args.oracle)
    candidate = _load_logprobs(args.candidate)
    if oracle.shape != candidate.shape:
        raise ValueError(f"shape mismatch: oracle={tuple(oracle.shape)} candidate={tuple(candidate.shape)}")

    diff = candidate - oracle
    abs_diff = diff.abs()
    metrics = {
        "implementation": args.implementation,
        "oracle": str(args.oracle),
        "candidate": str(args.candidate),
        "tokens_scored": int(diff.numel()),
        "max_abs_logprob_diff": float(abs_diff.max().item()),
        "mean_abs_logprob_diff": float(abs_diff.mean().item()),
        "rms_logprob_diff": float(torch.sqrt((diff * diff).mean()).item()),
        "tflops_per_gpu": args.tflops_per_gpu,
        "status": "measured",
        "notes": args.notes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
