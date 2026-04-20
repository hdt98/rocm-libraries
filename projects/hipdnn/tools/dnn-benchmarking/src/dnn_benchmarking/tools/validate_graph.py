# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Load and validate one or more hipDNN graph JSON files (globs accepted)."""

import glob
import sys
from pathlib import Path

from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.common.exceptions import GraphLoadError


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph.json> [<graph2.json> ...]", file=sys.stderr)
        print("  Patterns like 'graphs/*.json' are accepted.", file=sys.stderr)
        return 1

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        expanded = [Path(p) for p in glob.glob(arg, recursive=True)]
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(Path(arg))  # keep as-is so the error is reported below

    if not paths:
        print("Error: no files matched.", file=sys.stderr)
        return 1

    loader = GraphLoader()
    failed = False
    for path in paths:
        try:
            graph_json = loader.load_json(path)
            loader.validate(graph_json)
            print(f"OK: {loader.get_graph_name(graph_json)}")
        except GraphLoadError as e:
            print(f"FAIL: {path.name}: {e}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
