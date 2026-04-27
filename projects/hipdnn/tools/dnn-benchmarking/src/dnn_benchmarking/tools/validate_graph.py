# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Load and validate one or more hipDNN graph JSON files (globs and tarballs accepted)."""

import sys
from pathlib import Path
from typing import List

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph.loader import GraphLoader
from dnn_benchmarking.graph.resolver import resolve_graph_files


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph.json|tarball|glob> [...]", file=sys.stderr)
        print(
            "  Accepts .json files, tarballs (.tar.gz etc.), and glob patterns.",
            file=sys.stderr,
        )
        return 1

    all_tmpdirs = []
    all_paths: List[Path] = []
    failed = False
    for arg in sys.argv[1:]:
        try:
            tmpdirs, resolved, _ = resolve_graph_files(arg)
        except GraphLoadError as e:
            print(f"Error: {e}", file=sys.stderr)
            failed = True
            continue
        all_tmpdirs.extend(tmpdirs)
        if resolved:
            all_paths.extend(Path(p) for p in resolved)
        elif tmpdirs:
            # A tarball was opened but contained no JSON files.
            print(
                f"Warning: tarball '{arg}' contains no .json files",
                file=sys.stderr,
            )
        else:
            print(f"Warning: no files found for '{arg}'", file=sys.stderr)

    if not all_paths:
        print("Error: no files matched.", file=sys.stderr)
        for td in all_tmpdirs:
            td.cleanup()
        return 1

    loader = GraphLoader()
    try:
        for path in all_paths:
            try:
                graph_json = loader.load_json(path)
                loader.validate(graph_json)
                print(f"OK: {loader.get_graph_name(graph_json)}")
            except GraphLoadError as e:
                print(f"FAIL: {path.name}: {e}", file=sys.stderr)
                failed = True
    finally:
        for td in all_tmpdirs:
            td.cleanup()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
