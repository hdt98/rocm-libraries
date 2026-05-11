# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from importlib.resources import files


def sample_graphs_path() -> str:
    """Return the path to the bundled sample graphs directory."""
    return str(files("dnn_benchmarking.graphs"))
