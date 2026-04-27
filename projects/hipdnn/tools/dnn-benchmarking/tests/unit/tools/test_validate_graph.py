# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the dnn-validate-graph CLI tool."""

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from dnn_benchmarking.tools.validate_graph import main


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data))


def _valid_graph() -> Dict[str, Any]:
    return {
        "name": "test_graph",
        "compute_data_type": "float",
        "io_data_type": "float",
        "intermediate_data_type": "float",
        "tensors": [
            {
                "uid": 1,
                "name": "a",
                "dims": [4, 4],
                "strides": [4, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 2,
                "name": "b",
                "dims": [4, 4],
                "strides": [4, 1],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 3,
                "name": "c",
                "dims": [4, 4],
                "strides": [4, 1],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "matmul",
                "type": "MatmulAttributes",
                "compute_data_type": "float",
                "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                "outputs": {"c_tensor_uid": 3},
            }
        ],
    }


class TestValidateGraphMain:
    """Tests for validate_graph.main()."""

    def test_no_args_returns_one(self) -> None:
        with patch.object(sys, "argv", ["dnn-validate-graph"]):
            result = main()
        assert result == 1

    def test_valid_json_returns_zero(self, tmp_path: Path) -> None:
        graph_file = tmp_path / "graph.json"
        _write_json(graph_file, _valid_graph())

        with patch.object(sys, "argv", ["dnn-validate-graph", str(graph_file)]):
            result = main()
        assert result == 0

    def test_invalid_graph_returns_one(self, tmp_path: Path) -> None:
        # Missing required field: b_tensor_uid for MatmulAttributes
        bad_graph = {
            "tensors": [],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "name": "matmul",
                    "inputs": {"a_tensor_uid": 1},  # b_tensor_uid missing
                    "outputs": {"c_tensor_uid": 2},
                }
            ],
        }
        graph_file = tmp_path / "bad.json"
        _write_json(graph_file, bad_graph)

        with patch.object(sys, "argv", ["dnn-validate-graph", str(graph_file)]):
            result = main()
        assert result == 1

    def test_nonexistent_glob_returns_one(self) -> None:
        with patch.object(sys, "argv", ["dnn-validate-graph", "/nonexistent/*.json"]):
            result = main()
        assert result == 1
