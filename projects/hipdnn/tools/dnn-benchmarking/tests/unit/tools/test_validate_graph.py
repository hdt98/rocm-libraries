# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the dnn-validate-graph CLI tool."""

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.tools.validate_graph import main


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data))


def _make_tarball(dest: Path, members: Dict[str, str]) -> Path:
    """Create a real .tar.gz at dest containing the given {name: content} members."""
    with tarfile.open(str(dest), "w:gz") as tf:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return dest


def _fake_tmpdir() -> MagicMock:
    """Return a mock that quacks like tempfile.TemporaryDirectory."""
    td = MagicMock(spec=tempfile.TemporaryDirectory)
    td.cleanup = MagicMock()
    return td


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

    def test_valid_graph_prints_ok(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        graph_file = tmp_path / "graph.json"
        _write_json(graph_file, _valid_graph())

        with patch.object(sys, "argv", ["dnn-validate-graph", str(graph_file)]):
            result = main()

        assert result == 0
        assert "OK:" in capsys.readouterr().out

    def test_multiple_valid_graphs_all_pass(self, tmp_path: Path) -> None:
        for name in ("a.json", "b.json", "c.json"):
            _write_json(tmp_path / name, _valid_graph())

        with patch.object(
            sys, "argv", ["dnn-validate-graph", str(tmp_path / "a.json"), str(tmp_path / "b.json"), str(tmp_path / "c.json")]
        ):
            result = main()

        assert result == 0

    def test_partial_failure_across_multiple_files_returns_one(
        self, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.json"
        bad = tmp_path / "bad.json"
        _write_json(good, _valid_graph())
        _write_json(bad, {"tensors": [], "nodes": [{"type": "MatmulAttributes", "name": "m", "inputs": {}, "outputs": {}}]})

        with patch.object(sys, "argv", ["dnn-validate-graph", str(good), str(bad)]):
            result = main()

        assert result == 1

    def test_resolve_error_on_one_arg_continues_to_next(self, tmp_path: Path) -> None:
        """A GraphLoadError from resolve_graph_files marks failure but keeps going."""
        good = tmp_path / "good.json"
        _write_json(good, _valid_graph())

        real_resolve = __import__(
            "dnn_benchmarking.graph.resolver", fromlist=["resolve_graph_files"]
        ).resolve_graph_files

        call_count = 0

        def _resolve_side_effect(arg: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise GraphLoadError("simulated tarball error")
            return real_resolve(arg)

        with patch("dnn_benchmarking.tools.validate_graph.resolve_graph_files", side_effect=_resolve_side_effect):
            with patch.object(sys, "argv", ["dnn-validate-graph", "bad.tar.gz", str(good)]):
                result = main()

        assert result == 1  # failed flag set by first arg
        assert call_count == 2  # both args were processed

    def test_all_args_fail_resolve_with_no_paths_returns_one(self) -> None:
        """When every arg raises GraphLoadError, all_paths is empty → return 1."""
        with patch(
            "dnn_benchmarking.tools.validate_graph.resolve_graph_files",
            side_effect=GraphLoadError("boom"),
        ):
            with patch.object(sys, "argv", ["dnn-validate-graph", "a.tar.gz", "b.tar.gz"]):
                result = main()

        assert result == 1

    def test_tarball_with_no_json_emits_warning(self, capsys: pytest.CaptureFixture) -> None:
        """resolve_graph_files returning tmpdirs but no paths triggers the warning."""
        td = _fake_tmpdir()

        with patch(
            "dnn_benchmarking.tools.validate_graph.resolve_graph_files",
            return_value=([td], [], "graphs.tar.gz"),
        ):
            with patch.object(sys, "argv", ["dnn-validate-graph", "graphs.tar.gz"]):
                result = main()

        assert result == 1
        assert "no .json files" in capsys.readouterr().err.lower()
        td.cleanup.assert_called_once()

    def test_tarball_with_no_json_cleans_up_tmpdir(self) -> None:
        """Temporary directories are cleaned up even when no JSON files are found."""
        td = _fake_tmpdir()

        with patch(
            "dnn_benchmarking.tools.validate_graph.resolve_graph_files",
            return_value=([td], [], "graphs.tar.gz"),
        ):
            with patch.object(sys, "argv", ["dnn-validate-graph", "graphs.tar.gz"]):
                main()

        td.cleanup.assert_called_once()

    def test_tarball_with_valid_graph_returns_zero(self, tmp_path: Path) -> None:
        tb = _make_tarball(
            tmp_path / "graphs.tar.gz",
            {"graph.json": json.dumps(_valid_graph())},
        )

        with patch.object(sys, "argv", ["dnn-validate-graph", str(tb)]):
            result = main()

        assert result == 0

    def test_tarball_tmpdir_cleaned_up_on_success(self, tmp_path: Path) -> None:
        """Temporary directories from tarball extraction are cleaned up after validation."""
        tb = _make_tarball(
            tmp_path / "graphs.tar.gz",
            {"graph.json": json.dumps(_valid_graph())},
        )

        cleaned = []

        real_resolve = __import__(
            "dnn_benchmarking.graph.resolver", fromlist=["resolve_graph_files"]
        ).resolve_graph_files

        def _spying_resolve(arg: str):
            tmpdirs, files, source = real_resolve(arg)
            for td in tmpdirs:
                original_cleanup = td.cleanup

                def _tracked_cleanup(_td=td, _orig=original_cleanup):
                    cleaned.append(_td)
                    _orig()

                td.cleanup = _tracked_cleanup
            return tmpdirs, files, source

        with patch("dnn_benchmarking.tools.validate_graph.resolve_graph_files", side_effect=_spying_resolve):
            with patch.object(sys, "argv", ["dnn-validate-graph", str(tb)]):
                result = main()

        assert result == 0
        assert len(cleaned) == 1

    def test_glob_with_no_matches_emits_warning(self, capsys: pytest.CaptureFixture) -> None:
        """resolve_graph_files returning no tmpdirs and no paths triggers the no-files warning."""
        with patch(
            "dnn_benchmarking.tools.validate_graph.resolve_graph_files",
            return_value=([], [], None),
        ):
            with patch.object(sys, "argv", ["dnn-validate-graph", "/no/match/*.json"]):
                result = main()

        assert result == 1
        assert "no files found" in capsys.readouterr().err.lower()
