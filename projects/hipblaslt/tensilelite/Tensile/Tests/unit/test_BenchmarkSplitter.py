################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile.BenchmarkSplitter import BenchmarkSplitter


class TestBenchmarkSplitterSplitByProblem:
    """Tests for BenchmarkSplitter.__splitByProblem static method."""

    def test_splits_single_problem_into_separate_config(self):
        """Test splits single problem into its own config."""
        data = {
            "GlobalParameters": {"Key": "Value"},
            "BenchmarkProblems": [
                [{"Problem": 1}, {"Data": 1}]
            ]
        }

        # Access private method for testing
        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(data)

        assert len(result) == 1
        assert result[0]["GlobalParameters"] == {"Key": "Value"}
        assert len(result[0]["BenchmarkProblems"]) == 1
        assert result[0]["BenchmarkProblems"][0] == [{"Problem": 1}, {"Data": 1}]

    def test_splits_multiple_problems_into_separate_configs(self):
        """Test splits multiple problems into separate configs."""
        data = {
            "GlobalParameters": {"Key": "Value"},
            "BenchmarkProblems": [
                [{"Problem": 1}, {"Data": 1}],
                [{"Problem": 2}, {"Data": 2}],
                [{"Problem": 3}, {"Data": 3}]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(data)

        assert len(result) == 3
        # Each result should have same GlobalParameters
        for i, config in enumerate(result):
            assert config["GlobalParameters"] == {"Key": "Value"}
            assert len(config["BenchmarkProblems"]) == 1
            assert config["BenchmarkProblems"][0][0]["Problem"] == i + 1

    def test_preserves_other_keys_in_split(self):
        """Test preserves keys other than BenchmarkProblems."""
        data = {
            "GlobalParameters": {"Key": "Value"},
            "OtherKey": "OtherValue",
            "AnotherKey": [1, 2, 3],
            "BenchmarkProblems": [
                [{"Problem": 1}, {}]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(data)

        assert len(result) == 1
        assert result[0]["OtherKey"] == "OtherValue"
        assert result[0]["AnotherKey"] == [1, 2, 3]

    def test_deep_copies_data(self):
        """Test that split creates deep copies, not references."""
        data = {
            "GlobalParameters": {"Key": ["MutableList"]},
            "BenchmarkProblems": [
                [{"Problem": {"nested": "dict"}}, {}],
                [{"Problem": {"nested": "dict"}}, {}]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(data)

        # Modify one result
        result[0]["GlobalParameters"]["Key"].append("Modified")
        result[0]["BenchmarkProblems"][0][0]["Problem"]["nested"] = "modified"

        # Other result should be unchanged
        assert "Modified" not in result[1]["GlobalParameters"]["Key"]
        assert result[1]["BenchmarkProblems"][0][0]["Problem"]["nested"] == "dict"


class TestBenchmarkSplitterSplitByBenchmarkGroup:
    """Tests for BenchmarkSplitter.__splitByBenchmarkGroup static method."""

    def test_splits_into_benchmark_groups(self):
        """Test splits single problem into benchmark groups."""
        data = {
            "GlobalParameters": {},
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM"},  # Problem type
                    {"BenchmarkCommonParameters": [{"Key": "Value"}]},  # Common params
                    {"Param1": [1, 2]},  # Grouping 1
                    {"Param2": [3, 4]}   # Grouping 2
                ]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(data)

        # Should create separate config for each grouping (common params + each param group)
        assert len(result) >= 2  # At least the parameter groups

    def test_asserts_on_multiple_benchmark_problems(self):
        """Test asserts when more than one BenchmarkProblem."""
        data = {
            "BenchmarkProblems": [
                [{"OperationType": "GEMM"}],
                [{"OperationType": "Other"}]
            ]
        }

        with pytest.raises(AssertionError, match="one BenchmarkProblems"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(data)

    def test_handles_empty_benchmark_problems(self):
        """Test handles config with empty BenchmarkProblems."""
        data = {
            "BenchmarkProblems": [[]]
        }

        # Should raise error about problem type not found
        with pytest.raises((IndexError, AssertionError)):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(data)


class TestBenchmarkSplitterReadConfigFile:
    """Tests for BenchmarkSplitter.__readConfigFile method."""

    def test_reads_config_file(self, tmp_path):
        """Test __readConfigFile reads YAML correctly."""
        config_data = {
            "Key1": "Value1",
            "Key2": [1, 2, 3],
            "Key3": {"nested": "dict"}
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Access private method
        result = BenchmarkSplitter._BenchmarkSplitter__readConfigFile(str(config_file))

        assert result == config_data
        assert result["Key1"] == "Value1"
        assert result["Key2"] == [1, 2, 3]
        assert result["Key3"]["nested"] == "dict"
