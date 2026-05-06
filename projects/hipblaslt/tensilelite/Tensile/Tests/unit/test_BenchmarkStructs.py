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
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

from Tensile.BenchmarkStructs import (
    getDefaultsForMissingParameters,
    separateParameters,
    checkCDBufferAndStrides
)


class TestGetDefaultsForMissingParameters:
    """Tests for getDefaultsForMissingParameters function."""

    def test_returns_all_defaults_when_param_list_empty(self):
        """Test returns all defaults when paramList is empty."""
        paramList = []
        defaultParams = [
            {"Param1": "value1"},
            {"Param2": "value2"}
        ]

        result = getDefaultsForMissingParameters(paramList, defaultParams)

        assert result == {"Param1": "value1", "Param2": "value2"}

    def test_excludes_params_in_param_list(self):
        """Test excludes parameters already in paramList."""
        paramList = [{"Param1": "custom_value"}]
        defaultParams = [
            {"Param1": "default1"},
            {"Param2": "default2"}
        ]

        result = getDefaultsForMissingParameters(paramList, defaultParams)

        assert "Param1" not in result
        assert result["Param2"] == "default2"

    def test_always_includes_problem_sizes(self):
        """Test ProblemSizes is always included even if in paramList."""
        paramList = [{"ProblemSizes": "custom"}]
        defaultParams = [{"ProblemSizes": "default"}]

        result = getDefaultsForMissingParameters(paramList, defaultParams)

        assert result["ProblemSizes"] == "default"

    def test_handles_multiple_default_dicts(self):
        """Test handles multiple dictionaries in defaultParams."""
        paramList = []
        defaultParams = [
            {"A": 1, "B": 2},
            {"C": 3, "D": 4}
        ]

        result = getDefaultsForMissingParameters(paramList, defaultParams)

        assert result == {"A": 1, "B": 2, "C": 3, "D": 4}

    def test_later_defaults_override_earlier(self):
        """Test later default dicts override earlier ones for same key."""
        paramList = []
        defaultParams = [
            {"Key": "first"},
            {"Key": "second"}
        ]

        result = getDefaultsForMissingParameters(paramList, defaultParams)

        assert result["Key"] == "second"


class TestSeparateParameters:
    """Tests for separateParameters function."""

    def test_separates_single_and_multi_values(self):
        """Test separates parameters with single vs multiple values."""
        paramSetList = {
            "SingleParam": [10],
            "MultiParam": [1, 2, 3]
        }

        single, multi = separateParameters(paramSetList)

        assert single == {"SingleParam": 10}
        assert multi == {"MultiParam": [1, 2, 3]}

    def test_problem_sizes_not_in_single_values(self):
        """Test ProblemSizes with single value is not treated as single."""
        paramSetList = {
            "ProblemSizes": [100],
            "OtherParam": [5]
        }

        single, multi = separateParameters(paramSetList)

        assert "ProblemSizes" not in single
        assert "ProblemSizes" not in multi
        assert single == {"OtherParam": 5}

    def test_problem_sizes_not_in_multi_values(self):
        """Test ProblemSizes with multiple values is not in multi dict."""
        paramSetList = {
            "ProblemSizes": [100, 200],
            "OtherParam": [5, 10]
        }

        single, multi = separateParameters(paramSetList)

        assert "ProblemSizes" not in multi
        assert multi == {"OtherParam": [5, 10]}

    def test_exits_on_none_value(self):
        """Test exits when parameter value is None."""
        paramSetList = {
            "BadParam": None
        }

        with pytest.raises(SystemExit):
            separateParameters(paramSetList)

    def test_empty_param_list(self):
        """Test with empty parameter list."""
        paramSetList = {}

        single, multi = separateParameters(paramSetList)

        assert single == {}
        assert multi == {}

    def test_all_single_values(self):
        """Test when all parameters have single values."""
        paramSetList = {
            "A": [1],
            "B": [2],
            "C": [3]
        }

        single, multi = separateParameters(paramSetList)

        assert single == {"A": 1, "B": 2, "C": 3}
        assert multi == {}

    def test_all_multi_values(self):
        """Test when all parameters have multiple values."""
        paramSetList = {
            "A": [1, 2],
            "B": [3, 4],
            "C": [5, 6]
        }

        single, multi = separateParameters(paramSetList)

        assert single == {}
        assert multi == {
            "A": [1, 2],
            "B": [3, 4],
            "C": [5, 6]
        }


class TestCheckCDBufferAndStrides:
    """Tests for checkCDBufferAndStrides function."""

    def test_passes_when_c_not_equal_d(self):
        """Test passes validation when CEqualD is False."""
        problemType = {
            "OperationType": "GEMM",
            "IndexAssignmentsLD": [0, 1]
        }

        mock_problem = MagicMock()
        mock_problem.sizes = [100, 200]

        problemSizes = MagicMock()
        problemSizes.problems = [mock_problem]

        # Should not raise
        checkCDBufferAndStrides(problemType, problemSizes, isCEqualD=False)

    def test_passes_when_non_gemm_operation(self):
        """Test passes validation for non-GEMM operations."""
        problemType = {
            "OperationType": "CONTRACTION",
            "IndexAssignmentsLD": [0, 1]
        }

        mock_problem = MagicMock()
        mock_problem.sizes = [100, 200]  # Different values

        problemSizes = MagicMock()
        problemSizes.problems = [mock_problem]

        # Should not raise even with different LD values
        checkCDBufferAndStrides(problemType, problemSizes, isCEqualD=True)

    def test_passes_when_ldd_equals_ldc(self):
        """Test passes when LDD equals LDC with CEqualD."""
        problemType = {
            "OperationType": "GEMM",
            "IndexAssignmentsLD": [0, 1]
        }

        mock_problem = MagicMock()
        mock_problem.sizes = [100, 100]  # Same values

        problemSizes = MagicMock()
        problemSizes.problems = [mock_problem]

        # Should not raise
        checkCDBufferAndStrides(problemType, problemSizes, isCEqualD=True)

    def test_fails_when_ldd_not_equal_ldc_with_c_equal_d(self):
        """Test fails when LDD != LDC with CEqualD=True."""
        problemType = {
            "OperationType": "GEMM",
            "IndexAssignmentsLD": [0, 1]
        }

        mock_problem = MagicMock()
        mock_problem.sizes = [100, 200]  # Different values

        problemSizes = MagicMock()
        problemSizes.problems = [mock_problem]

        with pytest.raises(SystemExit):
            checkCDBufferAndStrides(problemType, problemSizes, isCEqualD=True)

    def test_checks_all_problems(self):
        """Test validates all problems in problemSizes."""
        problemType = {
            "OperationType": "GEMM",
            "IndexAssignmentsLD": [0, 1]
        }

        # First problem OK, second problem fails
        mock_problem1 = MagicMock()
        mock_problem1.sizes = [100, 100]

        mock_problem2 = MagicMock()
        mock_problem2.sizes = [100, 200]

        problemSizes = MagicMock()
        problemSizes.problems = [mock_problem1, mock_problem2]

        with pytest.raises(SystemExit):
            checkCDBufferAndStrides(problemType, problemSizes, isCEqualD=True)
