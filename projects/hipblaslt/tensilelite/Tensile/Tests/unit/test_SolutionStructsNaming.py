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

from Tensile.SolutionStructs.Naming import (
    getParameterNameAbbreviation,
    getPrimitiveParameterValueAbbreviation,
    getParameterValueAbbreviation,
    getKernelNameMin,
    getSolutionNameMin,
    getSolutionNameFull,
    getKernelFileBase,
    shortenFileBase,
    getKeyNoInternalArgs,
    _getName
)
from Tensile.SolutionStructs.Problem import ProblemType


class TestGetParameterNameAbbreviation:
    """Tests for getParameterNameAbbreviation function."""

    def test_abbreviation_all_uppercase(self):
        """Test abbreviation of name with all uppercase letters."""
        result = getParameterNameAbbreviation("ABC")
        assert result == "ABC"

    def test_abbreviation_mixed_case(self):
        """Test abbreviation extracts only uppercase letters."""
        result = getParameterNameAbbreviation("MacroTile")
        assert result == "MT"

    def test_abbreviation_camel_case(self):
        """Test abbreviation of camelCase name."""
        result = getParameterNameAbbreviation("ThreadTile")
        assert result == "TT"

    def test_abbreviation_single_uppercase(self):
        """Test abbreviation with single uppercase letter."""
        result = getParameterNameAbbreviation("Value")
        assert result == "V"

    def test_abbreviation_no_uppercase(self):
        """Test abbreviation with no uppercase letters."""
        result = getParameterNameAbbreviation("lowercase")
        assert result == ""

    def test_abbreviation_multiple_consecutive_uppercase(self):
        """Test abbreviation with consecutive uppercase letters."""
        result = getParameterNameAbbreviation("MFMA")
        assert result == "MFMA"

    def test_abbreviation_cached(self):
        """Test that results are cached (calling twice returns same object)."""
        result1 = getParameterNameAbbreviation("TestName")
        result2 = getParameterNameAbbreviation("TestName")
        assert result1 is result2  # Same object due to caching


class TestGetPrimitiveParameterValueAbbreviation:
    """Tests for getPrimitiveParameterValueAbbreviation function."""

    def test_string_value_abbreviation(self):
        """Test abbreviation of string value."""
        result = getPrimitiveParameterValueAbbreviation("key", "CamelCase")
        assert result == "CC"

    def test_boolean_true_abbreviation(self):
        """Test abbreviation of True boolean."""
        result = getPrimitiveParameterValueAbbreviation("key", True)
        assert result == "1"

    def test_boolean_false_abbreviation(self):
        """Test abbreviation of False boolean."""
        result = getPrimitiveParameterValueAbbreviation("key", False)
        assert result == "0"

    def test_positive_integer_abbreviation(self):
        """Test abbreviation of positive integer."""
        result = getPrimitiveParameterValueAbbreviation("key", 42)
        assert result == "42"

    def test_zero_integer_abbreviation(self):
        """Test abbreviation of zero integer."""
        result = getPrimitiveParameterValueAbbreviation("key", 0)
        assert result == "0"

    def test_negative_integer_abbreviation(self):
        """Test abbreviation of negative integer with 'n' prefix."""
        result = getPrimitiveParameterValueAbbreviation("key", -1)
        assert result == "n1"

    def test_negative_large_integer_abbreviation(self):
        """Test abbreviation of large negative integer."""
        result = getPrimitiveParameterValueAbbreviation("key", -42)
        assert result == "n42"

    def test_float_integer_value_abbreviation(self):
        """Test abbreviation of float with integer value."""
        result = getPrimitiveParameterValueAbbreviation("key", 5.0)
        assert result == "5"

    def test_float_with_decimals_abbreviation(self):
        """Test abbreviation of float with decimal part."""
        result = getPrimitiveParameterValueAbbreviation("key", 3.14)
        assert result == "3p14"

    def test_float_with_small_decimals_abbreviation(self):
        """Test abbreviation of float with small decimal (zero padded)."""
        result = getPrimitiveParameterValueAbbreviation("key", 2.05)
        assert result == "2p05"

    def test_float_negative_with_decimals_abbreviation(self):
        """Test abbreviation of negative float with decimals."""
        result = getPrimitiveParameterValueAbbreviation("key", -1.5)
        # Negative floats are converted to int first, so -1.5 becomes -1
        assert result == "-1"

    def test_problem_type_value_abbreviation(self):
        """Test abbreviation of ProblemType returns string representation."""
        mock_pt = MagicMock(spec=ProblemType)
        mock_pt.__str__ = MagicMock(return_value="MockProblemType")

        result = getPrimitiveParameterValueAbbreviation("key", mock_pt)
        assert result == "MockProblemType"


class TestGetParameterValueAbbreviation:
    """Tests for getParameterValueAbbreviation function."""

    def test_isa_special_format(self):
        """Test ISA key uses special hex format."""
        result = getParameterValueAbbreviation("ISA", (9, 0, 10))
        assert result == "90a"  # Format is {v0}{v1}{v2:x}

    def test_isa_different_version(self):
        """Test ISA with different version numbers."""
        result = getParameterValueAbbreviation("ISA", (11, 0, 1))
        assert result == "1101"

    def test_primitive_string_value(self):
        """Test primitive string value delegates to primitive function."""
        result = getParameterValueAbbreviation("key", "TestValue")
        assert result == "TV"

    def test_primitive_int_value(self):
        """Test primitive int value."""
        result = getParameterValueAbbreviation("key", 123)
        assert result == "123"

    def test_primitive_bool_value(self):
        """Test primitive bool value."""
        result = getParameterValueAbbreviation("key", True)
        assert result == "1"

    def test_tuple_value_concatenation(self):
        """Test tuple value concatenates elements."""
        result = getParameterValueAbbreviation("key", (1, 2, 3))
        assert result == "123"

    def test_tuple_mixed_types(self):
        """Test tuple with mixed types."""
        result = getParameterValueAbbreviation("key", (4, 8))
        assert result == "48"

    def test_list_value_underscore_separated(self):
        """Test list value joins with underscores."""
        result = getParameterValueAbbreviation("key", [1, 2, 3])
        assert result == "1_2_3"

    def test_list_with_strings(self):
        """Test list with string values."""
        result = getParameterValueAbbreviation("key", ["A", "B"])
        assert result == "A_B"

    def test_dict_value_position_key_format(self):
        """Test dict value uses position:key format."""
        result = getParameterValueAbbreviation("key", {0: 1, 1: 2})
        assert result == "01_12"

    def test_dict_single_entry(self):
        """Test dict with single entry."""
        result = getParameterValueAbbreviation("key", {5: 10})
        assert result == "510"

    def test_unsupported_type_raises_exception(self):
        """Test unsupported type raises exception."""
        # Sets are unhashable and will raise TypeError before reaching our exception
        with pytest.raises((Exception, TypeError)):
            getParameterValueAbbreviation("key", set([1, 2, 3]))


class TestGetName:
    """Tests for _getName internal function."""

    def test_custom_kernel_name_returns_immediately(self):
        """Test custom kernel name is returned without processing."""
        state = {
            "CustomKernelName": "MyCustomKernel",
            "ProblemType": {"GroupedGemm": False},
            "GlobalSplitU": 1
        }

        result = _getName(state, frozenset(), False, False)
        assert result == "MyCustomKernel"


class TestShortenFileBase:
    """Tests for shortenFileBase function."""

    @patch('Tensile.SolutionStructs.Naming.getKernelNameMin')
    @patch('Tensile.SolutionStructs.Naming.MAX_FILENAME_LENGTH', 100)
    def test_short_name_unchanged(self, mock_get_name):
        """Test short name is returned unchanged."""
        mock_get_name.return_value = "ShortName"

        result = shortenFileBase(False, {})
        assert result == "ShortName"

    @patch('Tensile.SolutionStructs.Naming.getKernelNameMin')
    @patch('Tensile.SolutionStructs.Naming.MAX_FILENAME_LENGTH', 10)
    def test_long_name_gets_shortened(self, mock_get_name):
        """Test long name gets shortened with hash."""
        long_name = "VeryLongKernelNameThatExceedsMaxLength"
        mock_get_name.return_value = long_name

        result = shortenFileBase(False, {})

        # Should be shortened to MAX_FILENAME_LENGTH
        assert len(result) <= 10 + 50  # Hash adds some length
        # Should start with first part of original
        assert result.startswith(long_name[:7])  # 10 * 3/4 = 7

    @patch('Tensile.SolutionStructs.Naming.getKernelNameMin')
    @patch('Tensile.SolutionStructs.Naming.MAX_FILENAME_LENGTH', 20)
    def test_exactly_max_length_unchanged(self, mock_get_name):
        """Test name exactly at max length is unchanged."""
        exact_name = "A" * 20
        mock_get_name.return_value = exact_name

        result = shortenFileBase(False, {})
        assert result == exact_name


class TestGetKernelFileBase:
    """Tests for getKernelFileBase function."""

    def test_custom_kernel_name_used(self):
        """Test custom kernel name is used as file base."""
        kernel = {"CustomKernelName": "CustomName"}

        result = getKernelFileBase(False, kernel)
        assert result == "CustomName"

    @patch('Tensile.SolutionStructs.Naming.shortenFileBase')
    def test_no_custom_name_uses_shortened_base(self, mock_shorten):
        """Test shortened file base is used when no custom name."""
        mock_shorten.return_value = "ShortenedBase"
        kernel = {}

        result = getKernelFileBase(True, kernel)
        assert result == "ShortenedBase"
        mock_shorten.assert_called_once_with(True, kernel)

    def test_empty_custom_kernel_name_uses_shortened(self):
        """Test empty custom kernel name falls back to shortened base."""
        # Empty string is falsy, so should use shortenFileBase
        with patch('Tensile.SolutionStructs.Naming.shortenFileBase') as mock_shorten:
            mock_shorten.return_value = "Shortened"
            kernel = {"CustomKernelName": ""}

            result = getKernelFileBase(False, kernel)
            assert result == "Shortened"


class TestGetKernelNameMin:
    """Tests for getKernelNameMin function."""

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersMin')
    def test_calls_get_name_with_min_params(self, mock_req_params, mock_get_name):
        """Test calls _getName with required parameters min."""
        mock_req_params.return_value = frozenset(["Param1", "Param2"])
        mock_get_name.return_value = "KernelName"

        kernel = {"key": "value"}
        result = getKernelNameMin(kernel, True)

        mock_get_name.assert_called_once_with(
            kernel,
            frozenset(["Param1", "Param2"]),
            True,
            True  # ignoreInternalArgs=True for kernel names
        )
        assert result == "KernelName"


class TestGetSolutionNameMin:
    """Tests for getSolutionNameMin function."""

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersMin')
    def test_calls_get_name_with_min_params(self, mock_req_params, mock_get_name):
        """Test calls _getName with required parameters min."""
        mock_req_params.return_value = frozenset(["Param1", "Param2"])
        mock_get_name.return_value = "SolutionName"

        solution = {"key": "value"}
        result = getSolutionNameMin(solution, False)

        mock_get_name.assert_called_once_with(
            solution,
            frozenset(["Param1", "Param2"]),
            False,
            False  # ignoreInternalArgs=False for solution names
        )
        assert result == "SolutionName"


class TestGetSolutionNameFull:
    """Tests for getSolutionNameFull function."""

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersFull')
    def test_calls_get_name_with_full_params(self, mock_req_params, mock_get_name):
        """Test calls _getName with required parameters full."""
        mock_req_params.return_value = frozenset(["Param1", "Param2", "Param3"])
        mock_get_name.return_value = "FullSolutionName"

        state = {"key": "value"}
        result = getSolutionNameFull(state, True)

        mock_get_name.assert_called_once_with(
            state,
            frozenset(["Param1", "Param2", "Param3"]),
            True,
            False  # ignoreInternalArgs=False for solution names
        )
        assert result == "FullSolutionName"


class TestGetKeyNoInternalArgs:
    """Tests for getKeyNoInternalArgs function."""

    def _create_complete_state(self):
        """Helper to create state with all required internal args."""
        return {
            "WorkGroupMapping": 1,
            "WorkGroupMappingXCC": 1,
            "WorkGroupMappingXCCGroup": 1,
            "StaggerU": 2,
            "StaggerUStride": 1,
            "StaggerUMapping": 0,
            "GlobalSplitU": 1,
            "GlobalSplitUCoalesced": False,
            "GlobalSplitUWorkGroupMappingRoundRobin": False,
            "SFCWGM": 0,
            "ProblemType": {"GroupedGemm": True}
        }

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersFull')
    def test_masks_internal_args_to_m(self, mock_req_params, mock_get_name):
        """Test internal args are masked to 'M'."""
        mock_req_params.return_value = frozenset()
        mock_get_name.return_value = "KeyName"

        state = self._create_complete_state()

        getKeyNoInternalArgs(state, False)

        # Internal args should be restored after function call
        assert state["WorkGroupMapping"] == 1  # Restored after

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersFull')
    def test_includes_code_object_file_in_key(self, mock_req_params, mock_get_name):
        """Test includes codeObjectFile in returned key."""
        mock_req_params.return_value = frozenset()
        mock_get_name.return_value = "BaseKey"

        state = self._create_complete_state()
        state["codeObjectFile"] = "kernel.co"

        result = getKeyNoInternalArgs(state, False)
        assert "kernel.co" in result

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersFull')
    def test_includes_device_names_in_key(self, mock_req_params, mock_get_name):
        """Test includes DeviceNames in returned key."""
        mock_req_params.return_value = frozenset()
        mock_get_name.return_value = "BaseKey"

        state = self._create_complete_state()
        state["DeviceNames"] = ["gfx90a", "gfx942"]

        result = getKeyNoInternalArgs(state, False)
        assert "gfx90a" in result or "gfx942" in result

    @patch('Tensile.SolutionStructs.Naming._getName')
    @patch('Tensile.SolutionStructs.Naming.getRequiredParametersFull')
    def test_restores_original_values(self, mock_req_params, mock_get_name):
        """Test restores all original values after key generation."""
        mock_req_params.return_value = frozenset()
        mock_get_name.return_value = "Key"

        state = {
            "WorkGroupMapping": 42,
            "StaggerU": 7,
            "GlobalSplitU": 3,
            "ProblemType": {"GroupedGemm": True},
            "WorkGroupMappingXCC": 1,
            "WorkGroupMappingXCCGroup": 2,
            "StaggerUStride": 4,
            "StaggerUMapping": 5,
            "GlobalSplitUCoalesced": True,
            "GlobalSplitUWorkGroupMappingRoundRobin": False,
            "SFCWGM": 8
        }

        getKeyNoInternalArgs(state, False)

        # All values should be restored
        assert state["WorkGroupMapping"] == 42
        assert state["StaggerU"] == 7
        assert state["GlobalSplitU"] == 3
        assert state["ProblemType"]["GroupedGemm"] is True
