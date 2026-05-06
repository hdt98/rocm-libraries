################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
import pytest
from io import StringIO
import sys

from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Utilities import (
    getMiInputType,
    reject,
    pvar,
    roundupRatio,
    getRealDataTypeA,
    getRealDataTypeB
)


def _build_kernel(*, enable_f32_xdl_math_op=False, use_f32x_emulation=False,
                  data_type=None, f32_xdl_math_op=None):
    """Build a minimal kernel dict for getMiInputType tests."""
    return {
        "EnableF32XdlMathOp": enable_f32_xdl_math_op,
        "UseF32XEmulation": use_f32x_emulation,
        "ProblemType": {
            "DataType": data_type or DataType(DataTypeEnum.Float),
            "F32XdlMathOp": f32_xdl_math_op or DataType(DataTypeEnum.XFloat32),
        },
    }


class TestGetMiInputType:
    """Verify getMiInputType selects the correct MFMA operand type.

    EnableF32XdlMathOp=False                         → ProblemType["DataType"]
    EnableF32XdlMathOp=True,  UseF32XEmulation=False → ProblemType["F32XdlMathOp"]
    EnableF32XdlMathOp=True,  UseF32XEmulation=True  → DataType(BFloat16)
    """

    def test_plain_datatype(self):
        """Neither flag set → returns ProblemType.DataType."""
        kernel = _build_kernel()

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.Float)

    def test_native_xf32(self):
        """EnableF32XdlMathOp without emulation → returns F32XdlMathOp."""
        kernel = _build_kernel(enable_f32_xdl_math_op=True)

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.XFloat32)

    def test_use_f32x_emulation(self):
        """UseF32XEmulation + EnableF32XdlMathOp → returns DataType(BFloat16)."""
        kernel = _build_kernel(
            enable_f32_xdl_math_op=True,
            use_f32x_emulation=True,
        )

        result = getMiInputType(kernel)

        assert result == DataType(DataTypeEnum.BFloat16)

    def test_missing_enable_flag_raises(self):
        """EnableF32XdlMathOp must exist; absent key means broken caller."""
        kernel = _build_kernel()
        del kernel["EnableF32XdlMathOp"]

        with pytest.raises(KeyError):
            getMiInputType(kernel)

    def test_missing_emulation_flag_raises(self):
        """UseF32XEmulation must exist; absent key means broken caller."""
        kernel = _build_kernel(enable_f32_xdl_math_op=True)
        del kernel["UseF32XEmulation"]

        with pytest.raises(KeyError):
            getMiInputType(kernel)


class TestReject:
    """Tests for reject function."""

    def test_reject_with_no_reject_flag_returns_false(self):
        """Test reject returns False when NoReject is True."""
        state = {"NoReject": True, "Valid": True}

        result = reject(state, False)

        assert result is False
        assert state["Valid"] is True  # Should not be modified

    def test_reject_with_none_state_prints_nothing(self):
        """Test reject with None state doesn't error and doesn't set Valid."""
        result = reject(None, False)

        # reject() only sets state["Valid"] = False if state is not None
        # When state is None, it just returns without setting anything
        # The function returns True only if it sets Valid to False
        # Looking at the code: if state != None: state["Valid"] = False; return True
        # So if state is None, it returns None (implicit)
        assert result is None

    def test_reject_sets_valid_to_false(self):
        """Test reject sets Valid to False in state."""
        state = {"Valid": True}

        result = reject(state, False)

        assert result is True
        assert state["Valid"] is False

    def test_reject_prints_reason_when_enabled(self, capsys):
        """Test reject prints rejection reason when printSolutionRejectionReason is True."""
        state = {"Valid": True}

        result = reject(state, True, "reason1", "reason2")

        captured = capsys.readouterr()
        assert "reject:" in captured.out
        assert "reason1" in captured.out
        assert "reason2" in captured.out
        assert result is True

    def test_reject_no_print_when_disabled(self, capsys):
        """Test reject doesn't print when printSolutionRejectionReason is False."""
        state = {"Valid": True}

        result = reject(state, False, "reason1")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert result is True

    def test_reject_raises_exception_for_library_logic_rejection(self):
        """Test reject raises exception when SolutionIndex is present."""
        state = {
            "Valid": True,
            "SolutionIndex": 42,
            "SolutionNameMin": "TestSolution",
            "ProblemType": "TestProblem"
        }

        with pytest.raises(Exception, match="!! Warning: Any rejection of a LibraryLogic"):
            reject(state, True, "test reason")

    def test_reject_uses_problem_type_when_no_solution_name_min(self):
        """Test reject uses ProblemType when SolutionNameMin is not present."""
        state = {
            "Valid": True,
            "SolutionIndex": 42,
            "ProblemType": "TestProblemType"
        }

        with pytest.raises(Exception, match="TestProblemType"):
            reject(state, True, "test reason")

    def test_reject_skips_exception_when_solution_index_minus_one(self):
        """Test reject doesn't raise exception when SolutionIndex is -1."""
        state = {
            "Valid": True,
            "SolutionIndex": -1
        }

        result = reject(state, True, "test reason")

        assert result is True
        assert state["Valid"] is False


class TestPvar:
    """Tests for pvar function."""

    def test_pvar_formats_string_value(self):
        """Test pvar formats field with string value."""
        state = {"name": "test"}

        result = pvar(state, "name")

        assert result == "name=test"

    def test_pvar_formats_integer_value(self):
        """Test pvar formats field with integer value."""
        state = {"count": 42}

        result = pvar(state, "count")

        assert result == "count=42"

    def test_pvar_formats_boolean_value(self):
        """Test pvar formats field with boolean value."""
        state = {"flag": True}

        result = pvar(state, "flag")

        assert result == "flag=True"

    def test_pvar_formats_none_value(self):
        """Test pvar formats field with None value."""
        state = {"value": None}

        result = pvar(state, "value")

        assert result == "value=None"

    def test_pvar_formats_list_value(self):
        """Test pvar formats field with list value."""
        state = {"items": [1, 2, 3]}

        result = pvar(state, "items")

        assert result == "items=[1, 2, 3]"


class TestRoundupRatio:
    """Tests for roundupRatio function."""

    def test_roundup_ratio_exact_division(self):
        """Test roundupRatio with exact division."""
        result = roundupRatio(10, 5)
        assert result == 2

    def test_roundup_ratio_rounds_up(self):
        """Test roundupRatio rounds up."""
        result = roundupRatio(10, 3)
        assert result == 4  # ceil(10/3) = 4

    def test_roundup_ratio_one_remainder(self):
        """Test roundupRatio with remainder of 1."""
        result = roundupRatio(7, 3)
        assert result == 3  # ceil(7/3) = 3

    def test_roundup_ratio_dividend_one(self):
        """Test roundupRatio with dividend of 1."""
        result = roundupRatio(1, 10)
        assert result == 1  # ceil(0.1) = 1

    def test_roundup_ratio_same_values(self):
        """Test roundupRatio with same dividend and divisor."""
        result = roundupRatio(5, 5)
        assert result == 1

    def test_roundup_ratio_large_numbers(self):
        """Test roundupRatio with large numbers."""
        result = roundupRatio(1000000, 999999)
        assert result == 2  # ceil(1.000001) = 2

    def test_roundup_ratio_returns_int(self):
        """Test roundupRatio returns int type."""
        result = roundupRatio(10, 3)
        assert isinstance(result, int)


class TestGetRealDataTypeA:
    """Tests for getRealDataTypeA function."""

    def test_get_real_datatype_a_float8_bfloat8(self):
        """Test getRealDataTypeA returns Float8 for Float8BFloat8."""
        dt = DataType(DataTypeEnum.Float8BFloat8)

        result = getRealDataTypeA(dt)

        assert result == DataType(DataTypeEnum.Float8)

    def test_get_real_datatype_a_bfloat8_float8(self):
        """Test getRealDataTypeA returns BFloat8 for BFloat8Float8."""
        dt = DataType(DataTypeEnum.BFloat8Float8)

        result = getRealDataTypeA(dt)

        assert result == DataType(DataTypeEnum.BFloat8)

    def test_get_real_datatype_a_float8_bfloat8_fnuz(self):
        """Test getRealDataTypeA returns Float8_fnuz for Float8BFloat8_fnuz."""
        dt = DataType(DataTypeEnum.Float8BFloat8_fnuz)

        result = getRealDataTypeA(dt)

        assert result == DataType(DataTypeEnum.Float8_fnuz)

    def test_get_real_datatype_a_bfloat8_float8_fnuz(self):
        """Test getRealDataTypeA returns BFloat8_fnuz for BFloat8Float8_fnuz."""
        dt = DataType(DataTypeEnum.BFloat8Float8_fnuz)

        result = getRealDataTypeA(dt)

        assert result == DataType(DataTypeEnum.BFloat8_fnuz)

    def test_get_real_datatype_a_passthrough_float(self):
        """Test getRealDataTypeA returns same type for standard Float."""
        dt = DataType(DataTypeEnum.Float)

        result = getRealDataTypeA(dt)

        assert result == dt

    def test_get_real_datatype_a_passthrough_bfloat16(self):
        """Test getRealDataTypeA returns same type for BFloat16."""
        dt = DataType(DataTypeEnum.BFloat16)

        result = getRealDataTypeA(dt)

        assert result == dt

    def test_get_real_datatype_a_passthrough_int8(self):
        """Test getRealDataTypeA returns same type for Int8."""
        dt = DataType(DataTypeEnum.Int8)

        result = getRealDataTypeA(dt)

        assert result == dt


class TestGetRealDataTypeB:
    """Tests for getRealDataTypeB function."""

    def test_get_real_datatype_b_float8_bfloat8(self):
        """Test getRealDataTypeB returns BFloat8 for Float8BFloat8."""
        dt = DataType(DataTypeEnum.Float8BFloat8)

        result = getRealDataTypeB(dt)

        assert result == DataType(DataTypeEnum.BFloat8)

    def test_get_real_datatype_b_bfloat8_float8(self):
        """Test getRealDataTypeB returns Float8 for BFloat8Float8."""
        dt = DataType(DataTypeEnum.BFloat8Float8)

        result = getRealDataTypeB(dt)

        assert result == DataType(DataTypeEnum.Float8)

    def test_get_real_datatype_b_float8_bfloat8_fnuz(self):
        """Test getRealDataTypeB returns BFloat8_fnuz for Float8BFloat8_fnuz."""
        dt = DataType(DataTypeEnum.Float8BFloat8_fnuz)

        result = getRealDataTypeB(dt)

        assert result == DataType(DataTypeEnum.BFloat8_fnuz)

    def test_get_real_datatype_b_bfloat8_float8_fnuz(self):
        """Test getRealDataTypeB returns Float8_fnuz for BFloat8Float8_fnuz."""
        dt = DataType(DataTypeEnum.BFloat8Float8_fnuz)

        result = getRealDataTypeB(dt)

        assert result == DataType(DataTypeEnum.Float8_fnuz)

    def test_get_real_datatype_b_passthrough_float(self):
        """Test getRealDataTypeB returns same type for standard Float."""
        dt = DataType(DataTypeEnum.Float)

        result = getRealDataTypeB(dt)

        assert result == dt

    def test_get_real_datatype_b_passthrough_double(self):
        """Test getRealDataTypeB returns same type for Double."""
        dt = DataType(DataTypeEnum.Double)

        result = getRealDataTypeB(dt)

        assert result == dt

    def test_get_real_datatype_b_passthrough_int32(self):
        """Test getRealDataTypeB returns same type for Int32."""
        dt = DataType(DataTypeEnum.Int32)

        result = getRealDataTypeB(dt)

        assert result == dt
