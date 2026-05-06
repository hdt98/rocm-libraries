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
import json as json_stdlib
from io import StringIO
from unittest.mock import patch, MagicMock
from rocisa.enum import DataTypeEnum

pytestmark = pytest.mark.unit

from Tensile.LibraryIO import (
    _fast_yaml_scalar,
    _fast_yaml_str,
    _fast_yaml_flow_list,
    fast_yaml_dump,
    write,
    writeYAML,
    writeJson,
    read,
    readYAML,
    readJson,
    getRealDataTypeA,
    getRealDataTypeB,
    parseLibraryLogicList,
    rawLibraryLogic,
    _findBodyOffset,
    getCUCount
)


class TestFastYamlScalar:
    """Tests for _fast_yaml_scalar function."""

    def test_none_returns_null(self):
        """Test None value converts to 'null'."""
        assert _fast_yaml_scalar(None) == "null"

    def test_bool_true_returns_true(self):
        """Test True converts to 'true'."""
        assert _fast_yaml_scalar(True) == "true"

    def test_bool_false_returns_false(self):
        """Test False converts to 'false'."""
        assert _fast_yaml_scalar(False) == "false"

    def test_int_converts_to_string(self):
        """Test integer converts to string."""
        assert _fast_yaml_scalar(42) == "42"
        assert _fast_yaml_scalar(0) == "0"
        assert _fast_yaml_scalar(-100) == "-100"

    def test_float_uses_repr(self):
        """Test float uses repr() for conversion."""
        result = _fast_yaml_scalar(3.14)
        assert result == repr(3.14)

    def test_string_calls_fast_yaml_str(self):
        """Test string values use _fast_yaml_str."""
        result = _fast_yaml_scalar("hello")
        assert result == "hello"

    def test_list_calls_fast_yaml_flow_list(self):
        """Test list values use _fast_yaml_flow_list."""
        result = _fast_yaml_scalar([1, 2, 3])
        assert result == "[1, 2, 3]"


class TestFastYamlStr:
    """Tests for _fast_yaml_str function."""

    def test_empty_string_quoted(self):
        """Test empty string is quoted."""
        assert _fast_yaml_str("") == "''"

    def test_bool_keyword_quoted(self):
        """Test YAML boolean keywords are quoted."""
        assert _fast_yaml_str("true") == "'true'"
        assert _fast_yaml_str("false") == "'false'"
        assert _fast_yaml_str("yes") == "'yes'"
        assert _fast_yaml_str("no") == "'no'"

    def test_null_keyword_quoted(self):
        """Test YAML null keywords are quoted."""
        assert _fast_yaml_str("null") == "'null'"
        assert _fast_yaml_str("~") == "'~'"

    def test_leading_space_quoted(self):
        """Test string with leading space is quoted."""
        assert _fast_yaml_str(" hello") == "' hello'"

    def test_trailing_space_quoted(self):
        """Test string with trailing space is quoted."""
        assert _fast_yaml_str("hello ") == "'hello '"

    def test_colon_space_quoted(self):
        """Test string with ': ' is quoted."""
        assert _fast_yaml_str("key: value") == "'key: value'"

    def test_space_hash_quoted(self):
        """Test string with ' #' is quoted."""
        assert _fast_yaml_str("text # comment") == "'text # comment'"

    def test_single_quote_escaped(self):
        """Test single quotes are escaped by doubling when string is quoted."""
        # String with quote AND special char needs quoting
        assert _fast_yaml_str("don't: value") == "'don''t: value'"

    def test_number_like_string_quoted(self):
        """Test strings that look like numbers are quoted."""
        assert _fast_yaml_str("123") == "'123'"
        assert _fast_yaml_str("3.14") == "'3.14'"

    def test_plain_string_unquoted(self):
        """Test plain strings are not quoted."""
        assert _fast_yaml_str("hello") == "hello"
        assert _fast_yaml_str("my_key") == "my_key"


class TestFastYamlFlowList:
    """Tests for _fast_yaml_flow_list function."""

    def test_empty_list_returns_brackets(self):
        """Test empty list returns '[]'."""
        assert _fast_yaml_flow_list([]) == "[]"

    def test_single_element_list(self):
        """Test list with single element."""
        assert _fast_yaml_flow_list([42]) == "[42]"

    def test_multiple_elements(self):
        """Test list with multiple elements."""
        assert _fast_yaml_flow_list([1, 2, 3]) == "[1, 2, 3]"

    def test_mixed_types(self):
        """Test list with mixed types."""
        result = _fast_yaml_flow_list([1, True, "hello"])
        assert result == "[1, true, hello]"


class TestFastYamlDump:
    """Tests for fast_yaml_dump function."""

    def test_writes_simple_dict(self):
        """Test writing simple dictionary."""
        f = StringIO()
        data = [{"key1": "value1", "key2": 42}]

        fast_yaml_dump(data, f)
        result = f.getvalue()

        assert "- key1: value1" in result
        assert "  key2: 42" in result

    def test_writes_nested_dict(self):
        """Test writing dictionary with nested dict."""
        f = StringIO()
        data = [{"outer": {"inner1": 1, "inner2": 2}}]

        fast_yaml_dump(data, f)
        result = f.getvalue()

        assert "- outer:" in result
        assert "    inner1: 1" in result
        assert "    inner2: 2" in result

    def test_sorts_keys(self):
        """Test keys are sorted alphabetically."""
        f = StringIO()
        data = [{"z": 1, "a": 2, "m": 3}]

        fast_yaml_dump(data, f)
        result = f.getvalue()

        lines = result.strip().split('\n')
        assert "- a: 2" in lines[0]
        assert "  m: 3" in lines[1]
        assert "  z: 1" in lines[2]

    def test_multiple_solutions(self):
        """Test writing multiple solution dicts."""
        f = StringIO()
        data = [{"key1": 1}, {"key2": 2}]

        fast_yaml_dump(data, f)
        result = f.getvalue()

        assert result.count("- key") == 2


class TestWriteFunctions:
    """Tests for write, writeYAML, writeJson functions."""

    def test_write_yaml_format(self, tmp_path):
        """Test write function with YAML format."""
        file_base = tmp_path / "test"
        data = {"key": "value"}

        write(str(file_base), data, format="yaml")

        assert (tmp_path / "test.yaml").exists()

    def test_write_json_format(self, tmp_path):
        """Test write function with JSON format."""
        file_base = tmp_path / "test"
        data = {"key": "value"}

        write(str(file_base), data, format="json")

        assert (tmp_path / "test.json").exists()

    def test_write_msgpack_format(self, tmp_path):
        """Test write function with msgpack format."""
        pytest.importorskip("msgpack")
        file_base = tmp_path / "test"
        data = {"key": "value"}

        write(str(file_base), data, format="msgpack")

        assert (tmp_path / "test.dat").exists()

    def test_write_invalid_format_exits(self, tmp_path):
        """Test write function with invalid format exits."""
        file_base = tmp_path / "test"
        data = {"key": "value"}

        with pytest.raises(SystemExit):
            write(str(file_base), data, format="invalid")

    def test_writeYAML_creates_file(self, tmp_path):
        """Test writeYAML creates valid YAML file."""
        yaml_file = tmp_path / "test.yaml"
        data = {"key": "value", "number": 42}

        writeYAML(str(yaml_file), data)

        assert yaml_file.exists()
        with open(yaml_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_writeYAML_with_explicit_start(self, tmp_path):
        """Test writeYAML includes explicit start marker."""
        yaml_file = tmp_path / "test.yaml"
        data = {"key": "value"}

        writeYAML(str(yaml_file), data)

        content = yaml_file.read_text()
        assert content.startswith("---")

    def test_writeJson_creates_file(self, tmp_path):
        """Test writeJson creates valid JSON file."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        writeJson(str(json_file), data)

        assert json_file.exists()
        with open(json_file) as f:
            loaded = json_stdlib.load(f)
        assert loaded == data


class TestReadFunctions:
    """Tests for read, readYAML, readJson functions."""

    def test_read_yaml_file(self, tmp_path):
        """Test read function with YAML file."""
        yaml_file = tmp_path / "test.yaml"
        data = {"key": "value"}
        yaml_file.write_text("key: value\n")

        result = read(str(yaml_file))

        assert result == data

    def test_read_json_file(self, tmp_path):
        """Test read function with JSON file."""
        json_file = tmp_path / "test.json"
        data = {"key": "value"}
        json_file.write_text('{"key": "value"}')

        result = read(str(json_file))

        assert result == data

    def test_read_invalid_extension_exits(self, tmp_path):
        """Test read function with invalid extension exits."""
        invalid_file = tmp_path / "test.invalid"
        invalid_file.write_text("data")

        with pytest.raises(SystemExit):
            read(str(invalid_file))

    def test_readYAML_loads_file(self, tmp_path):
        """Test readYAML loads YAML file correctly."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n")

        result = readYAML(str(yaml_file))

        assert result == {"key": "value", "number": 42}

    def test_readJson_loads_file(self, tmp_path):
        """Test readJson loads JSON file correctly."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = readJson(str(json_file))

        assert result == {"key": "value", "number": 42}


class TestGetRealDataType:
    """Tests for getRealDataTypeA and getRealDataTypeB functions."""

    def test_getRealDataTypeA_float8bfloat8(self):
        """Test Float8BFloat8 returns Float8 for A."""
        result = getRealDataTypeA(DataTypeEnum.Float8BFloat8.value)
        assert result == DataTypeEnum.Float8.value

    def test_getRealDataTypeA_bfloat8float8(self):
        """Test BFloat8Float8 returns BFloat8 for A."""
        result = getRealDataTypeA(DataTypeEnum.BFloat8Float8.value)
        assert result == DataTypeEnum.BFloat8.value

    def test_getRealDataTypeA_float8bfloat8_fnuz(self):
        """Test Float8BFloat8_fnuz returns Float8_fnuz for A."""
        result = getRealDataTypeA(DataTypeEnum.Float8BFloat8_fnuz.value)
        assert result == DataTypeEnum.Float8_fnuz.value

    def test_getRealDataTypeA_bfloat8float8_fnuz(self):
        """Test BFloat8Float8_fnuz returns BFloat8_fnuz for A."""
        result = getRealDataTypeA(DataTypeEnum.BFloat8Float8_fnuz.value)
        assert result == DataTypeEnum.BFloat8_fnuz.value

    def test_getRealDataTypeA_passthrough(self):
        """Test other data types pass through unchanged."""
        result = getRealDataTypeA(DataTypeEnum.Float.value)
        assert result == DataTypeEnum.Float.value

    def test_getRealDataTypeB_float8bfloat8(self):
        """Test Float8BFloat8 returns BFloat8 for B."""
        result = getRealDataTypeB(DataTypeEnum.Float8BFloat8.value)
        assert result == DataTypeEnum.BFloat8.value

    def test_getRealDataTypeB_bfloat8float8(self):
        """Test BFloat8Float8 returns Float8 for B."""
        result = getRealDataTypeB(DataTypeEnum.BFloat8Float8.value)
        assert result == DataTypeEnum.Float8.value

    def test_getRealDataTypeB_float8bfloat8_fnuz(self):
        """Test Float8BFloat8_fnuz returns BFloat8_fnuz for B."""
        result = getRealDataTypeB(DataTypeEnum.Float8BFloat8_fnuz.value)
        assert result == DataTypeEnum.BFloat8_fnuz.value

    def test_getRealDataTypeB_bfloat8float8_fnuz(self):
        """Test BFloat8Float8_fnuz returns Float8_fnuz for B."""
        result = getRealDataTypeB(DataTypeEnum.BFloat8Float8_fnuz.value)
        assert result == DataTypeEnum.Float8_fnuz.value

    def test_getRealDataTypeB_passthrough(self):
        """Test other data types pass through unchanged."""
        result = getRealDataTypeB(DataTypeEnum.Float.value)
        assert result == DataTypeEnum.Float.value


class TestParseLibraryLogicList:
    """Tests for parseLibraryLogicList function."""

    def test_exits_on_short_data(self):
        """Test exits when data has less than 9 fields."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]  # Only 8 elements

        with pytest.raises(SystemExit):
            parseLibraryLogicList(data)

    def test_parses_minimum_fields(self):
        """Test parses data with minimum required fields."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},  # 0
            "ScheduleName",  # 1
            "gfx900",  # 2
            ["Device1"],  # 3
            {"DataType": 0},  # 4
            [],  # 5 - Solutions
            None,  # 6 - indexOrder
            None,  # 7 - exactLogic
            None,  # 8 - rangeLogic
            None,  # 9
            None,  # 10
            "FreeSize",  # 11 - library type (required)
        ]

        result = parseLibraryLogicList(data)

        assert result["MinimumRequiredVersion"] == "1.0.0"
        assert result["ScheduleName"] == "ScheduleName"
        assert result["ArchitectureName"] == "gfx900"
        assert result["DeviceNames"] == ["Device1"]

    def test_parses_architecture_dict(self):
        """Test parses architecture as dict with CUCount."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            {"Architecture": "gfx90a", "CUCount": 110},
            ["Device1"],
            {"DataType": 0},
            [],
            None,
            None,
            None,
            None,
            None,
            "FreeSize",  # library type (required)
        ]

        result = parseLibraryLogicList(data)

        assert result["ArchitectureName"] == "gfx90a"
        assert result["CUCount"] == 110

    def test_parses_optional_perf_metric(self):
        """Test parses optional PerfMetric field."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            "gfx900",
            ["Device1"],
            {"DataType": 0},
            [],
            None,
            None,
            None,
            None,
            {"metric": "value"},  # PerfMetric
            "FreeSize",  # library type (required)
        ]

        result = parseLibraryLogicList(data)

        assert result["PerfMetric"] == {"metric": "value"}

    def test_exits_on_missing_library_type(self):
        """Test exits when library type field is missing."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            "gfx900",
            ["Device1"],
            {"DataType": 0},
            [],
            None,
            None,
            None,
            None,
            None,
            None,  # Missing library type
        ]

        with pytest.raises(SystemExit):
            parseLibraryLogicList(data)

    def test_parses_freesize_library_type(self):
        """Test parses FreeSize library type."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            "gfx900",
            ["Device1"],
            {"DataType": 0},
            [1, 2, 3],  # 3 solutions
            None,
            None,
            None,
            None,
            None,
            "FreeSize",
        ]

        result = parseLibraryLogicList(data)

        assert result["LibraryType"] == "FreeSize"
        assert result["Library"]["table"] == [0, 3]

    def test_parses_prediction_library_type(self):
        """Test parses Prediction library type."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            "gfx900",
            ["Device1"],
            {"DataType": 0},
            [1, 2],  # 2 solutions
            None,
            None,
            None,
            None,
            None,
            "Prediction",
        ]

        result = parseLibraryLogicList(data)

        assert result["LibraryType"] == "Prediction"
        assert result["Library"]["table"] == [0, 2]

    def test_parses_matching_library_type(self):
        """Test parses Matching library type."""
        data = [
            {"MinimumRequiredVersion": "1.0.0"},
            "ScheduleName",
            "gfx900",
            ["Device1"],
            {"DataType": 0},
            [],
            ["index1", "index2"],
            {"exact": "logic"},
            None,
            None,
            None,
            "Euclidean",  # distance type
        ]

        result = parseLibraryLogicList(data)

        assert result["LibraryType"] == "Matching"
        assert result["Library"]["indexOrder"] == ["index1", "index2"]
        assert result["Library"]["distance"] == "Euclidean"


class TestRawLibraryLogic:
    """Tests for rawLibraryLogic function."""

    def test_returns_tuple_with_9_elements(self):
        """Test returns tuple with 9 base elements for minimum data."""
        data = [
            "version",
            "schedule",
            "arch",
            ["device"],
            {"problem": "type"},
            [],
            None,
            None,
            None,
        ]

        result = rawLibraryLogic(data)

        assert len(result) == 10  # 9 fields + otherFields list
        assert result[0] == "version"
        assert result[1] == "schedule"
        assert result[2] == "arch"
        assert result[9] == []  # otherFields empty

    def test_includes_other_fields(self):
        """Test includes additional fields in otherFields."""
        data = [
            "version",
            "schedule",
            "arch",
            ["device"],
            {"problem": "type"},
            [],
            None,
            None,
            None,
            "extra1",
            "extra2",
        ]

        result = rawLibraryLogic(data)

        assert result[9] == ["extra1", "extra2"]


class TestFindBodyOffset:
    """Tests for _findBodyOffset function."""

    def test_finds_first_non_header_line(self, tmp_path):
        """Test finds offset of first solution entry."""
        yaml_file = tmp_path / "test.yaml"
        content = """- MinimumRequiredVersion: 1.0.0
- ProblemSizes: []
- SolutionKey: value
"""
        yaml_file.write_text(content)

        offset = _findBodyOffset(str(yaml_file), {"MinimumRequiredVersion", "ProblemSizes"})

        # Should point to "- SolutionKey: value\n"
        with open(yaml_file) as f:
            f.seek(offset)
            line = f.readline()
        assert line.startswith("- SolutionKey")

    def test_returns_end_if_no_body(self, tmp_path):
        """Test returns end position if only header present."""
        yaml_file = tmp_path / "test.yaml"
        content = """- MinimumRequiredVersion: 1.0.0
- ProblemSizes: []
"""
        yaml_file.write_text(content)

        offset = _findBodyOffset(str(yaml_file), {"MinimumRequiredVersion", "ProblemSizes"})

        # Should point to EOF
        assert offset == len(content)


class TestGetCUCount:
    """Tests for getCUCount function."""

    def test_reads_from_environment(self, monkeypatch):
        """Test reads CU count from environment variable."""
        monkeypatch.setenv("CU", "64")

        result = getCUCount()

        assert result == 64

    @patch('Tensile.LibraryIO.subprocess.run')
    def test_reads_from_rocminfo_when_env_not_set(self, mock_run):
        """Test reads from rocminfo when CU env var not set."""
        mock_result = MagicMock()
        mock_result.stdout = b"  Compute Unit:     110\n"
        mock_run.return_value = mock_result

        result = getCUCount()

        assert result == 110

    @patch('Tensile.LibraryIO.subprocess.run')
    def test_exits_when_rocminfo_fails(self, mock_run):
        """Test exits when rocminfo fails and no env var."""
        mock_run.side_effect = Exception("rocminfo failed")

        with pytest.raises(SystemExit):
            getCUCount()
