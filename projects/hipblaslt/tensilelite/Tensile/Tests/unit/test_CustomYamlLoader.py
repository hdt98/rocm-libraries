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
from pathlib import Path

pytestmark = pytest.mark.unit

from Tensile import CustomYamlLoader


class TestIsFloat:
    """Tests for is_float helper function."""

    def test_is_float_valid_float(self):
        """Test is_float with valid float string."""
        assert CustomYamlLoader.is_float("3.14") is True

    def test_is_float_valid_negative_float(self):
        """Test is_float with negative float."""
        assert CustomYamlLoader.is_float("-3.14") is True

    def test_is_float_valid_scientific_notation(self):
        """Test is_float with scientific notation."""
        assert CustomYamlLoader.is_float("1e-5") is True

    def test_is_float_integer_string(self):
        """Test is_float with integer string (valid float)."""
        assert CustomYamlLoader.is_float("42") is True

    def test_is_float_invalid_string(self):
        """Test is_float with invalid string."""
        assert CustomYamlLoader.is_float("hello") is False

    def test_is_float_empty_string(self):
        """Test is_float with empty string."""
        assert CustomYamlLoader.is_float("") is False

    def test_is_float_mixed_string(self):
        """Test is_float with mixed alphanumeric string."""
        assert CustomYamlLoader.is_float("3.14abc") is False


class TestParseScalar:
    """Tests for parse_scalar function."""

    def test_parse_scalar_true_lowercase(self):
        """Test parsing 'true' returns boolean True."""
        yaml_str = "true"
        loader = yaml.SafeLoader(yaml_str)
        loader.check_event()  # StreamStartEvent
        loader.get_event()
        loader.check_event()  # DocumentStartEvent
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is True
        assert isinstance(result, bool)

    def test_parse_scalar_true_uppercase(self):
        """Test parsing 'TRUE' returns boolean True."""
        yaml_str = "TRUE"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()  # StreamStartEvent
        loader.get_event()  # DocumentStartEvent

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is True

    def test_parse_scalar_yes(self):
        """Test parsing 'yes' returns boolean True."""
        yaml_str = "yes"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is True

    def test_parse_scalar_false_lowercase(self):
        """Test parsing 'false' returns boolean False."""
        yaml_str = "false"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is False
        assert isinstance(result, bool)

    def test_parse_scalar_false_uppercase(self):
        """Test parsing 'FALSE' returns boolean False."""
        yaml_str = "FALSE"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is False

    def test_parse_scalar_no(self):
        """Test parsing 'no' returns boolean False."""
        yaml_str = "no"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is False

    def test_parse_scalar_null(self):
        """Test parsing 'null' returns None."""
        yaml_str = "null"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is None

    def test_parse_scalar_tilde(self):
        """Test parsing '~' returns None."""
        yaml_str = "~"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is None

    def test_parse_scalar_empty_unquoted(self):
        """Test parsing empty unquoted value returns None."""
        yaml_str = "key: "
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()
        loader.get_event()  # MappingStartEvent
        loader.get_event()  # key scalar

        result = CustomYamlLoader.parse_scalar(loader)
        assert result is None

    def test_parse_scalar_positive_integer(self):
        """Test parsing positive integer string."""
        yaml_str = "42"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == 42
        assert isinstance(result, int)

    def test_parse_scalar_negative_integer(self):
        """Test parsing negative integer string."""
        yaml_str = "-42"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == -42
        assert isinstance(result, int)

    def test_parse_scalar_positive_float(self):
        """Test parsing positive float string."""
        yaml_str = "3.14"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_scalar_negative_float(self):
        """Test parsing negative float string."""
        yaml_str = "-3.14"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == -3.14
        assert isinstance(result, float)

    def test_parse_scalar_scientific_notation(self):
        """Test parsing scientific notation."""
        yaml_str = "1e-5"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == 1e-5
        assert isinstance(result, float)

    def test_parse_scalar_regular_string(self):
        """Test parsing regular string value."""
        yaml_str = "hello"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == "hello"
        assert isinstance(result, str)

    def test_parse_scalar_quoted_null(self):
        """Test parsing quoted 'null' returns string not None."""
        yaml_str = "'null'"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_scalar(loader)
        assert result == "null"
        assert isinstance(result, str)


class TestParseSequence:
    """Tests for parse_sequence function."""

    def test_parse_sequence_empty(self):
        """Test parsing empty sequence."""
        yaml_str = "[]"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()  # StreamStartEvent
        loader.get_event()  # DocumentStartEvent

        result = CustomYamlLoader.parse_sequence(loader)
        assert result == []

    def test_parse_sequence_integers(self):
        """Test parsing sequence of integers."""
        yaml_str = "[1, 2, 3]"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_sequence(loader)
        assert result == [1, 2, 3]

    def test_parse_sequence_mixed_types(self):
        """Test parsing sequence with mixed types."""
        yaml_str = "[1, 3.14, true, null, hello]"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_sequence(loader)
        assert result == [1, 3.14, True, None, "hello"]

    def test_parse_sequence_nested(self):
        """Test parsing nested sequences."""
        yaml_str = "[[1, 2], [3, 4]]"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_sequence(loader)
        assert result == [[1, 2], [3, 4]]


class TestParseMapping:
    """Tests for parse_mapping function."""

    def test_parse_mapping_empty(self):
        """Test parsing empty mapping."""
        yaml_str = "{}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {}

    def test_parse_mapping_simple(self):
        """Test parsing simple mapping."""
        yaml_str = "{key: value}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {"key": "value"}

    def test_parse_mapping_multiple_entries(self):
        """Test parsing mapping with multiple entries."""
        yaml_str = "{a: 1, b: 2, c: 3}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_parse_mapping_mixed_types(self):
        """Test parsing mapping with mixed value types."""
        yaml_str = "{int: 42, float: 3.14, bool: true, str: hello}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {
            "int": 42,
            "float": 3.14,
            "bool": True,
            "str": "hello"
        }

    def test_parse_mapping_nested(self):
        """Test parsing nested mapping."""
        yaml_str = "{outer: {inner: value}}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {"outer": {"inner": "value"}}

    def test_parse_mapping_with_sequence_value(self):
        """Test parsing mapping with sequence as value."""
        yaml_str = "{key: [1, 2, 3]}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_mapping(loader)
        assert result == {"key": [1, 2, 3]}


class TestParseGeneral:
    """Tests for parse_general dispatcher function."""

    def test_parse_general_scalar(self):
        """Test parse_general with scalar."""
        yaml_str = "42"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_general(loader)
        assert result == 42

    def test_parse_general_sequence(self):
        """Test parse_general with sequence."""
        yaml_str = "[1, 2, 3]"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_general(loader)
        assert result == [1, 2, 3]

    def test_parse_general_mapping(self):
        """Test parse_general with mapping."""
        yaml_str = "{key: value}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_general(loader)
        assert result == {"key": "value"}

    def test_parse_general_complex_structure(self):
        """Test parse_general with complex nested structure."""
        yaml_str = "{items: [1, 2, {nested: true}], flag: false}"
        loader = yaml.SafeLoader(yaml_str)
        loader.get_event()
        loader.get_event()

        result = CustomYamlLoader.parse_general(loader)
        assert result == {
            "items": [1, 2, {"nested": True}],
            "flag": False
        }


class TestLoadYamlStream:
    """Tests for load_yaml_stream function."""

    def test_load_yaml_stream_simple_mapping(self, tmp_path):
        """Test loading simple YAML mapping from file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n")

        result = CustomYamlLoader.load_yaml_stream(yaml_file, yaml.SafeLoader)
        assert result == {"key": "value", "number": 42}

    def test_load_yaml_stream_simple_sequence(self, tmp_path):
        """Test loading simple YAML sequence from file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- item1\n- item2\n- 42\n")

        result = CustomYamlLoader.load_yaml_stream(yaml_file, yaml.SafeLoader)
        assert result == ["item1", "item2", 42]

    def test_load_yaml_stream_complex_structure(self, tmp_path):
        """Test loading complex YAML structure from file."""
        yaml_content = """
GlobalParameters:
  MinimumRequiredVersion: 4.4.0
  PrintLevel: 1

BenchmarkProblems:
  - # Problem 1
    OperationType: GEMM
    DataType: s
    TransposeA: false
    TransposeB: true
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_yaml_stream(yaml_file, yaml.SafeLoader)
        assert isinstance(result, dict)
        assert "GlobalParameters" in result
        assert "BenchmarkProblems" in result
        assert result["GlobalParameters"]["PrintLevel"] == 1

    def test_load_yaml_stream_type_preservation(self, tmp_path):
        """Test that load_yaml_stream preserves custom type conversions."""
        yaml_content = """
bool_true: yes
bool_false: no
int_value: 42
float_value: 3.14
null_value: null
string_value: hello
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_yaml_stream(yaml_file, yaml.SafeLoader)
        assert result["bool_true"] is True
        assert result["bool_false"] is False
        assert result["int_value"] == 42
        assert isinstance(result["int_value"], int)
        assert result["float_value"] == 3.14
        assert isinstance(result["float_value"], float)
        assert result["null_value"] is None
        assert result["string_value"] == "hello"


class TestLoadYamlSequenceItem:
    """Tests for load_yaml_sequence_item function."""

    def test_load_yaml_sequence_item_first_element(self, tmp_path):
        """Test loading first element from YAML sequence."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- first\n- second\n- third\n")

        result = CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 0)
        assert result == "first"

    def test_load_yaml_sequence_item_middle_element(self, tmp_path):
        """Test loading middle element from YAML sequence."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- first\n- second\n- third\n")

        result = CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 1)
        assert result == "second"

    def test_load_yaml_sequence_item_last_element(self, tmp_path):
        """Test loading last element from YAML sequence."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- first\n- second\n- third\n")

        result = CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 2)
        assert result == "third"

    def test_load_yaml_sequence_item_complex_element(self, tmp_path):
        """Test loading complex object from YAML sequence."""
        yaml_content = """
- simple_item
- {key: value, number: 42}
- [1, 2, 3]
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 1)
        assert result == {"key": "value", "number": 42}

    def test_load_yaml_sequence_item_out_of_bounds(self, tmp_path):
        """Test loading element beyond sequence length returns None."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- first\n- second\n")

        result = CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 10)
        assert result is None

    def test_load_yaml_sequence_item_not_sequence_raises(self, tmp_path):
        """Test loading from non-sequence raises RuntimeError."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\n")

        with pytest.raises(RuntimeError, match="Root of YAML is not a sequence"):
            CustomYamlLoader.load_yaml_sequence_item(yaml_file, yaml.SafeLoader, 0)


class TestLoadYamlDictItem:
    """Tests for load_yaml_dict_item function."""

    def test_load_yaml_dict_item_existing_key(self, tmp_path):
        """Test loading value by existing key."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: value2\n")

        result = CustomYamlLoader.load_yaml_dict_item(yaml_file, yaml.SafeLoader, "key1")
        assert result == "value1"

    def test_load_yaml_dict_item_second_key(self, tmp_path):
        """Test loading value by second key."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: value2\nkey3: value3\n")

        result = CustomYamlLoader.load_yaml_dict_item(yaml_file, yaml.SafeLoader, "key2")
        assert result == "value2"

    def test_load_yaml_dict_item_complex_value(self, tmp_path):
        """Test loading complex value from dict."""
        yaml_content = """
SimpleKey: simple_value
ComplexKey:
  nested: data
  list: [1, 2, 3]
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_yaml_dict_item(yaml_file, yaml.SafeLoader, "ComplexKey")
        assert result == {"nested": "data", "list": [1, 2, 3]}

    def test_load_yaml_dict_item_nonexistent_key(self, tmp_path):
        """Test loading nonexistent key returns None."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: value2\n")

        result = CustomYamlLoader.load_yaml_dict_item(yaml_file, yaml.SafeLoader, "nonexistent")
        assert result is None

    def test_load_yaml_dict_item_not_mapping_raises(self, tmp_path):
        """Test loading from non-mapping raises RuntimeError."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("- item1\n- item2\n")

        with pytest.raises(RuntimeError, match="Root of YAML is not a map"):
            CustomYamlLoader.load_yaml_dict_item(yaml_file, yaml.SafeLoader, "key")


class TestLoadLogicGfxArch:
    """Tests for load_logic_gfx_arch function."""

    def test_load_logic_gfx_arch_sequence_with_dict(self, tmp_path):
        """Test loading gfx arch from sequence with dict at index 2."""
        yaml_content = """
- item0
- item1
- Architecture: gfx90a
  Other: value
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_logic_gfx_arch(yaml_file, yaml.SafeLoader)
        assert result == "gfx90a"

    def test_load_logic_gfx_arch_sequence_with_scalar(self, tmp_path):
        """Test loading gfx arch from sequence with scalar at index 2."""
        yaml_content = """
- item0
- item1
- gfx942
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_logic_gfx_arch(yaml_file, yaml.SafeLoader)
        assert result == "gfx942"

    def test_load_logic_gfx_arch_fallback_to_dict(self, tmp_path):
        """Test fallback to ArchitectureName key when sequence fails."""
        yaml_content = """
ArchitectureName: gfx1100
OtherKey: value
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        result = CustomYamlLoader.load_logic_gfx_arch(yaml_file, yaml.SafeLoader)
        assert result == "gfx1100"

    def test_load_logic_gfx_arch_default_loader(self, tmp_path):
        """Test load_logic_gfx_arch uses DEFAULT_YAML_LOADER by default."""
        yaml_content = """
- item0
- item1
- gfx950
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        # Call without specifying loader
        result = CustomYamlLoader.load_logic_gfx_arch(yaml_file)
        assert result == "gfx950"


class TestDefaultYamlLoader:
    """Tests for DEFAULT_YAML_LOADER constant."""

    def test_default_yaml_loader_is_set(self):
        """Test that DEFAULT_YAML_LOADER is set to a valid loader."""
        assert CustomYamlLoader.DEFAULT_YAML_LOADER is not None
        # CSafeLoader and SafeLoader inherit from different base classes but are both valid
        loader_name = CustomYamlLoader.DEFAULT_YAML_LOADER.__name__
        assert loader_name in ('CSafeLoader', 'SafeLoader')

    def test_default_yaml_loader_is_safe(self):
        """Test that DEFAULT_YAML_LOADER is CSafeLoader or SafeLoader."""
        loader = CustomYamlLoader.DEFAULT_YAML_LOADER
        # Should be either CSafeLoader (if available) or SafeLoader
        assert loader in (yaml.CSafeLoader, yaml.SafeLoader) or loader.__name__ in ('CSafeLoader', 'SafeLoader')
