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
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from Tensile.Component import PartialMatch, LraTileProperties


class TestPartialMatch:
    """Tests for PartialMatch function."""

    def test_exact_value_match(self):
        """Test PartialMatch with exact values."""
        assert PartialMatch(42, 42) is True
        assert PartialMatch("value", "value") is True
        assert PartialMatch(True, True) is True

    def test_exact_value_mismatch(self):
        """Test PartialMatch with different values."""
        assert PartialMatch(42, 43) is False
        assert PartialMatch("value", "other") is False
        assert PartialMatch(True, False) is False

    def test_callable_pattern_returns_true(self):
        """Test PartialMatch with callable that returns True."""
        pattern = lambda x: x > 0
        assert PartialMatch(pattern, 10) is True
        assert PartialMatch(pattern, 1) is True

    def test_callable_pattern_returns_false(self):
        """Test PartialMatch with callable that returns False."""
        pattern = lambda x: x > 0
        assert PartialMatch(pattern, -5) is False
        assert PartialMatch(pattern, 0) is False

    def test_dict_pattern_all_keys_present(self):
        """Test PartialMatch with dict where all keys match."""
        pattern = {"key1": "value1", "key2": "value2"}
        obj = {"key1": "value1", "key2": "value2", "key3": "extra"}

        assert PartialMatch(pattern, obj) is True

    def test_dict_pattern_missing_key(self):
        """Test PartialMatch with dict missing required key."""
        pattern = {"key1": "value1", "key2": "value2"}
        obj = {"key1": "value1"}

        assert PartialMatch(pattern, obj) is False

    def test_dict_pattern_wrong_value(self):
        """Test PartialMatch with dict having wrong value."""
        pattern = {"key1": "value1"}
        obj = {"key1": "wrong"}

        assert PartialMatch(pattern, obj) is False

    def test_nested_dict_pattern(self):
        """Test PartialMatch with nested dictionaries."""
        pattern = {"outer": {"inner": "value"}}
        obj = {"outer": {"inner": "value", "extra": "data"}}

        assert PartialMatch(pattern, obj) is True

    def test_nested_dict_pattern_mismatch(self):
        """Test PartialMatch with nested dict mismatch."""
        pattern = {"outer": {"inner": "value"}}
        obj = {"outer": {"inner": "wrong"}}

        assert PartialMatch(pattern, obj) is False

    def test_callable_in_dict_pattern(self):
        """Test PartialMatch with callable in dict pattern."""
        pattern = {"key": lambda x: x > 10}
        obj = {"key": 20}

        assert PartialMatch(pattern, obj) is True

    def test_callable_in_dict_pattern_fails(self):
        """Test PartialMatch with callable in dict that fails."""
        pattern = {"key": lambda x: x > 10}
        obj = {"key": 5}

        assert PartialMatch(pattern, obj) is False

    def test_complex_nested_pattern(self):
        """Test PartialMatch with complex nested structure."""
        pattern = {
            "level1": {
                "level2": {
                    "value": lambda x: x % 2 == 0
                }
            }
        }
        obj = {
            "level1": {
                "level2": {
                    "value": 10,
                    "other": "data"
                }
            }
        }

        assert PartialMatch(pattern, obj) is True

    def test_debug_mode_prints(self, capsys):
        """Test PartialMatch debug mode prints messages."""
        pattern = {"key": "value"}
        obj = {"key": "wrong"}

        result = PartialMatch(pattern, obj, debug=True)

        captured = capsys.readouterr()
        assert result is False
        assert "!=" in captured.out or "recursing" in captured.out

    def test_debug_mode_with_success(self, capsys):
        """Test PartialMatch debug mode with successful match."""
        pattern = {"key": "value"}
        obj = {"key": "value"}

        result = PartialMatch(pattern, obj, debug=True)

        captured = capsys.readouterr()
        assert result is True
        assert "True" in captured.out

    def test_empty_dict_pattern(self):
        """Test PartialMatch with empty dict pattern."""
        pattern = {}
        obj = {"key": "value"}

        assert PartialMatch(pattern, obj) is True

    def test_both_empty_dicts(self):
        """Test PartialMatch with both empty dicts."""
        pattern = {}
        obj = {}

        assert PartialMatch(pattern, obj) is True

    def test_none_pattern_matches_none(self):
        """Test PartialMatch with None values."""
        assert PartialMatch(None, None) is True

    def test_none_pattern_mismatches_value(self):
        """Test PartialMatch None vs value."""
        assert PartialMatch(None, "value") is False

    def test_list_not_treated_as_mapping(self):
        """Test PartialMatch with lists (not treated as Mapping)."""
        # Lists are not Mappings, so they use exact match
        assert PartialMatch([1, 2], [1, 2]) is True
        assert PartialMatch([1, 2], [1, 2, 3]) is False


class TestLraTileProperties:
    """Tests for LraTileProperties dataclass."""

    def test_initialization(self):
        """Test LraTileProperties can be instantiated."""
        # LraTileProperties is an empty dataclass, just verify it exists
        props = LraTileProperties()
        assert props is not None
        assert isinstance(props, LraTileProperties)
