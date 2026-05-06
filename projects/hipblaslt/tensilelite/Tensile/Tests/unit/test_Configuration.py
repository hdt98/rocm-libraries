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
from copy import copy, deepcopy

pytestmark = pytest.mark.unit

from Tensile.Configuration import ReadWriteTransformDict


class TestReadWriteTransformDict:
    """Tests for ReadWriteTransformDict class."""

    def test_initialization_empty(self):
        """Test initialization without transform functions."""
        d = ReadWriteTransformDict()

        assert len(d) == 0
        assert not d.hasReadTransform()
        assert not d.hasWriteTransform()

    def test_initialization_with_read_transform(self):
        """Test initialization with read transform function."""
        def read_func(name):
            return None

        d = ReadWriteTransformDict(readTransformFunc=read_func)

        assert d.hasReadTransform()
        assert not d.hasWriteTransform()

    def test_initialization_with_write_transform(self):
        """Test initialization with write transform function."""
        def write_func(name, value):
            pass

        d = ReadWriteTransformDict(writeTransformFunc=write_func)

        assert not d.hasReadTransform()
        assert d.hasWriteTransform()

    def test_initialization_with_both_transforms(self):
        """Test initialization with both transform functions."""
        def read_func(name):
            return None

        def write_func(name, value):
            pass

        d = ReadWriteTransformDict(readTransformFunc=read_func, writeTransformFunc=write_func)

        assert d.hasReadTransform()
        assert d.hasWriteTransform()

    def test_basic_getitem_setitem(self):
        """Test basic dictionary get/set operations."""
        d = ReadWriteTransformDict()
        d["key"] = "value"

        assert d["key"] == "value"

    def test_getattr_setattr(self):
        """Test attribute-style access."""
        d = ReadWriteTransformDict()
        d.key = "value"

        assert d.key == "value"
        assert d["key"] == "value"

    def test_read_transform_is_applied(self):
        """Test read transform function is applied on access."""
        def read_transform(self, name):
            # Add prefix to all reads
            raw_value = ReadWriteTransformDict.readNoTransform(self, name)
            return f"transformed_{raw_value}"

        d = ReadWriteTransformDict(readTransformFunc=read_transform)
        d.writeNoTransform("key", "value")

        assert d["key"] == "transformed_value"

    def test_write_transform_is_applied(self):
        """Test write transform function is applied on write."""
        def write_transform(self, name, value):
            # Add suffix to all writes
            ReadWriteTransformDict.writeNoTransform(self, name, f"{value}_modified")

        d = ReadWriteTransformDict(writeTransformFunc=write_transform)
        d["key"] = "value"

        assert d.readNoTransform("key") == "value_modified"

    def test_get_with_default(self):
        """Test get method with default value."""
        d = ReadWriteTransformDict()
        d["exists"] = "value"

        assert d.get("exists") == "value"
        assert d.get("missing") is None
        assert d.get("missing", "default") == "default"

    def test_set_method(self):
        """Test set method for setting values."""
        d = ReadWriteTransformDict()
        d.set("key", "value")

        assert d["key"] == "value"

    def test_set_method_silences_exceptions(self):
        """Test set method catches and silences exceptions."""
        def write_transform(self, name, value):
            raise ValueError("Transform error")

        d = ReadWriteTransformDict(writeTransformFunc=write_transform)
        d.set("key", "value")  # Should not raise

        # Value should not be set due to exception
        with pytest.raises(KeyError):
            _ = d.readNoTransform("key")

    def test_copy_shallow(self):
        """Test shallow copy creates new dict with same values."""
        d = ReadWriteTransformDict()
        d["key"] = [1, 2, 3]

        d_copy = copy(d)

        assert d_copy is not d
        assert d_copy["key"] is d["key"]  # Same list object
        assert isinstance(d_copy, ReadWriteTransformDict)

    def test_deepcopy_creates_new_objects(self):
        """Test deepcopy creates completely independent copy."""
        d = ReadWriteTransformDict()
        d["key"] = [1, 2, 3]

        d_copy = deepcopy(d)

        assert d_copy is not d
        assert d_copy["key"] is not d["key"]  # Different list object
        assert d_copy["key"] == d["key"]  # But same values
        assert isinstance(d_copy, ReadWriteTransformDict)

    def test_deepcopy_handles_nested_dict(self):
        """Test deepcopy handles nested dictionaries correctly."""
        d = ReadWriteTransformDict()
        d["nested"] = {"inner": "value"}

        d_copy = deepcopy(d)

        assert d_copy is not d
        assert d_copy["nested"] is not d["nested"]
        assert d_copy["nested"] == d["nested"]

    def test_repr_shows_type_and_contents(self):
        """Test __repr__ shows readable representation."""
        d = ReadWriteTransformDict()
        d["key1"] = "value1"
        d["key2"] = "value2"

        repr_str = repr(d)

        assert "ReadWriteTransformDict" in repr_str
        assert "key1" in repr_str
        assert "value1" in repr_str

    def test_repr_nested_dicts(self):
        """Test __repr__ handles nested ReadWriteTransformDict."""
        outer = ReadWriteTransformDict()
        inner = ReadWriteTransformDict()
        inner["inner_key"] = "inner_value"
        outer["nested"] = inner

        repr_str = repr(outer)

        assert "ReadWriteTransformDict" in repr_str
        assert "nested" in repr_str
        assert "inner_key" in repr_str
        # Check indentation exists
        assert "\t" in repr_str

    def test_has_read_transform_detection(self):
        """Test hasReadTransform correctly detects presence."""
        d = ReadWriteTransformDict()
        assert not d.hasReadTransform()

        d.setReadTransform(lambda name: None)
        assert d.hasReadTransform()

    def test_has_write_transform_detection(self):
        """Test hasWriteTransform correctly detects presence."""
        d = ReadWriteTransformDict()
        assert not d.hasWriteTransform()

        d.setWriteTransform(lambda name, value: None)
        assert d.hasWriteTransform()

    def test_read_no_transform_bypasses_transform(self):
        """Test readNoTransform bypasses read transform."""
        def read_transform(self, name):
            return "transformed"

        d = ReadWriteTransformDict(readTransformFunc=read_transform)
        d.writeNoTransform("key", "original")

        assert d["key"] == "transformed"
        assert d.readNoTransform("key") == "original"

    def test_write_no_transform_bypasses_transform(self):
        """Test writeNoTransform bypasses write transform."""
        def write_transform(self, name, value):
            ReadWriteTransformDict.writeNoTransform(self, name, "transformed")

        d = ReadWriteTransformDict(writeTransformFunc=write_transform)
        d["key1"] = "value1"
        d.writeNoTransform("key2", "value2")

        assert d.readNoTransform("key1") == "transformed"
        assert d.readNoTransform("key2") == "value2"

    def test_multiple_keys(self):
        """Test dictionary with multiple keys."""
        d = ReadWriteTransformDict()
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3

        assert len(d) == 3
        assert set(d.keys()) == {"a", "b", "c"}
        assert set(d.values()) == {1, 2, 3}

    def test_update_values(self):
        """Test updating existing values."""
        d = ReadWriteTransformDict()
        d["key"] = "value1"
        d["key"] = "value2"

        assert d["key"] == "value2"

    def test_delete_key(self):
        """Test deleting keys from dictionary."""
        d = ReadWriteTransformDict()
        d["key"] = "value"

        del d["key"]

        assert "key" not in d
        with pytest.raises(KeyError):
            _ = d["key"]

    def test_in_operator(self):
        """Test 'in' operator for membership testing."""
        d = ReadWriteTransformDict()
        d["exists"] = "value"

        assert "exists" in d
        assert "missing" not in d

    def test_items_iteration(self):
        """Test iterating over items."""
        d = ReadWriteTransformDict()
        d["a"] = 1
        d["b"] = 2

        items = list(d.items())

        assert ("a", 1) in items
        assert ("b", 2) in items
        assert len(items) == 2

    def test_clear(self):
        """Test clear removes all items."""
        d = ReadWriteTransformDict()
        d["a"] = 1
        d["b"] = 2

        d.clear()

        assert len(d) == 0
        assert "a" not in d

    def test_pop(self):
        """Test pop removes and returns value."""
        d = ReadWriteTransformDict()
        d["key"] = "value"

        result = d.pop("key")

        assert result == "value"
        assert "key" not in d

    def test_setdefault(self):
        """Test setdefault sets value if key missing."""
        d = ReadWriteTransformDict()

        result1 = d.setdefault("new", "default")
        result2 = d.setdefault("new", "other")

        assert result1 == "default"
        assert result2 == "default"
        assert d["new"] == "default"
