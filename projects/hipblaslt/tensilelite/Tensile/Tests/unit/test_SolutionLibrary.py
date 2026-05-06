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

from Tensile.SolutionLibrary import (
    SingleSolutionLibrary,
    IndexSolutionLibrary,
    PlaceholderLibrary,
    MatchingLibrary,
    FreeSizeLibrary,
    PredictionLibrary
)


class TestSingleSolutionLibrary:
    """Tests for SingleSolutionLibrary class."""

    def test_initialization(self):
        """Test SingleSolutionLibrary initialization."""
        solution = MagicMock()
        solution.index = 42

        lib = SingleSolutionLibrary(solution)

        assert lib.solution == solution
        assert lib.tag == "Single"

    def test_state_returns_correct_structure(self):
        """Test state() returns correct structure."""
        solution = MagicMock()
        solution.index = 10

        lib = SingleSolutionLibrary(solution)
        state = lib.state()

        assert state["type"] == "Single"
        assert state["index"] == 10

    def test_remap_solution_indices_is_noop(self):
        """Test remapSolutionIndices is a no-op."""
        solution = MagicMock()
        lib = SingleSolutionLibrary(solution)

        # Should not raise
        lib.remapSolutionIndices({})


class TestIndexSolutionLibrary:
    """Tests for IndexSolutionLibrary class."""

    def test_tag_is_index(self):
        """Test IndexSolutionLibrary has correct tag."""
        solution = MagicMock()
        lib = IndexSolutionLibrary(solution)

        assert lib.tag == "Index"

    def test_state_returns_solution_index(self):
        """Test state() returns just the solution index."""
        solution = MagicMock()
        solution.index = 25

        lib = IndexSolutionLibrary(solution)
        state = lib.state()

        assert state == 25


class TestPlaceholderLibrary:
    """Tests for PlaceholderLibrary class."""

    def test_initialization(self):
        """Test PlaceholderLibrary initialization."""
        lib = PlaceholderLibrary("test_kernel")

        assert lib.filenamePrefix == "test_kernel"
        assert lib.tag == "Placeholder"

    def test_state_returns_correct_structure(self):
        """Test state() returns correct structure."""
        lib = PlaceholderLibrary("my_kernel")
        state = lib.state()

        assert state["type"] == "Placeholder"
        assert state["value"] == "my_kernel"

    def test_remap_solution_indices_is_noop(self):
        """Test remapSolutionIndices is a no-op."""
        lib = PlaceholderLibrary("kernel")

        # Should not raise
        lib.remapSolutionIndices({})

    def test_merge_is_noop(self):
        """Test merge is a no-op."""
        lib1 = PlaceholderLibrary("kernel1")
        lib2 = PlaceholderLibrary("kernel2")

        # Should not raise
        lib1.merge(lib2)


class TestFreeSizeLibrary:
    """Tests for FreeSizeLibrary class."""

    def test_initialization(self):
        """Test FreeSizeLibrary initialization."""
        table = [{"index": 1}, {"index": 2}]
        lib = FreeSizeLibrary(table)

        assert lib.table == table
        assert lib.tag == "FreeSize"

    def test_from_original_state(self):
        """Test FromOriginalState creates library correctly."""
        solutions = []
        for i in range(5):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {"table": [0, 3]}  # Start at index 0, count 3

        lib = FreeSizeLibrary.FromOriginalState(d, solutions)

        assert len(lib.table) == 3
        assert lib.tag == "FreeSize"

    def test_from_original_state_with_offset(self):
        """Test FromOriginalState with offset start index."""
        solutions = []
        for i in range(10):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {"table": [5, 3]}  # Start at index 5, count 3

        lib = FreeSizeLibrary.FromOriginalState(d, solutions)

        assert len(lib.table) == 3

    def test_merge_combines_tables(self):
        """Test merge combines two FreeSizeLibrary tables."""
        table1 = [{"index": 1}, {"index": 2}]
        table2 = [{"index": 3}, {"index": 4}]

        lib1 = FreeSizeLibrary(table1)
        lib2 = FreeSizeLibrary(table2)

        lib1.merge(lib2)

        assert len(lib1.table) == 4

    def test_remap_solution_indices_is_noop(self):
        """Test remapSolutionIndices is a no-op."""
        lib = FreeSizeLibrary([])

        # Should not raise
        lib.remapSolutionIndices({})


class TestPredictionLibrary:
    """Tests for PredictionLibrary class."""

    def test_initialization(self):
        """Test PredictionLibrary initialization."""
        table = [{"index": 1}, {"index": 2}]
        lib = PredictionLibrary(table)

        assert lib.table == table
        assert lib.tag == "Prediction"

    def test_from_original_state(self):
        """Test FromOriginalState creates library correctly."""
        solutions = []
        for i in range(5):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {"table": [1, 2]}  # Start at index 1, count 2

        lib = PredictionLibrary.FromOriginalState(d, solutions)

        assert len(lib.table) == 2
        assert lib.tag == "Prediction"

    def test_merge_combines_tables(self):
        """Test merge combines two PredictionLibrary tables."""
        table1 = [{"index": 1}]
        table2 = [{"index": 2}, {"index": 3}]

        lib1 = PredictionLibrary(table1)
        lib2 = PredictionLibrary(table2)

        lib1.merge(lib2)

        assert len(lib1.table) == 3

    def test_remap_solution_indices_is_noop(self):
        """Test remapSolutionIndices is a no-op."""
        lib = PredictionLibrary([])

        # Should not raise
        lib.remapSolutionIndices({})


class TestMatchingLibrary:
    """Tests for MatchingLibrary class."""

    def test_initialization(self):
        """Test MatchingLibrary initialization."""
        properties = ["prop1", "prop2"]
        table = [{"key": [1, 2], "index": 0, "speed": 1.0}]
        distance = "Equality"

        lib = MatchingLibrary(properties, table, distance)

        assert lib.properties == properties
        assert lib.table == table
        assert lib.distance == distance
        assert lib.tag == "Matching"

    def test_from_original_state_equality(self):
        """Test FromOriginalState with Equality distance."""
        solutions = []
        for i in range(3):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {
            "indexOrder": [0, 2, 3, 1],
            "distance": "Equality",
            "table": [
                [[100, 200, 300, 400], [0, 1.5]],
                [[150, 250, 350, 450], [1, 2.0]]
            ]
        }

        lib = MatchingLibrary.FromOriginalState(d, solutions)

        assert lib.distance == "Equality"
        assert len(lib.table) == 2
        # keyOrder = [0,1,2,3] since all indices values are in propertyKeys
        # key = row[0] for indices in keyOrder = [100, 200, 300, 400]
        assert lib.table[0]["key"] == [100, 200, 300, 400]

    def test_from_original_state_gridbased(self):
        """Test FromOriginalState with GridBased distance."""
        solutions = []
        for i in range(2):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {
            "indexOrder": [0, 2, 3],
            "distance": "GridBased",
            "table": [
                [[100, 200, 300], [0, 1.0]],
            ]
        }

        lib = MatchingLibrary.FromOriginalState(d, solutions)

        assert lib.distance == "GridBased"
        assert "speed" not in lib.table[0]  # GridBased doesn't include speed

    def test_from_original_state_range(self):
        """Test FromOriginalState with Range distance."""
        solutions = []
        for i in range(2):
            sol = MagicMock()
            sol.index = i
            solutions.append(sol)

        d = {
            "indexOrder": [0, 1, 2, 3, 4, 5, 6, 7],
            "distance": "Range",
            "table": [
                [[100, 200, 300, 400, 500, 600, 700, 800], [0, 1.5]],
            ]
        }

        lib = MatchingLibrary.FromOriginalState(d, solutions)

        assert lib.distance == "Range"
        assert len(lib.properties) == 8  # Range uses all 8 properties

    def test_merge_combines_tables(self):
        """Test merge combines two MatchingLibrary tables."""
        properties = ["prop1"]
        table1 = [{"key": [1], "index": 0, "speed": 1.0}]
        table2 = [{"key": [2], "index": 1, "speed": 2.0}]

        lib1 = MatchingLibrary(properties, table1, "Equality")
        lib2 = MatchingLibrary(properties, table2, "Equality")

        lib1.merge(lib2)

        assert len(lib1.table) == 2
        # Should be sorted by key
        assert lib1.table[0]["key"] == [1]
        assert lib1.table[1]["key"] == [2]

    def test_merge_asserts_on_incompatible(self):
        """Test merge asserts when libraries are incompatible."""
        lib1 = MatchingLibrary(["prop1"], [], "Equality")
        lib2 = MatchingLibrary(["prop2"], [], "Equality")  # Different properties

        with pytest.raises(AssertionError):
            lib1.merge(lib2)

    def test_table_sorted_after_merge(self):
        """Test table is sorted after merge."""
        properties = ["prop1"]
        table1 = [{"key": [3], "index": 0, "speed": 1.0}]
        table2 = [{"key": [1], "index": 1, "speed": 2.0}]

        lib1 = MatchingLibrary(properties, table1, "Equality")
        lib2 = MatchingLibrary(properties, table2, "Equality")

        lib1.merge(lib2)

        # Should be sorted by key
        assert lib1.table[0]["key"] == [1]
        assert lib1.table[1]["key"] == [3]

    def test_remap_solution_indices_is_noop(self):
        """Test remapSolutionIndices is a no-op."""
        lib = MatchingLibrary([], [], "Equality")

        # Should not raise
        lib.remapSolutionIndices({})
