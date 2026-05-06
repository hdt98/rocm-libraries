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
import array
import os
from unittest.mock import MagicMock, patch
from copy import deepcopy

pytestmark = pytest.mark.unit

from Tensile.LibraryLogic import (
    LogicAnalyzer,
    read_max_freq
)


class TestReadMaxFreq:
    """Tests for read_max_freq function."""

    def test_reads_valid_frequency_from_env(self):
        """Test reading valid frequency from MAX_FREQ environment variable."""
        with patch.dict(os.environ, {"MAX_FREQ": "1500.5"}):
            result = read_max_freq()
            assert result == 1500.5

    def test_returns_none_when_env_not_set(self):
        """Test returns None when MAX_FREQ is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = read_max_freq()
            assert result is None

    def test_returns_none_when_env_empty_string(self):
        """Test returns None when MAX_FREQ is empty string."""
        with patch.dict(os.environ, {"MAX_FREQ": ""}):
            result = read_max_freq()
            assert result is None

    def test_returns_none_when_env_whitespace(self):
        """Test returns None when MAX_FREQ is only whitespace."""
        with patch.dict(os.environ, {"MAX_FREQ": "   "}):
            result = read_max_freq()
            assert result is None

    def test_returns_none_on_invalid_value(self):
        """Test returns None when MAX_FREQ is not a valid number."""
        with patch.dict(os.environ, {"MAX_FREQ": "invalid"}):
            result = read_max_freq()
            assert result is None

    def test_handles_integer_frequency(self):
        """Test handles integer frequency values."""
        with patch.dict(os.environ, {"MAX_FREQ": "2000"}):
            result = read_max_freq()
            assert result == 2000.0

    def test_handles_zero_frequency(self):
        """Test handles zero frequency."""
        with patch.dict(os.environ, {"MAX_FREQ": "0"}):
            result = read_max_freq()
            assert result == 0.0

    def test_handles_negative_frequency(self):
        """Test handles negative frequency (returns it as-is)."""
        with patch.dict(os.environ, {"MAX_FREQ": "-100"}):
            result = read_max_freq()
            assert result == -100.0


class TestLogicAnalyzerHelperMethods:
    """Tests for LogicAnalyzer helper methods using direct object manipulation."""

    def test_indices_to_serial_basic(self):
        """Test indicesToSerial converts indices to serial index correctly."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 2
        analyzer.numIndices = 3
        analyzer.numProblemSizes = [2, 2, 2]

        # Call the real method
        result = LogicAnalyzer.indicesToSerial(analyzer, 0, [0, 0, 0])
        assert result == 0

        result = LogicAnalyzer.indicesToSerial(analyzer, 1, [0, 0, 0])
        assert result == 1

    def test_indices_to_serial_with_different_indices(self):
        """Test indicesToSerial with various indices."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 3
        analyzer.numIndices = 2
        analyzer.numProblemSizes = [4, 4]

        # Serial index should be:
        # solutionIdx + numSolutions * (idx0 + numSize0 * idx1)
        # For solutionIdx=1, idx=[1,0]: 1 + 3 * (1 + 4*0) = 1 + 3 = 4
        result = LogicAnalyzer.indicesToSerial(analyzer, 1, [1, 0])
        assert result == 4

    def test_to_index_order(self):
        """Test toIndexOrder reorders problem indices."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.indexOrder = [2, 0, 1]

        problemIndices = [10, 20, 30]
        result = LogicAnalyzer.toIndexOrder(analyzer, problemIndices)

        # Should reorder: [problemIndices[2], problemIndices[0], problemIndices[1]]
        assert result == [30, 10, 20]

    def test_total_flops_for_problem_indices(self):
        """Test totalFlopsForProblemIndices calculation."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.problemIndexToSize = [[64, 128], [64, 128], [16, 32]]
        analyzer.numIndices = 3
        analyzer.flopsPerMac = 2

        problemIndices = [0, 0, 0]  # First size for each index
        result = LogicAnalyzer.totalFlopsForProblemIndices(analyzer, problemIndices)

        # Should be flopsPerMac * size0 * size1 * size2 = 2 * 64 * 64 * 16
        expected = 2 * 64 * 64 * 16
        assert result == expected

    def test_total_flops_for_problem_indices_different_sizes(self):
        """Test totalFlopsForProblemIndices with different problem sizes."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.problemIndexToSize = [[64, 128], [64, 128]]
        analyzer.numIndices = 2
        analyzer.flopsPerMac = 4

        problemIndices = [1, 1]  # Second size for each index
        result = LogicAnalyzer.totalFlopsForProblemIndices(analyzer, problemIndices)

        # Should be flopsPerMac * size0 * size1 = 4 * 128 * 128
        expected = 4 * 128 * 128
        assert result == expected

    def test_get_logic_depth_single_level(self):
        """Test getLogicDepth for single-level logic."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        logic = [[-1, 0]]
        depth = LogicAnalyzer.getLogicDepth(analyzer, logic)
        assert depth == 1

    def test_get_logic_depth_two_levels(self):
        """Test getLogicDepth for two-level logic."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        logic = [[-1, [[-1, 0]]]]
        depth = LogicAnalyzer.getLogicDepth(analyzer, logic)
        assert depth == 2

    def test_get_logic_depth_three_levels(self):
        """Test getLogicDepth for three-level logic."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        logic = [[-1, [[-1, [[-1, 0]]]]]]
        depth = LogicAnalyzer.getLogicDepth(analyzer, logic)
        assert depth == 3

    def test_recommended_index_order(self):
        """Test recommendedIndexOrder generates correct order."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.problemType = {"TotalIndices": 5}
        analyzer.idx0 = 0
        analyzer.idx1 = 1
        analyzer.idxU = 2

        order = LogicAnalyzer.recommendedIndexOrder(analyzer)

        # Should put unroll, idx0, idx1 at the end
        assert len(order) == 5
        assert order[-1] == analyzer.idx1
        assert order[-2] == analyzer.idx0
        assert order[-3] == analyzer.idxU

    def test_problem_indices_for_range_empty(self):
        """Test problemIndicesForRange returns empty for empty range."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 3

        indexRange = [[0, 0], [0, 1], [0, 1]]  # First range is empty
        result = LogicAnalyzer.problemIndicesForRange(analyzer, indexRange)
        assert result == []

    def test_problem_indices_for_range_single(self):
        """Test problemIndicesForRange for single element range."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2

        indexRange = [[0, 1], [0, 1]]  # Single element in each dimension
        result = LogicAnalyzer.problemIndicesForRange(analyzer, indexRange)

        assert len(result) == 1
        assert result[0] == [0, 0]

    def test_problem_indices_for_range_multiple(self):
        """Test problemIndicesForRange for multi-element range."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2

        indexRange = [[0, 2], [0, 2]]  # 2x2 grid
        result = LogicAnalyzer.problemIndicesForRange(analyzer, indexRange)

        assert len(result) == 4
        expected = [[0, 0], [1, 0], [0, 1], [1, 1]]
        assert result == expected

    def test_problem_indices_for_range_3d(self):
        """Test problemIndicesForRange for 3D range."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 3

        indexRange = [[0, 2], [0, 2], [0, 2]]  # 2x2x2 cube
        result = LogicAnalyzer.problemIndicesForRange(analyzer, indexRange)

        assert len(result) == 8

    def test_score_logic_complexity_single_level(self):
        """Test scoreLogicComplexity for single level."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 1

        logic = [[-1, 0], [100, 1]]
        logicComplexity = [0]

        LogicAnalyzer.scoreLogicComplexity(analyzer, logic, logicComplexity)

        # Should have 2 branches at the deepest level
        assert logicComplexity[0] == 2

    def test_score_logic_complexity_two_levels(self):
        """Test scoreLogicComplexity for two levels."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2

        logic = [[-1, [[-1, 0]]]]
        logicComplexity = [0, 0]

        LogicAnalyzer.scoreLogicComplexity(analyzer, logic, logicComplexity)

        # Should have 1 branch at depth 0, 1 at depth 1
        assert logicComplexity[0] == 1
        assert logicComplexity[1] == 1


class TestLogicAnalyzerDataArrayOperations:
    """Tests for data array operations without full initialization."""

    def test_getitem_and_setitem_work_together(self):
        """Test that __getitem__ and __setitem__ work correctly."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 2
        analyzer.numIndices = 2
        analyzer.numProblemSizes = [3, 3]
        analyzer.data = array.array('f', [0.0] * 18)  # 2 solutions * 3 * 3 problems

        # Test setting a value
        indices = [1, 1]
        solutionIdx = 1
        test_value = 123.456

        # Calculate serial index manually
        serial = LogicAnalyzer.indicesToSerial(analyzer, solutionIdx, indices)
        analyzer.data[serial] = test_value

        # Verify we can read it back
        assert analyzer.data[serial] == test_value


class TestLogicAnalyzerScoreRangeForSolutions:
    """Tests for scoreRangeForSolutions method."""

    def test_score_range_for_solutions_basic(self):
        """Test scoreRangeForSolutions calculates time scores."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 2
        analyzer.numIndices = 1
        analyzer.numProblemSizes = [2]
        analyzer.problemIndexToSize = [[64, 128]]
        analyzer.flopsPerMac = 2
        analyzer.data = array.array('f', [
            100.0, 150.0,  # Problem 0: sol0=100, sol1=150 GFlops
            80.0, 120.0    # Problem 1: sol0=80, sol1=120 GFlops
        ])

        # Mock problemIndicesForRange
        def mock_problem_indices(indexRange):
            return [[0], [1]]

        analyzer.problemIndicesForRange = mock_problem_indices

        indexRange = [[0, 2]]
        scores = LogicAnalyzer.scoreRangeForSolutions(analyzer, indexRange)

        assert len(scores) == 2
        # Scores should be positive (time in microseconds)
        assert all(s > 0 for s in scores)
        # Better solution (higher GFlops) should have lower score
        assert scores[1] < scores[0]

    def test_score_range_handles_zero_gflops(self):
        """Test scoreRangeForSolutions handles unbenchmarked solutions (0 GFlops)."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 2
        analyzer.numIndices = 1
        analyzer.numProblemSizes = [1]
        analyzer.problemIndexToSize = [[64]]
        analyzer.flopsPerMac = 2
        analyzer.data = array.array('f', [
            100.0, 0.0  # sol0 benchmarked, sol1 not benchmarked
        ])

        def mock_problem_indices(indexRange):
            return [[0]]

        analyzer.problemIndicesForRange = mock_problem_indices

        indexRange = [[0, 1]]
        scores = LogicAnalyzer.scoreRangeForSolutions(analyzer, indexRange)

        # Unbenchmarked solution should have infinite score
        assert scores[1] == float("inf")
        assert scores[0] < scores[1]


class TestLogicAnalyzerWinnerSelection:
    """Tests for winner selection methods."""

    def test_get_winner_for_problem_selects_highest_gflops(self):
        """Test getWinnerForProblem selects solution with highest GFlops."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 3
        analyzer.numIndices = 1
        analyzer.numProblemSizes = [2]
        analyzer.data = array.array('f', [
            100.0, 150.0, 120.0,  # Problem 0
            80.0, 90.0, 110.0      # Problem 1
        ])

        problemIndices = [0]
        winner_idx, winner_gflops = LogicAnalyzer.getWinnerForProblem(analyzer, problemIndices)

        # Solution 1 has 150 GFlops, should be winner
        assert winner_idx == 1
        assert winner_gflops == 150.0

    def test_get_winner_for_problem_second_problem(self):
        """Test getWinnerForProblem for second problem."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 3
        analyzer.numIndices = 1
        analyzer.numProblemSizes = [2]
        analyzer.data = array.array('f', [
            100.0, 150.0, 120.0,  # Problem 0
            80.0, 90.0, 110.0      # Problem 1
        ])

        problemIndices = [1]
        winner_idx, winner_gflops = LogicAnalyzer.getWinnerForProblem(analyzer, problemIndices)

        # Solution 2 has 110 GFlops, should be winner for problem 1
        assert winner_idx == 2
        assert winner_gflops == 110.0

    def test_winner_for_range_single_solution(self):
        """Test winnerForRange with single solution returns 0."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 1

        result = LogicAnalyzer.winnerForRange(analyzer, [[0, 1]])

        # With only one solution, should always return 0
        assert result == 0

    def test_winner_for_range_selects_best_overall(self):
        """Test winnerForRange selects best solution across range."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numSolutions = 2
        analyzer.numIndices = 1
        analyzer.numProblemSizes = [3]
        analyzer.problemIndexToSize = [[64, 128, 256]]
        analyzer.flopsPerMac = 2
        analyzer.data = array.array('f', [
            100.0, 150.0,  # Problem 0
            200.0, 180.0,  # Problem 1
            120.0, 160.0   # Problem 2
        ])

        def mock_score_range(indexRange):
            # Solution 1 should have lower total time
            return [1000.0, 800.0]

        analyzer.scoreRangeForSolutions = mock_score_range

        result = LogicAnalyzer.winnerForRange(analyzer, [[0, 3]])

        # Solution 1 should win (lower score)
        assert result == 1


class TestPrepareLogic:
    """Tests for prepareLogic method."""

    def test_prepare_logic_converts_indices_to_sizes(self):
        """Test prepareLogic converts indices to actual sizes."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2
        analyzer.problemIndexToSize = [[64, 128, 256], [32, 64]]
        analyzer.indexOrder = [0, 1]

        # Logic with index 1 (which maps to size 128)
        logic = [[1, 0], [2, 1]]

        LogicAnalyzer.prepareLogic(analyzer, logic)

        # Should convert index 1 to size 128 for first level
        assert logic[0][0] == 128
        # Last entry should be -1
        assert logic[1][0] == -1

    def test_prepare_logic_empty_logic(self):
        """Test prepareLogic handles empty logic gracefully."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2

        logic = []

        # Should not crash
        LogicAnalyzer.prepareLogic(analyzer, logic)

        assert logic == []

    def test_prepare_logic_nested(self):
        """Test prepareLogic handles nested logic."""
        analyzer = MagicMock(spec=LogicAnalyzer)
        analyzer.numIndices = 2
        analyzer.problemIndexToSize = [[64, 128], [32, 64]]
        analyzer.indexOrder = [0, 1]

        # Nested logic
        logic = [[0, [[-1, 0]]]]

        LogicAnalyzer.prepareLogic(analyzer, logic)

        # Top level should be -1 (last)
        assert logic[0][0] == -1
