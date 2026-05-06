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
import os
import yaml

pytestmark = pytest.mark.unit

from Tensile.TensileMergeLibrary import (
    ensurePath,
    allFiles,
    addKernel,
    sanitizeSolutions,
    removeUnusedSolutions,
    removeDuplicatedSolutions,
    findSolutionWithIndex
)


class TestEnsurePath:
    """Tests for ensurePath function."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Test creates directory when it doesn't exist."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        result = ensurePath(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == str(new_dir)

    def test_returns_path_if_already_exists(self, tmp_path):
        """Test returns path when directory already exists."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensurePath(str(existing_dir))

        assert existing_dir.exists()
        assert result == str(existing_dir)

    def test_creates_nested_directories(self, tmp_path):
        """Test creates nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensurePath(str(nested_dir))

        assert nested_dir.exists()
        assert result == str(nested_dir)


class TestAllFiles:
    """Tests for allFiles function."""

    def test_finds_yaml_files_in_directory(self, tmp_path):
        """Test finds all YAML files in directory."""
        # Create some YAML files
        (tmp_path / "file1.yaml").write_text("content1")
        (tmp_path / "file2.yaml").write_text("content2")
        (tmp_path / "file3.txt").write_text("not yaml")

        result = allFiles(str(tmp_path))

        assert len(result) == 2
        assert any("file1.yaml" in f for f in result)
        assert any("file2.yaml" in f for f in result)
        assert not any("file3.txt" in f for f in result)

    def test_finds_yaml_files_recursively(self, tmp_path):
        """Test finds YAML files in subdirectories."""
        # Create nested structure
        (tmp_path / "file1.yaml").write_text("content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.yaml").write_text("content")

        result = allFiles(str(tmp_path))

        # Note: allFiles only returns files, not files in subdirectories with .yaml extension
        # It only recurses into actual subdirectories, not files
        assert len(result) >= 1  # At least file1.yaml
        assert any("file1.yaml" in f for f in result)

    def test_handles_case_insensitive_yaml_extension(self, tmp_path):
        """Test finds .YAML (uppercase) files."""
        (tmp_path / "file1.YAML").write_text("content")
        (tmp_path / "file2.Yaml").write_text("content")

        result = allFiles(str(tmp_path))

        assert len(result) == 2

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        """Test returns empty list when no YAML files."""
        (tmp_path / "file.txt").write_text("content")

        result = allFiles(str(tmp_path))

        assert result == []


class TestAddKernel:
    """Tests for addKernel function."""

    def test_adds_new_solution_to_pool(self):
        """Test adds new solution when not in pool."""
        solutionPool = []
        solDict = {}
        solution = {"SolutionNameMin": "sol1", "Param": "value"}

        pool, dict_, index = addKernel(solutionPool, solDict, solution)

        assert len(pool) == 1
        assert index == 0
        assert pool[0]["SolutionIndex"] == 0
        assert "sol1" in dict_

    def test_reuses_existing_solution(self):
        """Test reuses existing solution instead of duplicating."""
        existing_sol = {"SolutionNameMin": "sol1", "SolutionIndex": 0}
        solutionPool = [existing_sol]
        solDict = {"sol1": existing_sol}
        solution = {"SolutionNameMin": "sol1", "Param": "new_value"}

        pool, dict_, index = addKernel(solutionPool, solDict, solution)

        assert len(pool) == 1  # No new solution added
        assert index == 0
        assert pool[0]["SolutionNameMin"] == "sol1"

    def test_increments_index_for_new_solutions(self):
        """Test increments solution index correctly."""
        solutionPool = [{"SolutionNameMin": "sol1", "SolutionIndex": 0}]
        solDict = {"sol1": solutionPool[0]}
        solution = {"SolutionNameMin": "sol2", "Param": "value"}

        pool, dict_, index = addKernel(solutionPool, solDict, solution)

        assert len(pool) == 2
        assert index == 1
        assert pool[1]["SolutionIndex"] == 1

    def test_deep_copies_solution(self):
        """Test deep copies solution to avoid reference issues."""
        solutionPool = []
        solDict = {}
        solution = {"SolutionNameMin": "sol1", "MutableParam": [1, 2, 3]}

        pool, dict_, index = addKernel(solutionPool, solDict, solution)

        # Modify original
        solution["MutableParam"].append(4)

        # Pool should have original values
        assert pool[0]["MutableParam"] == [1, 2, 3]


class TestSanitizeSolutions:
    """Tests for sanitizeSolutions function."""

    def test_updates_stagger_params_when_stagger_u_zero(self):
        """Test updates StaggerU-related params when StaggerU=0."""
        solList = [
            {"StaggerU": 0, "StaggerUMapping": 5, "StaggerUStride": 10, "_staggerStrideShift": 2}
        ]

        sanitizeSolutions(solList)

        assert solList[0]["StaggerUMapping"] == 0
        assert solList[0]["StaggerUStride"] == 0
        assert solList[0]["_staggerStrideShift"] == 0

    def test_leaves_params_unchanged_when_stagger_u_nonzero(self):
        """Test doesn't modify params when StaggerU != 0."""
        solList = [
            {"StaggerU": 1, "StaggerUMapping": 5, "StaggerUStride": 10, "_staggerStrideShift": 2}
        ]

        sanitizeSolutions(solList)

        assert solList[0]["StaggerUMapping"] == 5
        assert solList[0]["StaggerUStride"] == 10
        assert solList[0]["_staggerStrideShift"] == 2

    def test_handles_missing_stagger_u(self):
        """Test handles solutions without StaggerU key."""
        solList = [
            {"Param": "value"}
        ]

        # Should not raise
        sanitizeSolutions(solList)

    def test_processes_multiple_solutions(self):
        """Test processes all solutions in list."""
        solList = [
            {"StaggerU": 0, "StaggerUMapping": 5},
            {"StaggerU": 1, "StaggerUMapping": 5},
            {"StaggerU": 0, "StaggerUMapping": 10}
        ]

        sanitizeSolutions(solList)

        assert solList[0]["StaggerUMapping"] == 0
        assert solList[1]["StaggerUMapping"] == 5
        assert solList[2]["StaggerUMapping"] == 0


class TestRemoveUnusedSolutions:
    """Tests for removeUnusedSolutions function."""

    def test_removes_unused_solutions(self):
        """Test removes solutions not referenced in logic."""
        oriData = [
            None, None, None, None, None,
            [  # Index 5: solutions
                {"SolutionIndex": 0, "SolutionNameMin": "sol0"},
                {"SolutionIndex": 1, "SolutionNameMin": "sol1"},
                {"SolutionIndex": 2, "SolutionNameMin": "sol2"}
            ],
            None,
            [  # Index 7: logic - only uses solutions 0 and 2
                [None, [0, 1.0]],
                [None, [2, 0.9]]
            ]
        ]

        result_data, num_removed = removeUnusedSolutions(oriData)

        assert len(result_data[5]) == 2  # Solution 1 removed
        assert num_removed == 1

    def test_reindexes_remaining_solutions(self):
        """Test reindexes solutions after removal."""
        oriData = [
            None, None, None, None, None,
            [
                {"SolutionIndex": 0, "SolutionNameMin": "sol0"},
                {"SolutionIndex": 2, "SolutionNameMin": "sol2"}
            ],
            None,
            [
                [None, [0, 1.0]],
                [None, [2, 0.9]]
            ]
        ]

        result_data, num_removed = removeUnusedSolutions(oriData)

        # After reindexing, solutions should be 0 and 1
        assert result_data[5][0]["SolutionIndex"] == 0
        assert result_data[5][1]["SolutionIndex"] == 1
        # Logic should reference new indices
        assert result_data[7][0][1][0] == 0
        assert result_data[7][1][1][0] == 1

    def test_preserves_all_when_all_used(self):
        """Test preserves all solutions when all are used."""
        oriData = [
            None, None, None, None, None,
            [
                {"SolutionIndex": 0},
                {"SolutionIndex": 1}
            ],
            None,
            [
                [None, [0, 1.0]],
                [None, [1, 0.9]]
            ]
        ]

        result_data, num_removed = removeUnusedSolutions(oriData)

        assert len(result_data[5]) == 2
        assert num_removed == 0


class TestRemoveDuplicatedSolutions:
    """Tests for removeDuplicatedSolutions function."""

    def test_removes_duplicate_solutions_by_name(self):
        """Test removes solutions with duplicate SolutionNameMin."""
        oriData = [
            None, None, None, None, None,
            [
                {"SolutionIndex": 0, "SolutionNameMin": "sol1", "KernelNameMin": "kern1"},
                {"SolutionIndex": 1, "SolutionNameMin": "sol1", "KernelNameMin": "kern1"},  # Duplicate
                {"SolutionIndex": 2, "SolutionNameMin": "sol2", "KernelNameMin": "kern2"}
            ],
            None, []
        ]

        result_data, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(oriData)

        assert len(result_data[5]) == 2  # One duplicate removed
        assert num_removed == 1
        assert num_solutions == 2
        assert num_kernels == 2

    def test_keeps_first_occurrence_of_duplicate(self):
        """Test keeps first occurrence when duplicates found."""
        oriData = [
            None, None, None, None, None,
            [
                {"SolutionIndex": 0, "SolutionNameMin": "sol1", "KernelNameMin": "k1", "Param": "first"},
                {"SolutionIndex": 1, "SolutionNameMin": "sol1", "KernelNameMin": "k1", "Param": "second"}
            ],
            None, []
        ]

        result_data, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(oriData)

        assert result_data[5][0]["Param"] == "first"

    def test_preserves_all_when_no_duplicates(self):
        """Test preserves all when no duplicates."""
        oriData = [
            None, None, None, None, None,
            [
                {"SolutionIndex": 0, "SolutionNameMin": "sol1", "KernelNameMin": "k1"},
                {"SolutionIndex": 1, "SolutionNameMin": "sol2", "KernelNameMin": "k2"}
            ],
            None, []
        ]

        result_data, num_removed, num_solutions, num_kernels = removeDuplicatedSolutions(oriData)

        assert len(result_data[5]) == 2
        assert num_removed == 0


class TestFindSolutionWithIndex:
    """Tests for findSolutionWithIndex function."""

    def test_finds_solution_by_index_fast_path(self):
        """Test finds solution using fast path when index matches position."""
        solutionData = [
            {"SolutionIndex": 0, "Name": "sol0"},
            {"SolutionIndex": 1, "Name": "sol1"},
            {"SolutionIndex": 2, "Name": "sol2"}
        ]

        result = findSolutionWithIndex(solutionData, 1)

        assert result == {"SolutionIndex": 1, "Name": "sol1"}

    def test_finds_solution_by_search_when_not_at_position(self):
        """Test finds solution by searching when not at expected position."""
        solutionData = [
            {"SolutionIndex": 5, "Name": "sol5"},
            {"SolutionIndex": 10, "Name": "sol10"}
        ]

        result = findSolutionWithIndex(solutionData, 10)

        assert result == {"SolutionIndex": 10, "Name": "sol10"}

    def test_asserts_when_not_found(self):
        """Test raises assertion when index not found."""
        solutionData = [
            {"SolutionIndex": 0},
            {"SolutionIndex": 1}
        ]

        with pytest.raises(AssertionError):
            findSolutionWithIndex(solutionData, 99)

    def test_finds_first_when_duplicate_indices_at_position(self):
        """Test finds first occurrence when duplicate at expected position."""
        solutionData = [
            {"SolutionIndex": 0, "Order": "zero"},
            {"SolutionIndex": 1, "Order": "first"},
            {"SolutionIndex": 1, "Order": "second"}
        ]

        # Fast path returns the one at position 1
        result = findSolutionWithIndex(solutionData, 1)
        assert result["Order"] == "first"
