from copy import deepcopy
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_matrix

# Store original project_map to restore between tests
ORIGINAL_PROJECT_MAP = deepcopy(therock_matrix.project_map)


class TheRockMatrixTest(unittest.TestCase):
    def setUp(self):
        therock_matrix.project_map = deepcopy(ORIGINAL_PROJECT_MAP)

    def test_collect_projects_to_run_without_additional_option(self):
        subtrees = ["projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run(self):
        subtrees = ["projects/rocsparse", "projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_additional_option(self):
        subtrees = ["projects/rocsparse"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_dependency_graph(self):
        subtrees = ["projects/miopen", "projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_dependency_graph_diff_projects(self):
        # miopen and rocwmma: rocwmma adds to blas, miopen combines with blas
        # via dependency_graph, so we end up with 1 combined project
        subtrees = ["projects/miopen", "projects/rocwmma"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)
        # Verify rocwmma tests are included in the combined project
        projects_to_test = project_to_run[0]["projects_to_test"].split(",")
        self.assertIn("rocwmma", projects_to_test)
        self.assertIn("miopen", projects_to_test)

    def test_collect_projects_to_run_truly_separate_projects(self):
        # prim and fft are truly separate projects with no dependency overlap
        subtrees = ["projects/rocprim", "projects/hipfft"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 2)


class TheRockDynamicDepsTest(unittest.TestCase):
    """Tests for dynamic test dependency resolution from TheRock."""

    def setUp(self):
        therock_matrix.project_map = deepcopy(ORIGINAL_PROJECT_MAP)

    @patch("therock_matrix.get_test_dependencies_from_therock")
    def test_rocblas_uses_therock_deps(self, mock_get_deps):
        """When TheRock returns deps for rocblas, use those deps."""
        mock_get_deps.return_value = ["rocblas", "hipblas", "rocsolver"]
        subtrees = ["projects/rocblas"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        projects_to_test = project_to_run[0]["projects_to_test"].split(",")
        self.assertIn("rocblas", projects_to_test)
        self.assertIn("hipblas", projects_to_test)
        self.assertIn("rocsolver", projects_to_test)
        # Should NOT include hipblaslt or rocroller (from fallback)
        self.assertNotIn("hipblaslt", projects_to_test)
        self.assertNotIn("rocroller", projects_to_test)

    @patch("therock_matrix.get_test_dependencies_from_therock")
    def test_hipblaslt_falls_back_when_therock_returns_empty(self, mock_get_deps):
        """When TheRock returns empty, fall back to project_map."""
        mock_get_deps.return_value = None
        subtrees = ["projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        projects_to_test = project_to_run[0]["projects_to_test"].split(",")
        # Should include all from project_map["blas"]["projects_to_test"]
        self.assertIn("hipblaslt", projects_to_test)
        self.assertIn("rocblas", projects_to_test)
        self.assertIn("hipblas", projects_to_test)
        self.assertIn("rocroller", projects_to_test)

    @patch("therock_matrix.get_test_dependencies_from_therock")
    def test_hipblaslt_falls_back_when_therock_returns_only_self(self, mock_get_deps):
        """When TheRock returns only the component itself, fall back to project_map."""
        mock_get_deps.return_value = ["hipblaslt"]
        subtrees = ["projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        projects_to_test = project_to_run[0]["projects_to_test"].split(",")
        # Should include all from project_map["blas"]["projects_to_test"]
        self.assertIn("hipblaslt", projects_to_test)
        self.assertIn("rocblas", projects_to_test)
        self.assertIn("hipblas", projects_to_test)
        self.assertIn("rocroller", projects_to_test)

    @patch("therock_matrix.get_test_dependencies_from_therock")
    def test_mixed_components_combines_deps(self, mock_get_deps):
        """When both rocblas and hipblaslt change, combine TheRock deps with fallback."""

        def mock_deps(component_names):
            component = component_names[0]
            if component == "rocblas":
                return ["rocblas", "hipblas", "rocsolver"]
            else:
                return None  # hipblaslt has no TheRock deps

        mock_get_deps.side_effect = mock_deps
        subtrees = ["projects/rocblas", "projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        projects_to_test = project_to_run[0]["projects_to_test"].split(",")
        # Should include rocblas deps from TheRock
        self.assertIn("rocblas", projects_to_test)
        self.assertIn("hipblas", projects_to_test)
        self.assertIn("rocsolver", projects_to_test)
        # Should also include fallback deps for hipblaslt
        self.assertIn("hipblaslt", projects_to_test)
        self.assertIn("rocroller", projects_to_test)


if __name__ == "__main__":
    unittest.main()
