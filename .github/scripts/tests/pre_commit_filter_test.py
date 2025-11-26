from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pre_commit_filter


class TestPreCommitFilter(unittest.TestCase):
    def test_filter_files_opted_in_project(self):
        changed_files = ["projects/hipdnn/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(filtered, ["projects/hipdnn/some_file.cpp"])
        self.assertEqual(projects, {"hipdnn"})

    def test_filter_files_opted_in_shared_project(self):
        changed_files = ["shared/hipdnn/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(filtered, ["shared/hipdnn/some_file.cpp"])
        self.assertEqual(projects, {"hipdnn"})

    def test_filter_files_non_opted_in_project(self):
        changed_files = ["projects/otherproject/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(filtered, [])
        self.assertEqual(projects, set())

    def test_filter_files_non_opted_in_shared_project(self):
        changed_files = ["shared/otherproject/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(filtered, [])
        self.assertEqual(projects, set())

    def test_filter_files_outside_projects(self):
        changed_files = [".github/workflows/pre-commit.yml", "README.md"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(filtered, [".github/workflows/pre-commit.yml", "README.md"])
        self.assertEqual(projects, set())

    def test_filter_files_mixed(self):
        changed_files = [
            "projects/hipdnn/file1.cpp",
            "shared/hipdnn/file2.cpp",
            "projects/otherproject/file3.cpp",
            "shared/otherproject/file4.cpp",
            "README.md",
        ]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        expected_filtered = [
            "projects/hipdnn/file1.cpp",
            "shared/hipdnn/file2.cpp",
            "README.md",
        ]
        self.assertEqual(sorted(filtered), sorted(expected_filtered))
        self.assertEqual(projects, {"hipdnn"})

    def test_filter_files_with_spaces(self):
        changed_files = ["projects/hipdnn/file with spaces.cpp", "file with spaces.md"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, ["hipdnn"])
        self.assertEqual(
            filtered, ["projects/hipdnn/file with spaces.cpp", "file with spaces.md"]
        )
        self.assertEqual(projects, {"hipdnn"})

    @patch("pre_commit_filter.subprocess.run")
    def test_get_changed_files(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "file1.txt\nfile2.txt\n"
        mock_run.return_value = mock_result

        files = pre_commit_filter.get_changed_files("base", "head")
        self.assertEqual(files, ["file1.txt", "file2.txt"])
        mock_run.assert_called_with(
            ["git", "diff", "--name-only", "base...head"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("pre_commit_filter.os.environ", {})
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_github_output(self, mock_file):
        # Mock GITHUB_OUTPUT environment variable
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "output.txt"}):
            pre_commit_filter.write_github_output("key", "value")
            mock_file.assert_called_with("output.txt", "a")
            mock_file().write.assert_called_with("key=value\n")

    @patch("pre_commit_filter.os.environ", {})
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_github_output_multiline(self, mock_file):
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "output.txt"}):
            pre_commit_filter.write_github_output("key", "line1\nline2")
            mock_file.assert_called_with("output.txt", "a")
            mock_file().write.assert_called_with("key<<EOF\nline1\nline2\nEOF\n")


if __name__ == "__main__":
    unittest.main()
