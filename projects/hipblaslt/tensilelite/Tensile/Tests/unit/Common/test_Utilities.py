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
from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path
from Tensile.Common.Utilities import (
    isRhel8,
    fastdeepcopy,
    setVerbosity,
    getVerbosity,
    hasParam,
    isExe,
    locateExe,
    ensurePath,
    roundUp,
    versionIsCompatible,
    ProgressBar,
    DataDirection,
    SpinnyThing
)

pytestmark = pytest.mark.unit

@pytest.fixture
def mock_openFile():
    with patch("builtins.open", mock_open()) as mock:
        yield mock

@pytest.fixture
def mock_exists():
    with patch.object(Path, "exists", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_notExists():
    with patch.object(Path, "exists", return_value=False) as mock:
        yield mock

# Test cases for isRhel8
def test_isRhel8_true(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.4"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_non_rhel(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Ubuntu" VERSION_ID="20.04"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_new_version(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="9.0"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_true_with_warning(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.5"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_file_not_found(mock_notExists, mock_openFile):
    assert isRhel8() is False
    mock_notExists.assert_called_once_with()
    mock_openFile.assert_not_called()  # No file open attempt if the file doesn't exist


class TestFastDeepCopy:
    """Tests for fastdeepcopy function."""

    def test_copy_simple_dict(self):
        """Test fastdeepcopy with simple dictionary."""
        original = {"a": 1, "b": 2}
        copied = fastdeepcopy(original)

        assert copied == original
        assert copied is not original

    def test_copy_nested_structure(self):
        """Test fastdeepcopy with nested structures."""
        original = {"outer": {"inner": [1, 2, 3]}}
        copied = fastdeepcopy(original)

        assert copied == original
        assert copied is not original
        assert copied["outer"] is not original["outer"]
        assert copied["outer"]["inner"] is not original["outer"]["inner"]

    def test_copy_list(self):
        """Test fastdeepcopy with list."""
        original = [1, 2, [3, 4]]
        copied = fastdeepcopy(original)

        assert copied == original
        assert copied is not original
        assert copied[2] is not original[2]


class TestVerbosity:
    """Tests for verbosity functions."""

    def test_default_verbosity(self):
        """Test default verbosity level."""
        # Default is 1
        verbosity = getVerbosity()
        assert verbosity >= 0

    def test_set_verbosity(self):
        """Test setting verbosity level."""
        original = getVerbosity()
        setVerbosity(2)
        assert getVerbosity() == 2
        # Restore original
        setVerbosity(original)

    def test_verbosity_zero(self):
        """Test setting verbosity to zero."""
        original = getVerbosity()
        setVerbosity(0)
        assert getVerbosity() == 0
        setVerbosity(original)


class TestHasParam:
    """Tests for hasParam function."""

    def test_param_in_dict(self):
        """Test hasParam finds parameter in dictionary."""
        structure = {"param1": "value1", "param2": "value2"}
        assert hasParam("param1", structure) is True
        assert hasParam("param3", structure) is False

    def test_param_in_list(self):
        """Test hasParam finds parameter in list."""
        structure = [{"param1": "value1"}, {"param2": "value2"}]
        assert hasParam("param1", structure) is True
        assert hasParam("param3", structure) is False

    def test_param_in_nested_list(self):
        """Test hasParam finds parameter in nested list."""
        structure = [[{"param1": "value1"}], [{"param2": "value2"}]]
        assert hasParam("param1", structure) is True
        assert hasParam("param2", structure) is True

    def test_param_equals_value(self):
        """Test hasParam with direct value comparison."""
        assert hasParam("value", "value") is True
        assert hasParam("value", "other") is False

    def test_param_in_empty_list(self):
        """Test hasParam with empty list."""
        assert hasParam("param", []) is False


class TestIsExe:
    """Tests for isExe function."""

    def test_executable_file_returns_true(self, tmp_path):
        """Test isExe returns True for executable file."""
        exe_file = tmp_path / "test_exe"
        exe_file.write_text("#!/bin/bash\necho test")
        exe_file.chmod(0o755)

        assert isExe(str(exe_file)) is True

    def test_non_executable_file_returns_false(self, tmp_path):
        """Test isExe returns False for non-executable file."""
        regular_file = tmp_path / "test_file.txt"
        regular_file.write_text("content")

        assert isExe(str(regular_file)) is False

    def test_non_existent_file_returns_false(self):
        """Test isExe returns False for non-existent file."""
        assert isExe("/nonexistent/path/file") is False


class TestLocateExe:
    """Tests for locateExe function."""

    def test_finds_exe_in_default_path(self, tmp_path):
        """Test locateExe finds executable in default path."""
        exe_dir = tmp_path / "bin"
        exe_dir.mkdir()
        exe_file = exe_dir / "my_exe"
        exe_file.write_text("#!/bin/bash")
        exe_file.chmod(0o755)

        result = locateExe(str(exe_dir), "my_exe")
        assert result == str(exe_file)

    def test_finds_exe_in_path(self, tmp_path, monkeypatch):
        """Test locateExe finds executable in PATH."""
        exe_dir = tmp_path / "path_bin"
        exe_dir.mkdir()
        exe_file = exe_dir / "path_exe"
        exe_file.write_text("#!/bin/bash")
        exe_file.chmod(0o755)

        monkeypatch.setenv("PATH", str(exe_dir))

        result = locateExe(None, "path_exe")
        assert result == str(exe_file)

    def test_raises_when_not_found(self, monkeypatch):
        """Test locateExe raises when executable not found."""
        monkeypatch.setenv("PATH", "/nonexistent/path")

        with pytest.raises(OSError, match="Failed to locate"):
            locateExe("/nonexistent", "missing_exe")


class TestEnsurePath:
    """Tests for ensurePath function."""

    def test_creates_new_directory(self, tmp_path):
        """Test ensurePath creates new directory."""
        new_dir = tmp_path / "new_directory"
        result = ensurePath(str(new_dir))

        assert new_dir.exists()
        assert result == str(new_dir)

    def test_handles_existing_directory(self, tmp_path):
        """Test ensurePath handles existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensurePath(str(existing_dir))
        assert result == str(existing_dir)

    def test_creates_nested_directories(self, tmp_path):
        """Test ensurePath creates nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = ensurePath(str(nested_dir))

        assert nested_dir.exists()
        assert result == str(nested_dir)


class TestRoundUp:
    """Tests for roundUp function."""

    def test_rounds_up_decimal(self):
        """Test roundUp rounds up decimal values."""
        assert roundUp(1.1) == 2
        assert roundUp(3.9) == 4

    def test_integer_unchanged(self):
        """Test roundUp keeps integers unchanged."""
        assert roundUp(5.0) == 5
        assert roundUp(10.0) == 10

    def test_negative_values(self):
        """Test roundUp with negative values."""
        assert roundUp(-1.5) == -1
        assert roundUp(-2.1) == -2


class TestVersionIsCompatible:
    """Tests for versionIsCompatible function."""

    @patch('Tensile.Common.Utilities.__version__', '4.20.0')
    def test_same_version_compatible(self):
        """Test same version is compatible."""
        assert versionIsCompatible("4.20.0") is True

    @patch('Tensile.Common.Utilities.__version__', '4.20.0')
    def test_older_minor_compatible(self):
        """Test older minor version is compatible."""
        assert versionIsCompatible("4.19.0") is True

    @patch('Tensile.Common.Utilities.__version__', '4.20.0')
    def test_older_patch_compatible(self):
        """Test older patch version is compatible."""
        assert versionIsCompatible("4.20.0") is True

    @patch('Tensile.Common.Utilities.__version__', '4.20.0')
    def test_different_major_incompatible(self):
        """Test different major version is incompatible."""
        assert versionIsCompatible("5.0.0") is False
        assert versionIsCompatible("3.20.0") is False

    @patch('Tensile.Common.Utilities.__version__', '4.20.0')
    def test_newer_minor_incompatible(self):
        """Test newer minor version is incompatible."""
        assert versionIsCompatible("4.21.0") is False

    @patch('Tensile.Common.Utilities.__version__', '4.20.5')
    def test_newer_patch_incompatible(self):
        """Test newer patch version is incompatible."""
        assert versionIsCompatible("4.20.6") is False


class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_initialization(self):
        """Test ProgressBar initialization."""
        pb = ProgressBar(100)

        assert pb.maxValue == 100
        assert pb.width == 80
        assert pb.numTicks == 0

    def test_increment(self):
        """Test ProgressBar increment method."""
        pb = ProgressBar(100)
        pb.increment(10)

        assert pb.priorValue == 10

    def test_update(self):
        """Test ProgressBar update method."""
        pb = ProgressBar(100)
        pb.update(50)

        assert pb.priorValue == 50
        assert pb.fraction == 0.5

    def test_custom_width(self):
        """Test ProgressBar with custom width."""
        pb = ProgressBar(100, width=50)

        assert pb.width == 50
        assert pb.maxTicks == 43  # width - 7


class TestDataDirection:
    """Tests for DataDirection enum."""

    def test_enum_values(self):
        """Test DataDirection enum values."""
        assert DataDirection.NONE.value == (0,)
        assert DataDirection.READ.value == (1,)
        assert DataDirection.WRITE.value == 2

    def test_enum_members(self):
        """Test DataDirection enum members."""
        assert DataDirection.NONE.name == "NONE"
        assert DataDirection.READ.name == "READ"
        assert DataDirection.WRITE.name == "WRITE"


class TestSpinnyThing:
    """Tests for SpinnyThing class."""

    def test_initialization(self):
        """Test SpinnyThing initialization."""
        spinner = SpinnyThing()

        assert spinner.index == 0
        assert len(spinner.chars) == 4

    def test_increment(self):
        """Test SpinnyThing increment cycles through chars."""
        spinner = SpinnyThing()

        initial_index = spinner.index
        spinner.increment()
        assert spinner.index == (initial_index + 1) % 4
