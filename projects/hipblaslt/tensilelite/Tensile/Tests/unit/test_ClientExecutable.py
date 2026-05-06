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
from unittest.mock import Mock, patch, MagicMock, call
import os
from pathlib import Path

pytestmark = pytest.mark.unit

from Tensile import ClientExecutable


class TestCMakeEnvironment:
    """Tests for CMakeEnvironment class."""

    def test_initialization(self):
        """Test CMakeEnvironment initialization with options."""
        env = ClientExecutable.CMakeEnvironment(
            "/src", "/build",
            CMAKE_BUILD_TYPE="Release",
            TENSILE_USE_MSGPACK="ON"
        )

        assert env.sourceDir == "/src"
        assert env.buildDir == "/build"
        assert env.options["CMAKE_BUILD_TYPE"] == "Release"
        assert env.options["TENSILE_USE_MSGPACK"] == "ON"

    def test_initialization_no_options(self):
        """Test CMakeEnvironment initialization without options."""
        env = ClientExecutable.CMakeEnvironment("/src", "/build")

        assert env.sourceDir == "/src"
        assert env.buildDir == "/build"
        assert env.options == {}

    @patch('Tensile.ClientExecutable.subprocess.check_call')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.ClientExecutionLock')
    @patch('Tensile.ClientExecutable.print2')
    @patch('Tensile.ClientExecutable.globalParameters', {'ClientExecutionLockPath': '/tmp/lock'})
    def test_generate(self, mock_print2, mock_lock, mock_ensure, mock_check_call, tmp_path):
        """Test CMake generation command construction."""
        mock_ensure.return_value = str(tmp_path)
        mock_lock_instance = MagicMock()
        mock_lock.return_value.__enter__.return_value = mock_lock_instance

        env = ClientExecutable.CMakeEnvironment(
            "/src", str(tmp_path),
            CMAKE_BUILD_TYPE="Release",
            TENSILE_USE_MSGPACK="ON"
        )
        env.generate()

        # Verify subprocess called with cmake command
        assert mock_check_call.called
        args = mock_check_call.call_args[0][0]
        assert args[0] == 'cmake'
        assert '-D' in args
        assert 'CMAKE_BUILD_TYPE=Release' in args
        assert 'TENSILE_USE_MSGPACK=ON' in args
        assert '/src' in args

        # Verify cwd parameter
        assert mock_check_call.call_args[1]['cwd'] == str(tmp_path)

        # Verify lock was acquired
        mock_lock.assert_called_once_with('/tmp/lock')

    @patch('Tensile.ClientExecutable.subprocess.check_call')
    @patch('Tensile.ClientExecutable.ClientExecutionLock')
    @patch('Tensile.ClientExecutable.print2')
    @patch('Tensile.ClientExecutable.globalParameters', {'ClientExecutionLockPath': '/tmp/lock'})
    def test_build(self, mock_print2, mock_lock, mock_check_call):
        """Test make build command."""
        mock_lock_instance = MagicMock()
        mock_lock.return_value.__enter__.return_value = mock_lock_instance

        env = ClientExecutable.CMakeEnvironment("/src", "/build")
        env.build()

        # Verify make command
        assert mock_check_call.called
        args = mock_check_call.call_args[0][0]
        assert args == ['make', '-j']

        # Verify cwd parameter
        assert mock_check_call.call_args[1]['cwd'] == "/build"

        # Verify lock was acquired
        mock_lock.assert_called_once_with('/tmp/lock')

    def test_builtPath_single_path(self):
        """Test builtPath with single path component."""
        env = ClientExecutable.CMakeEnvironment("/src", "/build")
        result = env.builtPath("client")

        assert result == os.path.join("/build", "client")

    def test_builtPath_multiple_paths(self):
        """Test builtPath with multiple path components."""
        env = ClientExecutable.CMakeEnvironment("/src", "/build")
        result = env.builtPath("client", "tensile_client")

        assert result == os.path.join("/build", "client", "tensile_client")

    def test_builtPath_nested_paths(self):
        """Test builtPath with nested paths."""
        env = ClientExecutable.CMakeEnvironment("/src", "/build")
        result = env.builtPath("dir1", "dir2", "dir3", "file.exe")

        assert result == os.path.join("/build", "dir1", "dir2", "dir3", "file.exe")


class TestClientExecutableEnvironment:
    """Tests for clientExecutableEnvironment function."""

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Release',
        'LibraryFormat': 'msgpack',
        'EnableMarker': 'ON',
        'ROCmBinPath': '/opt/rocm/bin'
    })
    def test_basic_environment_setup(self, mock_ensure):
        """Test basic environment setup without ccache."""
        mock_ensure.return_value = "/build"

        env = ClientExecutable.clientExecutableEnvironment("/builddir", "amdclang++", "amdclang")

        assert env.sourceDir == '/tensile/source'
        assert env.buildDir == "/build"
        assert env.options['CMAKE_BUILD_TYPE'] == 'Release'
        assert env.options['TENSILE_USE_MSGPACK'] == 'ON'
        assert env.options['Tensile_LIBRARY_FORMAT'] == 'msgpack'
        assert env.options['Tensile_ENABLE_MARKER'] == 'ON'
        assert env.options['CMAKE_CXX_COMPILER'] == '/opt/rocm/bin/amdclang++'
        assert env.options['CMAKE_C_COMPILER'] == '/opt/rocm/bin/amdclang'

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Debug',
        'LibraryFormat': 'yaml',
        'EnableMarker': 'OFF',
        'ROCmBinPath': '/custom/rocm/bin'
    })
    def test_different_global_parameters(self, mock_ensure):
        """Test with different globalParameters values."""
        mock_ensure.return_value = "/debug_build"

        env = ClientExecutable.clientExecutableEnvironment("/debugdir", "g++", "gcc")

        assert env.options['CMAKE_BUILD_TYPE'] == 'Debug'
        assert env.options['Tensile_LIBRARY_FORMAT'] == 'yaml'
        assert env.options['Tensile_ENABLE_MARKER'] == 'OFF'
        assert env.options['CMAKE_CXX_COMPILER'] == '/custom/rocm/bin/g++'
        assert env.options['CMAKE_C_COMPILER'] == '/custom/rocm/bin/gcc'

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Release',
        'LibraryFormat': 'msgpack',
        'EnableMarker': 'ON',
        'ROCmBinPath': '/opt/rocm/bin'
    })
    @patch.dict(os.environ, {'CCACHE_BASEDIR': '/some/path'})
    @patch('builtins.print')
    def test_with_ccache(self, mock_print, mock_ensure):
        """Test ccache configuration when CCACHE_BASEDIR is set."""
        mock_ensure.return_value = "/build"

        env = ClientExecutable.clientExecutableEnvironment("/builddir", "amdclang++", "amdclang")

        assert env.options['CMAKE_C_COMPILER_LAUNCHER'] == 'ccache'
        assert env.options['CMAKE_CXX_COMPILER_LAUNCHER'] == 'ccache'
        mock_print.assert_called_once_with('Is Using CCACHE')

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Release',
        'LibraryFormat': 'msgpack',
        'EnableMarker': 'ON',
        'ROCmBinPath': '/opt/rocm/bin'
    })
    @patch.dict(os.environ, {}, clear=True)
    def test_without_ccache(self, mock_ensure):
        """Test configuration without ccache."""
        mock_ensure.return_value = "/build"

        env = ClientExecutable.clientExecutableEnvironment("/builddir", "amdclang++", "amdclang")

        assert 'CMAKE_C_COMPILER_LAUNCHER' not in env.options
        assert 'CMAKE_CXX_COMPILER_LAUNCHER' not in env.options

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Release',
        'LibraryFormat': 'msgpack',
        'EnableMarker': 'ON',
        'ROCmBinPath': '/opt/rocm/bin'
    })
    @patch('os.name', 'nt')
    def test_windows_llvm_disabled(self, mock_ensure):
        """Test TENSILE_USE_LLVM=OFF on Windows."""
        mock_ensure.return_value = "/build"

        env = ClientExecutable.clientExecutableEnvironment("/builddir", "cl.exe", "cl.exe")

        assert env.options['TENSILE_USE_LLVM'] == 'OFF'

    @patch('Tensile.ClientExecutable.SOURCE_PATH', '/tensile/source')
    @patch('Tensile.ClientExecutable.ensurePath')
    @patch('Tensile.ClientExecutable.globalParameters', {
        'CMakeBuildType': 'Release',
        'LibraryFormat': 'msgpack',
        'EnableMarker': 'ON',
        'ROCmBinPath': '/opt/rocm/bin'
    })
    @patch('os.name', 'posix')
    def test_linux_llvm_enabled(self, mock_ensure):
        """Test TENSILE_USE_LLVM=ON on Linux."""
        mock_ensure.return_value = "/build"

        env = ClientExecutable.clientExecutableEnvironment("/builddir", "amdclang++", "amdclang")

        assert env.options['TENSILE_USE_LLVM'] == 'ON'


class TestGetClientExecutable:
    """Tests for getClientExecutable function."""

    def setup_method(self):
        """Reset global buildEnv before each test."""
        ClientExecutable.buildEnv = None

    @patch('Tensile.ClientExecutable.globalParameters', {'PrebuiltClient': '/path/to/prebuilt/client'})
    def test_returns_prebuilt_client(self):
        """Test returns PrebuiltClient if configured."""
        result = ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/build"))

        assert result == '/path/to/prebuilt/client'

    @patch('Tensile.ClientExecutable.globalParameters', {})
    @patch('Tensile.ClientExecutable.clientExecutableEnvironment')
    def test_builds_client_first_time(self, mock_client_env):
        """Test builds client on first call when no prebuilt client."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_client_env.return_value = mock_env

        result = ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/builddir"))

        # Verify environment was created
        mock_client_env.assert_called_once()

        # Verify generate and build were called
        mock_env.generate.assert_called_once()
        mock_env.build.assert_called_once()

        # Verify builtPath was called for client executable
        mock_env.builtPath.assert_called_once_with("client/tensile_client")

        assert result == "/build/client/tensile_client"

    @patch('Tensile.ClientExecutable.globalParameters', {})
    @patch('Tensile.ClientExecutable.clientExecutableEnvironment')
    def test_caches_build_environment(self, mock_client_env):
        """Test buildEnv is cached after first build."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_client_env.return_value = mock_env

        # First call
        result1 = ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/builddir"))

        # Second call
        result2 = ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/builddir"))

        # clientExecutableEnvironment should only be called once
        assert mock_client_env.call_count == 1

        # generate and build should only be called once
        mock_env.generate.assert_called_once()
        mock_env.build.assert_called_once()

        # But builtPath called twice (once per call)
        assert mock_env.builtPath.call_count == 2

        assert result1 == result2

    @patch('Tensile.ClientExecutable.globalParameters', {})
    @patch('Tensile.ClientExecutable.clientExecutableEnvironment')
    @patch('Tensile.ClientExecutable.CLIENT_BUILD_DIR', 'client_build')
    def test_uses_client_build_dir(self, mock_client_env):
        """Test uses CLIENT_BUILD_DIR constant in path."""
        mock_env = Mock()
        mock_env.builtPath.return_value = "/build/client/tensile_client"
        mock_client_env.return_value = mock_env

        ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/builddir"))

        # Verify builddir argument includes CLIENT_BUILD_DIR
        call_args = mock_client_env.call_args[0]
        expected_builddir = Path("/builddir") / "client_build"
        assert call_args[0] == expected_builddir

    @patch('Tensile.ClientExecutable.globalParameters', {'PrebuiltClient': '/prebuilt'})
    def test_prebuilt_skips_build(self):
        """Test PrebuiltClient skips environment creation entirely."""
        # Should not attempt to build
        result = ClientExecutable.getClientExecutable("amdclang++", "amdclang", Path("/builddir"))

        # buildEnv should remain None
        assert ClientExecutable.buildEnv is None
        assert result == '/prebuilt'
