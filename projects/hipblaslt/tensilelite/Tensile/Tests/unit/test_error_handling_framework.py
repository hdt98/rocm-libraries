################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice and this permission
# notice shall be included in all copies or substantial portions of the
# Software.
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

"""
Unit tests for the error handling framework.

These tests verify that the ClientExecutor and error reporting components
work correctly.
"""

import pytest
import json
import subprocess
from pathlib import Path
from Tensile.ClientExecutor import (
    run_tensilelite_client,
    ClientExecutionError,
    ClientError,
)
from Tensile.Common.ErrorReporting import setup_logging, get_logger


class TestClientError:
    """Tests for ClientError dataclass"""

    def test_client_error_creation(self):
        """Verify ClientError can be created with all fields"""
        error = ClientError(
            error_type="TestError",
            message="Test message",
            location="test.py:10",
            timestamp="2026-05-02T10:00:00Z",
            exit_code=42,
            stdout="test output",
            stderr="test error",
            pid=12345,
        )

        assert error.error_type == "TestError"
        assert error.exit_code == 42
        assert "TENSILELITE CLIENT ERROR" in str(error)

    def test_client_error_to_dict(self):
        """Verify ClientError converts to dict correctly"""
        error = ClientError(
            error_type="TestError",
            message="Test",
            location="test.py",
            timestamp="",
            exit_code=1,
            stdout="",
            stderr="",
        )

        d = error.to_dict()
        assert isinstance(d, dict)
        assert d["error_type"] == "TestError"
        assert d["exit_code"] == 1


class TestClientExecutionError:
    """Tests for ClientExecutionError exception"""

    def test_exception_creation(self):
        """Verify exception wraps ClientError correctly"""
        error = ClientError(
            error_type="TestError",
            message="Test message",
            location="test.py",
            timestamp="",
            exit_code=1,
            stdout="",
            stderr="",
        )

        exc = ClientExecutionError(error)
        assert exc.error == error
        assert "TestError" in str(exc)


class TestClientExecutor:
    """Tests for run_tensilelite_client function"""

    def test_nonexistent_client(self):
        """Verify FileNotFound error when client doesn't exist"""
        with pytest.raises(ClientExecutionError) as exc_info:
            run_tensilelite_client(
                client_path="/nonexistent/client", args=[], timeout=1
            )

        error = exc_info.value.error
        assert error.error_type == "FileNotFound"
        assert "not found" in error.message.lower()

    def test_error_file_detection(self, tmp_path):
        """Verify error file is detected and parsed"""
        # Create a mock script that writes an error file
        mock_client = tmp_path / "mock_client.sh"
        error_file = Path(".tensilelite_error.json")

        # Script writes error file and exits with code 11
        mock_client.write_text(
            """#!/bin/bash
cat > .tensilelite_error.json << 'EOF'
{
  "error_type": "LibraryLoadError",
  "message": "Mock library load failure",
  "location": "mock_client.sh:1",
  "timestamp": "2026-05-02T10:00:00Z",
  "pid": 12345,
  "exit_code": 11
}
EOF
exit 11
"""
        )
        mock_client.chmod(0o755)

        try:
            with pytest.raises(ClientExecutionError) as exc_info:
                run_tensilelite_client(
                    client_path=mock_client, args=[], timeout=5
                )

            error = exc_info.value.error
            assert error.error_type == "LibraryLoadError"
            assert error.message == "Mock library load failure"
            assert error.exit_code == 11

            # Verify error file was cleaned up
            assert not error_file.exists()

        finally:
            # Cleanup
            if error_file.exists():
                error_file.unlink()

    def test_timeout_handling(self, tmp_path):
        """Verify timeout errors are handled correctly"""
        # Create a script that sleeps forever
        mock_client = tmp_path / "sleep_client.sh"
        mock_client.write_text(
            """#!/bin/bash
sleep 3600
"""
        )
        mock_client.chmod(0o755)

        with pytest.raises(ClientExecutionError) as exc_info:
            run_tensilelite_client(
                client_path=mock_client, args=[], timeout=1
            )

        error = exc_info.value.error
        assert error.error_type == "TimeoutError"
        assert "timeout" in error.message.lower()
        assert error.exit_code == -1

    def test_successful_execution(self, tmp_path):
        """Verify successful client execution returns result"""
        # Create a script that succeeds
        mock_client = tmp_path / "success_client.sh"
        mock_client.write_text(
            """#!/bin/bash
echo "Success output"
exit 0
"""
        )
        mock_client.chmod(0o755)

        result = run_tensilelite_client(
            client_path=mock_client, args=[], timeout=5
        )

        assert result.returncode == 0
        assert "Success output" in result.stdout

    def test_nonzero_exit_without_error_file(self, tmp_path):
        """Verify non-zero exit code without error file is detected"""
        mock_client = tmp_path / "fail_client.sh"
        mock_client.write_text(
            """#!/bin/bash
echo "Error message" >&2
exit 42
"""
        )
        mock_client.chmod(0o755)

        with pytest.raises(ClientExecutionError) as exc_info:
            run_tensilelite_client(
                client_path=mock_client, args=[], timeout=5
            )

        error = exc_info.value.error
        assert error.error_type == "UnhandledError"
        assert error.exit_code == 42
        assert "Error message" in error.stderr


class TestLogging:
    """Tests for ErrorReporting logging module"""

    def test_logger_creation(self):
        """Verify logger can be created"""
        logger = get_logger("Test")
        assert logger.name == "Tensile.Test"

    def test_logger_hierarchy(self):
        """Verify logger follows hierarchy"""
        logger1 = get_logger("Tensile.Module1")
        logger2 = get_logger("Module2")

        assert logger1.name == "Tensile.Module1"
        assert logger2.name == "Tensile.Module2"

    def test_setup_logging(self, tmp_path):
        """Verify logging setup creates log file"""
        log_file = tmp_path / "test.log"
        logger = setup_logging("TestLogger", log_file=log_file)

        logger.info("Test message")

        # Log file should exist
        assert log_file.exists()

        # Should contain the message
        content = log_file.read_text()
        assert "Test message" in content
