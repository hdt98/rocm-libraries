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
TensileLite Client Executor

Provides robust subprocess execution for tensilelite-client with proper error
handling and reporting, especially for use with pytest-xdist parallel testing.
"""

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Union
import logging

logger = logging.getLogger("Tensile.ClientExecutor")


@dataclass
class ClientError:
    """Structured error information from tensilelite-client"""

    error_type: str
    message: str
    location: str
    timestamp: str
    exit_code: int
    stdout: str
    stderr: str
    pid: Optional[int] = None

    def __str__(self):
        """Human-readable error format"""
        lines = [
            "=" * 78,
            "TENSILELITE CLIENT ERROR",
            "=" * 78,
            f"Type:     {self.error_type}",
            f"Message:  {self.message}",
            f"Location: {self.location}",
            f"Exit Code: {self.exit_code}",
        ]

        if self.timestamp:
            lines.append(f"Timestamp: {self.timestamp}")

        if self.pid:
            lines.append(f"PID: {self.pid}")

        lines.extend(
            [
                "=" * 78,
                "STDERR:",
                self.stderr if self.stderr else "(empty)",
                "=" * 78,
            ]
        )

        if self.stdout:
            lines.extend(
                [
                    "STDOUT (last 50 lines):",
                    "\n".join(self.stdout.splitlines()[-50:]),
                    "=" * 78,
                ]
            )

        return "\n".join(lines)

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ClientExecutionError(Exception):
    """
    Exception raised when tensilelite-client execution fails.

    This exception preserves all error context from the C++ client
    and provides clear error messages for debugging.
    """

    def __init__(self, error: ClientError):
        self.error = error
        super().__init__(str(error))


def run_tensilelite_client(
    client_path: Union[str, Path],
    args: List[str],
    timeout: Optional[int] = None,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """
    Run tensilelite-client with robust error handling.

    This function provides comprehensive error detection and reporting:
    1. Always checks for .tensilelite_error.json first (C++ errors)
    2. Captures stdout/stderr for debugging
    3. Provides clear error context
    4. Never masks the real error with FileNotFoundError

    Args:
        client_path: Path to tensilelite-client executable
        args: Command-line arguments for the client
        timeout: Timeout in seconds (None for no timeout)
        capture_output: Whether to capture stdout/stderr
        cwd: Working directory for execution
        env: Environment variables (None to inherit current environment)

    Returns:
        subprocess.CompletedProcess object on success

    Raises:
        ClientExecutionError: If client execution fails for any reason

    Example:
        >>> result = run_tensilelite_client(
        ...     "build/client/tensilelite-client",
        ...     ["--config-file=test.yaml"],
        ...     timeout=300
        ... )
    """
    client_path = Path(client_path)
    error_file = Path(".tensilelite_error.json")

    # Validate client exists before attempting to run
    if not client_path.exists():
        error = ClientError(
            error_type="FileNotFound",
            message=f"Client executable not found: {client_path}",
            location="run_tensilelite_client",
            timestamp="",
            exit_code=1,
            stdout="",
            stderr=f"Expected client at: {client_path.absolute()}",
        )
        raise ClientExecutionError(error)

    # Clean up any old error file from previous run
    if error_file.exists():
        try:
            error_file.unlink()
        except OSError as e:
            logger.warning(f"Could not remove old error file: {e}")

    # Prepare environment
    run_env = os.environ.copy() if env is None else env.copy()

    logger.debug(f"Executing: {client_path} {' '.join(args)}")
    logger.debug(f"Working directory: {cwd or os.getcwd()}")

    try:
        # Run the client
        result = subprocess.run(
            [str(client_path)] + args,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False,  # Don't raise CalledProcessError, we handle errors manually
            cwd=cwd,
            env=run_env,
        )

        # CRITICAL: Always check for error file FIRST, before checking expected output
        # This ensures C++ errors are never masked by Python FileNotFoundError
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_data = json.load(f)

                error = ClientError(
                    error_type=error_data.get("error_type", "Unknown"),
                    message=error_data.get("message", "No message provided"),
                    location=error_data.get("location", "Unknown"),
                    timestamp=error_data.get("timestamp", ""),
                    exit_code=error_data.get("exit_code", result.returncode),
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    pid=error_data.get("pid"),
                )

                logger.error(f"Client reported error: {error.error_type}")
                raise ClientExecutionError(error)

            except json.JSONDecodeError as e:
                # Error file exists but is corrupted
                error = ClientError(
                    error_type="ErrorFileCorrupted",
                    message=f"Could not parse error file: {e}",
                    location="run_tensilelite_client",
                    timestamp="",
                    exit_code=result.returncode,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                )
                raise ClientExecutionError(error)
            finally:
                # Clean up error file
                try:
                    error_file.unlink()
                except OSError:
                    pass

        # Check exit code if no error file was written
        if result.returncode != 0:
            # C++ failed but didn't write error file - construct error from stderr
            error = ClientError(
                error_type="UnhandledError",
                message=f"Client exited with code {result.returncode} but did not write error file",
                location="Unknown (no error file written)",
                timestamp="",
                exit_code=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
            logger.error(
                f"Client failed without error file. Exit code: {result.returncode}"
            )
            raise ClientExecutionError(error)

        # Success!
        logger.debug(f"Client execution successful (exit code 0)")
        return result

    except subprocess.TimeoutExpired as e:
        # Timeout occurred
        error = ClientError(
            error_type="TimeoutError",
            message=f"Client execution exceeded {timeout}s timeout",
            location="subprocess.run",
            timestamp="",
            exit_code=-1,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )
        logger.error(f"Client execution timed out after {timeout}s")
        raise ClientExecutionError(error) from e

    except OSError as e:
        # OS-level error (e.g., permission denied, broken pipe)
        error = ClientError(
            error_type="OSError",
            message=f"OS error during client execution: {e}",
            location="subprocess.run",
            timestamp="",
            exit_code=-1,
            stdout="",
            stderr=str(e),
        )
        logger.error(f"OS error executing client: {e}")
        raise ClientExecutionError(error) from e


def run_tensilelite_client_parallel(
    client_path: Union[str, Path],
    configs: List[dict],
    timeout: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[subprocess.CompletedProcess]:
    """
    Run multiple tensilelite-client instances in parallel.

    This is a convenience wrapper around run_tensilelite_client for
    parallel execution scenarios.

    Args:
        client_path: Path to tensilelite-client executable
        configs: List of config dicts, each with 'args', 'cwd', 'env' keys
        timeout: Timeout per client instance in seconds
        max_workers: Maximum number of parallel workers

    Returns:
        List of CompletedProcess objects, one per config

    Raises:
        ClientExecutionError: If any client execution fails
    """
    import concurrent.futures

    if max_workers is None:
        max_workers = min(len(configs), os.cpu_count() or 4)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config in configs:
            future = executor.submit(
                run_tensilelite_client,
                client_path,
                config.get("args", []),
                timeout=timeout,
                cwd=config.get("cwd"),
                env=config.get("env"),
            )
            futures.append(future)

        # Wait for all to complete and collect results
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())  # This will raise if any failed

    return results
