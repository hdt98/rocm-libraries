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
Enhanced ClientWriter with robust error handling.

This module wraps the existing ClientWriter.runClient() function with the new
ClientExecutor error handling framework. It can be used as a drop-in replacement.
"""

from Tensile.ClientWriter import getClientExecutablePath
from Tensile.ClientExecutor import run_tensilelite_client, ClientExecutionError
from Tensile.Common.ErrorReporting import get_logger
from pathlib import Path

logger = get_logger("ClientWriterWithErrorHandling")


def runClientWithErrorHandling(
    clientParametersFile,
    forBenchmark=False,
    timeout=None,
    cwd=None
):
    """
    Run tensilelite-client with enhanced error handling.

    This is a drop-in replacement for ClientWriter.runClient() that uses
    the new ClientExecutor framework for better error reporting.

    Args:
        clientParametersFile: Path to client parameters YAML file
        forBenchmark: Whether this is a benchmark run (affects timeout)
        timeout: Timeout in seconds (None for no timeout)
        cwd: Working directory (None to use current directory)

    Returns:
        subprocess.CompletedProcess on success

    Raises:
        ClientExecutionError: If client execution fails
    """
    # Get client executable path (raises FileNotFoundError if not found)
    clientExe = getClientExecutablePath()

    # Build arguments
    args = [f"--config-file={clientParametersFile}"]

    # Set reasonable default timeout if not specified
    if timeout is None:
        timeout = 3600 if forBenchmark else 600  # 1 hour for benchmark, 10 min for validation

    logger.info(f"Running client: {clientExe}")
    logger.debug(f"Config file: {clientParametersFile}")
    logger.debug(f"Timeout: {timeout}s, Benchmark: {forBenchmark}")

    try:
        result = run_tensilelite_client(
            client_path=clientExe,
            args=args,
            timeout=timeout,
            cwd=cwd
        )

        logger.info(f"Client execution completed successfully")
        return result

    except ClientExecutionError as e:
        logger.error(f"Client execution failed: {e.error.error_type}")
        logger.debug(f"Error details: {e.error.message}")
        # Re-raise to let pytest handle it
        raise


def validateClientExecution(result, expectedOutputFile):
    """
    Validate that client execution produced expected output.

    This should be called AFTER runClientWithErrorHandling() to verify
    that expected output files exist.

    Args:
        result: CompletedProcess from runClientWithErrorHandling()
        expectedOutputFile: Path to expected output file

    Raises:
        FileNotFoundError: If expected output doesn't exist (with helpful message)
    """
    outputPath = Path(expectedOutputFile)

    if not outputPath.exists():
        # Client ran successfully but didn't create expected output
        # This is different from a client error - it's a logic error
        logger.error(f"Client completed but output file not found: {outputPath}")
        logger.debug(f"Client stdout: {result.stdout[:500]}")  # First 500 chars

        raise FileNotFoundError(
            f"Expected output file not created: {outputPath}\n"
            f"Client execution completed with exit code 0, but the expected "
            f"output file was not created. This may indicate:\n"
            f"  - Incorrect output file path\n"
            f"  - Client logic error (ran but didn't produce output)\n"
            f"  - Test configuration issue\n"
            f"\nCheck client stdout for more details."
        )

    logger.debug(f"Validated output file exists: {outputPath}")
    return True
