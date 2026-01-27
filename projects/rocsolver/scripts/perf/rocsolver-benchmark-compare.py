#!/usr/bin/env python3
# ##########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

"""
Automated benchmark comparison suite for rocSOLVER.

Compares two rocSOLVER versions by running the same benchmark suite on both
and generating speedup comparison graphs.
"""

import argparse
import os
import subprocess
import sys

from rocsolver_suites import SUITES


def validate_executable(exe_path, label):
    """
    Validate that executable exists and is executable.

    Args:
        exe_path: Path to the executable to validate
        label: Human-readable label for error messages (e.g., "Baseline", "Comparison")

    Exits:
        If the executable doesn't exist or is not executable
    """
    if not os.path.exists(exe_path):
        sys.exit(f"Error: {label} executable not found: {exe_path}")
    if not os.access(exe_path, os.X_OK):
        sys.exit(f"Error: {label} executable is not executable: {exe_path}")


def run_command(cmd, description, verbose=False):
    """
    Run a command and handle errors.

    Args:
        cmd: Command to run as a list of arguments
        description: Description of what the command does
        verbose: If True, print the command being run

    Exits:
        If the command returns a non-zero exit code
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    if verbose:
        print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Error: {description} failed with exit code {result.returncode}")


def run_benchmarks(exe_path, output_csv, suite, precision, case, verbose, script_dir):
    """
    Run benchmarks using rocsolver-perfoptim-suite.py.

    Args:
        exe_path: Path to rocsolver-bench executable
        output_csv: Output CSV file path
        suite: Benchmark suite name
        precision: Precision to use ('s', 'd', 'c', 'z')
        case: Size case to use ('small', 'medium', 'large')
        verbose: Whether to run in verbose mode
        script_dir: Directory containing the scripts

    Returns:
        Command list for running the benchmarks
    """
    perfoptim_script = os.path.join(script_dir, 'rocsolver-perfoptim-suite.py')

    cmd = [
        sys.executable,
        perfoptim_script,
        '-o', output_csv,
        '--exe', exe_path
    ]
    if verbose:
        cmd.append('-v')

    cmd.extend([suite, precision, case])

    return cmd


def generate_comparison_graphs(baseline_csv, comparison_csv, output_prefix,
                                separate_groups, script_dir, verbose):
    """
    Generate speedup comparison graphs using rocsolver_compare.py.

    Args:
        baseline_csv: Path to baseline CSV file
        comparison_csv: Path to comparison CSV file
        output_prefix: Prefix for output graph files
        separate_groups: Whether to generate separate graphs for each parameter group
        script_dir: Directory containing the scripts
        verbose: Whether to run in verbose mode

    Returns:
        Command list for generating comparison graphs
    """
    compare_script = os.path.join(script_dir, 'rocsolver_compare.py')

    cmd = [
        sys.executable,
        compare_script,
        baseline_csv,
        comparison_csv,
        '--output-prefix', f'{output_prefix}_speedup'
    ]

    if separate_groups:
        cmd.append('--separate-groups')

    return cmd


def generate_individual_graphs(csv_path, output_path, separate_groups, script_dir):
    """
    Generate individual performance graph using rocsolver_graph.py.

    Args:
        csv_path: Path to the CSV file to graph
        output_path: Output graph file path
        separate_groups: Whether to generate separate graphs for each parameter group
        script_dir: Directory containing the scripts

    Returns:
        Command list for generating individual graphs
    """
    graph_script = os.path.join(script_dir, 'rocsolver_graph.py')

    cmd = [
        sys.executable,
        graph_script,
        csv_path,
        '-o', output_path
    ]

    if separate_groups:
        cmd.append('--separate-groups')

    return cmd


def main():
    """Main entry point for the benchmark comparison script."""
    parser = argparse.ArgumentParser(
        prog='rocsolver-benchmark-compare',
        description='Compare two rocSOLVER versions by running benchmarks and generating speedup graphs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n'
               '  python rocsolver-benchmark-compare.py \\\n'
               '      --baseline-exe /path/to/old/rocsolver-bench \\\n'
               '      --comparison-exe /path/to/new/rocsolver-bench \\\n'
               '      geqrf s small\n')

    parser.add_argument('--baseline-exe',
                       required=True,
                       metavar='PATH',
                       help='Path to baseline/reference rocsolver-bench executable')
    parser.add_argument('--comparison-exe',
                       required=True,
                       metavar='PATH',
                       help='Path to comparison/new rocsolver-bench executable')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Display detailed information about operations')
    parser.add_argument('-o', '--output-base',
                       default=None,
                       help='Base name for output files (default: <suite>_<precision>_<case>_comparison)')
    parser.add_argument('--keep-csvs',
                       default=True,
                       action='store_true',
                       help='Keep intermediate CSV files (default: delete after comparison)')
    parser.add_argument('--no-individual-graphs',
                       action='store_true',
                       help='Skip generating individual performance graphs for each version')
    parser.add_argument('--separate-groups',
                       action='store_true',
                       help='Generate separate graph for each parameter group')
    parser.add_argument('suite',
                       choices=SUITES.keys(),
                       help='Benchmark suite to run')
    parser.add_argument('precision',
                       choices=['s', 'd', 'c', 'z'],
                       help='Precision to use for benchmarks')
    parser.add_argument('case',
                       choices=['small', 'medium', 'large'],
                       help='Size case to use for benchmarks')

    args = parser.parse_args()

    # Validate executables
    validate_executable(args.baseline_exe, "Baseline")
    validate_executable(args.comparison_exe, "Comparison")

    # Determine output base name
    if args.output_base:
        output_base = args.output_base
    else:
        output_base = f"{args.suite}_{args.precision}_{args.case}_comparison"

    baseline_csv = f"{output_base}_baseline.csv"
    comparison_csv = f"{output_base}_comparison.csv"

    # Find script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run benchmarks
    baseline_cmd = run_benchmarks(
        args.baseline_exe, baseline_csv, args.suite, args.precision,
        args.case, args.verbose, script_dir
    )
    run_command(baseline_cmd, "Running baseline benchmarks", args.verbose)

    comparison_cmd = run_benchmarks(
        args.comparison_exe, comparison_csv, args.suite, args.precision,
        args.case, args.verbose, script_dir
    )
    run_command(comparison_cmd, "Running comparison benchmarks", args.verbose)

    # Generate speedup comparison graphs
    if not os.path.exists(baseline_csv) or not os.path.exists(comparison_csv):
        sys.exit(f"Error: CSV files not found after benchmark runs")

    comparison_graph_cmd = generate_comparison_graphs(
        baseline_csv, comparison_csv, output_base,
        args.separate_groups, script_dir, args.verbose
    )
    run_command(comparison_graph_cmd, "Generating speedup comparison graphs", args.verbose)

    # Generate individual performance graphs (optional)
    if not args.no_individual_graphs:
        baseline_graph_path = f"{output_base}_baseline.png"
        baseline_graph_cmd = generate_individual_graphs(
            baseline_csv, baseline_graph_path, args.separate_groups, script_dir
        )
        run_command(baseline_graph_cmd, "Generating baseline performance graph", args.verbose)

        comparison_graph_path = f"{output_base}_comparison.png"
        comparison_graph_cmd = generate_individual_graphs(
            comparison_csv, comparison_graph_path, args.separate_groups, script_dir
        )
        run_command(comparison_graph_cmd, "Generating comparison performance graph", args.verbose)

    if not args.keep_csvs:
        os.remove(baseline_csv)
        os.remove(comparison_csv)
        if args.verbose:
            print(f"\nCleaned up intermediate CSV files")

    if args.keep_csvs:
        print(f"Baseline CSV:     {baseline_csv}")
        print(f"Comparison CSV:   {comparison_csv}")

    if args.separate_groups:
        print(f"Speedup graphs:   {output_base}_speedup_*.png")
    else:
        print(f"Speedup graph:    {output_base}_speedup.png")

    if not args.no_individual_graphs:
        if args.separate_groups:
            print(f"Baseline graphs:  {output_base}_baseline_*.png")
            print(f"Comparison graphs: {output_base}_comparison_*.png")
        else:
            print(f"Baseline graph:   {output_base}_baseline.png")
            print(f"Comparison graph: {output_base}_comparison.png")


if __name__ == '__main__':
    main()
