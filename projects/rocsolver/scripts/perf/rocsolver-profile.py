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
Profiling script for rocSOLVER benchmarks using rocprofv3.

This script executes selected benchmark suites with rocprofv3 to collect
detailed profiling information including kernel traces and statistics.
"""

import argparse
import collections
import os
import shlex
import sys
from subprocess import Popen, PIPE

from rocsolver_suites import SUITES, get_size_configurations


#################################################
############## Helper functions #################
#################################################

def setup_vprint(args):
    """
    SETUP_VPRINT defines the function vprint as the normal print function when
    verbose output is enabled, or alternatively as a function that does nothing.
    """
    global vprint
    vprint = print if args.verbose else lambda *a, **k: None


def call_rocsolver_bench(bench_executable, *args):
    """
    CALL_ROCSOLVER_BENCH executes system call to the benchmark
    client executable with the given list of arguments
    """
    cmd = [bench_executable]
    for arg in args:
        if isinstance(arg, str):
            cmd.extend(shlex.split(arg, False, False))
        elif isinstance(arg, collections.Sequence):
            cmd.extend(arg)
        else:
            cmd.push(str(arg))
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    vprint('executing {}'.format(' '.join(cmd)))
    stdout, stderr = process.communicate()
    return (str(stdout, encoding='utf-8', errors='surrogateescape'),
            str(stderr, encoding='utf-8', errors='surrogateescape'),
            process.returncode)


def parse_internal_profiler_output(output_text):
    """
    Parse rocSOLVER internal profiler output to extract kernel timing data.

    The profiler produces hierarchical output showing function calls and their timing.
    Kernels are identified as leaf nodes (functions with no nested calls).

    Args:
        output_text: Raw stdout string from rocsolver-bench with profiling enabled

    Returns:
        tuple: (timing_us, kernels) where:
            - timing_us is the total execution time in microseconds
            - kernels is a list of dicts with kernel timing information

    Raises:
        ValueError: If the output format is invalid or missing expected sections
    """
    lines = output_text.strip().split('\n')
    if not lines:
        raise ValueError("Empty profiler output")

    # Extract timing from first line
    try:
        timing_us = float(lines[0].strip())
    except (ValueError, IndexError):
        raise ValueError("Could not parse timing from first line")

    # Find profile section marker
    profile_start = -1
    for i, line in enumerate(lines):
        if "------- PROFILE -------" in line:
            profile_start = i + 1
            break

    if profile_start == -1:
        print(lines)
        raise ValueError("No profile section found in output")

    # Parse each line to build function list
    profile_lines = lines[profile_start:]
    all_functions = []

    for i, line in enumerate(profile_lines):
        if not line.strip():
            continue

        # Skip lines that don't have the expected format
        if 'Calls:' not in line or 'Total Time:' not in line:
            continue

        # Calculate indentation level (4 spaces per level)
        indent_level = (len(line) - len(line.lstrip())) // 4

        # Extract function name (everything before first ':')
        name_part = line.lstrip().split(':', 1)[0]

        # Check if name is in parentheses and strip them
        if name_part.startswith('(') and name_part.endswith(')'):
            name = name_part[1:-1]
        else:
            name = name_part

        # Extract calls count
        # Format: "Calls: 10"
        calls_start = line.find('Calls:') + 6
        calls_end = line.find(',', calls_start)
        calls = int(line[calls_start:calls_end].strip())

        # Extract total time
        # Format: "Total Time: 66.899 ms"
        time_start = line.find('Total Time:') + 11
        # Find end - either " ms (in" or just " ms"
        if '(in nested functions:' in line:
            time_end = line.find(' ms', time_start)
        else:
            time_end = line.find(' ms', time_start)
        total_time_ms = float(line[time_start:time_end].strip())

        # Extract nested time if present
        # Format: "(in nested functions: 1059.040 ms)"
        nested_time_ms = 0.0
        if '(in nested functions:' in line:
            nested_start = line.find('(in nested functions:') + 21
            nested_end = line.find(' ms)', nested_start)
            nested_time_ms = float(line[nested_start:nested_end].strip())

        # Check if this is a leaf node (no children at deeper indent level)
        is_leaf = True
        if i + 1 < len(profile_lines):
            next_line = profile_lines[i + 1]
            if next_line.strip() and 'Calls:' in next_line:
                next_indent = (len(next_line) - len(next_line.lstrip())) // 4
                if next_indent > indent_level:
                    is_leaf = False

        all_functions.append({
            'name': name,
            'calls': calls,
            'total_time_ms': total_time_ms,
            'nested_time_ms': nested_time_ms,
            'own_time_ms': total_time_ms - nested_time_ms,
            'is_leaf': is_leaf,
            'indent_level': indent_level
        })

    # Filter to only leaf nodes (kernels)
    kernels = [f for f in all_functions if f['is_leaf']]

    # Strip template parameters from names
    for kernel in kernels:
        clean_name = kernel['name']
        # Remove template parameters (everything after '<')
        if '<' in clean_name:
            clean_name = clean_name[:clean_name.index('<')]
        kernel['clean_name'] = clean_name

    return timing_us, kernels


def save_kernel_data_csv(kernels, output_file):
    """
    Save kernel data to CSV file for graphing and analysis.

    Args:
        kernels: List of kernel dicts from parse_internal_profiler_output()
        output_file: Path to output CSV file

    The CSV format includes:
        - Name: Kernel name (cleaned, without template parameters)
        - Calls: Number of kernel invocations
        - TotalTimeMs: Total execution time in milliseconds
        - OwnTimeMs: Own time (excluding nested calls)
        - Percentage: Percentage of total runtime
    """
    import csv

    # Calculate total time and percentages
    total_time = sum(k['total_time_ms'] for k in kernels)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Calls', 'TotalTimeMs', 'OwnTimeMs', 'Percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for kernel in kernels:
            percentage = 100.0 * kernel['total_time_ms'] / total_time if total_time > 0 else 0.0
            writer.writerow({
                'Name': kernel['clean_name'],
                'Calls': kernel['calls'],
                'TotalTimeMs': kernel['total_time_ms'],
                'OwnTimeMs': kernel['own_time_ms'],
                'Percentage': percentage
            })


def profile_with_rocprofv3(benchmark_string, bench_executable, bench_args,
                            profile_output, graph, top_n):
    """
    Profile using rocprofv3 external profiler.

    Args:
        benchmark_string: String identifier for the benchmark
        bench_executable: Path to rocsolver-bench executable
        bench_args: Arguments to pass to rocsolver-bench
        profile_output: Directory to store profiling results
        graph: Whether to generate graph
        top_n: Number of top kernels to show in graph

    Returns:
        tuple: (success, timing_us) where success is a boolean and timing_us is the execution time
    """
    # Run with rocprofv3
    command_args = (f'--kernel-trace --stats=ON -d {profile_output} '
                    f'-o {benchmark_string} -f csv -- {bench_executable} {bench_args}')

    out_rocprof, err_rocprof, exitcode_rocprof = call_rocsolver_bench("rocprofv3", command_args)

    if exitcode_rocprof != 0:
        print(f"Warning: rocprofv3 call failed for {benchmark_string}: {err_rocprof}", file=sys.stderr)
        return False, None

    rocprof_time = float(out_rocprof)

    # Generate graph if requested
    if graph:
        try:
            from rocsolver_graph import generate_rocprof_graph
            generate_rocprof_graph(profile_output, top_n)
        except Exception as e:
            print(f"Warning: Failed to generate graph for {benchmark_string}: {e}", file=sys.stderr)

    return True, rocprof_time


def profile_with_internal_logger(benchmark_string, bench_executable, bench_args,
                                   profile_output, graph, top_n):
    """
    Profile using rocSOLVER internal logger.

    Args:
        benchmark_string: String identifier for the benchmark
        bench_executable: Path to rocsolver-bench executable
        bench_args: Arguments to pass to rocsolver-bench
        profile_output: Directory to store profiling results
        graph: Whether to generate graph
        top_n: Number of top kernels to show in graph

    Returns:
        tuple: (success, timing_us) where success is a boolean and timing_us is the execution time
    """
    # Use a large --profile arg to get fully nested output
    profiler_args = f'{bench_args} --profile 50 --profile_kernels 1'

    # Execute rocsolver-bench
    out, err, exitcode = call_rocsolver_bench(bench_executable, profiler_args)

    if exitcode != 0:
        print(f"Warning: Benchmark failed for {benchmark_string}: {err}", file=sys.stderr)
        return False, None

    # Parse profiler output
    try:
        timing_us, kernels = parse_internal_profiler_output(out+err)
    except Exception as e:
        print(f"Warning: Failed to parse profiler output for {benchmark_string}: {e}", file=sys.stderr)
        return False, None

    # Create output directory if needed
    if not os.path.exists(profile_output):
        os.makedirs(profile_output)

    # Save raw profiler output
    raw_file = os.path.join(profile_output, f'{benchmark_string}_internal_profile.txt')
    with open(raw_file, 'w') as f:
        f.write(out)

    # Save parsed kernel data as CSV
    kernel_csv = os.path.join(profile_output, f'{benchmark_string}_kernel_data.csv')
    save_kernel_data_csv(kernels, kernel_csv)

    # Generate graph if requested
    if graph:
        try:
            from rocsolver_graph import generate_internal_profiler_graph
            generate_internal_profiler_graph(kernel_csv, profile_output, top_n)
        except Exception as e:
            print(f"Warning: Failed to generate graph for {benchmark_string}: {e}", file=sys.stderr)

    return True, timing_us


def profile_benchmarks(suite, precision, case, bench_executable, output_dir, graph,
                       profiler='rocprofv3', top_n=10):
    """
    Profile benchmarks using specified profiler.

    Args:
        suite: Benchmark suite name
        precision: Precision to use (s/d/c/z)
        case: Size case (small/medium/large)
        bench_executable: Path to rocsolver-bench executable
        output_dir: Directory to store profiling results
        graph: Whether to generate graphs
        profiler: Profiler to use ('rocprofv3' or 'internal')
        top_n: Number of top kernels to show in graphs
    """
    benchmark_generator = SUITES[suite]
    sizenormal, sizebatch = get_size_configurations(case)

    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for row, n, bench_args in benchmark_generator(suite=suite, precision=precision,
                                                    sizenormal=sizenormal, sizebatch=sizebatch):
        # Construct benchmark identifier string
        benchmark_string = "_".join([str(x) for x in list(row.values())])

        # Determine output path for this benchmark's profiling data
        if output_dir:
            profile_output = os.path.join(output_dir, benchmark_string)
        else:
            profile_output = benchmark_string

        print(f"Profiling: {benchmark_string}")

        # Select profiler type
        if profiler == 'rocprofv3':
            success, timing = profile_with_rocprofv3(benchmark_string, bench_executable, bench_args,
                                                      profile_output, graph, top_n)
        elif profiler == 'internal':
            success, timing = profile_with_internal_logger(benchmark_string, bench_executable, bench_args,
                                                             profile_output, graph, top_n)
        else:
            print(f"Error: Unknown profiler '{profiler}'", file=sys.stderr)
            continue

        if success:
            print(f'  Timing: {timing} us')
        else:
            continue


#################################################
######### Main functions ########################
#################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rocsolver-profile',
        description='Profile rocSOLVER benchmarks using rocprofv3 or internal logger.')
    parser.add_argument('-v','--verbose',
            action='store_true',
            help='display more information about operations being performed')
    parser.add_argument('--exe',
            default='../../build/release/clients/staging/rocsolver-bench',
            help='the benchmark executable to run')
    parser.add_argument('-o', '--output-dir',
            dest='output_dir',
            default=None,
            help='directory to store profiling results (default: current directory)')
    parser.add_argument('--graph',
            action='store_true',
            help='generate graphs of profiling results using matplotlib')
    parser.add_argument('--profiler',
            choices=['rocprofv3', 'internal'],
            default='rocprofv3',
            help='profiler to use: rocprofv3 (external) or internal (rocSOLVER logger)')
    parser.add_argument('--top-n',
            type=int,
            default=10,
            help='number of top kernels to show in profiling graphs (default: 10)')
    parser.add_argument('suite',
            choices=SUITES.keys(),
            help='the set of benchmarks to profile')
    parser.add_argument('precision',
            choices=['s', 'd', 'c' , 'z'],
            help='the precision to use for the benchmarks')
    parser.add_argument('case',
            choices=['small', 'medium', 'large'],
            help='the size case to use for the benchmarks')

    args = parser.parse_args()

    if args.top_n < 0:
        sys.exit(f"Error: --top-n must have a positive integer argument")

    setup_vprint(args)

    profile_benchmarks(args.suite, args.precision, args.case, args.exe,
                       args.output_dir, args.graph, args.profiler, args.top_n)
