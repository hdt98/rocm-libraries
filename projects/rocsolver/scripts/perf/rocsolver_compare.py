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
Speedup comparison script for rocSOLVER benchmark results.

This script compares two benchmark CSV files and generates bar charts showing
speedup for each parameter group. Speedup is calculated as baseline_time / comparison_time,
where values > 1.0 indicate performance improvements and values < 1.0 indicate regressions.
"""

import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt


def read_and_group_csv(csv_path):
    """
    Read a benchmark CSV file and group data by parameters.

    This function reads a CSV file containing benchmark results and groups
    the data by all parameters except 'n' (the input size variable).

    Args:
        csv_path: Path to the CSV file containing benchmark results

    Returns:
        tuple: (groups, graph_title, fieldnames)
        - groups: dict of {group_key: {'x': [n_values], 'y': [time_ms_values]}}
        - graph_title: title extracted from 'name' column
        - fieldnames: list of CSV column names

    Raises:
        SystemExit: If file doesn't exist, is empty, or missing required columns
    """
    if not os.path.exists(csv_path):
        sys.exit(f"Error: CSV file not found: {csv_path}")

    groups = {}
    graph_title = None
    fieldnames = None

    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        if not fieldnames or 'n' not in fieldnames or 'gpu_time_us' not in fieldnames:
            sys.exit(f"Error: CSV must contain 'n' and 'gpu_time_us' columns: {csv_path}")

        for row in reader:
            # Extract graph title from first row
            if graph_title is None:
                graph_title = row.get('name', 'Benchmark Results')

            # Create group key from all fields except 'n', 'gpu_time_us', and log/name fields
            group_keys = []
            for field in fieldnames:
                if field not in ['n', 'gpu_time_us', 'log_n', 'log_gpu_time_us', 'name', 'name_test']:
                    group_keys.append(f"{field}_{row[field]}")

            group_key = "_".join(group_keys)

            if group_key not in groups:
                groups[group_key] = {'x': [], 'y': []}

            try:
                n = int(row['n'])
                time_us = float(row['gpu_time_us'])
                groups[group_key]['x'].append(n)
                groups[group_key]['y'].append(time_us / 1000.0)  # Convert to ms
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid row in {csv_path}: {e}")
                continue

    if not groups:
        sys.exit(f"Error: No valid data found in CSV file: {csv_path}")

    return groups, graph_title, fieldnames


def validate_matching_groups(baseline_groups, comparison_groups, baseline_path, comparison_path):
    """
    Validate that both CSV files have matching groups and n values.

    This ensures that speedup comparisons are meaningful by verifying that
    both benchmark runs tested the same parameter configurations and input sizes.

    Args:
        baseline_groups: groups dict from baseline CSV
        comparison_groups: groups dict from comparison CSV
        baseline_path: path to baseline CSV (for error messages)
        comparison_path: path to comparison CSV (for error messages)

    Raises:
        SystemExit: If groups or n values don't match between files
    """
    baseline_keys = set(baseline_groups.keys())
    comparison_keys = set(comparison_groups.keys())

    # Check group keys match
    if baseline_keys != comparison_keys:
        missing_in_comparison = baseline_keys - comparison_keys
        missing_in_baseline = comparison_keys - baseline_keys

        error_msg = "Error: Benchmark groups don't match between files\n"
        if missing_in_comparison:
            error_msg += f"  Missing in {comparison_path}:\n"
            for key in sorted(missing_in_comparison):
                error_msg += f"    - {key}\n"
        if missing_in_baseline:
            error_msg += f"  Missing in {baseline_path}:\n"
            for key in sorted(missing_in_baseline):
                error_msg += f"    - {key}\n"
        sys.exit(error_msg)

    # Check n values match for each group
    for group_key in baseline_keys:
        baseline_n = baseline_groups[group_key]['x']
        comparison_n = comparison_groups[group_key]['x']

        if baseline_n != comparison_n:
            sys.exit(f"Error: n values don't match for group '{group_key}'\n"
                    f"  Baseline n:   {baseline_n}\n"
                    f"  Comparison n: {comparison_n}\n")


def calculate_speedups(baseline_groups, comparison_groups):
    """
    Calculate speedup for each group and n value.

    Speedup is calculated as baseline_time / comparison_time:
    - speedup > 1.0: Performance improvement (comparison is faster)
    - speedup = 1.0: No change
    - speedup < 1.0: Performance regression (comparison is slower)

    Args:
        baseline_groups: groups dict from baseline CSV
        comparison_groups: groups dict from comparison CSV

    Returns:
        dict: {group_key: {'x': [n_values], 'speedup': [speedup_values]}}
    """
    speedup_data = {}

    for group_key in baseline_groups.keys():
        baseline_times = baseline_groups[group_key]['y']
        comparison_times = comparison_groups[group_key]['y']
        n_values = baseline_groups[group_key]['x']

        # Calculate speedup = baseline / comparison (>1 means improvement)
        speedups = []
        for base, comp in zip(baseline_times, comparison_times):
            if comp == 0:
                print(f"Warning: Division by zero for group '{group_key}', n={n_values[len(speedups)]}. Skipping.")
                speedups.append(0)
            else:
                speedups.append(base / comp)

        speedup_data[group_key] = {'x': n_values, 'speedup': speedups}

    return speedup_data


def generate_combined_speedup_graph(speedup_data, title, output_path):
    """
    Generate a single graph with all groups plotted as separate lines.

    This creates a combined speedup graph showing all parameter groups
    on one plot for easy comparison.

    Args:
        speedup_data: dict of {group_key: {'x': [...], 'speedup': [...]}}
        title: base title for the graph
        output_path: output path for the graph
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group as a separate line
    for group_key, data in speedup_data.items():
        ax.plot(data['x'], data['speedup'], marker='o', label=group_key, linewidth=2)

    # Reference line at y=1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No change (1.0x)')

    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title(f'Speedup: {title}', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Show legend if multiple groups
    if len(speedup_data) > 1:
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Speedup graph saved to: {output_path}")


def generate_separate_speedup_graphs(speedup_data, title, output_path):
    """
    Generate separate graphs for each group, one graph per parameter configuration.

    Args:
        speedup_data: dict of {group_key: {'x': [...], 'speedup': [...]}}
        title: base title for the graph
        output_path: base output path for graphs
    """
    # Determine base output path
    base_path = os.path.splitext(output_path)[0]

    # Generate one graph per group
    for group_key, data in speedup_data.items():
        # Title includes group key to identify parameters
        graph_title = f'Speedup: {title} ({group_key})'

        # Filename includes group key
        group_output_path = f'{base_path}_{group_key}.png'

        # Create the graph
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(data['x'], data['speedup'], marker='o', linewidth=2)

        # Reference line at y=1.0
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No change (1.0x)')

        ax.set_xlabel('Input Size (n)', fontsize=12)
        ax.set_ylabel('Speedup Factor', fontsize=12)
        ax.set_title(graph_title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(group_output_path, dpi=150)
        plt.close()

        print(f"Speedup graph saved to: {group_output_path}")


def generate_speedup_comparison(baseline_csv, comparison_csv, output_dir=None, output_prefix="speedup_comparison", separate_groups=False):
    """
    Generate speedup comparison graphs from two benchmark CSVs.

    Reads both CSV files, validates they have matching groups and n values,
    calculates speedup for each configuration, and generates line graphs.

    Results are either:
    - plotted as separate lines on one graph (default), or
    - plotted as separate individual graphs (if separate_groups=True)

    Args:
        baseline_csv: path to baseline/reference benchmark results (denominator)
        comparison_csv: path to comparison/new benchmark results (numerator)
        output_dir: optional output directory (default: current directory)
        output_prefix: optional prefix for output filenames (default: "speedup_comparison")
        separate_groups: if True, generate one graph per group instead of one combined graph
    """
    print(f"Baseline:   {baseline_csv}")
    print(f"Comparison: {comparison_csv}\n")

    # Read and group both CSV files
    baseline_groups, baseline_title, _ = read_and_group_csv(baseline_csv)
    comparison_groups, comparison_title, _ = read_and_group_csv(comparison_csv)

    # Warn if function names differ
    if baseline_title != comparison_title:
        print(f"Warning: Comparing different benchmarks:")
        print(f"  Baseline:   {baseline_title}")
        print(f"  Comparison: {comparison_title}\n")

    # Validate matching groups and calculate speedups
    validate_matching_groups(baseline_groups, comparison_groups, baseline_csv, comparison_csv)
    speedup_data = calculate_speedups(baseline_groups, comparison_groups)

    # Set up output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    title = baseline_title or "Benchmark Results"

    # Choose plotting strategy based on flag
    if separate_groups:
        # Generate one graph per group
        base_output_path = os.path.join(output_dir, f"{output_prefix}.png")
        generate_separate_speedup_graphs(speedup_data, title, base_output_path)
        print(f"\nGenerated {len(speedup_data)} speedup comparison graph(s)")
    else:
        # Generate one combined graph with all groups
        output_path = os.path.join(output_dir, f"{output_prefix}.png")
        generate_combined_speedup_graph(speedup_data, title, output_path)


def main():
    """Main entry point for the speedup comparison script."""
    parser = argparse.ArgumentParser(
        prog='rocsolver-compare',
        description='Compare two rocSOLVER benchmark results and generate speedup graphs.\n\n'
                    'Speedup = baseline_time / comparison_time\n'
                    '  > 1.0 = Performance improvement (faster)\n'
                    '  < 1.0 = Performance regression (slower)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python rocsolver_compare.py old_version.csv new_version.csv\n'
               '  python rocsolver_compare.py baseline.csv optimized.csv -o results/')

    parser.add_argument('baseline',
                       metavar='BASELINE_CSV',
                       help='Path to baseline/reference benchmark CSV file (denominator in speedup calculation)')
    parser.add_argument('comparison',
                       metavar='COMPARISON_CSV',
                       help='Path to comparison/new benchmark CSV file (numerator in speedup calculation)')
    parser.add_argument('-o', '--output-dir',
                       default=None,
                       help='Output directory for graphs (default: current directory)')
    parser.add_argument('--output-prefix',
                       default='speedup_comparison',
                       help='Prefix for output filenames (default: speedup_comparison)')
    parser.add_argument('--separate-groups',
                       action='store_true',
                       help='Generate separate graph for each parameter group instead of combined graph')

    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        sys.exit(f"Error: Baseline CSV not found: {args.baseline}")
    if not os.path.exists(args.comparison):
        sys.exit(f"Error: Comparison CSV not found: {args.comparison}")

    generate_speedup_comparison(args.baseline, args.comparison, args.output_dir, args.output_prefix, args.separate_groups)


if __name__ == '__main__':
    main()
