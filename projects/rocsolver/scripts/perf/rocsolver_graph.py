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
Graphing script for rocSOLVER benchmark results.

This script reads CSV files containing benchmark results and generates
performance graphs showing runtime as a function of input size.
"""

import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt


def generate_benchmark_graph(csv_path, output_path=None):
    """
    Generate a graph of benchmark results from a CSV file.

    Results are grouped by all parameters except 'n', and separate lines
    are plotted for each group showing runtime vs. input size.

    Args:
        csv_path: Path to CSV file containing benchmark results
        output_path: Optional output path for the graph (defaults to CSV basename)
    """
    if not os.path.exists(csv_path):
        sys.exit(f"Error: CSV file not found: {csv_path}")

    # Read CSV and group data
    groups = {}
    precision_index = None
    size_index = None
    graph_title = None

    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames

        if not fieldnames or 'n' not in fieldnames or 'gpu_time_us' not in fieldnames:
            sys.exit("Error: CSV must contain 'n' and 'gpu_time_us' columns")

        for row in reader:
            # Find indices on first row
            if precision_index is None and 'precision' in fieldnames:
                precision_index = fieldnames.index('precision')
                size_index = fieldnames.index('n')
                graph_title = row.get('name', 'Benchmark Results')

            # Create group key from all fields except 'n', 'gpu_time_us', 'log_n', 'log_gpu_time_us'
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
                print(f"Warning: Skipping invalid row: {e}")
                continue

    if not groups:
        sys.exit("Error: No valid data found in CSV file")

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))

    for group_name, data in groups.items():
        ax.plot(data['x'], data['y'], marker='o', label=group_name, linewidth=2)

    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Total Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Only show legend if there are multiple groups
    if len(groups) > 1:
        ax.legend(fontsize=8, loc='best')

    title = graph_title or os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(f'{title} - Runtime by Size', fontsize=14)

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(csv_path)[0] + '.png'

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Graph saved to: {output_path}")


def generate_kernel_breakdown_graph(csv_path, output_path, title, n=10,
                                     strip_templates=False, sort_by_percentage=False):
    """
    Generate a bar chart showing kernel runtime breakdown.

    Generic function that creates a bar chart from a CSV file containing
    kernel performance data. Shows the top N kernels by percentage, with
    remaining kernels grouped into an "Other" category.

    Args:
        csv_path: Path to CSV file with 'Name' and 'Percentage' columns
        output_path: Path where the PNG graph will be saved
        title: Title for the graph
        n: Number of top kernels to display (default: 10)
        strip_templates: Whether to strip template parameters from names (default: False)
        sort_by_percentage: Whether to sort data by percentage before processing (default: False)
    """
    if not os.path.exists(csv_path):
        sys.exit(f"Error: CSV file not found: {csv_path}")

    # Read kernel data from CSV
    kernels = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['Name']

            # Strip template parameters if requested
            if strip_templates and '<' in name:
                name = name[:name.index('<')]

            kernels.append({
                'name': name,
                'percentage': float(row['Percentage'])
            })

    if not kernels:
        sys.exit("Error: No kernel data found in CSV")

    # Sort by percentage if requested (e.g., for unsorted CSV data)
    if sort_by_percentage:
        kernels.sort(key=lambda k: k['percentage'], reverse=True)

    # Extract top N kernels and accumulate rest into "Other"
    x = []
    y = []
    other_percentage = 0.0

    for i, kernel in enumerate(kernels):
        if i < n:
            x.append(kernel['name'])
            y.append(kernel['percentage'])
        else:
            other_percentage += kernel['percentage']

    # Add "Other" category if there are remaining kernels
    if other_percentage > 0:
        x.append("Other")
        y.append(other_percentage)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(5, 10))

    ax.set_ylabel('Proportion of Run Time (%)', fontsize=12)
    ax.set_xlabel('Kernel Name', fontsize=12)

    rects = ax.bar(x, y)
    ax.bar_label(rects)

    ax.set_ylim(0, 1.2*max(y))
    ax.tick_params("x", rotation=90)

    fig.suptitle(title, fontsize=14)

    plt.subplots_adjust(bottom=0.40)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Graph saved to: {output_path}")


def generate_rocprof_graph(rocprof_dir, n=10):
    """
    Generate a graph decomposing total runtime into individual kernels.

    This function uses results produced by rocprofv3 to show which kernels
    contribute most to the total runtime.

    Args:
        rocprof_dir: Directory containing rocprof results
        n: Number of top kernels to display (default: 10)
    """
    kernel_stats_file = os.path.join(rocprof_dir, f'{os.path.basename(rocprof_dir)}_kernel_stats.csv')
    output_path = os.path.join(rocprof_dir, f'{os.path.basename(rocprof_dir)}_kernel_breakdown.png')

    generate_kernel_breakdown_graph(
        csv_path=kernel_stats_file,
        output_path=output_path,
        title='Proportion of Time (%) by Kernel',
        n=n,
        strip_templates=True,
        sort_by_percentage=False  # rocprof CSV is already sorted
    )


def generate_internal_profiler_graph(kernel_csv, output_dir, n=10):
    """
    Generate a graph decomposing total runtime into individual kernels.

    This function uses results from rocSOLVER internal logger to show which
    kernels contribute most to the total runtime.

    Args:
        kernel_csv: Path to CSV file containing parsed kernel data
        output_dir: Directory to save the graph
        n: Number of top kernels to display (default: 10)
    """
    output_path = os.path.join(output_dir, f'{os.path.basename(output_dir)}_kernel_breakdown.png')

    generate_kernel_breakdown_graph(
        csv_path=kernel_csv,
        output_path=output_path,
        title='Proportion of Time (%) by Kernel (Internal Profiler)',
        n=n,
        strip_templates=False,  # Names already cleaned during CSV creation
        sort_by_percentage=True  # Need to sort the data
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rocsolver-graph',
        description='Generate performance graphs from rocSOLVER benchmark results.')
    parser.add_argument('input',
            help='Path to CSV file (for benchmark graphs) or directory (for rocprof graphs)')
    parser.add_argument('-o', '--output',
            dest='output_path',
            default=None,
            help='Output path for the generated graph (default: input basename with .png extension)')
    parser.add_argument('--rocprof',
            action='store_true',
            help='Generate graph from rocprof kernel stats instead of benchmark CSV')
    parser.add_argument('--top-n',
            type=int,
            default=10,
            help='Number of top kernels to show in rocprof graph (default: 10)')

    args = parser.parse_args()

    if args.rocprof:
        if not os.path.isdir(args.input):
            sys.exit(f"Error: For rocprof graphs, input must be a directory: {args.input}")
        generate_rocprof_graph(args.input, args.top_n)
    else:
        if not os.path.isfile(args.input):
            sys.exit(f"Error: For benchmark graphs, input must be a CSV file: {args.input}")
        generate_benchmark_graph(args.input, args.output_path)
