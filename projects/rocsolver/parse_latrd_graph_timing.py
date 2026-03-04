#!/usr/bin/env python3
"""
Parse LATRD/SYTRD timing results and generate CSV with averaged metrics.

Handles output from test_latrdforsytrd_hipgraph_timing.sh, which benchmarks
two functions per build:
  latrd_forsytrd  -- LATRD-level HIP graph caching (LATRD_GRAPH_MODE != 0)
  sytrd           -- SYTRD-level HIP graph caching (SYTRD_GRAPH_MODE=1)

Usage:
    ./parse_latrd_graph_timing.py [options]

Options:
    -i, --input   Input file containing timing results (default: stdin or first positional arg)
    -o, --output  Output CSV file (default: graph_timing_results.csv)

Output:
    CSV file with averaged timing metrics keyed by
    (function, size, latrd_mode, sytrd_mode, buffer_intrinsics)

    A cross-function comparison section is also printed to the console,
    showing how SYTRD_GRAPH_MODE=1 compares against each LATRD mode for
    the same matrix size and buffer_intrinsics setting.
"""

import sys
import re
from collections import defaultdict
from statistics import mean
import csv
import argparse


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_timing_line(line):
    """
    Parse a single log line and return (line_type, parsed_data).

    Recognised line types
    ---------------------
    'test_config'  -- [TEST_CONFIG] tag emitted by the shell script
    'config'       -- old-style n=N, GRAPH_MODE=M header (backward compat)
    'metric'       -- [LATRD_GRAPH TIMING] / [LATRD_NON-GRAPH TIMING] /
                      [SYTRD CACHE HIT] / [SYTRD CACHE MISS] timing line
    'table_header' -- rocsolver-bench "cpu_time_us  gpu_time_us" header row
    None           -- unrecognised / not relevant
    """

    # ------------------------------------------------------------------
    # [TEST_CONFIG] tag (new format, emitted by the shell script)
    # Example:
    #   [TEST_CONFIG] LATRD_GRAPH_MODE=4, SYTRD_GRAPH_MODE=0,
    #                 LATRD_USE_BUFFER_INTRINSICS=1, FUNCTION=latrd_forsytrd, SIZE=1024
    # ------------------------------------------------------------------
    test_config_match = re.search(
        r'\[TEST_CONFIG\]'
        r'.*LATRD_GRAPH_MODE=(-?\d+)'
        r'.*SYTRD_GRAPH_MODE=(\d+)'
        r'.*LATRD_USE_BUFFER_INTRINSICS=(\d+)'
        r'.*FUNCTION=(\w+)'
        r'.*SIZE=(\d+)',
        line
    )
    if test_config_match:
        return 'test_config', {
            'latrd_mode':        int(test_config_match.group(1)),
            'sytrd_mode':        int(test_config_match.group(2)),
            'buffer_intrinsics': int(test_config_match.group(3)),
            'function':          test_config_match.group(4),
            'size':              int(test_config_match.group(5)),
        }

    # ------------------------------------------------------------------
    # Old-style config header (backward compat, no SYTRD_GRAPH_MODE /
    # FUNCTION fields -- assume latrd_forsytrd, sytrd_mode=0)
    # Example:
    #   [LATRD_GRAPH TIMING] n=1024, k=32, GRAPH_MODE=4
    # ------------------------------------------------------------------
    config_match = re.search(r'n=(\d+).*GRAPH_MODE=(-?\d+)', line)
    if config_match:
        return 'config', {
            'size':     int(config_match.group(1)),
            'latrd_mode': int(config_match.group(2)),
        }

    # ------------------------------------------------------------------
    # Timing metric lines
    # LATRD graph timing:
    #   [LATRD_GRAPH TIMING] Total capture time: 1234.56 us
    #   [LATRD_NON-GRAPH TIMING] Total execution time: 5678.90 us
    # SYTRD cache timing (unconditional, emitted by rocsolver_sytrd_hetrd_template_cached):
    #   [SYTRD CACHE MISS] Graph capture time:        9876.54 us
    #   [SYTRD CACHE MISS] Initial graph launch time:  123.45 us
    #   [SYTRD CACHE HIT]  Graph launch time:           12.34 us
    # ------------------------------------------------------------------
    metric_match = re.search(r'\[(LATRD.*TIMING|SYTRD CACHE (?:HIT|MISS))\]\s+(.*?):\s+([\d.]+)\s*us', line)
    if metric_match:
        prefix      = metric_match.group(1)   # e.g. "SYTRD CACHE HIT"
        metric_name = metric_match.group(2).strip()
        value       = float(metric_match.group(3))
        return 'metric', {'prefix': prefix, 'name': metric_name, 'value': value}

    # ------------------------------------------------------------------
    # rocsolver-bench table header
    # ------------------------------------------------------------------
    if 'cpu_time_us' in line and 'gpu_time_us' in line:
        return 'table_header', None

    return None, None


def parse_results_file(filename):
    """
    Parse the results file and return collected timing data.

    Returns
    -------
    dict  {(function, size, latrd_mode, sytrd_mode, buffer_intrinsics):
               {metric_name: [float, ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))

    current = {
        'function':          'latrd_forsytrd',  # safe default for old logs
        'size':              None,
        'latrd_mode':        None,
        'sytrd_mode':        0,
        'buffer_intrinsics': None,
    }
    expecting_gpu_time = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip grep separator lines
            if line == '--':
                continue

            line_type, parsed = parse_timing_line(line)

            if line_type == 'test_config':
                current = {
                    'function':          parsed['function'],
                    'size':              parsed['size'],
                    'latrd_mode':        parsed['latrd_mode'],
                    'sytrd_mode':        parsed['sytrd_mode'],
                    'buffer_intrinsics': parsed['buffer_intrinsics'],
                }
                expecting_gpu_time = False

            elif line_type == 'config':
                # Old format — no function or sytrd_mode fields
                current['size']       = parsed['size']
                current['latrd_mode'] = parsed['latrd_mode']
                if current['sytrd_mode'] is None:
                    current['sytrd_mode'] = 0
                if current['buffer_intrinsics'] is None:
                    current['buffer_intrinsics'] = 0
                expecting_gpu_time = False

            elif line_type == 'metric':
                if current['size'] is not None and current['latrd_mode'] is not None:
                    key = (
                        current['function'],
                        current['size'],
                        current['latrd_mode'],
                        current['sytrd_mode'],
                        current['buffer_intrinsics'] if current['buffer_intrinsics'] is not None else 0,
                    )
                    # Prefix metric name with its source category so that
                    # SYTRD CACHE HIT / MISS lines don't collide with LATRD ones
                    full_name = f"[{parsed['prefix']}] {parsed['name']}"
                    data[key][full_name].append(parsed['value'])

            elif line_type == 'table_header':
                expecting_gpu_time = True

            elif expecting_gpu_time and line and current['size'] is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        gpu_time = float(parts[1])
                        key = (
                            current['function'],
                            current['size'],
                            current['latrd_mode'],
                            current['sytrd_mode'],
                            current['buffer_intrinsics'] if current['buffer_intrinsics'] is not None else 0,
                        )
                        data[key]['GPU time (benchmark)'].append(gpu_time)
                    except (ValueError, IndexError):
                        pass
                expecting_gpu_time = False

    return data


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def compute_averages(data):
    """Average each metric list into a single float."""
    return {
        key: {name: mean(vals) for name, vals in metrics.items() if vals}
        for key, metrics in data.items()
    }


def compute_percentages_and_dominant(averages):
    """
    Add percentage columns (relative to GPU benchmark time) and identify
    the dominant overhead component for LATRD graph builds.
    """
    latrd_overhead_metrics = [
        'Total capture time',
        'Total instantiation time',
        'Total launch time',
    ]

    for key, metrics in averages.items():
        gpu_time = metrics.get('GPU time (benchmark)', 0)
        if gpu_time <= 0:
            continue

        # Percentages for LATRD overhead components
        for base in latrd_overhead_metrics:
            full = f'[LATRD_GRAPH TIMING] {base}'
            if full in metrics:
                averages[key][f'{full} (%)'] = (metrics[full] / gpu_time) * 100.0

        # Dominant LATRD overhead
        max_val, dominant = 0.0, 'None'
        for base in latrd_overhead_metrics:
            full = f'[LATRD_GRAPH TIMING] {base}'
            if full in metrics and metrics[full] > max_val:
                max_val = metrics[full]
                dominant = base.replace('Total ', '').replace(' time', '')
        averages[key]['Dominant overhead component'] = dominant

    return averages


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

# Ordered list of all metric columns we want in the CSV (human-readable names).
# Rows that don't have a particular metric will get an empty cell.
_METRIC_COLUMNS = [
    'GPU time (benchmark)',
    # LATRD non-graph (baseline)
    '[LATRD_NON-GRAPH TIMING] Total execution time',
    # LATRD graph timing
    '[LATRD_GRAPH TIMING] Total capture time',
    '[LATRD_GRAPH TIMING] Total capture time (%)',
    '[LATRD_GRAPH TIMING] Total instantiation time',
    '[LATRD_GRAPH TIMING] Total instantiation time (%)',
    '[LATRD_GRAPH TIMING] Total launch time',
    '[LATRD_GRAPH TIMING] Total launch time (%)',
    '[LATRD_GRAPH TIMING] TOTAL OVERHEAD',
    '[LATRD_GRAPH TIMING] GRAPH EXECUTION TIME',
    'Dominant overhead component',
    # SYTRD cache timing (new)
    '[SYTRD CACHE MISS] Graph capture time',
    '[SYTRD CACHE MISS] Initial graph launch time',
    '[SYTRD CACHE HIT] Graph launch time',
]


def generate_csv(averages, output_file='graph_timing_results.csv'):
    """Write the full results table to a CSV file."""

    id_columns = [
        'Function', 'Matrix_Size',
        'LATRD_GRAPH_MODE', 'SYTRD_GRAPH_MODE',
        'LATRD_USE_BUFFER_INTRINSICS',
    ]

    # Collect any extra metric columns present in the data but not in
    # our predefined list (future-proofs against new timing lines).
    all_metric_cols = list(_METRIC_COLUMNS)
    known = set(_METRIC_COLUMNS)
    for metrics in averages.values():
        for name in metrics:
            if name not in known:
                all_metric_cols.append(name)
                known.add(name)

    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(id_columns + all_metric_cols)

        for (func, size, latrd_mode, sytrd_mode, buf_int) in sorted_keys:
            metrics = averages[(func, size, latrd_mode, sytrd_mode, buf_int)]
            row = [func, size, latrd_mode, sytrd_mode, buf_int]
            for col in all_metric_cols:
                val = metrics.get(col, '')
                if val == '':
                    row.append('')
                elif isinstance(val, str):
                    row.append(val)
                else:
                    row.append(f'{val:.2f}')
            writer.writerow(row)

    print(f"CSV file generated: {output_file}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(averages):
    """Print a human-readable per-key summary."""
    print("\n" + "=" * 80)
    print("TIMING RESULTS SUMMARY")
    print("=" * 80)

    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    for (func, size, latrd_mode, sytrd_mode, buf_int) in sorted_keys:
        print(
            f"\nFunction: {func}  |  Size: {size}  |  "
            f"LATRD_GRAPH_MODE: {latrd_mode}  |  SYTRD_GRAPH_MODE: {sytrd_mode}  |  "
            f"Buffer intrinsics: {buf_int}"
        )
        print("-" * 80)
        metrics = averages[(func, size, latrd_mode, sytrd_mode, buf_int)]

        if 'GPU time (benchmark)' in metrics:
            print(f"  {'GPU time (benchmark)':50s}: {metrics['GPU time (benchmark)']:10.2f} us\n")

        non_pct = {k: v for k, v in metrics.items()
                   if k not in ('GPU time (benchmark)', 'Dominant overhead component')
                   and not k.endswith('(%)')}
        for k in sorted(non_pct):
            print(f"  {k:50s}: {non_pct[k]:10.2f} us")

        pct_keys = [k for k in metrics if k.endswith('(%)')]
        if pct_keys:
            print()
            for k in sorted(pct_keys):
                print(f"  {k:50s}: {metrics[k]:10.2f} %")

        if 'Dominant overhead component' in metrics:
            print(f"\n  {'Dominant overhead component':50s}: {metrics['Dominant overhead component']}")


def print_cross_function_comparison(averages):
    """
    Compare sytrd GPU time across all modes (LATRD modes + SYTRD cache mode)
    and also compare latrd_forsytrd across LATRD modes.

    Groups by (size, buffer_intrinsics) and shows a speedup table with
    LATRD_GRAPH_MODE=0 / sytrd as the common baseline.
    """
    print("\n" + "=" * 80)
    print("CROSS-MODE COMPARISON  (GPU time, us)")
    print("Baseline = sytrd with LATRD_GRAPH_MODE=0, SYTRD_GRAPH_MODE=0")
    print("=" * 80)

    # Collect all (size, buf_int) pairs
    combos = sorted({(k[1], k[4]) for k in averages})

    for (size, buf_int) in combos:
        print(f"\n  Matrix size: {size}  |  Buffer intrinsics: {buf_int}")
        print(f"  {'Mode':<55s} {'Function':<20s} {'gpu_time_us':>12s} {'vs baseline':>12s}")
        print("  " + "-" * 100)

        # Baseline: sytrd, LATRD=0, SYTRD=0
        baseline_key = ('sytrd', size, 0, 0, buf_int)
        baseline_gpu = averages.get(baseline_key, {}).get('GPU time (benchmark)')

        # All keys matching this (size, buf_int)
        relevant = sorted(
            [k for k in averages if k[1] == size and k[4] == buf_int],
            key=lambda k: (k[2], k[3], k[0])  # latrd_mode, sytrd_mode, function
        )

        for (func, s, latrd_mode, sytrd_mode, bi) in relevant:
            gpu = averages[(func, s, latrd_mode, sytrd_mode, bi)].get('GPU time (benchmark)', None)
            if gpu is None:
                continue
            if func == 'sytrd' and latrd_mode == 0 and sytrd_mode == 0:
                mode_label = "LATRD_MODE=0, SYTRD_MODE=0  [BASELINE]"
            elif sytrd_mode == 1:
                mode_label = f"SYTRD_MODE=1 (full-call cache), LATRD_MODE={latrd_mode}"
            else:
                mode_label = f"LATRD_MODE={latrd_mode}, SYTRD_MODE={sytrd_mode}"

            if baseline_gpu and baseline_gpu > 0:
                speedup = baseline_gpu / gpu
                vs_str  = f"{speedup:+.3f}x"
            else:
                vs_str = "N/A"

            print(f"  {mode_label:<55s} {func:<20s} {gpu:>12.2f} {vs_str:>12s}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parse LATRD/SYTRD timing results and generate CSV with averaged metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse results from file and output to default CSV
  %(prog)s -i test_results.txt

  # Specify both input and output files
  %(prog)s -i test_results.txt -o my_results.csv

  # Use positional argument (backward compatibility)
  %(prog)s test_results.txt
        """
    )
    parser.add_argument('-i', '--input',  dest='input_file',
                        help='Input file containing timing results')
    parser.add_argument('-o', '--output', dest='output_file',
                        default='graph_timing_results.csv',
                        help='Output CSV file (default: graph_timing_results.csv)')
    parser.add_argument('input_positional', nargs='?',
                        help='Input file (positional argument for backward compatibility)')

    args = parser.parse_args()
    input_file  = args.input_file or args.input_positional
    output_file = args.output_file

    if not input_file:
        parser.print_help()
        sys.exit(1)

    try:
        print(f"Parsing results from: {input_file}")

        data = parse_results_file(input_file)

        if not data:
            print("ERROR: No timing data found in the input file.")
            print("\nExpected format (new):")
            print("  [TEST_CONFIG] LATRD_GRAPH_MODE=4, SYTRD_GRAPH_MODE=0, "
                  "LATRD_USE_BUFFER_INTRINSICS=1, FUNCTION=latrd_forsytrd, SIZE=1024")
            print("  [LATRD_GRAPH TIMING] Total capture time: 1234.56 us")
            print("  [SYTRD CACHE MISS] Graph capture time:         9876.54 us")
            print("  [SYTRD CACHE MISS] Initial graph launch time:   123.45 us")
            print("  [SYTRD CACHE HIT] Graph launch time:             12.34 us")
            print("  cpu_time_us     gpu_time_us")
            print("  395068          19394")
            sys.exit(1)

        print(f"Found {len(data)} unique (function, size, latrd_mode, sytrd_mode, buffer_intrinsics) combinations")

        averages = compute_averages(data)
        averages = compute_percentages_and_dominant(averages)

        print_summary(averages)
        print_cross_function_comparison(averages)
        generate_csv(averages, output_file)

        print("\n" + "=" * 80)
        print("Processing complete!")
        print("=" * 80)

    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
