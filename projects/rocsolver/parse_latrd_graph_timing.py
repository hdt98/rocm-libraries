#!/usr/bin/env python3
"""
Parse LATRD timing results and generate CSV with averaged metrics.

Usage:
    ./parse_latrd_graph_timing.py [options]
    
Options:
    -i, --input   Input file containing timing results (default: stdin or first positional arg)
    -o, --output  Output CSV file (default: graph_timing_results.csv)
    
Output:
    CSV file with averaged timing metrics including buffer intrinsics configuration
"""

import sys
import re
from collections import defaultdict
from statistics import mean
import csv
import argparse


def parse_timing_line(line):
    """Parse a timing line and extract metric name and value."""
    # Match patterns like:
    # [TEST_CONFIG] LATRD_GRAPH_MODE=4, LATRD_USE_BUFFER_INTRINSICS=1, SIZE=1024
    # [LATRD_GRAPH TIMING] n=1024, k=32, GRAPH_MODE=4
    # [LATRD_GRAPH TIMING] Total capture time: 1234.56 us
    # [LATRD_NON-GRAPH TIMING] Total execution time: 5678.90 us
    # Benchmark table: cpu_time_us     gpu_time_us
    #                  395068          19394
    
    # Check for test configuration line (new format with buffer intrinsics)
    test_config_match = re.search(r'\[TEST_CONFIG\].*LATRD_GRAPH_MODE=(-?\d+).*LATRD_USE_BUFFER_INTRINSICS=(\d+).*SIZE=(\d+)', line)
    if test_config_match:
        return 'test_config', {
            'mode': int(test_config_match.group(1)),
            'buffer_intrinsics': int(test_config_match.group(2)),
            'size': int(test_config_match.group(3))
        }
    
    # Check if this is a configuration line (old format)
    config_match = re.search(r'n=(\d+).*GRAPH_MODE=(-?\d+)', line)
    if config_match:
        return 'config', {'n': int(config_match.group(1)), 'mode': int(config_match.group(2))}
    
    # Check for timing metric lines (now in microseconds)
    metric_match = re.search(r'\[LATRD.*TIMING\]\s+(.*?):\s+([\d.]+)\s*us', line)
    if metric_match:
        metric_name = metric_match.group(1).strip()
        value = float(metric_match.group(2))
        return 'metric', {'name': metric_name, 'value': value}
    
    # Check for benchmark timing table header (mark that we should read next line)
    if 'cpu_time_us' in line and 'gpu_time_us' in line:
        return 'table_header', None
    
    return None, None


def parse_results_file(filename):
    """
    Parse the results file and organize timing data.
    
    Returns:
        dict: {(size, mode, buffer_intrinsics): {metric_name: [values]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    current_size = None
    current_mode = None
    current_buffer_intrinsics = None
    expecting_gpu_time = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip grep separator lines
            if line == '--':
                continue
            
            line_type, parsed = parse_timing_line(line)
            
            if line_type == 'test_config':
                # New format with explicit buffer intrinsics configuration
                current_size = parsed['size']
                current_mode = parsed['mode']
                current_buffer_intrinsics = parsed['buffer_intrinsics']
                expecting_gpu_time = False
            elif line_type == 'config':
                # Old format - try to infer from context or default to 0
                current_size = parsed['n']
                current_mode = parsed['mode']
                # If buffer_intrinsics wasn't set yet, default to 0
                if current_buffer_intrinsics is None:
                    current_buffer_intrinsics = 0
                expecting_gpu_time = False
            elif line_type == 'metric' and current_size is not None and current_mode is not None:
                metric_name = parsed['name']
                value = parsed['value']
                # Ensure buffer_intrinsics has a value (default to 0 if not set)
                if current_buffer_intrinsics is None:
                    current_buffer_intrinsics = 0
                data[(current_size, current_mode, current_buffer_intrinsics)][metric_name].append(value)
            elif line_type == 'table_header':
                # Next non-empty line should contain the timing values
                expecting_gpu_time = True
            elif expecting_gpu_time and line and current_size is not None and current_mode is not None:
                # Try to parse the timing data line (format: "395068          19394")
                # This should be the immediately following line due to grep -A1
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Second value is gpu_time_us
                        gpu_time = float(parts[1])
                        # Ensure buffer_intrinsics has a value
                        if current_buffer_intrinsics is None:
                            current_buffer_intrinsics = 0
                        data[(current_size, current_mode, current_buffer_intrinsics)]['GPU time (benchmark)'].append(gpu_time)
                        expecting_gpu_time = False
                    except (ValueError, IndexError):
                        # If parsing fails, reset flag and continue
                        expecting_gpu_time = False
    
    return data


def compute_averages(data):
    """
    Compute average for each metric across all iterations.
    
    Args:
        data: dict from parse_results_file
        
    Returns:
        dict: {(size, mode, buffer_intrinsics): {metric_name: avg_value}}
    """
    averages = {}
    
    for (size, mode, buffer_intrinsics), metrics in data.items():
        averages[(size, mode, buffer_intrinsics)] = {}
        for metric_name, values in metrics.items():
            if values:
                averages[(size, mode, buffer_intrinsics)][metric_name] = mean(values)
    
    return averages


def compute_percentages_and_dominant(averages):
    """
    Add percentage columns and identify dominant overhead component.
    
    Args:
        averages: dict from compute_averages
        
    Returns:
        dict: same structure with added percentage columns
    """
    overhead_metrics = ['Total capture time', 'Total instantiation time', 'Total launch time']
    
    for (size, mode, buffer_intrinsics), metrics in averages.items():
        gpu_time = metrics.get('GPU time (benchmark)', 0)
        
        if gpu_time > 0:
            # Calculate percentages for each overhead component
            for metric_name in overhead_metrics:
                if metric_name in metrics:
                    percentage = (metrics[metric_name] / gpu_time) * 100.0
                    averages[(size, mode, buffer_intrinsics)][f'{metric_name} (%)'] = percentage
            
            # Find dominant overhead component
            max_overhead = 0
            dominant_component = 'None'
            for metric_name in overhead_metrics:
                if metric_name in metrics and metrics[metric_name] > max_overhead:
                    max_overhead = metrics[metric_name]
                    dominant_component = metric_name.replace('Total ', '').replace(' time', '')
            
            averages[(size, mode, buffer_intrinsics)]['Dominant overhead component'] = dominant_component
    
    return averages


def generate_csv(averages, output_file='graph_timing_results.csv'):
    """
    Generate CSV file from averaged timing data.
    
    Args:
        averages: dict from compute_averages
        output_file: path to output CSV file
    """
    # Define column ordering
    base_columns = ['Matrix_Size', 'Graph_Mode', 'LATRD_USE_BUFFER_INTRINSICS', 'GPU time (benchmark)']
    
    timing_columns = [
        'Total capture time',
        'Total instantiation time', 
        'Total launch time',
        'TOTAL OVERHEAD',
        'GRAPH EXECUTION TIME'
    ]
    
    percentage_columns = [
        'Total capture time (%)',
        'Total instantiation time (%)',
        'Total launch time (%)'
    ]
    
    meta_columns = ['Dominant overhead component']
    
    # Sort keys by size then mode then buffer_intrinsics for consistent row ordering
    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1], x[2]))
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = base_columns + timing_columns + percentage_columns + meta_columns
        writer.writerow(header)
        
        # Write data rows
        for (size, mode, buffer_intrinsics) in sorted_keys:
            metrics = averages[(size, mode, buffer_intrinsics)]
            row = [size, mode, buffer_intrinsics]
            
            # Add all columns in order
            all_data_columns = (base_columns[3:] + timing_columns + 
                               percentage_columns + meta_columns)
            
            for col_name in all_data_columns:
                value = metrics.get(col_name, '')
                if value != '':
                    if isinstance(value, str):
                        row.append(value)
                    elif '(%)' in col_name:
                        row.append(f'{value:.2f}')
                    else:
                        row.append(f'{value:.2f}')
                else:
                    row.append('')
            
            writer.writerow(row)
    
    print(f"CSV file generated: {output_file}")


def print_summary(averages):
    """Print a summary of the parsed data."""
    print("\n" + "="*80)
    print("TIMING RESULTS SUMMARY")
    print("="*80)
    
    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1], x[2]))
    
    for (size, mode, buffer_intrinsics) in sorted_keys:
        print(f"\nMatrix Size: {size}, Graph Mode: {mode}, Buffer Intrinsics: {buffer_intrinsics}")
        print("-" * 80)
        
        metrics = averages[(size, mode, buffer_intrinsics)]
        
        # Print GPU time first
        if 'GPU time (benchmark)' in metrics:
            print(f"  {'GPU time (benchmark)':45s}: {metrics['GPU time (benchmark)']:10.2f} us")
            print()
        
        # Print timing metrics
        timing_keys = [k for k in metrics.keys() 
                      if k not in ['GPU time (benchmark)', 'Dominant overhead component'] 
                      and not k.endswith('(%)')]
        for metric_name in sorted(timing_keys):
            value = metrics[metric_name]
            print(f"  {metric_name:45s}: {value:10.2f} us")
        
        # Print percentages
        print()
        percent_keys = [k for k in metrics.keys() if k.endswith('(%)')]
        if percent_keys:
            for metric_name in sorted(percent_keys):
                value = metrics[metric_name]
                print(f"  {metric_name:45s}: {value:10.2f} %")
        
        # Print dominant component
        if 'Dominant overhead component' in metrics:
            print()
            print(f"  {'Dominant overhead component':45s}: {metrics['Dominant overhead component']}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse LATRD timing results and generate CSV with averaged metrics.',
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
    
    parser.add_argument('-i', '--input', dest='input_file', 
                       help='Input file containing timing results')
    parser.add_argument('-o', '--output', dest='output_file', 
                       default='graph_timing_results.csv',
                       help='Output CSV file (default: graph_timing_results.csv)')
    parser.add_argument('input_positional', nargs='?',
                       help='Input file (positional argument for backward compatibility)')
    
    args = parser.parse_args()
    
    # Determine input file (prioritize -i flag, then positional arg)
    input_file = args.input_file or args.input_positional
    
    if not input_file:
        parser.print_help()
        sys.exit(1)
    
    output_file = args.output_file
    
    try:
        print(f"Parsing results from: {input_file}")
        
        # Parse the file
        data = parse_results_file(input_file)
        
        if not data:
            print("ERROR: No timing data found in the input file.")
            print("\nExpected format:")
            print("  [TEST_CONFIG] LATRD_GRAPH_MODE=4, LATRD_USE_BUFFER_INTRINSICS=1, SIZE=1024")
            print("  [LATRD_GRAPH TIMING] n=1024, k=32, GRAPH_MODE=4")
            print("  [LATRD_GRAPH TIMING] Total capture time: 1234.56 us")
            print("  [LATRD_NON-GRAPH TIMING] Total execution time: 5678.90 us")
            print("  cpu_time_us     gpu_time_us")
            print("  395068          19394")
            sys.exit(1)
        
        print(f"Found {len(data)} unique size/mode/buffer_intrinsics combinations")
        
        # Compute averages
        averages = compute_averages(data)
        
        # Compute percentages and identify dominant component
        averages = compute_percentages_and_dominant(averages)
        
        # Print summary to console
        print_summary(averages)
        
        # Generate CSV
        generate_csv(averages, output_file)
        
        print("\n" + "="*80)
        print("Processing complete!")
        print("="*80)
        
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
