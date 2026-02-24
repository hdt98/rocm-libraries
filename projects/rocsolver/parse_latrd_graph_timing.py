#!/usr/bin/env python3
"""
Parse LATRD timing results and generate CSV with averaged metrics.

Usage:
    ./parse_latrd_graph_timing.py <results_file.txt>
    
Output:
    graph_timing_results.csv - CSV file with averaged timing metrics
"""

import sys
import re
from collections import defaultdict
from statistics import mean
import csv


def parse_timing_line(line):
    """Parse a timing line and extract metric name and value."""
    # Match patterns like:
    # [LATRD_GRAPH TIMING] n=1024, k=32, GRAPH_MODE=4
    # [LATRD_GRAPH TIMING] Total capture time: 1234.56 us
    # [LATRD_NON-GRAPH TIMING] Total execution time: 5678.90 us
    # Benchmark table: cpu_time_us     gpu_time_us
    #                  395068          19394
    
    # First check if this is a configuration line
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
        dict: {(size, mode): {metric_name: [values]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    current_size = None
    current_mode = None
    expecting_gpu_time = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip grep separator lines
            if line == '--':
                continue
            
            line_type, parsed = parse_timing_line(line)
            
            if line_type == 'config':
                current_size = parsed['n']
                current_mode = parsed['mode']
                expecting_gpu_time = False
            elif line_type == 'metric' and current_size is not None and current_mode is not None:
                metric_name = parsed['name']
                value = parsed['value']
                data[(current_size, current_mode)][metric_name].append(value)
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
                        data[(current_size, current_mode)]['GPU time (benchmark)'].append(gpu_time)
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
        dict: {(size, mode): {metric_name: avg_value}}
    """
    averages = {}
    
    for (size, mode), metrics in data.items():
        averages[(size, mode)] = {}
        for metric_name, values in metrics.items():
            if values:
                averages[(size, mode)][metric_name] = mean(values)
    
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
    
    for (size, mode), metrics in averages.items():
        gpu_time = metrics.get('GPU time (benchmark)', 0)
        
        if gpu_time > 0:
            # Calculate percentages for each overhead component
            for metric_name in overhead_metrics:
                if metric_name in metrics:
                    percentage = (metrics[metric_name] / gpu_time) * 100.0
                    averages[(size, mode)][f'{metric_name} (%)'] = percentage
            
            # Find dominant overhead component
            max_overhead = 0
            dominant_component = 'None'
            for metric_name in overhead_metrics:
                if metric_name in metrics and metrics[metric_name] > max_overhead:
                    max_overhead = metrics[metric_name]
                    dominant_component = metric_name.replace('Total ', '').replace(' time', '')
            
            averages[(size, mode)]['Dominant overhead component'] = dominant_component
    
    return averages


def generate_csv(averages, output_file='graph_timing_results.csv'):
    """
    Generate CSV file from averaged timing data.
    
    Args:
        averages: dict from compute_averages
        output_file: path to output CSV file
    """
    # Define column ordering
    base_columns = ['Matrix_Size', 'Graph_Mode', 'GPU time (benchmark)']
    
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
    
    # Sort keys by size then mode for consistent row ordering
    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1]))
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = base_columns + timing_columns + percentage_columns + meta_columns
        writer.writerow(header)
        
        # Write data rows
        for (size, mode) in sorted_keys:
            metrics = averages[(size, mode)]
            row = [size, mode]
            
            # Add all columns in order
            all_data_columns = (base_columns[2:] + timing_columns + 
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
    
    sorted_keys = sorted(averages.keys(), key=lambda x: (x[0], x[1]))
    
    for (size, mode) in sorted_keys:
        print(f"\nMatrix Size: {size}, Graph Mode: {mode}")
        print("-" * 80)
        
        metrics = averages[(size, mode)]
        
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
    if len(sys.argv) != 2:
        print("Usage: ./parse_latrd_graph_timing.py <results_file.txt>")
        print("\nExample:")
        print("  ./parse_latrd_graph_timing.py test_results.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        print(f"Parsing results from: {input_file}")
        
        # Parse the file
        data = parse_results_file(input_file)
        
        if not data:
            print("ERROR: No timing data found in the input file.")
            print("\nExpected format:")
            print("  [LATRD_GRAPH TIMING] n=1024, k=32, GRAPH_MODE=4")
            print("  [LATRD_GRAPH TIMING] Total capture time: 1234.56 us")
            print("  [LATRD_GRAPH TIMING] GRAPH EXECUTION TIME: 18000.00 us")
            print("  cpu_time_us     gpu_time_us")
            print("  395068          19394")
            sys.exit(1)
        
        print(f"Found {len(data)} unique size/mode combinations")
        
        # Compute averages
        averages = compute_averages(data)
        
        # Compute percentages and identify dominant component
        averages = compute_percentages_and_dominant(averages)
        
        # Print summary to console
        print_summary(averages)
        
        # Generate CSV
        generate_csv(averages)
        
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
