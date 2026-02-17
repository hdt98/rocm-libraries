#!/usr/bin/env python3

"""
Compare performance results from two profiler runs.

Usage:
    python compare_results.py results-baseline.txt results-improved-kernels.txt --output comparison.png
    python compare_results.py results-baseline.txt results-improved-kernels.txt --bin-width 5 --output comparison.png
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_results_file(filepath):
    """Parse a results file and extract command -> tflops mapping.
    
    Args:
        filepath: Path to the results file
        
    Returns:
        Dictionary mapping input commands to tflops values.
        Commands with errors are excluded.
    """
    results = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by separator lines
    blocks = re.split(r'={70,}', content)
    
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        
        # Look for input command
        if block.startswith('Input command:'):
            command = block.replace('Input command:', '').strip()
            
            # Get the next block which should contain the results
            if i + 1 < len(blocks):
                result_block = blocks[i + 1]
                
                # Check for errors
                if 'Error: Incorrect results!' in result_block:
                    i += 1
                    continue
                
                # Extract tflops
                tflops_match = re.search(r'tflops:\s*([\d.]+)', result_block)
                if tflops_match:
                    tflops = float(tflops_match.group(1))
                    results[command] = tflops
        
        i += 1
    
    return results


def bin_value(value, bin_width):
    """Bin a value to the nearest bin center.
    
    Args:
        value: The value to bin
        bin_width: Width of each bin
        
    Returns:
        Binned value (center of the bin)
    """
    if bin_width <= 0:
        return value
    return round(value / bin_width) * bin_width


def create_short_label(command, max_length=50):
    """Create a shortened label from a command.
    
    Args:
        command: Full MIOpen driver command
        max_length: Maximum label length
        
    Returns:
        Shortened label string
    """
    # Extract key parameters
    params = {}
    
    # Parse common parameters
    patterns = [
        (r'-n\s+(\d+)', 'n'),
        (r'-c\s+(\d+)', 'c'),
        (r'-H\s+(\d+)', 'H'),
        (r'-W\s+(\d+)', 'W'),
        (r'-k\s+(\d+)', 'k'),
        (r'-y\s+(\d+)', 'y'),
        (r'-x\s+(\d+)', 'x'),
        (r'--in_d\s+(\d+)', 'D'),
        (r'--fil_d\s+(\d+)', 'fD'),
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, command)
        if match:
            params[key] = match.group(1)
    
    # Build label
    parts = []
    if 'n' in params:
        parts.append(f"n{params['n']}")
    if 'c' in params:
        parts.append(f"c{params['c']}")
    if 'D' in params:
        parts.append(f"D{params['D']}")
    if 'H' in params and 'W' in params:
        parts.append(f"{params['H']}x{params['W']}")
    if 'k' in params:
        parts.append(f"k{params['k']}")
    if 'fD' in params:
        parts.append(f"f{params['fD']}")
    if 'y' in params and 'x' in params:
        parts.append(f"{params['y']}x{params['x']}")
    
    label = '_'.join(parts)
    
    if len(label) > max_length:
        label = label[:max_length-3] + '...'
    
    return label if label else command[:max_length]


def plot_comparison(baseline_results, improved_results, bin_width=0, output_file=None):
    """Create a comparison bar chart.
    
    Args:
        baseline_results: Dictionary of baseline command -> tflops
        improved_results: Dictionary of improved command -> tflops
        bin_width: Width for binning tflops values (0 = no binning)
        output_file: Output file path (None = show plot)
    """
    # Find common commands
    common_commands = set(baseline_results.keys()) & set(improved_results.keys())
    
    if not common_commands:
        print("No common commands found between the two result files.")
        return
    
    # Sort commands for consistent ordering
    commands = sorted(common_commands)
    
    # Extract and optionally bin values
    baseline_values = []
    improved_values = []
    labels = []
    
    for cmd in commands:
        baseline_val = baseline_results[cmd]
        improved_val = improved_results[cmd]
        
        if bin_width > 0:
            baseline_val = bin_value(baseline_val, bin_width)
            improved_val = bin_value(improved_val, bin_width)
        
        baseline_values.append(baseline_val)
        improved_values.append(improved_val)
        labels.append(create_short_label(cmd))
    
    # Calculate speedup
    speedups = [imp / base if base > 0 else 0 for base, imp in zip(baseline_values, improved_values)]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Bar chart comparison
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue')
    bars2 = ax1.bar(x + width/2, improved_values, width, label='Double buf added', color='darkorange')
    
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('TFLOPS')
    ax1.set_title('Baseline vs Double buffered kernels added')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Speedup chart
    colors = ['green' if s >= 1.0 else 'red' for s in speedups]
    bars3 = ax2.bar(x, speedups, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No change')
    
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Speedup (Improved / Baseline)')
    ax2.set_title('Speedup per Test Case')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add speedup values on bars
    for i, (bar, speedup) in enumerate(zip(bars3, speedups)):
        height = bar.get_height()
        ax2.annotate(f'{speedup:.2f}x',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total test cases compared: {len(commands)}")
    print(f"  Average speedup: {np.mean(speedups):.2f}x")
    print(f"  Median speedup: {np.median(speedups):.2f}x")
    print(f"  Min speedup: {min(speedups):.2f}x")
    print(f"  Max speedup: {max(speedups):.2f}x")
    print(f"  Cases improved (>1.0x): {sum(1 for s in speedups if s > 1.0)}")
    print(f"  Cases regressed (<1.0x): {sum(1 for s in speedups if s < 1.0)}")
    
    if bin_width > 0:
        print(f"  Bin width: {bin_width} TFLOPS")
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare performance results from two profiler runs.'
    )
    parser.add_argument('baseline', help='Path to baseline results file')
    parser.add_argument('improved', help='Path to improved results file')
    parser.add_argument('--bin-width', type=float, default=0,
                        help='Bin width for TFLOPS values (0 = no binning)')
    parser.add_argument('--output', '-o', help='Output file for the plot (e.g., comparison.png)')
    
    args = parser.parse_args()
    
    print(f"Parsing baseline results from: {args.baseline}")
    baseline_results = parse_results_file(args.baseline)
    print(f"  Found {len(baseline_results)} valid results")
    
    print(f"Parsing improved results from: {args.improved}")
    improved_results = parse_results_file(args.improved)
    print(f"  Found {len(improved_results)} valid results")
    
    plot_comparison(baseline_results, improved_results, args.bin_width, args.output)


if __name__ == "__main__":
    main()