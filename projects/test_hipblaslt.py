#!/usr/bin/env python3
"""
Simplified script to test hipblaslt-bench with verification.
Only runs hipblaslt-bench (no rocblas comparison) for faster testing.

Usage:
    python3 test_hipblaslt.py <yaml_file>
    python3 test_hipblaslt.py /path/to/aquavanjaram942_Cijk_Ailk_BjlkC_CB.yaml
"""

import yaml
import sys
import subprocess
import re
import argparse
from pathlib import Path
from datetime import datetime


def parse_matrix_operations_from_filename(filename):
    """
    Parse matrix operation patterns from Tensile YAML filename.
    
    Returns:
        Dictionary with keys: transA, transB, lda_formula, ldb_formula
    """
    filename = Path(filename).name
    
    result = {
        'transA': 'N',
        'transB': 'N',
        'lda_formula': 'm',
        'ldb_formula': 'k'
    }
    
    match = re.search(r'Cijk_([A-Za-z]+)_([A-Za-z]+)', filename)
    
    if not match:
        print(f"⚠️  Warning: Could not parse matrix operations from filename: {filename}")
        print(f"   Using defaults: transA=N, transB=N, lda=m, ldb=k")
        return result
    
    a_pattern = match.group(1)
    b_pattern = match.group(2)
    
    # Parse A pattern
    if a_pattern == "AlikC":
        result['transA'] = 'C'
        result['lda_formula'] = 'k'
    elif a_pattern == "Alik":
        result['transA'] = 'T'
        result['lda_formula'] = 'k'
    elif a_pattern == "Ailk":
        result['transA'] = 'N'
        result['lda_formula'] = 'm'
    else:
        print(f"⚠️  Warning: Unknown A pattern '{a_pattern}', using defaults for A")
    
    # Parse B pattern
    if b_pattern == "Bljk":
        result['transB'] = 'N'
        result['ldb_formula'] = 'k'
    elif b_pattern == "Bjlk":
        result['transB'] = 'T'
        result['ldb_formula'] = 'n'
    elif b_pattern == "BjlkC":
        result['transB'] = 'C'
        result['ldb_formula'] = 'n'
    else:
        print(f"⚠️  Warning: Unknown B pattern '{b_pattern}', using defaults for B")
    
    return result


def extract_problem_sizes(yaml_file_path):
    """
    Extract problem sizes from the YAML file.
    
    Returns:
        List of tuples containing (m, n, k, solution_index) values
    """
    print(f"Reading YAML file: {yaml_file_path}")
    
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    problem_sizes = []
    
    for item in data:
        if isinstance(item, list):
            for entry in item:
                if isinstance(entry, list) and len(entry) == 2:
                    problem_spec = entry[0]
                    metrics = entry[1]
                    
                    if isinstance(problem_spec, list) and len(problem_spec) == 4:
                        m, n, batch, k = problem_spec
                        solution_index = metrics[0] if isinstance(metrics, list) and len(metrics) >= 1 else -1
                        problem_sizes.append((m, n, k, solution_index))
                    elif isinstance(problem_spec, list) and len(problem_spec) == 8:
                        m, n, batch, k = problem_spec[0], problem_spec[1], problem_spec[2], problem_spec[3]
                        solution_index = metrics[0] if isinstance(metrics, list) and len(metrics) >= 1 else -1
                        problem_sizes.append((m, n, k, solution_index))
    
    return problem_sizes


def run_hipblaslt_bench(m, n, k, bench_path, data_type, matrix_ops):
    """
    Run hipblaslt-bench for a specific problem size with verification.
    
    Returns:
        Dictionary with kernel_name, gflops, time_us, norm_error, etc., or None if failed
    """
    lda = eval(matrix_ops['lda_formula'], {'m': m, 'n': n, 'k': k})
    ldb = eval(matrix_ops['ldb_formula'], {'m': m, 'n': n, 'k': k})
    
    command = [
        bench_path, "-f", "matmul",
        "--transA", matrix_ops['transA'], "--transB", matrix_ops['transB'],
        "-m", str(m), "-n", str(n), "-k", str(k),
        "-r", data_type,
        "--lda", str(lda), "--ldb", str(ldb), "--ldc", str(m), "--ldd", str(m),
        "--alpha", "1.0", "--beta", "0.0",
        "--cold_iters", "10", "--iters", "10",
        "--initialization", "trig_float",
        "--a_type", data_type, "--b_type", data_type,
        "--c_type", data_type, "--d_type", data_type,
        "--compute_type", data_type,
        "--rotating", "512",
        "--print_kernel_info",
        "--verify"
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Parse kernel name
        kernel_name = "Unknown"
        kernel_match = re.search(r'--kernel name:\s+(.+?)(?:\n|$)', output)
        if kernel_match:
            kernel_name = kernel_match.group(1).strip()
        
        # Parse results from CSV line
        gflops = 0.0
        time_us = 0.0
        norm_error = 0.0
        cpu_gflops = 0.0
        cpu_time_us = 0.0
        
        trans_pattern = f"{matrix_ops['transA']},{matrix_ops['transB']},"
        csv_line = None
        
        for line in output.split('\n'):
            if line.strip().startswith(trans_pattern):
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    try:
                        # With --verify: ...,hipblaslt-Gflops,hipblaslt-GB/s,us,CPU-Gflops,CPU-us,norm_error,atol,rtol
                        gflops = float(parts[-8])
                        time_us = float(parts[-6])
                        cpu_gflops = float(parts[-5])
                        cpu_time_us = float(parts[-4])
                        norm_error = float(parts[-3])
                        csv_line = line.strip()
                        break
                    except (ValueError, IndexError):
                        pass
        
        return {
            'kernel_name': kernel_name,
            'gflops': gflops,
            'time_us': time_us,
            'norm_error': norm_error,
            'cpu_gflops': cpu_gflops,
            'cpu_time_us': cpu_time_us,
            'csv_line': csv_line
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout for problem size M={m}, N={n}, K={k}")
        return None
    except Exception as e:
        print(f"  ❌ Error running hipblaslt-bench for M={m}, N={n}, K={k}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test hipblaslt-bench with verification from Tensile YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: %(prog)s /path/to/aquavanjaram942_Cijk_Ailk_BjlkC_CB.yaml"
    )
    
    parser.add_argument(
        'yaml_file',
        type=str,
        help='Path to the Tensile YAML file'
    )
    
    parser.add_argument(
        '--hipblaslt-bench',
        type=str,
        default='./hipblaslt/build/release/clients/hipblaslt-bench',
        help='Path to hipblaslt-bench executable'
    )
    
    args = parser.parse_args()
    yaml_file = args.yaml_file
    
    if not Path(yaml_file).exists():
        print(f"❌ Error: File not found: {yaml_file}")
        sys.exit(1)
    
    # Extract problem sizes
    problem_sizes = extract_problem_sizes(yaml_file)
    print(f"\n✓ Found {len(problem_sizes)} problem sizes\n")
    
    # Determine data type
    filename = Path(yaml_file).name
    if "ZB" in filename:
        data_type = "f64_c"
        print("📌 Detected 'ZB' - using f64_c (double precision complex)\n")
    elif "CB" in filename:
        data_type = "f32_c"
        print("📌 Detected 'CB' - using f32_c (single precision complex)\n")
    else:
        data_type = "f64_c"
        print("📌 Using default data type: f64_c\n")
    
    # Parse matrix operations
    matrix_ops = parse_matrix_operations_from_filename(filename)
    print(f"📌 Matrix ops: transA={matrix_ops['transA']}, transB={matrix_ops['transB']}, lda={matrix_ops['lda_formula']}, ldb={matrix_ops['ldb_formula']}\n")
    
    # Check executable
    hipblaslt_bench_path = args.hipblaslt_bench
    if not Path(hipblaslt_bench_path).exists():
        print(f"❌ Error: hipblaslt-bench not found: {hipblaslt_bench_path}")
        sys.exit(1)
    
    print("="*80)
    print("Running hipblaslt-bench with --verify...")
    print("="*80)
    print()
    
    # Collect results
    results = []
    threshold = 0.000000000001 if data_type == "f64_c" else 0.00001
    
    for idx, (m, n, k, sol_idx) in enumerate(problem_sizes):
        print(f"[{idx+1}/{len(problem_sizes)}] Testing M={m}, N={n}, K={k} (Solution: {sol_idx})...")
        
        result = run_hipblaslt_bench(m, n, k, hipblaslt_bench_path, data_type, matrix_ops)
        
        if result:
            validation_status = "PASS" if result['norm_error'] < threshold else "FAIL"
            print(f"  ✓ {result['gflops']:.2f} Gflops | Validation: {validation_status} | norm_error: {result['norm_error']:.2e}")
            print(f"    Kernel: {result['kernel_name'][:80]}")
            if result.get('csv_line'):
                print(f"    CSV: {result['csv_line']}")
            
            results.append({
                'M': m,
                'N': n,
                'K': k,
                'Solution_Index': sol_idx,
                'Gflops': result['gflops'],
                'Time_us': result['time_us'],
                'Validation': validation_status,
                'norm_error': result['norm_error'],
                'CPU_Gflops': result['cpu_gflops'],
                'Kernel': result['kernel_name']
            })
        else:
            print(f"  ✗ FAILED")
            results.append({
                'M': m,
                'N': n,
                'K': k,
                'Solution_Index': sol_idx,
                'Gflops': 0.0,
                'Time_us': 0.0,
                'Validation': 'FAILED',
                'norm_error': -1.0,
                'CPU_Gflops': 0.0,
                'Kernel': 'FAILED'
            })
        print()
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'M':<8} {'N':<8} {'K':<8} {'Sol':<6} {'Gflops':<12} {'Time(us)':<12} {'Validation':<12} {'norm_error':<12}")
    print("-"*90)
    
    for result in results:
        norm_str = f"{result['norm_error']:.2e}" if result['norm_error'] >= 0 else "N/A"
        print(f"{result['M']:<8} {result['N']:<8} {result['K']:<8} {result['Solution_Index']:<6} "
              f"{result['Gflops']:<12.2f} {result['Time_us']:<12.2f} {result['Validation']:<12} {norm_str:<12}")
    
    # Count results
    passed = sum(1 for r in results if r['Validation'] == 'PASS')
    failed = sum(1 for r in results if r['Validation'] in ['FAIL', 'FAILED'])
    
    print("\n" + "="*80)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
