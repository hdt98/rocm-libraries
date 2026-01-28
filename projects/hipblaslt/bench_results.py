#!/usr/bin/env python3
"""
Script to extract problem sizes (m, n, k values) from a Tensile YAML file.
The problem sizes are located at the end of the YAML file in the format:
- - [m, n, batch, k]
  - [index, efficiency]

This script will:
1. Extract all problem sizes from the Equality YAML file
2. Run hipblaslt-bench for each problem size
3. Collect performance results
4. Export to Excel file with columns: M, N, K, Kernel Name, hipblaslt-Gflops
"""

import yaml
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime


def extract_problem_sizes(yaml_file_path):
    """
    Extract problem sizes from the YAML file.
    
    Args:
        yaml_file_path: Path to the YAML file
        
    Returns:
        List of tuples containing (m, n, k) values
    """
    print(f"Reading YAML file: {yaml_file_path}")
    
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # The YAML structure has problem sizes near the end
    # We need to find the section with problem size data
    # Looking at the structure, it's a list where one element contains
    # nested lists with problem sizes in format [[m, n, batch, k], [index, efficiency]]
    
    problem_sizes = []
    
    # Iterate through the data to find problem size entries
    for item in data:
        if isinstance(item, list):
            # Check if this is a list of problem size entries
            for entry in item:
                if isinstance(entry, list) and len(entry) == 2:
                    problem_spec = entry[0]
                    metrics = entry[1]
                    
                    # Problem spec should be [m, n, batch, k]
                    if isinstance(problem_spec, list) and len(problem_spec) == 4:
                        m, n, batch, k = problem_spec
                        problem_sizes.append((m, n, k))
    
    return problem_sizes


def run_hipblaslt_bench(m, n, k, bench_path="./build/release/clients/hipblaslt-bench"):
    """
    Run hipblaslt-bench for a specific problem size and extract results.
    
    Args:
        m: M dimension
        n: N dimension
        k: K dimension
        bench_path: Path to hipblaslt-bench executable
        
    Returns:
        Dictionary with kernel_name and gflops, or None if failed
    """
    command = [
        bench_path, "-f", "matmul",
        "--transA", "C", "--transB", "N",
        "-m", str(m), "-n", str(n), "-k", str(k),
        "--lda", str(k), "--ldb", str(k), "--ldc", str(m), "--ldd", str(m),
        "--alpha", "1.0", "--beta", "1.0",
        "--cold_iters", "10", "--iters", "10",  # Reduced for faster execution
        "--initialization", "trig_float", "--rotating", "512",
        "--flush", "--use_gpu_timer",
        "--a_type", "f64_c", "--b_type", "f64_c",
        "--c_type", "f64_c", "--d_type", "f64_c",
        "--compute_type", "f64_c",
        "--print_kernel_info"
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Parse kernel name from line: --kernel name:    <kernel_name>
        kernel_name = "Unknown"
        kernel_match = re.search(r'--kernel name:\s+(.+?)(?:\n|$)', output)
        if kernel_match:
            kernel_name = kernel_match.group(1).strip()
        
        # Parse Gflops from CSV data line
        # Format: ...,f64_c,f64_c,512,1,1,182.379,34.5016,22.9977
        # Gflops is the 3rd value from the last (before GB/s and us)
        gflops = 0.0
        
        # Find lines that contain the results (look for lines with transA,transB pattern)
        for line in output.split('\n'):
            # Look for data line that starts with transpose config
            if line.strip().startswith('C,N,') or line.strip().startswith('T,N,'):
                # Split by comma and get the last 3 values
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        # Last 3 values are: hipblaslt-Gflops, hipblaslt-GB/s, us
                        gflops = float(parts[-3])
                        break
                    except (ValueError, IndexError):
                        pass
        
        return {
            'kernel_name': kernel_name,
            'gflops': gflops
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout for problem size M={m}, N={n}, K={k}")
        return None
    except Exception as e:
        print(f"  ❌ Error running hipblaslt-bench for M={m}, N={n}, K={k}: {e}")
        return None


def export_to_excel(results, output_file="hipblaslt_bench_results.xlsx"):
    """
    Export results to Excel file.
    
    Args:
        results: List of dictionaries with M, N, K, Kernel_Name, Gflops
        output_file: Output Excel filename
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Results exported to: {output_file}")
        return True
        
    except ImportError:
        print("\n⚠️  pandas or openpyxl not installed. Exporting to CSV instead...")
        try:
            import csv
            csv_file = output_file.replace('.xlsx', '.csv')
            with open(csv_file, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            print(f"✅ Results exported to: {csv_file}")
            return True
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return False
    except Exception as e:
        print(f"❌ Error exporting to Excel: {e}")
        return False


def main():
    yaml_file = "/data0/kamuruga/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942/Equality/aquavanjaram942_Cijk_AlikC_Bljk_ZB.yaml"
    
    # Check if file exists
    if not Path(yaml_file).exists():
        print(f"Error: File not found: {yaml_file}")
        sys.exit(1)
    
    # Extract problem sizes
    problem_sizes = extract_problem_sizes(yaml_file)
    
    print(f"\nFound {len(problem_sizes)} problem sizes:\n")
    print(f"{'Index':<8} {'M':<10} {'N':<10} {'K':<10}")
    print("-" * 40)
    
    for idx, (m, n, k) in enumerate(problem_sizes):
        print(f"{idx:<8} {m:<10} {n:<10} {k:<10}")
    
    print("\n" + "="*70)
    print("Running hipblaslt-bench for all problem sizes...")
    print("="*70)
    print("\nThis may take several minutes. Please wait...\n")
    
    bench_path = "./build/release/clients/hipblaslt-bench"
    
    # Check if hipblaslt-bench executable exists
    if not Path(bench_path).exists():
        print(f"❌ Error: Hipblaslt-Bench executable not found: {bench_path}")
        print("Please build the project first.")
        sys.exit(1)
    
    # Collect results
    results = []
    
    for idx, (m, n, k) in enumerate(problem_sizes):
        print(f"[{idx+1}/{len(problem_sizes)}] Running hipblaslt-bench for M={m}, N={n}, K={k}...", end=" ")
        
        bench_result = run_hipblaslt_bench(m, n, k, bench_path)
        
        if bench_result:
            results.append({
                'M': m,
                'N': n,
                'K': k,
                'Kernel_Name': bench_result['kernel_name'],
                'hipblaslt_Gflops': bench_result['gflops']
            })
            print(f"✓ {bench_result['gflops']:.2f} Gflops")
        else:
            results.append({
                'M': m,
                'N': n,
                'K': k,
                'Kernel_Name': 'FAILED',
                'hipblaslt_Gflops': 0.0
            })
            print("✗ Failed")
    
    print("\n" + "="*70)
    print("Hipblaslt-Bench execution completed!")
    print("="*70)
    
    # Display summary table
    print("\n\nResults Summary:")
    print("="*100)
    print(f"{'M':<10} {'N':<10} {'K':<10} {'Gflops':<15} {'Kernel Name':<50}")
    print("-"*100)
    for result in results:
        kernel_short = result['Kernel_Name'][:47] + "..." if len(result['Kernel_Name']) > 50 else result['Kernel_Name']
        print(f"{result['M']:<10} {result['N']:<10} {result['K']:<10} {result['hipblaslt_Gflops']:<15.2f} {kernel_short:<50}")
    
    # Export to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"equality_hipblaslt_bench_results_{timestamp}.xlsx"
    
    print("\n" + "="*70)
    export_to_excel(results, output_file)
    print("="*70)
    
    # Print sample command
    print("\n\nSample command used:")
    print("-"*70)
    m, n, k = problem_sizes[0]
    sample_cmd = (
        f"{bench_path} -f matmul "
        f"--transA C --transB N "
        f"-m {m} -n {n} -k {k} "
        f"--lda {k} --ldb {k} --ldc {m} --ldd {m} "
        f"--alpha 1.0 --beta 1.0 "
        f"--cold_iters 10 --iters 10 "
        f"--initialization trig_float --rotating 512 "
        f"--flush --use_gpu_timer "
        f"--a_type f64_c --b_type f64_c --c_type f64_c --d_type f64_c "
        f"--compute_type f64_c --print_kernel_info"
    )
    print(sample_cmd)
    print("-"*70)
    
    print("\n✅ All done! Check the Excel file for complete results.")
    
    return results


if __name__ == "__main__":
    main()
