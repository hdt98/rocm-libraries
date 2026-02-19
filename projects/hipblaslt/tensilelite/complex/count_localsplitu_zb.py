#!/usr/bin/env python3
import os
import glob
import re

def count_localsplitu_in_file(file_path):
    """Count solutions with LocalSplitU > 1 in a YAML file."""
    count_greater_than_1 = 0
    total_solutions = 0
    
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Find all LocalSplitU values
        matches = re.findall(r'\bLocalSplitU:\s*(\d+)', content)
        
        for value_str in matches:
            value = int(value_str)
            if value > 1:
                count_greater_than_1 += 1
            total_solutions += 1
    
    return count_greater_than_1, total_solutions

def main():
    # Directory containing the converted files
    output_dir = '/data0/kamuruga/rocm-libraries/projects/hipblaslt/tensilelite/complex/ported_tensile'
    
    # Find all YAML files with ZB in the name
    all_yaml_files = glob.glob(os.path.join(output_dir, '*.yaml'))
    zb_files = sorted([f for f in all_yaml_files if '_ZB' in os.path.basename(f)])
    
    print("=" * 100)
    print(f"{'ZB File Name':<60} {'LocalSplitU>1':<15} {'Total Solutions':<15}")
    print("=" * 100)
    
    total_greater_than_1 = 0
    total_solutions_all_files = 0
    
    for yaml_file in zb_files:
        filename = os.path.basename(yaml_file)
        count_gt_1, total_sols = count_localsplitu_in_file(yaml_file)
        
        total_greater_than_1 += count_gt_1
        total_solutions_all_files += total_sols
        
        print(f"{filename:<60} {count_gt_1:<15} {total_sols:<15}")
    
    print("=" * 100)
    print(f"{'TOTAL (ZB files only)':<60} {total_greater_than_1:<15} {total_solutions_all_files:<15}")
    print("=" * 100)
    print(f"\nSummary for ZB files:")
    print(f"  - Total ZB files analyzed: {len(zb_files)}")
    print(f"  - Total solutions with LocalSplitU > 1: {total_greater_than_1}")
    print(f"  - Total solutions: {total_solutions_all_files}")
    if total_solutions_all_files > 0:
        percentage = (total_greater_than_1 / total_solutions_all_files) * 100
        print(f"  - Percentage with LocalSplitU > 1: {percentage:.2f}%")

if __name__ == "__main__":
    main()
