#!/usr/bin/env python3
"""
Script to update PrefetchLocalRead and ClusterLocalRead values for specific solution indices in a Tensile YAML file.

Usage:
  python3 update_solution_params.py -y <yaml_file> -d <solution_indices> [--plr <value>] [--clr <value>]

Examples:
  # Update PLR for solution indices 0, 1, 2
  python3 update_solution_params.py -y input.yaml -d 0 1 2 --plr 8
  
  # Update both PLR and CLR for solution index 15
  python3 update_solution_params.py -y input.yaml -d 15 --plr 8 --clr 1
  
  # Update CLR only for multiple solutions
  python3 update_solution_params.py -y input.yaml -d 10 20 30 --clr 0
"""

import argparse
import re
import sys
from typing import List, Optional


def update_solution_params(
    yaml_file: str,
    solution_indices: List[int],
    plr: Optional[int] = None,
    clr: Optional[int] = None,
    output_file: Optional[str] = None
) -> None:
    """
    Update PrefetchLocalRead and/or ClusterLocalRead for specific solutions.
    
    Args:
        yaml_file: Path to input YAML file
        solution_indices: List of solution indices to update
        plr: New PrefetchLocalRead value (None = don't update)
        clr: New ClusterLocalRead value (None = don't update)
        output_file: Path to output file (None = overwrite input)
    """
    
    if plr is None and clr is None:
        print("Error: At least one of --plr or --clr must be specified")
        sys.exit(1)
    
    print(f"Loading {yaml_file}...")
    with open(yaml_file, 'r') as f:
        lines = f.readlines()
    
    solution_indices_set = set(solution_indices)
    current_solution_index = None
    in_solution = False
    solution_start_line = None
    updated_count = 0
    plr_updates = 0
    clr_updates = 0
    clr_insertions = 0
    
    # Track solutions that need CLR insertion
    solutions_needing_clr = {}
    solution_ranges = {}  # Maps solution index to (start_line, end_line)
    
    # First pass: identify solution boundaries and update existing parameters
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the start of a solution
        # Pattern 1: "- - 1LDSBuffer:" (first solution)
        # Pattern 2: "  - 1LDSBuffer:" (subsequent solutions)
        if re.match(r'^(- -|  -) 1LDSBuffer:', line):
            # Save the previous solution range if we were in one
            if in_solution and current_solution_index is not None and current_solution_index in solution_indices_set:
                solution_ranges[current_solution_index] = (solution_start_line, i - 1)
            
            solution_start_line = i
            in_solution = True
            current_solution_index = None  # Will be set when we find SolutionIndex
            
            # Look ahead to find the SolutionIndex for this solution
            for j in range(i, min(len(lines), i + 300)):
                if 'SolutionIndex:' in lines[j]:
                    match = re.search(r'SolutionIndex:\s*(\d+)', lines[j])
                    if match:
                        current_solution_index = int(match.group(1))
                        if current_solution_index in solution_indices_set:
                            print(f"Found solution {current_solution_index} at line {solution_start_line+1}")
                        break
                        if current_solution_index in solution_indices_set:
                            print(f"Found solution {current_solution_index} at line {solution_start_line+1}")
                        break
        
        # If we're in a target solution, look for parameters to update
        if in_solution and current_solution_index in solution_indices_set:
            
            # Update PrefetchLocalRead
            if plr is not None:
                plr_match = re.match(r'^(\s*)PrefetchLocalRead:\s*\d+', lines[i])
                if plr_match:
                    indent = plr_match.group(1)
                    old_line = lines[i].strip()
                    lines[i] = f'{indent}PrefetchLocalRead: {plr}\n'
                    print(f"  Line {i+1}: Updated PrefetchLocalRead: {old_line} -> PrefetchLocalRead: {plr}")
                    plr_updates += 1
            
            # Update ClusterLocalRead
            if clr is not None:
                clr_match = re.match(r'^(\s*)ClusterLocalRead:\s*\d+', lines[i])
                if clr_match:
                    indent = clr_match.group(1)
                    old_line = lines[i].strip()
                    lines[i] = f'{indent}ClusterLocalRead: {clr}\n'
                    print(f"  Line {i+1}: Updated ClusterLocalRead: {old_line} -> ClusterLocalRead: {clr}")
                    clr_updates += 1
        
        i += 1
    
    # Handle the last solution if it was a target
    if in_solution and current_solution_index in solution_indices_set:
        solution_ranges[current_solution_index] = (solution_start_line, len(lines) - 1)
    
    # Handle the last solution if it was a target
    if in_solution and current_solution_index in solution_indices_set:
        solution_ranges[current_solution_index] = (solution_start_line, len(lines) - 1)
    
    # Check which solutions need CLR inserted (if CLR was specified but not found)
    if clr is not None:
        for sol_idx in solution_indices_set:
            if sol_idx in solution_ranges:
                start_line, end_line = solution_ranges[sol_idx]
                
                # Check if this solution already has CLR
                has_clr = False
                
                # Search within the solution for ClusterLocalRead
                for j in range(start_line, end_line + 1):
                    if 'ClusterLocalRead:' in lines[j]:
                        has_clr = True
                        break
                
                if not has_clr:
                    solutions_needing_clr[sol_idx] = (start_line, end_line)
    
    # Second pass: insert ClusterLocalRead for solutions that don't have it
    if solutions_needing_clr:
        print(f"\nInserting ClusterLocalRead for {len(solutions_needing_clr)} solutions...")
        
        # Process in reverse order to maintain line numbers
        for sol_idx in sorted(solutions_needing_clr.keys(), reverse=True):
            start_line, end_line = solutions_needing_clr[sol_idx]
            
            # Find insertion point after BufferStore and before CodeObjectVersion
            insert_line = None
            indent = '    '
            buffer_store_line = None
            code_object_line = None
            
            # Look for BufferStore and CodeObjectVersion
            for j in range(start_line, end_line + 1):
                if 'BufferStore:' in lines[j]:
                    buffer_store_line = j
                    match = re.match(r'^(\s*)', lines[j])
                    if match:
                        indent = match.group(1)
                elif 'CodeObjectVersion:' in lines[j]:
                    code_object_line = j
                    if not indent or indent == '    ':  # Update indent if not set yet
                        match = re.match(r'^(\s*)', lines[j])
                        if match:
                            indent = match.group(1)
                    break
            
            # Determine insert position
            if buffer_store_line is not None and code_object_line is not None:
                # Insert after BufferStore
                insert_line = buffer_store_line + 1
            elif code_object_line is not None:
                # Insert before CodeObjectVersion
                insert_line = code_object_line
            elif buffer_store_line is not None:
                # Insert after BufferStore
                insert_line = buffer_store_line + 1
            else:
                # Insert after the first line of the solution (1LDSBuffer line)
                insert_line = start_line + 1
                match = re.match(r'^(\s*)', lines[start_line])
                if match:
                    # Get base indent from 1LDSBuffer line and add 4 spaces
                    base_indent = match.group(1)
                    if base_indent.startswith('- - ') or base_indent.startswith('  - '):
                        indent = '    '
                    else:
                        indent = base_indent + '    '
            
            # Insert the ClusterLocalRead line
            new_line = f'{indent}ClusterLocalRead: {clr}\n'
            lines.insert(insert_line, new_line)
            print(f"  Line {insert_line+1}: Inserted ClusterLocalRead: {clr} for solution {sol_idx}")
            clr_insertions += 1
    
    # Write output
    if output_file is None:
        output_file = yaml_file
    
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    # Summary
    total_updates = plr_updates + clr_updates + clr_insertions
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Target solutions: {sorted(solution_indices)}")
    if plr is not None:
        print(f"  PrefetchLocalRead updates: {plr_updates} (new value: {plr})")
    if clr is not None:
        print(f"  ClusterLocalRead updates: {clr_updates} (new value: {clr})")
        print(f"  ClusterLocalRead insertions: {clr_insertions} (new value: {clr})")
    print(f"  Total changes: {total_updates}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Update PrefetchLocalRead and ClusterLocalRead values for specific solutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update PLR for solution indices 0, 1, 2
  python3 update_solution_params.py -y input.yaml -d 0 1 2 --plr 8
  
  # Update both PLR and CLR for solution index 15
  python3 update_solution_params.py -y input.yaml -d 15 --plr 8 --clr 1
  
  # Update CLR only for multiple solutions
  python3 update_solution_params.py -y input.yaml -d 10 20 30 --clr 0
  
  # Save to a different file
  python3 update_solution_params.py -y input.yaml -d 5 --plr 16 -o output.yaml
        """
    )
    
    parser.add_argument('-y', '--yaml', required=True,
                        help='Path to Tensile YAML file')
    parser.add_argument('-d', '--solutions', nargs='+', type=int, required=True,
                        metavar='INDEX',
                        help='Solution indices to update (space-separated)')
    parser.add_argument('--plr', type=int,
                        help='New PrefetchLocalRead value')
    parser.add_argument('--clr', type=int,
                        help='New ClusterLocalRead value')
    parser.add_argument('-o', '--output',
                        help='Output file path (default: overwrite input file)')
    
    args = parser.parse_args()
    
    if args.plr is None and args.clr is None:
        parser.error("At least one of --plr or --clr must be specified")
    
    update_solution_params(
        yaml_file=args.yaml,
        solution_indices=args.solutions,
        plr=args.plr,
        clr=args.clr,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
