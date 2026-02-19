#!/usr/bin/env python3
"""
Script to analyze Tensile YAML files and find solution indices by parameter combinations
"""

import yaml
import argparse
from typing import Dict, List, Any, Optional


def load_yaml_file(file_path: str) -> Any:
    """Load YAML file"""
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def extract_solutions(data: Any) -> List[Dict]:
    """Extract solutions from the YAML data"""
    # The structure is: [header, arch, gfx, devices, problem_type, [solutions...]]
    # Solutions are typically in index 5 (0-indexed) as a list
    if isinstance(data, list) and len(data) > 5:
        solutions = data[5]
        if isinstance(solutions, list):
            return solutions
    return []


def analyze_solutions(solutions: List[Dict]) -> None:
    """Print statistics about solutions"""
    print(f"\nTotal solutions found: {len(solutions)}")
    
    if not solutions:
        return
    
    # Collect unique values for key parameters
    mt0_values = set()
    mt1_values = set()
    depth_u_values = set()
    wg_values = set()
    mi_values = set()
    gsu_values = set()
    plr_values = set()
    li_values = set()
    lrvw_values = set()
    
    for sol in solutions:
        if 'MacroTile0' in sol:
            mt0_values.add(sol['MacroTile0'])
        if 'MacroTile1' in sol:
            mt1_values.add(sol['MacroTile1'])
        if 'DepthU' in sol:
            depth_u_values.add(sol['DepthU'])
        if 'WorkGroup' in sol:
            wg_values.add(tuple(sol['WorkGroup']))
        if 'MatrixInstruction' in sol:
            mi_values.add(tuple(sol['MatrixInstruction']))
        if 'GlobalSplitU' in sol:
            gsu_values.add(sol['GlobalSplitU'])
        if 'PrefetchLocalRead' in sol:
            plr_values.add(sol['PrefetchLocalRead'])
        if 'LoopIters' in sol:
            li_values.add(sol['LoopIters'])
        if 'LocalReadVectorWidth' in sol:
            lrvw_values.add(sol['LocalReadVectorWidth'])
    
    print(f"\nUnique MacroTile0 values: {sorted(mt0_values)}")
    print(f"Unique MacroTile1 values: {sorted(mt1_values)}")
    print(f"Unique DepthU values: {sorted(depth_u_values)}")
    print(f"Unique GlobalSplitU values: {sorted(gsu_values)}")
    print(f"Unique PrefetchLocalRead values: {sorted(plr_values)}")
    print(f"Unique LoopIters values: {sorted(li_values)}")
    print(f"Unique LocalReadVectorWidth values: {sorted(lrvw_values)}")
    print(f"\nUnique WorkGroup combinations: {len(wg_values)}")
    print(f"Unique MatrixInstruction combinations: {len(mi_values)}")


def find_solutions(
    solutions: List[Dict],
    macro_tile_0: Optional[int] = None,
    macro_tile_1: Optional[int] = None,
    depth_u: Optional[int] = None,
    global_split_u: Optional[int] = None,
    work_group: Optional[List[int]] = None,
    matrix_instruction: Optional[List[int]] = None,
    prefetch_local_read: Optional[int] = None,
    loop_iters: Optional[int] = None,
    local_read_vector_width: Optional[int] = None,
    direct_to_vgpr_a: Optional[int] = None,
    **kwargs
) -> List[Dict]:
    """
    Find solutions matching the given parameters
    
    Args:
        macro_tile_0: MacroTile0 value
        macro_tile_1: MacroTile1 value
        depth_u: DepthU value
        global_split_u: GlobalSplitU value
        work_group: WorkGroup as [x, y, z]
        matrix_instruction: MatrixInstruction as [M, N, K, B]
        prefetch_local_read: PrefetchLocalRead value
        loop_iters: LoopIters value
        local_read_vector_width: LocalReadVectorWidth value
        direct_to_vgpr_a: DirectToVgprA value
        **kwargs: Any other parameter to filter by
    """
    matching = []
    
    for sol in solutions:
        match = True
        
        if macro_tile_0 is not None and sol.get('MacroTile0') != macro_tile_0:
            match = False
        if macro_tile_1 is not None and sol.get('MacroTile1') != macro_tile_1:
            match = False
        if depth_u is not None and sol.get('DepthU') != depth_u:
            match = False
        if global_split_u is not None and sol.get('GlobalSplitU') != global_split_u:
            match = False
        if work_group is not None and sol.get('WorkGroup') != work_group:
            match = False
        if matrix_instruction is not None and sol.get('MatrixInstruction') != matrix_instruction:
            match = False
        if prefetch_local_read is not None and sol.get('PrefetchLocalRead') != prefetch_local_read:
            match = False
        
        # Calculate LoopIters using the formula instead of direct comparison
        if loop_iters is not None:
            # Extract necessary values from solution
            depth_u_val = sol.get('DepthU', 0)
            local_split_u = sol.get('LocalSplitU', 1)
            inner_unroll = sol.get('InnerUnroll', 1)
            matrix_inst = sol.get('MatrixInstruction', [])
            
            # MatrixInstK is the 3rd element (index 2) of MatrixInstruction
            matrix_inst_k = matrix_inst[2] if len(matrix_inst) > 2 else 1
            
            # Calculate LoopUnroll
            if local_split_u > 0 and inner_unroll > 0:
                loop_unroll = depth_u_val // local_split_u // inner_unroll
            else:
                loop_unroll = 0
            
            # Calculate LoopIters
            if matrix_inst_k > 0:
                calculated_loop_iters = loop_unroll // matrix_inst_k
            else:
                calculated_loop_iters = loop_unroll
            
            # Compare calculated LoopIters with the input parameter
            if calculated_loop_iters != loop_iters:
                match = False
        
        if local_read_vector_width is not None and sol.get('LocalReadVectorWidth') != local_read_vector_width:
            match = False
        
        if direct_to_vgpr_a is not None:
            # Convert boolean to int if needed
            sol_dtva = sol.get('DirectToVgprA', False)
            if isinstance(sol_dtva, bool):
                sol_dtva = 1 if sol_dtva else 0
            if sol_dtva != direct_to_vgpr_a:
                match = False
        
        # Check any additional kwargs
        for key, value in kwargs.items():
            if sol.get(key) != value:
                match = False
                break
        
        if match:
            matching.append(sol)
    
    return matching


def print_solutions(solutions: List[Dict], max_print: int = 20, index_only: bool = False, filter_params: dict = None) -> None:
    """Print solution information"""
    if not solutions:
        print("No matching solutions found.")
        return
    
    if index_only:
        # Print summary with parameter values
        if filter_params:
            parts = []
            if filter_params.get('plr') is not None:
                parts.append(f"PrefetchLocalRead={filter_params['plr']}")
            if filter_params.get('li') is not None:
                parts.append(f"LoopIters={filter_params['li']}")
            if filter_params.get('lrvw') is not None:
                parts.append(f"LocalReadVectorWidth={filter_params['lrvw']}")
            
            if parts:
                param_str = ", ".join(parts)
                print(f"With {param_str}, we found {len(solutions)} matching solution indices:")
        
        # Print only solution indices
        indices = [sol.get('SolutionIndex', 'N/A') for sol in solutions]
        print(" ".join(map(str, indices)))
        return
    
    print(f"\nFound {len(solutions)} matching solution(s):")
    print("=" * 100)
    
    for i, sol in enumerate(solutions[:max_print]):
        idx = sol.get('SolutionIndex', 'N/A')
        mt0 = sol.get('MacroTile0', 'N/A')
        mt1 = sol.get('MacroTile1', 'N/A')
        du = sol.get('DepthU', 'N/A')
        wg = sol.get('WorkGroup', 'N/A')
        gsu = sol.get('GlobalSplitU', 'N/A')
        mi = sol.get('MatrixInstruction', 'N/A')
        plr = sol.get('PrefetchLocalRead', 'N/A')
        li = sol.get('LoopIters', 'N/A')
        lrvw = sol.get('LocalReadVectorWidth', 'N/A')
        name = sol.get('SolutionNameMin', 'N/A')
        
        print(f"\nSolution #{i+1}:")
        print(f"  SolutionIndex: {idx}")
        print(f"  MacroTile: {mt0}x{mt1}")
        print(f"  DepthU: {du}")
        print(f"  WorkGroup: {wg}")
        print(f"  GlobalSplitU: {gsu}")
        print(f"  MatrixInstruction: {mi}")
        print(f"  PrefetchLocalRead: {plr}")
        print(f"  LoopIters: {li}")
        print(f"  LocalReadVectorWidth: {lrvw}")
        print(f"  Name: {name}")
    
    if len(solutions) > max_print:
        print(f"\n... and {len(solutions) - max_print} more solutions (use --all to see all)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Tensile YAML files and find solutions by parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all solutions statistics
  python3 analyze_solutions.py -y file.yaml
  
  # Find solutions with MacroTile 128x128 and DepthU 8
  python3 analyze_solutions.py -y file.yaml --mt0 128 --mt1 128 --du 8
  
  # Find solutions with specific WorkGroup
  python3 analyze_solutions.py -y file.yaml --wg 128 4 1
  
  # Find solutions with GlobalSplitU = 1
  python3 analyze_solutions.py -y file.yaml --gsu 1
        """
    )
    
    parser.add_argument('-y', '--yaml', required=True, help='Path to Tensile YAML file')
    parser.add_argument('--mt0', type=int, help='MacroTile0 value')
    parser.add_argument('--mt1', type=int, help='MacroTile1 value')
    parser.add_argument('--du', type=int, help='DepthU value')
    parser.add_argument('--gsu', type=int, help='GlobalSplitU value')
    parser.add_argument('--wg', nargs=3, type=int, metavar=('X', 'Y', 'Z'), 
                        help='WorkGroup as three integers (e.g., 128 4 1)')
    parser.add_argument('--mi', nargs=4, type=int, metavar=('M', 'N', 'K', 'B'),
                        help='MatrixInstruction as four integers (e.g., 32 32 2 1)')
    parser.add_argument('--plr', type=int, help='PrefetchLocalRead value')
    parser.add_argument('--li', type=int, help='LoopIters value')
    parser.add_argument('--lrvw', type=int, help='LocalReadVectorWidth value')
    parser.add_argument('--dtvA', type=int, choices=[0, 1], help='DirectToVgprA value (0 or 1)')
    parser.add_argument('--all', action='store_true', help='Print all matching solutions')
    parser.add_argument('-d', '--index-only', action='store_true', help='Print only solution indices')
    parser.add_argument('--stats-only', action='store_true', help='Only show statistics')
    parser.add_argument('--param', nargs=2, action='append', 
                        metavar=('KEY', 'VALUE'),
                        help='Additional parameter to filter (can be used multiple times)')
    
    args = parser.parse_args()
    
    # Load YAML
    data = load_yaml_file(args.yaml)
    solutions = extract_solutions(data)
    
    # Show statistics
    analyze_solutions(solutions)
    
    if args.stats_only:
        return
    
    # If any filters are specified, find matching solutions
    has_filters = any([
        args.mt0 is not None,
        args.mt1 is not None,
        args.du is not None,
        args.gsu is not None,
        args.wg is not None,
        args.mi is not None,
        args.plr is not None,
        args.li is not None,
        args.lrvw is not None,
        args.dtvA is not None,
        args.param is not None
    ])
    
    if has_filters:
        # Prepare additional parameters
        extra_params = {}
        if args.param:
            for key, value in args.param:
                # Try to convert to int
                try:
                    extra_params[key] = int(value)
                except ValueError:
                    extra_params[key] = value
        
        matching = find_solutions(
            solutions,
            macro_tile_0=args.mt0,
            macro_tile_1=args.mt1,
            depth_u=args.du,
            global_split_u=args.gsu,
            work_group=args.wg,
            matrix_instruction=args.mi,
            prefetch_local_read=args.plr,
            loop_iters=args.li,
            local_read_vector_width=args.lrvw,
            direct_to_vgpr_a=args.dtvA,
            **extra_params
        )
        
        # Prepare filter params for display
        filter_params = {
            'plr': args.plr,
            'li': args.li,
            'lrvw': args.lrvw
        }
        
        max_print = len(matching) if args.all else 20
        print_solutions(matching, max_print=max_print, index_only=args.index_only, filter_params=filter_params)


if __name__ == '__main__':
    main()
