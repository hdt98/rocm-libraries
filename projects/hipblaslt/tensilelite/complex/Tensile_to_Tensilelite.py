import os
import glob
import re

def process_solution(solution_lines):
    """Process a single solution to validate and fix LoopIters and PrefetchLocalRead."""
    loop_iters_value = None
    prefetch_local_read_value = None
    prefetch_local_read_line_idx = None
    
    # Find LoopIters and PrefetchLocalRead values
    for i, line in enumerate(solution_lines):
        if 'LoopIters:' in line:
            match = re.search(r'LoopIters:\s*(\d+)', line)
            if match:
                loop_iters_value = int(match.group(1))
        if 'PrefetchLocalRead:' in line:
            match = re.search(r'PrefetchLocalRead:\s*(\d+)', line)
            if match:
                prefetch_local_read_value = int(match.group(1))
                prefetch_local_read_line_idx = i
    
    corrected = False
    # Check condition: LoopIters - PrefetchLocalRead < 0 must NOT be satisfied
    # If LoopIters - PrefetchLocalRead < 0, then PrefetchLocalRead > LoopIters
    # Fix by decrementing PrefetchLocalRead to LoopIters - 1
    if loop_iters_value is not None and prefetch_local_read_value is not None:
        if loop_iters_value - prefetch_local_read_value < 0:
            # Fix: PrefetchLocalRead = LoopIters - 1 (to satisfy the original condition)
            new_prefetch_local_read = loop_iters_value - 1
            solution_lines[prefetch_local_read_line_idx] = re.sub(
                r'PrefetchLocalRead:\s*\d+',
                f'PrefetchLocalRead: {new_prefetch_local_read}',
                solution_lines[prefetch_local_read_line_idx]
            )
            corrected = True
    
    return {'lines': solution_lines, 'corrected': corrected}

def convert_liblogic_tensile_to_tensilelite(first_words_in_tensilelite_lib_logic, tensile_path, output_file_path, removed_param_path, filename):
    try:
        # Determine ActivationComputeDataType based on filename
        if 'CB' in filename:
            activation_compute_data_type = 2
        elif 'ZB' in filename:
            activation_compute_data_type = 3
        else:
            activation_compute_data_type = None  # Don't add if neither CB nor ZB
        
        # create set to ignore duplicate removed params
        seen_removed_word = set()
        
        # Read all lines first to process solutions
        with open(tensile_path, 'r') as tensile_lib_logic:
            all_lines = tensile_lib_logic.readlines()
        
        # Process lines and track solution boundaries
        processed_lines = []
        current_solution = []
        in_solution = False
        prefetch_corrections = 0
        device_list_found = False
        
        for i, line in enumerate(all_lines):
            original_line = line
            line_stripped = line.strip()
            
            # Detect device list line to insert ActivationComputeDataType after it
            if line_stripped.startswith('- [Device') and activation_compute_data_type is not None:
                processed_lines.append(original_line)
                device_list_found = True
                continue
            
            # Insert ActivationComputeDataType after device list
            if device_list_found and line_stripped.startswith('- ') and ':' in line_stripped:
                processed_lines.append(f"- ActivationComputeDataType: {activation_compute_data_type}\n")
                device_list_found = False  # Only insert once
            
            # Detect start of a solution
            if line_stripped.startswith('- ') and '1LDSBuffer:' in line_stripped:
                # Process previous solution if exists
                if current_solution:
                    corrected_solution = process_solution(current_solution)
                    if corrected_solution['corrected']:
                        prefetch_corrections += 1
                    processed_lines.extend(corrected_solution['lines'])
                # Start new solution
                current_solution = [original_line]
                in_solution = True
            elif in_solution:
                current_solution.append(original_line)
            else:
                processed_lines.append(original_line)
        
        # Process last solution
        if current_solution:
            corrected_solution = process_solution(current_solution)
            if corrected_solution['corrected']:
                prefetch_corrections += 1
            processed_lines.extend(corrected_solution['lines'])
        
        # Now filter and write the processed lines
        with open(output_file_path, 'w') as output_file, open(removed_param_path, 'w') as removed_file:
            for line in processed_lines:
                original_line = line
                line_stripped = line.strip()
                if line_stripped:  # Check if line is not empty
                    first_word = line_stripped.split()[0].strip()
                    # If the first word is in the set of first words from the tensilelite file, write the line to the output file
                    if first_word in first_words_in_tensilelite_lib_logic or first_word == 'ProblemType:':
                        # Convert MIArchVgpr to always be false (regardless of its original value)
                        if 'MIArchVgpr:' in original_line:
                            original_line = re.sub(r'MIArchVgpr:\s*\w+', 'MIArchVgpr: false', original_line)
                        # Convert LocalSplitU to 1 if value is > 1
                        if 'LocalSplitU:' in original_line:
                            original_line = re.sub(r'LocalSplitU:\s*([2-9]\d*|\d{2,})', 'LocalSplitU: 1', original_line)
                        # Convert GlobalSplitU to 1 if value is > 1
                        if 'GlobalSplitU:' in original_line:
                            original_line = re.sub(r'GlobalSplitU:\s*([2-9]\d*|\d{2,})', 'GlobalSplitU: 1', original_line)
                        # Convert WorkGroup[1] (second value) to 1 if it's not 1 (for complex GEMM compatibility)
                        if 'WorkGroup:' in original_line:
                            original_line = re.sub(r'(WorkGroup:\s*\[)(\d+),\s*([2-9]\d*|\d{2,}),\s*(\d+)(\])', r'\g<1>\2, 1, \4\5', original_line)
                        # Convert BufferLoad from 0 to 1
                        if 'BufferLoad:' in original_line:
                            original_line = re.sub(r'BufferLoad:\s*0\b', 'BufferLoad: 1', original_line)
                        # Convert BufferStore from 0 to 1
                        if 'BufferStore:' in original_line:
                            original_line = re.sub(r'BufferStore:\s*0\b', 'BufferStore: 1', original_line)
                        output_file.write(original_line)
                    elif first_word not in seen_removed_word:
                        removed_file.write(first_word + '\n')
                        seen_removed_word.add(first_word)
                else:
                    # Write empty lines if they're part of kept content
                    if not line_stripped:
                        output_file.write(original_line)
            
            # Add -Equality at the end of the file
            output_file.write('- Equality\n')
        
        if prefetch_corrections > 0:
            print(f"  → Fixed PrefetchLocalRead in {prefetch_corrections} solutions")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check that all file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_all_tensile_files():
    # Configuration
    tensilelite_path = '/data0/kamuruga/rocm-libraries/projects/hipblaslt/tensilelite/complex/aquavanjaram_Cijk_Ailk_Bjlk_DB_UserArgs.yaml'
    tensile_source_dir = '/data0/kamuruga/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/aquavanjaram/aquavanjaram942'
    output_dir = '/data0/kamuruga/rocm-libraries/projects/hipblaslt/tensilelite/complex/ported_tensile'
    
    # Read the tensilelite reference file ONCE (not for every conversion)
    print(f"Reading tensilelite reference file: {tensilelite_path}")
    with open(tensilelite_path, 'r') as tensilelite_lib_logic:
        first_words_in_tensilelite_lib_logic = set(line.split()[0].strip() for line in tensilelite_lib_logic if line.strip())
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all YAML files with CB or ZB in their name
    all_files = glob.glob(os.path.join(tensile_source_dir, '*.yaml'))
    filtered_files = [f for f in all_files if 'CB' in os.path.basename(f) or 'ZB' in os.path.basename(f)]
    
    print(f"\nFound {len(filtered_files)} files with CB or ZB in their names")
    print("=" * 80)
    
    # Process each file
    for i, tensile_file in enumerate(filtered_files, 1):
        filename = os.path.basename(tensile_file)
        print(f"\n[{i}/{len(filtered_files)}] Processing: {filename}")
        
        # Define output paths
        base_name, ext = os.path.splitext(filename)
        output_file_path = os.path.join(output_dir, filename)
        removed_param_path = os.path.join(output_dir, f"{base_name}_removed.txt")
        
        # Convert the file
        convert_liblogic_tensile_to_tensilelite(
            first_words_in_tensilelite_lib_logic=first_words_in_tensilelite_lib_logic,
            tensile_path=tensile_file,
            output_file_path=output_file_path,
            removed_param_path=removed_param_path,
            filename=filename
        )
    
    print("\n" + "=" * 80)
    print(f"✓ All done! Ported {len(filtered_files)} files.")
    print(f"✓ Output files saved to: {os.path.abspath(output_dir)}")

# Run the batch processing
if __name__ == "__main__":
    process_all_tensile_files()
