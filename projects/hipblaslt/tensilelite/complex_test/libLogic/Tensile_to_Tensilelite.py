import os
import glob

def convert_liblogic_tensile_to_tensilelite(first_words_in_tensilelite_lib_logic, tensile_path, output_file_path, removed_param_path):
    try:
        # create set to ignore duplicate removed params
        seen_removed_word = set()
        # Open the second file and the output file
        with open(tensile_path, 'r') as tensile_lib_logic, open(output_file_path, 'w') as output_file, open(removed_param_path, 'w') as removed_file:
            # Iterate over the lines in the second file
            for line in tensile_lib_logic:
                original_line = line
                line = line.strip()
                if line:  # Check if line is not empty
                    first_word = line.split()[0].strip()
                    # If the first word is in the set of first words from the tensilelite file, write the line to the output file
                    if first_word in first_words_in_tensilelite_lib_logic or first_word == 'ProblemType:':
                        output_file.write(original_line)
                    elif first_word not in seen_removed_word:
                        removed_file.write(first_word + '\n')
                        seen_removed_word.add(first_word)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check that all file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_all_tensile_files():
    # Configuration
    tensilelite_path = 'Tensilelite_hipblaslt/aquavanjaram_Cijk_Ailk_Bjlk_DB_UserArgs.yaml'
    tensile_source_dir = '/home/AMD/kamuruga/workspace/rocm-libraries/projects/rocblas/library/src/blas3/Tensile/Logic/aquavanjaram/aquavanjaram942'
    output_dir = 'ported_tensile'
    
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
        output_file_path = os.path.join(output_dir, filename)
        removed_param_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_removed.txt")
        
        # Convert the file
        convert_liblogic_tensile_to_tensilelite(
            first_words_in_tensilelite_lib_logic=first_words_in_tensilelite_lib_logic,
            tensile_path=tensile_file,
            output_file_path=output_file_path,
            removed_param_path=removed_param_path
        )
    
    print("\n" + "=" * 80)
    print(f"✓ All done! Ported {len(filtered_files)} files.")
    print(f"✓ Output files saved to: {os.path.abspath(output_dir)}")

# Run the batch processing
if __name__ == "__main__":
    process_all_tensile_files()
