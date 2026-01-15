def write_common_lines_based_on_first_word(file1_path, file2_path, output_file_path, removed_param_path):
    try:
        # Read the first file and store the first word of each line in a set
        with open(file1_path, 'r') as file1:
            first_words_in_file1 = set(line.split()[0].strip() for line in file1 if line.strip())

        # create set to ignore duplicate removed params
        seen_removed_word = set()
        # Open the second file and the output file
        with open(file2_path, 'r') as file2, open(output_file_path, 'w') as output_file, open(removed_param_path, 'w') as removed_file:
            # Iterate over the lines in the second file
            for line in file2:
                original_line = line
                line = line.strip()
                if line:  # Check if line is not empty
                    first_word = line.split()[0].strip()
                    # If the first word is in the set of first words from the first file, write the line to the output file
                    if first_word in first_words_in_file1 or first_word == 'ProblemType:':
                        output_file.write(original_line)
                    elif first_word not in seen_removed_word:
                        removed_file.write(first_word + '\n')
                        seen_removed_word.add(first_word)

        print(f"Lines with common first words have been written to {output_file_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check that all file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file1_path = 'Tensilelite_hipblaslt/aquavanjaram_Cijk_Ailk_Bjlk_DB_UserArgs.yaml'
file2_path = 'Tensile_rocblas/aquavanjaram942_Cijk_Ailk_Bjlk_CB.yaml'
removed_param_path = 'removed.txt'
output_file_path = 'aquavanjaram942_Cijk_Ailk_Bjlk_CB.yaml'

write_common_lines_based_on_first_word(file1_path, file2_path, output_file_path, removed_param_path)