# MIOpen Convolution Coverage Scripts

This directory contains scripts to analyze convolution source files in the composablekernel library.

## Scripts

### 1. list_convolution_files.sh
Lists all convolution source files from the develop branch in `projects/composablekernel/library/src`.

**Usage:**
```bash
./list_convolution_files.sh
```

**Output:**
- `convolution_files_list.txt` - List of all convolution source files (.cpp, .in)

### 2. check_cmake_coverage.sh
Verifies that all convolution files from the list are referenced in CMakeLists.txt files.

**Usage:**
```bash
./check_cmake_coverage.sh
```

**Requirements:**
- Run `list_convolution_files.sh` first to generate the input file

**Output:**
- `found_in_cmake.txt` - Files found in CMakeLists.txt with their corresponding CMakeLists.txt path (format: `<file> | <cmake_path>`)
- `missing_from_cmake.txt` - Files not referenced in any CMakeLists.txt (only created if there are missing files)
- Console report with coverage statistics

## Example Workflow

```bash
# Navigate to this directory
cd projects/composablekernel/miopen_coverage

# Step 1: Generate list of convolution files from develop branch
./list_convolution_files.sh

# Step 2: Check CMake coverage on current branch
./check_cmake_coverage.sh
```

## Notes

- `list_convolution_files.sh` searches the **develop** branch
- `check_cmake_coverage.sh` searches CMakeLists.txt on the **current** branch
- Only searches within `projects/composablekernel/library/src` for both files and CMakeLists.txt
