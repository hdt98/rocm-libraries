# HipBLAS CTest Integration Documentation

## Overview

The HipBLAS project has been enhanced with a comprehensive CTest integration system that allows for selective test execution based on categories. This system provides flexible test organization and execution capabilities using YAML-based configuration files.

## Architecture

The CTest integration consists of several key components:

### 1. Core Components

- **`test_categories.yaml`**: YAML configuration file defining test categories and their properties
- **`parse_test_categories.py`**: Python script that parses the YAML file and generates CMake test definitions
- **`cmake/TestCategories.cmake`**: CMake module containing functions for applying test categories
- **CTest Integration**: Built into the main CMake build system for automatic test discovery

### 2. Integration Flow

```
test_categories.yaml → parse_test_categories.py → Generated CMake Code → CTest Execution
```

## Test Categories Structure

### Available Categories (arbitrarily added - change according to needs)

The HipBLAS project organizes tests into the following categories:

| Category | Description | Typical Timeout |
|----------|-------------|-----------------|
| `auxil` | Auxiliary functions (set/get operations, utilities) | 120 seconds |
| `blas1` | BLAS Level 1 operations (vector-vector) | 300 seconds |
| `blas2` | BLAS Level 2 operations (matrix-vector) | 600 seconds |
| `blas3` | BLAS Level 3 operations (matrix-matrix) | 900 seconds |
| `blas_ex` | Extended BLAS operations (mixed precision) | 600 seconds |
| `solver` | Linear algebra solver operations | 1200 seconds |

### Labels System (arbitrarily added - change according to needs)

Each test category can have multiple labels for flexible selection:
- **Category-specific labels**: `auxil`, `blas1`, `blas2`, `blas3`, `blas_ex`, `solver`
- **Level-based labels**: `level1`, `level2`, `level3`, `level4`, `level5`
- **Feature-based labels**: `basic`, `extended`, `utility`
- **Grouping labels**: `suite`

## Creating test_categories.yaml

### File Structure

The `test_categories.yaml` file follows this structure:

```yaml
# HipBLAS Test Categories Configuration
test_categories:
  category_name:
    description: "Human-readable description of the test category"
    test_patterns:
      - "*pattern1*"
      - "*pattern2*"
    test_files:
      - "path/to/test_file1.cpp"
      - "path/to/test_file2.cpp"
    exclude:
      - "*excluded_pattern*"
    labels:
      - "label1"
      - "label2"

execution_settings:
  default_timeout: 300
  category_timeouts:
    category_name: 600
```

### Configuration Sections

#### 1. test_categories

**Required fields:**
- `description`: Brief description of what this category tests
- `test_patterns`: List of GTest filter patterns to include
- `labels`: List of CTest labels to apply

**Optional fields:**
- `test_files`: List of source files containing these tests (for documentation)
- `exclude`: List of patterns to exclude from the category

#### 2. execution_settings

**Optional section for test execution configuration:**
- `default_timeout`: Default timeout for all tests (seconds)
- `category_timeouts`: Per-category timeout overrides

### Example Configuration

```yaml
test_categories:
  blas1:
    description: "BLAS Level 1 operations (vector-vector operations)"
    test_patterns:
      - "*asum*"
      - "*axpy*"
      - "*dot*"
      - "*nrm2*"
    test_files:
      - "blas1/asum_gtest.cpp"
      - "blas1/axpy_gtest.cpp"
    exclude:
      - "*dot*"  # Exclude this pattern
    labels:
      - "blas1"
      - "level1"
      - "basic"

execution_settings:
  default_timeout: 300
  category_timeouts:
    blas1: 600
    blas3: 1200
```

## Running Tests

### Prerequisites

1. **Build the tests**
   ```bash
   git clone https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/hipblas
   ./install.sh -c
   ```

2. **Ensure Python 3** is available (required for YAML parsing)


### Basic CTest Commands

#### Run Tests
```bash
cd rocm-libraries/projects/hipblas/build/release/clients
#Run all tests
ctest

# Run all BLAS1 tests
ctest -L blas1

# Run all Level 3 operations
ctest -L level3

# Run auxiliary tests
ctest -L auxil
```
## Integration with Build System

The test categories are automatically integrated when building with tests.
parse_test_categories.py will create the file test_categories.cmake at build time which adds the tests to CTest system.

```cmake
# In gtest/CMakeLists.txt
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TestCategories.cmake)

#If the YAML file is missing, the system falls back to hardcoded test categories defined in `TestCategories.cmake`.

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories.yaml")
    message(STATUS "Using YAML-based test categorization")
    apply_test_category_labels(hipblas-test "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories.yaml")
else()
    message(STATUS "Using hardcoded test categorization (YAML missing)")
    apply_hardcoded_test_categories()
endif()
```

## File Locations

```
projects/hipblas/clients/
            │       ├── test_categories.yaml                        # Main configuration file
            │       ├── parse_test_categories.py                    # YAML parser script
            │       ├── cmake/
            │       │   └── TestCategories.cmake                    # CMake integration functions
            │       └── gtest/
            │            └── CMakeLists.txt                         # Test integration point
            └── /build/release/clients/gtest/test_categories.cmake  #created at build time    
```
## Sample run
```
root@dell-rack-13:/dockerx/rocm-libraries/projects/hipblas/build/release/clients# ctest -N
Test project /dockerx/rocm-libraries/projects/hipblas/build/release/clients
  Test #1: hipblas-auxil-suite
  Test #2: hipblas-blas1-suite
  Test #3: hipblas-blas2-suite
  Test #4: hipblas-blas3-suite
  Test #5: hipblas-blas_ex-suite
  Test #6: hipblas-solver-suite

Total Tests: 6
root@dell-rack-13:/dockerx/rocm-libraries/projects/hipblas/build/release/clients# ctest -L blas1
Test project /dockerx/rocm-libraries/projects/hipblas/build/release/clients
    Start 2: hipblas-blas1-suite
1/1 Test #2: hipblas-blas1-suite ..............   Passed    4.71 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
blas1     =   4.71 sec*proc (1 test)
level1    =   4.71 sec*proc (1 test)

Total Test time (real) =   4.71 sec
```
