# `hipblaslt` Tests

## 1. Test Overview

hipblaslt and tensilelite have multiple independent test stacks that reflect the different
components of the project. That is, there are tests of the C++ APIs and tests of the
assemebly code generation python libraries/applications:

![test-stack-expressive](https://github.com/user-attachments/assets/0fea6feb-ddfa-4538-aef3-a25e2678dce8)

The tests are organized in a directory parallel to the corresponding source code.
The hipblaslt C++ API tests and test orchestration are located in `projects/hipblaslt/clients`
while all of the other tests are located in a subdirectory of `projects/hipblaslt/tensilelite`
such as:

- `projects/hipblaslt/tensilelite/tests`
- `projects/hipblaslt/tensilelite/Tensile/Tests`
- `projects/hipblaslt/tensilelite/rocisa/test`

In the following sections, we describe each test stack including: 

- a summary of the tests and test flow
- how to setup a test environment, configure, build
- - how to run the tests
- how to extend the existing tests
- what testing automation is in place for the tests

## 2. `hipblaslt-test`

The test code and meta data for the `hipblaslt-test` are located in the `clients` directory:

```
projects/hipblaslt/clients/tests/
├── src/
│   ├── hipblaslt_gtest_main.cpp      # Entry point, device setup, signal handling
│   ├── hipblaslt_test.cpp            # Test infrastructure (thread pool, stream pool)
│   ├── matmul_gtest.cpp              # GEMM / matmul test cases
│   ├── auxiliary_gtest.cpp           # Auxiliary and extended operations
│   ├── matrix_transform_gtest.cpp    # Matrix transformation tests
│   └── hipblaslt_gtest_ext_op.cpp    # Extended operation tests
└── data/
    ├── matmul_gtest.yaml             # Primary test matrix (2,440 lines)
    ├── smoke_gtest.yaml              # Abbreviated smoke suite (404 lines)
    ├── matmul_common.yaml            # Shared matmul configurations
    ├── hipblaslt_common.yaml         # Common test definitions
    ├── auxiliary_gtest.yaml          # Auxiliary op test cases
    ├── rocroller_gtest.yaml          # ROCroller kernel tests
    └── known_bugs.yaml               # Known failures excluded from standard runs
```

Tests are data-driven. YAML files define parameter combinations (problem sizes, data types,
transpose modes, epilogue options, etc.), and `hipblaslt_gentest.py` expands them into a
binary `hipblaslt_gtest.data` file at build time. The GTest binary reads this file at
runtime to generate parameterized test cases.

This means adding a new test case often requires only editing a YAML file, not C++ code.

All matmul tests compute a reference result on CPU (via a BLAS librarysuch as openblas or blis)
and compare to the GPU result using `assert_allclose`-equivalent logic. Tolerances are
data-type-aware: stricter for FP32, relaxed for FP16/BF16/FP8.

### 1. Test Flow

### 2. Test envinronment, configuration and build

There are several alternatives to generating a test environment such as using a rocm image hosted
on dockerhub or building from source using TheRock. We we demonstrate both.

**rocm image**

```
cd rocm-libraries
docker run --rm -it -v $(pwd):/mnt/host/rocm-libraries rocm/dev-ubuntu-24.04:7.2-complete
cd /mnt/host/rocm-libraries
cmake -B build -S . \
 \
 \
cmake --build build --parallel <N>
```

### 3. Running the tests

### 4. Adding a new test

### 5. Testing automation


## 3. `tensilelite-test`

## 4. Tensile Code Generation Tests

## 5. `rocIsa`
