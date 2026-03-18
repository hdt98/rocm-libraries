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

This section explains how `hipblaslt-test` works end-to-end, from YAML authoring
through binary test data generation, GTest registration, and GPU kernel execution.
The following figure offers a high-level description of the program flow:

![test-flow-diagram](https://github.com/user-attachments/assets/63d61b1e-ec7a-4493-9c5d-dfc31595eed5)

***1. YAML File Structure***

`matmul_gtest.yaml` opens with `include:` directives that pull in shared definitions:

```yaml
include: hipblaslt_common.yaml   # Argument struct layout, global defaults, enum defs
include: known_bugs.yaml         # Known failures to exclude or flag
include: matmul_common.yaml      # Shared matmul parameter anchors (&transA_range, etc.)

Definitions:
  - &alpha_beta_range
    - { alpha: 1, beta: 0 }
    - { alpha: 1, beta: 1 }

Tests:
- name: matmul_bad_arg
  category: pre_checkin
  function:
    - matmul_bad_arg: *real_precisions   # anchor expands to list of type combos
  transA: N
  transB: N

- name: matmul_f16
  category: quick
  function: matmul
  a_type: f16_r
  b_type: f16_r
  M: [256, 512, 1024]    # list → expanded into 3 separate test cases
  N: [256]
  K: [256]
  transA: [N, T]         # list → expanded, combined with M values
  <<: *alpha_beta_range  # merged → 2 alpha/beta combos each
```

***2. What*** `hipblaslt_gentest.py` ***Does***

The script is a **test case expander and serializer**. It is not part of the test
binary — it runs at build time to produce the binary data file the test binary reads.


### Expansion Example

A YAML entry like:

```yaml
- name: matmul_f16
  function: matmul
  a_type: f16_r
  M: [256, 512]
  N: [128]
  transA: [N, T]
  alpha: 1.0
  beta: 0.0
```

Produces **4 binary records** (2 M values × 2 transA values):

```
{function=matmul, a_type=f16_r, M=256, N=128, transA=N, alpha=1.0, beta=0.0, lda=256, ...}
{function=matmul, a_type=f16_r, M=256, N=128, transA=T, alpha=1.0, beta=0.0, lda=256, ...}
{function=matmul, a_type=f16_r, M=512, N=128, transA=N, alpha=1.0, beta=0.0, lda=512, ...}
{function=matmul, a_type=f16_r, M=512, N=128, transA=T, alpha=1.0, beta=0.0, lda=512, ...}
```

---

## Phase 2: Binary Loading and GTest Registration (Runtime)

### Binary File Format

```
hipblaslt_gtest.data
┌─────────────────────────────────────────────────────┐
│ Signature block                                     │
│  "hipBLASLt\0" + per-field offset/size pattern     │
│  + "HIPblaslT\0"                                    │
│  (used by Arguments::validate() to check compat)   │
├─────────────────────────────────────────────────────┤
│ Arguments record 0   (sizeof(Arguments) bytes)      │
├─────────────────────────────────────────────────────┤
│ Arguments record 1                                  │
├─────────────────────────────────────────────────────┤
│ ...                                                 │
└─────────────────────────────────────────────────────┘
```

### Test Suite Registration

Each operation type registers a test suite via a macro in its `*_gtest.cpp` file:

```cpp
// From matmul_gtest.cpp

struct matmul_testing : hipblaslt_test_valid {
    void operator()(const Arguments& arg) {
        if(!strcmp(arg.function, "matmul"))
            testing_matmul(arg);          // ← actual test logic
        else if(!strcmp(arg.function, "matmul_bad_arg"))
            testing_matmul_bad_arg(arg);
    }
};

struct matmul_test : RocBlasLt_Test<matmul_test, matmul_testing> {
    static bool type_filter(const Arguments& arg);     // match data types
    static bool function_filter(const Arguments& arg); // match "matmul" function
    static std::string name_suffix(const Arguments& arg); // build test name
};

INSTANTIATE_TEST_CATEGORIES(matmul_test);
// expands to:
// INSTANTIATE_TEST_SUITE_P(_, matmul_test,
//     testing::ValuesIn(HipBlasLt_TestData::begin(filter),
//                       HipBlasLt_TestData::end()))
```





### 2. Test envinronment, configuration and build

There are several alternatives to generating a test environment such as using a rocm image hosted
on dockerhub or building from source using TheRock. We we demonstrate both.

**rocm image**

```
cd rocm-libraries
docker run --rm -it -v $(pwd):/mnt/host/rocm-libraries rocm/dev-ubuntu-24.04:7.2-complete
apt update
apt install -y cmake git gfortran libopenblas-dev libmsgpack-dev libgtest-dev python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r tensilelite/requirements.txt
cd /mnt/host/rocm-libraries/projects/hipblaslt
cmake -B build -S . \
  -DHIPBLASLT_ENABLE_BLIS=OFF \
  -DGPU_TARGETS=<targets> \
  -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH="/opt/rocm/lib/llvm;/opt/rocm" \
  -DROCM_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel <N>
```

### 3. Running the tests

### 4. Adding a new test

### 5. Testing automation


## 3. `tensilelite-test`

## 4. Tensile Code Generation Tests

## 5. `rocIsa`
