# Origami: Analytical GEMM Solution Selection

The name "origami" still evokes the elegance of transforming a flat (2-D) sheet into intricate higher dimensional structures. In this context, however, Origami has evolved into a toolset for **GEMM solution selection and optimization**. Inspired by the art of paper folding, the library now enables users to explore a range of tiling and mapping configurations and to make informed decisions on data and computation mapping for high-performance GEMM operations.

Origami provides a rigorous methodology to analyze and select optimal GEMM parameters. It does so by evaluating both **compute** and **memory latencies** across a wide range of tile sizes. The framework computes essential metrics such as:
- **Matrix Instruction (MI) Tiling Counts** and the associated compute latencies.
- **Memory Load Latencies**, considering both L2 and main memory bandwidth constraints.
- **Active Compute Unit Occupancy and Wave Counts**, ensuring realistic mapping to the underlying hardware.
- **Tie-breaking Strategies** using arithmetic intensity and more refined L2 hit rate estimations to resolve candidate configuration ties.

This approach allows programmers to achieve near-optimal performance without manually exploring every possibility. While default (implicit) parameters provide "out-of-the-box" performance, expert users can dive deeper into the analytical model to tune for their specific hardware and problem sizes.

## Documentation

- [Quick Start Guide](#quick-start-guide)
- [API Example](#api-example)
- [Building Origami](#building-origami)
  - [C++](#build--install-origami-c)
  - [Python](#installing-origami-python)
- [Origami Tests](#origami-tests)
- [Contribute](#contribute)

## Quick Start Guide

**Origami** provides an end-to-end analytical solution to GEMM parameter selection. It estimates performance by sweeping over candidate tile sizes and selecting the optimal configuration based on latency and arithmetic intensity. The easiest way to get started with Origami is using python development, for detail build instructions please see [Building Origami](#building-origami).

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

## API Example

### C++ API

```cpp
#include "origami/origami.hpp"
#include "origami/types.hpp"
#include <vector>
#include <iostream>

int main() {
    // Get hardware information for device 0
    auto hardware = origami::hardware_t::get_hardware_for_device(0);
    
    // Create a problem description
    origami::problem_t problem;
    problem.size.m = 2048;  // M dimension
    problem.size.n = 2048;  // N dimension
    problem.size.k = 2048;  // K dimension
    problem.batch = 1;
    problem.a_transpose = origami::transpose_t::T;
    problem.b_transpose = origami::transpose_t::N;
    problem.a_dtype = origami::data_type_t::Half;
    problem.b_dtype = origami::data_type_t::Half;
    problem.c_dtype = origami::data_type_t::Half;
    problem.d_dtype = origami::data_type_t::Half;
    problem.mi_dtype = origami::data_type_t::Half;
    problem.a_mx_block_size = 0;
    problem.b_mx_block_size = 0;
    
    // Create candidate configurations
    std::vector<origami::config_t> configs;
    origami::config_t config;
    config.mt.m = 256;  // Macro tile M
    config.mt.n = 256;  // Macro tile N
    config.mt.k = 64;   // Macro tile K
    config.mi.m = 16;   // Matrix instruction M
    config.mi.n = 16;   // Matrix instruction N
    config.mi.k = 32;   // Matrix instruction K
    config.occupancy = 4;
    configs.push_back(config);
    
    // Select best configuration
    auto best_result = origami::select_config(problem, hardware, configs);
    std::cout << "Best latency: " << best_result.latency << std::endl;
    std::cout << "Best config: MT=(" 
              << best_result.config.mt.m << ", " 
              << best_result.config.mt.n << ", " 
              << best_result.config.mt.k << ")" << std::endl;
    
    // Alternative: Simple selection using just M, N, K
    auto best_result_simple = origami::select_config_mnk(2048, 2048, 2048, hardware, configs);
    
    // Rank all configurations by performance
    auto ranked_configs = origami::rank_configs(problem, hardware, configs);
    std::cout << "Top 5 configs:" << std::endl;
    for (size_t i = 0; i < std::min(ranked_configs.size(), size_t(5)); ++i) {
        const auto& result = ranked_configs[i];
        std::cout << "  Rank " << (i+1) << ": latency=" << result.latency 
                  << ", MT=(" << result.config.mt.m << ", " 
                  << result.config.mt.n << ", " << result.config.mt.k << ")" << std::endl;
    }
    
    // Compute performance in GFLOPS
    double gflops = origami::compute_perf_gflops(hardware, problem, best_result.latency);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    return 0;
}
```

### Python API

```python
import origami

# Get hardware information for device 0
hardware = origami.get_hardware_for_device(0)

# Create a problem description
problem = origami.problem_t()
problem.size = origami.dim3_t(2048, 2048, 2048)  # M, N, K dimensions
problem.batch = 1
problem.a_transpose = origami.transpose_t.T
problem.b_transpose = origami.transpose_t.N
problem.a_dtype = origami.data_type_t.Half
problem.b_dtype = origami.data_type_t.Half
problem.c_dtype = origami.data_type_t.Half
problem.d_dtype = origami.data_type_t.Half
problem.mi_dtype = origami.data_type_t.Half
problem.a_mx_block_size = 0
problem.b_mx_block_size = 0

# Create candidate configurations
configs = []
config = origami.config_t()
config.mt = origami.dim3_t(256, 256, 64)  # Macro tile dimensions
config.mi = origami.dim3_t(16, 16, 32)    # Matrix instruction dimensions
config.occupancy = 4
configs.append(config)

# Select best configuration
best_result = origami.select_config(problem, hardware, configs)
print(f"Best latency: {best_result.latency}")
print(f"Best config: MT=({best_result.config.mt.m}, {best_result.config.mt.n}, {best_result.config.mt.k})")
```

## Building Origami

### Build & Install Origami (C++)

Assuming you are in the repository root, run:

```bash
# configure
cmake -S . -B build/ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_INSTALL_PREFIX=/opt/rocm 
# build
cmake --build build/ --parallel
```

After configuring and building, run the following command to install:

```bash
# install
cmake --target install
```

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ORIGAMI_BUILD_SHARED_LIBS` | Build shared libraries | `ON` (standalone), `OFF` (as part of rocm-libraries) |
| `ORIGAMI_ENABLE_PYTHON` | Enable Python bindings | `OFF` |
| `ORIGAMI_BUILD_TESTING` | Enable Python binding tests | `OFF` |
| `ORIGAMI_ENABLE_INSTALL` | Configure origami installation | `ON` |
| `ORIGAMI_ENABLE_FETCH` | Auto-fetch dependencies with FetchContent | `ON` |


### Installing Origami (Python)

Origami provides Python bindings that allow you to use Origami's functionality directly from Python.

#### Option 1: Install from Git (Recommended)

Install directly from the rocm-libraries repository without cloning:

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

Or install from a specific branch:

```bash
pip install git+https://github.com/ROCm/rocm-libraries.git@branch-name#subdirectory=shared/origami/python
```

#### Option 2: Install from local source

If you have cloned the repository:

```bash
cd shared/origami/python
pip install --user .
```

Or for editable/development installation:

```bash
cd shared/origami/python
pip install --user -e .
```

#### Option 3: Manual build (alternative)

If you prefer to build manually:

Make sure `ROCM_PATH` is set (e.g. `export ROCM_PATH=/opt/rocm`) and `nanobind` is installed. Then, build the bindings using the following command:

```bash
cd shared/origami/python
python setup.py build_ext --inplace
export PYTHONPATH=$(pwd):$PYTHONPATH
```

#### Option 4: Build with CMake

```bash
# configure with python bindings and tests enabled 
cmake -S . -B build/ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_INSTALL_PREFIX=/opt/rocm -D ORIGAMI_ENABLE_PYTHON=ON -D ORIGAMI_BUILD_TESTING=ON

# build 
cmake --build build/ --parallel

# run tests
cd build/
ctest --output-on-failure
```

## Origami Tests

Use CMake to configure the build system and compile Origami with testing enabled. From origami root `<rocm-libraries-root>/shared/origami`:

```bash
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -D ORIGAMI_BUILD_TESTING=ON -D ORIGAMI_ENABLE_PYTHON=ON

cmake --build build/ --parallel
```

```bash
cd build/
ctest --output-on-failure
```

## Contribute

If you want to submit an issue, you can do so on
[GitHub](https://github.com/ROCm/rocm-libraries/issues). To contribute to our repository, you can create a GitHub pull request.
