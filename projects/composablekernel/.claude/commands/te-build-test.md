Clean build and test a tile engine operator.

Usage: /te-build-test <operator_name>
  operator_name: e.g. gemm_aquant, gemm_bquant

Example: /te-build-test gemm_aquant

## Instructions

The user wants to clean-build and benchmark a tile engine operator. The operator name is provided as: $ARGUMENTS

If no operator name is provided, ask the user which operator to test.

Run the following 4 steps sequentially from the project root (`/dockerx/rocm-libraries/projects/composablekernel`). Stop and report if any step fails.

### Step 1: Clean and create build directory

```
cd /dockerx/rocm-libraries/projects/composablekernel && rm -rf build && mkdir build
```

### Step 2: Configure with CMake

```
cd /dockerx/rocm-libraries/projects/composablekernel/build && ../script/cmake-ck-dev.sh ../ gfx942 -G Ninja
```

### Step 3: Build the benchmark target

```
cd /dockerx/rocm-libraries/projects/composablekernel/build && ninja -j256 benchmark_${OPERATOR}_all
```

Where `${OPERATOR}` is the operator name (e.g. `gemm_aquant`).

### Step 4: Run the benchmark

```
cd /dockerx/rocm-libraries/projects/composablekernel/build && python3 ../tile_engine/ops/gemm/block_scale_gemm/${OPERATOR}/${OPERATOR}_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json
```

Report the benchmark results to the user when done.
