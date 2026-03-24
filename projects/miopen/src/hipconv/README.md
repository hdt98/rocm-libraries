# hipconv

Convolution kernels written in HIP for AMD Instinct MI355X (CDNA 4, gfx950).

## What's here

- **hipconv/** — static library with the kernel, dispatch logic, and public API

## Supported configurations

- **Data type**: fp16
- **Layout**: NHWC input, KRSC weights, NPQK output
- **Direction**: Fprop and Dgrad (Backward Data)
- **Filter**: 3x3, stride 1, dilation 1
- **Group size**: 4, 8, and 16 channels per group
- **Constraint**: input channels == output channels (`c == k`)

The kernels use MFMA 16x16x16 (4c, 16c) and MFMA 16x16x32 (8c) instructions with buffer-load-to-LDS for input staging.

## MIOpen integration

hipconv is integrated into MIOpen as the `ConvHipConv` solver. It is registered as a
Direct algorithm solver and selected automatically when it is the fastest option for a
supported problem configuration.

### Benchmarking with MIOpenDriver

```bash
# Force MIOpen to benchmark all solvers and bypass the find-db cache,
# ensuring the fastest kernel is selected from scratch.
export MIOPEN_FIND_MODE=1
export MIOPEN_DISABLE_CACHE=1

# Grouped conv2d, fp16, NHWC layout
# n=32 c=128 H=200 W=200 k=128 filter=3x3 groups=32 forward-only
# 100 warmup iterations, 100 timed iterations
./bin/MIOpenDriver convfp16 \
    -n 32 -c 128 -H 200 -W 200 -k 128 -x 3 -y 3 -g 32 \
    -F 1 -t 100 -i 100 -m conv -V 0 \
    -I NHWC -O NHWC -f NHWC
PRNG seed: 12345678
MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvHipConv
GPU Kernel Time Forward Conv. Elapsed: 0.241975 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: fwd-conv3x3u1, 32, 128, 198, 198, 3, 3, 128, 11561730048, 327689216, 321159168, 47781, 2681, 0.241975    
```

## Project structure

```
hipconv/
  include/hipconv/
    hipconv.hpp          — public API (get_valid_configs, launch, etc.)
    conv2d_params.hpp    — Conv2dParams struct
  src/
    algo_config.h        — AlgoConfig struct (shared by all algorithms)
    algo_entry.h         — AlgorithmEntry struct (algorithm dispatch table)
    kernel_variant.h     — KernelVariant struct (kernel variant dispatch table)
    registry.cpp         — config enumeration, launch, and tolerance dispatch
    grouped/
      grouped_conv.hpp   — internal grouped conv interface (namespace grouped)
      grouped_conv.cpp   — variant table and dispatch
      grouped_4c_fp16.h  — 4-channel kernel (MFMA 16x16x16)
      grouped_8c_fp16.h  — 8-channel kernel (MFMA 16x16x32)
      grouped_8c_transforms.h — GT matrix transform helpers for 8c kernel
      grouped_16c_fp16.h — 16-channel kernel (MFMA 16x16x16)
```

## Adding a new algorithm

1. Create a new directory in `hipconv/src/` (e.g. `hipconv/src/newalgo/`)
2. Define `namespace newalgo` with `get_valid_configs()`, `launch()`, `get_tolerance()`
3. Define an `inline constexpr AlgorithmEntry algo_entry` in the algorithm's header
4. Add the new `Algorithm` enum value to `hipconv.hpp`
5. Add `&newalgo::algo_entry` to the `algorithms[]` table in `registry.cpp`

## Adding a new kernel variant (within an existing algorithm)

1. Create a new header in the algorithm's directory with its own namespace
2. Implement `launch()` and `make_variant()` returning a `KernelVariant`
3. In the algorithm's `.cpp`: add one `#include` and one entry in `variants[]`
