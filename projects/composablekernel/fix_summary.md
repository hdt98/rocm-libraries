# Fix Summary for gemm.hpp Removal

## Problem
After removing gemm.hpp, two targets failed to build:
- `tile_example_gemm_universal`
- `tile_example_gemm_weight_preshuffle`

Both targets failed because various files were including the now-empty `ck_tile/ops/gemm.hpp`.

## Root Cause Analysis

### Missing Symbols in gemm_utils.hpp
The file `example/ck_tile/03_gemm/gemm_utils.hpp` uses many GEMM pipeline types:
1. `ck_tile::GemmPipeline` enum
2. `ck_tile::GemmPipelineScheduler` enum
3. `ck_tile::get_k_warp_tile()` function
4. Pipeline classes: `GemmPipelineAgBgCrMem`, `BaseGemmPipelineAgBgCrMem`
5. Pipeline classes: `GemmPipelineAgBgCrCompV3`, `BaseGemmPipelineAgBgCrCompV3`
6. Pipeline classes: `GemmPipelineAgBgCrCompV4`, `BaseGemmPipelineAgBgCrCompV4`
7. Pipeline classes: `GemmPipelineAgBgCrCompV5`, `BaseGemmPipelineAgBgCrCompV5`
8. Pipeline classes: `GemmPipelineAgBgCrCompV6`, `BaseGemmPipelineAgBgCrCompV6`
9. Pipeline classes: `WeightPreshufflePipelineAGmemBGmemCRegV2`, `BaseWeightPreshufflePipelineAGmemBGmemCRegV2`
10. `ck_tile::ArgParser` class
11. `ck_tile::GemmHostArgs` struct
12. `ck_tile::TileGemmUniversalTraits` struct
13. `ck_tile::UniversalGemmPipelineProblem` struct

### Missing Symbols in tensor_shuffle_utils.hpp
The file `include/ck_tile/host/tensor_shuffle_utils.hpp` uses:
1. `ck_tile::HostTensor` class
2. `ck_tile::reference_permute` function (from host_tensor.hpp)

## Solution Applied

### File 1: example/ck_tile/03_gemm/gemm_utils.hpp
Replaced the empty `#include "ck_tile/ops/gemm.hpp"` with specific includes:

```cpp
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp"
```

### File 2: include/ck_tile/host/tensor_shuffle_utils.hpp
Added missing include for HostTensor:

```cpp
#include "host_tensor.hpp"
```

## Files Modified
1. `example/ck_tile/03_gemm/gemm_utils.hpp` - Replaced gemm.hpp include with 13 specific headers
2. `include/ck_tile/host/tensor_shuffle_utils.hpp` - Added host_tensor.hpp include

## Testing
Run the build script to verify the fix:
```bash
./build_failed_targets.sh
```

Both targets should now build successfully.

## Header Mapping Reference
For future fixes in other files, here's where each symbol is defined:

| Symbol | Header File |
|--------|-------------|
| `GemmPipeline` | `ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp` |
| `GemmPipelineScheduler` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp` |
| `get_k_warp_tile()` | `ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp` |
| `GemmPipelineAgBgCrMem` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp` |
| `GemmPipelineAgBgCrCompV3` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp` |
| `GemmPipelineAgBgCrCompV4` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp` |
| `GemmPipelineAgBgCrCompV5` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp` |
| `GemmPipelineAgBgCrCompV6` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp` |
| `WeightPreshufflePipelineAGmemBGmemCRegV2` | `ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp` |
| `ArgParser` | `ck_tile/host/arg_parser.hpp` |
| `GemmHostArgs` | `ck_tile/ops/gemm/kernel/gemm_kernel.hpp` |
| `TileGemmUniversalTraits` | `ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp` |
| `UniversalGemmPipelineProblem` | `ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp` |
| `HostTensor` | `ck_tile/host/host_tensor.hpp` |

## Notes
There are 74+ other files in examples and tests that still include the empty `gemm.hpp`. These may need similar fixes if they are built and fail. Use the header mapping table above to identify the correct includes.
