## Motivation

The GPU reference convolution tests lived in a single binary
(`hipdnn_gpu_ref_tests`) that mixed small, medium, and large shapes together.
The medium/large shapes were marked with `DISABLED_` prefixes so they never
actually ran — they were compiled but invisible to CI and easy to forget about.
There was no way to run them selectively, and the single monolithic binary
mixed test infrastructure with shape data, making it harder to add new shape
tiers in the future.

## What this PR does

1. **Replaces DISABLED_ prefixes with manual ctest entries using gtest filters.**
   A single binary (`hipdnn_gpu_ref_tests`) now covers all shapes. Four
   cumulative ctest entries are defined directly in `tests/CMakeLists.txt`
   with gtest filters and labels. Category names (`quick`, `standard`,
   `comprehensive`, `full`) match `test_runner.py`'s `VALID_TEST_CATEGORIES`
   for TheRock CI integration.

   | ctest entry | gtest filter | What it runs | Label |
   |-------------|--------------|--------------|-------|
   | `hipdnn_gpu_ref_tests_quick` | `Small*` | Small shapes | `quick` |
   | `hipdnn_gpu_ref_tests_standard` | `Small*:Test*` | + standalone tests | `standard`, `unit_test` |
   | `hipdnn_gpu_ref_tests_comprehensive` | `Small*:Test*:Medium*` | + medium shapes | `comprehensive` |
   | `hipdnn_gpu_ref_tests_full` | `*` | All tests | `full` |

   Each tier includes everything from the tier below it. All tiers cover the
   same algorithmic code paths (padding, stride, dilation, groups, depthwise,
   pointwise, multi-batch, 5x5 kernels, non-square spatial). The difference
   is scale — small shapes verify correctness at minimal dimensions,
   medium/large shapes verify at realistic model dimensions.

   The installed CTestTestfile includes only quick entries because TheRock CI
   runs bare `ctest` (no `-L` filter) and higher tiers would time out.
   Other tiers run locally via `ctest -L standard` / `comprehensive` / `full`.

2. **Extracts shape catalogs into dedicated headers.**
   Each operation's fixture header (`GpuConv{Fwd,Bwd,Wgrad}RefTestFixture.hpp`)
   is split into:
   - `*TestFixture.hpp` — test infrastructure (helpers, struct, fixture class,
     type aliases)
   - `*ShapeCatalog.hpp` — shape data (`getSmall2d*Cases()`,
     `getMedium2d*Cases()`, etc.) + `withChannelLastLayout()`

   Test `.cpp` files include the catalog header (which transitively includes
   the fixture). This makes it straightforward to add new shape tiers by
   just adding functions to the catalog.

3. **Full type x dimensionality coverage across all shape tiers.**
   Previously, medium/large shapes were only instantiated for fp32 and only
   for 2D convolutions. This PR adds fp16 and bfp16 instantiations and
   extends coverage to 1D and 3D for all tiers, across both default (NCHW)
   and channel-last (NHWC) layouts, for all three operations:

   | Shape tier | 1D | 2D | 3D | Types | First appears in |
   |------------|----|----|-----|-------|-----------------|
   | Small | yes | yes | yes | fp32/fp16/bfp16 | quick |
   | Test (standalone) | — | — | — | — | standard |
   | Medium | yes | yes | yes | fp32/fp16/bfp16 | comprehensive |
   | Large | yes | yes | yes | fp32/fp16/bfp16 | full |

4. **Adds MIOpen/CK regression edge case shapes to Large catalogs.**
   Large shape tiers now include shapes derived from real MIOpen and
   Composable Kernel regression bugs:
   - Odd input channels C=5 — vector alignment (SWDEV-502833)
   - Asymmetric filter 5x10 (MIOpen #540)
   - Stride-2 on tiny spatial producing 1x1 output
   - Prime output channels K=127
   - High-channel 1x1 reduction (MIOpen #2012 / SWDEV-305815)
   - Asymmetric stride (1,2) — CK edge case
   - D=1 degeneracy in 3D — collapses depth to 2D-like behavior
   - 1D analogs: odd channels, kernel=spatial (output=1), large dilation, prime K

5. **Removes dead performance tests.**
   `MediumTensorTimingComparison` and `DISABLED_LargeTensorTimingComparison`
   were benchmark artifacts with no consumers — `RecordProperty` output was
   never collected, and correctness checks were redundant with the shape tests.

6. **Bug fix: dgrad tolerance rejected 1D convolution weights.**
   `DynamicTolerancesConv.hpp` used `wDims.size() < 4` to guard spatial-dim
   extraction, which excluded valid 1D weights (rank 3). Changed to `< 3`.

7. **Adds explicit rank validation to `withChannelLastLayout()`.**
   Throws `std::invalid_argument` for unsupported tensor ranks instead of
   silently falling through.

## Files changed

| File | Change |
|------|--------|
| `tests/CMakeLists.txt` | Manual ctest entries with gtest filters, quick-only install staging |
| `tests/gpu_ref/GpuConv{Fwd,Bwd,Wgrad}RefTestFixture.hpp` | Keep infrastructure only |
| `tests/gpu_ref/GpuConv{Fwd,Bwd,Wgrad}RefShapeCatalog.hpp` | New — shape catalogs with regression edge cases |
| `tests/gpu_ref/TestGpuFpReference{Convolution,Dgrad,Wgrad}.cpp` | Remove dead perf tests, medium/large shapes now use catalog |
| `CMakeLists.txt` | Append quick-only staging to installed CTestTestfile |
| `README.md` | New — documents test commands |
| `hipdnn/.../DynamicTolerancesConv.hpp` | Fix 1D weight rank guard |
| `hipdnn/.../TestDynamicTolerancesConv.cpp` | Fix dgrad tolerance unit tests for rank-3 weights |

## How to run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja unit-check` | Standard tests only | Fast developer iteration |
| `ninja check` | All tests (all four tiers) | Full validation |
| `ctest -L quick` | Small shapes only | Pre-commit sanity |
| `ctest -L standard` | Small + standalone tests | PR validation |
| `ctest -L comprehensive` | Small + standalone + medium | Nightly |
| `ctest -L full` | All tests | Weekly |

Use `--gtest_filter` to select tests manually when running the binary directly.

## Test plan

- [ ] `ninja hipdnn_gpu_ref_tests` builds cleanly
- [ ] `ninja unit-check` runs only standard tests
- [ ] `ninja check` runs all four tiers
- [ ] `ctest -L quick` runs only small shapes
- [ ] `ctest -L full` runs all tests
- [ ] Installed CTestTestfile contains only quick entries
- [ ] GPU run: all tests pass on MI300
