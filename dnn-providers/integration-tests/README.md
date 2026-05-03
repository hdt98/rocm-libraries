# Integration Tests

Shared integration tests for hipDNN provider implementations.
Providers are expected to follow the same prefix convention so that a single
`GTEST_FILTER` value works across the superbuild and all sub-projects.

## Test Tiers

Tests are categorized by how long they take to run. Only slow tests need
explicit prefixes — everything else runs by default.

| Tier | GTest prefix | What it covers | CI cadence |
|------|-------------|----------------|------------|
| Quick | *(default)* | Smoke shapes, standalone tests, TYPED_TESTs | Every PR |
| Comprehensive | `Medium` | Medium shapes | Nightly |
| Full | `Full` | Large shapes | Weekly |

**Design principle: opt-in to slow, everything else runs by default.**
The quick tier uses an exclusion filter (`-Medium*:-Full*`) instead of a
positive match. This means any test — `TEST()`, `TEST_F()`, `TYPED_TEST()`,
or `INSTANTIATE_TEST_SUITE_P` with a non-slow prefix — automatically runs
in quick without needing a specific naming convention. No test is silently
dropped.

**Only two prefixes to learn:** `Medium` and `Full`. The `Smoke` prefix is
used in `INSTANTIATE_TEST_SUITE_P` for readability (marks quick parameterized
tests) but the ctest filter does not depend on it.

**Current CI reality:** TheRock CI already runs with `GTEST_FILTER=-Full*`
via `test_miopenprovider.py`. The tier system adds finer granularity on top
of that existing split.

### How to assign a tier

For parameterized tests, set the first argument of `INSTANTIATE_TEST_SUITE_P`:

```cpp
// Quick — use Smoke (or any prefix that isn't Medium/Full)
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getSmall2dConvCases()),
                         byTag());

// Comprehensive — use Medium
INSTANTIATE_TEST_SUITE_P(Medium,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getMedium2dConvCases()),
                         byTag());

// Full — use Full
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuConvFwdRef2dFp32,
                         ::testing::ValuesIn(getLarge2dConvCases()),
                         byTag());
```

Standalone `TEST()` / `TEST_F()` / `TYPED_TEST()` suites are quick by default.
No prefix or annotation needed.

### Adding a new shape

Add it to the appropriate function in `ConvShapeCatalog.hpp`
(e.g. `getSmall2dConvCases()` for smoke, `getLarge2dConvCases()` for full).
The existing `INSTANTIATE_TEST_SUITE_P` picks it up — no other changes needed.

## Running Tests

### Build & run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja unit-check` | Quick tier (smoke + standalone) | Fast developer iteration |
| `ninja check` | All tiers | Full validation |
| `ctest -L quick` | Quick tier | Pre-commit / PR |
| `ctest -L comprehensive` | Quick + medium | Nightly |
| `ctest -L full` | All tiers | Weekly |

### Direct binary invocation

```bash
# Quick tier (everything except medium/full)
./bin/hipdnn_gpu_ref_tests --gtest_filter="-Medium*:-Full*"

# Smoke shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Smoke*"

# Medium shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Medium*"

# Full shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Full*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```
