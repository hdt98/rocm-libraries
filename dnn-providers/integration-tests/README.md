# Integration Tests

Integration tests validate hipDNN provider plugins (engine libraries such as
`libmiopen_plugin.so` or `libhipblaslt_plugin.so`) by running graphs through
the plugin and comparing results against a reference executor.

## Quick Start

```bash
# Run smoke-tier tests against a built provider plugin
./bin/hipdnn_integration_tests \
  --test-article /path/to/libmiopen_plugin.so

# If running from a superbuild, plugin discovery is automatic
./bin/hipdnn_integration_tests
```

## Test Tiers

| Tier | GTest prefix | Shape catalog | CI cadence | Timeout |
|------|-------------|---------------|------------|---------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `getSmall*()` + standalone tests | Every commit / PR | 600s (10 min) |
| Standard | `Standard` | `getMedium*()` | PR gate | 1800s (30 min) |
| Comprehensive | `Comprehensive` | `getLargeEdge*()` | Nightly | 3600s (60 min) |
| Full | `Full` | `getLargeStress*()` | Weekly | 7200s (120 min) |

Timeouts are defaults and can be overridden per binary via
`SMOKE_TIMEOUT`, `STANDARD_TIMEOUT`, `COMPREHENSIVE_TIMEOUT`, and
`FULL_TIMEOUT` arguments to `add_tiered_test_target()`.

### Smoke is a catch-all

The smoke ctest entry uses an exclusion filter
(`-Standard*:Comprehensive*:Full*`). Every test that does **not** start with
`Standard`, `Comprehensive`, or `Full` runs in smoke automatically. This
includes standalone tests and any `Smoke`-prefixed parameterized suites:

```cpp
// Runs in smoke — has Smoke prefix
INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture, ...);

// Also runs in smoke — no tier prefix, caught by the exclusion filter
TEST(MyFeature, BasicBehavior) { ... }
TEST_F(MyFixture, EdgeCase) { ... }
```

**The 600-second timeout on smoke is a safety net.** If smoke starts timing
out, it means a large shape is missing its tier prefix and running where it
shouldn't. Treat a smoke timeout as a signal to check tier categorization.

### How tiers cascade

Each higher ctest label includes all lower tiers:

```
ctest -L quick           →  [smoke]
ctest -L standard        →  [smoke + standard]
ctest -L comprehensive   →  [smoke + standard + comprehensive]
ctest -L full            →  [smoke + standard + comprehensive + full]
```

> **Note:** The ctest label uses `quick` for the smoke tier
> (backlog: rename to `smoke` for consistency).

## Running Tests

**Which runner to use:**
- **ctest** — CI and local full-tier runs
- **ninja targets** — local shortcut
- **Direct binary** — debugging a specific test with `--gtest_filter`

### Via ctest (recommended)

| Command | What runs |
|---------|-----------|
| `ctest -L quick` | Smoke tier |
| `ctest -L standard` | Smoke + standard |
| `ctest -L comprehensive` | Smoke + standard + comprehensive |
| `ctest -L full` | All tiers |

### Via ninja targets

| Command | What runs |
|---------|-----------|
| `ninja unit-check` | Smoke tier |
| `ninja check` | All tiers |

### Direct binary invocation

```bash
# Smoke tier (everything except standard/comprehensive/full)
./bin/hipdnn_gpu_ref_tests --gtest_filter="-Standard*:Comprehensive*:Full*"

# Specific tier only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Smoke*"
./bin/hipdnn_gpu_ref_tests --gtest_filter="Standard*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```

> **GTest filter syntax note:** The smoke exclusion filter uses
> `-Standard*:Comprehensive*:Full*` (single leading dash). In GTest, only
> the **first** `-` starts the negative section; all colon-separated patterns
> after it are excluded. Using `:-` between patterns (e.g.,
> `:-Comprehensive*`) does **not** negate — the dash becomes a literal
> character in the pattern name.

### Example output

```
[==========] 42 tests from 6 test suites ran. (12345 ms total)
[  PASSED  ] 42 tests.
```

## Adding a New Operation

### Directory layout

Each operation has its own test file and shape catalog:

```
tests/
  gpu_ref/
    ConvShapeCase.hpp              # Shared shape struct + byTag()
    ConvShapeCatalog.hpp           # getSmall/getMedium/getLargeEdge/getLargeStress
    TestGpuFpReferenceConvolution.cpp
    TestGpuFpReferenceDgrad.cpp
    TestGpuFpReferenceWgrad.cpp
  my_new_op/
    MyNewOpShapeCase.hpp           # Shape struct for the new op
    MyNewOpShapeCatalog.hpp        # Shape catalogs by tier
    TestMyNewOp.cpp                # TEST_P suites + INSTANTIATE_TEST_SUITE_P
```

### Step 1 — CMake registration

Register the test binary with `add_tiered_test_target()` in
`tests/CMakeLists.txt`. This creates the four ctest entries, install staging,
and RPATH setup automatically:

```cmake
add_tiered_test_target(hipdnn_my_new_op_tests ${CMAKE_CURRENT_BINARY_DIR})
```

### Step 2 — Shape catalog

Create a shape catalog following the same tier pattern. See
[`tests/gpu_ref/ConvShapeCatalog.hpp`](tests/gpu_ref/ConvShapeCatalog.hpp)
for a complete example. The function-to-tier mapping is documented in its
file header.

### Step 3 — C++ test tiers

New parameterized test suites **must** define all four tiers explicitly:

```cpp
INSTANTIATE_TEST_SUITE_P(Smoke,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getSmallMyNewOp2dCases()),
                         byTag());

INSTANTIATE_TEST_SUITE_P(Standard,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getMediumMyNewOp2dCases()),
                         byTag());

INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getLargeEdgeMyNewOp2dCases()),
                         byTag());

INSTANTIATE_TEST_SUITE_P(Full,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getLargeStressMyNewOp2dCases()),
                         byTag());
```

`byTag()` is a name generator that uses the shape's `tag` field as the test
name. Without it, a failing test shows as `Smoke/MyOp2dTestFp32.Runs/7` —
with it, you get `Smoke/MyOp2dTestFp32.Runs/n8c64k32_f3x3_s1_p1` and
immediately know which shape failed.

### Adding a new convolution shape

Add to the appropriate function in
[`tests/gpu_ref/ConvShapeCatalog.hpp`](tests/gpu_ref/ConvShapeCatalog.hpp).
The existing `INSTANTIATE_TEST_SUITE_P` calls pick up new shapes
automatically for forward, dgrad, and wgrad.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Engine 'X' is not loaded` | Plugin not found — standalone build has no auto-discovery | Pass `--test-article /path/to/plugin.so`, or run from a superbuild where plugins are in `build/lib/hipdnn_plugins/engines/` |
| Smoke tier timing out | A large shape is running in smoke because it is missing its tier prefix | Check that all `INSTANTIATE_TEST_SUITE_P` calls use `Smoke`, `Standard`, `Comprehensive`, or `Full` as the prefix |
| `No tests matched the filter` | Incorrect `--gtest_filter` syntax | Use a single `-` to start negative filters: `-Standard*:Comprehensive*:Full*`. Do **not** use `:-` between patterns |
