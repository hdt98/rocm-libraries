# Integration Tests

Shared integration tests for hipDNN provider implementations.

## Test Tiers

| Tier | GTest prefix | Shape catalog | CI cadence | Timeout |
|------|-------------|---------------|------------|---------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `getSmall*()` | Every commit / PR | 600s |
| Standard | `Standard` | `getMedium*()` | PR gate | 1800s |
| Comprehensive | `Comprehensive` | `getLargeEdge*()` | Nightly | 3600s |
| Full | `Full` | `getLargeStress*()` | Weekly | 7200s |

Timeouts are defaults and can be overridden per binary via
`SMOKE_TIMEOUT`, `STANDARD_TIMEOUT`, `COMPREHENSIVE_TIMEOUT`, and
`FULL_TIMEOUT` arguments to `add_tiered_test_target()`.

**Smoke is a catch-all.** The smoke ctest entry uses an exclusion filter
(`-Standard*:Comprehensive*:Full*`). Every test that does not start with
`Standard`, `Comprehensive`, or `Full` runs in smoke automatically —
including standalone `TEST()`, `TEST_F()`, `TYPED_TEST()`, and any
`INSTANTIATE_TEST_SUITE_P` with a `Smoke` prefix.

**The 600-second timeout on smoke is a safety net.** If smoke starts timing
out, it means a large shape is missing its tier prefix and running where it
shouldn't. Treat a smoke timeout as a signal to check tier categorization.

**Cumulative ctest labels** — each higher tier includes all lower tiers.
The ctest label names use `quick` for the smoke tier (backlog: rename to
`smoke` for consistency):

```
ctest -L quick           # smoke only
ctest -L standard        # smoke + standard
ctest -L comprehensive   # smoke + standard + comprehensive
ctest -L full            # everything
```

### Adding a new operation

Each operation owns its own shape catalog (e.g., convolution uses
`ConvShapeCatalog.hpp`). New operations should create a similar catalog
following the same small / medium / largeEdge / largeStress pattern.

**Step 1 — CMake registration.** Register the test binary with
`add_tiered_test_target()` in `tests/CMakeLists.txt`. This creates
the four ctest entries, install staging, and RPATH setup automatically:

```cmake
add_tiered_test_target(hipdnn_my_new_op_tests ${CMAKE_CURRENT_BINARY_DIR})
```

**Step 2 — C++ test tiers.** New parameterized test suites **must** define
all four tiers explicitly:

```cpp
// Smoke — small shapes for fast smoke testing
INSTANTIATE_TEST_SUITE_P(Smoke,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getSmallMyNewOp2dCases()),
                         byTag());

// Standard — medium shapes for PR validation
INSTANTIATE_TEST_SUITE_P(Standard,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getMediumMyNewOp2dCases()),
                         byTag());

// Comprehensive — large edge-case shapes for nightly
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getLargeEdgeMyNewOp2dCases()),
                         byTag());

// Full — large stress-test shapes for weekly
INSTANTIATE_TEST_SUITE_P(Full,
                         MyNewOp2dTestFp32,
                         ::testing::ValuesIn(getLargeStressMyNewOp2dCases()),
                         byTag());
```

### Adding a new convolution shape

Add to the appropriate function in
`tests/gpu_ref/ConvShapeCatalog.hpp` — the function-to-tier mapping is
documented in its file header. The existing `INSTANTIATE_TEST_SUITE_P`
calls pick up new shapes automatically for forward, dgrad, and wgrad.

## Running Tests

### Via ctest (recommended)

| Command | What runs |
|---------|-----------|
| `ctest -L quick` | Smoke tier (smoke + standalone) |
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
./bin/hipdnn_gpu_ref_tests --gtest_filter="Comprehensive*"
./bin/hipdnn_gpu_ref_tests --gtest_filter="Full*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```

### GTest filter syntax note

The smoke exclusion filter uses `-Standard*:Comprehensive*:Full*` (single
leading dash). In GTest, only the **first** `-` starts the negative section;
all colon-separated patterns after it are excluded. Using `:-` between
patterns (e.g., `:-Comprehensive*`) does **not** negate — the dash becomes a
literal character in the pattern name.
