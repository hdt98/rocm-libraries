# Integration Tests

Shared integration tests for hipDNN provider implementations.

## Test Tiers

| Tier | GTest prefix | Shape catalog | CI cadence |
|------|-------------|---------------|------------|
| Smoke | `Smoke` *(or no prefix)* — **catch-all** | `getSmall*()` | Every commit / PR |
| Standard | `Standard` | `getMedium*()` | PR gate |
| Comprehensive | `Comprehensive` | `getLargeEdge*()` | Nightly |
| Full | `Full` | `getLargeStress*()` | Weekly |

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

New parameterized test suites **must** define all four tiers explicitly:

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

Convolution shapes live in `ConvShapeCatalog.hpp`. Add to the appropriate
function — the existing `INSTANTIATE_TEST_SUITE_P` calls pick it up
automatically for forward, dgrad, and wgrad.

| Function family | Tier | Purpose |
|----------------|------|---------|
| `getSmall*ConvCases()` | Smoke | Minimal shapes, fast |
| `getMedium*ConvCases()` | Standard | Moderate shapes, PR-level coverage |
| `getLargeEdge*ConvCases()` | Comprehensive | Corner cases (odd channels, asymmetric filters, prime K) |
| `getLargeStress*ConvCases()` | Full | Real-workload shapes (ResNeXt, DeepSpeech, large stem) |
| `getLarge*ConvCases()` | *(union)* | Returns edge + stress combined (backward compat) |

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
