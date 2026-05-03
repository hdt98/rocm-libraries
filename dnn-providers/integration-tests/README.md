# Integration Tests

Shared integration tests for hipDNN provider implementations.

## Test Tiers

Tests use four tiers, controlled by the first argument of
`INSTANTIATE_TEST_SUITE_P`. The quick ctest entry uses an **exclusion filter**
(`-Standard*:Comprehensive*:Full*`), so standalone `TEST()` / `TEST_F()` /
`TYPED_TEST()` suites run in quick by default — no prefix needed.

| Tier | GTest prefix | Shape catalog | CI cadence |
|------|-------------|---------------|------------|
| Quick | `Smoke` *(or no prefix)* | `getSmall*()` | Every commit / PR |
| Standard | `Standard` | `getMedium*()` | PR gate |
| Comprehensive | `Comprehensive` | `getLargeEdge*()` | Nightly |
| Full | `Full` | `getLargeStress*()` | Weekly |

**Cumulative labels** — each higher tier includes all lower tiers:

```
ctest -L quick           # quick only
ctest -L standard        # quick + standard
ctest -L comprehensive   # quick + standard + comprehensive
ctest -L full            # everything
```

### Adding a new operation

New parameterized test suites **must** define all four tiers explicitly:

```cpp
// Quick — small shapes for fast smoke testing
INSTANTIATE_TEST_SUITE_P(Smoke,
                         MyNewOpTestFp32,
                         ::testing::ValuesIn(getSmall2dCases()),
                         byTag());

// Standard — medium shapes for PR validation
INSTANTIATE_TEST_SUITE_P(Standard,
                         MyNewOpTestFp32,
                         ::testing::ValuesIn(getMedium2dCases()),
                         byTag());

// Comprehensive — large edge-case shapes for nightly
INSTANTIATE_TEST_SUITE_P(Comprehensive,
                         MyNewOpTestFp32,
                         ::testing::ValuesIn(getLargeEdge2dCases()),
                         byTag());

// Full — large stress-test shapes for weekly
INSTANTIATE_TEST_SUITE_P(Full,
                         MyNewOpTestFp32,
                         ::testing::ValuesIn(getLargeStress2dCases()),
                         byTag());
```

### Adding a new shape

Add shapes to the appropriate function in `ConvShapeCatalog.hpp`:

| Function family | Tier | Purpose |
|----------------|------|---------|
| `getSmall*()` | Quick | Minimal shapes, fast |
| `getMedium*()` | Standard | Moderate shapes, PR-level coverage |
| `getLargeEdge*()` | Comprehensive | Corner cases (odd channels, asymmetric filters, prime K) |
| `getLargeStress*()` | Full | Real-workload shapes (ResNeXt, DeepSpeech, large stem) |
| `getLarge*()` | *(union)* | Returns edge + stress combined (backward compat) |

The existing `INSTANTIATE_TEST_SUITE_P` calls pick up new shapes automatically.

## Running Tests

### Via ctest (recommended)

| Command | What runs |
|---------|-----------|
| `ctest -L quick` | Quick tier (smoke + standalone) |
| `ctest -L standard` | Quick + standard |
| `ctest -L comprehensive` | Quick + standard + comprehensive |
| `ctest -L full` | All tiers |

### Via ninja targets

| Command | What runs |
|---------|-----------|
| `ninja unit-check` | Quick tier |
| `ninja check` | All tiers |

### Direct binary invocation

```bash
# Quick tier (everything except standard/comprehensive/full)
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

The quick exclusion filter uses `-Standard*:Comprehensive*:Full*` (single
leading dash). In GTest, only the **first** `-` starts the negative section;
all colon-separated patterns after it are excluded. Using `:-` between
patterns (e.g., `:-Comprehensive*`) does **not** negate — the dash becomes a
literal character in the pattern name.
