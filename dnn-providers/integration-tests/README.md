# Integration Tests

Integration tests for hipDNN provider implementations.

## Running Tests

### Build & run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja check` | All tests | Full validation |
| `ninja unit-check` | All unit tests | CI gate on every PR |
| `ctest` | fast + slow (all tests) | Alternative to ninja check |
| `ctest -LE slow` | fast only | Excludes tests labeled slow |
| `ctest -L slow` | slow only | Medium + Large shapes |

### Filtering GPU reference tests by category

All GPU reference shape tests live in a single binary (`hipdnn_gpu_ref_tests`).
Test instantiation prefixes encode the category (`{Size}{Layout}{Dim}`):

| Prefix | Category | Typical run |
|--------|----------|-------------|
| `Small*` | Fast / small shapes | Every PR |
| `Medium*` | Medium shapes | Nightly |
| `Large*` | Large shapes | Weekly |

Use `--gtest_filter` to select categories:

```bash
# Fast only: small shapes + standalone tests
./bin/hipdnn_gpu_ref_tests --gtest_filter="Small*:Test*"

# Slow only: medium + large shapes
./bin/hipdnn_gpu_ref_tests --gtest_filter="Medium*:Large*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```
