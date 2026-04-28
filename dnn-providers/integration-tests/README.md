# Integration Tests

Integration tests for hipDNN provider implementations.

## Running Tests

### Build & run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja check` | All tests | Full validation |
| `ninja unit-check` | All unit tests | CI gate on every PR |
| `ctest -L quick` | Quick tests only | Small shapes + standalone tests |
| `ctest -L nightly` | Nightly tests only | Medium + Large shapes |

### Filtering GPU reference tests by category

All GPU reference shape tests live in a single binary (`hipdnn_gpu_ref_tests`).
Test instantiation prefixes encode the shape size (`{Size}{Layout}{Dim}`):

| Prefix | Size | Typical run |
|--------|------|-------------|
| `Small*` | Small shapes | Every PR (quick) |
| `Medium*` | Medium shapes | Nightly |
| `Large*` | Large shapes | Nightly |

Use `--gtest_filter` to select manually:

```bash
# Quick only: small shapes + standalone tests
./bin/hipdnn_gpu_ref_tests --gtest_filter="Small*:Test*"

# Nightly only: medium + large shapes
./bin/hipdnn_gpu_ref_tests --gtest_filter="Medium*:Large*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```
