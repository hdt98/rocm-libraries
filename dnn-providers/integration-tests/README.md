# Integration Tests

Integration tests for hipDNN provider implementations.

## Running Tests

### Build & run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja unit-check` | Small shapes + standalone tests | Fast developer iteration |
| `ninja check` | All tests (all four tiers) | Full validation |
| `ctest -L quick` | Small shapes | Pre-commit sanity |
| `ctest -L standard` | Small + standalone tests | PR validation |
| `ctest -L comprehensive` | Small + standalone + medium | Nightly |
| `ctest -L full` | All tests | Weekly |

### Direct binary invocation

```bash
# Small shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Small*"

# Standalone tests only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Test*"

# Medium shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Medium*"

# Large shapes only
./bin/hipdnn_gpu_ref_tests --gtest_filter="Large*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```
