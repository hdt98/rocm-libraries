# Integration Tests

Integration tests for hipDNN provider implementations.

## Running Tests

### Build & run

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja unit-check` | Quick tests only | Fast developer iteration |
| `ninja check` | All tests (quick + nightly) | Full validation |
| `ctest -L quick` | Quick gpu_ref only | Small shapes + standalone tests |
| `ctest -L nightly` | Nightly gpu_ref only | Medium + Large shapes |

### Direct binary invocation

```bash
# Quick only: small shapes + standalone tests
./bin/hipdnn_gpu_ref_tests --gtest_filter="Small*:Test*"

# Nightly only: medium + large shapes
./bin/hipdnn_gpu_ref_tests --gtest_filter="Medium*:Large*"

# Run everything
./bin/hipdnn_gpu_ref_tests
```
