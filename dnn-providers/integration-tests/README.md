# Integration Tests

Integration tests for hipDNN provider implementations.

## Running Tests

| Command | What runs | Use case |
|---------|-----------|----------|
| `ninja check` | fast + slow (all tests) | Full validation |
| `ninja unit-check` | fast only | CI gate on every PR |
| `ninja integration-check` | slow only | Nightly / manual |
