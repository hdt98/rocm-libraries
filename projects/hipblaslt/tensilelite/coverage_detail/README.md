#### This file contain the detail of changes added. Once we decide to merge the change in `develop`, we need to cherry-pick the specific changes.

1> a095e165068c5910608fe05561230fdf464da63a : This contain changes for enabling coverage with tox command. 

We have following options:

```bash
tox -e coverage
```

This will:
- Run all unit tests with coverage
- Run all common tests (184 YAML configs) with coverage
- Generate HTML, XML, and JSON reports
- Display a summary in the terminal

## Coverage Options

### Option 1: Unit Tests Only (Fast - ~1-2 minutes)
```bash
tox -e coverage-unit
```

Best for quick iteration during development. Only runs Python unit tests.

### Option 2: Common Tests Only (Requires pre-built client)
```bash
tox -e coverage-common
```

Runs integration tests with YAML configurations. Requires GPU and builds the client.

### Option 3: Full Coverage (Unit + Common - Recommended)
```bash
tox -e coverage
```

Comprehensive coverage including both unit and integration tests.

---

### Current Code Coverage status on `develop` branch

On GFX942 : 46.84%
On GFX950 : 25.51%
Combined:   47.66%


