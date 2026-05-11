# hipDNN

> [!CAUTION]
> **hipDNN is in the early stages of development. There is currently limited functionality available to execute graphs. See [Operation Support](./docs/OperationSupport.md) for reference.**

> [!NOTE]
> The published hipDNN documentation is available [here](https://rocm.docs.amd.com/projects/hipdnn/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Overview

hipDNN is a graph-based deep learning library for AMD GPUs that leverages a flexible plugin architecture to provide optimized implementations and utilities for various routines.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Documentation](#documentation)
  - [User Guides](#user-guides)
  - [Developer Guides](#developer-guides)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)

---

## Getting Started

The fastest way to get started with hipDNN is to follow the [quick start steps in the build guide](./docs/Building.md#quick-start-guide).

---

## Documentation

### User Guides
- **[Building](./docs/Building.md)** - Prerequisites, build configurations, and platform-specific instructions
- **[Consumer Quick Start](./docs/ConsumerQuickStart.md)** - Using an installed hipDNN in your CMake project
- **[How-To](./docs/HowTo.md)** - Using hipDNN components and extending the framework
- **[Environment Configuration](./docs/Environment.md)** - Runtime configuration and logging setup
- **[Operation Support](./docs/OperationSupport.md)** - Currently supported operations and their status
- **[Samples](./samples/README.md)** - Frontend usage examples
- **[API Reference](#api-documentation)** - Doxygen-generated API documentation

### Developer Guides
- **[Design Overview](./docs/Design.md)** - Architecture and design descriptions and diagrams
- **[Extending hipDNN](./docs/HowTo.md#extending-hipdnn)** - How to extend hipDNN functionality
- **[Plugin Development](./docs/PluginDevelopment.md)** - Creating and using custom plugins for hipDNN
- **[Roadmap](./docs/Roadmap.md)** - Feature priorities and development plans

### Testing
- **[Testing](./docs/Testing.md)** - Synopsis of testing information
- **[Testing Strategy](./docs/testing/TestingStrategy.md)** - Specific testing approach
- **[Test Plan](./docs/testing/TestPlan.md)** - Detailed test planning
- **[Test Run Template](./docs/testing/TestRunTemplate.md)** - Guidelines for test execution

---

## Project Structure

hipDNN is organized into several key components. For detailed architecture descriptions, see the [Design Overview](./docs/Design.md).

| Component | Description |
|-----------|-------------|
| **[Backend](./backend/)** | Core shared library providing C API for operation graphs and managing plugins |
| **[Data SDK](./data_sdk/)** | Header-only library with shared types, tensor utilities, logging, and the engine name registry |
| **[Flatbuffers SDK](./flatbuffers_sdk/)** | Header-only library with FlatBuffers schemas, generated headers, graph wrappers, and optional JSON helpers |
| **[Frontend](./frontend/)** | Header-only C++ API wrapper around the backend |
| **[Plugin SDK](./plugin_sdk/)** | Header-only library for plugin development |
| **[Samples](./samples/)** | Example implementations demonstrating hipDNN usage |
| **[Tests](./tests/)** | Tests for the public API (incl. frontend integration tests) |
| **[Python Bindings](./python_bindings/)** | Python bindings for the frontend via nanobind (see [README](./python_bindings/README.md) for installation methods) |
| **[Tools](./tools/)** | Experimental utilities (e.g., benchmarking, engine listing) — subject to change |

> [!NOTE]
> Official hipDNN plugins can be found in the [dnn-providers](../../../dnn-providers/) folder (e.g., [MIOpen Plugin](../../../dnn-providers/miopen-provider/)).

### Docker Support
See [Docker README](./dockerfiles/README.md) for containerized development environments.

---

## API Documentation

hipDNN includes Doxygen-generated API documentation for the public C++ frontend.

### Building the Documentation

1. Install Doxygen:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install doxygen
   ```
   For Windows, download the installer from [doxygen.nl/download](https://www.doxygen.nl/download.html).

2. Generate the documentation:
   ```bash
   cd <project-root>
   doxygen Doxyfile
   ```

3. Open the generated documentation:
   ```bash
   # Linux
   xdg-open build/docs/html/index.html

   # Windows
   start build\docs\html\index.html
   ```

The documentation covers the frontend API including:
- Graph construction and execution
- Tensor and operation attributes
- Engine configuration and knobs
- Error handling

---

## Python Bindings — Packaging and Distribution

The Python bindings are built via nanobind and distributed through two
separate channels depending on the consumer:

### Build Phase (rocm-libraries)

`hipdnn/CMakeLists.txt` guards the bindings behind
`HIPDNN_BUILD_PYTHON_BINDINGS` (default OFF). When enabled, it calls
`add_subdirectory(python_bindings)`, which:

- Fetches nanobind via `FetchContent` and builds `hipdnn_frontend_python.*.so`
- If `SKBUILD` is set (pip install path): installs to the wheel root
- If `SKBUILD` is not set (cmake/superbuild): installs to
  `python_bindings/hipdnn_frontend/` under the install prefix

### Distribution Channels

```
┌─ tar.xz (C/C++ consumers) ───────────┐   ┌─ wheel (Python consumers) ────────┐
│                                       │   │                                   │
│  install_rocm_from_artifacts.py       │   │  pip install rocm[libraries]      │
│    → /opt/rocm/lib/libhipdnn_*.so     │   │    → hipdnn_frontend importable   │
│    → NO python bindings               │   │    → libhipdnn_backend.so included│
│                                       │   │                                   │
└───────────────────────────────────────┘   └───────────────────────────────────┘
```

- **tar.xz** — native `.so` libraries, headers, cmake configs.
  `python_bindings/**` is excluded via `artifact-hipdnn.toml`.
- **wheel (`rocm-sdk-libraries`)** — native `.so` libraries +
  Python bindings. No duplication of binding files.

### TheRock Wheel Build Flow

1. TheRock sets `-DHIPDNN_BUILD_PYTHON_BINDINGS=ON` in the hipDNN subproject
2. cmake installs to `python_bindings/hipdnn_frontend/` in the stage tree
3. `populate_python_bindings()` copies those files into the wheel's `src/`
4. `setup.py`'s `find_packages(where="./src")` discovers `hipdnn_frontend`
5. The nanobind `.so` extension goes into the wheel alongside `__init__.py`
6. `libhipdnn_backend.so` is separately included via `populate_runtime_files()`

### CI Testing

- `rocm-sdk test` runs `libraries_test.py::testHipDNNFrontendImport`
  which imports `hipdnn_frontend`
- Skips gracefully if the package is not present in the build
- Full CI chain: **build wheels** → **upload** → **pip install** →
  **`rocm-sdk test`**

See [python_bindings/README.md](./python_bindings/README.md) for all
installation methods.

---

## Contributing

For information about contributing to the hipDNN project, please see the [Contributing Guide](./CONTRIBUTING.md).
