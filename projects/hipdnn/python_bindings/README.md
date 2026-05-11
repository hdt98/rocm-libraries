# hipDNN Python Bindings

This project provides Python bindings for the hipDNN frontend library using
the nanobind library. The bindings allow users to access the functionalities
of the hipDNN library directly from Python, enabling seamless integration of
deep learning operations.

## Project Structure

```
python_bindings/
├── src/
│   ├── module.cpp               # Main entry point for the nanobind module
│   ├── graph_bindings.cpp       # Bindings for the Graph class and its methods
│   ├── handle_bindings.cpp      # Bindings for handle management
│   ├── memory_bindings.cpp      # Bindings for device memory management
│   ├── tensor_bindings.cpp      # Bindings for tensor-related functionalities
│   ├── attributes_bindings.cpp  # Bindings for attribute classes
│   └── types_bindings.cpp       # Bindings for custom types and enums
├── hipdnn_frontend/
│   ├── __init__.py              # Initializes the hipdnn_frontend package
│   └── samples/
│       ├── bn_inference.py      # Batch normalization inference sample (DISABLED)
│       ├── conv_fprop.py        # Convolution forward propagation sample
│       ├── conv_dgrad.py        # Convolution backward data gradient sample
│       └── conv_wgrad.py        # Convolution backward weight gradient sample
├── CMakeLists.txt               # CMake configuration file
├── pyproject.toml               # Python project configuration
└── README.md                    # This file
```

## Prerequisites

- CMake 3.18 or higher
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.8 or higher
- ROCm/HIP runtime and libraries
- hipDNN frontend and backend libraries (built and installed)

## Installation Methods

There are three ways to install the Python bindings, depending on your setup:

| Method | Command | When to use |
|---|---|---|
| [TheRock wheel](#1-therock-wheel-recommended) | `pip install rocm[libraries]` | End users, CI |
| [Standalone pip install](#2-standalone-pip-install) | `pip install .` from this directory | Developers building hipDNN from source |
| [cmake install](#3-cmake-install) | Automatic via `ninja install` | Intermediate staging only (not directly importable) |

### 1. TheRock Wheel (Recommended)

When hipDNN is built as part of [TheRock](https://github.com/ROCm/TheRock),
the Python bindings are automatically included in the `rocm-sdk-libraries`
wheel. Install with:

```bash
pip install rocm[libraries]
```

This is the recommended method for end users. The wheel includes both the
native hipDNN libraries and the Python bindings. After installation:

```python
import hipdnn_frontend
```

For more details on the TheRock packaging flow, see the
[Packaging and Distribution](../README.md#python-bindings--packaging-and-distribution)
section of the hipDNN README.

### 2. Standalone pip Install

For developers building hipDNN from source, the bindings can be built and
installed directly with pip using scikit-build-core:

```bash
cd python_bindings
pip install -v . -Ccmake.define.CMAKE_PREFIX_PATH=/opt/rocm
```

This triggers scikit-build-core which:
1. Sets the `SKBUILD` variable
2. Runs cmake to build the nanobind extension
3. Packages the extension into a wheel
4. Installs it into your active Python environment's `site-packages/`

After installation, `import hipdnn_frontend` works immediately.

#### Development Installation

For development work where you want to rebuild after changes:

**Editable install:**
```bash
pip install -e .
```

**Full reinstall after C++ changes:**
```bash
pip uninstall hipdnn-frontend -y
pip install -v . -Ccmake.define.CMAKE_PREFIX_PATH=/opt/rocm
```

### 3. cmake Install

When building hipDNN with cmake, enable the Python bindings with:

```bash
cmake -DHIPDNN_BUILD_PYTHON_BINDINGS=ON ...
```

When hipDNN is built as part of a superbuild (e.g. TheRock or rocm-libraries),
this option is enabled automatically. The bindings are installed to
`<install_prefix>/python_bindings/hipdnn_frontend/`. This directory is an
intermediate staging location used by TheRock's wheel builder — it is **not**
directly importable by Python.

To use the bindings from a cmake install without the wheel, you must
manually add the path:

```bash
export PYTHONPATH=/opt/rocm/python_bindings:$PYTHONPATH
python -c "import hipdnn_frontend"
```

However, using the wheel (method 1 or 2) is strongly recommended over this
approach.

## How the cmake Build Detects the Install Context

The `CMakeLists.txt` uses the `SKBUILD` variable (set by scikit-build-core)
to determine the install layout:

- **`SKBUILD` defined** (pip install): installs to the wheel root, mapped to
  `hipdnn_frontend/` by `pyproject.toml`'s `wheel.install-dir`
- **`SKBUILD` not defined** (cmake/superbuild): installs to
  `python_bindings/hipdnn_frontend/` under the cmake install prefix

## Running the Sample Applications

The repository includes several sample applications demonstrating different
operations:

### Convolution Forward Propagation
```bash
python conv_fprop.py
```

This sample demonstrates:
- Setting up a convolution forward pass
- Configuring padding, stride, and dilation parameters
- Executing the convolution and displaying results

### Convolution Backward Data Gradient
```bash
python conv_dgrad.py
```

This sample demonstrates:
- Computing input gradients (dx) given output gradients (dy) and weights
- Used in backpropagation for training neural networks

### Convolution Backward Weight Gradient
```bash
python conv_wgrad.py
```

This sample demonstrates:
- Computing weight gradients (dw) given output gradients (dy) and input (x)
- Used for updating convolution filter weights during training
