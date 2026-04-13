# hipRAND

> [!NOTE]
> The published hipRAND documentation is available [here](https://rocm.docs.amd.com/projects/hipRAND/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipRAND is a RAND marshalling library with multiple supported backends. It sits between your
application and the backend RAND library, where it marshals inputs to the backend and results to the
application. hipRAND exports an interface that doesn't require the client to change, regardless of the
chosen backend.

hipRAND supports [rocRAND](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand).

## Requirements

You must have the following installed to use hipRAND:

* CMake (3.16 or later)
* For AMD GPUs:
  * AMD ROCm Software (5.0.0 or later)
  * rocRAND library

## Build and install

You can download pre-built packages by following the [ROCm Install Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html),
or by clicking the github releases tab (this option could have a newer version).

Once downloaded, use the following command to install hipRAND:

`sudo apt update && sudo apt install hiprand`

To build hipRAND, you can use the bash helper script (Ubuntu only) or build manually (for all
supported platforms):

* Bash helper build script:

  The helper script `install` is located in the root of the `projects/hiprand` folder. Note that this method doesn't take many
  options and hard-codes a configuration that you can specify by invoking CMake directly.

  A few commands in the script need sudo access, so it may prompt you for a password.

  * `./install -h`: Shows help
  * `./install -id`: Builds library, dependencies, and installs (the `-d` flag only needs to be passed once on
    a system)

* Manual build:

  For information on cloning and building the hipRAND library, see the
  [hipRAND installation documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/install/installation.html) for version 7.0 or later. It has helpful information on how to configure CMake and build manually.

## Running unit tests

```shell
# Go to hipCUB build directory
cd projects/hiprand; cd build

# To run all tests
ctest

# To run unit tests for hipRAND
./test/<unit-test-name>
```

### Parallel testing using multiple GPUs

hipRAND can make use of [CTest's resource allocation feature](https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation) to distribute tests across multiple GPUs in
a balanced way. This can decrease the amount of time required to run the full test suite.

#### Auto resource spec generation

An executable named `generate_resource_spec` will be built when you run `cmake -DBUILD_TEST=ON`. It can be used to generate the resource spec file, which describes the GPU resources available on your system:

```shell
# Go to hipRAND build directory
cd projects/hipRAND; cd build

# Invoke the executable with a name for the output json file
./generate_resource_spec resources.json

# Run tests in parallel with specified number of jobs
ctest --resource-spec-file ./resources.json --parallel <number-of-jobs>
```

This will launch up to the specified number of jobs to run tests in parallel, while honoring the available GPU resources defined by the resource spec file and their allocation defined by the `RESOURCE_GROUPS` property on the tests.

When using the `GPU_TEST_TARGETS` cmake option, test names are prefixed with the gfx ID of the device (eg. `gfx1100-hiprand.BasicTest`). This allows you
run device-specific groups of tests with ctest's `-R` option (eg. to run all gfx1100 tests, you could use `ctest -j<num jobs> -R "gfx1100-.*"`).

#### Manual

Assuming you have two GPUs from the gfx900 family and they are the first devices enumerated by the
system, you can specify `-D AMDGPU_TEST_TARGETS=gfx900` during configuration to specify that you
want only one family to be tested. If you leave this var empty (default), all GPUs in the system
are targeted. To specify that there are two GPUs that should be targeted, you must feed cmake a JSON file
similar to the generated `resources.json` file. For example:

```json
{
  "version": {
    "major": 1,
    "minor": 0
  },
  "local": [
    {
      "gfx900": [
        {
          "id": "0"
        },
        {
          "id": "1"
        }
      ]
    }
  ]
}
```

Pass this file to CTest using the `--resource-spec-file` flag:

```shell
# Run tests on specified GPU family
ctest --resource-spec-file <path-to-your-resources.json> --parallel <number-of-jobs>
```

## Interface examples

The hipRAND interface is compatible with rocRAND and cuRAND-v2 APIs. Porting a CUDA application
that calls the cuRAND API to an application that calls the hipRAND API is relatively straightforward. For
example, to create a generator:

### Host API

```c
hiprandStatus_t
hiprandCreateGenerator(
  hiprandGenerator_t* generator,
  hiprandRngType_t rng_type
)
```

### Device API

Here is an example that generates a log-normally distributed float from a generator (these functions
are templated for all generators).

```c
__device__ double
hiprand_log_normal_double(
  hiprandStateSobol64_t* state,
  double mean,
  double stddev
)
```

## Building the documentation locally

### Requirements

#### Doxygen

The build system uses Doxygen [version 1.9.4](https://github.com/doxygen/doxygen/releases/tag/Release_1_9_4). You can try using a newer version, but that might cause issues.

After you have downloaded Doxygen version 1.9.4:

```shell
# Add doxygen to your PATH
echo 'export PATH=<doxygen 1.9.4 path>/bin:$PATH' >> ~/.bashrc

# Apply the updated .bashrc
source ~/.bashrc

# Confirm that you are using version 1.9.4
doxygen --version
```

#### Python

The build system uses Python version 3.10. You can try using a newer version, but that might cause issues.

You can install Python 3.10 alongside your other Python versions using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

```shell
# Install Python 3.10
pyenv install 3.10

# Create a Python 3.10 virtual environment
pyenv virtualenv 3.10 venv_hiprand

# Activate the virtual environment
pyenv activate venv_hiprand
```

### Building

After cloning this repository and navigating into its directory:

```shell
# Install Python dependencies
python3 -m pip install -r docs/sphinx/requirements.txt

# Build the documentation
python3 -m sphinx -T -E -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can then open `docs/_build/html/index.html` in your browser to view the documentation.
