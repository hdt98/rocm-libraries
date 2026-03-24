# SDPA Kernel Provider Plugin
A minimal plugin proof of concept for running standalone SDPA kernels. This plugin is not yet production ready and only contains a single MI300 kernel, provided by the AITER project.

### Building as a standalone plugin
In order to build the plugin standalone, you will need to have installed hipDNN on the system first.

1. Navigate to the `dnn-providers/sdpa-kernel-provider` directory.
1. Make a build directory, `mkdir build && cd build`.
1. Run `cmake -DCMAKE_CXX_COMPILER=<path to amdclang>/clang++ ..` to configure the build.
1. Run `ninja` to build the plugin.

### Running tests
Having used the building instructions as before, run `ninja check` to run the tests
