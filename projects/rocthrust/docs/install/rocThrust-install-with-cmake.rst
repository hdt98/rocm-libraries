.. meta::
  :description: Build and install rocThrust with CMake
  :keywords: install, building, rocThrust, AMD, ROCm, source code, cmake

.. _install-with-cmake:

********************************************************************
Building and installing rocThrust on Windows and Linux with CMake
********************************************************************

You can build and install rocThrust with CMake on either Windows or Linux.

Set ``CXX`` to ``hipcc`` and set ``CMAKE_CXX_COMPILER`` to hipcc's absolute path. For example:

.. code:: shell

    CXX=hipcc
    CMAKE_CXX_COMPILER=/usr/bin/hipcc

After :doc:`cloning the project <./rocThrust-install-overview>`, create the ``build`` directory under the ``rocthrust`` root directory, then change directory to the ``build`` directory:

.. code:: shell

    mkdir build
    cd build

Generate the rocThrust makefile using the ``cmake`` command:

.. code:: shell

    cmake ../. [-D<OPTION1=VALUE1> [-D<OPTION2=VALUE2>] ...]

The build options are:

* ``DISABLE_WERROR``: Set this to ``OFF`` to pass ``-Werror`` to the compiler. Default is ``ON``.
* ``BUILD_TEST``: Set this to ``ON`` to enable rocThrust tests. Default is ``OFF``.
* ``BUILD_HIPSTDPAR_TEST``: Set this to ``ON`` to enable HIPSTDPAR tests. Default is ``OFF``.
* ``BUILD_BENCHMARK``: Set this to ``ON`` to build rocThrust benchmarks. Default is ``OFF``.
* ``BUILD_EXAMPLE``: Set this to ``ON`` to build the rocThrust examples. Default is ``OFF``.
* ``BUILD_OFFLOAD_COMPRESS``: Set this to ``OFF`` to prevent the ``--offload-compress`` switch from being passed to the compiler and compressing the binary. On by default.
* ``USE_SYSTEM_LIB``: Set this to ``ON`` to use the installed ``ROCm`` libraries when building the tests. For this option to take effect, ``BUILD_TEST`` must be set to ``ON``. Default is ``OFF``.
* ``RNG_SEED_COUNT``: Set this to the non-repeatable random dataset count. Default is 0.
* ``PRNG_SEEDS``: Set this to the RNG seeds. The seeds must be passed as a semicolon-delimited array of 32-bit unsigned integers. To avoid command line parsing errors, enclose the entire option in quotation marks. For example, ``cmake "-DPRNG_SEEDS=1;2;3;4"``. ``-DPRNG_SEEDS=1`` is used by default.
* ``BUILD_ADDRESS_SANITIZER``: Set this to ``ON`` to build with the Clang address sanitizer enabled. Default is ``OFF``.
* ``EXTERNAL_DEPS_FORCE_DOWNLOAD``: Set this to ``ON`` to download the non-ROCm dependencies such as Google Test even if they're already installed. Default is ``OFF``.
* ``USE_HIPCXX``: Set this to ``ON`` to build with CMake HIP language support. Setting this to ``ON`` eliminates the need to use ``CXX=hipcc``. Default is ``OFF``.
* ``AMDGPU_TARGETS``: Set this to build the library, examples, tests, and benchmarks for specific architecture targets. 
* ``AMDGPU_TEST_TARGETS``: Set this to build tests for a subset of the architectures specified by ``AMDGPU_TARGETS``. When set, copies of the same test will be generated for each of the architectures listed. These tests can be run using ``ctest -R "TARGET_ARCHITECTURE"``. The list of targets must be separated by a semicolon (``;``).
* ``ROCPRIM_FETCH_METHOD`` and ``ROCRAND_FETCH_METHOD``: Set these to the method to use to download the rocPRIM and rocRAND components, respectively. Can be set to ``PACKAGE``, ``DOWNLOAD``, or ``MONOREPO``. Set to ``MONOREPO`` if the component isn't already installed and you're building rocThrust from within a clone of the `rocm-libraries <https://github.com/ROCm/rocm-libraries/>`_ repository that includes the component. Set to ``DOWNLOAD`` if the component isn't installed and you aren't in a clone of the ``rocm-libraries`` repository that includes the component. ``DOWNLOAD`` will clone the repository using sparse checkout so that only the necessary files are downloaded. Set to ``PACKAGE`` if the component is already installed. If the component isn't installed, it'll be downloaded from the repository in the same way as using the ``DOWNLOAD`` option. The default method is ``PACKAGE``.
* ``LINK_HIP_DEVICE_LIBS``: Set to ``OFF`` to prevent linking against device dependencies, including rocPRIM, and use the build for host-side computation only. Must be set to ``OFF`` when ``THRUST_DEVICE_SYSTEM`` is set to anything other than ``HIP``. Default is ``ON``.
* ``THRUST_DEVICE_SYSTEM``: Sets the device backend that will be used. Can be set to ``HIP`` for AMD GPU acceleration, ``CPP`` (C++) for sequential host-side compute, ``TBB`` (Thread Building Blocks) for parallel host-side compute, and ``OMP`` (OpenMP) for parallel host-side compute. For values other than ``HIP``,  ``LINK_HIP_DEVICE_LIBS`` must be set to ``OFF``.
* ``THRUST_HOST_SYSTEM``: Sets the host backend that will be used. If unspecified, this option defaults to ``CPP``. Valid values include ``CPP`` (C++), ``OMP`` (OpenMP), and ``TBB`` (Thread Building Blocks).

.. note::

    If you're using a version of git earlier than 2.25, ``-DROCPRIM_FETCH_METHOD=DOWNLOAD`` and ``-DROCRAND_FETCH_METHOD=DOWNLOAD`` will download the entire ``rocm-libraries`` repository.

Build rocThrust using the generated make file:

.. code:: shell

    make -j4

After you've built rocThrust, you can optionally generate tar, zip, and deb packages:

.. code:: shell

    make package

Finally, install rocThrust:

.. code:: shell

    make install
