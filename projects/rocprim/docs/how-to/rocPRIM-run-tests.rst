.. meta::
  :description: Run rocPRIM unit tests and benchmarks with CTest, multiple GPUs, and custom seeds
  :keywords: ROCm libraries, rocPRIM, ROCm, GoogleTest, CTest, benchmarks, testing

*********************************************
Running rocPRIM unit tests over multiple GPUs
*********************************************

The `CTest resource allocation feature <https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-allocation>`__ can be used to distribute tests across multiple GPUs, accelerating testing when multiple GPUs of the same family are in a system. It can also be used to test multiple product families without having to set ``HIP_VISIBLE_DEVICES``.

.. note::

   CMake 3.18 or later is required.

When rocPRIM is built with either :doc:`rmake -c <../install/rocPRIM-build-install-windows>` or :doc:`cmake -DBUILD_TEST=ON <../install/rocPRIM-build-install-linux>`, a ``generate_resource_spec`` file will be created. 

Use  ``generate_resource_spec`` to create a resource specification file. The resource specification file is a JSON file that describes the GPU resources available on your system. For example:

.. code:: shell

    ./generate_resource_spec resources.json

To run tests in parallel, pass the resource specification file and the number of tests to run in parallel to ``ctest``:

.. code:: shell

    ctest --resource-spec-file PATH_TO_RESOURCE_SPECIFICATION_FILE --parallel NUMBER_OF_PARALLEL_TESTS


You can use the ``-DAMDGPU_TEST_TARGETS`` when building rocPRIM to restrict testing to a family of GPUs.