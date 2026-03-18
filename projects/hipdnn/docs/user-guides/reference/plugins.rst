.. meta::
  :description: Learn about the operations supported in hipDNN.
  :keywords: hipDNN, ROCm, operations

.. _plugin-support:

************************
hipDNN operation support
************************

hipDNN operations are implemented through plugins. Each plugin provides its own set of supported operations. For detailed information about what operations are available, refer to the plugin-specific documentation.

Plugins
=======

- :ref:`miopen-provider`: Provides integration with AMD's `MIOpen library <https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html>`_ for GPU-accelerated deep learning operations.

  - Convolution operations (Forward, Dgrad, Wgrad)
  - Batchnorm operations (Training, Backward, Inference)
  - Fused operation graphs

- :ref:`hipblaslt`: Provides integration with AMD's `hipBLASLt library <https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html>`_ that provides optimized GEMM operations.
