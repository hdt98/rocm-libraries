.. meta::
  :description: rocSPARSE generic functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, generic functions

.. _rocsparse_generic_functions_:

********************************************************************
Sparse generic functions
********************************************************************

This module contains all sparse generic routines.

The sparse generic routines describe some of the most common operations that manipulate sparse matrices and
vectors. The generic API is more flexible than the other rocSPARSE APIs because it is easy to set
different index types, data types, and compute types. For some generic routines, for example, SpMV, the generic
API also lets users select different algorithms which have different performance characteristics depending
on the sparse matrix being operated on.

rocsparse_axpby()
-----------------

.. doxygenfunction:: rocsparse_axpby

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_axpby.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_axpby_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gather()
------------------

.. doxygenfunction:: rocsparse_gather

rocsparse_scatter()
-------------------

.. doxygenfunction:: rocsparse_scatter

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_scatter.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_scatter_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_rot()
---------------

.. doxygenfunction:: rocsparse_rot

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_rot.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_rot_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_spvv()
----------------

.. doxygenfunction:: rocsparse_spvv

rocsparse_spmv()
----------------

.. doxygenfunction:: rocsparse_spmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_v2_spmv_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_v2_spmv_buffer_size

rocsparse_v2_spmv()
-------------------

.. doxygenfunction:: rocsparse_v2_spmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_v2_spmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_v2_spmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_spmv_set_extra()
--------------------------

.. doxygenfunction:: rocsparse_spmv_set_extra

rocsparse_spmv_clear_extra()
----------------------------

.. doxygenfunction:: rocsparse_spmv_clear_extra

rocsparse_spsv()
----------------

.. doxygenfunction:: rocsparse_spsv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spsv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spsv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_sptrsv_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsv_buffer_size

rocsparse_sptrsv()
------------------

.. doxygenfunction:: rocsparse_sptrsv

rocsparse_spsm()
----------------

.. doxygenfunction:: rocsparse_spsm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spsm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spsm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_sptrsm_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sptrsm_buffer_size

rocsparse_sptrsm()
------------------

.. doxygenfunction:: rocsparse_sptrsm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sptrsm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sptrsm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_spmm()
----------------

.. doxygenfunction:: rocsparse_spmm

rocsparse_spgemm()
------------------

.. doxygenfunction:: rocsparse_spgemm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spgemm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_spgemm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_spgeam_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_spgeam_buffer_size

rocsparse_spgeam()
------------------

.. doxygenfunction:: rocsparse_spgeam

rocsparse_sddmm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sddmm_buffer_size

rocsparse_sddmm_preprocess()
----------------------------

.. doxygenfunction:: rocsparse_sddmm_preprocess

rocsparse_sddmm()
-----------------

.. doxygenfunction:: rocsparse_sddmm

rocsparse_dense_to_sparse()
---------------------------

.. doxygenfunction:: rocsparse_dense_to_sparse

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_dense_to_sparse.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_dense_to_sparse_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_sparse_to_dense()
---------------------------

.. doxygenfunction:: rocsparse_sparse_to_dense

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sparse_to_dense.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sparse_to_dense_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_sparse_to_sparse_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse_buffer_size

rocsparse_sparse_to_sparse()
----------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sparse_to_sparse.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_sparse_to_sparse_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_extract_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_extract_buffer_size

rocsparse_extract_nnz
---------------------

.. doxygenfunction:: rocsparse_extract_nnz

rocsparse_extract()
-------------------

.. doxygenfunction:: rocsparse_extract

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_extract.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_rocsparse_extract_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_check_spmat
---------------------

.. doxygenfunction:: rocsparse_check_spmat

rocsparse_spitsv
----------------

.. doxygenfunction:: rocsparse_spitsv
