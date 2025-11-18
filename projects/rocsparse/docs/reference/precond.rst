.. meta::
  :description: rocSPARSE preconditioner functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, preconditioner functions

.. _rocsparse_precond_functions_:

********************************************************************
Sparse preconditioner functions
********************************************************************

This module contains all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

rocsparse_bsric0_zero_pivot()
-----------------------------

.. doxygenfunction:: rocsparse_bsric0_zero_pivot

rocsparse_bsric0_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_sbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsric0_buffer_size

rocsparse_bsric0_analysis()
---------------------------

.. doxygenfunction:: rocsparse_sbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsric0_analysis

rocsparse_bsric0()
------------------

.. doxygenfunction:: rocsparse_sbsric0
  :outline:
.. doxygenfunction:: rocsparse_dbsric0
  :outline:
.. doxygenfunction:: rocsparse_cbsric0
  :outline:
.. doxygenfunction:: rocsparse_zbsric0

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsric0_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsric0.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsric0_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_bsric0_clear()
------------------------

.. doxygenfunction:: rocsparse_bsric0_clear

rocsparse_bsrilu0_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_bsrilu0_zero_pivot

rocsparse_bsrilu0_numeric_boost()
---------------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_numeric_boost

rocsparse_bsrilu0_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_buffer_size

rocsparse_bsrilu0_analysis()
----------------------------

.. doxygenfunction:: rocsparse_sbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0_analysis

rocsparse_bsrilu0()
-------------------

.. doxygenfunction:: rocsparse_sbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_dbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_cbsrilu0
  :outline:
.. doxygenfunction:: rocsparse_zbsrilu0

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsrilu0_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsrilu0.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_bsrilu0_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_bsrilu0_clear()
-------------------------

.. doxygenfunction:: rocsparse_bsrilu0_clear

rocsparse_csric0_zero_pivot()
-----------------------------

.. doxygenfunction:: rocsparse_csric0_zero_pivot

rocsparse_csric0_singular_pivot()
---------------------------------

.. doxygenfunction:: rocsparse_csric0_singular_pivot

rocsparse_csric0_set_tolerance()
--------------------------------

.. doxygenfunction:: rocsparse_csric0_set_tolerance

rocsparse_csric0_get_tolerance()
--------------------------------

.. doxygenfunction:: rocsparse_csric0_get_tolerance

rocsparse_csric0_buffer_size()
------------------------------

.. doxygenfunction:: rocsparse_scsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsric0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsric0_buffer_size

rocsparse_csric0_analysis()
---------------------------

.. doxygenfunction:: rocsparse_scsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsric0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsric0_analysis

rocsparse_csric0()
------------------

.. doxygenfunction:: rocsparse_scsric0
  :outline:
.. doxygenfunction:: rocsparse_dcsric0
  :outline:
.. doxygenfunction:: rocsparse_ccsric0
  :outline:
.. doxygenfunction:: rocsparse_zcsric0

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csric0_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csric0.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csric0_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csric0_clear()
------------------------

.. doxygenfunction:: rocsparse_csric0_clear

rocsparse_csritilu0_buffer_size()
---------------------------------

.. doxygenfunction:: rocsparse_csritilu0_buffer_size

rocsparse_csritilu0_preprocess()
--------------------------------

.. doxygenfunction:: rocsparse_csritilu0_preprocess

rocsparse_csritilu0_history()
-----------------------------

.. doxygenfunction:: rocsparse_scsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_dcsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_ccsritilu0_history
  :outline:
.. doxygenfunction:: rocsparse_zcsritilu0_history


rocsparse_csritilu0_compute()
-----------------------------

.. doxygenfunction:: rocsparse_scsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_dcsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_ccsritilu0_compute
  :outline:
.. doxygenfunction:: rocsparse_zcsritilu0_compute


rocsparse_csritilu0_compute_ex()
--------------------------------

.. doxygenfunction:: rocsparse_scsritilu0_compute_ex
  :outline:
.. doxygenfunction:: rocsparse_dcsritilu0_compute_ex
  :outline:
.. doxygenfunction:: rocsparse_ccsritilu0_compute_ex
  :outline:
.. doxygenfunction:: rocsparse_zcsritilu0_compute_ex


rocsparse_csrilu0_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_csrilu0_zero_pivot

rocsparse_csrilu0_singular_pivot()
----------------------------------

.. doxygenfunction:: rocsparse_csrilu0_singular_pivot

rocsparse_csrilu0_set_tolerance()
---------------------------------

.. doxygenfunction:: rocsparse_csrilu0_set_tolerance

rocsparse_csrilu0_get_tolerance()
---------------------------------

.. doxygenfunction:: rocsparse_csrilu0_get_tolerance

rocsparse_csrilu0_numeric_boost()
---------------------------------

.. doxygenfunction:: rocsparse_scsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_numeric_boost
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_numeric_boost

rocsparse_csrilu0_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_scsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_buffer_size

rocsparse_csrilu0_analysis()
----------------------------

.. doxygenfunction:: rocsparse_scsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrilu0_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrilu0_analysis

rocsparse_csrilu0()
-------------------

.. doxygenfunction:: rocsparse_scsrilu0

  :outline:
   .. tab:: Fortran
.. doxygenfunction:: rocsparse_dcsrilu0

  :outline:
      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csrilu0_fortran.f90
.. doxygenfunction:: rocsparse_ccsrilu0
         :language: fortran
  :outline:
         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_zcsrilu0
         :end-before: ! [doc example end]

         :linenos:
.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csrilu0_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csrilu0.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_csrilu0_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csrilu0_clear()
-------------------------

.. doxygenfunction:: rocsparse_csrilu0_clear

rocsparse_gtsv_buffer_size()
----------------------------

.. doxygenfunction:: rocsparse_sgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_buffer_size

rocsparse_gtsv()
----------------

.. doxygenfunction:: rocsparse_sgtsv
  :outline:
.. doxygenfunction:: rocsparse_dgtsv
  :outline:
.. doxygenfunction:: rocsparse_cgtsv
  :outline:
.. doxygenfunction:: rocsparse_zgtsv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_gtsv_no_pivot_buffer_size()
-------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_buffer_size
  :outline:

.. doxygenfunction:: rocsparse_cgtsv_no_pivot_buffer_size
   .. tab:: Fortran
  :outline:

.. doxygenfunction:: rocsparse_zgtsv_no_pivot_buffer_size
      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_fortran.f90

         :language: fortran
rocsparse_gtsv_no_pivot()
         :start-after: ! [doc example start]
-------------------------
         :end-before: ! [doc example end]

         :linenos:
.. doxygenfunction:: rocsparse_sgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


   .. tab:: Fortran
rocsparse_gtsv_no_pivot_strided_batch_buffer_size()

---------------------------------------------------
      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_strided_batch_fortran.f90

         :language: fortran
.. doxygenfunction:: rocsparse_sgtsv_no_pivot_strided_batch_buffer_size
         :start-after: ! [doc example start]
  :outline:
         :end-before: ! [doc example end]
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_strided_batch_buffer_size
         :linenos:
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot_strided_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot_strided_batch_buffer_size

rocsparse_gtsv_no_pivot_strided_batch()
---------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_no_pivot_strided_batch
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_no_pivot_strided_batch

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_strided_batch_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_strided_batch.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_no_pivot_strided_batch_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gtsv_interleaved_batch_buffer_size()
----------------------------------------------

.. doxygenfunction:: rocsparse_sgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_interleaved_batch_buffer_size

rocsparse_gtsv_interleaved_batch()
----------------------------------

.. doxygenfunction:: rocsparse_sgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_dgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_cgtsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_zgtsv_interleaved_batch

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_interleaved_batch.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_rocsparse_gtsv_interleaved_batch_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gpsv_interleaved_batch_buffer_size()
----------------------------------------------

.. doxygenfunction:: rocsparse_sgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgpsv_interleaved_batch_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgpsv_interleaved_batch_buffer_size

rocsparse_gpsv_interleaved_batch()
----------------------------------

.. doxygenfunction:: rocsparse_sgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_dgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_cgpsv_interleaved_batch
  :outline:
.. doxygenfunction:: rocsparse_zgpsv_interleaved_batch
