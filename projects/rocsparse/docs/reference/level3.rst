.. meta::
  :description: rocSPARSE level 3 functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, level 3 functions

.. _rocsparse_level3_functions_:

********************************************************************
Sparse level 3 functions
********************************************************************

This module contains all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format
that can also be seen as a dense matrix.

rocsparse_bsrmm()
-----------------

.. doxygenfunction:: rocsparse_sbsrmm
  :outline:
.. doxygenfunction:: rocsparse_dbsrmm
  :outline:
.. doxygenfunction:: rocsparse_cbsrmm
  :outline:
.. doxygenfunction:: rocsparse_zbsrmm

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_bsrmm_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_bsrmm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_bsrmm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


rocsparse_gebsrmm()
-------------------

.. doxygenfunction:: rocsparse_sgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_dgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_cgebsrmm
  :outline:
.. doxygenfunction:: rocsparse_zgebsrmm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_gebsrmm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_gebsrmm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


rocsparse_csrmm()
-----------------

.. doxygenfunction:: rocsparse_scsrmm
  :outline:
.. doxygenfunction:: rocsparse_dcsrmm
  :outline:
.. doxygenfunction:: rocsparse_ccsrmm
  :outline:
.. doxygenfunction:: rocsparse_zcsrmm

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_csrmm_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_csrmm.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_csrmm_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csrsm_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_csrsm_zero_pivot

rocsparse_csrsm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_scsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_buffer_size

rocsparse_csrsm_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_analysis

rocsparse_csrsm_solve()
-----------------------

.. doxygenfunction:: rocsparse_scsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_ccsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_zcsrsm_solve

rocsparse_csrsm_clear()
-----------------------

.. doxygenfunction:: rocsparse_csrsm_clear

rocsparse_bsrsm_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_bsrsm_zero_pivot

rocsparse_bsrsm_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_buffer_size

rocsparse_bsrsm_analysis()
--------------------------

.. doxygenfunction:: rocsparse_sbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_analysis

rocsparse_bsrsm_solve()
-----------------------

.. doxygenfunction:: rocsparse_sbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_dbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_cbsrsm_solve
  :outline:
.. doxygenfunction:: rocsparse_zbsrsm_solve

rocsparse_bsrsm_clear()
-----------------------

.. doxygenfunction:: rocsparse_bsrsm_clear

rocsparse_gemmi()
-----------------

.. doxygenfunction:: rocsparse_sgemmi
  :outline:
.. doxygenfunction:: rocsparse_dgemmi
  :outline:
.. doxygenfunction:: rocsparse_cgemmi
  :outline:
.. doxygenfunction:: rocsparse_zgemmi

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_gemmi_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_gemmi.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_rocsparse_gemmi_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:
