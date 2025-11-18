.. meta::
  :description: rocSPARSE level 2 functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, level 2 functions

.. _rocsparse_level2_functions_:

********************************************************************
Sparse level 2 functions
********************************************************************

This module contains all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse format and a vector in dense format.

rocsparse_bsrmv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_sbsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrmv_analysis

rocsparse_bsrmv()
-----------------

.. doxygenfunction:: rocsparse_sbsrmv
  :outline:
.. doxygenfunction:: rocsparse_dbsrmv
  :outline:
.. doxygenfunction:: rocsparse_cbsrmv
  :outline:
.. doxygenfunction:: rocsparse_zbsrmv

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_bsrmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_bsrmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_bsrmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_bsrmv_clear()
-----------------------

.. doxygenfunction:: rocsparse_bsrmv_clear

rocsparse_bsrxmv()
------------------

.. doxygenfunction:: rocsparse_sbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_dbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_cbsrxmv
  :outline:
.. doxygenfunction:: rocsparse_zbsrxmv

rocsparse_bsrsv_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_bsrsv_zero_pivot

rocsparse_bsrsv_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_buffer_size

rocsparse_bsrsv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_sbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_analysis

rocsparse_bsrsv_solve()
-----------------------

.. doxygenfunction:: rocsparse_sbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_cbsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_zbsrsv_solve

rocsparse_bsrsv_clear()
-----------------------

.. doxygenfunction:: rocsparse_bsrsv_clear

rocsparse_coomv()
-----------------

.. doxygenfunction:: rocsparse_scoomv
  :outline:
.. doxygenfunction:: rocsparse_dcoomv
  :outline:
.. doxygenfunction:: rocsparse_ccoomv
  :outline:
.. doxygenfunction:: rocsparse_zcoomv

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_coomv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_coomv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_coomv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csrmv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrmv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrmv_analysis

rocsparse_csrmv()
-----------------

.. doxygenfunction:: rocsparse_scsrmv
  :outline:
.. doxygenfunction:: rocsparse_dcsrmv
  :outline:
.. doxygenfunction:: rocsparse_ccsrmv
  :outline:
.. doxygenfunction:: rocsparse_zcsrmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_csrmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_csrmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_csrmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_csrmv_analysis_clear()
--------------------------------

.. doxygenfunction:: rocsparse_csrmv_clear

rocsparse_csrsv_zero_pivot()
----------------------------

.. doxygenfunction:: rocsparse_csrsv_zero_pivot

rocsparse_csrsv_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_scsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_buffer_size

rocsparse_csrsv_analysis()
--------------------------

.. doxygenfunction:: rocsparse_scsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_analysis

rocsparse_csrsv_solve()
-----------------------

.. doxygenfunction:: rocsparse_scsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_ccsrsv_solve
  :outline:
.. doxygenfunction:: rocsparse_zcsrsv_solve

rocsparse_csrsv_clear()
-----------------------

.. doxygenfunction:: rocsparse_csrsv_clear


rocsparse_csritsv_zero_pivot()
------------------------------

.. doxygenfunction:: rocsparse_csritsv_zero_pivot

rocsparse_csritsv_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_scsritsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsritsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsritsv_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsritsv_buffer_size

rocsparse_csritsv_analysis()
----------------------------

.. doxygenfunction:: rocsparse_scsritsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_dcsritsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_ccsritsv_analysis
  :outline:
.. doxygenfunction:: rocsparse_zcsritsv_analysis

rocsparse_csritsv_solve()
-------------------------

.. doxygenfunction:: rocsparse_scsritsv_solve
  :outline:
.. doxygenfunction:: rocsparse_dcsritsv_solve
  :outline:
.. doxygenfunction:: rocsparse_ccsritsv_solve
  :outline:
.. doxygenfunction:: rocsparse_zcsritsv_solve

rocsparse_csritsv_solve_ex()
----------------------------

.. doxygenfunction:: rocsparse_scsritsv_solve_ex
  :outline:
.. doxygenfunction:: rocsparse_dcsritsv_solve_ex
  :outline:
.. doxygenfunction:: rocsparse_ccsritsv_solve_ex
  :outline:
.. doxygenfunction:: rocsparse_zcsritsv_solve_ex

rocsparse_csritsv_clear()
-------------------------

.. doxygenfunction:: rocsparse_csritsv_clear

rocsparse_ellmv()
-----------------

.. doxygenfunction:: rocsparse_sellmv
  :outline:
.. doxygenfunction:: rocsparse_dellmv
  :outline:
.. doxygenfunction:: rocsparse_cellmv
  :outline:
.. doxygenfunction:: rocsparse_zellmv

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_ellmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_ellmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_ellmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_hybmv()
-----------------

.. doxygenfunction:: rocsparse_shybmv

  :outline:
   .. tab:: Fortran
.. doxygenfunction:: rocsparse_dhybmv

  :outline:
      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_hybmv_fortran.f90
.. doxygenfunction:: rocsparse_chybmv
         :language: fortran
  :outline:
         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_zhybmv
         :end-before: ! [doc example end]

         :linenos:
.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_hybmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_hybmv_c.c
         :language: c

         :start-after: /*! [doc example] */
   .. tab:: Fortran
         :end-before: /*! [doc example] */

         :linenos:
      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gebsrmv_fortran.f90

         :language: fortran
rocsparse_gebsrmv()
         :start-after: ! [doc example start]
-------------------
         :end-before: ! [doc example end]

         :linenos:
.. doxygenfunction:: rocsparse_sgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_dgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_cgebsrmv
  :outline:
.. doxygenfunction:: rocsparse_zgebsrmv

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gebsrmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gebsrmv.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gebsrmv_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gemvi_buffer_size()
-----------------------------

.. doxygenfunction:: rocsparse_sgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgemvi_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgemvi_buffer_size

rocsparse_gemvi()
-----------------

.. doxygenfunction:: rocsparse_sgemvi
  :outline:
.. doxygenfunction:: rocsparse_dgemvi
  :outline:
.. doxygenfunction:: rocsparse_cgemvi
  :outline:
.. doxygenfunction:: rocsparse_zgemvi

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gemvi.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_rocsparse_gemvi_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:
