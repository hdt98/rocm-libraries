.. meta::
  :description: rocSPARSE conversion functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, conversion functions

.. _rocsparse_conversion_functions_:

********************************************************************
Sparse conversion functions
********************************************************************

This module contains all sparse conversion routines.

The sparse conversion routines describe operations on a matrix in sparse format to obtain a matrix in a different sparse format.

rocsparse_csr2coo()
-------------------

.. doxygenfunction:: rocsparse_csr2coo

rocsparse_coo2csr()
-------------------

.. doxygenfunction:: rocsparse_coo2csr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2csr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2csr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csr2csc_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csr2csc_buffer_size

rocsparse_csr2csc()
-------------------

.. doxygenfunction:: rocsparse_scsr2csc
  :outline:
.. doxygenfunction:: rocsparse_dcsr2csc
  :outline:
.. doxygenfunction:: rocsparse_ccsr2csc
  :outline:
.. doxygenfunction:: rocsparse_zcsr2csc

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csc_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csc.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csc_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gebsr2gebsc_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc_buffer_size

rocsparse_gebsr2gebsc()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsc
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsc

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsc_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsc.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsc_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csr2ell_width()
-------------------------

.. doxygenfunction:: rocsparse_csr2ell_width

rocsparse_csr2ell()
-------------------

.. doxygenfunction:: rocsparse_scsr2ell
  :outline:
.. doxygenfunction:: rocsparse_dcsr2ell
  :outline:
.. doxygenfunction:: rocsparse_ccsr2ell
  :outline:
.. doxygenfunction:: rocsparse_zcsr2ell

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2ell_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2ell.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2ell_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_ell2csr_nnz()
-----------------------

.. doxygenfunction:: rocsparse_ell2csr_nnz

rocsparse_ell2csr()
-------------------

.. doxygenfunction:: rocsparse_sell2csr

  :outline:
   .. tab:: Fortran
.. doxygenfunction:: rocsparse_dell2csr

  :outline:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_ell2csr_fortran.f90
.. doxygenfunction:: rocsparse_cell2csr
         :language: fortran
  :outline:
         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_zell2csr
         :end-before: ! [doc example end]

         :linenos:
.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_ell2csr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_ell2csr_c.c
         :language: c

         :start-after: /*! [doc example] */
   .. tab:: Fortran
         :end-before: /*! [doc example] */

         :linenos:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2hyb_fortran.f90

         :language: fortran
rocsparse_csr2hyb()
         :start-after: ! [doc example start]
-------------------
         :end-before: ! [doc example end]

         :linenos:
.. doxygenfunction:: rocsparse_scsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_dcsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_ccsr2hyb
  :outline:
.. doxygenfunction:: rocsparse_zcsr2hyb

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2hyb.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C


   .. tab:: Fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2hyb_c.c

         :language: c
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_hyb2csr_fortran.f90
         :start-after: /*! [doc example] */
         :language: fortran
         :end-before: /*! [doc example] */
         :start-after: ! [doc example start]
         :linenos:
         :end-before: ! [doc example end]

         :linenos:
rocsparse_hyb2csr_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_hyb2csr_buffer_size

rocsparse_hyb2csr()
-------------------

.. doxygenfunction:: rocsparse_shyb2csr
  :outline:
.. doxygenfunction:: rocsparse_dhyb2csr
  :outline:
.. doxygenfunction:: rocsparse_chyb2csr
  :outline:

.. doxygenfunction:: rocsparse_zhyb2csr
   .. tab:: Fortran


.. tabs::
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_bsr2csr_fortran.f90

         :language: fortran
   .. tab:: C++
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_hyb2csr.cpp
         :linenos:
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_hyb2csr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_bsr2csr()

-------------------
   .. tab:: Fortran


.. doxygenfunction:: rocsparse_sbsr2csr
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2csr_fortran.f90
  :outline:
         :language: fortran
.. doxygenfunction:: rocsparse_dbsr2csr
         :start-after: ! [doc example start]
  :outline:
         :end-before: ! [doc example end]
.. doxygenfunction:: rocsparse_cbsr2csr
         :linenos:
  :outline:
.. doxygenfunction:: rocsparse_zbsr2csr

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_bsr2csr_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_bsr2csr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_bsr2csr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gebsr2csr()
---------------------

.. doxygenfunction:: rocsparse_sgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2csr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2csr
  :outline:

.. doxygenfunction:: rocsparse_zgebsr2csr
   .. tab:: Fortran


.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2csr_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsr_fortran.f90

         :language: fortran
   .. tab:: C++
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2csr.cpp
         :linenos:
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2csr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_gebsr2gebsr_buffer_size()
-----------------------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr_buffer_size
  :outline:

.. doxygenfunction:: rocsparse_cgebsr2gebsr_buffer_size
   .. tab:: Fortran
  :outline:

.. doxygenfunction:: rocsparse_zgebsr2gebsr_buffer_size
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2bsr_fortran.f90

         :language: fortran
rocsparse_gebsr2gebsr_nnz()
         :start-after: ! [doc example start]
---------------------------
         :end-before: ! [doc example end]

         :linenos:
.. doxygenfunction:: rocsparse_gebsr2gebsr_nnz

rocsparse_gebsr2gebsr()
-----------------------

.. doxygenfunction:: rocsparse_sgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_cgebsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zgebsr2gebsr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_gebsr2gebsr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


   .. tab:: Fortran


rocsparse_csr2bsr_nnz()
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2gebsr_fortran.f90
-----------------------
         :language: fortran

         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_csr2bsr_nnz
         :end-before: ! [doc example end]

         :linenos:
rocsparse_csr2bsr()
-------------------

.. doxygenfunction:: rocsparse_scsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2bsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2bsr

.. tabs::

   .. tab:: C++


   .. tab:: Fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2bsr.cpp

         :language: cpp
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csr_compress_fortran.f90
         :start-after: //! [doc example]
         :language: fortran
         :end-before: //! [doc example]
         :start-after: ! [doc example start]
         :linenos:
         :end-before: ! [doc example end]

         :linenos:
   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2bsr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csr2gebsr_nnz()
-------------------------

.. doxygenfunction:: rocsparse_csr2gebsr_nnz

rocsparse_csr2gebsr_buffer_size()
---------------------------------

.. doxygenfunction:: rocsparse_scsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr_buffer_size

rocsparse_csr2gebsr()
---------------------

.. doxygenfunction:: rocsparse_scsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_dcsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_ccsr2gebsr
  :outline:
.. doxygenfunction:: rocsparse_zcsr2gebsr

.. tabs::

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2gebsr_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2gebsr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2gebsr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csr2csr_compress()
----------------------------

.. doxygenfunction:: rocsparse_scsr2csr_compress

  :outline:
   .. tab:: Fortran
.. doxygenfunction:: rocsparse_dcsr2csr_compress

  :outline:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csr_compress_fortran.f90
.. doxygenfunction:: rocsparse_ccsr2csr_compress
         :language: fortran
  :outline:
         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_zcsr2csr_compress
         :end-before: ! [doc example end]

         :linenos:
.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csr_compress.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2csr_compress_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_inverse_permutation()
-------------------------------

.. doxygenfunction:: rocsparse_inverse_permutation

rocsparse_create_identity_permutation()
---------------------------------------

.. doxygenfunction:: rocsparse_create_identity_permutation

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_create_identity_permutation.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_create_identity_permutation_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_csrsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_csrsort_buffer_size

rocsparse_csrsort()
-------------------

.. doxygenfunction:: rocsparse_csrsort

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csrsort.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C


   .. tab:: Fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csrsort_c.c

         :language: c
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz_compress_fortran.f90
         :start-after: /*! [doc example] */
         :language: fortran
         :end-before: /*! [doc example] */
         :start-after: ! [doc example start]
         :linenos:
         :end-before: ! [doc example end]

         :linenos:
rocsparse_cscsort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_cscsort_buffer_size

rocsparse_cscsort()
-------------------

.. doxygenfunction:: rocsparse_cscsort

.. tabs::

   .. tab:: C++


      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_cscsort.cpp
   .. tab:: Fortran
         :language: cpp

         :start-after: //! [doc example]
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz_fortran.f90
         :end-before: //! [doc example]
         :language: fortran
         :linenos:
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
   .. tab:: C
         :linenos:

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_cscsort_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_coosort_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_coosort_buffer_size

rocsparse_coosort_by_row()
--------------------------


.. doxygenfunction:: rocsparse_coosort_by_row
   .. tab:: Fortran


.. tabs::
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csr_fortran.f90

         :language: fortran
   .. tab:: C++
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coosort_by_row.cpp
         :linenos:
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coosort_by_row_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_coosort_by_column()
-----------------------------


   .. tab:: Fortran
.. doxygenfunction:: rocsparse_coosort_by_column


      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csc_fortran.f90
rocsparse_nnz_compress()
         :language: fortran
------------------------
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
.. doxygenfunction:: rocsparse_snnz_compress
         :linenos:
  :outline:
.. doxygenfunction:: rocsparse_dnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_cnnz_compress
  :outline:
.. doxygenfunction:: rocsparse_znnz_compress

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz_compress.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]

         :linenos:
   .. tab:: Fortran


   .. tab:: C
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2coo_fortran.f90

         :language: fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz_compress_c.c
         :start-after: ! [doc example start]
         :language: c
         :end-before: ! [doc example end]
         :start-after: /*! [doc example] */
         :linenos:
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_nnz()
---------------

.. doxygenfunction:: rocsparse_snnz
  :outline:
.. doxygenfunction:: rocsparse_dnnz
  :outline:
.. doxygenfunction:: rocsparse_cnnz
  :outline:
.. doxygenfunction:: rocsparse_znnz

.. tabs::


   .. tab:: Fortran
   .. tab:: C++


      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2dense_fortran.f90
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz.cpp
         :language: fortran
         :language: cpp
         :start-after: ! [doc example start]
         :start-after: //! [doc example]
         :end-before: ! [doc example end]
         :end-before: //! [doc example]
         :linenos:
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_nnz_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


rocsparse_dense2csr()
---------------------

.. doxygenfunction:: rocsparse_sdense2csr

  :outline:
   .. tab:: Fortran
.. doxygenfunction:: rocsparse_ddense2csr

  :outline:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csc2dense_fortran.f90
.. doxygenfunction:: rocsparse_cdense2csr
         :language: fortran
  :outline:
         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_zdense2csr
         :end-before: ! [doc example end]

         :linenos:
.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csr_c.c
         :language: c

         :start-after: /*! [doc example] */
   .. tab:: Fortran
         :end-before: /*! [doc example] */

         :linenos:
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2dense_fortran.f90

         :language: fortran

         :start-after: ! [doc example start]
rocsparse_dense2csc()
         :end-before: ! [doc example end]
---------------------
         :linenos:

.. doxygenfunction:: rocsparse_sdense2csc
  :outline:
.. doxygenfunction:: rocsparse_ddense2csc
  :outline:
.. doxygenfunction:: rocsparse_cdense2csc
  :outline:
.. doxygenfunction:: rocsparse_zdense2csc

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csc.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2csc_c.c
         :language: c
         :start-after: /*! [doc example] */

         :end-before: /*! [doc example] */
   .. tab:: Fortran
         :linenos:


      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_fortran.f90

         :language: fortran
rocsparse_dense2coo()
         :start-after: ! [doc example start]
---------------------
         :end-before: ! [doc example end]

         :linenos:
.. doxygenfunction:: rocsparse_sdense2coo
  :outline:
.. doxygenfunction:: rocsparse_ddense2coo
  :outline:
.. doxygenfunction:: rocsparse_cdense2coo
  :outline:
.. doxygenfunction:: rocsparse_zdense2coo

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2coo.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_dense2coo_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */

         :linenos:
   .. tab:: Fortran



      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_fortran.f90
rocsparse_csr2dense()
         :language: fortran
---------------------
         :start-after: ! [doc example start]

         :end-before: ! [doc example end]
.. doxygenfunction:: rocsparse_scsr2dense
         :linenos:
  :outline:
.. doxygenfunction:: rocsparse_dcsr2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsr2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsr2dense

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2dense.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csr2dense_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


   .. tab:: Fortran


rocsparse_csc2dense()
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_by_percentage_fortran.f90
---------------------
         :language: fortran

         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_scsc2dense
         :end-before: ! [doc example end]
  :outline:
         :linenos:
.. doxygenfunction:: rocsparse_dcsc2dense
  :outline:
.. doxygenfunction:: rocsparse_ccsc2dense
  :outline:
.. doxygenfunction:: rocsparse_zcsc2dense

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csc2dense.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_csc2dense_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:


rocsparse_coo2dense()
   .. tab:: Fortran
---------------------


   .. tab:: Fortran


      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_by_percentage_fortran.f90
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2dense_fortran.f90
.. doxygenfunction:: rocsparse_scoo2dense
         :language: fortran
         :language: fortran
         :start-after: ! [doc example start]
  :outline:
         :end-before: ! [doc example end]
         :start-after: ! [doc example start]
         :linenos:
.. doxygenfunction:: rocsparse_dcoo2dense
         :end-before: ! [doc example end]
  :outline:
         :linenos:
.. doxygenfunction:: rocsparse_ccoo2dense
  :outline:
.. doxygenfunction:: rocsparse_zcoo2dense

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2dense.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_coo2dense_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */

         :linenos:
   .. tab:: Fortran


rocsparse_prune_dense2csr_buffer_size()
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_fortran.f90
---------------------------------------
         :language: fortran

         :start-after: ! [doc example start]
.. doxygenfunction:: rocsparse_sprune_dense2csr_buffer_size
         :end-before: ! [doc example end]
  :outline:
         :linenos:
.. doxygenfunction:: rocsparse_dprune_dense2csr_buffer_size

rocsparse_prune_dense2csr_nnz()
-------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz

rocsparse_prune_dense2csr()
---------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]

         :linenos:
   .. tab:: Fortran


   .. tab:: C
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_fortran.f90

         :language: fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_c.c
         :start-after: ! [doc example start]
         :language: c
         :end-before: ! [doc example end]
         :start-after: /*! [doc example] */
         :linenos:
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_prune_csr2csr_buffer_size()
-------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_buffer_size

rocsparse_prune_csr2csr_nnz()
-----------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz

rocsparse_prune_csr2csr()
-------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr


.. tabs::
   .. tab:: Fortran


   .. tab:: C++
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_by_percentage_fortran.f90

         :language: fortran
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr.cpp
         :start-after: ! [doc example start]
         :language: cpp
         :end-before: ! [doc example end]
         :start-after: //! [doc example]
         :linenos:
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_prune_dense2csr_by_percentage_buffer_size()
-----------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage_buffer_size

rocsparse_prune_dense2csr_nnz_by_percentage()
---------------------------------------------

.. doxygenfunction:: rocsparse_sprune_dense2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_dense2csr_nnz_by_percentage


   .. tab:: Fortran
rocsparse_prune_dense2csr_by_percentage()

-----------------------------------------
      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_by_percentage_fortran.f90

         :language: fortran
.. doxygenfunction:: rocsparse_sprune_dense2csr_by_percentage
         :start-after: ! [doc example start]
  :outline:
         :end-before: ! [doc example end]
.. doxygenfunction:: rocsparse_dprune_dense2csr_by_percentage
         :linenos:

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_by_percentage.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_dense2csr_by_percentage_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_prune_csr2csr_by_percentage_buffer_size()
---------------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage_buffer_size

rocsparse_prune_csr2csr_nnz_by_percentage()
-------------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_nnz_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_nnz_by_percentage

rocsparse_prune_csr2csr_by_percentage()
---------------------------------------

.. doxygenfunction:: rocsparse_sprune_csr2csr_by_percentage
  :outline:
.. doxygenfunction:: rocsparse_dprune_csr2csr_by_percentage

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_by_percentage.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_rocsparse_prune_csr2csr_by_percentage_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

rocsparse_rocsparse_bsrpad_value()
----------------------------------

.. doxygenfunction:: rocsparse_sbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_dbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_cbsrpad_value
  :outline:
.. doxygenfunction:: rocsparse_zbsrpad_value
