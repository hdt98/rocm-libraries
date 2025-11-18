.. meta::
  :description: rocSPARSE level 1 functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, level 1 functions

.. _rocsparse_level1_functions_:

********************************************************************
Sparse level 1 functions
********************************************************************

The sparse level 1 routines describe operations between a vector in sparse format and a vector in dense format.
This section describes all rocSPARSE level 1 sparse linear algebra functions.

rocsparse_axpyi()
-----------------

.. doxygenfunction:: rocsparse_saxpyi
  :outline:
.. doxygenfunction:: rocsparse_daxpyi
  :outline:
.. doxygenfunction:: rocsparse_caxpyi
  :outline:
.. doxygenfunction:: rocsparse_zaxpyi

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_axpyi.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_axpyi_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_axpyi_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_doti()
----------------

.. doxygenfunction:: rocsparse_sdoti
  :outline:
.. doxygenfunction:: rocsparse_ddoti
  :outline:
.. doxygenfunction:: rocsparse_cdoti
  :outline:
.. doxygenfunction:: rocsparse_zdoti

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_doti.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_doti_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_doti_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_dotci()
-----------------

.. doxygenfunction:: rocsparse_cdotci
  :outline:
.. doxygenfunction:: rocsparse_zdotci

rocsparse_gthr()
----------------

.. doxygenfunction:: rocsparse_sgthr
  :outline:
.. doxygenfunction:: rocsparse_dgthr
  :outline:
.. doxygenfunction:: rocsparse_cgthr
  :outline:
.. doxygenfunction:: rocsparse_zgthr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_gthr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_gthr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_gthr_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_gthrz()
-----------------

.. doxygenfunction:: rocsparse_sgthrz
  :outline:
.. doxygenfunction:: rocsparse_dgthrz
  :outline:
.. doxygenfunction:: rocsparse_cgthrz
  :outline:
.. doxygenfunction:: rocsparse_zgthrz

rocsparse_roti()
----------------

.. doxygenfunction:: rocsparse_sroti
  :outline:
.. doxygenfunction:: rocsparse_droti

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_roti.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_roti_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_roti_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:

rocsparse_sctr()
----------------

.. doxygenfunction:: rocsparse_ssctr
  :outline:
.. doxygenfunction:: rocsparse_dsctr
  :outline:
.. doxygenfunction:: rocsparse_csctr
  :outline:
.. doxygenfunction:: rocsparse_zsctr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_sctr.cpp
         :language: cpp
         :start-after: //! [doc example]
         :end-before: //! [doc example]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_sctr_c.c
         :language: c
         :start-after: /*! [doc example] */
         :end-before: /*! [doc example] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_rocsparse_sctr_fortran.f90
         :language: fortran
         :start-after: ! [doc example]
         :end-before: ! [doc example]
         :linenos:
