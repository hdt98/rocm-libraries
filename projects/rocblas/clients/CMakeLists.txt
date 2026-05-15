# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(ROCBLAS_ENABLE_THREADS)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

add_subdirectory(common)

if(ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    if(ROCBLAS_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
        # If AOCL was found as a shared library, add its directory to RPATH
        # so client binaries can locate it at runtime without LD_LIBRARY_PATH.
        if(BLIS_LIB MATCHES "\\.so$")
            get_filename_component(_blis_lib_dir "${BLIS_LIB}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH "${_blis_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_blis_lib_dir}")
            unset(_blis_lib_dir)
        endif()
    else()
        find_package(BLAS REQUIRED)
        find_package(cblas)
    endif()

    if(ROCBLAS_ENABLE_AMD_SMI)
        find_package(amd_smi CONFIG REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_library(rocblas-testing-common STATIC)
    add_library(rocblas::testing-common ALIAS rocblas-testing-common)

    target_link_libraries(rocblas-testing-common
        PRIVATE
            $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
            $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
            hip::device
            roc::rocblas
            rocblas::clients-common
            rocblas::warnings
    )

    if(ROCBLAS_ENABLE_BLIS)
        target_link_libraries(rocblas-testing-common PRIVATE BLIS::BLIS)
        target_sources(rocblas-testing-common
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/common/blis_interface.cpp"
        )
    else()
        target_link_libraries(rocblas-testing-common PRIVATE ${BLAS_LIBRARIES})
        if(TARGET cblas)
            target_link_libraries(rocblas-testing-common PRIVATE cblas)
# === MERGE: develop additions, resolve later ===
# # Client tests/benchmarks with LINK_BLIS=OFF: default to ILP64 OpenBLAS via find_package(BLAS) + BLA_SIZEOF_INTEGER=8
# # (TheRock / bundled OpenBLAS64). When OFF, Linux uses generic find_package(BLAS) (LP64); Windows uses legacy OPENBLAS_DIR + LP64.
# if( (BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS) AND NOT LINK_BLIS )
#   set( _rocblas_clients_ref_ilp64_default ON )
# else()
#   set( _rocblas_clients_ref_ilp64_default OFF )
# endif()
# option( ROCBLAS_CLIENTS_REFERENCE_ILP64
#   "Use ILP64 OpenBLAS for client tests/benchmarks (find_package(BLAS) with BLA_SIZEOF_INTEGER=8). When OFF, Linux uses LP64 BLAS; Windows may use legacy OPENBLAS_DIR / LP64."
#   ${_rocblas_clients_ref_ilp64_default} )
#
# if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
#   if ( NOT WIN32 )
#     unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
#     if (LINK_BLIS)
#       set(BLAS_LIBRARY)
#       set(BLIS_INCLUDE_DIR)
#
#       # Try AOCL 5.x unified library (static only)
#       if(NOT BLAS_LIBRARY)
#         # Check build dependencies
#         if(EXISTS "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
#           set(BLAS_LIBRARY "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
#           set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/aocl/install_package/include")
#           message(STATUS "Found AOCL 5.x unified library at: ${BLAS_LIBRARY}")
# === END MERGE ===
        endif()
        if(BLAS_INCLUDE_DIRS)
            target_include_directories(rocblas-testing-common SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS})
        endif()
    endif()

    if(ROCBLAS_ENABLE_FORTRAN)
        target_link_libraries(rocblas-testing-common PRIVATE rocblas-fortran-client)
# === MERGE: develop additions, resolve later ===
#       endif()
#     elseif( ROCBLAS_CLIENTS_REFERENCE_ILP64 )
#       # ILP64 OpenBLAS via CMake FindBLAS (TheRock finders: OpenBLAS64 when BLA_SIZEOF_INTEGER=8).
#       set( WARN_NOT_ILP64_PREFERRED false )
#       set( BLA_SIZEOF_INTEGER 8 )
#       set( BLA_VENDOR OpenBLAS )
#       find_package( BLAS REQUIRED )
#       if( TARGET BLAS::BLAS )
#         set( BLAS_LIBRARY BLAS::BLAS )
#       else()
#         set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
#       endif()
#       set( BLAS_INCLUDE_DIR "" )
#       if( TARGET OpenBLAS64::OpenBLAS )
#         get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
#       elseif( TARGET OpenBLAS::OpenBLAS )
#         get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
#       elseif( TARGET BLAS::BLAS )
#         get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
#       endif()
#       if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
#         set( BLAS_INCLUDE_DIR "" )
#       endif()
#       if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
#         set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
#       endif()
#       if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
#         set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
#       elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
#         set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
#       endif()
#       set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
#     else()
#       find_package( BLAS REQUIRED )
#       set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
#     endif()
#   else() # WIN32
#     unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
#     file(TO_CMAKE_PATH "C:/Program\ Files/AMD/AOCL-Windows" AOCL_ROOT)
#     file(TO_CMAKE_PATH "${AOCL_ROOT}/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib" AOCL_BLAS_LIBRARY)
#     if (LINK_BLIS AND EXISTS ${AOCL_BLAS_LIBRARY})
#       set( BLAS_LIBRARY "-l\"${AOCL_ROOT}/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT\"" )
#       set( BLIS_INCLUDE_DIR "${AOCL_ROOT}/amd-blis/include/ILP64" )
#       set( BLIS_DEFINES BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS )
#     elseif( NOT LINK_BLIS AND ROCBLAS_CLIENTS_REFERENCE_ILP64 )
#       # ILP64 OpenBLAS via CMake FindBLAS (CMAKE_MODULE_PATH / CMAKE_PREFIX_PATH unchanged so TheRock finders apply).
#       set( WARN_NOT_ILP64_PREFERRED false )
#       set( BLA_SIZEOF_INTEGER 8 )
#       set( BLA_VENDOR OpenBLAS )
#       find_package( BLAS REQUIRED )
#       if( TARGET BLAS::BLAS )
#         set( BLAS_LIBRARY BLAS::BLAS )
#       else()
#         set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
#       endif()
#       set( BLAS_INCLUDE_DIR "" )
#       if( TARGET OpenBLAS64::OpenBLAS )
#         get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
#       elseif( TARGET OpenBLAS::OpenBLAS )
#         get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
#       elseif( TARGET BLAS::BLAS )
#         get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
#       endif()
#       if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
#         set( BLAS_INCLUDE_DIR "" )
#       endif()
#       if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
#         set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
#       endif()
#       if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
#         set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
#       elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
#         set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
#       endif()
#       set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
#     else()
#       # Legacy: LP64 OpenBLAS via OPENBLAS_DIR / vcpkg (may warn about 64-bit stress tests).
#       set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
#       find_library( BLAS_LIBRARY libopenblas
#                     PATHS ${OPENBLAS_DIR}/lib
#                     NO_DEFAULT_PATH
#                   )
#       if (NOT BLAS_LIBRARY)
#         find_package( OpenBLAS CONFIG REQUIRED )
#         set( BLAS_LIBRARY OpenBLAS::OpenBLAS )
#         set( BLAS_INCLUDE_DIR "" )
#       endif()
#       set( WARN_NOT_ILP64_PREFERRED true )
# === END MERGE ===
    endif()

    target_include_directories(rocblas-testing-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas1"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas2"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas3"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas_ex"
            "${PROJECT_SOURCE_DIR}/library/src"
            "${PROJECT_SOURCE_DIR}/library/src/include"
    )

    target_compile_definitions(rocblas-testing-common
        PRIVATE
            GOOGLE_TEST
            ROCBLAS_INTERNAL_API
            ROCBLAS_NO_DEPRECATED_WARNINGS
            ROCM_USE_FLOAT16
            __HIP_HCC_COMPAT_MODE__=1
    )
# === MERGE: develop additions, resolve later ===
#   # One token for ROCBLAS_REFERENCE_LIB (runtime log stringification). BLAS_LIBRARY may be an
#   # imported target or a CMake list when falling back to BLAS_LIBRARIES; the latter breaks
#   # target_compile_definitions if passed through verbatim.
#   if( TARGET "${BLAS_LIBRARY}" )
#     set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
#   elseif( BLAS_LIBRARIES )
#     list( GET BLAS_LIBRARIES 0 ROCBLAS_REFERENCE_LIB_LABEL )
#   else()
#     set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
#   endif()
#
#   message(STATUS "Linking reference BLAS LIB: ${BLAS_LIBRARY}")
# === END MERGE ===

    if(ROCBLAS_ENABLE_TENSILE OR ROCBLAS_ENABLE_HIPBLASLT)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                ROCBLAS_TENSILE_LAZY_LOAD=1
                ROCBLAS_TENSILE_SEPARATE_ARCH=1
        )
    endif()

    if(ROCBLAS_ENABLE_TENSILE)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                BUILD_WITH_TENSILE
        )
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                BUILD_WITH_HIPBLASLT
        )
        target_link_libraries(rocblas-testing-common PRIVATE roc::hipblaslt)
    endif()

    if(ROCBLAS_ENABLE_ASAN)
        rocblas_target_configure_sanitizers(rocblas-testing-common PRIVATE)
    endif()

    target_sources(rocblas-testing-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/common/cblas_interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/gtest_helpers.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/client_utility.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_matrix.cpp"
    )

    add_subdirectory(common/blas1)
    add_subdirectory(common/blas2)
    add_subdirectory(common/blas3)
    add_subdirectory(common/blas_ex)
    add_subdirectory(gtest)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_smoke.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_smoke.yaml" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_extras.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_extras.yaml" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.py" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.py" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.xml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.xml" COPYONLY)
endif()

if(ROCBLAS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ROCBLAS_ENABLE_OPENMP)
    add_subdirectory(openmp-support)

    if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
        target_link_libraries(rocblas-test PRIVATE rocblas::clients-openmp)
    endif()

    if(ROCBLAS_ENABLE_BENCHMARKS)
        target_link_libraries(rocblas-bench PRIVATE rocblas::clients-openmp)
        if(ROCBLAS_ENABLE_TENSILE)
            target_link_libraries(rocblas-gemm-tune PRIVATE rocblas::clients-openmp)
        endif()
    endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_general.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_clients_readme.txt" COPYONLY)

if(NOT ROCBLAS_ENABLE_FORTRAN)
  target_compile_definitions(rocblas-testing-common PUBLIC CLIENTS_NO_FORTRAN)
  target_compile_definitions(rocblas-clients-common PUBLIC CLIENTS_NO_FORTRAN)
endif()

# Install rules for test support files
if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
  rocm_install(
    FILES
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_smoke.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_extras.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.xml"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT tests
  )
  rocm_install(
    PROGRAMS "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.py"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT tests
  )
endif()
