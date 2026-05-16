# rocBLAS CMake merge conflicts — develop additions, deferred

Generated during merge of `origin/develop` into
`users/davidd-amd/tensile-rocblas-cmake-update` (merge commit `65b3fbedef`).

These are the lines develop added inside the rocBLAS CMake files that this
branch's wholesale CMake refactor had removed or replaced. The merge kept
HEAD's version active and stashed develop's lines here for later review.

**Delete this file before opening the PR.**

## `projects/rocblas/CMakeLists.txt`

### Block at original line 192 (after marker removal)

```cmake

if( BUILD_WITH_HIPBLASLT )
  list( APPEND HIPBLASLT_DEFINES BUILD_WITH_HIPBLASLT )
  rocm_package_add_dependencies(DEPENDS "hipblaslt >= ${HIPBLASLT_VERSION}")
endif()

if(BUILD_CLIENTS_TESTS)
  enable_testing()
endif()

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
```

## `projects/rocblas/clients/CMakeLists.txt`

### Block at original line 62 (after marker removal)

```cmake
# Client tests/benchmarks with LINK_BLIS=OFF: default to ILP64 OpenBLAS via find_package(BLAS) + BLA_SIZEOF_INTEGER=8
# (TheRock / bundled OpenBLAS64). When OFF, Linux uses generic find_package(BLAS) (LP64); Windows uses legacy OPENBLAS_DIR + LP64.
if( (BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS) AND NOT LINK_BLIS )
  set( _rocblas_clients_ref_ilp64_default ON )
else()
  set( _rocblas_clients_ref_ilp64_default OFF )
endif()
option( ROCBLAS_CLIENTS_REFERENCE_ILP64
  "Use ILP64 OpenBLAS for client tests/benchmarks (find_package(BLAS) with BLA_SIZEOF_INTEGER=8). When OFF, Linux uses LP64 BLAS; Windows may use legacy OPENBLAS_DIR / LP64."
  ${_rocblas_clients_ref_ilp64_default} )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if ( NOT WIN32 )
    unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
    if (LINK_BLIS)
      set(BLAS_LIBRARY)
      set(BLIS_INCLUDE_DIR)

      # Try AOCL 5.x unified library (static only)
      if(NOT BLAS_LIBRARY)
        # Check build dependencies
        if(EXISTS "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
          set(BLAS_LIBRARY "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
          set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/aocl/install_package/include")
          message(STATUS "Found AOCL 5.x unified library at: ${BLAS_LIBRARY}")
```

### Block at original line 97 (after marker removal)

```cmake
      endif()
    elseif( ROCBLAS_CLIENTS_REFERENCE_ILP64 )
      # ILP64 OpenBLAS via CMake FindBLAS (TheRock finders: OpenBLAS64 when BLA_SIZEOF_INTEGER=8).
      set( WARN_NOT_ILP64_PREFERRED false )
      set( BLA_SIZEOF_INTEGER 8 )
      set( BLA_VENDOR OpenBLAS )
      find_package( BLAS REQUIRED )
      if( TARGET BLAS::BLAS )
        set( BLAS_LIBRARY BLAS::BLAS )
      else()
        set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
      endif()
      set( BLAS_INCLUDE_DIR "" )
      if( TARGET OpenBLAS64::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET OpenBLAS::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET BLAS::BLAS )
        get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
      endif()
      if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
        set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
      elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
      endif()
      set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
    else()
      find_package( BLAS REQUIRED )
      set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
    endif()
  else() # WIN32
    unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
    file(TO_CMAKE_PATH "C:/Program\ Files/AMD/AOCL-Windows" AOCL_ROOT)
    file(TO_CMAKE_PATH "${AOCL_ROOT}/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib" AOCL_BLAS_LIBRARY)
    if (LINK_BLIS AND EXISTS ${AOCL_BLAS_LIBRARY})
      set( BLAS_LIBRARY "-l\"${AOCL_ROOT}/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT\"" )
      set( BLIS_INCLUDE_DIR "${AOCL_ROOT}/amd-blis/include/ILP64" )
      set( BLIS_DEFINES BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS )
    elseif( NOT LINK_BLIS AND ROCBLAS_CLIENTS_REFERENCE_ILP64 )
      # ILP64 OpenBLAS via CMake FindBLAS (CMAKE_MODULE_PATH / CMAKE_PREFIX_PATH unchanged so TheRock finders apply).
      set( WARN_NOT_ILP64_PREFERRED false )
      set( BLA_SIZEOF_INTEGER 8 )
      set( BLA_VENDOR OpenBLAS )
      find_package( BLAS REQUIRED )
      if( TARGET BLAS::BLAS )
        set( BLAS_LIBRARY BLAS::BLAS )
      else()
        set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
      endif()
      set( BLAS_INCLUDE_DIR "" )
      if( TARGET OpenBLAS64::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET OpenBLAS::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET BLAS::BLAS )
        get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
      endif()
      if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
        set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
      elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
      endif()
      set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
    else()
      # Legacy: LP64 OpenBLAS via OPENBLAS_DIR / vcpkg (may warn about 64-bit stress tests).
      set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
      find_library( BLAS_LIBRARY libopenblas
                    PATHS ${OPENBLAS_DIR}/lib
                    NO_DEFAULT_PATH
                  )
      if (NOT BLAS_LIBRARY)
        find_package( OpenBLAS CONFIG REQUIRED )
        set( BLAS_LIBRARY OpenBLAS::OpenBLAS )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      set( WARN_NOT_ILP64_PREFERRED true )
```

### Block at original line 208 (after marker removal)

```cmake
  # One token for ROCBLAS_REFERENCE_LIB (runtime log stringification). BLAS_LIBRARY may be an
  # imported target or a CMake list when falling back to BLAS_LIBRARIES; the latter breaks
  # target_compile_definitions if passed through verbatim.
  if( TARGET "${BLAS_LIBRARY}" )
    set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
  elseif( BLAS_LIBRARIES )
    list( GET BLAS_LIBRARIES 0 ROCBLAS_REFERENCE_LIB_LABEL )
  else()
    set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
  endif()

  message(STATUS "Linking reference BLAS LIB: ${BLAS_LIBRARY}")
```

## `projects/rocblas/clients/common/CMakeLists.txt`

### Block at original line 73 (after marker removal)

```cmake
if ( ROCBLAS_REFERENCE_BLAS_DEFINES )
  target_compile_definitions( rocblas_clients_common PRIVATE ${ROCBLAS_REFERENCE_BLAS_DEFINES} )
  target_compile_definitions( rocblas_clients_testing_common PRIVATE ${ROCBLAS_REFERENCE_BLAS_DEFINES} )
endif()
target_compile_definitions( rocblas_clients_common PRIVATE ROCBLAS_REFERENCE_LIB=${ROCBLAS_REFERENCE_LIB_LABEL} )
```

## `projects/rocblas/clients/gtest/CMakeLists.txt`

### Block at original line 5 (after marker removal)

```cmake
# CTest categorization: see ROCBLAS_ENABLE_CTEST / ROCBLAS_HAS_CTEST_CATEGORIES in cmake/build-options.cmake
find_package( GTest REQUIRED )

if( BUILD_WITH_TENSILE )
  set(rocblas_tensile_test_source
      # Keep multiheaded_gtest.cpp first as we want
      # to allow it to create the first TensileHost !
      # Current GTESTs are run in the linking order as added with global variables
      multiheaded_gtest.cpp
      # use of tensile based functions (gemm)
      atomics_mode_gtest.cpp
      get_solutions_gtest.cpp

  )
endif()

set(rocblas_no_tensile_test_source
    # general
    rocblas_gtest_main.cpp
    rocblas_test.cpp
    asan_helpers_gtest.cpp
    general_gtest.cpp
    set_get_pointer_mode_gtest.cpp
    set_get_atomics_mode_gtest.cpp
    logging_mode_gtest.cpp
    ostream_threadsafety_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    # blas1
    blas1/asum_gtest.cpp
    blas1/axpy_gtest.cpp
    blas1/copy_gtest.cpp
    blas1/dot_gtest.cpp
    blas1/iamaxmin_gtest.cpp
    blas1/nrm2_gtest.cpp
    blas1/rot_gtest.cpp
    blas1/scal_gtest.cpp
    blas1/swap_gtest.cpp
    # blas1_ex
    blas_ex/axpy_ex_gtest.cpp
    blas_ex/dot_ex_gtest.cpp
    blas_ex/nrm2_ex_gtest.cpp
    blas_ex/rot_ex_gtest.cpp
    blas_ex/scal_ex_gtest.cpp
    # blas2
    blas2/trsv_gtest.cpp
    blas2/gbmv_gtest.cpp
    blas2/gemv_gtest.cpp
    blas2/hbmv_gtest.cpp
    blas2/hemv_gtest.cpp
    blas2/her_gtest.cpp
    blas2/her2_gtest.cpp
    blas2/hpmv_gtest.cpp
    blas2/hpr_gtest.cpp
    blas2/hpr2_gtest.cpp
    blas2/trmv_gtest.cpp
    blas2/tpmv_gtest.cpp
    blas2/tbmv_gtest.cpp
    blas2/tbsv_gtest.cpp
    blas2/tpsv_gtest.cpp
    blas2/ger_gtest.cpp
    blas2/geru_gtest.cpp
    blas2/gerc_gtest.cpp
    blas2/spr_gtest.cpp
    blas2/spr2_gtest.cpp
    blas2/syr_gtest.cpp
    blas2/syr2_gtest.cpp
    blas2/sbmv_gtest.cpp
    blas2/spmv_gtest.cpp
    blas2/symv_gtest.cpp
    # blas3 may use tensile or source gemm
    blas3/gemm_gtest.cpp
    blas_ex/gemm_ex_gtest.cpp
    blas3/symm_gtest.cpp
    blas3/hemm_gtest.cpp
    blas3/trsm_gtest.cpp
    blas3/trtri_gtest.cpp
    blas3/trmm_gtest.cpp
    blas3/syrk_gtest.cpp
    blas3/syrkx_gtest.cpp
    blas3/syr2k_gtest.cpp
    blas3/herk_gtest.cpp
    blas3/herkx_gtest.cpp
    blas3/her2k_gtest.cpp
    blas3/dgmm_gtest.cpp
    blas3/geam_gtest.cpp
    blas_ex/gemmt_gtest.cpp
    blas_ex/geam_ex_gtest.cpp
    blas_ex/syrk_ex_gtest.cpp
    blas_ex/herk_ex_gtest.cpp
  )

# Keep ${rocblas_tensile_test_source} first, so that multiheaded tests are the
# first to initialize Tensile.
```

### Block at original line 128 (after marker removal)

```cmake
if (CREATE_TEST_APP_LOCAL_DEPLOY AND WIN32)
  # HIP and tooling (OpenBLAS DLLs come from the same install CMake linked when using ILP64).
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/amd*.dll
    ${HIP_DIR}/bin/hiprt*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )

  if( NOT LINK_BLIS AND ROCBLAS_CLIENTS_REFERENCE_ILP64 )
    if( CMAKE_VERSION VERSION_GREATER_EQUAL "3.21" )
      add_custom_command(
        TARGET rocblas-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_RUNTIME_DLLS:rocblas-test>
          $<TARGET_FILE_DIR:rocblas-test>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying reference BLAS and other runtime DLLs for rocblas-test (ILP64)" )
    elseif( DEFINED OpenBLAS64_DIR AND EXISTS "${OpenBLAS64_DIR}/bin" )
      file( GLOB _openblas64_dlls LIST_DIRECTORIES OFF CONFIGURE_DEPENDS "${OpenBLAS64_DIR}/bin/*.dll" )
      foreach( file_i ${_openblas64_dlls})
        add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
      endforeach()
    else()
      message( WARNING "rocblas-test local deploy: CMake < 3.21 and OpenBLAS64_DIR not set; reference OpenBLAS DLLs may be missing from staging. Set OpenBLAS64_DIR or use CMake 3.21+." )
    endif()
  elseif( NOT LINK_BLIS )
    file( GLOB third_party_openblas_dlls LIST_DIRECTORIES OFF CONFIGURE_DEPENDS ${OPENBLAS_DIR}/bin/*.dll )
    foreach( file_i ${third_party_openblas_dlls})
      add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
    endforeach()
  endif()
```

### Block at original line 211 (after marker removal)

```cmake
# CTest categories: quick (= smoke), standard (= precheckin), comprehensive (= extended), stress (= weekly)
set(ROCBLAS_CTEST_INSTALL_FILE "")
if(ROCBLAS_HAS_CTEST_CATEGORIES)
  message(STATUS "rocBLAS: YAML-based test categorization")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
  # Install-time CTestTestfile for TheRock: relative paths so ctest can run from installed location
  set(ROCBLAS_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
  file(WRITE "${ROCBLAS_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by rocBLAS for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
  )
  apply_test_category_labels(
    rocblas-test
    "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
    "${PROJECT_BINARY_DIR}/staging"
    "${ROCBLAS_CTEST_INSTALL_FILE}"
  )
elseif(BUILD_CLIENTS_TESTS AND NOT ROCBLAS_ENABLE_CTEST)
  message(STATUS "rocBLAS: ROCBLAS_ENABLE_CTEST=OFF (CTest category labels not applied)")
```

### Block at original line 338 (after marker removal)

```cmake
# Install CTestTestfile.cmake to bin/rocblas/ for TheRock (relocatable tests; run ctest from that dir)
if(ROCBLAS_ENABLE_CTEST AND ROCBLAS_CTEST_INSTALL_FILE)
  install(
    FILES "${ROCBLAS_CTEST_INSTALL_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/rocblas"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
  )
```

## `projects/rocblas/library/CMakeLists.txt`

### Block at original line 17 (after marker removal)

```cmake
# Configure Windows version resource file
if(WIN32)
  string(TIMESTAMP ROCBLAS_COPYRIGHT_YEAR "%Y")
  configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/src/version.rc.in"
    "${PROJECT_BINARY_DIR}/version.rc" @ONLY )
endif()

```

## `projects/rocblas/library/src/CMakeLists.txt`

### Block at original line 44 (after marker removal)

```cmake
set( rocblas64_blas1_source
  src64/blas1/rocblas_iamax_iamin_kernels_64.cpp
  src64/blas1/rocblas_iamin_64.cpp
  src64/blas1/rocblas_iamin_batched_64.cpp
  src64/blas1/rocblas_iamin_strided_batched_64.cpp
  src64/blas1/rocblas_iamax_64.cpp
  src64/blas1/rocblas_iamax_batched_64.cpp
  src64/blas1/rocblas_iamax_strided_batched_64.cpp
  src64/blas1/rocblas_axpy_64.cpp
  src64/blas1/rocblas_axpy_kernels_64.cpp
  src64/blas1/rocblas_axpy_batched_64.cpp
  src64/blas1/rocblas_axpy_strided_batched_64.cpp
  src64/blas1/rocblas_copy_64.cpp
  src64/blas1/rocblas_copy_kernels_64.cpp
  src64/blas1/rocblas_copy_batched_64.cpp
  src64/blas1/rocblas_copy_strided_batched_64.cpp
  src64/blas1/rocblas_asum_64.cpp
  src64/blas1/rocblas_asum_nrm2_kernels_64.cpp
  src64/blas1/rocblas_asum_batched_64.cpp
  src64/blas1/rocblas_asum_strided_batched_64.cpp
  src64/blas1/rocblas_dot_64.cpp
  src64/blas1/rocblas_dot_kernels_64.cpp
  src64/blas1/rocblas_dot_strided_batched_64.cpp
  src64/blas1/rocblas_dot_batched_64.cpp
  src64/blas1/rocblas_nrm2_64.cpp
  src64/blas1/rocblas_nrm2_batched_64.cpp
  src64/blas1/rocblas_nrm2_strided_batched_64.cpp
  src64/blas1/rocblas_rot_64.cpp
  src64/blas1/rocblas_rot_kernels_64.cpp
  src64/blas1/rocblas_rot_batched_64.cpp
  src64/blas1/rocblas_rot_strided_batched_64.cpp
  src64/blas1/rocblas_rotg_64.cpp
  src64/blas1/rocblas_rotg_kernels_64.cpp
  src64/blas1/rocblas_rotg_batched_64.cpp
  src64/blas1/rocblas_rotg_strided_batched_64.cpp
  src64/blas1/rocblas_rotm_64.cpp
  src64/blas1/rocblas_rotm_kernels_64.cpp
  src64/blas1/rocblas_rotm_batched_64.cpp
  src64/blas1/rocblas_rotm_strided_batched_64.cpp
  src64/blas1/rocblas_rotmg_64.cpp
  src64/blas1/rocblas_rotmg_kernels_64.cpp
  src64/blas1/rocblas_rotmg_batched_64.cpp
  src64/blas1/rocblas_rotmg_strided_batched_64.cpp
  src64/blas1/rocblas_scal_64.cpp
  src64/blas1/rocblas_scal_kernels_64.cpp
  src64/blas1/rocblas_scal_batched_64.cpp
  src64/blas1/rocblas_scal_strided_batched_64.cpp
  src64/blas1/rocblas_swap_64.cpp
  src64/blas1/rocblas_swap_kernels_64.cpp
  src64/blas1/rocblas_swap_batched_64.cpp
  src64/blas1/rocblas_swap_strided_batched_64.cpp
)

set( subdir_src64_list
  ${rocblas64_ex_source}
  ${rocblas64_ex_source_no_tensile}
  ${rocblas64_blas3_source}
  ${rocblas64_blas3_source_no_tensile}
  ${rocblas64_blas2_source}
  ${rocblas64_blas1_source}
  ${rocblas64_auxiliary_source}
)

prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )

add_library( rocblas
  # ordered from generally slowest to compile to faster to increase parallelism
  ${rocblas_blas3_source}
  ${rocblas_blas3_source_no_tensile}
  ${rocblas_blas2_source}
  ${rocblas_ex_source}
  ${rocblas_ex_source_no_tensile}
  ${rocblas_blas1_source}
  ${relative_rocblas_headers_public}
  ${rocblas_auxiliary_source}
  ${subdir_src64_list}
)

# Add Windows version resource to library
if(WIN32 AND BUILD_SHARED_LIBS)
  target_sources(rocblas PRIVATE "${PROJECT_BINARY_DIR}/version.rc")
endif()

foreach( file ${subdir_src64_list} )
  SET_SOURCE_FILES_PROPERTIES( ${file} PROPERTIES COMPILE_DEFINITIONS ROCBLAS_INTERNAL_ILP64 )
endforeach()

message(STATUS "** NOTE: blas2/rocblas_ger_kernels.cpp is compiled with the verbose flag -v for QC purposes.")
SET_SOURCE_FILES_PROPERTIES( blas2/rocblas_ger_kernels.cpp PROPERTIES COMPILE_FLAGS "-v" )

#if( WIN32 )
#  set_target_properties(rocblas_fortran PROPERTIES LINKER_LANGUAGE CXX)
#  target_link_directories(rocblas_fortran PRIVATE "C:\\cygwin64\\lib\\gcc\\x86_64-pc-cygwin\\9.3.0" "C:\\cygwin64\\lib" "C:\\cygwin64\\lib\\w32api")
#endif( )

add_library( roc::rocblas ALIAS rocblas )

rocblas_library_settings( rocblas )

target_link_libraries( rocblas PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols


if( BUILD_WITH_TENSILE )

  if( BUILD_SHARED_LIBS )
    target_link_libraries( rocblas PRIVATE TensileHost )
  endif()
```

### Block at original line 226 (after marker removal)

```cmake

if( BUILD_WITH_TENSILE )
  if (WIN32)
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/rocblas" CACHE PATH "path to tensile library" )
  else()
    set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas" CACHE PATH "path to tensile library" )
  endif()
  # For ASAN package, Tensile library files(which are not shared libraries) are not required
  if( NOT ENABLE_ASAN_PACKAGING )
    if( BUILD_SHARED_LIBS )
      set( TENSILE_DATA_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME} )
    else()
      set( TENSILE_DATA_COMPONENT_NAME devel )
    endif()
    # TensileManifest.txt is a build-time verification artifact whose contents
    # are absolute paths into the build tree (e.g. /work/build-.../Tensile/library/...).
    # Nothing at runtime reads it (rocBLAS opens TensileLibrary_lazy_<arch>.dat
    # directly), and shipping it would also collide across kpack per-arch shards
    # since the filename has no arch suffix.
    rocm_install(
      DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
      DESTINATION ${ROCBLAS_TENSILE_LIBRARY_DIR}
      COMPONENT ${TENSILE_DATA_COMPONENT_NAME} # Use this cmake variable to be compatible with rocm-cmake 0.6 and 0.7
      PATTERN "TensileManifest.txt" EXCLUDE)
  endif()
endif()

if(NOT WIN32)
  if(RUN_HEADER_TESTING)
  # Compilation tests to ensure that header files work independently,
  # and that public header files work across several languages
  add_custom_command(
    TARGET rocblas
    POST_BUILD
    COMMAND ${CMAKE_HOME_DIRECTORY}/header_compilation_tests.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  endif()
endif()
```

