# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
# === MERGE: develop additions, resolve later ===
# # CTest categorization: see ROCBLAS_ENABLE_CTEST / ROCBLAS_HAS_CTEST_CATEGORIES in cmake/build-options.cmake
# find_package( GTest REQUIRED )
#
# if( BUILD_WITH_TENSILE )
#   set(rocblas_tensile_test_source
#       # Keep multiheaded_gtest.cpp first as we want
#       # to allow it to create the first TensileHost !
#       # Current GTESTs are run in the linking order as added with global variables
#       multiheaded_gtest.cpp
#       # use of tensile based functions (gemm)
#       atomics_mode_gtest.cpp
#       get_solutions_gtest.cpp
#
#   )
# endif()
#
# set(rocblas_no_tensile_test_source
#     # general
#     rocblas_gtest_main.cpp
#     rocblas_test.cpp
#     asan_helpers_gtest.cpp
#     general_gtest.cpp
#     set_get_pointer_mode_gtest.cpp
#     set_get_atomics_mode_gtest.cpp
#     logging_mode_gtest.cpp
#     ostream_threadsafety_gtest.cpp
#     set_get_vector_gtest.cpp
#     set_get_matrix_gtest.cpp
#     # blas1
#     blas1/asum_gtest.cpp
#     blas1/axpy_gtest.cpp
#     blas1/copy_gtest.cpp
#     blas1/dot_gtest.cpp
#     blas1/iamaxmin_gtest.cpp
#     blas1/nrm2_gtest.cpp
#     blas1/rot_gtest.cpp
#     blas1/scal_gtest.cpp
#     blas1/swap_gtest.cpp
#     # blas1_ex
#     blas_ex/axpy_ex_gtest.cpp
#     blas_ex/dot_ex_gtest.cpp
#     blas_ex/nrm2_ex_gtest.cpp
#     blas_ex/rot_ex_gtest.cpp
#     blas_ex/scal_ex_gtest.cpp
#     # blas2
#     blas2/trsv_gtest.cpp
#     blas2/gbmv_gtest.cpp
#     blas2/gemv_gtest.cpp
#     blas2/hbmv_gtest.cpp
#     blas2/hemv_gtest.cpp
#     blas2/her_gtest.cpp
#     blas2/her2_gtest.cpp
#     blas2/hpmv_gtest.cpp
#     blas2/hpr_gtest.cpp
#     blas2/hpr2_gtest.cpp
#     blas2/trmv_gtest.cpp
#     blas2/tpmv_gtest.cpp
#     blas2/tbmv_gtest.cpp
#     blas2/tbsv_gtest.cpp
#     blas2/tpsv_gtest.cpp
#     blas2/ger_gtest.cpp
#     blas2/geru_gtest.cpp
#     blas2/gerc_gtest.cpp
#     blas2/spr_gtest.cpp
#     blas2/spr2_gtest.cpp
#     blas2/syr_gtest.cpp
#     blas2/syr2_gtest.cpp
#     blas2/sbmv_gtest.cpp
#     blas2/spmv_gtest.cpp
#     blas2/symv_gtest.cpp
#     # blas3 may use tensile or source gemm
#     blas3/gemm_gtest.cpp
#     blas_ex/gemm_ex_gtest.cpp
#     blas3/symm_gtest.cpp
#     blas3/hemm_gtest.cpp
#     blas3/trsm_gtest.cpp
#     blas3/trtri_gtest.cpp
#     blas3/trmm_gtest.cpp
#     blas3/syrk_gtest.cpp
#     blas3/syrkx_gtest.cpp
#     blas3/syr2k_gtest.cpp
#     blas3/herk_gtest.cpp
#     blas3/herkx_gtest.cpp
#     blas3/her2k_gtest.cpp
#     blas3/dgmm_gtest.cpp
#     blas3/geam_gtest.cpp
#     blas_ex/gemmt_gtest.cpp
#     blas_ex/geam_ex_gtest.cpp
#     blas_ex/syrk_ex_gtest.cpp
#     blas_ex/herk_ex_gtest.cpp
#   )
#
# # Keep ${rocblas_tensile_test_source} first, so that multiheaded tests are the
# # first to initialize Tensile.
# === END MERGE ===

add_executable(rocblas-test)

if(ROCBLAS_ENABLE_BLIS)
    target_link_libraries(rocblas-test PRIVATE BLIS::BLIS)
else()
    target_link_libraries(rocblas-test PRIVATE ${BLAS_LIBRARIES})
    if(TARGET cblas)
        target_link_libraries(rocblas-test PRIVATE cblas)
    endif()
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(rocblas-test SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
endif()

target_link_libraries(rocblas-test
    PRIVATE
        $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
        $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
        hip::device
        roc::rocblas
        rocblas::testing-common
        rocblas::warnings
)

if(ROCBLAS_ENABLE_THREADS)
    target_link_libraries(rocblas-test PRIVATE Threads::Threads)
# === MERGE: develop additions, resolve later ===
# if (CREATE_TEST_APP_LOCAL_DEPLOY AND WIN32)
#   # HIP and tooling (OpenBLAS DLLs come from the same install CMake linked when using ILP64).
#   file( GLOB third_party_dlls
#     LIST_DIRECTORIES OFF
#     CONFIGURE_DEPENDS
#     ${HIP_DIR}/bin/amd*.dll
#     ${HIP_DIR}/bin/hiprt*.dll
#     ${HIP_DIR}/bin/hipinfo.exe
#     ${CMAKE_SOURCE_DIR}/rtest.*
#     C:/Windows/System32/libomp140*.dll
#   )
#   foreach( file_i ${third_party_dlls})
#     add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
#   endforeach( file_i )
#
#   if( NOT LINK_BLIS AND ROCBLAS_CLIENTS_REFERENCE_ILP64 )
#     if( CMAKE_VERSION VERSION_GREATER_EQUAL "3.21" )
#       add_custom_command(
#         TARGET rocblas-test POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#           $<TARGET_RUNTIME_DLLS:rocblas-test>
#           $<TARGET_FILE_DIR:rocblas-test>
#         COMMAND_EXPAND_LISTS
#         COMMENT "Copying reference BLAS and other runtime DLLs for rocblas-test (ILP64)" )
#     elseif( DEFINED OpenBLAS64_DIR AND EXISTS "${OpenBLAS64_DIR}/bin" )
#       file( GLOB _openblas64_dlls LIST_DIRECTORIES OFF CONFIGURE_DEPENDS "${OpenBLAS64_DIR}/bin/*.dll" )
#       foreach( file_i ${_openblas64_dlls})
#         add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
#       endforeach()
#     else()
#       message( WARNING "rocblas-test local deploy: CMake < 3.21 and OpenBLAS64_DIR not set; reference OpenBLAS DLLs may be missing from staging. Set OpenBLAS64_DIR or use CMake 3.21+." )
#     endif()
#   elseif( NOT LINK_BLIS )
#     file( GLOB third_party_openblas_dlls LIST_DIRECTORIES OFF CONFIGURE_DEPENDS ${OPENBLAS_DIR}/bin/*.dll )
#     foreach( file_i ${third_party_openblas_dlls})
#       add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
#     endforeach()
#   endif()
# === END MERGE ===
endif()

target_include_directories(rocblas-test
    PRIVATE
        "${PROJECT_SOURCE_DIR}/clients/common"
        "${PROJECT_SOURCE_DIR}/clients/include"
        "${PROJECT_SOURCE_DIR}/clients/include/blas1"
        "${PROJECT_SOURCE_DIR}/clients/include/blas_ex"
        "${PROJECT_SOURCE_DIR}/library/src"
        "${PROJECT_SOURCE_DIR}/library/src/include"
)

target_compile_definitions(rocblas-test
    PRIVATE
        GOOGLE_TEST
        ROCBLAS_INTERNAL_API
        ROCBLAS_NO_DEPRECATED_WARNINGS
        ROCM_USE_FLOAT16
        __HIP_HCC_COMPAT_MODE__=1
)

if(ROCBLAS_ENABLE_TENSILE)
    target_sources(rocblas-test
        PRIVATE
            # Keep multiheaded_gtest.cpp first to initialize TensileHost
            "${CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.cpp"
    )

    target_compile_definitions(rocblas-test
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas-test
        PRIVATE
            BUILD_WITH_HIPBLASLT
    )
# === MERGE: develop additions, resolve later ===
# # CTest categories: quick (= smoke), standard (= precheckin), comprehensive (= extended), stress (= weekly)
# set(ROCBLAS_CTEST_INSTALL_FILE "")
# if(ROCBLAS_HAS_CTEST_CATEGORIES)
#   message(STATUS "rocBLAS: YAML-based test categorization")
#   file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
#   # Install-time CTestTestfile for TheRock: relative paths so ctest can run from installed location
#   set(ROCBLAS_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
#   file(WRITE "${ROCBLAS_CTEST_INSTALL_FILE}"
# [=[
# # This is a test file generated by rocBLAS for install time.
# # Tests are defined with relative paths to work in the installed location.
# ]=]
#   )
#   apply_test_category_labels(
#     rocblas-test
#     "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
#     "${PROJECT_BINARY_DIR}/staging"
#     "${ROCBLAS_CTEST_INSTALL_FILE}"
#   )
# elseif(BUILD_CLIENTS_TESTS AND NOT ROCBLAS_ENABLE_CTEST)
#   message(STATUS "rocBLAS: ROCBLAS_ENABLE_CTEST=OFF (CTest category labels not applied)")
# === END MERGE ===
endif()

target_sources(rocblas-test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/asan_helpers_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/general_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/asum_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/axpy_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/copy_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/dot_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/iamaxmin_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/nrm2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/rot_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/scal_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/swap_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/axpy_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/dot_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/nrm2_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rot_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/scal_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/trsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gemv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hemv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/her_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/her2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/trmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tpmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tbsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tpsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/ger_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/geru_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gerc_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/syr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/syr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/sbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/symv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/gemm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/symm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/hemm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trsm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trtri_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trmm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syrk_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syrkx_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syr2k_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/herk_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/herkx_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/her2k_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/dgmm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/geam_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemm_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemmt_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/syrk_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/geam_ex_gtest.cpp"
)

set_target_properties(rocblas-test
    PROPERTIES
        IMPORT_PREFIX ""
        IMPORT_SUFFIX ".lib"
        LINKER_LANGUAGE CXX
)

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
set(ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gtest.data")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(
    OUTPUT "${ROCBLAS_TEST_DATA}"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/clients/common/rocblas_gentest.py"
            -I "${PROJECT_SOURCE_DIR}/clients/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
            -o "${ROCBLAS_TEST_DATA}"
    DEPENDS 
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
        "${PROJECT_SOURCE_DIR}/clients/common/rocblas_gentest.py"
    COMMENT "Generating test data: ${ROCBLAS_TEST_DATA}"
    VERBATIM
)

add_custom_target(rocblas-test-data ALL
    DEPENDS "${ROCBLAS_TEST_DATA}"
)

add_dependencies(rocblas-test rocblas-test-data)

if(ROCBLAS_ENABLE_ASAN)
    rocblas_target_configure_sanitizers(rocblas-test PRIVATE)
# === MERGE: develop additions, resolve later ===
# # Install CTestTestfile.cmake to bin/rocblas/ for TheRock (relocatable tests; run ctest from that dir)
# if(ROCBLAS_ENABLE_CTEST AND ROCBLAS_CTEST_INSTALL_FILE)
#   install(
#     FILES "${ROCBLAS_CTEST_INSTALL_FILE}"
#     DESTINATION "${CMAKE_INSTALL_BINDIR}/rocblas"
#     COMPONENT tests
#     RENAME "CTestTestfile.cmake"
#   )
# === END MERGE ===
endif()
