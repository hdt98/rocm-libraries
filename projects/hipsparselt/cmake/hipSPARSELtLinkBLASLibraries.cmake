# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(hipsparselt_link_blas_libraries target_name)
    if(HIPSPARSELT_ENABLE_THEROCK)
        # TheRock's cblas package provides OpenBLAS which includes BLAS, CBLAS, and LAPACK
        target_link_libraries(${target_name} PRIVATE cblas)
        message(STATUS "Linking ${target_name} with TheRock OpenBLAS (${OpenBLAS_DIR})")
    elseif(HIPSPARSELT_ENABLE_BLIS AND BLIS_FOUND)
        target_link_libraries(${target_name} PRIVATE BLIS::blis ${LAPACK_LIBRARIES})
        message(STATUS "Linking ${target_name} with BLIS and LAPACK")
    else()
        # LAPACK_LIBRARIES implicitly includes BLAS (find_package(LAPACK) calls find_package(BLAS))
        target_link_libraries(${target_name} PRIVATE ${LAPACK_LIBRARIES})
        message(STATUS "Linking ${target_name} with LAPACK (includes BLAS)")
    endif()
endfunction()
