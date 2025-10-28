# This script reads the test categories YAML file and applies labels to CTest

# Function to apply category labels to discovered GTest tests
function(apply_test_category_labels target_name yaml_file)
    # Execute the Python script to generate CMake code
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/ctest/hipblas/parse_test_categories.py ${yaml_file}
        OUTPUT_VARIABLE CMAKE_CATEGORY_CODE
        ERROR_VARIABLE PARSE_ERROR
        RESULT_VARIABLE PARSE_RESULT
    )

    if(NOT PARSE_RESULT EQUAL 0)
        message(WARNING "Failed to parse test categories YAML: ${PARSE_ERROR}")
        return()
    endif()

    # Write the generated CMake code to a file and include it
    set(CATEGORY_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake")
    file(WRITE "${CATEGORY_CMAKE}" "${CMAKE_CATEGORY_CODE}")

    message(STATUS "Generated test category configuration: ${CATEGORY_CMAKE}")

    # Include and execute the generated CMake code
    include("${CATEGORY_CMAKE}")
endfunction()

# Simplified version that works without Python dependency
function(apply_hardcoded_test_categories)
	  # Add category-specific test runners using GTest filters
  # These allow running tests by category using ctest -L <category>

  # Auxil tests
  add_test(
    NAME auto-hipblas-auxil-suite
    COMMAND hipblas-test --gtest_filter="*auxiliary*:*set_get*:*auxil*"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
  )
  set_tests_properties(auto-hipblas-auxil-suite PROPERTIES
    LABELS "auxil;basic;suite"
    TIMEOUT 120
  )

  # BLAS1 tests
  add_test(
    NAME hipblas-blas1-suite
    COMMAND hipblas-test --gtest_filter="*asum*:*axpy*:*copy*:*dot*:*nrm2*:*rot*:*scal*:*swap*:*iamax*:*iamin*"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
  )
  set_tests_properties(hipblas-blas1-suite PROPERTIES
    LABELS "blas1;level1;suite"
    TIMEOUT 600
  )

  # BLAS2 tests
  add_test(
    NAME hipblas-blas2-suite
    COMMAND hipblas-test --gtest_filter="*gbmv*:*gemv*:*ger*:*hbmv*:*hemv*:*her*:*hpmv*:*hpr*:*sbmv*:*spmv*:*spr*:*symv*:*syr*:*tbmv*:*tbsv*:*tpmv*:*tpsv*:*trmv*:*trsv*"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
  )
  set_tests_properties(hipblas-blas2-suite PROPERTIES
    LABELS "blas2;level2;suite"
    TIMEOUT 900
  )

  # BLAS3 tests
  add_test(
    NAME hipblas-blas3-suite
    COMMAND hipblas-test --gtest_filter="*dgmm*:*geam*:*gemm*:*hemm*:*herk*:*her2k*:*herkx*:*symm*:*syrk*:*syr2k*:*syrkx*:*trmm*:*trsm*:*trtri*"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
  )
  set_tests_properties(hipblas-blas3-suite PROPERTIES
    LABELS "blas3;level3;suite"
    TIMEOUT 1200
  )

  # BLAS_EX tests
  add_test(
    NAME hipblas-blas_ex-suite
    COMMAND hipblas-test --gtest_filter="*_ex*"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
  )
  set_tests_properties(hipblas-blas_ex-suite PROPERTIES
    LABELS "blas_ex;level4;suite"
    TIMEOUT 900
  )

  # Solver tests (if enabled)
  if(BUILD_WITH_SOLVER)
    add_test(
      NAME hipblas-solver-suite
      COMMAND hipblas-test --gtest_filter="*gels*:*geqrf*:*getrf*:*getri*:*getrs*:*solver*"
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging
    )
    set_tests_properties(hipblas-solver-suite PROPERTIES
      LABELS "solver;level5;suite"
      TIMEOUT 1500
    )
  endif()
endfunction()

