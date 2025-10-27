# This script reads the test categories YAML file and applies labels to CTest

# Function to apply category labels to discovered GTest tests
function(apply_test_category_labels target_name yaml_file)
    # Execute the Python script to generate CMake code
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../parse_test_categories.py ${CMAKE_CURRENT_SOURCE_DIR}/../test_categories.yaml
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

    add_test(NAME rocfft_all_tests
           COMMAND rocfft-test
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  # Set test properties
  set_tests_properties(rocfft_all_tests PROPERTIES
    TIMEOUT 3600  # 1 hour timeout
    LABELS "accuracy;rocfft"
  )

  # Add specific test suites that can be run individually
  # Power of 2 tests
  add_test(NAME rocfft_pow2_1D
           COMMAND rocfft-test --gtest_filter="pow2_1D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_pow2_1D PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;pow2_1D")

  add_test(NAME rocfft_pow2_2D
           COMMAND rocfft-test --gtest_filter="pow2_2D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_pow2_2D PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;pow2_2D")

  add_test(NAME rocfft_pow2_3D
           COMMAND rocfft-test --gtest_filter="pow2_3D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_pow2_3D PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;pow2_3D")

  # Prime number tests
  add_test(NAME rocfft_prime_1D
           COMMAND rocfft-test --gtest_filter="prime_1D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_prime_1D PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;prime_1D" )

  # Mixed radix tests
  add_test(NAME rocfft_mix_1D
           COMMAND rocfft-test --gtest_filter="mix_1D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_mix_1D PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;mix_1D" )

  # Callback tests
  add_test(NAME rocfft_callback_tests
           COMMAND rocfft-test --gtest_filter="*callback*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_callback_tests PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;callback" )

  # Unit tests
  add_test(NAME rocfft_unit_tests
           COMMAND rocfft-test --gtest_filter="unit_test*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_unit_tests PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;unit_test" )

  # Hermitian tests
  add_test(NAME rocfft_hermitian_tests
           COMMAND rocfft-test --gtest_filter="hermitian*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})
  set_tests_properties(rocfft_hermitian_tests PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;hermitian" )

  # Set properties for all individual test suites
  #set_tests_properties(
  #  rocfft_pow2_2D rocfft_pow2_3D
  #  rocfft_prime_1D rocfft_mix_1D rocfft_callback_tests
  #  rocfft_unit_tests rocfft_hermitian_tests
  #  PROPERTIES
  #  TIMEOUT 1800  # 30 minutes timeout for individual suites
  #  LABELS "rocfft"
  #)

  # Quick smoke test with limited parameters
  add_test(NAME rocfft_smoke_test
           COMMAND rocfft-test --gtest_filter="pow2_1D*"
           WORKING_DIRECTORY ${TESTS_OUT_DIR})

  set_tests_properties(rocfft_smoke_test PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    LABELS "rocfft;smoke;quick"
  )

endfunction()

