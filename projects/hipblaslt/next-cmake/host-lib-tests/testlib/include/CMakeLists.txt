set(TESTLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tensilelite/HostLibraryTests/testlib")

target_include_directories(testlib
    PRIVATE
        # "${TESTLIB_SRC_DIR}/include/GEMMKernelTest_impl.hpp"
        # "${TESTLIB_SRC_DIR}/include/GEMMKernelTest.hpp"
        # "${TESTLIB_SRC_DIR}/include/RunGEMMKernelTest_impl.hpp"
        # "${TESTLIB_SRC_DIR}/include/RunGEMMKernelTest.hpp"
        # "${TESTLIB_SRC_DIR}/include/TestData.hpp"
        # "${TESTLIB_SRC_DIR}/include/TestUtils.hpp"
        "${TESTLIB_SRC_DIR}/include"
)

target_compile_definitions(testlib
    PRIVATE
        BOOST_ERROR_CODE_HEADER_ONLY
)
