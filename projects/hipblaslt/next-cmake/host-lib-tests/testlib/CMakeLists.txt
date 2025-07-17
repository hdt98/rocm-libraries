set(TESTLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/HostLibraryTests/testlib")

# add_library(hostlib-tests STATIC)

target_include_directories(hostlib-tests
    PRIVATE
        # "${TESTLIB_SRC_DIR}/include/GEMMKernelTest_impl.hpp"
        # "${TESTLIB_SRC_DIR}/include/GEMMKernelTest.hpp"
        # "${TESTLIB_SRC_DIR}/include/RunGEMMKernelTest_impl.hpp"
        # "${TESTLIB_SRC_DIR}/include/RunGEMMKernelTest.hpp"
        # "${TESTLIB_SRC_DIR}/include/TestData.hpp"
        # "${TESTLIB_SRC_DIR}/include/TestUtils.hpp"
        "${TESTLIB_SRC_DIR}/include"
)
target_sources(hostlib-tests
    PRIVATE
        "${TESTLIB_SRC_DIR}/source/TestData.cpp"
)


target_compile_definitions(hostlib-tests
    PRIVATE
        BOOST_ERROR_CODE_HEADER_ONLY
)

target_link_libraries(hostlib-tests
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hostlib-tests PRIVATE OpenMP::OpenMP_CXX)
endif()
