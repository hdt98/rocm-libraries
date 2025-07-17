set(TESTLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tensilelite/HostLibraryTests/testlib")

target_sources(testlib
    PRIVATE
        "${TESTLIB_SRC_DIR}/source/TestData.cpp"
)
