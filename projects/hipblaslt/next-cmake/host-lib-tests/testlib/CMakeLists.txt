add_library(testlib STATIC)

add_subdirectory(include)
add_subdirectory(source)

target_link_libraries(testlib
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(testlib PRIVATE OpenMP::OpenMP_CXX)
endif()
