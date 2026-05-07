# Fetch and build the third-party dependencies MIOpen requires when no
# system / superbuild copies are available. Activated by MIOPEN_STANDALONE_BUILD
# from projects/MIOpen/CMakeLists.txt. All upstream URLs and hashes are taken
# from D:/develop/src/TheRock/third-party (and .../sysdeps/common). Upstream
# sources are preferred; AMD bucket URLs are noted in comments as fallback.
#
# Each dep is declared with OVERRIDE_FIND_PACKAGE so the existing
# find_package(<name> [REQUIRED]) calls in projects/MIOpen/CMakeLists.txt
# resolve transparently against the populated content.
#
# Requires CMake 3.24 (OVERRIDE_FIND_PACKAGE). MIOpen's top-level
# cmake_minimum_required stays at 3.15; this floor only applies in standalone
# mode where we know we control the environment.

cmake_minimum_required(VERSION 3.24)
include(FetchContent)

message(STATUS "MIOpen: standalone build — fetching third-party dependencies via FetchContent")

# Suppress sub-project noise: only CMake's own one-liner per dep.
set(FETCHCONTENT_QUIET ON)

# Per-dep build options must be set before MakeAvailable.
set(JSON_BuildTests       OFF CACHE BOOL "" FORCE)
set(JSON_MultipleHeaders  ON  CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK           ON  CACHE BOOL "" FORCE)
set(INSTALL_GTEST         OFF CACHE BOOL "" FORCE)

# Eigen3: disable testing/doc to avoid creating a 'check' target that
# collides with MIOpen's own 'check' target in test/CMakeLists.txt.
set(BUILD_TESTING         OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)

# frugally-deep's CMake floor is older than 3.30's policy default.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
# Suppress deprecation warnings from third-party CMakeLists (frugally-deep).
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Header-only / CMake-aware deps (have an upstream CMakeLists.txt at the root).
# -----------------------------------------------------------------------------
# Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/eigen-3.4.0.tar.bz2
FetchContent_Declare(Eigen3
    URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
    URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
    OVERRIDE_FIND_PACKAGE)

# Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/FunctionalPlus-0.2.25.tar.gz
FetchContent_Declare(FunctionalPlus
    URL      https://github.com/Dobiasd/FunctionalPlus/archive/refs/tags/v0.2.25.tar.gz
    URL_HASH SHA256=9b5e24bbc92f43b977dc83efbc173bcf07dbe07f8718fc2670093655b56fcee3
    OVERRIDE_FIND_PACKAGE)

# Note: TheRock pins the GitHub release asset (json-3.12.0.tar.gz, hash 42f6e9...);
# we use the auto-generated tag archive instead (different SHA, identical content).
FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
    URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
    OVERRIDE_FIND_PACKAGE)

# Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/frugally-deep-0.15.31.tar.gz
# Note: frugally-deep is populated and add_subdirectory'd manually below so we can
# stub out its upstream cmake/pkgconfig.cmake. That file unconditionally registers
# install(EXPORT fdeepTargets) referencing nlohmann_json — which is itself a
# FetchContent target not in any export set, causing a generate-time error.
FetchContent_Declare(frugally-deep
    URL      https://github.com/Dobiasd/frugally-deep/archive/refs/tags/v0.15.31.tar.gz
    URL_HASH SHA256=49bf5e30ad2d33e464433afbc8b6fe8536fc959474004a1ce2ac03d7c54bc8ba
    OVERRIDE_FIND_PACKAGE)

# Originally mirrored from https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/googletest-1.17.0.tar.gz
FetchContent_Declare(GTest
    URL      https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    OVERRIDE_FIND_PACKAGE)

# -----------------------------------------------------------------------------
# C-source-only deps (no upstream CMakeLists.txt). We fetch the sources and
# add_subdirectory() a small wrapper under cmake/thirdparty/<dep>/ that
# defines the same alias target find_package would have created.
# -----------------------------------------------------------------------------
FetchContent_Declare(BZip2
    URL      https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
    URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
    OVERRIDE_FIND_PACKAGE)

FetchContent_Declare(SQLite3
    URL      https://sqlite.org/2026/sqlite-amalgamation-3510300.zip
    URL_HASH SHA256=acb1e6f5d832484bf6d32b681e858c38add8b2acdfd42ac5df24b8afb46552b4
    OVERRIDE_FIND_PACKAGE)

# Populate the deps that have upstream CMake support — this also writes the
# FetchContent redirect stubs into ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}, so the
# later find_package(...) calls in MIOpen's main CMakeLists succeed.
FetchContent_MakeAvailable(
    Eigen3
    FunctionalPlus
    nlohmann_json
    BZip2
    SQLite3
)

# GTest must be built static on Windows: gmock has class-static members
# (e.g. testing::internal::g_gmock_mutex) that lack __declspec(dllexport),
# so a shared gmock_main.dll fails to link. Save/restore BUILD_SHARED_LIBS
# so only the gtest subdirectory sees the override.
set(_miopen_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(GTest)
set(BUILD_SHARED_LIBS ${_miopen_saved_BUILD_SHARED_LIBS})
unset(_miopen_saved_BUILD_SHARED_LIBS)

# frugally-deep: populate but stub upstream cmake/pkgconfig.cmake before
# add_subdirectory() so its install(EXPORT) referencing un-exported nlohmann_json
# doesn't trip CMake's generate-time export-set validation.
FetchContent_GetProperties(frugally-deep)
if(NOT frugally-deep_POPULATED)
    FetchContent_Populate(frugally-deep)
endif()
file(WRITE "${frugally-deep_SOURCE_DIR}/cmake/pkgconfig.cmake"
     "# install rules disabled by MIOpen standalone wrapper (cmake/ThirdParty.cmake)\n")
add_subdirectory("${frugally-deep_SOURCE_DIR}" "${frugally-deep_BINARY_DIR}" EXCLUDE_FROM_ALL)

# Mimic what FetchContent_MakeAvailable would have written for OVERRIDE_FIND_PACKAGE,
# so the later find_package(frugally-deep CONFIG REQUIRED) call resolves to the
# in-tree fdeep target created by the add_subdirectory() above.
file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-config.cmake"
     "include(\"\${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-extra.cmake\" OPTIONAL)\n")
file(WRITE "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/frugally-deep-config-version.cmake"
     "set(PACKAGE_VERSION \"0.15.31\")\n"
     "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
     "set(PACKAGE_VERSION_EXACT TRUE)\n")

# Restore settings that should not leak into MIOpen's own CMakeLists.
unset(BUILD_TESTING CACHE)
unset(CMAKE_WARN_DEPRECATED CACHE)

# BZip2 and SQLite3 have no CMakeLists.txt in their upstream tarballs, so
# MakeAvailable populated the sources but didn't build anything. Build them
# from our vendored wrappers, which create the BZip2::BZip2 / SQLite::SQLite3
# alias targets that the existing find_package(...) consumers expect.
set(BZIP2_UPSTREAM_SOURCE_DIR "${bzip2_SOURCE_DIR}")
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/thirdparty/bzip2"
    "${bzip2_BINARY_DIR}-wrapper"
)

set(SQLITE3_UPSTREAM_SOURCE_DIR "${sqlite3_SOURCE_DIR}")
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/thirdparty/sqlite3"
    "${sqlite3_BINARY_DIR}-wrapper"
)

# Some of MIOpen's existing find_package consumers expect GTest::gtest /
# GTest::gtest_main aliases (the names FindGTest creates). googletest's own
# CMakeLists exports plain gtest / gtest_main targets, so add the aliases.
if(TARGET gtest AND NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
endif()
if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()
if(TARGET gmock AND NOT TARGET GTest::gmock)
    add_library(GTest::gmock ALIAS gmock)
endif()
if(TARGET gmock_main AND NOT TARGET GTest::gmock_main)
    add_library(GTest::gmock_main ALIAS gmock_main)
endif()
