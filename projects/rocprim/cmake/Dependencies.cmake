# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(FetchContent)

# Stores the content of '_variable' in 'OLD_${_variable}' and
# sets '_variable' to '_value'.
macro(override_cache_variable _variable _value _type)
  if(DEFINED ${_variable})
    set(OLD_${_variable} ${${_variable}})
  endif()
  set(${_variable} ${value} CACHE ${_type} "" FORCE)
endmacro(override_cache_variable)

# Restores the content of '_variable'. If original was unset,
# then unsets '_variable'.
macro(restore_cache_variable _variable _type)
  if(DEFINED OLD_${_variable})
    set(${_variable} ${OLD_${_variable}} CACHE ${_type} "" FORCE)
  else()
    unset(${_variable} CACHE)
  endif()
endmacro(restore_cache_variable)

macro(remove_warning_flags _flags _var)
set(${_var} ${flags})
list(REMOVE_ITEM ${_var} /WX -Werror -Werror=pendantic -pedantic-errors)
if(MSVC)
  list(FILTER ${_var} EXCLUDE REGEX "/[Ww]([0-4]?)(all)?") # Remove MSVC warning flags
  list(APPEND ${_var} /w)
else()
  list(FILTER ${_var} EXCLUDE REGEX "-W(all|extra|everything)") # Remove GCC/LLVM flags
  list(APPEND ${_var} -w)
endif()
set(${_var})
endmacro(remove_warning_flags)

# This function checks to see if the download branch given by "branch" exists in the repository.
# It does so using the git ls-remote command.
# If the branch cannot be found, the variable described by "branch" is changed to "develop" in the host scope.
function(find__branch git_path branch)
  set(branch_value ${${branch}})
  execute_process(COMMAND ${git_path} "ls-remote" "https://github.com/ROCm/rocm-libraries.git" "refs/heads/${branch_value}" RESULT_VARIABLE ret_code OUTPUT_VARIABLE output)

  if(NOT ${ret_code} STREQUAL "0")
    message(WARNING "Unable to check if release branch exists, defaulting to the develop branch.")
    set(${branch} "develop" PARENT_SCOPE)
  else()
    if(${output})
      string(STRIP ${output} output)
    endif()

    if(NOT (${output} MATCHES "[\t ]+refs/heads/${branch_value}(\n)?$"))
      message(WARNING "Unable to locate requested release branch \"${branch_value}\" in repository. Defaulting to the develop branch.")
      set(${branch} "develop" PARENT_SCOPE)
    else()
      message(STATUS "Found release branch \"${branch_value}\" in repository.")
    endif()
  endif()
endfunction()

function(check_git_version git_path)
  execute_process(COMMAND ${git_path} "--version" OUTPUT_VARIABLE git_version_output)
  string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" GIT_VERSION_STRING ${git_version_output})
  if(DEFINED CMAKE_MATCH_0)
    set(GIT_VERSION ${CMAKE_MATCH_0} PARENT_SCOPE)
  else()
    set(GIT_VERSION "" PARENT_SCOPE)
  endif()
endfunction()

macro(fetch_rocmcmake)
  # Try to find package first.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(ROCmCMakeBuildTools 0.14.0 CONFIG QUIET PATHS "${ROCM_ROOT}")
  endif()
  # Otherwise fetch from source
  if(NOT ROCmCMakeBuildTools_FOUND)
    # We don't really want to consume the build and test targets of ROCm CMake.
    # CMake 3.18 allows omitting them, even though there's a CMakeLists.txt in source root.
    # TODO: simplify this once we upgrade minimum CMake to >=3.18
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
    else()
      set(SOURCE_SUBDIR_ARG)
    endif()
    FetchContent_Declare(
      rocm-cmake
      GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
      GIT_TAG        rocm-7.1.1
      ${SOURCE_SUBDIR_ARG}
    )
    FetchContent_GetProperties(rocm-cmake)
    if(NOT rocm-cmake_POPULATED)
      # rocm-cmake 0.12.0 and higher needs to built from source
      FetchContent_Populate(rocm-cmake)
      message("Populated: ${rocm-cmake_SOURCE_DIR}")
      execute_process(
        WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} ${rocm-cmake_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=.
      )
      execute_process(
        WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${rocm-cmake_SOURCE_DIR} --target install
      )
    endif()
    FetchContent_MakeAvailable(rocm-cmake)
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
  else()
    find_package(ROCmCMakeBuildTools 0.11.0 CONFIG REQUIRED PATHS "${ROCM_ROOT}")
  endif()

  # Expose available includes.
  include(ROCMSetupVersion)
  include(ROCMCreatePackage)
  include(ROCMInstallTargets)
  include(ROCMPackageConfigHelpers)
  include(ROCMInstallSymlinks)
  include(ROCMCheckTargetIds)
  include(ROCMClients)
  if(BUILD_DOCS)
    include(ROCMSphinxDoc)
  endif()
endmacro(fetch_rocmcmake)

# Google Test
macro(fetch_googletest)
  # Try to find package first.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    if(WIN32)
      # Older versions of gtest on Windows does not support printing of 128-bit values,
      # Causing compilation errors.
      find_package(GTest 1.11.0 REQUIRED)
    else()
      find_package(GTest QUIET)
    endif()
  endif()
  # Otherwise fetch from source.
  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    if(EXISTS /usr/src/googletest AND NOT DEPENDENCIES_FORCE_DOWNLOAD)
      # TODO: do we still want this behaviour?
      FetchContent_Declare(
        googletest
        SOURCE_DIR /usr/src/googletest
      )
    else()
      message(STATUS "Google Test not found. Fetching...")
      FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
      )
    endif()
    # Set build options.
    set(googletest_CMAKE_CXX_FLAGS OFF CACHE INTERNAL "")
    set(googletest_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(googletest_BUILD_GTEST ON CACHE INTERNAL "")
    set(googletest_BUILD_GMOCK OFF CACHE INTERNAL "")
    set(googletest_INSTALL_GTEST ON CACHE INTERNAL  "")

    FetchContent_MakeAvailable(googletest)

    # Add build targets.
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main  ALIAS gtest_main)
  endif()
endmacro(fetch_googletest)

# Google Benchmark
macro(fetch_googlebench)
  # Try to find package first.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark 1.8.0 CONFIG QUIET)
  endif()
  # Otherwise fetch from source.
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found. Fetching...")
    FetchContent_Declare(
      googlebench
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.0
    )
    # Set build options.
    set(googlebench_CMAKE_CXX_FLAGS OFF CACHE INTERNAL "")
    set(googlebench_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(googlebench_BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
    set(googlebench_BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
    set(googlebench_HAVE_STD_REGEX ON CACHE INTERNAL "")
    set(googlebench_RUN_HAVE_STD_REGEX 1 CACHE INTERNAL "")

    FetchContent_MakeAvailable(googlebench)

    # Add build targets.
    add_library(benchmark::benchmark ALIAS benchmark)
  endif()
endmacro(fetch_googlebench)

# This function fetches the project "_project" using the method specified by "_method".
# The result is stored in the parent scope version of "_path".
# It does not build the repo.
macro(fetch_monorepo _method _project _version _path _branch)
  set(FETCH_METHOD_OPTIONS "PACKAGE" "MONOREPO" "DOWNLOAD")
  if (DEFINED ${_method} AND NOT "${${_method}}" IN_LIST FETCH_METHOD_OPTIONS)
    message(FATAL_ERROR "Unrecognized ${_method}: \"${${_method}}\". Valid options are: ${FETCH_METHOD_OPTIONS}.")
  endif()

  # Since the monorepo is large, we want to avoid downloading the whole thing if possible.
  # We can do this if we have access to git's sparse-checkout functionality, which was added in git 2.25.
  # On some Linux systems (eg. Ubuntu), the git in /usr/bin tends to be newer than the git in /usr/local/bin,
  # and the latter is what gets picked up by find_package(Git), since it's what's in PATH.
  # Check for a git binary in /usr/bin first, then if git < 2.25 is not found, use find_package(Git) to search
  # other locations.
  if (NOT(GIT_PATH))
    message(STATUS "Checking git version")
    set(GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT 2.25)

    find_program(find_result git PATHS /usr/bin NO_DEFAULT_PATH)
    if(NOT (${find_result} STREQUAL "find_result-NOTFOUND"))
      set(GIT_PATH ${find_result} CACHE INTERNAL "Path to the git executable")
      check_git_version(${GIT_PATH})
    endif()

    if(NOT GIT_VERSION OR "${GIT_VERSION}" LESS ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      find_package(Git QUIET)
      if(GIT_FOUND)
        set(GIT_PATH ${GIT_EXECUTABLE} CACHE INTERNAL "Path to the git executable")
        check_git_version(${GIT_PATH})
      endif()
    endif()

    if(NOT GIT_VERSION OR "${GIT_VERSION}" LESS ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      set(USE_SPARSE_CHECKOUT "OFF" CACHE INTERNAL "Records whether git supports sparse checkout functionality")
    else()
      set(USE_SPARSE_CHECKOUT "ON" CACHE INTERNAL "Records whether git supports sparse checkout functionality")
    endif()

    if(NOT GIT_VERSION)
      # Warn the user that we were unable to find git. This will only actually be a problem if we use one of the
      # fetch _methods (download, or monorepo with dependency not present) that requires it. If we end up running
      # into one of those scenarios, a fatal error will be issued at that point.
      message(WARNING "Unable to find git.")
    else()
      message(STATUS "Found git at: ${GIT_PATH}, version: ${GIT_VERSION}")
    endif()
  endif()

  if(${${_method}} STREQUAL "PACKAGE")
    message(STATUS "Searching for ${_project} package")

    # Add default install location for WIN32 and non-WIN32 as hint
    find_package(${_project} ${_version} CONFIG QUIET PATHS "${ROCM_ROOT}/lib/cmake/${_project}")

    if(NOT ${${_project}_FOUND})
      # TODO: Remove fallback behavior. This simplifies the code and makes sure we don't download code
      # without the user's explicit perm
      message(STATUS "No existing ${_project} package meeting the minimum version requirement (${_version}) was found. Falling back to downloading it.")
      # Update local and parent variable values
      set(${_method} "DOWNLOAD")
    else()
      message(STATUS "Package found (${${_project}_DIR})")
    endif()

  elseif(${${_method}} STREQUAL "MONOREPO")
    message(STATUS "Searching for ${_project} in the parent monorepo directory")

    # Check if this looks like a monorepo checkout
    find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${_project}/" NO_CACHE NO_DEFAULT_PATH)

    # If not, see if the local monorepo is a sparse-checkout.
    # If it is a sparse-checkout, try to add the dependency to the sparse-checkout list.
    # If it's not a sparse-checkout (or adding to the sparse-checkout list fails), fall back to downloading the dependency.
    if(${found_path} STREQUAL "found_path-NOTFOUND")
      set(FALLBACK_TO_DOWNLOAD ON)
      message(WARNING "Unable to locate ${_project} in parent monorepo (it's not at \"${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${_project}/\").")
      message(STATUS "Checking if local monorepo is a sparse-checkout that we can add ${_project} to.")
      if(NOT(GIT_PATH))
        message(FATAL_ERROR "Git could not be found on the system. Since ${_project} could not be found in the local monorepo, git is required to download it.")
      endif()

      if(USE_SPARSE_CHECKOUT)
        execute_process(COMMAND ${GIT_PATH} "sparse-checkout" "list" OUTPUT_VARIABLE sparse_list ERROR_VARIABLE git_error RESULT_VARIABLE git_result
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)

        if(NOT(git_result EQUAL 0) OR git_error)
          message(STATUS "The local monorepo does not appear to be a sparse-checkout.")
        else()
          message(STATUS "The local monorepo appears to be a sparse checkout. Attempting to add \"projects/${_project}\" to the checkout list.")
          # Check if the dependency is already present in the checkout list.
          # Git lists sparse checkout directories each on a separate line.
          # Take care not to match something in the middle of a path, eg. "other_dir/projects/${_project}/sub_dir".
          string(REGEX MATCH "(^|\n)projects/${_project}($|\n)" find_result ${sparse_list})
          if(find_result)
            message(STATUS "Found existing entry for \"projects/${_project}\" in sparse-checkout list - has the directory structure been modified?")
          else()
            # Add project/${_project} to the sparse checkout
            execute_process(COMMAND ${GIT_PATH} "sparse-checkout" "add" "projects/${_project}" RESULT_VARIABLE sparse_checkout_result
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)
            # Note that in this case, we are forced to checkout the same branch that the sparse-checkout was created with.
            execute_process(COMMAND ${GIT_PATH} "checkout" RESULT_VARIABLE checkout_result
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)

            if(sparse_checkout_result EQUAL 0 AND checkout_result EQUAL 0)
              message(STATUS "Added new checkout list entry.")
              set(FALLBACK_TO_DOWNLOAD OFF)
            else()
              message(STATUS "Unable to add new checkout list entry.")
            endif()
            # Save the monorepo path in the parent scope
            set(${_path} "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${_project}")
          endif()
        endif()
      else()
        message(STATUS "The version of git installed on the system (${GIT_VERSION}) does not support sparse-checkout.")
      endif()

      if (FALLBACK_TO_DOWNLOAD)
        message(WARNING "Unable to locate/fetch dependency ${_project} from monorepo. Falling back to downloading it.")
        # Update local and parent variable values
        set(${_method} "DOWNLOAD")
      endif()

    else()
      message(STATUS "Found ${_project} at ${found_path}")

      # Save the monorepo path in the parent scope
      set(${_path} ${found_path})
    endif()
  endif()

  if(${${_method}} STREQUAL "DOWNLOAD")
    if(NOT DEFINED GIT_PATH)
      message(FATAL_ERROR "Git could not be found on the system. Git is required for downloading ${_project}.")
    endif()

    message(STATUS "Checking if repository contains requested branch ${${_branch}}")
    find__branch(${GIT_PATH} ${_branch})
    set(_branch_value ${${_branch}})

    message(STATUS "Downloading ${_project} from https://github.com/ROCm/rocm-libraries.git")
    if(${USE_SPARSE_CHECKOUT})
      # In this case, we have access to git sparse-checkout.
      # Check if the dependency has already been downloaded in the past:
      find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_BINARY_DIR}/${_project}-src/" NO_CACHE NO_DEFAULT_PATH)
      if(${found_path} STREQUAL "found_path-NOTFOUND")
        # First, git clone with options "--no-checkout" and "--filter=tree:0" to prevent files from being pulled immediately.
        # Use option "--depth=1" to avoid downloading past commit history.
        execute_process(COMMAND ${GIT_PATH} clone --branch ${_branch_value} --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git ${CMAKE_CURRENT_BINARY_DIR}/${_project}-src)

        # Next, use git sparse-checkout to ensure we only pull the directory containing the desired repo.
        execute_process(COMMAND ${GIT_PATH} sparse-checkout init --cone
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_project}-src)

        execute_process(COMMAND ${GIT_PATH} sparse-checkout set projects/${_project}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_project}-src)

        # Finally, download the files using git checkout.
        execute_process(COMMAND ${GIT_PATH} checkout ${_branch_value}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_project}-src)

        message(STATUS "${_project} download complete")
      else()
        message("Found previously downloaded directory, skipping download step.")
      endif()

      # Save the downloaded path in the parent scope
      set(${_path} "${CMAKE_CURRENT_BINARY_DIR}/${_project}-src/projects/${_project}")
    else()
      # In this case, we do not have access to sparse-checkout, so we need to download the whole monorepo.
      # Check if the monorepo has already been downloaded to satisfy a previous dependency
      find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_BINARY_DIR}/monorepo-src/" NO_CACHE NO_DEFAULT_PATH)
      if(${found_path} STREQUAL "found_path-NOTFOUND")
        # Warn the user that this will take some time.
        message(WARNING "The detected version of git (${GIT_VERSION}) is older than 2.25 and does not provide sparse-checkout functionality. Falling back to checking out the whole rocm-libraries repository (this may take a long time).")
        # Avoid downloading anything related to branches other than the target branch (--single-branch), and avoid any past commit history information (--depth=1)
        execute_process(COMMAND ${GIT_PATH} clone --single-branch --branch=${_branch_value} --depth=1 https://github.com/ROCm/rocm-libraries.git ${CMAKE_CURRENT_BINARY_DIR}/monorepo-src)
        message(STATUS "rocm-libraries download complete")
      else()
        message("Found previously downloaded directory, skipping download step.")
      endif()

      # Save the downloaded path in the parent scope
      set(${_path} "${CMAKE_CURRENT_BINARY_DIR}/monorepo-src/projects/${_project}")
    endif()
  endif()
endmacro(fetch_monorepo)

macro(fetch_rocrand)
  override_cache_variable(ROCM_PACKAGE_CREATED FALSE BOOL)
  fetch_monorepo(ROCRAND_FETCH_METHOD rocrand 4.2.0 ROCRAND_PATH ROCRAND_FETCH_BRANCH)
  if(NOT rocrand_FOUND)
    FetchContent_Declare(
      rocrand
      SOURCE_DIR    ${ROCRAND_PATH}
      INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand
      LOG_CONFIGURE TRUE
      LOG_BUILD     TRUE
      LOG_INSTALL   TRUE
    )
    set(rocrand_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(rocrand_CMAKE_CXX_FLAGS "" CACHE INTERNAL "")
    set(rocrand_BUILD_TEST OFF CACHE INTERNAL "")
    set(rocrand_BUILD_BENCHMARK OFF "" CACHE INTERNAL "")
    FetchContent_MakeAvailable(rocrand)
    if(NOT TARGET roc::rocrand)
      add_library(roc::rocrand ALIAS rocrand)
      target_compile_options(rocrand PRIVATE "-Wno-pass-failed")
    endif()
  endif()
  restore_cache_variable(ROCM_PACKAGE_CREATED BOOL)
endmacro(fetch_rocrand)
