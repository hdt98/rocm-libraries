# probe_archive_archs.cmake
# Detects which requested architectures have device code in a CK fat archive.
#
# Required -D arguments:
#   FAT_ARCHIVE   - Path to the fat .a file
#   ARCHS         - Semicolon-delimited list of GPU architectures to probe
#   RESULT_FILE   - Path to a CMake file containing FOUND_DEVICE_CODE/FOUND_ARCHS
#   BUNDLER       - Path to clang-offload-bundler
#   AR            - Path to ar tool
#   LLVM_OBJCOPY  - Path to llvm-objcopy

cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_LIST_DIR}/ck_archive_common.cmake")

miopen_ck_require(FAT_ARCHIVE "FAT_ARCHIVE is required")
miopen_ck_require(ARCHS "ARCHS is required")
miopen_ck_require(RESULT_FILE "RESULT_FILE is required")
miopen_ck_require(BUNDLER "BUNDLER is required")
miopen_ck_require(AR "AR is required")
miopen_ck_require(LLVM_OBJCOPY "LLVM_OBJCOPY is required")

miopen_ck_prepare_work_dir("${RESULT_FILE}" "_probe_archive" work_dir)
miopen_ck_extract_archive("${FAT_ARCHIVE}" "${AR}" "${work_dir}")
miopen_ck_collect_objects("${work_dir}" obj_files)

set(found_device_code FALSE)
set(found_archs "")
# Track only the archs we still need to find so large archives can short-circuit
# once every requested target has been seen.
set(unmatched_probe_archs ${ARCHS})

foreach(obj IN LISTS obj_files)
    miopen_ck_list_fatbin_targets(
        "${obj}"
        "${work_dir}"
        "${LLVM_OBJCOPY}"
        "${BUNDLER}"
        targets
        target_list_status
        target_list_exit_code
        target_list_error)

    if(target_list_status STREQUAL "NO_FATBIN")
        continue()
    endif()

    if(target_list_status STREQUAL "LIST_FAILED")
        message(FATAL_ERROR
            "Failed to list fatbin targets while probing ${FAT_ARCHIVE}\n"
            "  object: ${obj}\n"
            "  bundler: ${BUNDLER}\n"
            "  exit code: ${target_list_exit_code}\n"
            "  stderr: ${target_list_error}")
    endif()

    if(NOT target_list_status STREQUAL "OK")
        message(FATAL_ERROR
            "Unexpected target-listing status '${target_list_status}' while probing ${obj}")
    endif()

    if(NOT targets)
        continue()
    endif()

    foreach(probe_arch IN LISTS unmatched_probe_archs)
        set(_probe_arch_found FALSE)
        foreach(target IN LISTS targets)
            if(target MATCHES "^host-")
                continue()
            endif()

            miopen_ck_arch_matches("${probe_arch}" "${target}" _probe_arch_found)
            if(_probe_arch_found)
                break()
            endif()
        endforeach()

        if(_probe_arch_found)
            set(found_device_code TRUE)
            list(APPEND found_archs "${probe_arch}")
            list(REMOVE_ITEM unmatched_probe_archs "${probe_arch}")
        endif()
    endforeach()

    if(NOT unmatched_probe_archs)
        break()
    endif()
endforeach()

# The caller includes this file during configure to recover the discovered arch list.
file(WRITE "${RESULT_FILE}" "set(FOUND_DEVICE_CODE ${found_device_code})\n")
file(APPEND "${RESULT_FILE}" "set(FOUND_ARCHS \"${found_archs}\")\n")

file(REMOVE_RECURSE "${work_dir}")
