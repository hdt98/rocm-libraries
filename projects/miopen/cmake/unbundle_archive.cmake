# unbundle_archive.cmake
# Extracts single-arch objects from a fat static archive.
#
# Handles HIP objects where device bundles are stored in .hip_fatbin sections
# inside normal host objects (CCOB model).
#
# Required -D arguments:
#   FAT_ARCHIVE   - Path to the fat .a file
#   ARCH          - GPU architecture (e.g., gfx90a)
#   OUTPUT        - Path for the output per-arch .a file
#   BUNDLER       - Path to clang-offload-bundler
#   AR            - Path to ar tool
#   LLVM_OBJCOPY  - Path to llvm-objcopy (NOT GNU objcopy, which rewrites ELFs)

cmake_minimum_required(VERSION 3.16)

# Script entrypoint used via `cmake -P`; helper functions live in
# ck_archive_common.cmake.
include("${CMAKE_CURRENT_LIST_DIR}/ck_archive_common.cmake")

miopen_ck_require(FAT_ARCHIVE "FAT_ARCHIVE is required")
miopen_ck_require(ARCH "ARCH is required")
miopen_ck_require(OUTPUT "OUTPUT is required")
miopen_ck_require(BUNDLER "BUNDLER is required")
miopen_ck_require(AR "AR is required")
miopen_ck_require(LLVM_OBJCOPY "LLVM_OBJCOPY is required")

miopen_ck_prepare_work_dir("${OUTPUT}" "_unbundle_${ARCH}" work_dir)
miopen_ck_extract_archive("${FAT_ARCHIVE}" "${AR}" "${work_dir}")
miopen_ck_collect_objects("${work_dir}" obj_files)

set(thin_objs)

foreach(obj IN LISTS obj_files)
    get_filename_component(obj_name "${obj}" NAME)

    miopen_ck_list_fatbin_targets(
        "${obj}"
        "${work_dir}"
        "${LLVM_OBJCOPY}"
        "${BUNDLER}"
        targets
        target_list_status
        target_list_exit_code
        target_list_error)

    # No .hip_fatbin means this archive member is not a HIP object.
    if(target_list_status STREQUAL "NO_FATBIN")
        continue()
    endif()

    # A bundler list failure after successful extraction is a real tool/data
    # error and should fail the build immediately.
    if(target_list_status STREQUAL "LIST_FAILED")
        message(FATAL_ERROR
            "Failed to list fatbin targets for ${obj_name} (${ARCH})\n"
            "  object: ${obj}\n"
            "  bundler: ${BUNDLER}\n"
            "  exit code: ${target_list_exit_code}\n"
            "  stderr: ${target_list_error}")
    endif()

    if(NOT target_list_status STREQUAL "OK")
        message(FATAL_ERROR
            "Unexpected target-listing status '${target_list_status}' for ${obj_name} (${ARCH})")
    endif()

    # Empty target output is unexpected but harmless; ignore this object.
    if(NOT targets)
        continue()
    endif()

    miopen_ck_find_targets_for_arch("${targets}" "${ARCH}" matched_target host_target)
    # This object does not contain device code for the requested arch.
    if(NOT matched_target)
        continue()
    endif()

    # clang-offload-bundler requires every target in the original bundle to be
    # listed during unbundle, even though the rebundle step keeps only host +
    # one matched device target.
    set(unbundle_targets "")
    set(unbundle_outputs "")
    set(device_output "")
    set(host_output "")
    foreach(target IN LISTS targets)
        string(REPLACE "/" "_" safe_target "${target}")
        set(out_file "${work_dir}/${obj_name}.${safe_target}")
        if(unbundle_targets)
            set(unbundle_targets "${unbundle_targets},${target}")
        else()
            set(unbundle_targets "${target}")
        endif()
        list(APPEND unbundle_outputs "--output=${out_file}")
        if(target STREQUAL matched_target)
            set(device_output "${out_file}")
        elseif(target STREQUAL host_target)
            set(host_output "${out_file}")
        endif()
    endforeach()

    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${unbundle_targets}"
            "--input=${work_dir}/${obj_name}.fatbin"
            ${unbundle_outputs}
            --unbundle
        OUTPUT_VARIABLE unbundle_output
        ERROR_VARIABLE  unbundle_error
        RESULT_VARIABLE unbundle_result)

    if(NOT unbundle_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to unbundle ${obj_name} for ${ARCH}\n"
            "  exit code: ${unbundle_result}\n"
            "  stderr: ${unbundle_error}\n"
            "  stdout: ${unbundle_output}")
    endif()

    # Re-bundle to a host+single-device fatbin for this arch.
    set(rebundle_targets "${host_target},${matched_target}")
    set(thin_fatbin "${work_dir}/${obj_name}.thin_fatbin")
    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${rebundle_targets}"
            "--input=${host_output}" "--input=${device_output}"
            "--output=${thin_fatbin}"
            --compress
        OUTPUT_VARIABLE rebundle_output
        ERROR_VARIABLE  rebundle_error
        RESULT_VARIABLE rebundle_result)

    if(NOT rebundle_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to re-bundle ${obj_name} for ${ARCH}\n"
            "  exit code: ${rebundle_result}\n"
            "  stderr: ${rebundle_error}\n"
            "  stdout: ${rebundle_output}")
    endif()

    set(fatbin_file "${work_dir}/${obj_name}.fatbin")

    # Replace the .hip_fatbin section with the single-arch fatbin.
    # --update-section preserves relocations (.hipFatBinSegment references
    # .hip_fatbin) and works on both ELF and COFF, but requires the new
    # content to be no larger than the original section.
    # For multi-arch bundles this holds: we removed device code blobs.
    # For single-arch bundles, re-encoding adds overhead that can make the
    # thin fatbin larger than the original. In that case the object is already
    # target-specific, so keep the original object unchanged.
    file(SIZE "${fatbin_file}" _orig_fatbin_size)
    file(SIZE "${thin_fatbin}" _thin_fatbin_size)

    if(_thin_fatbin_size GREATER _orig_fatbin_size)
        # Count device targets (those containing "gfx"). If more than one,
        # the thin fatbin should have been smaller — something is wrong.
        set(_num_device_targets 0)
        foreach(_t IN LISTS targets)
            if(_t MATCHES "gfx")
                math(EXPR _num_device_targets "${_num_device_targets} + 1")
            endif()
        endforeach()
        if(NOT _num_device_targets EQUAL 1)
            message(FATAL_ERROR
                "Thin fatbin for ${obj_name} is larger than original "
                "(${_thin_fatbin_size} > ${_orig_fatbin_size}) but bundle "
                "has ${_num_device_targets} device targets")
        endif()
        list(APPEND thin_objs "${obj}")
        continue()
    endif()

    set(thin_obj "${work_dir}/thin_${obj_name}")
    execute_process(
        COMMAND ${LLVM_OBJCOPY}
            --update-section=.hip_fatbin=${thin_fatbin}
            "${obj}" "${thin_obj}"
        OUTPUT_VARIABLE patch_output
        ERROR_VARIABLE  patch_error
        RESULT_VARIABLE patch_result)

    if(patch_result EQUAL 0)
        list(APPEND thin_objs "${thin_obj}")
    else()
        message(FATAL_ERROR
            "Failed to patch ${obj_name} for ${ARCH}\n"
            "  exit code: ${patch_result}\n"
            "  stderr: ${patch_error}\n"
            "  stdout: ${patch_output}")
    endif()
endforeach()

if(thin_objs)
    list(LENGTH thin_objs count)

    # Use a response file so large CK archives do not overflow command-line
    # length limits while re-archiving filtered objects.
    set(rsp_file "${work_dir}/thin_objs.rsp")
    file(WRITE "${rsp_file}" "")
    foreach(_obj IN LISTS thin_objs)
        file(APPEND "${rsp_file}" "${_obj}\n")
    endforeach()

    execute_process(
        COMMAND ${AR} rcs "${OUTPUT}" "@${rsp_file}"
        OUTPUT_VARIABLE ar_output
        ERROR_VARIABLE  ar_error
        RESULT_VARIABLE ar_result)
    if(NOT ar_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to create ${OUTPUT}\n"
            "  exit code: ${ar_result}\n"
            "  stderr: ${ar_error}\n"
            "  stdout: ${ar_output}\n"
            "  object count: ${count}")
    endif()
    message(STATUS "Created ${OUTPUT} with ${count} objects")
else()
    message(FATAL_ERROR
        "No device code objects were found for ${ARCH} in ${FAT_ARCHIVE}. "
        "Refusing to create ${OUTPUT}. Did you build CK to include ${ARCH}?")
endif()

file(REMOVE_RECURSE "${work_dir}")
