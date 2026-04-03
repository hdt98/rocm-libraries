# Shared helpers used by:
# 1) configure-time archive probing, and
# 2) build-time per-arch unbundling.

# Validate that a required variable was provided by the caller.
function(miopen_ck_require VAR_NAME DESCRIPTION)
    # Keep script entrypoints explicit about required -D arguments.
    if(NOT DEFINED ${VAR_NAME})
        message(FATAL_ERROR "${DESCRIPTION}")
    endif()
endfunction()

# Normalize an arch string to its base gfx name (strip feature suffixes).
function(miopen_ck_base_arch ARCH OUT_VAR)
    # CK target strings may carry feature suffixes; callers usually need the
    # base gfx name when matching requested and discovered targets.
    string(FIND "${ARCH}" ":" _colon_pos)
    if(_colon_pos GREATER -1)
        string(SUBSTRING "${ARCH}" 0 ${_colon_pos} _base)
    else()
        set(_base "${ARCH}")
    endif()
    set(${OUT_VAR} "${_base}" PARENT_SCOPE)
endfunction()

# Return whether a requested arch matches an available target/triple string.
function(miopen_ck_arch_matches REQUESTED_ARCH AVAILABLE_ARCH OUT_VAR)
    miopen_ck_base_arch("${REQUESTED_ARCH}" _requested_base_arch)
    miopen_ck_base_arch("${AVAILABLE_ARCH}" _available_base_arch)

    # Fast path: exact string or exact base-arch match.
    if("${REQUESTED_ARCH}" STREQUAL "${AVAILABLE_ARCH}" OR
       "${_requested_base_arch}" STREQUAL "${_available_base_arch}")
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()

    # Offload bundle targets are often full triples (e.g.
    # hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), so allow a literal
    # substring match for both the full requested arch and its base arch.
    string(FIND "${AVAILABLE_ARCH}" "${REQUESTED_ARCH}" _requested_arch_pos)
    if(_requested_arch_pos GREATER -1)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
        return()
    endif()

    string(FIND "${AVAILABLE_ARCH}" "${_requested_base_arch}" _requested_base_arch_pos)
    if(_requested_base_arch_pos GREATER -1)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Create a clean temporary work directory anchored near an output/result file.
function(miopen_ck_prepare_work_dir ANCHOR_PATH SUFFIX OUT_VAR)
    # Unpack archive members into a throwaway directory near the caller's
    # output/result path so reruns always start from a clean state.
    get_filename_component(_parent_dir "${ANCHOR_PATH}" DIRECTORY)
    set(_work_dir "${_parent_dir}/${SUFFIX}")
    file(REMOVE_RECURSE "${_work_dir}")
    file(MAKE_DIRECTORY "${_work_dir}")
    set(${OUT_VAR} "${_work_dir}" PARENT_SCOPE)
endfunction()

# Extract all members from a static archive into the given work directory.
function(miopen_ck_extract_archive FAT_ARCHIVE AR WORK_DIR)
    execute_process(
        COMMAND ${AR} x "${FAT_ARCHIVE}"
        WORKING_DIRECTORY "${WORK_DIR}"
        OUTPUT_VARIABLE _ar_output
        ERROR_VARIABLE _ar_error
        RESULT_VARIABLE _ar_result)
    if(NOT _ar_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to extract ${FAT_ARCHIVE}\n"
            "  exit code: ${_ar_result}\n"
            "  stderr: ${_ar_error}\n"
            "  stdout: ${_ar_output}")
    endif()
endfunction()

# Collect extracted object files from the work directory.
function(miopen_ck_collect_objects WORK_DIR OUT_VAR)
    # CK archive members are object files; collect both ELF/COFF suffixes.
    file(GLOB _obj_files "${WORK_DIR}/*.o" "${WORK_DIR}/*.obj")
    set(${OUT_VAR} "${_obj_files}" PARENT_SCOPE)
endfunction()

# Extract and list offload targets from an object's .hip_fatbin section.
function(miopen_ck_list_fatbin_targets
    OBJ
    WORK_DIR
    LLVM_OBJCOPY
    BUNDLER
    OUT_TARGETS_VAR
    OUT_STATUS_VAR
    OUT_EXIT_CODE_VAR
    OUT_ERROR_VAR)
    # Status contract:
    # - OK: targets were listed successfully
    # - NO_FATBIN: object has no HIP device bundle section
    # - LIST_FAILED: .hip_fatbin exists, but bundler -list failed
    # Callers use this to distinguish "skip non-HIP object" from real errors.
    get_filename_component(_obj_name "${OBJ}" NAME)
    set(_fatbin_file "${WORK_DIR}/${_obj_name}.fatbin")

    # Non-HIP objects do not contain .hip_fatbin and are intentionally skipped.
    execute_process(
        COMMAND ${LLVM_OBJCOPY} --dump-section .hip_fatbin=${_fatbin_file} "${OBJ}"
        RESULT_VARIABLE _extract_result
        ERROR_VARIABLE _extract_error)
    if(NOT _extract_result EQUAL 0)
        set(${OUT_TARGETS_VAR} "" PARENT_SCOPE)
        set(${OUT_STATUS_VAR} "NO_FATBIN" PARENT_SCOPE)
        set(${OUT_EXIT_CODE_VAR} "${_extract_result}" PARENT_SCOPE)
        set(${OUT_ERROR_VAR} "${_extract_error}" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND ${BUNDLER} --type=o "--input=${_fatbin_file}" -list
        OUTPUT_VARIABLE _targets_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _list_result
        ERROR_VARIABLE _list_error)
    if(NOT _list_result EQUAL 0)
        set(${OUT_TARGETS_VAR} "" PARENT_SCOPE)
        set(${OUT_STATUS_VAR} "LIST_FAILED" PARENT_SCOPE)
        set(${OUT_EXIT_CODE_VAR} "${_list_result}" PARENT_SCOPE)
        set(${OUT_ERROR_VAR} "${_list_error}" PARENT_SCOPE)
        return()
    endif()

    string(REPLACE "\n" ";" _target_lines "${_targets_output}")
    set(_targets "")
    foreach(_line IN LISTS _target_lines)
        string(STRIP "${_line}" _line)
        if(NOT _line STREQUAL "")
            list(APPEND _targets "${_line}")
        endif()
    endforeach()

    set(${OUT_TARGETS_VAR} "${_targets}" PARENT_SCOPE)
    set(${OUT_STATUS_VAR} "OK" PARENT_SCOPE)
    set(${OUT_EXIT_CODE_VAR} "0" PARENT_SCOPE)
    set(${OUT_ERROR_VAR} "" PARENT_SCOPE)
endfunction()

# Find host target and best-matching device target for a requested arch.
function(miopen_ck_find_targets_for_arch TARGETS ARCH OUT_MATCHED_TARGET OUT_HOST_TARGET)
    # Unbundling needs both the requested device target and the companion host
    # target so the rebundled object keeps a valid host/device pair.
    # If multiple targets match the same base arch, keep the last one seen so
    # feature-qualified targets override generic ones.
    set(_matched_target "")
    set(_host_target "")

    foreach(_target IN LISTS TARGETS)
        if(_target MATCHES "^host-")
            set(_host_target "${_target}")
            continue()
        endif()

        miopen_ck_arch_matches("${ARCH}" "${_target}" _target_matches)
        if(_target_matches)
            set(_matched_target "${_target}")
        endif()
    endforeach()

    set(${OUT_MATCHED_TARGET} "${_matched_target}" PARENT_SCOPE)
    set(${OUT_HOST_TARGET} "${_host_target}" PARENT_SCOPE)
endfunction()
