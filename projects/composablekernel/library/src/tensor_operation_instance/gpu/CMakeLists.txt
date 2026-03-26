# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(CheckCXXCompilerFlag)

# We need to pass '-x hip' since check_cxx_compiler_flag assumes c++ and not HIP. 
check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)

function(add_instance_library INSTANCE_NAME)
    message(DEBUG "adding instance ${INSTANCE_NAME}")
    set(result 1)
    if(DEFINED DTYPES)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)
            set(test 0)
            foreach(type IN LISTS DTYPES)
                if(type MATCHES "fp16")
                    set(type1 "_f16")
                elseif(type MATCHES "fp32")
                    set(type1 "_f32")
                elseif(type MATCHES "tf32")
                    set(type1 "_tf32")
                elseif(type MATCHES "fp8")
                    set(type1 "_f8")
                elseif(type MATCHES "bf16")
                    set(type1 "_b16")
                elseif(type MATCHES "fp64")
                    set(type1 "_f64")
                elseif(type MATCHES "int8")
                    set(type1 "_i8")
                endif()
                #make an exception for reduction kernels
                if("${source_name}" MATCHES "${type}" OR "${source_name}" MATCHES "${type1}" OR "${source_name}" MATCHES "device_reduce_instance" OR ${source_name} MATCHES "device_image_to_column")
                    #if filename matches any selected type, exit type loop and do no exclude the file from the list
                    set(test 0)
                    break()
                elseif((source_name MATCHES "fp8" OR source_name MATCHES "fp32" OR source_name MATCHES "tf32" OR source_name MATCHES "fp64" OR source_name MATCHES "bf16" OR source_name MATCHES "int8" OR source_name MATCHES "fp16" OR
                         source_name MATCHES "_f8" OR source_name MATCHES "_f32" OR source_name MATCHES "_tf32" OR source_name MATCHES "_f64" OR source_name MATCHES "_i8" OR source_name MATCHES "_f16" OR source_name MATCHES "_b16") AND
                    NOT (source_name MATCHES type OR source_name MATCHES type1))
                    #if filename contains a type which doesn't match any selected type, mark it for removal
                    set(test 1)
                endif()
            endforeach()
            if(test EQUAL 1)
                message(DEBUG "removing instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endforeach()
    endif()

    set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})

    foreach(source IN LISTS ARGN)
        get_filename_component(source_name ${source} NAME)

        # Do not build DPP instances if DPP_KERNELS macro is not set
        if(NOT DEFINED DPP_KERNELS AND source_name MATCHES "_dpp")
            message(DEBUG "removing dpp instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build DL instances if DL_KERNELS macro is not set
        if(NOT DEFINED DL_KERNELS AND source_name MATCHES "_dl")
            message(DEBUG "removing dl instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build XDL instances if gfx9 targets are not on the target list
        if(((NOT INST_TARGETS MATCHES "gfx9" AND NOT INST_TARGETS MATCHES "gfx11" AND NOT INST_TARGETS MATCHES "gfx12") OR FORCE_DISABLE_XDL) AND source_name MATCHES "_xdl")
            message(DEBUG "removing xdl instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build MX instances if gfx950 targets are not on the target list
        if(NOT INST_TARGETS MATCHES "gfx950" AND source_name MATCHES "_mx")
            message(DEBUG "removing MX instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build WMMA instances if gfx11 targets are not on the target list
        if(((NOT INST_TARGETS MATCHES "gfx11" AND NOT INST_TARGETS MATCHES "gfx12") OR FORCE_DISABLE_WMMA) AND source_name MATCHES "_wmma")
            message(DEBUG "removing wmma instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build mha instances if gfx94 or gfx90a targets are not on the target list
        if((NOT BUILD_MHA_LIB OR (NOT INST_TARGETS MATCHES "gfx94" AND NOT INST_TARGETS MATCHES "gfx90a" AND NOT INST_TARGETS MATCHES "gfx95")) AND source_name MATCHES "mha")
            message(DEBUG "removing mha instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build XDL gemm_universal_f8 or gemm_multiply_multiply_f8 for any targets except gfx94, gfx95 and gfx12
        if(NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
            if(NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx12" AND source_name MATCHES "gemm_multiply_multiply" AND source_name MATCHES "_f8_")
                message(DEBUG "removing gemm_multiply_multiply_f8 instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
            if(NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx12" AND source_name MATCHES "gemm_xdl_universal" AND source_name MATCHES "_f8_")
                message(DEBUG "removing gemm_universal_f8 instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
            if(NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx12" AND source_name MATCHES "gemm_blockscale" AND source_name MATCHES "_f8_")
                message(DEBUG "removing gemm_blockscale_f8 instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endif()
        # Do not build WMMA gemm_universal_f8 for any targets except gfx12+
        if((NOT INST_TARGETS MATCHES "gfx12" OR FORCE_DISABLE_WMMA) AND source_name MATCHES "gemm_wmma_universal" AND source_name MATCHES "_f8_")
            message(DEBUG "removing gemm_wmma_universal_f8 instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build WMMA grouped conv 3d fwd fp8 / bf8 for any targets except gfx12+
        if(NOT INST_TARGETS MATCHES "gfx12" AND source_name MATCHES "grouped_conv3d_fwd_wmma" AND source_name MATCHES "_(f8|fp8|bf8)_")
            message(DEBUG "removing grouped_conv3d_fwd_wmma fp8/bf8 instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build gemm_universal_preshuffle_f8 for any targets except gfx94, gfx95 and gfx12
        if(NOT (INST_TARGETS MATCHES "gfx94|gfx95|gfx12") AND (source_name MATCHES "gemm_universal_preshuffle") AND source_name MATCHES "_f8_" )
            message(DEBUG "removing gemm_universal_preshuffle_f8 instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build gemm_ab_scale_f8 for any targets except gfx94, gfx95 and gfx12
        if(NOT (INST_TARGETS MATCHES "gfx942" OR INST_TARGETS MATCHES "gfx950" OR INST_TARGETS MATCHES "gfx12") AND (source_name MATCHES "gemm_ab_scale") AND (source_name MATCHES "_f8_f8_"))
            message(DEBUG "removing gemm_ab_scale_f8 instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Do not build gemm_blockscale_wp_f8 for any targets except gfx94, gfx95 and gfx12
        if(NOT (INST_TARGETS MATCHES "gfx942" OR INST_TARGETS MATCHES "gfx950" OR INST_TARGETS MATCHES "gfx12") AND (source_name MATCHES "gemm_blockscale_wp") AND (source_name MATCHES "_f8_f8_"))
            message(DEBUG "removing gemm_blockscale_wp_f8 instance ${source} ")
            list(REMOVE_ITEM ARGN "${source}")
        endif()
        # Only build tf32 instances for gfx942 & gfx950
        if(source_name MATCHES "_tf32_")
            if(NOT ((INST_TARGETS MATCHES "gfx942|gfx950") AND CK_ENABLE_TF32))
                message(DEBUG "removing tf32 instance ${source} ")
                list(REMOVE_ITEM ARGN "${source}")
            endif()
        endif()

    endforeach()

    message(DEBUG "remaining instances: ${ARGN}")
    #only continue if there are some source files left on the list
    if(ARGN)
        set(INST_OBJ)
        foreach(source IN LISTS ARGN)
            get_filename_component(source_name ${source} NAME)

            set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
            if(source_name MATCHES "_xdl")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx1030 gfx10-3-generic)
            elseif(source_name MATCHES "_wmma")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack+ gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx950)
            elseif(source_name MATCHES "mha")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx908 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic)
            endif()

            if(source_name MATCHES "_mx")
                list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx10-3-generic gfx11-generic gfx12-generic)
            endif()

            #only build the fp8 gemm instances for gfx90a if the build argument is set, otherwise only build for gfx942/gfx950 and gfx1200/gfx1201
            if(NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH)
                if(source_name MATCHES "gemm_xdl_universal|gemm_multiply_multiply|gemm_universal_preshuffle|gemm_blockscale" AND source_name MATCHES "_f8_")
                    list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx90a:xnack+ gfx90a:xnack- gfx908 gfx90a gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx10-3-generic gfx11-generic)
                endif()
            else()
                if(source_name MATCHES "gemm_xdl_universal|gemm_multiply_multiply|gemm_universal_preshuffle|gemm_blockscale" AND source_name MATCHES "_f8_")
                    list(REMOVE_ITEM INST_TARGETS gfx900 gfx906 gfx906:xnack- gfx908:xnack- gfx908:xnack+ gfx908 gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx10-3-generic gfx11-generic)
                endif()
            endif()
            if(source_name MATCHES "gemm_wmma_universal" AND source_name MATCHES "f8")
                list(FILTER INST_TARGETS INCLUDE REGEX "gfx12")
            endif()
            set(offload_targets)
            foreach(target IN LISTS INST_TARGETS)
                string(APPEND offload_targets "--offload-arch=${target} ")
            endforeach()
            set_source_files_properties(${source} PROPERTIES COMPILE_FLAGS ${offload_targets})
            list(APPEND INST_OBJ ${source})
        endforeach()
        add_library(${INSTANCE_NAME} OBJECT ${INST_OBJ})

        # Allow comparing floating points directly in order to check sentinel values
        if(${INSTANCE_NAME} STREQUAL "device_mha_instance")
            if(NOT DEFINED FMHA_FWD_FAST_EXP2)
                set(FMHA_FWD_FAST_EXP2 true)
            endif()
            if(FMHA_FWD_FAST_EXP2)
                list(APPEND FMHA_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero)
            else()
                list(APPEND FMHA_COMPILE_OPTIONS -Wno-undefined-func-template -DCK_TILE_FMHA_FWD_FAST_EXP2=0)
            endif()
            list(APPEND FMHA_COMPILE_OPTIONS -Wno-float-equal)
            list(APPEND FMHA_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_SPLITKV_API=1)
            list(APPEND FMHA_COMPILE_OPTIONS -DCK_TILE_FMHA_FWD_APPENDKV_API=1)
            target_compile_options(device_mha_instance PRIVATE ${FMHA_COMPILE_OPTIONS})
        endif()

        target_compile_features(${INSTANCE_NAME} PUBLIC)

        # splits debug information into separate .dwo files to reduce debug section size
        if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(${INSTANCE_NAME} PRIVATE -gsplit-dwarf)
        endif()

        # flags to compress the library
        if(NOT DISABLE_OFFLOAD_COMPRESS AND CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
            message(DEBUG "Adding --offload-compress flag for ${INSTANCE_NAME}")
            target_compile_options(${INSTANCE_NAME} PRIVATE --offload-compress)
        endif()

        set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        clang_tidy_check(${INSTANCE_NAME})
        set(result 0)
        message(DEBUG "add_instance_library ${INSTANCE_NAME}")
    else()
        message(DEBUG "skip_instance_libary ${INSTANCE_NAME}")
    endif()
    set(result ${result} PARENT_SCOPE)
endfunction(add_instance_library INSTANCE_NAME)


file(GLOB dir_list LIST_DIRECTORIES true *)
set(CK_DEVICE_OTHER_INSTANCES)
set(CK_DEVICE_GEMM_INSTANCES)
set(CK_DEVICE_CONV_INSTANCES)
set(CK_DEVICE_MHA_INSTANCES)
set(CK_DEVICE_CONTRACTION_INSTANCES)
set(CK_DEVICE_REDUCTION_INSTANCES)
FOREACH(subdir_path ${dir_list})
    set(target_dir)
    IF(IS_DIRECTORY "${subdir_path}")
        set(cmake_instance)
        file(READ "${subdir_path}/CMakeLists.txt" cmake_instance)
        set(add_inst 0)
        if(("${cmake_instance}" MATCHES "_fp8" OR "${cmake_instance}" MATCHES "_f8") AND DTYPES MATCHES "fp8")
            message(DEBUG "fp8 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_bf8" OR "${cmake_instance}" MATCHES "_b8") AND DTYPES MATCHES "bf8")
            message(DEBUG "bf8 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_bf16" OR "${cmake_instance}" MATCHES "_b16") AND DTYPES MATCHES "bf16")
            message(DEBUG "bf16 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp16" OR "${cmake_instance}" MATCHES "_f16") AND DTYPES MATCHES "fp16")
            message(DEBUG "fp16 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp32" OR "${cmake_instance}" MATCHES "_f32") AND DTYPES MATCHES "fp32")
            message(DEBUG "fp32 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_tf32" OR "${cmake_instance}" MATCHES "_tf32") AND DTYPES MATCHES "tf32")
            message(DEBUG "tf32 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_fp64" OR "${cmake_instance}" MATCHES "_f64") AND DTYPES MATCHES "fp64")
            message(DEBUG "fp64 instance found!")
            set(add_inst 1)
        endif()
        if(("${cmake_instance}" MATCHES "_int8" OR "${cmake_instance}" MATCHES "_i8") AND DTYPES MATCHES "int8")
            message(DEBUG "int8 instance found!")
            set(add_inst 1)
        endif()
        if(NOT ("${cmake_instance}" MATCHES "_fp8" OR
                "${cmake_instance}" MATCHES "_f8" OR
                "${cmake_instance}" MATCHES "_fp16" OR
                "${cmake_instance}" MATCHES "_f16" OR
                "${cmake_instance}" MATCHES "_fp32" OR
                "${cmake_instance}" MATCHES "_f32" OR
                "${cmake_instance}" MATCHES "_tf32" OR
                "${cmake_instance}" MATCHES "_fp64" OR
                "${cmake_instance}" MATCHES "_f64" OR
                "${cmake_instance}" MATCHES "_bf16" OR
                "${cmake_instance}" MATCHES "_int8" OR
                "${cmake_instance}" MATCHES "_i8" OR
                "${cmake_instance}" MATCHES "_int4"))
            message(DEBUG "instance should be built for all types!")
            set(add_inst 1)
        endif()
        if(NOT DEFINED DTYPES)
            set(add_inst 1)
        endif()

        set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})

        if(("${cmake_instance}" MATCHES "quantization") AND (DEFINED DTYPES) AND (NOT DTYPES MATCHES "int8"))
            message(DEBUG "quantization instances will not be built!")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY DL_KERNELS") AND (NOT DEFINED DL_KERNELS))
            message(DEBUG "Found only dl instances, but DL_KERNELS is not set. Skipping.")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY XDL_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx9|gfx11|gfx12" OR FORCE_DISABLE_XDL))
            message(DEBUG "Found only xdl instances, but gfx9 is not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY MX_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx950"))
            message(DEBUG "Found only MX instances, but gfx950 is not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
	    if(("${cmake_instance}" MATCHES "ONLY WMMA_KERNELS") AND (((NOT INST_TARGETS MATCHES "gfx11") AND (NOT INST_TARGETS MATCHES "gfx12")) OR FORCE_DISABLE_WMMA))
            message(DEBUG "Found only wmma instances, but gfx11 is not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "ONLY XDL_AND_DL_KERNELS") AND (NOT DEFINED DL_KERNELS) AND (NOT INST_TARGETS MATCHES "gfx9|gfx11|gfx12"))
            message(DEBUG "Found only xdl and dl instances, but gfx9 is not on the targets listand DL_KERNELS is not set. Skipping.")
            set(add_inst 0)
        endif()
	    if(("${cmake_instance}" MATCHES "ONLY XDL_AND_WMMA_KERNELS") AND ((NOT INST_TARGETS MATCHES "gfx9|gfx11|gfx12") OR (FORCE_DISABLE_XDL AND FORCE_DISABLE_WMMA)))
            message(DEBUG "Found only xdl and wmma instances, but gfx11 and gfx9 are not on the targets list. Skipping.")
            set(add_inst 0)
        endif()
	    if(("${cmake_instance}" MATCHES "XDL_DL_WMMA_KERNELS") AND (NOT INST_TARGETS MATCHES "gfx9|gfx11|gfx12") AND (NOT DEFINED DL_KERNELS))
            message(DEBUG "Found xdl, dl, and wmma instances, but none of those meet the target list. Skipping.")
            set(add_inst 0)
        endif()
        if("${cmake_instance}" MATCHES "gemm_multiply_multiply_wp" AND (NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx12"))
            message(DEBUG "Found gemm_multiply_multiply_wp instances, but gfx94/gfx95/gfx12 not on the target list. Skipping. ${cmake_instance}")
            set(add_inst 0)
        elseif("${cmake_instance}" MATCHES "gemm_multiply_multiply" AND (NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx11|gfx12"))
            message(DEBUG "Found gemm_multiply_multiply instances, but gfx94/gfx95/gfx11/gfx12 not on the target list. Skipping. ${cmake_instance}")
            set(add_inst 0)
        endif()
        if(("${cmake_instance}" MATCHES "gemm_universal_preshuffle|gemm_blockscale|gemm_ab_scale" AND "${cmake_instance}" MATCHES "_f8_" ) AND (NOT INST_TARGETS MATCHES "gfx94|gfx95|gfx12") AND (NOT CK_USE_FP8_ON_UNSUPPORTED_ARCH))
            message(DEBUG "Found gemm_f8 instances, but gfx94/gfx95 not on the target list. Skipping.")
            set(add_inst 0)
        endif()
        if ("${cmake_instance}" MATCHES "gemm_bilinear")
            set(add_inst 0)
            if((SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12") AND (DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES))
                set(add_inst 1)
            endif()
            if((SUPPORTED_GPU_TARGETS MATCHES "gfx1[12]") AND (DTYPES MATCHES "int8" OR NOT DEFINED DTYPES))
                set(add_inst 1)
            endif()
        endif()

        # Build the required pattern based on library settings
        set(required_pattern "")
        set(pattern_parts "")
        if(MIOPEN_REQ_LIBS_ONLY)
            message(STATUS "Removing all sources that are not required for MIOpen")
            list(APPEND pattern_parts "conv")
        endif()
        if(HIPTENSOR_REQ_LIBS_ONLY)
            message(STATUS "Removing all sources that are not required for HipTensor")
            list(APPEND pattern_parts "contract" "reduce" "element")
        endif()
        if(pattern_parts)
            string(JOIN "|" required_pattern ${pattern_parts})
        endif()
        # Apply the pattern if one was set
        if(required_pattern AND NOT "${cmake_instance}" MATCHES "${required_pattern}")
            set(add_inst 0)
        endif()

        if((add_inst EQUAL 1))
            get_filename_component(target_dir ${subdir_path} NAME)

            # Skip convolution operations that are now split by layout
            # These are handled explicitly below
            set(layout_split_ops
                "grouped_conv1d_fwd"
                "grouped_conv1d_bwd_weight"
                "grouped_conv2d_fwd"
                "grouped_conv2d_bwd_data"
                "grouped_conv2d_bwd_weight"
                "grouped_conv2d_fwd_clamp"
                "grouped_conv2d_fwd_bias_clamp"
                "grouped_conv2d_fwd_bias_bnorm_clamp"
                "grouped_conv2d_fwd_dynamic_op"
                "grouped_conv3d_fwd"
                "grouped_conv3d_bwd_data"
                "grouped_conv3d_bwd_weight"
                "grouped_conv3d_fwd_scale"
                "grouped_conv3d_fwd_clamp"
                "grouped_conv3d_fwd_bias_clamp"
                "grouped_conv3d_fwd_bias_bnorm_clamp"
                "grouped_conv3d_fwd_bilinear"
                "grouped_conv3d_fwd_convscale"
                "grouped_conv3d_fwd_convscale_add"
                "grouped_conv3d_fwd_convscale_relu"
                "grouped_conv3d_fwd_convinvscale"
                "grouped_conv3d_fwd_dynamic_op"
                "grouped_conv3d_fwd_scaleadd_ab"
                "grouped_conv3d_fwd_scaleadd_scaleadd_relu"
                "grouped_conv3d_bwd_data_bilinear"
                "grouped_conv3d_bwd_data_scale"
                "grouped_conv3d_bwd_weight_bilinear"
                "grouped_conv3d_bwd_weight_scale"
            )

            if(NOT target_dir IN_LIST layout_split_ops)
                add_subdirectory(${target_dir})
                if (TARGET device_${target_dir}_instance)
                    if("${cmake_instance}" MATCHES "gemm")
                        list(APPEND CK_DEVICE_GEMM_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    elseif("${cmake_instance}" MATCHES "conv")
                        list(APPEND CK_DEVICE_CONV_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    elseif("${cmake_instance}" MATCHES "mha")
                        list(APPEND CK_DEVICE_MHA_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    elseif("${cmake_instance}" MATCHES "contr")
                        list(APPEND CK_DEVICE_CONTRACTION_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    elseif("${cmake_instance}" MATCHES "reduce")
                        list(APPEND CK_DEVICE_REDUCTION_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    else()
                        list(APPEND CK_DEVICE_OTHER_INSTANCES $<TARGET_OBJECTS:device_${target_dir}_instance>)
                    endif()
                    message(DEBUG "add_instance_directory ${subdir_path}")
                endif()
            endif()  # NOT IN_LIST check
        else()
            message(DEBUG "skip_instance_directory ${subdir_path}")
        endif()
    ENDIF()
ENDFOREACH()

# =============================================================================
# Layout-Split Convolution Operations (Explicit Includes)
# =============================================================================
# These operations are now split by data layout for selective linking.
# Each operation includes multiple layout subdirectories (gnhwc, nhwgc, etc.)

if(NOT HIPTENSOR_REQ_LIBS_ONLY OR MIOPEN_REQ_LIBS_ONLY)
    message(STATUS "Including layout-split convolution operations...")

    # 1D convolutions
    add_subdirectory(grouped_conv1d_fwd)
    add_subdirectory(grouped_conv1d_bwd_weight)

    # 2D convolutions - core operations
    add_subdirectory(grouped_conv2d_fwd)
    add_subdirectory(grouped_conv2d_bwd_data)
    add_subdirectory(grouped_conv2d_bwd_weight)

    # 2D convolutions - variants
    add_subdirectory(grouped_conv2d_fwd_clamp)
    add_subdirectory(grouped_conv2d_fwd_bias_clamp)
    add_subdirectory(grouped_conv2d_fwd_bias_bnorm_clamp)
    add_subdirectory(grouped_conv2d_fwd_dynamic_op)

    # 3D convolutions - core operations
    add_subdirectory(grouped_conv3d_fwd)
    add_subdirectory(grouped_conv3d_bwd_data)
    add_subdirectory(grouped_conv3d_bwd_weight)

    # 3D convolutions - variants
    add_subdirectory(grouped_conv3d_fwd_scale)
    add_subdirectory(grouped_conv3d_fwd_clamp)
    add_subdirectory(grouped_conv3d_fwd_bias_clamp)
    add_subdirectory(grouped_conv3d_fwd_bias_bnorm_clamp)
    add_subdirectory(grouped_conv3d_fwd_bilinear)
    add_subdirectory(grouped_conv3d_fwd_convscale)
    add_subdirectory(grouped_conv3d_fwd_convscale_add)
    add_subdirectory(grouped_conv3d_fwd_convscale_relu)
    add_subdirectory(grouped_conv3d_fwd_convinvscale)
    add_subdirectory(grouped_conv3d_fwd_dynamic_op)
    add_subdirectory(grouped_conv3d_fwd_scaleadd_ab)
    add_subdirectory(grouped_conv3d_fwd_scaleadd_scaleadd_relu)
    add_subdirectory(grouped_conv3d_bwd_data_bilinear)
    add_subdirectory(grouped_conv3d_bwd_data_scale)
    add_subdirectory(grouped_conv3d_bwd_weight_bilinear)
    add_subdirectory(grouped_conv3d_bwd_weight_scale)
endif()

# =============================================================================
# End of Layout-Split Operations
# =============================================================================




if(CK_DEVICE_OTHER_INSTANCES AND NOT MIOPEN_REQ_LIBS_ONLY)
        add_library(device_other_operations ${CK_DEVICE_OTHER_INSTANCES})
        add_library(composablekernels::device_other_operations ALIAS device_other_operations)
        set_target_properties(device_other_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_other_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_other_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/problem_transform>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device/impl>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/grid>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/block>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/warp>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/thread>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/element>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/quantization>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/softmax>
        )
        rocm_install(TARGETS device_other_operations
            EXPORT device_other_operationsTargets)
        rocm_install(EXPORT device_other_operationsTargets
            FILE composable_kerneldevice_other_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
if(CK_DEVICE_GEMM_INSTANCES AND NOT MIOPEN_REQ_LIBS_ONLY AND NOT HIPTENSOR_REQ_LIBS_ONLY)
        add_library(device_gemm_operations ${CK_DEVICE_GEMM_INSTANCES})
        add_library(composablekernels::device_gemm_operations ALIAS device_gemm_operations)
        target_compile_features(device_gemm_operations PUBLIC)
        set_target_properties(device_gemm_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(device_gemm_operations
            PROPERTIES
            VERSION ${CMAKE_PROJECT_VERSION}
            SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
        )
        target_include_directories(device_gemm_operations PUBLIC
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
        )
        rocm_install(TARGETS device_gemm_operations
            EXPORT device_gemm_operationsTargets)
        rocm_install(EXPORT device_gemm_operationsTargets
            FILE composable_kerneldevice_gemm_operationsTargets.cmake
            NAMESPACE composable_kernel::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
        )
endif()
# OLD: Monolithic device_conv_operations (replaced by layout-specific libraries)
# if(CK_DEVICE_CONV_INSTANCES AND (NOT HIPTENSOR_REQ_LIBS_ONLY OR MIOPEN_REQ_LIBS_ONLY))
#         add_library(device_conv_operations ${CK_DEVICE_CONV_INSTANCES})
#         add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
#         target_compile_features(device_conv_operations PUBLIC)
#         set_target_properties(device_conv_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
#         set_target_properties(device_conv_operations
#             PROPERTIES
#             VERSION ${CMAKE_PROJECT_VERSION}
#             SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
#         )
#         target_include_directories(device_conv_operations PUBLIC
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd>
#         )
#         rocm_install(TARGETS device_conv_operations
#             EXPORT device_conv_operationsTargets)
#         rocm_install(EXPORT device_conv_operationsTargets
#             FILE composable_kerneldevice_conv_operationsTargets.cmake
#             NAMESPACE composable_kernel::
#             DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
#         )
# endif()
# if(CK_DEVICE_MHA_INSTANCES AND NOT MIOPEN_REQ_LIBS_ONLY AND NOT HIPTENSOR_REQ_LIBS_ONLY AND BUILD_MHA_LIB)
#         set(gpu_list ${INST_TARGETS})
#         if(gpu_list MATCHES "gfx94" OR gpu_list MATCHES "gfx90a" OR gpu_list MATCHES "gfx95")
#             add_library(device_mha_operations ${CK_DEVICE_MHA_INSTANCES})
#             set_target_properties(device_mha_operations
#                 PROPERTIES
#                 VERSION ${CMAKE_PROJECT_VERSION}
#                 SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
#             )
#             add_library(composablekernels::device_mha_operations ALIAS device_mha_operations)
#             target_compile_features(device_mha_operations PUBLIC)
#             set_target_properties(device_mha_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)

#             rocm_install(TARGETS device_mha_operations
#                 EXPORT device_mha_operationsTargets)
#             rocm_install(EXPORT device_mha_operationsTargets
#                 FILE composable_kerneldevice_mha_operationsTargets.cmake
#                 NAMESPACE composable_kernel::
#                 DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
#             )
#         endif()
# endif()
# if(CK_DEVICE_CONTRACTION_INSTANCES AND NOT MIOPEN_REQ_LIBS_ONLY)
#         add_library(device_contraction_operations ${CK_DEVICE_CONTRACTION_INSTANCES})
#         add_library(composablekernels::device_contraction_operations ALIAS device_contraction_operations)
#         target_compile_features(device_contraction_operations PUBLIC)
#         set_target_properties(device_contraction_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
#         set_target_properties(device_contraction_operations
#             PROPERTIES
#             VERSION ${CMAKE_PROJECT_VERSION}
#             SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
#         )
#         target_include_directories(device_contraction_operations PUBLIC
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/contraction>
#         )
#         rocm_install(TARGETS device_contraction_operations
#             EXPORT device_contraction_operationsTargets)
#         rocm_install(EXPORT device_contraction_operationsTargets
#             FILE composable_kerneldevice_contraction_operationsTargets.cmake
#             NAMESPACE composable_kernel::
#             DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
#         )
# endif()
# if(CK_DEVICE_REDUCTION_INSTANCES AND NOT MIOPEN_REQ_LIBS_ONLY)
#         add_library(device_reduction_operations ${CK_DEVICE_REDUCTION_INSTANCES})
#         add_library(composablekernels::device_reduction_operations ALIAS device_reduction_operations)
#         target_compile_features(device_reduction_operations PUBLIC)
#         set_target_properties(device_reduction_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
#         set_target_properties(device_reduction_operations
#             PROPERTIES
#             VERSION ${CMAKE_PROJECT_VERSION}
#             SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
#         )
#         target_include_directories(device_reduction_operations PUBLIC
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/reduce>
#         )
#         rocm_install(TARGETS device_reduction_operations
#             EXPORT device_reduction_operationsTargets)
#         rocm_install(EXPORT device_reduction_operationsTargets
#             FILE composable_kerneldevice_reduction_operationsTargets.cmake
#             NAMESPACE composable_kernel::
#             DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
#         )
# endif()

# if(NOT MIOPEN_REQ_LIBS_ONLY AND NOT HIPTENSOR_REQ_LIBS_ONLY)
#     add_library(device_operations INTERFACE)
#     target_link_libraries(device_operations INTERFACE
#         device_contraction_operations
#         device_conv_operations
#         device_gemm_operations
#         device_other_operations
#         device_reduction_operations
#         utility)
# endif()

# set(DEV_OPS_INC_DIRS
#     ${PROJECT_SOURCE_DIR}/include/ck/
#     ${PROJECT_SOURCE_DIR}/library/include/ck/
# )
# rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)

# =============================================================================
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-Licen# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Layout-Specific Convolution Libraries (Static Libraries with Conditional Sources)
# =============================================================================

# Helper macro to conditionally add target objects
macro(ADD_TARGET_OBJECTS_IF_EXISTS VAR_NAME TARGET_NAME)
    if(TARGET ${TARGET_NAME})
        list(APPEND ${VAR_NAME} $<TARGET_OBJECTS:${TARGET_NAME}>)
    endif()
endmacro()

# -----------------------------------------------------------------------------
# 2D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# NHWGC Layout (2D) - ⭐ Required by MIOpen
set(CONV2D_NHWGC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_fwd_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_bwd_data_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_bwd_weight_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_fwd_clamp_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_fwd_bias_clamp_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_fwd_bias_bnorm_clamp_nhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NHWGC_OBJECTS device_grouped_conv2d_fwd_dynamic_op_nhwgc_instance)
if(CONV2D_NHWGC_OBJECTS)
    add_library(device_conv2d_nhwgc_operations ${CONV2D_NHWGC_OBJECTS})
    set_target_properties(device_conv2d_nhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv2d_nhwgc_operations ALIAS device_conv2d_nhwgc_operations)
    message(STATUS "Created device_conv2d_nhwgc_operations (MIOpen required) with ${CONV2D_NHWGC_OBJECTS}")
endif()

# GNHWC Layout (2D)
set(CONV2D_GNHWC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_GNHWC_OBJECTS device_grouped_conv2d_fwd_gnhwc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_GNHWC_OBJECTS device_grouped_conv2d_bwd_data_gnhwc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_GNHWC_OBJECTS device_grouped_conv2d_bwd_weight_gnhwc_instance)
if(CONV2D_GNHWC_OBJECTS)
    add_library(device_conv2d_gnhwc_operations ${CONV2D_GNHWC_OBJECTS})
    set_target_properties(device_conv2d_gnhwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv2d_gnhwc_operations ALIAS device_conv2d_gnhwc_operations)
    message(STATUS "Created device_conv2d_gnhwc_operations")
endif()

# NGCHW Layout (2D)
set(CONV2D_NGCHW_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NGCHW_OBJECTS device_grouped_conv2d_fwd_ngchw_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NGCHW_OBJECTS device_grouped_conv2d_bwd_data_ngchw_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV2D_NGCHW_OBJECTS device_grouped_conv2d_bwd_weight_ngchw_instance)
if(CONV2D_NGCHW_OBJECTS)
    add_library(device_conv2d_ngchw_operations ${CONV2D_NGCHW_OBJECTS})
    set_target_properties(device_conv2d_ngchw_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv2d_ngchw_operations ALIAS device_conv2d_ngchw_operations)
    message(STATUS "Created device_conv2d_ngchw_operations")
endif()

# -----------------------------------------------------------------------------
# 3D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# NDHWGC Layout (3D) - ⭐ Required by MIOpen
set(CONV3D_NDHWGC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_data_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_weight_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_scale_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_clamp_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_bias_clamp_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_bias_bnorm_clamp_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_bilinear_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_convscale_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_convscale_add_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_convscale_relu_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_convinvscale_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_dynamic_op_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_scaleadd_ab_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_fwd_scaleadd_scaleadd_relu_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_data_bilinear_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_data_scale_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_weight_bilinear_ndhwgc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NDHWGC_OBJECTS device_grouped_conv3d_bwd_weight_scale_ndhwgc_instance)
if(CONV3D_NDHWGC_OBJECTS)
    add_library(device_conv3d_ndhwgc_operations ${CONV3D_NDHWGC_OBJECTS})
    set_target_properties(device_conv3d_ndhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv3d_ndhwgc_operations ALIAS device_conv3d_ndhwgc_operations)
    message(STATUS "Created device_conv3d_ndhwgc_operations (MIOpen required)")
endif()

# GNDHWC Layout (3D)
set(CONV3D_GNDHWC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_GNDHWC_OBJECTS device_grouped_conv3d_fwd_gndhwc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_GNDHWC_OBJECTS device_grouped_conv3d_bwd_data_gndhwc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_GNDHWC_OBJECTS device_grouped_conv3d_bwd_weight_gndhwc_instance)
if(CONV3D_GNDHWC_OBJECTS)
    add_library(device_conv3d_gndhwc_operations ${CONV3D_GNDHWC_OBJECTS})
    set_target_properties(device_conv3d_gndhwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv3d_gndhwc_operations ALIAS device_conv3d_gndhwc_operations)
    message(STATUS "Created device_conv3d_gndhwc_operations")
endif()

# NGCDHW Layout (3D)
set(CONV3D_NGCDHW_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NGCDHW_OBJECTS device_grouped_conv3d_fwd_ngcdhw_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NGCDHW_OBJECTS device_grouped_conv3d_bwd_data_ngcdhw_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NGCDHW_OBJECTS device_grouped_conv3d_bwd_weight_ngcdhw_instance)
if(CONV3D_NGCDHW_OBJECTS)
    add_library(device_conv3d_ngcdhw_operations ${CONV3D_NGCDHW_OBJECTS})
    set_target_properties(device_conv3d_ngcdhw_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv3d_ngcdhw_operations ALIAS device_conv3d_ngcdhw_operations)
    message(STATUS "Created device_conv3d_ngcdhw_operations")
endif()

# NHWGC Layout (3D)
set(CONV3D_NHWGC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV3D_NHWGC_OBJECTS device_grouped_conv3d_fwd_bias_bnorm_clamp_nhwgc_instance)
if(CONV3D_NHWGC_OBJECTS)
    add_library(device_conv3d_nhwgc_operations ${CONV3D_NHWGC_OBJECTS})
    set_target_properties(device_conv3d_nhwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv3d_nhwgc_operations ALIAS device_conv3d_nhwgc_operations)
    message(STATUS "Created device_conv3d_nhwgc_operations")
endif()

# -----------------------------------------------------------------------------
# 1D Convolution Layout Libraries
# -----------------------------------------------------------------------------

# GNWC Layout (1D)
set(CONV1D_GNWC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV1D_GNWC_OBJECTS device_grouped_conv1d_fwd_gnwc_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV1D_GNWC_OBJECTS device_grouped_conv1d_bwd_weight_gnwc_instance)
if(CONV1D_GNWC_OBJECTS)
    add_library(device_conv1d_gnwc_operations ${CONV1D_GNWC_OBJECTS})
    set_target_properties(device_conv1d_gnwc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv1d_gnwc_operations ALIAS device_conv1d_gnwc_operations)
    message(STATUS "Created device_conv1d_gnwc_operations")
endif()

# NWGC Layout (1D)
set(CONV1D_NWGC_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV1D_NWGC_OBJECTS device_grouped_conv1d_bwd_weight_nwgc_instance)
if(CONV1D_NWGC_OBJECTS)
    add_library(device_conv1d_nwgc_operations ${CONV1D_NWGC_OBJECTS})
    set_target_properties(device_conv1d_nwgc_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv1d_nwgc_operations ALIAS device_conv1d_nwgc_operations)
    message(STATUS "Created device_conv1d_nwgc_operations")
endif()

# -----------------------------------------------------------------------------
# Special Libraries (not split by layout)
# -----------------------------------------------------------------------------

# Non-grouped convolutions (old/legacy) - ⭐ Required by MIOpen
# Combine all non-grouped conv instance libraries into one static library
set(CONV_OLD_OBJECTS)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv1d_bwd_data_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv2d_bwd_data_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv2d_fwd_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv2d_fwd_bias_relu_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv2d_fwd_bias_relu_add_instance)
ADD_TARGET_OBJECTS_IF_EXISTS(CONV_OLD_OBJECTS device_conv3d_bwd_data_instance)
if(CONV_OLD_OBJECTS)
    add_library(device_conv_old_operations ${CONV_OLD_OBJECTS})
    set_target_properties(device_conv_old_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_conv_old_operations ALIAS device_conv_old_operations)
    message(STATUS "Created device_conv_old_operations (MIOpen required)")
endif()

# N-dimensional generic convolutions (layout-agnostic)
if(TARGET device_grouped_convnd_bwd_weight_instance)
    add_library(device_convnd_generic_operations
        $<TARGET_OBJECTS:device_grouped_convnd_bwd_weight_instance>
    )
    set_target_properties(device_convnd_generic_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_convnd_generic_operations ALIAS device_convnd_generic_operations)
    message(STATUS "Created device_convnd_generic_operations")
endif()

# Quantization operations
if(TARGET device_quantization_instance)
    add_library(device_quantization_operations
        $<TARGET_OBJECTS:device_quantization_instance>
    )
    set_target_properties(device_quantization_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(composablekernels::device_quantization_operations ALIAS device_quantization_operations)
    message(STATUS "Created device_quantization_operations")
endif()

# -----------------------------------------------------------------------------
# Umbrella Library (Backward Compatibility)
# -----------------------------------------------------------------------------
# Combines all layout libraries - use INTERFACE since layout libs are now static

add_library(device_conv_operations INTERFACE)

# Link all layout libraries that exist
if(TARGET device_conv1d_gnwc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv1d_gnwc_operations)
endif()
if(TARGET device_conv1d_nwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv1d_nwgc_operations)
endif()
if(TARGET device_conv2d_gnhwc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv2d_gnhwc_operations)
endif()
if(TARGET device_conv2d_nhwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv2d_nhwgc_operations)
endif()
if(TARGET device_conv2d_ngchw_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv2d_ngchw_operations)
endif()
if(TARGET device_conv3d_gndhwc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv3d_gndhwc_operations)
endif()
if(TARGET device_conv3d_ndhwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv3d_ndhwgc_operations)
endif()
if(TARGET device_conv3d_ngcdhw_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv3d_ngcdhw_operations)
endif()
if(TARGET device_conv3d_nhwgc_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv3d_nhwgc_operations)
endif()
if(TARGET device_conv_old_operations)
    target_link_libraries(device_conv_operations INTERFACE device_conv_old_operations)
endif()
if(TARGET device_convnd_generic_operations)
    target_link_libraries(device_conv_operations INTERFACE device_convnd_generic_operations)
endif()
if(TARGET device_quantization_operations)
    target_link_libraries(device_conv_operations INTERFACE device_quantization_operations)
endif()

add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
message(STATUS "Created umbrella device_conv_operations (includes all layouts)")

# =============================================================================
# End of Layout-Specific Libraries
# =============================================================================

# =============================================================================
