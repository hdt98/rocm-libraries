# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Support both standalone (scikit-build-core / tox) and subdirectory
# (hipblaslt CMake) builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.25)
    project(rocisa LANGUAGES CXX)
    set(ROCISA_STANDALONE ON)
else()
    set(ROCISA_STANDALONE OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rocisa-cpp INTERFACE)
add_library(rocisa::rocisa-cpp ALIAS rocisa-cpp)
target_sources(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include/enum.hpp>
)
target_include_directories(rocisa-cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Build the Python extension when bundling deps (hipblaslt subdir build)
# or when building standalone.
if(HIPBLASLT_BUNDLE_PYTHON_DEPS OR ROCISA_STANDALONE)
    if(ROCISA_STANDALONE)
        # scikit-build-core pre-installs nanobind (from build-system.requires)
        # and injects its CMake path, so find_package works without FetchContent.
        # nanobind requires Python to be found first.
        find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
        find_package(nanobind REQUIRED CONFIG)
    else()
        include(FetchContent)
        FetchContent_Declare(
          nanobind
          GIT_REPOSITORY https://github.com/wjakob/nanobind.git
          GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
        )
        FetchContent_MakeAvailable(nanobind)
    endif()

    # HIP is required to build _rocisa (via hip::host).  Set the search prefix
    # so find_package works both in developer environments and in the temporary
    # build sandbox created by scikit-build-core / uv.
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip REQUIRED)

    # Locate origami: prefer the monorepo sibling, fall back to installed package.
    if(NOT TARGET roc::origami)
        set(_origami_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/origami")
        if(IS_DIRECTORY "${_origami_sibling}")
            add_subdirectory("${_origami_sibling}" origami EXCLUDE_FROM_ALL)
        else()
            find_package(origami REQUIRED)
        endif()
    endif()

    set(ROCISAINST_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/common.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mem.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/instruction/extension.cpp")
    set(ROCISAFUNC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/functions.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/argument.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_math.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_branch.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/functions/f_cast.cpp")
    set(ROCISAPASS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/pass.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/graph.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/composite.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/cycle.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/remove.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/pass/insert_delay_alu.cpp")

    nanobind_add_module(_rocisa NOMINSIZE NB_SUPPRESS_WARNINGS
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/main.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/base.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/code.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/container.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/count.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/enum.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/helper.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/label.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/macro.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/src/register.cpp"
                            ${ROCISAINST_SOURCE}
                            ${ROCISAPASS_SOURCE}
                            ${ROCISAFUNC_SOURCE})
    target_include_directories(_rocisa PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/include"
    )

    # Optional inclusion of external shared/stinkytofu CMake project
    set(STINKYTOFU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/stinkytofu")
    if(EXISTS "${STINKYTOFU_DIR}/CMakeLists.txt")
        set(BUILD_SHARED_LIBS ON)
        # Force the stinkytofu Python binding ON before we descend.
        # ``shared/stinkytofu`` keeps ``STINKYTOFU_BUILD_PYTHON`` defaulted
        # OFF in sub-project mode (see shared/stinkytofu/CMakeLists.txt:157)
        # to stay neutral for consumers that don't need Python. The
        # rocisa→stinkytofu adapter (``ROCISA_BACKEND=stinkytofu``) DOES
        # need it: ``rocisa_stinkytofu_adaptor.code`` calls
        # ``stinkytofu.getSrdUpperValue125X / getSrdUpperDesc125X`` from
        # the ``_stinkytofu.so`` nanobind module. Forcing the option here
        # (in the consumer) keeps the dependency direction one-way.
        set(STINKYTOFU_BUILD_PYTHON ON CACHE BOOL
            "Build StinkyTofu Python bindings (required by ROCISA_BACKEND=stinkytofu adapter)"
            FORCE)
        # Add the external directory; build output goes into a local binary dir 'stinkytofu'
        add_subdirectory("${STINKYTOFU_DIR}" stinkytofu EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "External directory not found: ${STINKYTOFU_DIR} (skipping)")
    endif()

    target_link_libraries(_rocisa PUBLIC roc::origami)
    target_link_libraries(_rocisa PRIVATE hip::host stinkytofu)
    # rocisa conversion sources include stinkytofu public headers that
    # transitively include generated files (hardware/gfxIsa.inc) from
    # stinkytofu's build directory.
    target_include_directories(_rocisa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu")
    set_target_properties(_rocisa PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/rocisa"
        # At install time (wheel/site-packages): libstinkytofu.so sits next to
        # _rocisa.so in the same rocisa/ directory.
        INSTALL_RPATH "$ORIGIN"
        # At build time: libstinkytofu.so is in the stinkytofu/ subdirectory of
        # the binary dir, not co-located with _rocisa.so.
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu"
    )
    # On Windows there is no RPATH; copy stinkytofu.dll next to _rocisa.pyd
    # so the dynamic linker can find it at runtime.
    if(WIN32)
        add_custom_command(TARGET _rocisa POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:stinkytofu>
                $<TARGET_FILE_DIR:_rocisa>
        )
    endif()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/__init__.py"
        COPYONLY
    )
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/_build_info.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/rocisa/_build_info.py"
        @ONLY
    )
    # _build_info.py is intentionally NOT in install() rules so it is absent
    # from wheels and pre-built packages, acting as the source-build sentinel.

    # ---------------------------------------------------------------------
    # Sibling staging for the stinkytofu Python package.
    #
    # ``ROCISA_BACKEND=stinkytofu`` activates ``rocisa_stinkytofu_adaptor``
    # which imports ``stinkytofu.getSrdUpperValue125X`` etc. from the
    # ``_stinkytofu.so`` nanobind module. We want the consumer to be able to
    # do
    #
    #     PYTHONPATH=<binary_dir>/tensilelite/rocisa python -m Tensile ...
    #
    # and have BOTH ``import rocisa`` and ``import stinkytofu`` succeed —
    # the same PYTHONPATH that already locates ``rocisa/_rocisa.so`` should
    # also locate ``stinkytofu/_stinkytofu.so``. To make that true:
    #
    #   1. ``add_dependencies(_rocisa stinkytofu_python)`` defeats the
    #      ``EXCLUDE_FROM_ALL`` on the stinkytofu sub-project (line 113):
    #      ``stinkytofu_python`` only gets built because ``_rocisa`` (which
    #      IS in ALL) now depends on it.
    #   2. Override ``LIBRARY_OUTPUT_DIRECTORY`` from python_module's
    #      default of ``<top_binary>/lib/stinkytofu/`` to
    #      ``<here>/stinkytofu/`` so the .so lands next to libstinkytofu.so
    #      (BUILD_RPATH = $ORIGIN, no RPATH gymnastics needed).
    #   3. Re-stage ``__init__.py`` and ``_build_info.py`` next to the .so;
    #      python_module's own ``configure_file`` calls put them in
    #      ``<top_binary>/lib/stinkytofu/`` which is the wrong location for
    #      the rocisa-adjacent layout. The staleness check in
    #      ``__init__.py`` then sees a fresh ``_build_info.py`` pointing at
    #      our build dir and works correctly.
    #
    # Other sub-project consumers of stinkytofu are unaffected — we only
    # touch the ``stinkytofu_python`` target's properties from the rocisa
    # side, and only when that target exists (gated on STINKYTOFU_BUILD_PYTHON).
    # ---------------------------------------------------------------------
    if(TARGET stinkytofu_python)
        add_dependencies(_rocisa stinkytofu_python)

        set_target_properties(stinkytofu_python PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu"
            # libstinkytofu.so lives in the same directory at build time,
            # so $ORIGIN is enough; matches the install-time layout used
            # for _rocisa above.
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )

        # ``_build_info.py.in`` reads ``@STINKYTOFU_TOP_DIR@`` and
        # ``@CMAKE_BINARY_DIR@``. The former is sub-project-scoped inside
        # shared/stinkytofu/CMakeLists.txt (line 36) and not visible here,
        # so re-set it locally before configure_file. CMAKE_BINARY_DIR is
        # the top binary dir (the tensilelite preset's build root) which is
        # what the staleness check needs to skip its own generated files.
        set(STINKYTOFU_TOP_DIR "${STINKYTOFU_DIR}")
        configure_file(
            "${STINKYTOFU_DIR}/python_module/stinkytofu/__init__.py"
            "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/__init__.py"
            COPYONLY
        )
        configure_file(
            "${STINKYTOFU_DIR}/python_module/stinkytofu/_build_info.py.in"
            "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/_build_info.py"
            @ONLY
        )

        # -----------------------------------------------------------------
        # intrinsics.st.bc — runtime bitcode loaded by IntrinsicRegistry
        # at _stinkytofu.so import time
        # (shared/stinkytofu/src/ir/logical/IntrinsicRegistry.cpp:39-54).
        # Without it the module prints
        #   "IntrinsicRegistry: intrinsics.st.bc not found in search paths"
        # and any IntrinsicExpansionPass user (e.g. ``Intrinsic('ReluF32',
        # ...)``) eventually fails. ``rocisa_stinkytofu_adaptor`` doesn't
        # touch intrinsics today, but the warning is noisy and the
        # follow-on breakage is invisible until KernelWriter goes there,
        # so we wire it up alongside the rest of the stinkytofu sibling
        # staging to keep the integration self-contained.
        #
        # The bitcode is produced by the ``intrinsics_compiled`` custom
        # target (shared/stinkytofu/tools/intrinsic-compiler/CMakeLists.txt:40)
        # at ``${CMAKE_BINARY_DIR}/intrinsics.st.bc``. Two things here:
        #
        #   1. ``add_dependencies(_rocisa intrinsics_compiled)`` — same
        #      EXCLUDE_FROM_ALL bypass trick as for stinkytofu_python:
        #      _rocisa is in ALL, so intrinsics_compiled gets built.
        #
        #   2. Stage a copy alongside _stinkytofu.so. The search-path
        #      walk-up loop (IntrinsicRegistry.cpp:142-159) WOULD find
        #      the bc at ``new_dbg_build_mi450/intrinsics.st.bc`` (libDir
        #      walks up 2 levels), but that distance is fragile to any
        #      future build-tree restructuring. A same-dir copy hits the
        #      first search-path entry (libDir + "/intrinsics.st.bc",
        #      IntrinsicRegistry.cpp:138) and is robust.
        # -----------------------------------------------------------------
        if(TARGET intrinsics_compiled)
            add_dependencies(_rocisa intrinsics_compiled)

            set(_intrinsics_bc_src "${CMAKE_BINARY_DIR}/intrinsics.st.bc")
            set(_intrinsics_bc_dst "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu/intrinsics.st.bc")
            add_custom_command(
                OUTPUT "${_intrinsics_bc_dst}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_intrinsics_bc_src}" "${_intrinsics_bc_dst}"
                DEPENDS intrinsics_compiled "${_intrinsics_bc_src}"
                COMMENT "Staging intrinsics.st.bc next to _stinkytofu.so"
                VERBATIM
            )
            add_custom_target(rocisa_stage_intrinsics ALL
                DEPENDS "${_intrinsics_bc_dst}")
            add_dependencies(_rocisa rocisa_stage_intrinsics)
        endif()
    endif()

    if(ROCISA_STANDALONE)
        # Install rules for scikit-build-core: place _rocisa.so, libstinkytofu.so,
        # and __init__.py into the rocisa/ package directory in the wheel.
        install(TARGETS _rocisa DESTINATION rocisa)
        # Co-install libstinkytofu.so next to _rocisa.so so the dynamic linker
        # can find it at runtime (wheel layout: both files sit in rocisa/).
        install(FILES $<TARGET_FILE:stinkytofu> DESTINATION rocisa)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocisa/__init__.py" DESTINATION rocisa)
    endif()
endif()
