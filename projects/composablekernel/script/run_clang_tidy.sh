#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Runs clang-tidy with the readability-avoid-unconditional-preprocessor-if check
# on files introduced or modified by the current branch relative to the base branch,
# using a 3-dot git diff (git diff origin/<base>...HEAD).
#
# If compile_commands.json is not found under a build directory, a minimal cmake
# configure (no build) is run automatically to produce it.
#
# Called by pre-commit with pass_filenames: false.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECK="readability-avoid-unconditional-preprocessor-if"

# --- locate clang-tidy binary ---------------------------------------------
# Prefer 'clang-tidy' (installed via pip/additional_dependencies in pre-commit),
# fall back to 'clang-tidy-18' (system apt package).
CLANG_TIDY=""
for candidate in clang-tidy clang-tidy-18; do
    if command -v "${candidate}" &>/dev/null; then
        CLANG_TIDY="${candidate}"
        break
    fi
done

if [ -z "${CLANG_TIDY}" ]; then
    echo "clang-tidy: WARNING: no clang-tidy binary found, skipping check."
    echo "  Install with: apt-get install clang-tidy-18"
    exit 0
fi

echo "clang-tidy: using $(${CLANG_TIDY} --version | head -1)"

# --- determine base branch ------------------------------------------------
# Try common base branch names in order; fall back to 'develop'.
BASE_BRANCH=""
for candidate in develop main master; do
    if git rev-parse --verify "origin/${candidate}" &>/dev/null 2>&1; then
        BASE_BRANCH="${candidate}"
        break
    fi
done
BASE_BRANCH="${BASE_BRANCH:-develop}"

# --- collect changed C++ files via 3-dot diff -----------------------------
# --diff-filter=d  : exclude deleted files (clang-tidy cannot run on them)
# -- '*.cpp' '*.hpp': restrict to C++ source and header files
mapfile -t CHANGED_FILES < <(
    git -C "${PROJECT_ROOT}" diff --name-only --diff-filter=d \
        "origin/${BASE_BRANCH}...HEAD" \
        -- '*.cpp' '*.hpp' 2>/dev/null \
    | grep -v 'include/rapidjson'
)

if [ ${#CHANGED_FILES[@]} -eq 0 ]; then
    echo "clang-tidy: no C++ files changed relative to origin/${BASE_BRANCH}, skipping."
    exit 0
fi

echo "clang-tidy: checking ${#CHANGED_FILES[@]} file(s) against origin/${BASE_BRANCH}...HEAD"

# --- locate compile_commands.json -----------------------------------------
BUILD_DIR=""
for candidate in build build_rel build_release build_debug; do
    if [ -f "${PROJECT_ROOT}/${candidate}/compile_commands.json" ]; then
        BUILD_DIR="${PROJECT_ROOT}/${candidate}"
        break
    fi
done

if [ -z "${BUILD_DIR}" ]; then
    if ! command -v cmake &>/dev/null; then
        echo "clang-tidy: ERROR: compile_commands.json not found and cmake is unavailable."
        echo "  Run cmake configure in the project build directory first."
        exit 1
    fi

    echo "clang-tidy: compile_commands.json not found; running minimal cmake configure..."
    BUILD_DIR="${PROJECT_ROOT}/build"
    mkdir -p "${BUILD_DIR}"

    GPU_TARGET=$(rocminfo 2>/dev/null | awk '/^\s+Name: +gfx/{print $2; exit}')
    GPU_TARGET="${GPU_TARGET:-gfx942}"

    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=release \
        -DBUILD_DEV=On \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DGPU_TARGETS="${GPU_TARGET}" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DENABLE_CLANG_CPP_CHECKS=OFF \
        > "${BUILD_DIR}/cmake_configure.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "clang-tidy: ERROR: cmake configure failed. See ${BUILD_DIR}/cmake_configure.log"
        exit 1
    fi
    echo "clang-tidy: cmake configure done (${BUILD_DIR})."
fi

# --- run clang-tidy -------------------------------------------------------
# Paths from git diff are relative to the git root (not PROJECT_ROOT).
GIT_ROOT="$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel)"
exit_code=0

for file in "${CHANGED_FILES[@]}"; do
    abs_file="${GIT_ROOT}/${file}"
    [ -f "${abs_file}" ] || continue

    if ! "${CLANG_TIDY}" \
            -p "${BUILD_DIR}" \
            -checks="-*,${CHECK}" \
            -warnings-as-errors="${CHECK}" \
            "${abs_file}"; then
        exit_code=1
    fi
done

exit ${exit_code}
