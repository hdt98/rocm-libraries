################################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#
################################################################################

# INTERNAL BUILD-ONLY library — not installed, not exported, not part of the public MIOpen API.
# Header-only pure C++ utilities shared by MIOpen, MIOpenDriver, and tests.
# Contains NO MIOpen or GPU dependencies.
# Do NOT add install(TARGETS miopen_common_utils ...) — headers live in the build tree only.

add_library(miopen_common_utils INTERFACE)
set_target_properties(miopen_common_utils PROPERTIES EXCLUDE_FROM_ALL TRUE)

target_include_directories(miopen_common_utils INTERFACE
    # BUILD_INTERFACE only — no install interface; these headers are not installed.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# bfloat16.hpp needs MIOPEN_USE_RNE_BFLOAT16 at compile time.
# The option is declared in the top-level CMakeLists.txt.
if(MIOPEN_USE_RNE_BFLOAT16)
    target_compile_definitions(miopen_common_utils INTERFACE MIOPEN_USE_RNE_BFLOAT16=1)
else()
    target_compile_definitions(miopen_common_utils INTERFACE MIOPEN_USE_RNE_BFLOAT16=0)
endif()
