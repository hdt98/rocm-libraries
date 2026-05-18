# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

set(rocRAND_INTERNAL_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_log_normal_distribution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_normal_distribution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_poisson_distribution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_config_dispatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_discrete.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_generator_type.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_lfsr113_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_log_normal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mrg31k3p_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mrg32k3a_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mrg_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mt19937_octo_engine_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mt19937_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_mtgp32_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_normal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_philox_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_poisson.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_scrambled_sobol32_qrng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_scrambled_sobol64_qrng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_sobol32_qrng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_sobol64_qrng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_threefry2x32_20_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_threefry2x64_20_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_threefry4x32_20_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_threefry4x64_20_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand_xorwow_prng.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_uniform_distribution.cpp"
)

foreach(test_src ${rocRAND_INTERNAL_TEST_SRCS})
    add_test_target(${test_src} ON)
endforeach()
