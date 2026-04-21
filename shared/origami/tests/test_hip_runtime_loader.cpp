/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <cstring>

#include "origami/hip_runtime_loader.hpp"

TEST_CASE("hip_runtime_loader: hip_get_error_string is never null", "[hip_runtime_loader]") {
  const char* s = origami::hip_get_error_string(hipSuccess);
  REQUIRE(s != nullptr);
  REQUIRE(std::strlen(s) > 0);
}

TEST_CASE("hip_runtime_loader: hip_runtime_available is idempotent", "[hip_runtime_loader]") {
  const bool a = origami::hip_runtime_available();
  const bool b = origami::hip_runtime_available();
  REQUIRE(a == b);
}

TEST_CASE("hip_runtime_loader: behavior when HIP runtime is unavailable", "[hip_runtime_loader]") {
  if(origami::hip_runtime_available()) return;

  hipDeviceProp_t prop{};
  REQUIRE(origami::hip_get_device_properties(&prop, 0) == hipErrorNotInitialized);

  int value = 0;
  REQUIRE(origami::hip_get_device_attribute(&value, hipDeviceAttributeWarpSize, 0)
          == hipErrorNotInitialized);
}

TEST_CASE("hip_runtime_loader: behavior when HIP runtime is available", "[hip_runtime_loader]") {
  if(!origami::hip_runtime_available()) return;

  hipDeviceProp_t prop{};
  const hipError_t props_err = origami::hip_get_device_properties(&prop, 0);
  REQUIRE((props_err == hipSuccess || props_err == hipErrorNoDevice
           || props_err == hipErrorInvalidDevice));

  if(props_err != hipSuccess) return;

  REQUIRE(prop.name[0] != '\0');

  int warp = 0;
  const hipError_t attr_err =
      origami::hip_get_device_attribute(&warp, hipDeviceAttributeWarpSize, 0);
  REQUIRE((attr_err == hipSuccess || attr_err == hipErrorNotSupported));
  if(attr_err == hipSuccess) REQUIRE(warp > 0);
}

TEST_CASE("hip_runtime_loader: invalid device index returns error", "[hip_runtime_loader]") {
  if(!origami::hip_runtime_available()) return;

  hipDeviceProp_t prop{};
  const hipError_t err = origami::hip_get_device_properties(&prop, 1'000'000);
  REQUIRE(err != hipSuccess);
}
