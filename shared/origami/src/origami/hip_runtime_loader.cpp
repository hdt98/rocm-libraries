// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/hip_runtime_loader.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace origami::detail {

namespace {

using hipGetDeviceProperties_fn = hipError_t (*)(hipDeviceProp_t*, int);
using hipGetErrorString_fn      = const char* (*)(hipError_t);

#if defined(_WIN32)
using dl_handle_t = HMODULE;
#else
using dl_handle_t = void*;
#endif

dl_handle_t g_hip_lib                    = nullptr;
hipGetDeviceProperties_fn g_get_props    = nullptr;
hipGetErrorString_fn g_get_error_string  = nullptr;
std::string g_load_error;
std::mutex g_mutex;

std::vector<std::string> hip_library_candidates() {
  std::vector<std::string> out;
#if defined(_WIN32)
  if(const char* hip_path = std::getenv("HIP_PATH")) {
    std::string base(hip_path);
    if(!base.empty() && base.back() != '\\' && base.back() != '/') base += '\\';
    out.push_back(base + "bin\\amdhip64.dll");
  }
  out.push_back("amdhip64.dll");
#else
  if(const char* rocm = std::getenv("ROCM_PATH")) {
    std::string base(rocm);
    if(!base.empty() && base.back() != '/') base += '/';
    out.push_back(base + "lib/libamdhip64.so.6");
    out.push_back(base + "lib/libamdhip64.so");
  }
  if(const char* hip_path = std::getenv("HIP_PATH")) {
    std::string base(hip_path);
    if(!base.empty() && base.back() != '/') base += '/';
    out.push_back(base + "lib/libamdhip64.so.6");
    out.push_back(base + "lib/libamdhip64.so");
  }
  out.push_back("libamdhip64.so.6");
  out.push_back("libamdhip64.so");
#endif
  return out;
}

bool try_open_and_resolve() {
  if(g_get_props && g_get_error_string) return true;

  std::string last_open_error;
  for(const auto& path : hip_library_candidates()) {
#if defined(_WIN32)
    g_hip_lib = LoadLibraryA(path.c_str());
    if(!g_hip_lib) continue;
    g_get_props =
        reinterpret_cast<hipGetDeviceProperties_fn>(GetProcAddress(g_hip_lib, "hipGetDeviceProperties"));
    g_get_error_string =
        reinterpret_cast<hipGetErrorString_fn>(GetProcAddress(g_hip_lib, "hipGetErrorString"));
#else
    dlerror();
    g_hip_lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!g_hip_lib) {
      if(const char* err = dlerror()) last_open_error.assign(err);
      continue;
    }
    dlerror();
    g_get_props =
        reinterpret_cast<hipGetDeviceProperties_fn>(dlsym(g_hip_lib, "hipGetDeviceProperties"));
    g_get_error_string =
        reinterpret_cast<hipGetErrorString_fn>(dlsym(g_hip_lib, "hipGetErrorString"));
#endif
    if(g_get_props && g_get_error_string) {
      g_load_error.clear();
      return true;
    }
#if defined(_WIN32)
    if(g_hip_lib) {
      FreeLibrary(g_hip_lib);
      g_hip_lib = nullptr;
    }
#else
    if(g_hip_lib) {
      dlclose(g_hip_lib);
      g_hip_lib = nullptr;
    }
#endif
  }

  g_load_error.clear();
#if !defined(_WIN32)
  if(const char* sym_err = dlerror()) g_load_error.assign(sym_err);
#endif
  if(g_load_error.empty() && !last_open_error.empty()) g_load_error = last_open_error;
  if(g_load_error.empty())
    g_load_error = "failed to load HIP runtime (libamdhip64 / amdhip64.dll)";
  g_get_props        = nullptr;
  g_get_error_string = nullptr;
  return false;
}

void ensure_loaded_unlocked() {
  static bool attempted = false;
  if(attempted) return;
  attempted = true;
  try_open_and_resolve();
}

}  // namespace

bool hip_runtime_available() {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_loaded_unlocked();
  return g_get_props != nullptr && g_get_error_string != nullptr;
}

hipError_t hip_get_device_properties(hipDeviceProp_t* prop, int deviceId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_loaded_unlocked();
  if(!g_get_props) return hipErrorNotInitialized;
  return g_get_props(prop, deviceId);
}

const char* hip_get_error_string(hipError_t err) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_loaded_unlocked();
  if(!g_get_error_string) {
    if(!g_load_error.empty()) return g_load_error.c_str();
    return "HIP runtime is not loaded";
  }
  return g_get_error_string(err);
}

}  // namespace origami::detail
