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

namespace origami {

namespace {

using hipGetDeviceProperties_fn = hipError_t (*)(hipDeviceProp_t*, int);
using hipDeviceGetAttribute_fn  = hipError_t (*)(int*, hipDeviceAttribute_t, int);
using hipGetErrorString_fn      = const char* (*)(hipError_t);

#if defined(_WIN32)
using dl_handle_t = HMODULE;
#else
using dl_handle_t = void*;
#endif

dl_handle_t g_hip_lib                     = nullptr;
hipGetDeviceProperties_fn g_get_props     = nullptr;
hipDeviceGetAttribute_fn g_get_device_attribute = nullptr;
hipGetErrorString_fn g_get_error_string   = nullptr;
std::string g_load_error;
std::mutex g_mutex;

void append_path_sep(std::string& base) {
  if(base.empty()) return;
#if defined(_WIN32)
  if(base.back() != '\\' && base.back() != '/') base += '\\';
#else
  if(base.back() != '/') base += '/';
#endif
}

/** Split env (e.g. LD_LIBRARY_PATH or PATH) and append full paths to lib under each directory. */
#if defined(_WIN32)
void append_path_env_candidates(std::vector<std::string>& out, const char* env_val, const char* dll_name) {
  if(!env_val) return;
  std::string s(env_val);
  size_t start = 0;
  while(start <= s.size()) {
    // PATH uses ';' only; do not split on ':' (drive letters use C:...).
    size_t end = s.find(';', start);
    if(end == std::string::npos) end = s.size();
    std::string dir = s.substr(start, end - start);
    while(!dir.empty() && (dir.back() == ' ' || dir.back() == '\t')) dir.pop_back();
    if(!dir.empty()) {
      append_path_sep(dir);
      out.push_back(dir + dll_name);
    }
    if(end == s.size()) break;
    start = end + 1;
  }
}
#else
void append_ld_library_path_candidates(std::vector<std::string>& out,
                                       const char* lib6,
                                       const char* lib) {
  const char* env_val = std::getenv("LD_LIBRARY_PATH");
  if(!env_val) return;
  std::string s(env_val);
  size_t start = 0;
  while(start <= s.size()) {
    size_t end = s.find(':', start);
    if(end == std::string::npos) end = s.size();
    std::string dir = s.substr(start, end - start);
    while(!dir.empty() && dir.back() == '/') dir.pop_back();
    if(!dir.empty()) {
      dir += '/';
      out.push_back(dir + lib6);
      out.push_back(dir + lib);
    }
    if(end == s.size()) break;
    start = end + 1;
  }
}
#endif

std::vector<std::string> hip_library_candidates() {
  std::vector<std::string> out;
#if defined(_WIN32)
  constexpr const char* kDll = "amdhip64.dll";

  if(const char* hip_path = std::getenv("HIP_PATH")) {
    std::string base(hip_path);
    append_path_sep(base);
    out.push_back(base + kDll);
  }
  if(const char* rocm = std::getenv("ROCM_PATH")) {
    std::string base(rocm);
    append_path_sep(base);
    out.push_back(base + "bin\\" + kDll);
  }
  // PATH before default install locations (parallel to LD_LIBRARY_PATH before /opt/rocm on Linux).
  append_path_env_candidates(out, std::getenv("PATH"), kDll);
  // Typical ROCm layout under Program Files
  if(const char* pf = std::getenv("ProgramFiles")) {
    std::string b(pf);
    append_path_sep(b);
    out.push_back(b + "AMD\\ROCm\\bin\\" + kDll);
    out.push_back(b + "ROCm\\bin\\" + kDll);
  }
  if(const char* pfx86 = std::getenv("ProgramFiles(x86)")) {
    std::string b(pfx86);
    append_path_sep(b);
    out.push_back(b + "AMD\\ROCm\\bin\\" + kDll);
  }
  out.push_back(kDll);
#else
  constexpr const char* kSo6 = "libamdhip64.so.6";
  constexpr const char* kSo  = "libamdhip64.so";

  if(const char* rocm = std::getenv("ROCM_PATH")) {
    std::string base(rocm);
    append_path_sep(base);
    out.push_back(base + "lib/" + kSo6);
    out.push_back(base + "lib/" + kSo);
  }
  if(const char* hip_path = std::getenv("HIP_PATH")) {
    std::string base(hip_path);
    append_path_sep(base);
    out.push_back(base + "lib/" + kSo6);
    out.push_back(base + "lib/" + kSo);
  }
  append_ld_library_path_candidates(out, kSo6, kSo);
  {
    std::string opt("/opt/rocm/");
    out.push_back(opt + "lib/" + kSo6);
    out.push_back(opt + "lib/" + kSo);
  }
  out.push_back(kSo6);
  out.push_back(kSo);
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
#if defined(_WIN32)
      g_get_device_attribute = reinterpret_cast<hipDeviceGetAttribute_fn>(
          GetProcAddress(g_hip_lib, "hipDeviceGetAttribute"));
#else
      dlerror();
      g_get_device_attribute = reinterpret_cast<hipDeviceGetAttribute_fn>(
          dlsym(g_hip_lib, "hipDeviceGetAttribute"));
#endif
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
  g_get_props            = nullptr;
  g_get_device_attribute = nullptr;
  g_get_error_string     = nullptr;
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

hipError_t hip_get_device_attribute(int* value, hipDeviceAttribute_t attr, int deviceId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_loaded_unlocked();
  if(!g_get_props) return hipErrorNotInitialized;
  if(!g_get_device_attribute) return hipErrorNotSupported;
  return g_get_device_attribute(value, attr, deviceId);
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

}  // namespace origami
