// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/hip_runtime_loader.hpp"
#include "origami/logger.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
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
using hipGetProcAddress_fn =
    hipError_t (*)(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                   hipDriverProcAddressQueryResult* symbolStatus);

#if defined(_WIN32)
using dl_handle_t = HMODULE;
#else
using dl_handle_t = void*;
#endif

// Intentionally never dlclose'd / FreeLibrary'd after a successful load. Unloading the HIP
// runtime while GPU state may still be live (or during static destruction) can cause crashes
// at process exit. The OS reclaims the mapping when the process terminates.
dl_handle_t g_hip_lib                     = nullptr;
hipGetDeviceProperties_fn g_get_props     = nullptr;
hipDeviceGetAttribute_fn g_get_device_attribute = nullptr;
hipGetErrorString_fn g_get_error_string   = nullptr;
std::once_flag g_init_flag;

#if defined(_WIN32)
constexpr const char* k_hip_runtime_load_failed_message =
    "HIP runtime is not loaded (failed to load amdhip64.dll)";
#else
constexpr const char* k_hip_runtime_load_failed_message =
    "HIP runtime is not loaded (failed to load libamdhip64.so)";
#endif

bool hip_library_loaded() {
  return g_get_props != nullptr && g_get_error_string != nullptr;
}

/** Append @p path only if it is a bare library name or the file exists. Bare names rely on the OS loader search path. */
void push_hip_candidate(std::vector<std::string>& out, const std::string& path) {
  if(path.find_first_of("/\\") != std::string::npos) {
    std::error_code ec;
    if(!std::filesystem::exists(std::filesystem::path(path), ec)) return;
  }
  out.push_back(path);
}

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
      push_hip_candidate(out, dir + dll_name);
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
      push_hip_candidate(out, dir + lib6);
      push_hip_candidate(out, dir + lib);
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
    push_hip_candidate(out, base + kDll);
  }
  if(const char* rocm = std::getenv("ROCM_PATH")) {
    std::string base(rocm);
    append_path_sep(base);
    push_hip_candidate(out, base + "bin\\" + kDll);
  }
  // PATH before default install locations (parallel to LD_LIBRARY_PATH before /opt/rocm on Linux).
  append_path_env_candidates(out, std::getenv("PATH"), kDll);
  // Typical ROCm layout under Program Files
  if(const char* pf = std::getenv("ProgramFiles")) {
    std::string b(pf);
    append_path_sep(b);
    push_hip_candidate(out, b + "AMD\\ROCm\\bin\\" + kDll);
    push_hip_candidate(out, b + "ROCm\\bin\\" + kDll);
  }
  if(const char* pfx86 = std::getenv("ProgramFiles(x86)")) {
    std::string b(pfx86);
    append_path_sep(b);
    push_hip_candidate(out, b + "AMD\\ROCm\\bin\\" + kDll);
  }
  push_hip_candidate(out, kDll);
#else
  constexpr const char* kSo6 = "libamdhip64.so.6";
  constexpr const char* kSo  = "libamdhip64.so";

  if(const char* rocm = std::getenv("ROCM_PATH")) {
    std::string base(rocm);
    append_path_sep(base);
    push_hip_candidate(out, base + "lib/" + kSo6);
    push_hip_candidate(out, base + "lib/" + kSo);
  }
  if(const char* hip_path = std::getenv("HIP_PATH")) {
    std::string base(hip_path);
    append_path_sep(base);
    push_hip_candidate(out, base + "lib/" + kSo6);
    push_hip_candidate(out, base + "lib/" + kSo);
  }
  append_ld_library_path_candidates(out, kSo6, kSo);
  {
    std::string opt("/opt/rocm/");
    push_hip_candidate(out, opt + "lib/" + kSo6);
    push_hip_candidate(out, opt + "lib/" + kSo);
  }
  push_hip_candidate(out, kSo6);
  push_hip_candidate(out, kSo);
#endif
  return out;
}

void* bootstrap_get_proc_address(dl_handle_t mod) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(mod, "hipGetProcAddress"));
#else
  dlerror();
  return dlsym(mod, "hipGetProcAddress");
#endif
}

void* hip_get_proc_address_only(hipGetProcAddress_fn get_proc, const char* name) {
  if(!get_proc) return nullptr;
  void* pfn = nullptr;
  if(get_proc(name, &pfn, static_cast<int>(HIP_VERSION), 0u, nullptr) != hipSuccess || !pfn)
    return nullptr;
  return pfn;
}

bool try_open_and_resolve() {
  if(g_get_props && g_get_error_string) return true;

  for(const auto& path : hip_library_candidates()) {
#if defined(_WIN32)
    g_hip_lib = LoadLibraryA(path.c_str());
    if(!g_hip_lib) {
      OLOG_WARNING("Failed to load HIP library \"" << path << "\": GetLastError="
                   << static_cast<unsigned long>(GetLastError()));
      continue;
    }
#else
    dlerror();
    g_hip_lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!g_hip_lib) {
      const char* err = dlerror();
      if(err) {
        OLOG_WARNING("Failed to load HIP library \"" << path << "\": " << err);
      } else {
        OLOG_WARNING("Failed to load HIP library \"" << path << "\"");
      }
      continue;
    }
#endif
    hipGetProcAddress_fn hip_get_proc_address =
        reinterpret_cast<hipGetProcAddress_fn>(bootstrap_get_proc_address(g_hip_lib));
    if(!hip_get_proc_address) {
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
      OLOG_WARNING("HIP library \"" << path
                                   << "\" loaded but hipGetProcAddress was not found (not a usable HIP "
                                      "runtime)");
      continue;
    }
    g_get_props = reinterpret_cast<hipGetDeviceProperties_fn>(
        hip_get_proc_address_only(hip_get_proc_address, "hipGetDeviceProperties"));
    g_get_error_string = reinterpret_cast<hipGetErrorString_fn>(
        hip_get_proc_address_only(hip_get_proc_address, "hipGetErrorString"));
    if(g_get_props && g_get_error_string) {
      g_get_device_attribute = reinterpret_cast<hipDeviceGetAttribute_fn>(
          hip_get_proc_address_only(hip_get_proc_address, "hipDeviceGetAttribute"));
      return true;
    }
    OLOG_WARNING("HIP library \"" << path
                                  << "\" loaded but required HIP symbols could not be resolved via "
                                     "hipGetProcAddress");
    g_get_props            = nullptr;
    g_get_device_attribute = nullptr;
    g_get_error_string     = nullptr;
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

  g_get_props            = nullptr;
  g_get_device_attribute = nullptr;
  g_get_error_string     = nullptr;
  OLOG_WARNING(k_hip_runtime_load_failed_message);
  return false;
}

}  // namespace

bool hip_runtime_available() {
  std::call_once(g_init_flag, try_open_and_resolve);
  return hip_library_loaded();
}

hipError_t hip_get_device_properties(hipDeviceProp_t* prop, int deviceId) {
  std::call_once(g_init_flag, try_open_and_resolve);
  if(!hip_library_loaded()) return hipErrorNotInitialized;
  return g_get_props(prop, deviceId);
}

hipError_t hip_get_device_attribute(int* value, hipDeviceAttribute_t attr, int deviceId) {
  std::call_once(g_init_flag, try_open_and_resolve);
  if(!hip_library_loaded()) return hipErrorNotInitialized;
  if(!g_get_device_attribute) return hipErrorNotSupported;
  return g_get_device_attribute(value, attr, deviceId);
}

const char* hip_get_error_string(hipError_t err) {
  std::call_once(g_init_flag, try_open_and_resolve);
  if(!hip_library_loaded()) return k_hip_runtime_load_failed_message;
  return g_get_error_string(err);
}

}  // namespace origami
