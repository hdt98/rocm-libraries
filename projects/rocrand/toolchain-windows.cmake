#set(CMAKE_MAKE_PROGRAM "nmake.exe")
#set(CMAKE_GENERATOR "Ninja")
# Ninja doesn't support platform
#set(CMAKE_GENERATOR_PLATFORM x64)

if (DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
elseif (DEFINED ENV{HIP_DIR})
  file(TO_CMAKE_PATH "$ENV{HIP_DIR}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
else()
  set(HIP_DIR "C:/hip")
  set(rocm_bin "C:/hip/bin")
endif()

set(CMAKE_CXX_COMPILER "${rocm_bin}/clang++.exe")

# Combined toolchain flags (usage flags, clang direct-use flags, platform defines).
# Using _INIT so projects can still append without losing toolchain defaults.
# -Wno-ignored-attributes: suppress __declspec 'dllexport' warning from msvc-compat mode
set(CMAKE_CXX_FLAGS_INIT "-DWIN32 -D_CRT_SECURE_NO_WARNINGS -std=c++17 -fms-extensions -fms-compatibility -Wno-ignored-attributes -D__HIP_PLATFORM_AMD__ -D__HIP_ROCclr__")

if (DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")
