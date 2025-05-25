# TODO: the user must specify this manually if using c++23 and the linker cannot find it manually
# TODO: why does this have to be duplicated twice?
add_link_options(-L /usr/local/gcc-15.1.0/lib64/)
if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation.")
  set(rocm_bin "/opt/rocm/bin")
endif()

if (NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER "${rocm_bin}/amdclang++")
else()
  set(CMAKE_CXX_COMPILER "$ENV{CXX}")
endif()

if (NOT DEFINED ENV{CC})
  set(CMAKE_C_COMPILER "${rocm_bin}/amdclang")
else()
  set(CMAKE_C_COMPILER "$ENV{CC}")
endif()
