# Part 1: LDS probe kernel
add_executable(tile_bank_profiler_lds_probe EXCLUDE_FROM_ALL lds_probe.cpp)

# Part 2: GEMM profiler variants
add_executable(tile_bank_profiler_gemm_128x128x32 EXCLUDE_FROM_ALL gemm_bank_profile_128x128x32.cpp)
add_executable(tile_bank_profiler_gemm_256x128x32 EXCLUDE_FROM_ALL gemm_bank_profile_256x128x32.cpp)
add_executable(tile_bank_profiler_gemm_128x256x32 EXCLUDE_FROM_ALL gemm_bank_profile_128x256x32.cpp)
add_executable(tile_bank_profiler_gemm_256x256x32 EXCLUDE_FROM_ALL gemm_bank_profile_256x256x32.cpp)
add_executable(tile_bank_profiler_gemm_64x64x32   EXCLUDE_FROM_ALL gemm_bank_profile_64x64x32.cpp)
add_executable(tile_bank_profiler_gemm_128x128x64 EXCLUDE_FROM_ALL gemm_bank_profile_128x128x64.cpp)

# Convenience meta-target to build all profiler binaries
add_custom_target(tile_bank_profiler
    DEPENDS
        tile_bank_profiler_lds_probe
        tile_bank_profiler_gemm_128x128x32
        tile_bank_profiler_gemm_256x128x32
        tile_bank_profiler_gemm_128x256x32
        tile_bank_profiler_gemm_256x256x32
        tile_bank_profiler_gemm_64x64x32
        tile_bank_profiler_gemm_128x128x64
)
