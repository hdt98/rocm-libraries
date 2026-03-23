# Reproduce directly from dockerfile
Run docker
```
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v /home/AMD/barkocot:/root/workspace  \
rocm/miopen:ci_d262a5_navi \
/bin/bash
```
Cmake and make in miopen repository
```
cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
-D CMAKE_CXX_FLAGS=""   \
-D MIOPEN_USE_ROCBLAS=Off \
-D MIOPEN_USE_HIPBLASLT=Off \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-DCMAKE_INSTALL_PREFIX=/opt/rocm \
..
make -j128 MIOpenDriver
```
What I see
```
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 ./bin/MIOpenDriver conv -n 1 -c 896 -H 1 -W 1 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1  -in_layout NHWC -iter 50
MIOpen Error: HIP runtime error: invalid device function. hip_check_error.hpp: 18in function: hip_check_error
Error getting workspace size, status = 7
AllocateBuffersAndCopy() FAILED, rc = 7
```

# Reproduce from dockerfile but with CK from source
Run docker
```
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v /home/AMD/barkocot:/root/workspace  \
rocm/miopen:ci_d262a5_navi \
/bin/bash
```
Install CK
```
cd ../projects/composablekernel/build
make -j128 install
```
Clean, Cmake and make in miopen repository
```
rm * -r
cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
-D CMAKE_CXX_FLAGS=""   \
-D MIOPEN_USE_ROCBLAS=Off \
-D MIOPEN_USE_HIPBLASLT=Off \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-DCMAKE_INSTALL_PREFIX=/opt/rocm \
..
make -j128 MIOpenDriver
```
What I see
```
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 ./bin/MIOpenDriver conv -n 1 -c 896 -H 1 -W 1 -k 224 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1  -in_layout NHWC -iter 50
PRNG seed: 12345678
Timestamp: 2026-03-23 11:04:12 UTC; Host Name: def956a8ccf5; Operating System: Linux 6.8.0-31-generic; ROCm: 7.12.60700; MIOpen Driver: 3.5.1; CPU Vendor: AMD; CPU Model: 1 x Ryzen Threadripper; RAM Size: 125 GB; GPU Model: 1 x AMD Radeon RX 9070 XT; AMDGPU Driver: 6.16.13
MIOpen(HIP): Warning [FindSolutionImpl] Perf Db: load skipped: ConvBinWinogradRxSf3x2, enforce: SEARCH_DB_UPDATE(4)
MIOpen(HIP): Warning [FindSolutionImpl] Perf Db: load skipped: ConvBinWinogradRxSf2x3, enforce: SEARCH_DB_UPDATE(4)
MIOpen Backward Weights Conv. Algorithm: 1, Solution: 87/ConvDirectNaiveConvWrw
GPU Kernel Time Backward Weights Conv. Elapsed: 0.012380 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: bwdw-conv1x1u1, 1, 896, 1, 1, 1, 1, 224, 401408, 0, 0, 32, 0, 0.012380
Backward Convolution Weights Verifies OK on GPU reference (0 < 3e-06)
```
