/* **************************************************************************
 * CUDA Compatibility Layer for rocSOLVER Kernel Sandbox
 *
 * Maps HIP runtime APIs and macros to CUDA equivalents.
 * This allows rocSOLVER kernels to be compiled with nvcc for testing
 * with CUDA Compute Sanitizer.
 * *************************************************************************/

#pragma once

#include <cuda_runtime.h>

// Map HIP thread/block indexing macros to CUDA
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z
#define hipBlockIdx_x  blockIdx.x
#define hipBlockIdx_y  blockIdx.y
#define hipBlockIdx_z  blockIdx.z
#define hipBlockDim_x  blockDim.x
#define hipBlockDim_y  blockDim.y
#define hipBlockDim_z  blockDim.z
#define hipGridDim_x   gridDim.x
#define hipGridDim_y   gridDim.y
#define hipGridDim_z   gridDim.z

// Map HIP memory management to CUDA
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyHostToDevice   cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define hipMemset               cudaMemset
#define hipDeviceSynchronize    cudaDeviceSynchronize
#define hipGetLastError         cudaGetLastError
#define hipSuccess              cudaSuccess
#define hipError_t              cudaError_t
#define hipStream_t             cudaStream_t
#define hipGetErrorString       cudaGetErrorString
