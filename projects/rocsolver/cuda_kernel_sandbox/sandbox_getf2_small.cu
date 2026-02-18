/* **************************************************************************
 * rocSOLVER Kernel Sandbox - getf2_small_kernel Driver
 *
 * This program launches the getf2_small_kernel for testing with CUDA
 * Compute Sanitizer tools (memcheck, racecheck, initcheck, synccheck).
 *
 * Usage:
 *   ./sandbox_getf2_small [m] [n] [batch_count]
 *
 * Where:
 *   m           - number of rows (default: 16, max: 512)
 *   n           - number of columns / DIM (default: 8, max: 64)
 *   batch_count - number of matrices in batch (default: 1)
 *
 * Example:
 *   compute-sanitizer --tool racecheck ./sandbox_getf2_small 32 16 4
 * *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "cuda_compat.cuh"
#include "kernels/getf2_small_kernel.cuh"

using namespace rocsolver;

// Macro to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Template dispatch macro for kernel launch based on n (DIM)
#define LAUNCH_GETF2_SMALL(DIM_VAL) \
    case DIM_VAL: \
        getf2_small_kernel<DIM_VAL, float, int, int, float*><<<grid, block, shmem_size>>>( \
            m, d_A, 0, lda, strideA, d_ipiv, 0, strideP, d_info, batch_count, 0, nullptr, 0); \
        break

void print_usage(const char* prog_name)
{
    printf("Usage: %s [m] [n] [batch_count]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  m           - number of rows (default: 16, max: 512)\n");
    printf("  n           - number of columns / DIM (default: 8, max: 64)\n");
    printf("  batch_count - number of matrices in batch (default: 1)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  # Use defaults: m=16, n=8, batch=1\n", prog_name);
    printf("  %s 32 16 4          # m=32, n=16, batch=4\n", prog_name);
    printf("\n");
    printf("Run with CUDA Compute Sanitizer:\n");
    printf("  compute-sanitizer --tool racecheck %s 32 16 4\n", prog_name);
}

int main(int argc, char** argv)
{
    // Check for help flag
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    // Default parameters
    int m = 16;           // Number of rows
    int n = 8;            // Number of columns (DIM template parameter)
    int batch_count = 1;  // Number of matrices in batch

    // Parse command line args
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) batch_count = atoi(argv[3]);

    printf("=== rocSOLVER Kernel Sandbox: getf2_small_kernel ===\n");
    printf("Parameters: m=%d, n=%d, batch_count=%d\n", m, n, batch_count);

    // Validate parameters
    if (n < 1 || n > 64) {
        fprintf(stderr, "Error: n must be in range [1, 64], got %d\n", n);
        return 1;
    }
    if (m < n) {
        fprintf(stderr, "Error: m must be >= n for this kernel (m=%d, n=%d)\n", m, n);
        return 1;
    }
    if (m > 512) {
        fprintf(stderr, "Error: m must be <= 512, got %d\n", m);
        return 1;
    }
    if (batch_count < 1) {
        fprintf(stderr, "Error: batch_count must be >= 1, got %d\n", batch_count);
        return 1;
    }

    // Allocate host memory
    int lda = m;
    size_t matrix_size = (size_t)lda * n;
    size_t total_size = matrix_size * batch_count;

    printf("Matrix size: %dx%d, total elements: %zu\n", m, n, total_size);

    float* h_A = (float*)malloc(total_size * sizeof(float));
    int* h_ipiv = (int*)malloc(n * batch_count * sizeof(int));
    int* h_info = (int*)calloc(batch_count, sizeof(int));

    if (!h_A || !h_ipiv || !h_info) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize matrix with random values
    printf("Initializing random matrix data...\n");
    srand(42);
    for (size_t i = 0; i < total_size; i++) {
        h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Initialize ipiv to zeros
    memset(h_ipiv, 0, n * batch_count * sizeof(int));

    // Allocate device memory
    float* d_A;
    int* d_ipiv;
    int* d_info;

    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_count * sizeof(int)));

    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ipiv, 0, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_count * sizeof(int)));

    // Calculate grid/block dimensions
    // Based on getf2_run_small launcher logic from rocSOLVER
    int opval[] = {GETF2_OPTIM_NGRP};
    int ngrp = (batch_count < 2 || m > 32) ? 1 : opval[m - 1];
    int blocks = (batch_count - 1) / ngrp + 1;

    dim3 grid(1, blocks, 1);
    dim3 block(m, ngrp, 1);

    rocblas_stride strideA = matrix_size;
    rocblas_stride strideP = n;

    // Shared memory size: max(m, DIM) * ngrp * sizeof(float)
    size_t shmem_size = std::max(m, n) * ngrp * sizeof(float);

    printf("Kernel launch configuration:\n");
    printf("  Grid:  (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("  Block: (%d, %d, %d)\n", block.x, block.y, block.z);
    printf("  Shared memory: %zu bytes\n", shmem_size);
    printf("  ngrp: %d, blocks: %d\n", ngrp, blocks);

    printf("\nLaunching kernel...\n");

    // Launch kernel - dispatch based on n (DIM template parameter)
    switch(n) {
        LAUNCH_GETF2_SMALL(1);
        LAUNCH_GETF2_SMALL(2);
        LAUNCH_GETF2_SMALL(3);
        LAUNCH_GETF2_SMALL(4);
        LAUNCH_GETF2_SMALL(5);
        LAUNCH_GETF2_SMALL(6);
        LAUNCH_GETF2_SMALL(7);
        LAUNCH_GETF2_SMALL(8);
        LAUNCH_GETF2_SMALL(9);
        LAUNCH_GETF2_SMALL(10);
        LAUNCH_GETF2_SMALL(11);
        LAUNCH_GETF2_SMALL(12);
        LAUNCH_GETF2_SMALL(13);
        LAUNCH_GETF2_SMALL(14);
        LAUNCH_GETF2_SMALL(15);
        LAUNCH_GETF2_SMALL(16);
        LAUNCH_GETF2_SMALL(17);
        LAUNCH_GETF2_SMALL(18);
        LAUNCH_GETF2_SMALL(19);
        LAUNCH_GETF2_SMALL(20);
        LAUNCH_GETF2_SMALL(21);
        LAUNCH_GETF2_SMALL(22);
        LAUNCH_GETF2_SMALL(23);
        LAUNCH_GETF2_SMALL(24);
        LAUNCH_GETF2_SMALL(25);
        LAUNCH_GETF2_SMALL(26);
        LAUNCH_GETF2_SMALL(27);
        LAUNCH_GETF2_SMALL(28);
        LAUNCH_GETF2_SMALL(29);
        LAUNCH_GETF2_SMALL(30);
        LAUNCH_GETF2_SMALL(31);
        LAUNCH_GETF2_SMALL(32);
        LAUNCH_GETF2_SMALL(33);
        LAUNCH_GETF2_SMALL(34);
        LAUNCH_GETF2_SMALL(35);
        LAUNCH_GETF2_SMALL(36);
        LAUNCH_GETF2_SMALL(37);
        LAUNCH_GETF2_SMALL(38);
        LAUNCH_GETF2_SMALL(39);
        LAUNCH_GETF2_SMALL(40);
        LAUNCH_GETF2_SMALL(41);
        LAUNCH_GETF2_SMALL(42);
        LAUNCH_GETF2_SMALL(43);
        LAUNCH_GETF2_SMALL(44);
        LAUNCH_GETF2_SMALL(45);
        LAUNCH_GETF2_SMALL(46);
        LAUNCH_GETF2_SMALL(47);
        LAUNCH_GETF2_SMALL(48);
        LAUNCH_GETF2_SMALL(49);
        LAUNCH_GETF2_SMALL(50);
        LAUNCH_GETF2_SMALL(51);
        LAUNCH_GETF2_SMALL(52);
        LAUNCH_GETF2_SMALL(53);
        LAUNCH_GETF2_SMALL(54);
        LAUNCH_GETF2_SMALL(55);
        LAUNCH_GETF2_SMALL(56);
        LAUNCH_GETF2_SMALL(57);
        LAUNCH_GETF2_SMALL(58);
        LAUNCH_GETF2_SMALL(59);
        LAUNCH_GETF2_SMALL(60);
        LAUNCH_GETF2_SMALL(61);
        LAUNCH_GETF2_SMALL(62);
        LAUNCH_GETF2_SMALL(63);
        LAUNCH_GETF2_SMALL(64);
        default:
            fprintf(stderr, "Unsupported n=%d. Must be in range [1, 64].\n", n);
            return 1;
    }

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    printf("Waiting for kernel completion...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    printf("Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_A, d_A, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ipiv, d_ipiv, n * batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_info, d_info, batch_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results summary
    printf("\n=== Results ===\n");
    printf("info[0] = %d", h_info[0]);
    if (h_info[0] == 0) {
        printf(" (success)\n");
    } else {
        printf(" (singular at column %d)\n", h_info[0]);
    }

    printf("First %d pivot indices: ", std::min(n, 8));
    for (int i = 0; i < std::min(n, 8); i++) {
        printf("%d ", h_ipiv[i]);
    }
    printf("\n");

    // Print first few elements of factored matrix
    printf("First 4x4 block of factored matrix (batch 0):\n");
    for (int i = 0; i < std::min(m, 4); i++) {
        printf("  ");
        for (int j = 0; j < std::min(n, 4); j++) {
            printf("%8.4f ", h_A[i + j * lda]);
        }
        printf("\n");
    }

    // Cleanup
    printf("\nCleaning up...\n");
    free(h_A);
    free(h_ipiv);
    free(h_info);
    cudaFree(d_A);
    cudaFree(d_ipiv);
    cudaFree(d_info);

    printf("Done.\n");
    return 0;
}
