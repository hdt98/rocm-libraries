/* **************************************************************************
 * rocSOLVER Kernel Sandbox - STEBZ Driver
 *
 * This program launches the STEBZ kernels (eigenvalue computation using
 * bisection method for symmetric tridiagonal matrices) for testing with
 * CUDA Compute Sanitizer tools (memcheck, racecheck, initcheck, synccheck).
 *
 * Usage:
 *   ./sandbox_stebz [n] [batch_count] [range]
 *
 * Where:
 *   n           - matrix size (default: 64, max: 1024)
 *   batch_count - number of matrices in batch (default: 1)
 *   range       - eigenvalue range: all, value, index (default: all)
 *
 * Example:
 *   compute-sanitizer --tool racecheck ./sandbox_stebz 128 4 all
 * *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "cuda_compat.cuh"
#include "kernels/stebz_kernels.cuh"

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

void print_usage(const char* prog_name)
{
    printf("Usage: %s [n] [batch_count] [range]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  n           - matrix size (default: 64, max: 1024)\n");
    printf("  batch_count - number of matrices in batch (default: 1)\n");
    printf("  range       - eigenvalue range: all, value, index (default: all)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  # Use defaults: n=64, batch=1, range=all\n", prog_name);
    printf("  %s 128 4 all        # n=128, batch=4, find all eigenvalues\n", prog_name);
    printf("  %s 64 1 value       # n=64, batch=1, eigenvalues in (vl,vu]\n", prog_name);
    printf("  %s 64 1 index       # n=64, batch=1, eigenvalues il through iu\n", prog_name);
    printf("\n");
    printf("Run with CUDA Compute Sanitizer:\n");
    printf("  compute-sanitizer --tool racecheck %s 128 4\n", prog_name);
}

int main(int argc, char** argv)
{
    // Check for help flag
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    // Default parameters
    int n = 64;           // Matrix size
    int batch_count = 1;  // Number of matrices in batch
    rocblas_erange range = rocblas_erange_all;

    // Parse command line args
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) batch_count = atoi(argv[2]);
    if (argc > 3) {
        if (strcmp(argv[3], "value") == 0) {
            range = rocblas_erange_value;
        } else if (strcmp(argv[3], "index") == 0) {
            range = rocblas_erange_index;
        } else {
            range = rocblas_erange_all;
        }
    }

    printf("=== rocSOLVER Kernel Sandbox: STEBZ (eigenvalue bisection) ===\n");
    printf("Parameters: n=%d, batch_count=%d, range=%s\n", n, batch_count,
           range == rocblas_erange_all ? "all" :
           range == rocblas_erange_value ? "value" : "index");

    // Validate parameters
    if (n < 1 || n > 1024) {
        fprintf(stderr, "Error: n must be in range [1, 1024], got %d\n", n);
        return 1;
    }
    if (batch_count < 1) {
        fprintf(stderr, "Error: batch_count must be >= 1, got %d\n", batch_count);
        return 1;
    }

    // For range=value, we need vlow and vup
    float vlow = -10.0f;  // Lower bound for eigenvalue search
    float vup = 10.0f;    // Upper bound for eigenvalue search

    // For range=index, we need ilow and iup
    int ilow = 1;         // First eigenvalue index
    int iup = std::min(n, 10);  // Last eigenvalue index (first 10 or all if n < 10)

    if (range == rocblas_erange_value) {
        printf("Value range: eigenvalues in (%f, %f]\n", vlow, vup);
    } else if (range == rocblas_erange_index) {
        printf("Index range: eigenvalues %d through %d\n", ilow, iup);
    }

    // Allocate host memory
    printf("Allocating host memory...\n");

    // Diagonal (D) and off-diagonal (E) of tridiagonal matrix
    float* h_D = (float*)malloc(n * batch_count * sizeof(float));
    float* h_E = (float*)malloc((n - 1) * batch_count * sizeof(float));

    // Output arrays
    float* h_W = (float*)malloc(n * batch_count * sizeof(float));  // Eigenvalues
    int* h_IB = (int*)malloc(n * batch_count * sizeof(int));      // Block index for each eigenvalue
    int* h_IS = (int*)malloc(n * batch_count * sizeof(int));      // Split point indices

    // Scalar outputs per batch
    int* h_nev = (int*)calloc(batch_count, sizeof(int));          // Number of eigenvalues found
    int* h_nsplit = (int*)calloc(batch_count, sizeof(int));       // Number of split blocks
    int* h_info = (int*)calloc(batch_count, sizeof(int));         // Info status

    if (!h_D || !h_E || !h_W || !h_IB || !h_IS || !h_nev || !h_nsplit || !h_info) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize tridiagonal matrix with Wilkinson-style matrix
    // (known to have eigenvalues that cluster and are good for testing)
    printf("Initializing tridiagonal matrix data...\n");
    srand(42);
    for (int b = 0; b < batch_count; b++) {
        for (int i = 0; i < n; i++) {
            // Diagonal: gradually increasing values with some noise
            h_D[b * n + i] = (float)i - (float)(n-1)/2 + 0.1f * ((float)rand() / RAND_MAX);
        }
        for (int i = 0; i < n - 1; i++) {
            // Off-diagonal: constant with small perturbation
            h_E[b * (n-1) + i] = 1.0f + 0.01f * ((float)rand() / RAND_MAX - 0.5f);
        }
    }

    // Allocate device memory
    printf("Allocating device memory...\n");

    // Input arrays
    float* d_D;
    float* d_E;
    CUDA_CHECK(cudaMalloc(&d_D, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_E, (n - 1) * batch_count * sizeof(float)));

    // Output arrays
    float* d_W;
    int* d_IB;
    int* d_IS;
    int* d_nev;
    int* d_nsplit;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_W, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_IB, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_IS, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nev, batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nsplit, batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_count * sizeof(int)));

    // Workspace arrays
    int* d_work;        // Temporary indices/sizes
    float* d_pivmin;    // Minimum pivot values
    float* d_Esqr;      // Squared off-diagonal
    float* d_bounds;    // Interval bounds
    float* d_inter;     // Interval bounds during bisection
    int* d_ninter;      // Eigenvalue counts for intervals

    CUDA_CHECK(cudaMalloc(&d_work, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivmin, batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Esqr, (n - 1) * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bounds, 2 * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inter, 4 * n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ninter, 4 * n * batch_count * sizeof(int)));

    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_D, h_D, n * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, (n - 1) * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_W, 0, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_IB, 0, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_IS, 0, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nev, 0, batch_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nsplit, 0, batch_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_count * sizeof(int)));

    // Numeric constants
    float eps = get_epsilon<float>();
    float sfmin = get_safemin<float>();
    float abstol = 2.0f * sfmin;  // Absolute tolerance

    // Strides
    rocblas_stride strideD = n;
    rocblas_stride strideE = n - 1;
    rocblas_stride strideW = n;
    rocblas_stride strideIB = n;
    rocblas_stride strideIS = n;

    printf("\nLaunching STEBZ kernels...\n");

    // Handle n = 1 case separately
    if (n == 1) {
        printf("  Case n=1: single diagonal element\n");
        dim3 grid1((batch_count - 1) / BS1 + 1, 1, 1);
        dim3 block1(BS1, 1, 1);

        stebz_case1_kernel<float, float*><<<grid1, block1>>>(
            range, vlow, vup, d_D, 0, strideD, d_nev, d_nsplit,
            d_W, strideW, d_IB, strideIB, d_IS, strideIS, batch_count);
        CUDA_CHECK(cudaGetLastError());
    }
    else {
        // Step 1: Splitting kernel
        printf("  Step 1: Matrix splitting...\n");
        dim3 gridSplit(1, batch_count);
        dim3 blockSplit(STEBZ_SPLIT_THDS);

        stebz_splitting_kernel<float, float*><<<gridSplit, blockSplit>>>(
            range, n, vlow, vup, ilow, iup,
            d_D, 0, strideD,
            d_E, 0, strideE,
            d_nsplit, d_W, strideW, d_IS, strideIS, d_work,
            d_pivmin, d_Esqr, d_bounds, d_inter, d_ninter,
            eps, sfmin);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Bisection kernel
        printf("  Step 2: Iterative bisection...\n");
        dim3 gridBisec(IBISEC_BLKS, batch_count);
        dim3 blockBisec(IBISEC_THDS);

        stebz_bisection_kernel<float, float*><<<gridBisec, blockBisec>>>(
            range, n, abstol,
            d_D, 0, strideD,
            d_E, 0, strideE,
            d_nsplit, d_W, strideW, d_IB, strideIB, d_IS, strideIS,
            d_info, d_work, d_pivmin, d_Esqr, d_bounds, d_inter, d_ninter,
            eps, sfmin);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Synthesis kernel
        printf("  Step 3: Result synthesis...\n");
        dim3 gridSynth((batch_count - 1) / BS1 + 1, 1, 1);
        dim3 blockSynth(BS1, 1, 1);

        stebz_synthesis_kernel<float, float*><<<gridSynth, blockSynth>>>(
            range, rocblas_eorder_entire, n, ilow, iup,
            d_D, 0, strideD,
            d_nev, d_nsplit, d_W, strideW, d_IB, strideIB, d_IS, strideIS,
            batch_count, d_work, d_pivmin, d_Esqr, d_bounds, d_inter, d_ninter,
            eps);
        CUDA_CHECK(cudaGetLastError());
    }

    printf("Waiting for kernel completion...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    printf("Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_W, d_W, n * batch_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_IB, d_IB, n * batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_IS, d_IS, n * batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nev, d_nev, batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_nsplit, d_nsplit, batch_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_info, d_info, batch_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results summary
    printf("\n=== Results ===\n");
    for (int b = 0; b < std::min(batch_count, 2); b++) {
        printf("\nBatch %d:\n", b);
        printf("  info = %d%s\n", h_info[b], h_info[b] == 0 ? " (success)" : " (some eigenvalues did not converge)");
        printf("  nsplit = %d (number of split blocks)\n", h_nsplit[b]);
        printf("  nev = %d (number of eigenvalues found)\n", h_nev[b]);

        if (h_nev[b] > 0) {
            int num_to_print = std::min(h_nev[b], 10);
            printf("  First %d eigenvalues: ", num_to_print);
            for (int i = 0; i < num_to_print; i++) {
                printf("%.4f ", h_W[b * n + i]);
            }
            printf("\n");

            printf("  Block indices (IB): ");
            for (int i = 0; i < num_to_print; i++) {
                printf("%d ", h_IB[b * n + i]);
            }
            printf("\n");
        }
    }

    if (batch_count > 2) {
        printf("\n... (%d more batches not shown)\n", batch_count - 2);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    free(h_D);
    free(h_E);
    free(h_W);
    free(h_IB);
    free(h_IS);
    free(h_nev);
    free(h_nsplit);
    free(h_info);

    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_W);
    cudaFree(d_IB);
    cudaFree(d_IS);
    cudaFree(d_nev);
    cudaFree(d_nsplit);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_pivmin);
    cudaFree(d_Esqr);
    cudaFree(d_bounds);
    cudaFree(d_inter);
    cudaFree(d_ninter);

    printf("Done.\n");
    return 0;
}
