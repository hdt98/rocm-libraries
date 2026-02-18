/* **************************************************************************
 * rocSOLVER Kernel Sandbox - STEDC Secular Equation Solver Driver (laed4_alt)
 *
 * This program tests the STEDC secular equation solver kernel laed4_alt
 * for use with CUDA Compute Sanitizer tools (memcheck, racecheck, initcheck,
 * synccheck).
 *
 * IMPORTANT: laed4_alt is a parallelized version of the slaed4 solver that
 * uses STEDC_SOLVE_BDIM threads per eigenvalue. When STEDC_SOLVE_BDIM is a
 * multiple of 32 (warp size), this enables block-level reductions which may
 * have race conditions. This is intentional to test sanitizer detection.
 *
 * The secular equation solvers are used in the merge phase of the
 * divide-and-conquer algorithm for computing eigenvalues/eigenvectors
 * of symmetric tridiagonal matrices.
 *
 * Usage:
 *   ./sandbox_stedc_solve [n] [batch_count] [use_reference]
 *
 * Where:
 *   n             - problem size (number of eigenvalues, default: 16)
 *   batch_count   - number of problems in batch (default: 1)
 *   use_reference - 1 for single-thread behavior, 0 for parallel (default: 0)
 *
 * Example:
 *   compute-sanitizer --tool racecheck ./sandbox_stedc_solve 32 4 0
 * *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "cuda_compat.cuh"
#include "kernels/stedc_solve_kernels.cuh"

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
    printf("Usage: %s [n] [batch_count] [use_reference]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  n             - problem size (number of eigenvalues, default: 16, max: 256)\n");
    printf("  batch_count   - number of problems in batch (default: 1)\n");
    printf("  use_reference - 1 for single-thread behavior, 0 for parallel laed4_alt (default: 0)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  # Use defaults: n=16, batch=1, seq_solve\n", prog_name);
    printf("  %s 32 4 0           # n=32, batch=4, use seq_solve\n", prog_name);
    printf("  %s 16 1 1           # n=16, batch=1, use slaed4 (reference)\n", prog_name);
    printf("\n");
    printf("Run with CUDA Compute Sanitizer:\n");
    printf("  compute-sanitizer --tool racecheck %s 32 4 1\n", prog_name);
}

int main(int argc, char** argv)
{
    // Check for help flag
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    // Default parameters
    int n = 16;           // Problem size
    int batch_count = 1;  // Number of problems in batch
    bool use_reference = false;

    // Parse command line args
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) batch_count = atoi(argv[2]);
    if (argc > 3) use_reference = (atoi(argv[3]) != 0);

    printf("=== rocSOLVER Kernel Sandbox: STEDC Secular Equation Solver ===\n");
    printf("Parameters: n=%d, batch_count=%d, solver=%s\n", n, batch_count,
           use_reference ? "slaed4 (reference)" : "seq_solve (optimized)");

    // Validate parameters
    if (n < 2 || n > 256) {
        fprintf(stderr, "Error: n must be in range [2, 256], got %d\n", n);
        return 1;
    }
    if (batch_count < 1) {
        fprintf(stderr, "Error: batch_count must be >= 1, got %d\n", batch_count);
        return 1;
    }

    // Allocate host memory
    printf("Allocating host memory...\n");

    // For the secular equation D + rho * z * z^T:
    // D - sorted diagonal elements (poles)
    // z - rank-1 modification vector
    // rho - scaling factor (p value)
    // evs - computed eigenvalues (output)

    float* h_D = (float*)malloc(n * batch_count * sizeof(float));
    float* h_z = (float*)malloc(n * batch_count * sizeof(float));
    float* h_r1p = (float*)malloc(n * batch_count * sizeof(float));  // p values
    float* h_evs = (float*)malloc(n * batch_count * sizeof(float));  // eigenvalue output
    int* h_nps = (int*)malloc(n * batch_count * sizeof(int));        // start positions
    int* h_ndd = (int*)malloc(n * batch_count * sizeof(int));        // non-deflated counts
    float* h_etmpd = (float*)malloc(n * n * batch_count * sizeof(float));  // delta workspace

    if (!h_D || !h_z || !h_r1p || !h_evs || !h_nps || !h_ndd || !h_etmpd) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize test data
    printf("Initializing test data...\n");
    srand(42);

    // Create a simple test case: secular equation for a rank-1 perturbation
    // of a diagonal matrix
    float rho = 1.0f;  // Rank-1 modification factor

    for (int b = 0; b < batch_count; b++) {
        // Generate sorted diagonal elements (poles)
        // These must be in increasing order
        for (int i = 0; i < n; i++) {
            // Create well-separated poles
            h_D[b * n + i] = (float)i * 2.0f + 0.1f * ((float)rand() / RAND_MAX);
        }

        // Sort to ensure strictly increasing order
        std::sort(h_D + b * n, h_D + (b + 1) * n);

        // Add small perturbation to ensure no duplicates
        for (int i = 1; i < n; i++) {
            if (h_D[b * n + i] <= h_D[b * n + i - 1]) {
                h_D[b * n + i] = h_D[b * n + i - 1] + 0.01f;
            }
        }

        // Generate z vector (rank-1 modification)
        // Must be non-zero for each component
        float z_norm = 0.0f;
        for (int i = 0; i < n; i++) {
            h_z[b * n + i] = 0.5f + 0.5f * ((float)rand() / RAND_MAX);
            z_norm += h_z[b * n + i] * h_z[b * n + i];
        }
        // Normalize z
        z_norm = sqrtf(z_norm);
        for (int i = 0; i < n; i++) {
            h_z[b * n + i] /= z_norm;
        }

        // Set p values (all same for this test)
        for (int i = 0; i < n; i++) {
            h_r1p[b * n + i] = rho;
        }

        // Set positions and degrees (all eigenvalues are non-deflated for this test)
        for (int i = 0; i < n; i++) {
            h_nps[b * n + i] = 0;  // All start at position 0
            h_ndd[b * n + i] = n;  // All n eigenvalues are non-deflated
        }

        // Initialize delta/etmpd workspace with copies of D
        // Each eigenvalue needs its own copy of D (delta)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h_etmpd[b * n * n + i * n + j] = h_D[b * n + j];
            }
        }

        // Initialize eigenvalue output
        for (int i = 0; i < n; i++) {
            h_evs[b * n + i] = h_D[b * n + i];  // Initial guess
        }
    }

    // Allocate device memory
    printf("Allocating device memory...\n");

    float* d_D;
    float* d_z;
    float* d_r1p;
    float* d_evs;
    int* d_nps;
    int* d_ndd;
    float* d_etmpd;

    CUDA_CHECK(cudaMalloc(&d_D, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1p, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_evs, n * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nps, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ndd, n * batch_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_etmpd, n * n * batch_count * sizeof(float)));

    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_D, h_D, n * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, n * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r1p, h_r1p, n * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_evs, h_evs, n * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nps, h_nps, n * batch_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ndd, h_ndd, n * batch_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_etmpd, h_etmpd, n * n * batch_count * sizeof(float), cudaMemcpyHostToDevice));

    // Numeric constants
    float eps = get_epsilon<float>();
    float ssfmin = get_safemin<float>();
    float ssfmax = 1.0f / ssfmin;
    ssfmin = sqrtf(ssfmin) / (eps * eps);
    ssfmax = sqrtf(ssfmax) / 3.0f;

    rocblas_stride strideD = n;

    printf("\nLaunching STEDC solve kernel (laed4_alt)...\n");

    // Launch kernel - one block per eigenvalue, STEDC_SOLVE_BDIM threads per block
    // laed4_alt uses all threads in the block to parallelize the secular equation solve
    int threads_per_block = STEDC_SOLVE_BDIM;
    int blocks_x = n;  // One block per eigenvalue

    dim3 grid(blocks_x, batch_count);
    dim3 block(threads_per_block);

    printf("  Grid: (%d, %d), Block: %d threads (STEDC_SOLVE_BDIM=%d)\n",
           blocks_x, batch_count, threads_per_block, STEDC_SOLVE_BDIM);
    printf("  eps=%.6e, ssfmin=%.6e, ssfmax=%.6e\n", eps, ssfmin, ssfmax);
    printf("  use_reference=%s (when true, only thread 0 participates)\n",
           use_reference ? "true" : "false");

    // Note: Template parameter BDIM must match the block size
    stedc_mergeValues_Solve_kernel<STEDC_SOLVE_BDIM, float><<<grid, block>>>(
        n,
        d_D, strideD,
        d_evs,
        d_etmpd,
        d_z,
        d_r1p,
        d_nps,
        d_ndd,
        eps, ssfmin, ssfmax,
        use_reference);

    CUDA_CHECK(cudaGetLastError());

    printf("Waiting for kernel completion...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    printf("Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_evs, d_evs, n * batch_count * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results summary
    printf("\n=== Results ===\n");
    for (int b = 0; b < std::min(batch_count, 2); b++) {
        printf("\nBatch %d:\n", b);

        // Print original diagonal (poles)
        int num_to_print = std::min(n, 8);
        printf("  Original D (first %d): ", num_to_print);
        for (int i = 0; i < num_to_print; i++) {
            printf("%.4f ", h_D[b * n + i]);
        }
        if (n > num_to_print) printf("...");
        printf("\n");

        // Print z vector
        printf("  z vector   (first %d): ", num_to_print);
        for (int i = 0; i < num_to_print; i++) {
            printf("%.4f ", h_z[b * n + i]);
        }
        if (n > num_to_print) printf("...");
        printf("\n");

        // Print computed eigenvalues
        printf("  Eigenvalues (first %d): ", num_to_print);
        for (int i = 0; i < num_to_print; i++) {
            printf("%.4f ", h_evs[b * n + i]);
        }
        if (n > num_to_print) printf("...");
        printf("\n");

        // Check if eigenvalues are interlaced with poles (basic sanity check)
        // For D + rho*z*z^T with rho > 0, eigenvalues should satisfy:
        // D[0] < lambda[0] < D[1] < lambda[1] < ... < lambda[n-1]
        bool interlacing_ok = true;
        for (int i = 0; i < n - 1; i++) {
            if (h_evs[b * n + i] <= h_D[b * n + i] ||
                h_evs[b * n + i] >= h_D[b * n + i + 1]) {
                interlacing_ok = false;
                printf("  WARNING: Interlacing violated at index %d: D[%d]=%.4f < lambda[%d]=%.4f < D[%d]=%.4f\n",
                       i, i, h_D[b * n + i], i, h_evs[b * n + i], i+1, h_D[b * n + i + 1]);
                break;
            }
        }
        if (interlacing_ok) {
            printf("  Interlacing check: PASSED\n");
        }
    }

    if (batch_count > 2) {
        printf("\n... (%d more batches not shown)\n", batch_count - 2);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    free(h_D);
    free(h_z);
    free(h_r1p);
    free(h_evs);
    free(h_nps);
    free(h_ndd);
    free(h_etmpd);

    cudaFree(d_D);
    cudaFree(d_z);
    cudaFree(d_r1p);
    cudaFree(d_evs);
    cudaFree(d_nps);
    cudaFree(d_ndd);
    cudaFree(d_etmpd);

    printf("Done.\n");
    return 0;
}
