/* **************************************************************************
 * rocSOLVER Kernel Sandbox - POTF2 Driver (Cholesky Factorization)
 *
 * This program tests the POTF2 Cholesky factorization kernels for use
 * with CUDA Compute Sanitizer tools (memcheck, racecheck, initcheck,
 * synccheck).
 *
 * POTF2 computes the Cholesky factorization of a symmetric/Hermitian
 * positive definite matrix A:
 *   A = L * L'  (lower triangular)
 *   A = U' * U  (upper triangular)
 *
 * The potf2_kernel_small kernel uses a blocked algorithm with:
 * - Register-based storage for the triangular matrix
 * - Shared memory for panel factorization
 * - Support for NB panels of size PANEL_SIZE (32)
 *
 * This version is derived from rocm-libraries-angelo-potf2.
 *
 * Usage:
 *   ./sandbox_potf2 [n] [batch_count] [uplo]
 *
 * Where:
 *   n           - matrix dimension (default: 64, max: 256)
 *   batch_count - number of matrices in batch (default: 1)
 *   uplo        - triangular part: lower or upper (default: lower)
 *
 * Example:
 *   compute-sanitizer --tool racecheck ./sandbox_potf2 64 4 lower
 * *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "cuda_compat.cuh"
#include "kernels/potf2_kernels.cuh"

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
    printf("Usage: %s [n] [batch_count] [uplo]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  n           - matrix dimension (default: 64, max: %d)\n", POTF2_MAX_SMALL_SIZE);
    printf("  batch_count - number of matrices in batch (default: 1)\n");
    printf("  uplo        - triangular part: lower or upper (default: lower)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                  # Use defaults: n=64, batch=1, uplo=lower\n", prog_name);
    printf("  %s 64 4 lower       # n=64, batch=4, lower triangular\n", prog_name);
    printf("  %s 32 1 upper       # n=32, batch=1, upper triangular\n", prog_name);
    printf("\n");
    printf("Kernel uses blocked algorithm with:\n");
    printf("  - Panel size: %d\n", POTF2_PANEL_SIZE);
    printf("  - Max NB panels: %d\n", POTF2_MAX_NB);
    printf("  - Thread block: %dx%d\n", POTF2_PANEL_SIZE, POTF2_PANEL_SIZE);
    printf("\n");
    printf("Run with CUDA Compute Sanitizer:\n");
    printf("  compute-sanitizer --tool racecheck %s 64 4\n", prog_name);
}

// Generate a symmetric positive definite matrix
// A = B * B' + n * I (ensures positive definiteness)
void generate_spd_matrix(float* A, int n, int lda)
{
    // First, fill with random values
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[i + j * lda] = (float)rand() / RAND_MAX - 0.5f;
        }
    }

    // Compute A = A * A' (makes it symmetric positive semi-definite)
    float* temp = (float*)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i + k * lda] * A[j + k * lda];
            }
            temp[i + j * n] = sum;
        }
    }

    // Copy back and add n*I to make it positive definite
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[i + j * lda] = temp[i + j * n];
        }
        A[j + j * lda] += (float)n;  // Add n to diagonal
    }

    free(temp);
}

// Verify Cholesky factorization by computing L*L' (or U'*U) and comparing to A
float verify_cholesky(const float* A_orig, const float* L, int n, int lda, bool is_upper)
{
    float max_error = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float computed = 0.0f;

            if (is_upper) {
                // A = U' * U
                // A[i,j] = sum_k U[k,i] * U[k,j] for k <= min(i,j)
                int kmax = std::min(i, j) + 1;
                for (int k = 0; k < kmax; k++) {
                    computed += L[k + i * lda] * L[k + j * lda];
                }
            } else {
                // A = L * L'
                // A[i,j] = sum_k L[i,k] * L[j,k] for k <= min(i,j)
                int kmax = std::min(i, j) + 1;
                for (int k = 0; k < kmax; k++) {
                    computed += L[i + k * lda] * L[j + k * lda];
                }
            }

            float error = fabsf(computed - A_orig[i + j * lda]);
            if (error > max_error) {
                max_error = error;
            }
        }
    }

    return max_error;
}

int main(int argc, char** argv)
{
    // Check for help flag
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }

    // Default parameters
    int n = 64;           // Matrix dimension
    int batch_count = 1;  // Number of matrices in batch
    bool is_upper = false;  // Lower triangular by default

    // Parse command line args
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) batch_count = atoi(argv[2]);
    if (argc > 3) {
        if (strcmp(argv[3], "upper") == 0 || strcmp(argv[3], "U") == 0) {
            is_upper = true;
        }
    }

    printf("=== rocSOLVER Kernel Sandbox: POTF2 (Cholesky Factorization) ===\n");
    printf("Parameters: n=%d, batch_count=%d, uplo=%s\n", n, batch_count,
           is_upper ? "upper" : "lower");

    // Calculate number of panels
    int nb = (n + POTF2_PANEL_SIZE - 1) / POTF2_PANEL_SIZE;
    printf("Blocked algorithm: NB=%d panels of size %d\n", nb, POTF2_PANEL_SIZE);

    // Validate parameters
    if (n < 1 || n > POTF2_MAX_SMALL_SIZE) {
        fprintf(stderr, "Error: n must be in range [1, %d], got %d\n", POTF2_MAX_SMALL_SIZE, n);
        return 1;
    }
    if (batch_count < 1) {
        fprintf(stderr, "Error: batch_count must be >= 1, got %d\n", batch_count);
        return 1;
    }

    int lda = n;
    rocblas_stride strideA = (rocblas_stride)lda * n;

    // Allocate host memory
    printf("Allocating host memory...\n");

    float* h_A = (float*)malloc(strideA * batch_count * sizeof(float));
    float* h_A_orig = (float*)malloc(strideA * batch_count * sizeof(float));  // For verification
    int* h_info = (int*)calloc(batch_count, sizeof(int));

    if (!h_A || !h_A_orig || !h_info) {
        fprintf(stderr, "Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize test data - generate SPD matrices
    printf("Generating symmetric positive definite test matrices...\n");
    srand(42);

    for (int b = 0; b < batch_count; b++) {
        generate_spd_matrix(h_A + b * strideA, n, lda);

        // Save original for verification
        memcpy(h_A_orig + b * strideA, h_A + b * strideA, strideA * sizeof(float));
    }

    // Allocate device memory
    printf("Allocating device memory...\n");

    float* d_A;
    int* d_info;

    CUDA_CHECK(cudaMalloc(&d_A, strideA * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_count * sizeof(int)));

    // Copy data to device
    printf("Copying data to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, strideA * batch_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_count * sizeof(int)));

    // Calculate shared memory size
    size_t shared_mem_size = sizeof(float) * nb * POTF2_PANEL_SIZE * POTF2_PANEL_SIZE;

    printf("\nLaunching POTF2 kernel (blocked algorithm)...\n");
    printf("  Grid: (1, 1, %d), Block: (%d, %d, 1)\n",
           batch_count, POTF2_PANEL_SIZE, POTF2_PANEL_SIZE);
    printf("  Shared memory: %zu bytes\n", shared_mem_size);
    printf("  Register storage: %d elements per thread\n", (nb * (nb + 1)) / 2);

    // Launch the kernel using the helper function
    launch_potf2_kernel_small<float, int, int>(
        is_upper,
        n,
        d_A,
        0,  // shiftA
        lda,
        strideA,
        d_info,
        batch_count);

    CUDA_CHECK(cudaGetLastError());

    printf("Waiting for kernel completion...\n");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    printf("Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(h_A, d_A, strideA * batch_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_info, d_info, batch_count * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results summary
    printf("\n=== Results ===\n");
    bool all_success = true;

    for (int b = 0; b < std::min(batch_count, 2); b++) {
        printf("\nBatch %d:\n", b);
        printf("  info = %d%s\n", h_info[b],
               h_info[b] == 0 ? " (success)" : " (not positive definite)");

        if (h_info[b] == 0) {
            // Verify factorization
            float max_error = verify_cholesky(h_A_orig + b * strideA,
                                              h_A + b * strideA,
                                              n, lda, is_upper);
            printf("  Max reconstruction error: %.6e\n", max_error);

            if (max_error > 1e-4f * n) {
                printf("  WARNING: Error seems high!\n");
                all_success = false;
            } else {
                printf("  Verification: PASSED\n");
            }

            // Print first few elements of the factor
            int num_to_print = std::min(n, 5);
            printf("  %s factor (first %dx%d):\n",
                   is_upper ? "Upper" : "Lower", num_to_print, num_to_print);

            for (int i = 0; i < num_to_print; i++) {
                printf("    ");
                for (int j = 0; j < num_to_print; j++) {
                    if (is_upper) {
                        if (i <= j)
                            printf("%8.4f ", h_A[b * strideA + i + j * lda]);
                        else
                            printf("%8s ", "0");
                    } else {
                        if (i >= j)
                            printf("%8.4f ", h_A[b * strideA + i + j * lda]);
                        else
                            printf("%8s ", "0");
                    }
                }
                printf("\n");
            }
        } else {
            all_success = false;
        }
    }

    if (batch_count > 2) {
        printf("\n... (%d more batches not shown)\n", batch_count - 2);
    }

    printf("\n=== Summary ===\n");
    int success_count = 0;
    for (int b = 0; b < batch_count; b++) {
        if (h_info[b] == 0) success_count++;
    }
    printf("Successful factorizations: %d/%d\n", success_count, batch_count);

    // Cleanup
    printf("\nCleaning up...\n");
    free(h_A);
    free(h_A_orig);
    free(h_info);

    cudaFree(d_A);
    cudaFree(d_info);

    printf("Done.\n");
    return all_success ? 0 : 1;
}
