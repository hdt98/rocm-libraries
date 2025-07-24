#include <iostream>
#include <rocsparse.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(stat)                                                        \
    {                                                                          \
        if(stat != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                         \
        }                                                                      \
    }

#define ROCSPARSE_CHECK(stat)                                                        \
    {                                                                                \
        if(stat != rocsparse_status_success)                                         \
        {                                                                            \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                               \
        }                                                                            \
    }

int main()
{
    // Number of non-zeros of the sparse vector
    rocsparse_int nnz = 3;

    // Sparse index vector
    rocsparse_int hx_ind[3] = {0, 3, 5};

    // Sparse value vector
    float hx_val[3] = {9.0, 2.0, 3.0};

    // Dense vector
    float hy[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    // Index base
    rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // Offload data to device
    rocsparse_int* dx_ind;
    float*         dx_val;
    float*         dy;

    HIP_CHECK(hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx_val, sizeof(float) * nnz));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(float) * 9));

    HIP_CHECK(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx_val, hx_val, sizeof(float) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice));

    // rocSPARSE handle
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Call ssctr
    ROCSPARSE_CHECK(rocsparse_ssctr(handle, nnz, dx_val, dx_ind, dy, idx_base));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(float) * 9, hipMemcpyDeviceToHost));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dx_ind));
    HIP_CHECK(hipFree(dx_val));
    HIP_CHECK(hipFree(dy));

    return 0;
}