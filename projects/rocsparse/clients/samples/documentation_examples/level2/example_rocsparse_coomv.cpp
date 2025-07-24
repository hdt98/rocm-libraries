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
    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // A sparse matrix
    // 1 0 3 4
    // 0 0 5 1
    // 0 2 0 0
    // 4 0 0 8
    rocsparse_int hArow[8] = {0, 0, 0, 1, 1, 2, 3, 3};
    rocsparse_int hAcol[8] = {0, 2, 3, 2, 3, 1, 0, 3};
    double        hAval[8] = {1.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 8.0};

    rocsparse_int m = 4;
    rocsparse_int n = 4;
    rocsparse_int nnz = 8;

    double halpha = 1.0;
    double hbeta  = 0.0;

    double  hx[4] = {1.0, 2.0, 3.0, 4.0};

    // Matrix descriptor
    rocsparse_mat_descr descrA;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrA));

    // Offload data to device
    rocsparse_int* dArow = NULL;
    rocsparse_int* dAcol = NULL;
    double*        dAval = NULL;
    double*        dx    = NULL;
    double*        dy    = NULL;

    HIP_CHECK(hipMalloc((void**)&dArow, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAcol, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dAval, sizeof(double) * nnz));
    HIP_CHECK(hipMalloc((void**)&dx, sizeof(double) * n));
    HIP_CHECK(hipMalloc((void**)&dy, sizeof(double) * m));

    HIP_CHECK(hipMemcpy(dArow, hArow, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAcol, hAcol, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dAval, hAval, sizeof(double) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, hx, sizeof(double) * n, hipMemcpyHostToDevice));

    // Call rocsparse coomv
    ROCSPARSE_CHECK(rocsparse_dcoomv(handle,
                    rocsparse_operation_none,
                    m,
                    n,
                    nnz,
                    &halpha,
                    descrA,
                    dAval,
                    dArow,
                    dAcol,
                    dx,
                    &hbeta,
                    dy));

    // Copy back to host
    double hy[4];
    HIP_CHECK(hipMemcpy(hy, dy, sizeof(double) * m, hipMemcpyDeviceToHost));

    // Clear up on device
    HIP_CHECK(hipFree(dArow));
    HIP_CHECK(hipFree(dAcol));
    HIP_CHECK(hipFree(dAval));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    return 0;
}