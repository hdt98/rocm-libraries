// example_rocsparse_saxpyi.cpp

#include <rocsparse.h>
#include <hip/hip_runtime.h>

int main()
{
    // Number of non-zeros of the sparse vector
    rocsparse_int nnz = 3;

    // Sparse index vector
    rocsparse_int hx_ind[3] = {0, 3, 5};

    // Sparse value vector
    float hx_val[3] = {1.0f, 2.0f, 3.0f};

    // Dense vector
    float hy[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // Scalar alpha
    float alpha = 3.7f;

    // Index base
    rocsparse_index_base idx_base = rocsparse_index_base_zero;

    // Offload data to device
    rocsparse_int* dx_ind;
    float*        dx_val;
    float*        dy;

    hipMalloc((void**)&dx_ind, sizeof(rocsparse_int) * nnz);
    hipMalloc((void**)&dx_val, sizeof(float) * nnz);
    hipMalloc((void**)&dy, sizeof(float) * 9);

    hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx_val, hx_val, sizeof(float) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy, sizeof(float) * 9, hipMemcpyHostToDevice);

    // rocSPARSE handle
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Call saxpyi to perform y = y + alpha * x
    rocsparse_saxpyi(handle, nnz, &alpha, dx_val, dx_ind, dy, idx_base);

    // Copy result back to host
    hipMemcpy(hy, dy, sizeof(float) * 9, hipMemcpyDeviceToHost);

    // Clear rocSPARSE
    rocsparse_destroy_handle(handle);

    // Clear device memory
    hipFree(dx_ind);
    hipFree(dx_val);
    hipFree(dy);

    return 0;
}