#include <iostream>
#include <vector>
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
    //     1 4 0 0 0 0
    // A = 0 2 3 0 0 0
    //     5 0 0 7 8 0
    //     0 0 9 0 6 0
    int m   = 4;
    int n   = 6;

    std::vector<float> hdense = {1, 0, 5, 0, 4, 2, 0, 0, 0, 3, 0, 9, 0, 0, 7, 0, 0, 0, 8, 6, 0, 0, 0, 0};

    // Offload data to device
    int* dcsr_row_ptr;
    float* ddense;
    HIP_CHECK(hipMalloc((void**)&dcsr_row_ptr, sizeof(int) * (m + 1)));
    HIP_CHECK(hipMalloc((void**)&ddense, sizeof(float) * m * n));

    HIP_CHECK(hipMemcpy(ddense, hdense.data(), sizeof(float) * m * n, hipMemcpyHostToDevice));

    rocsparse_handle     handle;
    rocsparse_dnmat_descr matA;
    rocsparse_spmat_descr matB;

    rocsparse_indextype row_idx_type = rocsparse_indextype_i32;
    rocsparse_indextype col_idx_type = rocsparse_indextype_i32;
    rocsparse_datatype  data_type = rocsparse_datatype_f32_r;
    rocsparse_index_base idx_base = rocsparse_index_base_zero;

    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    // Create sparse matrix A
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(&matA, m, n, m, ddense, data_type, rocsparse_order_column));

    // Create dense matrix B
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matB,
                                m,
                                n,
                                0,
                                dcsr_row_ptr,
                                nullptr,
                                nullptr,
                                row_idx_type,
                                col_idx_type,
                                idx_base,
                                data_type));

    // Call dense_to_sparse to get required buffer size
    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_dense_to_sparse(handle,
                                matA,
                                matB,
                                rocsparse_dense_to_sparse_alg_default,
                                &buffer_size,
                                nullptr));

    void* temp_buffer;
    HIP_CHECK(hipMalloc((void**)&temp_buffer, buffer_size));

    // Call dense_to_sparse to perform analysis
    ROCSPARSE_CHECK(rocsparse_dense_to_sparse(handle,
                                matA,
                                matB,
                                rocsparse_dense_to_sparse_alg_default,
                                nullptr,
                                temp_buffer));

    int64_t num_rows_tmp, num_cols_tmp, nnz;
    ROCSPARSE_CHECK(rocsparse_spmat_get_size(matB, &num_rows_tmp, &num_cols_tmp, &nnz));

    int* dcsr_col_ind;
    float* dcsr_val;
    HIP_CHECK(hipMalloc((void**)&dcsr_col_ind, sizeof(int) * nnz));
    HIP_CHECK(hipMalloc((void**)&dcsr_val, sizeof(float) * nnz));

    ROCSPARSE_CHECK(rocsparse_csr_set_pointers(matB, dcsr_row_ptr, dcsr_col_ind, dcsr_val));

    // Call dense_to_sparse to complete conversion
    ROCSPARSE_CHECK(rocsparse_dense_to_sparse(handle,
                                matA,
                                matB,
                                rocsparse_dense_to_sparse_alg_default,
                                &buffer_size,
                                temp_buffer));

    std::vector<int> hcsr_row_ptr(m + 1, 0);
    std::vector<int> hcsr_col_ind(nnz, 0);
    std::vector<float> hcsr_val(nnz, 0);

    // Copy result back to host
    HIP_CHECK(hipMemcpy(hcsr_row_ptr.data(), dcsr_row_ptr, sizeof(int) * (m + 1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_col_ind.data(), dcsr_col_ind, sizeof(int) * nnz, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hcsr_val.data(), dcsr_val, sizeof(float) * nnz, hipMemcpyDeviceToHost));

    // Clear rocSPARSE
    ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(matA));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matB));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));

    // Clear device memory
    HIP_CHECK(hipFree(dcsr_row_ptr));
    HIP_CHECK(hipFree(dcsr_col_ind));
    HIP_CHECK(hipFree(dcsr_val));
    HIP_CHECK(hipFree(ddense));

    return 0;
}