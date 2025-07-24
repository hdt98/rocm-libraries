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
    // // It assumes the CSR arrays (ptr, ind, val) have already been allocated and filled.
    // // Build Source
    // rocsparse_spmat_descr source;
    // ROCSPARSE_CHECK(rocsparse_create_csr_descr(&source, M, N, nnz, ptr, ind, val, rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r));

    // // Build target
    // void * ell_ind, * ell_val;
    // int64_t ell_width = 0;
    // rocsparse_spmat_descr target;
    // ROCSPARSE_CHECK(rocsparse_create_ell_descr(&target, M, N, ell_ind, ell_val, ell_width, rocsparse_indextype_i32, rocsparse_index_base_zero, rocsparse_datatype_f32_r));

    // // Create descriptor
    // rocsparse_sparse_to_sparse_descr descr;
    // ROCSPARSE_CHECK(rocsparse_create_sparse_to_sparse_descr(&descr, source, target,  rocsparse_sparse_to_sparse_alg_default));

    // // Analysis phase
    // ROCSPARSE_CHECK(rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size));
    // HIP_CHECK(hipMalloc(&buffer,buffer_size));
    // ROCSPARSE_CHECK(rocsparse_sparse_to_sparse(handle, descr, source, target, rocsparse_sparse_to_sparse_stage_analysis, buffer_size, buffer));
    // HIP_CHECK(hipFree(buffer));

    // // the user is responsible to allocate target arrays after the analysis phase.
    // { 
    //     int64_t rows, cols, ell_width;
    //     void * ind, * val;
    //     rocsparse_indextype        idx_type;
    //     rocsparse_index_base       idx_base;
    //     rocsparse_datatype         data_type;

    //     ROCSPARSE_CHECK(rocsparse_ell_get(target,
    //                     &rows,
    //                     &cols,
    //                     &ind,
    //                     &val,
    //                     &ell_width,
    //                     &idx_type,
    //                     &idx_base,
    //                     &data_type));
    //     HIP_CHECK(hipMalloc(&ell_ind,ell_width * M * sizeof(int32_t)));
    //     HIP_CHECK(hipMalloc(&ell_val,ell_width * M * sizeof(float)));
    //     ROCSPARSE_CHECK(rocsparse_ell_set_pointers(target, ell_ind, ell_val)); 
    // }

    // // Calculation phase
    // ROCSPARSE_CHECK(rocsparse_sparse_to_sparse_buffer_size(handle, descr, source, target, rocsparse_sparse_to_sparse_stage_compute, &buffer_size));
    // HIP_CHECK(hipMalloc(&buffer,buffer_size));
    // ROCSPARSE_CHECK(rocsparse_sparse_to_sparse(handle, descr, source, target, rocsparse_sparse_to_sparse_stage_compute, buffer_size, buffer));
    // HIP_CHECK(hipFree(buffer));

    return 0;
}