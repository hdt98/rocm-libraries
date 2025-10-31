template <int MAX_THDS, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) lange_max_kernel(const rocblas_int m,
                                                                   const rocblas_int n,
                                                                   const U A,
                                                                   const rocblas_int lda,
                                                                   const rocblas_int shiftA,
                                                                   const rocblas_int strideA,
                                                                   T* final_norm)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* a = load_ptr_batch<T>(A, bid, shiftA, strideA);

    // shared variables
    __shared__ T sval[MAX_THDS / WarpSize];

    // dot
    T norm_max = 0;
    for(I i = tid; i < m * n; i += MAX_THDS)
    {
        int row = i % m;
        int col = i / m;
        
        norm_max = std::max(norm_max, rocblas_abs(a[row + col * lda]));
    }

    // reduce squared entries to find squared norm of x
    norm_max = std::max(norm_max, shift_left(norm_max, 1));
    norm_max = std::max(norm_max, shift_left(norm_max, 2));
    norm_max = std::max(norm_max, shift_left(norm_max, 4));
    norm_max = std::max(norm_max, shift_left(norm_max, 8));
    norm_max = std::max(norm_max, shift_left(norm_max, 16));
    if(warpSize > 32)
        norm_max = std::max(norm_max, shift_left(norm_max, 32));
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm_max;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm_max = std::max(norm_max, sval[k]);
        final_norm[bid] = norm_max;        
    }
}
