#pragma once
// ----------------------------------------------
// determine block sizes for simple_gemm_kernel()
// try to fit submatrices in LDS
// ----------------------------------------------

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

template <typename T, typename I>
__host__ __device__ static void get_gemm_nb(char const transA,
                                            char const transB,
                                            I const m,
                                            I const n,
                                            I const k,
                                            I* p_nb_m,
                                            I* p_nb_n,
                                            I* p_nb_k)
{
    assert(p_nb_m != nullptr);
    assert(p_nb_n != nullptr);
    assert(p_nb_k != nullptr);

    I const max_lds = (64 * 1024) / sizeof(T);
    I const nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    assert((3 * nb * nb) <= max_lds);

    *p_nb_m = nb;
    *p_nb_n = nb;
    *p_nb_k = nb;
}

// ------------------------------------------
// scale_beta_kernel to scale a matrix by beta
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// ------------------------------------------

template <typename T, typename I, typename Istride, typename UA>
static __global__ void scale_beta_kernel(I const m,
                                         I const n,
                                         T const beta,
                                         UA Amat,
                                         Istride const shift_Amat,
                                         I const ldA,
                                         Istride const stride_Amat,
                                         I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    if(beta == 1)
    {
        return;
    }

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipGridDim_x * hipBlockDim_x;

    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    I const j_inc = hipGridDim_y * hipBlockDim_y;

    bool const is_beta_zero = (beta == 0);
    T const zero = 0;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_ = load_ptr_batch(Amat, bid, shift_Amat, stride_Amat);

        auto A = [=](auto i, auto j) -> T& { return (A_[i + j * static_cast<int64_t>(ldA)]); };

        if(is_beta_zero)
        {
            for(auto j = j_start; j < n; j += j_inc)
            {
                for(auto i = i_start; i < m; i += i_inc)
                {
                    A(i, j) = zero;
                }
            }
        }
        else
        {
            // ----------------
            // multiply by beta
            // ----------------
            for(auto j = j_start; j < n; j += j_inc)
            {
                for(auto i = i_start; i < m; i += i_inc)
                {
                    A(i, j) *= beta;
                }
            }
        }
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UC>
static void scale_beta_template(rocblas_handle handle,

                                I const m,
                                I const n,
                                T const beta,

                                UC Cmat,
                                Istride const shift_Cmat,
                                I const ldC,
                                Istride const stride_Cmat,

                                I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    if(beta == 1)
    {
        return;
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    I const nx = 32;
    I const ny = 32;

    I const max_blocks = 1024;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    ROCSOLVER_LAUNCH_KERNEL((scale_beta_kernel<T, I, Istride, UC>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, 1), 0, stream,

                            m, n, beta,

                            Cmat, shift_Cmat, ldC, stride_Cmat,

                            batch_count);
}

// ------------------------------------------
// simple_gemm_kernel
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// ------------------------------------------

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static __global__ void simple_gemm_kernel(

    char const transA,
    char const transB,

    I const m,
    I const n,
    I const k,

    T const alpha,

    UA Amat,
    Istride const shift_Amat,
    I const ldA,
    Istride const stride_Amat,

    UB Bmat,
    Istride const shift_Bmat,
    I const ldB,
    Istride const stride_Bmat,

    // note no beta

    UC Cmat,
    Istride const shift_Cmat,
    I const ldC,
    Istride const stride_Cmat,

    I const batch_count)
{
    using S = decltype(std::real(T{}));
    bool constexpr is_complex = rocblas_is_complex<T>;

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1);
    if(!has_work)
    {
        return;
    };

    bool const is_transpose_A = (transA == 'T') || (transA == 't');
    bool const is_conj_transpose_A = (transA == 'C') || (transA == 'c');
    bool const is_no_transpose_A = (!is_transpose_A) && (!is_conj_transpose_A);

    bool const is_transpose_B = (transB == 'T') || (transB == 't');
    bool const is_conj_transpose_B = (transB == 'C') || (transB == 'c');
    bool const is_no_transpose_B = (!is_transpose_B) && (!is_conj_transpose_B);

    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    I const i_inc = nx;
    I const j_inc = ny;

    I const i_start = hipThreadIdx_x;
    I const j_start = hipThreadIdx_y;

    I nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    I nb_m = nb;
    I nb_n = nb;
    I nb_k = nb;
    get_gemm_nb<T>(transA, transB, m, n, k, &nb_m, &nb_n, &nb_k);

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    size_t const max_lds = 64 * 1024;
    size_t const lmem_size = max_lds / sizeof(T);
    __shared__ T lmem[lmem_size];
    size_t total_len = 0;

    I const mblocks = ceil(m, nb_m);
    I const nblocks = ceil(n, nb_n);
    I const kblocks = ceil(k, nb_k);

    I const nbx = hipGridDim_x;
    I const nby = hipGridDim_y;
    I const nbz = hipGridDim_z;

    I const ibx = hipBlockIdx_x;
    I const iby = hipBlockIdx_y;
    I const ibz = hipBlockIdx_z;
    // ---------------------------
    // default values without batch
    // ---------------------------
    I bid_start = 0;
    I bid_inc = 1;

    // ----------------------------------
    // use z-dimension for k-dimension without batch
    // ----------------------------------

    I ib_start = ibx;
    I ib_inc = nbx;

    I jb_start = iby;
    I jb_inc = nby;

    I kb_start = ibz;
    I kb_inc = nbz;

    if(batch_count > 1)
    {
        bid_start = ibz;
        bid_inc = nbz;

        // ----------------------------------
        // y-dimension blocks for k-dimension with batch
        // ----------------------------------
        kb_start = iby;
        kb_inc = nby;

        // --------------------------------------
        // split x-dimension blocks as
        // ib_inc by jb_inc grid of thread blocks
        //
        // want ibx == ib_start + jb_start * ib_inc
        // --------------------------------------
        ib_inc = std::min(nbx, mblocks);
        jb_inc = std::max(I(1), nbx / ib_inc);

        ib_start = ibx % ib_inc;
        jb_start = std::max(I(0), (ibx - ib_start) / ib_inc);
    }

    T* pfree = reinterpret_cast<T*>(&(lmem[0]));

    auto const ldAsh = nb_m;
    auto const size_Ash = nb_m * nb_k;
    T* const Ash_ = pfree;
    pfree += size_Ash;
    total_len += size_Ash;
    auto Ash = [=](auto i, auto j) -> T& { return (Ash_[(i - 1) + (j - 1) * ldAsh]); };

    auto const ldBsh = nb_k;
    auto const size_Bsh = nb_k * nb_n;
    T* const Bsh_ = pfree;
    pfree += size_Bsh;
    total_len += size_Bsh;
    auto Bsh = [=](auto i, auto j) -> T& { return (Bsh_[(i - 1) + (j - 1) * ldBsh]); };

    auto const ldCsh = nb_m;
    auto const size_Csh = nb_m * nb_n;
    T* const Csh_ = pfree;
    pfree += size_Csh;
    total_len += size_Csh;
    auto Csh = [=](auto i, auto j) -> T& { return (Csh_[(i - 1) + (j - 1) * ldCsh]); };

    assert(total_len <= lmem_size);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const __restrict__ A_ = load_ptr_batch(Amat, bid, shift_Amat, stride_Amat);
        T const* const __restrict__ B_ = load_ptr_batch(Bmat, bid, shift_Bmat, stride_Bmat);
        T* const __restrict__ C_ = load_ptr_batch(Cmat, bid, shift_Cmat, stride_Cmat);

        auto A = [=](auto i, auto j) -> const T& { return (A_[idx2F(i, j, ldA)]); };
        auto B = [=](auto i, auto j) -> const T& { return (B_[idx2F(i, j, ldB)]); };
        auto C = [=](auto i, auto j) -> T& { return (C_[idx2F(i, j, ldC)]); };

        for(I jb = 1 + jb_start; jb <= nblocks; jb += jb_inc)
        {
            for(I ib = 1 + ib_start; ib <= mblocks; ib += ib_inc)
            {
                auto const ic_start = 1 + (ib - 1) * nb_m;
                auto const ic_end = std::min(m, ic_start + nb_m - 1);

                auto const jc_start = 1 + (jb - 1) * nb_n;
                auto const jc_end = std::min(n, jc_start + nb_n - 1);

                auto const ic1 = ic_start;
                auto const ic2 = ic_end;

                auto const jc1 = jc_start;
                auto const jc2 = jc_end;

                auto const nrows_C = (ic2 - ic1 + 1);
                auto const ncols_C = (jc2 - jc1 + 1);

                // --------------------------------
                // Csh( 1:nrows_C, 1:ncols_C ) = 0;
                // --------------------------------

                __syncthreads();

                for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                {
                    for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                    {
                        Csh(i, j) = 0;
                    }
                }

                __syncthreads();

                for(auto kb = 1 + kb_start; kb <= kblocks; kb += kb_inc)
                {
                    auto const ik_start = 1 + (kb - 1) * nb_k;
                    auto const ik_end = std::min(k, ik_start + nb_k - 1);

                    auto const ik1 = ik_start;
                    auto const ik2 = ik_end;

                    // -----------------------------------------------------------------
                    // C(ic1:ic2, jc1:jc2) <- op(A(ia1:ia2, ja1:ja2)) * op(B( ib1:ib2,
                    // jb1:jb2))
                    // -----------------------------------------------------------------

                    auto const ia1 = (is_no_transpose_A) ? ic1 : ik1;
                    auto const ia2 = (is_no_transpose_A) ? ic2 : ik2;

                    auto const ja1 = (is_no_transpose_A) ? ik1 : ic1;
                    auto const ja2 = (is_no_transpose_A) ? ik2 : ic2;

                    auto const ib1 = (is_no_transpose_B) ? ik1 : jc1;
                    auto const ib2 = (is_no_transpose_B) ? ik2 : jc2;

                    auto const jb1 = (is_no_transpose_B) ? jc1 : ik1;
                    auto const jb2 = (is_no_transpose_B) ? jc2 : ik2;

                    auto const nrows_A = (ia2 - ia1 + 1);
                    auto const ncols_A = (ja2 - ja1 + 1);

                    // ---------------------------------------------------
                    // Ash( 1:nrows_A, 1:ncols_A ) = A( ia1:ia2, ja1:ja2 );
                    // ---------------------------------------------------

                    __syncthreads();

                    for(auto j = 1 + j_start; j <= ncols_A; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_A; i += i_inc)
                        {
                            auto const ia = (ia1 - 1) + i;
                            auto const ja = (ja1 - 1) + j;
                            Ash(i, j) = A(ia, ja);
                        }
                    }

                    auto const nrows_B = (ib2 - ib1 + 1);
                    auto const ncols_B = (jb2 - jb1 + 1);

                    // -----------------------------------------------
                    // Bsh(1:nrows_B, 1:ncols_B) = B(ib1:ib2, jb1:jb2);
                    // -----------------------------------------------

                    for(auto j = 1 + j_start; j <= ncols_B; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_B; i += i_inc)
                        {
                            auto const ib = (ib1 - 1) + i;
                            auto const jb = (jb1 - 1) + j;
                            Bsh(i, j) = B(ib, jb);
                        }
                    }
                    __syncthreads();

                    for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                        {
                            T cij = 0;
                            auto const nk = (is_no_transpose_A) ? ncols_A : nrows_A;

                            bool constexpr use_pointers = true;
                            if(use_pointers)
                            {
                                I const kk = 1;
                                T const* __restrict__ ap
                                    = (is_no_transpose_A) ? &(Ash(i, kk)) : &(Ash(kk, i));
                                I ap_inc = (is_no_transpose_A) ? ldAsh : 1;

                                T const* __restrict__ bp
                                    = (is_no_transpose_B) ? &(Bsh(kk, j)) : &(Bsh(j, kk));
                                I const bp_inc = (is_no_transpose_B) ? 1 : ldBsh;
                                for(I kk = 1; kk <= nk; kk++)
                                {
                                    T const aval = *ap;
                                    T const bval = *bp;
                                    T const aik = (is_conj_transpose_A) ? conj(aval) : aval;
                                    T const bkj = (is_conj_transpose_B) ? conj(bval) : bval;

                                    cij += aik * bkj;

                                    ap += ap_inc;
                                    bp += bp_inc;
                                }
                            }
                            else
                            {
                                for(I kk = 1; kk <= nk; kk++)
                                {
                                    T const aik = (is_no_transpose_A) ? Ash(i, kk)
                                        : (is_transpose_A)            ? Ash(kk, i)
                                                                      : conj(Ash(kk, i));

                                    T const bkj = (is_no_transpose_B) ? Bsh(kk, j)
                                        : (is_transpose_B)            ? Bsh(j, kk)
                                                                      : conj(Bsh(j, kk));

                                    cij += aik * bkj;
                                } // end for kk
                            }

                            Csh(i, j) += cij;
                        } // end for i
                    } // end for j

                    __syncthreads();
                } // for kb

                // -----------------------------------------------------------
                // C(ic1:ic2, jc1:jc2) +=  alpha * Csh( 1:nrows_C, 1:ncols_C );
                // -----------------------------------------------------------

                auto gatomicAdd = [](T* p, T value) {
                    S* px = (S*)p;
                    if constexpr(is_complex)
                    {
                        atomicAdd(px, std::real(value));
                        atomicAdd(px + 1, std::imag(value));
                    }
                    else
                    {
                        atomicAdd(px, value);
                    }
                };

                __syncthreads();
                for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                {
                    for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                    {
                        auto const ic = (ic1 - 1) + i;
                        auto const jc = (jc1 - 1) + j;

                        bool const need_atomic_update = (kb_inc > 1);
                        if(need_atomic_update)
                        {
                            gatomicAdd(&(C(ic, jc)), (alpha * Csh(i, j)));
                        }
                        else
                        {
                            C(ic, jc) += alpha * Csh(i, j);
                        }
                    }
                }
                __syncthreads();

            } // for ib
        } // for jb

    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static rocblas_status roclapack_simple_gemm_template(rocblas_handle handle,
                                                     rocblas_operation const transA,
                                                     rocblas_operation const transB,
                                                     I const m,
                                                     I const n,
                                                     I const k,

                                                     T* const p_alpha,

                                                     UA Amat,
                                                     Istride const shift_Amat,
                                                     I const ldA,
                                                     Istride const stride_Amat,

                                                     UB Bmat,
                                                     Istride const shift_Bmat,
                                                     I const ldB,
                                                     Istride const stride_Bmat,

                                                     T* const p_beta,

                                                     UC Cmat,
                                                     Istride const shift_Cmat,
                                                     I const ldC,
                                                     Istride const stride_Cmat,

                                                     I const batch_count,
                                                     void* workArr = nullptr)
{
    auto op2char = [](rocblas_operation transA) -> char {
        char const c_transA = (transA == rocblas_operation_none) ? 'N'
            : (transA == rocblas_operation_transpose)            ? 'T'
            : (transA == rocblas_operation_conjugate_transpose)  ? 'C'
                                                                 : 'X';
        return (c_transA);
    };

    char const c_transA = op2char(transA);
    char const c_transB = op2char(transB);

    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    assert(p_beta != nullptr);
    assert(p_alpha != nullptr);

    T const beta = *p_beta;
    T const alpha = *p_alpha;

    auto ceil = [](auto n, auto nb) { return (1 + ((n - 1) / nb)); };

    scale_beta_template<T, I, Istride, UC>(handle, m, n, beta, Cmat, shift_Cmat, ldC, stride_Cmat,
                                           batch_count);

    I nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    I nb_m = nb;
    I nb_n = nb;
    I nb_k = nb;

    get_gemm_nb<T>(c_transA, c_transB, m, n, k, &nb_m, &nb_n, &nb_k);

    auto const kblocks = ceil(k, nb_k);
    auto const mblocks = ceil(m, nb_m);
    auto const nblocks = ceil(n, nb_n);

    I const max_blocks = 64 * 1000;
    I const max_threads = 1024;

    I const nx = 32;
    I const ny = 32;

    auto const num_cu = get_num_cu();

    // --------------------------------------
    // if no batch (or equivalently batch_count == 1),
    // then use z-dimension as "k" dimension
    // --------------------------------------
    I const nbx_nobatch = std::min(max_blocks, mblocks);
    I const nby_nobatch = std::min(max_blocks, nblocks);
    I const nbz_nobatch
        = std::max(I(1), std::min(max_blocks, std::min(kblocks, num_cu / (mblocks * nblocks))));

    I const nbx_batch = std::max(I(1), mblocks * nblocks);
    I const nby_batch = std::max(
        I(1), std::min(max_blocks, std::min(kblocks, num_cu / (mblocks * nblocks * batch_count))));
    I const nbz_batch = std::max(I(1), std::min(max_blocks, batch_count));

    bool const has_batch = (batch_count > 1);
    I const nbx = (has_batch) ? nbx_batch : nbx_nobatch;
    I const nby = (has_batch) ? nby_batch : nby_nobatch;
    I const nbz = (has_batch) ? nbz_batch : nbz_nobatch;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    size_t const lmem_size = 64 * 1024;
    ROCSOLVER_LAUNCH_KERNEL((simple_gemm_kernel<T, I, Istride, UA, UB, UC>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, 1), 0, stream,

                            c_transA, c_transB, m, n, k, alpha,

                            Amat, shift_Amat, ldA, stride_Amat,

                            Bmat, shift_Bmat, ldB, stride_Bmat,

                            // note no beta

                            Cmat, shift_Cmat, ldC, stride_Cmat, batch_count);

    return (rocblas_status_success);
}
