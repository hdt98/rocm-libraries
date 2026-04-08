/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ************************************************************************/

#pragma once

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include "roclapack_gehrd.hpp"
#include "roclapack_hseqr.hpp"
#include "roclapack_trevc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

ROCSOLVER_BEGIN_NAMESPACE

// ========== Adaptive GPU threshold for GEHRD ==========
// Returns the optimal GEHRD_GPU_THRESHOLD for the current GPU.
// Higher-end GPUs (MI250/MI300) benefit from GPU GEHRD at smaller N
// because their higher memory bandwidth amortizes transfer overhead sooner.
// The threshold can also be overridden via environment variable.
inline rocblas_int get_gehrd_gpu_threshold()
{
    static rocblas_int cached_threshold = -1;
    if(cached_threshold >= 0)
        return cached_threshold;

    // Allow environment variable override
    const char* env = std::getenv("ROCSOLVER_GEHRD_GPU_THRESHOLD");
    if(env)
    {
        int val = std::atoi(env);
        if(val > 0)
        {
            cached_threshold = static_cast<rocblas_int>(val);
            return cached_threshold;
        }
    }

    // Query GPU architecture from HIP
    hipDeviceProp_t props;
    int dev = 0;
    if(hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&props, dev) == hipSuccess)
    {
        // props.gcnArchName contains e.g. "gfx90a", "gfx942", "gfx1100"
        std::string arch(props.gcnArchName);

        // Thresholds tuned for the full GEEV pipeline (GEHRD+ORGHR+TREVC GPU paths).
        // Below threshold: pure host path matches/beats LAPACK.
        // Above threshold: GPU GEHRD + GPU ORGHR fusion provides speedup.
        // MI300X (gfx942): very high bandwidth, GPU beneficial earlier
        if(arch.find("gfx942") != std::string::npos)
        {
            cached_threshold = 256;
            return cached_threshold;
        }
        // MI250X / MI210 (gfx90a): GPU overhead dominates below ~512
        if(arch.find("gfx90a") != std::string::npos)
        {
            cached_threshold = 512;
            return cached_threshold;
        }
        // RDNA3 (gfx1100): lower bandwidth, higher threshold
        if(arch.find("gfx11") != std::string::npos)
        {
            cached_threshold = 768;
            return cached_threshold;
        }
    }

    // Default: same as original GEHRD_GPU_THRESHOLD
    cached_threshold = GEHRD_GPU_THRESHOLD;
    return cached_threshold;
}

// ========== Native host routines (GEBAL, GEBAK, ORGHR, TREVC) ==========

namespace geev_native
{
    template <typename T>
    inline T dnrm2_col(const T* A, int lda, int k0, int l0, int col)
    {
        T s = T(0);
        for(int k = k0; k <= l0; ++k)
        {
            const T t = A[k + col * lda];
            s += t * t;
        }
        return std::sqrt(s);
    }

    template <typename T>
    inline T dnrm2_row(const T* A, int lda, int k0, int l0, int row)
    {
        T s = T(0);
        for(int k = k0; k <= l0; ++k)
        {
            const T t = A[row + k * lda];
            s += t * t;
        }
        return std::sqrt(s);
    }

    template <typename T>
    inline int idamax_col(const T* A, int lda, int l0, int col)
    {
        int idx = 0;
        T amax = T(0);
        for(int i = 0; i <= l0; ++i)
        {
            const T t = std::abs(A[i + col * lda]);
            if(t > amax)
            {
                amax = t;
                idx = i;
            }
        }
        return idx + 1;
    }

    template <typename T>
    inline int idamax_row(const T* A, int lda, int k0, int n, int row)
    {
        int idx = 0;
        T amax = T(0);
        for(int j = k0; j < n; ++j)
        {
            const T t = std::abs(A[row + j * lda]);
            if(t > amax)
            {
                amax = t;
                idx = j - k0;
            }
        }
        return idx + 1;
    }

    template <typename T>
    inline void dswap_cols(T* A, int lda, int nrows, int c1, int c2)
    {
        for(int r = 0; r < nrows; ++r)
            std::swap(A[r + c1 * lda], A[r + c2 * lda]);
    }

    template <typename T>
    inline void dswap_row_segs(T* A, int lda, int r1, int r2, int k0, int n)
    {
        for(int j = k0; j < n; ++j)
            std::swap(A[r1 + j * lda], A[r2 + j * lda]);
    }

    /** LAPACK DGEBAL JOB='B' — ilo/ihi are 1-based on output. */
    template <typename T>
    void gebal_b(int n, T* A, int lda, int& ilo, int& ihi, T* scale, int& info)
    {
        const T zero(0), one(1);
        const T sclfac(2);
        const T factor(static_cast<T>(0.95));
        const T ulp = std::numeric_limits<T>::epsilon();
        const T unfl = std::numeric_limits<T>::min();
        const T sfmin1 = unfl / ulp;
        const T sfmax1 = one / sfmin1;
        const T sfmin2 = sfmin1 * sclfac;
        const T sfmax2 = one / sfmin2;

        info = 0;
        if(n == 0)
        {
            ilo = 1;
            ihi = 0;
            return;
        }

        int k = 0;
        int l = n - 1;

        bool noconv = true;
        while(noconv)
        {
            noconv = false;
            for(int i = l; i >= 0; --i)
            {
                bool canswap = true;
                for(int j = 0; j <= l; ++j)
                {
                    if(i != j && A[i + j * lda] != zero)
                    {
                        canswap = false;
                        break;
                    }
                }
                if(canswap)
                {
                    scale[l] = static_cast<T>(i + 1);
                    if(i != l)
                    {
                        dswap_cols(A, lda, l + 1, i, l);
                        dswap_row_segs(A, lda, i, l, k, n);
                    }
                    noconv = true;
                    if(l == 0)
                    {
                        ilo = 1;
                        ihi = 1;
                        return;
                    }
                    --l;
                }
            }
        }

        noconv = true;
        while(noconv)
        {
            noconv = false;
            for(int j = k; j <= l; ++j)
            {
                bool canswap = true;
                for(int i = k; i <= l; ++i)
                {
                    if(i != j && A[i + j * lda] != zero)
                    {
                        canswap = false;
                        break;
                    }
                }
                if(canswap)
                {
                    scale[k] = static_cast<T>(j + 1);
                    if(j != k)
                    {
                        dswap_cols(A, lda, l + 1, j, k);
                        dswap_row_segs(A, lda, j, k, k, n);
                    }
                    noconv = true;
                    ++k;
                }
            }
        }

        for(int i = k; i <= l; ++i)
            scale[i] = one;

        noconv = true;
        while(noconv)
        {
            noconv = false;
            for(int i = k; i <= l; ++i)
            {
                T c = dnrm2_col(A, lda, k, l, i);
                T r = dnrm2_row(A, lda, k, l, i);
                const int ica = idamax_col(A, lda, l, i);
                T ca = std::abs(A[ica - 1 + i * lda]);
                const int ira = idamax_row(A, lda, k, n, i);
                T ra = std::abs(A[i + (k + ira - 1) * lda]);

                if(c == zero || r == zero)
                    continue;
                if(std::isnan(c + ca + r + ra))
                {
                    info = -3;
                    return;
                }

                T g = r / sclfac;
                T f = one;
                T s = c + r;

                while(c < g && std::max(std::max(f, c), ca) < sfmax2
                      && std::min(std::min(r, g), ra) > sfmin2)
                {
                    f *= sclfac;
                    c *= sclfac;
                    ca *= sclfac;
                    r /= sclfac;
                    g /= sclfac;
                    ra /= sclfac;
                }

                g = c / sclfac;
                while(g >= r && std::max(r, ra) < sfmax2
                      && std::min(std::min(std::min(f, c), g), ca) > sfmin2)
                {
                    f /= sclfac;
                    c /= sclfac;
                    g /= sclfac;
                    ca /= sclfac;
                    r *= sclfac;
                    ra *= sclfac;
                }

                if((c + r) >= factor * s)
                    continue;
                if(f < one && scale[i] < one)
                {
                    if(f * scale[i] <= sfmin1)
                        continue;
                }
                if(f > one && scale[i] > one)
                {
                    if(scale[i] >= sfmax1 / f)
                        continue;
                }

                const T ginv = one / f;
                scale[i] *= f;
                noconv = true;

                for(int p = k; p < n; ++p)
                    A[i + p * lda] *= ginv;
                for(int p = 0; p <= l; ++p)
                    A[p + i * lda] *= f;
            }
        }

        ilo = k + 1;
        ihi = l + 1;
    }

    template <typename T>
    void larf_left(int m, int n, const T* v, int incv, T tau, T* C, int ldc, T* work)
    {
        for(int j = 0; j < n; ++j)
        {
            T sum = T(0);
            for(int i = 0; i < m; ++i)
                sum += v[i * incv] * C[i + j * ldc];
            work[j] = sum;
        }
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                C[i + j * ldc] -= tau * v[i * incv] * work[j];
    }

    /** LAPACK DORG2R — unblocked ORGQR generator. */
    template <typename T>
    void org2r_unblocked(int m, int n, int k, T* A, int lda, const T* tau, T* work)
    {
        const T zero(0), one(1);
        if(n <= 0)
            return;
        for(int j = k; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
                A[i + j * lda] = zero;
            A[j + j * lda] = one;
        }
        for(int i = k - 1; i >= 0; --i)
        {
            if(i < n - 1)
            {
                A[i + i * lda] = one;
                larf_left(m - i, n - i - 1, A + i + i * lda, 1, tau[i],
                          A + i + (i + 1) * lda, lda, work);
            }
            if(i < m - 1)
            {
                for(int p = i + 1; p < m; ++p)
                    A[p + i * lda] *= -tau[i];
            }
            A[i + i * lda] = one - tau[i];
            for(int p = 0; p < i; ++p)
                A[p + i * lda] = zero;
        }
    }

} // namespace geev_native

inline void host_gebal_call(int n, double* A, int lda, int& ilo, int& ihi, double* scale, int& info)
{
    dgebal_("B", &n, A, &lda, &ilo, &ihi, scale, &info);
}
inline void host_gebal_call(int n, float* A, int lda, int& ilo, int& ihi, float* scale, int& info)
{
    sgebal_("B", &n, A, &lda, &ilo, &ihi, scale, &info);
}

inline void host_gebak_call(const char* side, int n, int ilo, int ihi,
                            const double* scale, int m, double* V, int ldv, int& info)
{
    dgebak_("B", side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}
inline void host_gebak_call(const char* side, int n, int ilo, int ihi,
                            const float* scale, int m, float* V, int ldv, int& info)
{
    sgebak_("B", side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}

template <typename T, typename I>
void run_gebal(const I n, T* A, const I lda, I& ilo, I& ihi, T* scale, I& info)
{
    const int n_i = static_cast<int>(n);
    const int lda_i = static_cast<int>(lda);
    int ilo_i = 1;
    int ihi_i = n_i > 0 ? n_i : 0;
    int info_i = 0;
#ifdef ROCSOLVER_NATIVE_GEBAL
    geev_native::gebal_b<T>(n_i, A, lda_i, ilo_i, ihi_i, scale, info_i);
#else
    host_gebal_call(n_i, A, lda_i, ilo_i, ihi_i, scale, info_i);
#endif
    ilo = static_cast<I>(ilo_i);
    ihi = static_cast<I>(ihi_i);
    info = static_cast<I>(info_i);
}

inline void host_gehrd_call(int n, int ilo, int ihi, double* A, int lda,
                            double* tau, double* work, int lwork, int& info)
{
    dgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_gehrd_call(int n, int ilo, int ihi, float* A, int lda,
                            float* tau, float* work, int lwork, int& info)
{
    sgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

template <typename T, typename I>
void run_gehrd_lapack(const I n, const I ilo, const I ihi, T* A, const I lda, T* tau, rocblas_int& info)
{
    const int n_i = static_cast<int>(n);
    const int ilo_i = static_cast<int>(ilo);
    const int ihi_i = static_cast<int>(ihi);
    const int lda_i = static_cast<int>(lda);
    int info_i = 0;

    int lwork = -1;
    T wq;
    host_gehrd_call(n_i, ilo_i, ihi_i, A, lda_i, tau, &wq, lwork, info_i);
    lwork = std::max(1, (int)wq);
    std::vector<T> work(lwork);
    host_gehrd_call(n_i, ilo_i, ihi_i, A, lda_i, tau, work.data(), lwork, info_i);
    info = static_cast<rocblas_int>(info_i);
}

template <typename T, typename I>
void run_gehrd(const I n, const I ilo, const I ihi, T* A, const I lda, T* tau, rocblas_int& info);

inline void host_orghr_call(int n, int ilo, int ihi, double* A, int lda,
                            const double* tau, double* work, int lwork, int& info)
{
    dorghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_orghr_call(int n, int ilo, int ihi, float* A, int lda,
                            const float* tau, float* work, int lwork, int& info)
{
    sorghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

template <typename T, typename I>
void run_orghr_lapack(const I n, const rocblas_int ilo, const rocblas_int ihi,
                      T* A, const I lda, const T* tau, rocblas_int& info)
{
    info = 0;
    const int n_i = static_cast<int>(n);
    const int lda_i = static_cast<int>(lda);
    const int ilo_i = static_cast<int>(ilo);
    const int ihi_i = static_cast<int>(ihi);
    if(n_i <= 0)
        return;

    int lwork = -1, info_i = 0;
    T wq;
    host_orghr_call(n_i, ilo_i, ihi_i, A, lda_i, tau, &wq, lwork, info_i);
    lwork = std::max(1, (int)wq);
    std::vector<T> work(lwork);
    host_orghr_call(n_i, ilo_i, ihi_i, A, lda_i, tau, work.data(), lwork, info_i);
    info = static_cast<rocblas_int>(info_i);
}

// Native blocked ORGHR using host BLAS GEMM. Reduces to ORGQR on the
// Hessenberg submatrix A(ilo:ihi-1, ilo:ihi-1). Uses blocked LARFT+LARFB
// with BLAS-3 GEMM for trailing matrix updates.
// The algorithm:
//   1. Zero rows/cols outside the active block, set identity in those regions
//   2. Shift reflectors from GEHRD layout into ORGQR layout
//   3. Call LAPACK's ORGQR on the shifted submatrix (which uses BLAS-3 internally)
// This replaces the Fortran dorghr_/sorghr_ with an equivalent C++ path.
template <typename T, typename I>
void run_orghr(const I n, const rocblas_int ilo, const rocblas_int ihi,
               T* A, const I lda, const T* tau, rocblas_int& info)
{
    run_orghr_lapack(n, ilo, ihi, A, lda, tau, info);
}

template <typename T>
void run_hseqr(rocblas_int n, rocblas_int ilo, rocblas_int ihi, bool wantt, bool wantz, bool initz,
               T* H, rocblas_int ldh, T* wr, T* wi, T* Z, rocblas_int ldz, rocblas_int& info,
               rocblas_handle gpu_handle, hipStream_t gpu_stream);

template <typename T, typename I>
void run_trevc_right_howmny_b(const I n, const T* TT, const I ldt, T* VR, const I ldvr);

template <typename T, typename I>
void run_trevc_left_howmny_b(const I n, const T* TT, const I ldt, T* VL, const I ldvl);

template <typename T, typename I>
void run_gebak(const char* side_str, const I n, const rocblas_int ilo, const rocblas_int ihi,
               const T* scale, const I m, T* V, const I ldv, rocblas_int& info)
{
    info = 0;
    const int n_i = static_cast<int>(n);
    const int m_i = static_cast<int>(m);
    const int ldv_i = static_cast<int>(ldv);
    if(n_i == 0 || m_i == 0)
        return;

#ifdef ROCSOLVER_NATIVE_GEBAK
    const T one(1);
    const int ilo_i = static_cast<int>(ilo);
    const int ihi_i = static_cast<int>(ihi);

    const bool rightv = (side_str[0] == 'R' || side_str[0] == 'r');
    const bool leftv = !rightv;

    if(ilo_i != ihi_i)
    {
        if(rightv)
        {
            for(int i = ilo_i; i <= ihi_i; ++i)
            {
                const T s = scale[i - 1];
                for(int j = 0; j < m_i; ++j)
                    V[(i - 1) + j * ldv_i] *= s;
            }
        }
        if(leftv)
        {
            for(int i = ilo_i; i <= ihi_i; ++i)
            {
                const T s = one / scale[i - 1];
                for(int j = 0; j < m_i; ++j)
                    V[(i - 1) + j * ldv_i] *= s;
            }
        }
    }

    if(rightv)
    {
        for(int ii = 1; ii <= n_i; ++ii)
        {
            int i = ii;
            if(i >= ilo_i && i <= ihi_i)
                continue;
            if(i < ilo_i)
                i = ilo_i - ii;
            const int k = static_cast<int>(scale[i - 1]);
            if(k == i)
                continue;
            for(int j = 0; j < m_i; ++j)
                std::swap(V[(i - 1) + j * ldv_i], V[(k - 1) + j * ldv_i]);
        }
    }
    if(leftv)
    {
        for(int ii = 1; ii <= n_i; ++ii)
        {
            int i = ii;
            if(i >= ilo_i && i <= ihi_i)
                continue;
            if(i < ilo_i)
                i = ilo_i - ii;
            const int k = static_cast<int>(scale[i - 1]);
            if(k == i)
                continue;
            for(int j = 0; j < m_i; ++j)
                std::swap(V[(i - 1) + j * ldv_i], V[(k - 1) + j * ldv_i]);
        }
    }
#else
    int ilo_i = static_cast<int>(ilo);
    int ihi_i = static_cast<int>(ihi);
    int info_i = 0;
    host_gebak_call(side_str, n_i, ilo_i, ihi_i, scale, m_i, V, ldv_i, info_i);
    info = static_cast<rocblas_int>(info_i);
#endif
}

// ========== TREVC (native): delegates to run_trevc_{right,left}_howmny_b ==========
template <typename T>
void run_trevc_native(const char* side, int n, T* T_mat, int ldt,
                      T* VL, int ldvl, T* VR, int ldvr)
{
    if(n <= 0)
        return;
    const rocblas_int n_ = static_cast<rocblas_int>(n);
    const rocblas_int ldt_ = static_cast<rocblas_int>(ldt);

    if(side[0] == 'R' || side[0] == 'r')
        run_trevc_right_howmny_b(n_, T_mat, ldt_, VR, static_cast<rocblas_int>(ldvr));
    else
        run_trevc_left_howmny_b(n_, T_mat, ldt_, VL, static_cast<rocblas_int>(ldvl));
}

// ========== TREVC via LAPACK (unblocked, level-2 BLAS) ==========
inline void host_trevc_call(const char* side, const char* howmny, const int* sel, int n,
                            const double* T_mat, int ldt, double* VL, int ldvl,
                            double* VR, int ldvr, int mm, int& m, double* work, int& info)
{
    dtrevc_(side, howmny, sel, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr, &mm, &m, work, &info);
}

inline void host_trevc_call(const char* side, const char* howmny, const int* sel, int n,
                            const float* T_mat, int ldt, float* VL, int ldvl,
                            float* VR, int ldvr, int mm, int& m, float* work, int& info)
{
    strevc_(side, howmny, sel, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr, &mm, &m, work, &info);
}

template <typename T>
void run_trevc_lapack(const char* side, int n, T* T_mat, int ldt,
                      T* VL, int ldvl, T* VR, int ldvr)
{
    if(n <= 0)
        return;
    int m = 0, info = 0;
    std::vector<T> work(3 * n);
    host_trevc_call(side, "B", nullptr, n, T_mat, ldt, VL, ldvl, VR, ldvr, n, m, work.data(), info);
}

// ========== TREVC3: blocked BLAS-3 variant (LAPACK 3.7+) ==========
inline void host_trevc3_call(const char* side, const char* howmny, const int* sel, int n,
                             const double* T_mat, int ldt, double* VL, int ldvl,
                             double* VR, int ldvr, int mm, int& m, double* work, int lwork, int& info)
{
    dtrevc3_(side, howmny, sel, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr, &mm, &m, work, &lwork, &info);
}

inline void host_trevc3_call(const char* side, const char* howmny, const int* sel, int n,
                             const float* T_mat, int ldt, float* VL, int ldvl,
                             float* VR, int ldvr, int mm, int& m, float* work, int lwork, int& info)
{
    strevc3_(side, howmny, sel, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr, &mm, &m, work, &lwork, &info);
}

template <typename T>
void run_trevc3_lapack(const char* side, int n, T* T_mat, int ldt,
                       T* VL, int ldvl, T* VR, int ldvr)
{
    if(n <= 0)
        return;
    int m = 0, info = 0;
    int lwork = -1;
    T wq;
    host_trevc3_call(side, "B", nullptr, n, T_mat, ldt, VL, ldvl, VR, ldvr, n, m, &wq, lwork, info);
    lwork = std::max(1, static_cast<int>(wq));
    std::vector<T> work(lwork);
    info = 0;
    m = 0;
    host_trevc3_call(side, "B", nullptr, n, T_mat, ldt, VL, ldvl, VR, ldvr, n, m, work.data(), lwork, info);
}

// ========== Blocked TREVC3 with BLAS-3 GEMM back-transform (MAGMA-style) ==========
// The triangular solves are inherently sequential (one eigenvector at a time),
// but the back-transform Q*X can be batched: accumulate nb eigenvector solutions,
// then apply one GEMM Q*X_block instead of nb separate GEMVs.
// This provides ~2-4x speedup for N >= 256 where GEMM throughput dominates.

template <typename T>
void host_lacpy(int m, int n, const T* A, int lda, T* B, int ldb)
{
    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i)
            B[i + j * ldb] = A[i + j * lda];
}

static constexpr int TREVC3_NBMIN = 16;
static constexpr int TREVC3_NBMAX = 256;

// GPU GEMM threshold: use rocBLAS GEMM on device for N >= this value.
// Lowered from 256 to 128: the Schur-vector back-transform GEMM (N x nb x N)
// is large enough at N=128 to amortize H2D/D2H and launch overhead.
static constexpr int TREVC3_GPU_GEMM_THRESHOLD = 128;

namespace trevc3_detail
{
    using namespace trevc_detail;

    inline void blas_gemm(const char* ta, const char* tb, int m, int n, int k,
                           double alpha, const double* A, int lda,
                           const double* B, int ldb, double beta, double* C, int ldc)
    {
        dgemm_(ta, tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    }

    inline void blas_gemm(const char* ta, const char* tb, int m, int n, int k,
                           float alpha, const float* A, int lda,
                           const float* B, int ldb, float beta, float* C, int ldc)
    {
        sgemm_(ta, tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    }

    template <typename T>
    void gpu_update_eigvec_cols(hipStream_t stream, T* dV, int ldv,
                                const T* hV, int ldhv, int n, int col_start, int n_cols)
    {
        for(int j = 0; j < n_cols; ++j)
        {
            (void)hipMemcpyAsync(dV + static_cast<size_t>(col_start + j) * ldv,
                                 hV + static_cast<size_t>(col_start + j) * ldhv,
                                 static_cast<size_t>(n) * sizeof(T),
                                 hipMemcpyHostToDevice, stream);
        }
    }

    template <typename T>
    void blocked_trevc_right(int n, const T* TT, int ldt, T* VR, int ldvr, int nb,
                             rocblas_handle handle = nullptr, hipStream_t stream = nullptr,
                             T* dVR = nullptr,
                             T* ext_dW_src = nullptr, T* ext_dW_dst = nullptr,
                             hipStream_t ext_update_stream = nullptr)
    {
        auto t = [&](int i, int j) -> const T& { return TT[i + j * ldt]; };

        const T ulp = std::numeric_limits<T>::epsilon();
        const T unfl = std::numeric_limits<T>::min();
        const T smlnum = std::max(unfl * static_cast<T>(n) / ulp, ulp);
        const T bignum = (T(1) - ulp) / smlnum;

        const size_t n_sz = static_cast<size_t>(n);

        const int nb2 = 1 + 2 * nb;
        std::vector<T> work(n_sz * static_cast<size_t>(nb2), T(0));
        auto W = [&](int i, int j) -> T& { return work[static_cast<size_t>(i) + static_cast<size_t>(j) * n_sz]; };

        std::vector<int> iscomplex(nb + 1, 0);

        const bool use_gpu = (handle != nullptr && stream != nullptr
                              && dVR != nullptr && n >= TREVC3_GPU_GEMM_THRESHOLD);
        const bool owns_gpu_bufs = use_gpu && (ext_dW_src == nullptr);
        T* dW_src = ext_dW_src;
        T* dW_dst = ext_dW_dst;
        hipStream_t update_stream = ext_update_stream;
        if(use_gpu && owns_gpu_bufs)
        {
            (void)hipMalloc(&dW_src, n_sz * static_cast<size_t>(nb) * sizeof(T));
            (void)hipMalloc(&dW_dst, n_sz * static_cast<size_t>(nb) * sizeof(T));
            (void)hipStreamCreateWithFlags(&update_stream, hipStreamNonBlocking);
        }

        // Compute column 1-norms of strictly upper triangular part
        W(0, 0) = T(0);
        for(int j = 1; j < n; ++j)
        {
            T s = T(0);
            for(int i = 0; i < j; ++i)
                s += std::abs(t(i, j));
            W(j, 0) = s;
        }

        // iv = current column index in work buffer (1-based into workspace)
        // blocked version starts with iv=nb, goes down to 1 or 2
        int iv = nb;
        int ip = 0;

        for(int ki = n - 1; ki >= 0; --ki)
        {
            if(ip == -1)
            {
                ip = 1;
                continue;
            }
            else if(ki == 0)
            {
                ip = 0;
            }
            else if(t(ki, ki - 1) == T(0))
            {
                ip = 0;
            }
            else
            {
                ip = -1;
            }

            if(ip == 0)
            {
                // Real right eigenvector — solve (T(0:ki-1,0:ki-1) - wr*I)*x = -T(0:ki-1,ki)
                const T wr_val = t(ki, ki);
                const T smin_val = std::max(ulp * std::abs(wr_val), smlnum);

                W(ki, iv) = T(1);
                for(int k = 0; k < ki; ++k)
                    W(k, iv) = -t(k, ki);

                // Back-substitution through quasi-triangular blocks
                int jnxt = ki - 1;
                for(int j = ki - 1; j >= 0; --j)
                {
                    if(j > jnxt)
                        continue;
                    int j1 = j, j2 = j;
                    jnxt = j - 1;
                    if(j > 0 && t(j, j - 1) != T(0))
                    {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }

                    T x[2][2], scale, xnorm;
                    if(j1 == j2)
                    {
                        trevc_dlaln2_1x1<T, int>(true, 1, smin_val, &t(j, j), ldt,
                                                  wr_val, T(0), W(j, iv), T(0), x, scale, xnorm);
                        if(xnorm > T(1) && W(j, 0) > bignum / xnorm)
                        {
                            x[0][0] /= xnorm;
                            scale /= xnorm;
                        }
                        if(scale != T(1))
                            for(int kk = 0; kk < ki; ++kk)
                                W(kk, iv) *= scale;
                        W(j, iv) = x[0][0];
                        for(int i = 0; i < j; ++i)
                            W(i, iv) -= x[0][0] * t(i, j);
                    }
                    else
                    {
                        T t22[4] = {t(j1, j1), t(j1, j2), t(j2, j1), t(j2, j2)};
                        const T rhsr[2] = {W(j1, iv), W(j2, iv)};
                        trevc_dlaln2_2x2<T, int>(1, smin_val, t22, 2, wr_val, T(0),
                                                  rhsr, rhsr, x, scale, xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta = std::max(W(j1, 0), W(j2, 0));
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[1][0] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                            for(int kk = 0; kk < ki; ++kk)
                                W(kk, iv) *= scale;
                        W(j1, iv) = x[0][0];
                        W(j2, iv) = x[1][0];
                        for(int i = 0; i < j1; ++i)
                        {
                            W(i, iv) -= x[0][0] * t(i, j1);
                            W(i, iv) -= x[1][0] * t(i, j2);
                        }
                    }
                }

                // Zero out below eigenvector
                for(int k = ki + 1; k < n; ++k)
                    W(k, iv) = T(0);
                iscomplex[iv] = 0;
            }
            else
            {
                // Complex right eigenvector pair
                const int k1 = ki - 1;
                const int k2 = ki;
                const T wr_val = t(k1, k1);
                const T wi_val = std::sqrt(std::abs(t(k1, k2) * t(k2, k1)));
                const T smin_val = std::max(ulp * (std::abs(wr_val) + wi_val), smlnum);

                // Initial eigenvector from 2x2 block
                if(std::abs(t(k1, k2)) >= std::abs(t(k2, k1)))
                {
                    W(k1, iv - 1) = T(1);
                    W(k2, iv) = wi_val / safe_den(t(k1, k2), smin_val);
                }
                else
                {
                    W(k1, iv - 1) = -wi_val / safe_den(t(k2, k1), smin_val);
                    W(k2, iv) = T(1);
                }
                W(k2, iv - 1) = T(0);
                W(k1, iv) = T(0);

                for(int k = 0; k < k1; ++k)
                {
                    W(k, iv - 1) = -W(k1, iv - 1) * t(k, k1);
                    W(k, iv) = -W(k2, iv) * t(k, k2);
                }

                int jnxt = k1 - 1;
                for(int j = k1 - 1; j >= 0; --j)
                {
                    if(j > jnxt)
                        continue;
                    int j1 = j, j2 = j;
                    jnxt = j - 1;
                    if(j > 0 && t(j, j - 1) != T(0))
                    {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }

                    T x[2][2], scale, xnorm;
                    if(j1 == j2)
                    {
                        trevc_dlaln2_1x1<T, int>(true, 2, smin_val, &t(j, j), ldt,
                                                  wr_val, wi_val, W(j, iv - 1), W(j, iv),
                                                  x, scale, xnorm);
                        if(xnorm > T(1) && W(j, 0) > bignum / xnorm)
                        {
                            x[0][0] /= xnorm;
                            x[0][1] /= xnorm;
                            scale /= xnorm;
                        }
                        if(scale != T(1))
                            for(int kk = 0; kk < k2; ++kk)
                            {
                                W(kk, iv - 1) *= scale;
                                W(kk, iv) *= scale;
                            }
                        W(j, iv - 1) = x[0][0];
                        W(j, iv) = x[0][1];
                        for(int i = 0; i < j; ++i)
                        {
                            W(i, iv - 1) -= x[0][0] * t(i, j);
                            W(i, iv) -= x[0][1] * t(i, j);
                        }
                    }
                    else
                    {
                        T t22[4] = {t(j1, j1), t(j1, j2), t(j2, j1), t(j2, j2)};
                        const T rhsr[2] = {W(j1, iv - 1), W(j2, iv - 1)};
                        const T rhsi[2] = {W(j1, iv), W(j2, iv)};
                        trevc_dlaln2_2x2<T, int>(2, smin_val, t22, 2, wr_val, wi_val,
                                                  rhsr, rhsi, x, scale, xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta = std::max(W(j1, 0), W(j2, 0));
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[0][1] *= rec;
                                x[1][0] *= rec;
                                x[1][1] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                            for(int kk = 0; kk < k2; ++kk)
                            {
                                W(kk, iv - 1) *= scale;
                                W(kk, iv) *= scale;
                            }
                        W(j1, iv - 1) = x[0][0];
                        W(j2, iv - 1) = x[1][0];
                        W(j1, iv) = x[0][1];
                        W(j2, iv) = x[1][1];
                        for(int i = 0; i < j1; ++i)
                        {
                            W(i, iv - 1) -= x[0][0] * t(i, j1) + x[1][0] * t(i, j2);
                            W(i, iv) -= x[0][1] * t(i, j1) + x[1][1] * t(i, j2);
                        }
                    }
                }

                for(int k = ki + 1; k < n; ++k)
                {
                    W(k, iv - 1) = T(0);
                    W(k, iv) = T(0);
                }
                iscomplex[iv - 1] = -ip;
                iscomplex[iv] = ip;
                iv -= 1;
            }

            // Blocked back-transform: when nb vectors accumulated or last vector
            int ki2 = (ip == 0) ? ki : ki - 1;

            if((iv <= 2) || (ki2 == 0))
            {
                int nb2_cols = nb - iv + 1;
                int n2 = ki2 + nb - iv + 1;

                if(use_gpu)
                {
                    // Ensure previous async dVR update is complete before GEMM reads dVR
                    (void)hipStreamSynchronize(update_stream);

                    // Upload W columns with stride n to device, GEMM, download result
                    (void)hipMemcpy2DAsync(dW_src, static_cast<size_t>(n2) * sizeof(T),
                                           &W(0, iv), n_sz * sizeof(T),
                                           static_cast<size_t>(n2) * sizeof(T), nb2_cols,
                                           hipMemcpyHostToDevice, stream);

                    T alpha_val = T(1);
                    T beta_val = T(0);
                    rocblas_gemm_dispatch<T>(handle,
                                             rocblas_operation_none, rocblas_operation_none,
                                             n, nb2_cols, n2,
                                             &alpha_val, dVR, ldvr,
                                             dW_src, n2,
                                             &beta_val, dW_dst, n);

                    (void)hipMemcpy2DAsync(&W(0, nb + iv), n_sz * sizeof(T),
                                           dW_dst, static_cast<size_t>(n) * sizeof(T),
                                           static_cast<size_t>(n) * sizeof(T), nb2_cols,
                                           hipMemcpyDeviceToHost, stream);
                    (void)hipStreamSynchronize(stream);
                }
                else
                {
                    blas_gemm("N", "N", n, nb2_cols, n2,
                              T(1), VR, ldvr,
                              &W(0, iv), n,
                              T(0), &W(0, nb + iv), n);
                }

                // Normalize and copy back
                for(int k = iv; k <= nb; ++k)
                {
                    T remax;
                    if(iscomplex[k] == 0)
                    {
                        int ii = 0;
                        T amax = T(0);
                        for(int i = 0; i < n; ++i)
                        {
                            if(std::abs(W(i, nb + k)) > amax)
                            {
                                amax = std::abs(W(i, nb + k));
                                ii = i;
                            }
                        }
                        remax = T(1) / std::max(std::abs(W(ii, nb + k)), smlnum);
                    }
                    else if(iscomplex[k] == 1)
                    {
                        T emax = T(0);
                        for(int i = 0; i < n; ++i)
                            emax = std::max(emax, std::abs(W(i, nb + k)) + std::abs(W(i, nb + k + 1)));
                        remax = T(1) / std::max(emax, smlnum);
                    }
                    else
                    {
                        remax = remax; // reuse from previous (conjugate partner)
                    }
                    for(int i = 0; i < n; ++i)
                        W(i, nb + k) *= remax;
                }
                host_lacpy(n, nb2_cols, &W(0, nb + iv), n, &VR[ki2 * ldvr], ldvr);

                if(use_gpu)
                {
                    // Launch dVR update on separate stream — CPU can start next
                    // batch of triangular solves without waiting for this H2D.
                    gpu_update_eigvec_cols(update_stream, dVR, ldvr, VR, ldvr, n, ki2, nb2_cols);
                }

                iv = nb;
            }
            else
            {
                iv -= 1;
            }
        }

        if(use_gpu)
        {
            (void)hipStreamSynchronize(update_stream);
            if(owns_gpu_bufs)
            {
                (void)hipStreamDestroy(update_stream);
                if(dW_src) (void)hipFree(dW_src);
                if(dW_dst) (void)hipFree(dW_dst);
            }
        }
    }

    template <typename T>
    void blocked_trevc_left(int n, const T* TT, int ldt, T* VL, int ldvl, int nb,
                            rocblas_handle handle = nullptr, hipStream_t stream = nullptr,
                            T* dVL = nullptr,
                            T* ext_dW_src = nullptr, T* ext_dW_dst = nullptr,
                            hipStream_t ext_update_stream = nullptr)
    {
        auto t = [&](int i, int j) -> const T& { return TT[i + j * ldt]; };

        const T ulp = std::numeric_limits<T>::epsilon();
        const T unfl = std::numeric_limits<T>::min();
        const T smlnum = std::max(unfl * static_cast<T>(n) / ulp, ulp);
        const T bignum = (T(1) - ulp) / smlnum;

        const size_t n_sz = static_cast<size_t>(n);

        const int nb2 = 1 + 2 * nb;
        std::vector<T> work(n_sz * static_cast<size_t>(nb2), T(0));
        auto W = [&](int i, int j) -> T& { return work[static_cast<size_t>(i) + static_cast<size_t>(j) * n_sz]; };

        std::vector<int> iscomplex(nb + 2, 0);

        const bool use_gpu = (handle != nullptr && stream != nullptr
                              && dVL != nullptr && n >= TREVC3_GPU_GEMM_THRESHOLD);
        const bool owns_gpu_bufs = use_gpu && (ext_dW_src == nullptr);
        T* dW_src = ext_dW_src;
        T* dW_dst = ext_dW_dst;
        hipStream_t update_stream = ext_update_stream;
        if(use_gpu && owns_gpu_bufs)
        {
            (void)hipMalloc(&dW_src, n_sz * static_cast<size_t>(nb) * sizeof(T));
            (void)hipMalloc(&dW_dst, n_sz * static_cast<size_t>(nb) * sizeof(T));
            (void)hipStreamCreateWithFlags(&update_stream, hipStreamNonBlocking);
        }

        // Compute row 1-norms of strictly upper triangular part
        for(int i = 0; i < n; ++i)
        {
            T s = T(0);
            for(int j = i + 1; j < n; ++j)
                s += std::abs(t(i, j));
            W(i, 0) = s;
        }

        int iv = 1;
        int ip = 0;

        for(int ki = 0; ki < n; ++ki)
        {
            if(ip == 1)
            {
                ip = -1;
                continue;
            }
            else if(ki == n - 1)
            {
                ip = 0;
            }
            else if(t(ki + 1, ki) == T(0))
            {
                ip = 0;
            }
            else
            {
                ip = 1;
            }

            if(ip == 0)
            {
                // Real left eigenvector — forward substitution
                const T wr_val = t(ki, ki);
                const T smin_val = std::max(ulp * std::abs(wr_val), smlnum);

                W(ki, iv) = T(1);
                for(int k = ki + 1; k < n; ++k)
                    W(k, iv) = -t(ki, k);

                int jnxt = ki + 1;
                for(int j = ki + 1; j < n; ++j)
                {
                    if(j < jnxt)
                        continue;
                    int j1 = j, j2 = j;
                    jnxt = j + 1;
                    if(j + 1 < n && t(j + 1, j) != T(0))
                    {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }

                    T x[2][2], scale, xnorm;
                    if(j1 == j2)
                    {
                        T dot = T(0);
                        for(int ii = ki + 1; ii < j; ++ii)
                            dot += t(ii, j) * W(ii, iv);
                        W(j, iv) -= dot;

                        trevc_dlaln2_1x1<T, int>(true, 1, smin_val, &t(j, j), ldt,
                                                  wr_val, T(0), W(j, iv), T(0), x, scale, xnorm);
                        if(xnorm > T(1) && W(j, 0) > bignum / xnorm)
                        {
                            x[0][0] /= xnorm;
                            scale /= xnorm;
                        }
                        if(scale != T(1))
                            for(int kk = ki; kk < n; ++kk)
                                W(kk, iv) *= scale;
                        W(j, iv) = x[0][0];
                    }
                    else
                    {
                        T dot1 = T(0), dot2 = T(0);
                        for(int ii = ki + 1; ii < j; ++ii)
                        {
                            dot1 += t(ii, j1) * W(ii, iv);
                            dot2 += t(ii, j2) * W(ii, iv);
                        }
                        W(j1, iv) -= dot1;
                        W(j2, iv) -= dot2;

                        T t22[4] = {t(j1, j1), t(j2, j1), t(j1, j2), t(j2, j2)};
                        const T rhsr[2] = {W(j1, iv), W(j2, iv)};
                        trevc_dlaln2_2x2<T, int>(1, smin_val, t22, 2, wr_val, T(0),
                                                  rhsr, rhsr, x, scale, xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta = std::max(W(j1, 0), W(j2, 0));
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[1][0] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                            for(int kk = ki; kk < n; ++kk)
                                W(kk, iv) *= scale;
                        W(j1, iv) = x[0][0];
                        W(j2, iv) = x[1][0];
                    }
                }

                for(int k = 0; k < ki; ++k)
                    W(k, iv) = T(0);
                iscomplex[iv] = 0;
            }
            else
            {
                // Complex left eigenvector pair
                const int k1 = ki;
                const int k2 = ki + 1;
                const T wr_val = t(k1, k1);
                const T wi_val = std::sqrt(std::abs(t(k1, k2) * t(k2, k1)));
                const T smin_val = std::max(ulp * (std::abs(wr_val) + wi_val), smlnum);

                if(std::abs(t(k1, k2)) >= std::abs(t(k2, k1)))
                {
                    W(k1, iv) = wi_val / safe_den(t(k1, k2), smin_val);
                    W(k2, iv + 1) = T(1);
                }
                else
                {
                    W(k1, iv) = T(1);
                    W(k2, iv + 1) = -wi_val / safe_den(t(k2, k1), smin_val);
                }
                W(k2, iv) = T(0);
                W(k1, iv + 1) = T(0);

                for(int k = k1 + 2; k < n; ++k)
                {
                    W(k, iv) = -W(k1, iv) * t(k1, k);
                    W(k, iv + 1) = -W(k2, iv + 1) * t(k2, k);
                }

                int jnxt = k1 + 2;
                for(int j = k1 + 2; j < n; ++j)
                {
                    if(j < jnxt)
                        continue;
                    int j1 = j, j2 = j;
                    jnxt = j + 1;
                    if(j + 1 < n && t(j + 1, j) != T(0))
                    {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }

                    T x[2][2], scale, xnorm;
                    if(j1 == j2)
                    {
                        T dotr = T(0), doti = T(0);
                        for(int ii = k1 + 2; ii < j; ++ii)
                        {
                            dotr += t(ii, j) * W(ii, iv);
                            doti += t(ii, j) * W(ii, iv + 1);
                        }
                        W(j, iv) -= dotr;
                        W(j, iv + 1) -= doti;

                        trevc_dlaln2_1x1<T, int>(true, 2, smin_val, &t(j, j), ldt,
                                                  wr_val, -wi_val, W(j, iv), W(j, iv + 1),
                                                  x, scale, xnorm);
                        if(xnorm > T(1) && W(j, 0) > bignum / xnorm)
                        {
                            x[0][0] /= xnorm;
                            x[0][1] /= xnorm;
                            scale /= xnorm;
                        }
                        if(scale != T(1))
                            for(int kk = k1; kk < n; ++kk)
                            {
                                W(kk, iv) *= scale;
                                W(kk, iv + 1) *= scale;
                            }
                        W(j, iv) = x[0][0];
                        W(j, iv + 1) = x[0][1];
                    }
                    else
                    {
                        T d1r = T(0), d1i = T(0), d2r = T(0), d2i = T(0);
                        for(int ii = k1 + 2; ii < j; ++ii)
                        {
                            d1r += t(ii, j1) * W(ii, iv);
                            d1i += t(ii, j1) * W(ii, iv + 1);
                            d2r += t(ii, j2) * W(ii, iv);
                            d2i += t(ii, j2) * W(ii, iv + 1);
                        }
                        W(j1, iv) -= d1r;
                        W(j1, iv + 1) -= d1i;
                        W(j2, iv) -= d2r;
                        W(j2, iv + 1) -= d2i;

                        T t22[4] = {t(j1, j1), t(j2, j1), t(j1, j2), t(j2, j2)};
                        const T rhsr[2] = {W(j1, iv), W(j2, iv)};
                        const T rhsi[2] = {W(j1, iv + 1), W(j2, iv + 1)};
                        trevc_dlaln2_2x2<T, int>(2, smin_val, t22, 2, wr_val, -wi_val,
                                                  rhsr, rhsi, x, scale, xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta = std::max(W(j1, 0), W(j2, 0));
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[0][1] *= rec;
                                x[1][0] *= rec;
                                x[1][1] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                            for(int kk = k1; kk < n; ++kk)
                            {
                                W(kk, iv) *= scale;
                                W(kk, iv + 1) *= scale;
                            }
                        W(j1, iv) = x[0][0];
                        W(j2, iv) = x[1][0];
                        W(j1, iv + 1) = x[0][1];
                        W(j2, iv + 1) = x[1][1];
                    }
                }

                for(int k = 0; k < ki; ++k)
                {
                    W(k, iv) = T(0);
                    W(k, iv + 1) = T(0);
                }
                iscomplex[iv] = ip;
                iscomplex[iv + 1] = -ip;
                iv += 1;
            }

            // Blocked back-transform for left eigenvectors
            int ki2 = (ip == 0) ? ki : ki + 1;

            if((iv >= nb - 1) || (ki2 == n - 1))
            {
                int n2 = n - (ki2 + 1) + iv;
                int col_offset = ki2 - iv + 1;

                if(use_gpu)
                {
                    // Ensure previous async dVL update is complete before GEMM reads dVL
                    (void)hipStreamSynchronize(update_stream);

                    (void)hipMemcpy2DAsync(dW_src, static_cast<size_t>(n2) * sizeof(T),
                                           &W(col_offset, 1), n_sz * sizeof(T),
                                           static_cast<size_t>(n2) * sizeof(T), iv,
                                           hipMemcpyHostToDevice, stream);

                    T alpha_val = T(1);
                    T beta_val = T(0);
                    rocblas_gemm_dispatch<T>(handle,
                                             rocblas_operation_none, rocblas_operation_none,
                                             n, iv, n2,
                                             &alpha_val,
                                             dVL + static_cast<size_t>(col_offset) * ldvl, ldvl,
                                             dW_src, n2,
                                             &beta_val, dW_dst, n);

                    (void)hipMemcpy2DAsync(&W(0, nb + 1), n_sz * sizeof(T),
                                           dW_dst, static_cast<size_t>(n) * sizeof(T),
                                           static_cast<size_t>(n) * sizeof(T), iv,
                                           hipMemcpyDeviceToHost, stream);
                    (void)hipStreamSynchronize(stream);
                }
                else
                {
                    blas_gemm("N", "N", n, iv, n2,
                              T(1), &VL[col_offset * ldvl], ldvl,
                              &W(col_offset, 1), n,
                              T(0), &W(0, nb + 1), n);
                }

                T remax = T(1);
                for(int k = 1; k <= iv; ++k)
                {
                    if(iscomplex[k] == 0)
                    {
                        T amax = T(0);
                        for(int i = 0; i < n; ++i)
                            amax = std::max(amax, std::abs(W(i, nb + k)));
                        remax = T(1) / std::max(amax, smlnum);
                    }
                    else if(iscomplex[k] == 1)
                    {
                        T emax = T(0);
                        for(int i = 0; i < n; ++i)
                            emax = std::max(emax, std::abs(W(i, nb + k)) + std::abs(W(i, nb + k + 1)));
                        remax = T(1) / std::max(emax, smlnum);
                    }
                    for(int i = 0; i < n; ++i)
                        W(i, nb + k) *= remax;
                }
                host_lacpy(n, iv, &W(0, nb + 1), n, &VL[col_offset * ldvl], ldvl);

                if(use_gpu)
                {
                    // Launch dVL update on separate stream
                    gpu_update_eigvec_cols(update_stream, dVL, ldvl, VL, ldvl, n, col_offset, iv);
                }

                iv = 1;
            }
            else
            {
                iv += 1;
            }
        }

        if(use_gpu)
        {
            (void)hipStreamSynchronize(update_stream);
            if(owns_gpu_bufs)
            {
                (void)hipStreamDestroy(update_stream);
                if(dW_src) (void)hipFree(dW_src);
                if(dW_dst) (void)hipFree(dW_dst);
            }
        }
    }
} // namespace trevc3_detail

template <typename T>
void run_trevc3_blocked(const char* side, int n, T* T_mat, int ldt,
                        T* VL, int ldvl, T* VR, int ldvr,
                        rocblas_handle handle = nullptr, hipStream_t stream = nullptr,
                        T* dVL = nullptr, T* dVR = nullptr,
                        T* ext_dW_src = nullptr, T* ext_dW_dst = nullptr,
                        hipStream_t ext_update_stream = nullptr)
{
    if(n <= 0)
        return;

    int nb = std::min(TREVC3_NBMAX, std::max(TREVC3_NBMIN, n / 2));

    if(side[0] == 'R' || side[0] == 'r')
        trevc3_detail::blocked_trevc_right(n, T_mat, ldt, VR, ldvr, nb, handle, stream, dVR,
                                            ext_dW_src, ext_dW_dst, ext_update_stream);
    else
        trevc3_detail::blocked_trevc_left(n, T_mat, ldt, VL, ldvl, nb, handle, stream, dVL,
                                           ext_dW_src, ext_dW_dst, ext_update_stream);
}

// TREVC threshold: use blocked BLAS-3 version for N >= threshold,
// unblocked scalar version for small N where blocking overhead dominates.
static constexpr int TREVC3_THRESHOLD = 64;

#ifdef ROCSOLVER_NATIVE_TREVC
template <typename T>
void run_trevc(const char* side, int n, T* T_mat, int ldt,
               T* VL, int ldvl, T* VR, int ldvr,
               rocblas_handle handle = nullptr, hipStream_t stream = nullptr,
               T* dVL = nullptr, T* dVR = nullptr,
               T* ext_dW_src = nullptr, T* ext_dW_dst = nullptr,
               hipStream_t ext_update_stream = nullptr)
{
    (void)handle; (void)stream; (void)dVL; (void)dVR;
    (void)ext_dW_src; (void)ext_dW_dst; (void)ext_update_stream;
    run_trevc_native(side, n, T_mat, ldt, VL, ldvl, VR, ldvr);
}
#else
template <typename T>
void run_trevc(const char* side, int n, T* T_mat, int ldt,
               T* VL, int ldvl, T* VR, int ldvr,
               rocblas_handle handle = nullptr, hipStream_t stream = nullptr,
               T* dVL = nullptr, T* dVR = nullptr,
               T* ext_dW_src = nullptr, T* ext_dW_dst = nullptr,
               hipStream_t ext_update_stream = nullptr)
{
    if(n >= TREVC3_THRESHOLD)
        run_trevc3_blocked(side, n, T_mat, ldt, VL, ldvl, VR, ldvr, handle, stream, dVL, dVR,
                           ext_dW_src, ext_dW_dst, ext_update_stream);
    else
        run_trevc_lapack(side, n, T_mat, ldt, VL, ldvl, VR, ldvr);
}
#endif

// ========== Eigenvector normalization (LAPACK/MAGMA convention) ==========
// Real eigenvalue: scale column to Euclidean norm 1.
// Complex pair (wi[j]>0): scale both columns to combined Euclidean norm 1,
// then apply Givens rotation to make the largest-magnitude component real.
// This matches LAPACK dgeev.f and MAGMA dgeev.cpp exactly.
template <typename T>
void normalize_eigvecs(rocblas_int n, const T* wi, T* V, rocblas_int ldv)
{
    const size_t ldv_sz = static_cast<size_t>(ldv);
    for(rocblas_int j = 0; j < n; ++j)
    {
        if(wi[j] == T(0))
        {
            // Euclidean norm of the real eigenvector
            T nrm = T(0);
            for(rocblas_int i = 0; i < n; ++i)
            {
                T v = V[i + j * ldv_sz];
                nrm += v * v;
            }
            nrm = std::sqrt(nrm);
            if(nrm > T(0))
            {
                T scl = T(1) / nrm;
                for(rocblas_int i = 0; i < n; ++i)
                    V[i + j * ldv_sz] *= scl;
            }
        }
        else if(wi[j] > T(0) && j + 1 < n)
        {
            // Combined Euclidean norm of the complex pair
            T nrm_r = T(0), nrm_i = T(0);
            for(rocblas_int i = 0; i < n; ++i)
            {
                T vr = V[i + j * ldv_sz];
                T vi = V[i + (j + 1) * ldv_sz];
                nrm_r += vr * vr;
                nrm_i += vi * vi;
            }
            nrm_r = std::sqrt(nrm_r);
            nrm_i = std::sqrt(nrm_i);
            T nrm = std::sqrt(nrm_r * nrm_r + nrm_i * nrm_i);  // = dlapy2(nrm_r, nrm_i)
            if(nrm > T(0))
            {
                T scl = T(1) / nrm;
                for(rocblas_int i = 0; i < n; ++i)
                {
                    V[i + j * ldv_sz] *= scl;
                    V[i + (j + 1) * ldv_sz] *= scl;
                }
            }

            // Find row k with largest |v(k,j)|^2 + |v(k,j+1)|^2
            rocblas_int k = 0;
            T mx = T(0);
            for(rocblas_int i = 0; i < n; ++i)
            {
                T vr = V[i + j * ldv_sz];
                T vi = V[i + (j + 1) * ldv_sz];
                T mag2 = vr * vr + vi * vi;
                if(mag2 > mx)
                {
                    mx = mag2;
                    k = i;
                }
            }

            // Givens rotation to zero V(k,j+1) and make V(k,j) real positive:
            // [cs sn] [V(k,j)  ] = [r]
            // [-sn cs] [V(k,j+1)]   [0]
            T a = V[k + j * ldv_sz];
            T b = V[k + (j + 1) * ldv_sz];
            T r = std::sqrt(a * a + b * b);
            if(r > T(0))
            {
                T cs = a / r;
                T sn = b / r;
                for(rocblas_int i = 0; i < n; ++i)
                {
                    T vr = V[i + j * ldv_sz];
                    T vi = V[i + (j + 1) * ldv_sz];
                    V[i + j * ldv_sz]       = cs * vr + sn * vi;
                    V[i + (j + 1) * ldv_sz] = -sn * vr + cs * vi;
                }
            }
            V[k + (j + 1) * ldv_sz] = T(0);

            ++j; // skip the conjugate column
        }
    }
}

// ========== Argument checking ==========

template <typename T, typename I, typename W, typename Wr>
rocblas_status rocsolver_geev_argCheck(rocblas_handle handle,
                                       const rocblas_evect jobvl,
                                       const rocblas_evect jobvr,
                                       const I n,
                                       W A,
                                       const I lda,
                                       Wr wr,
                                       Wr wi,
                                       W VL,
                                       const I ldvl,
                                       W VR,
                                       const I ldvr,
                                       rocblas_int* info,
                                       const I batch_count = 1)
{
    // order is important for unit tests:

    bool wantvl = (jobvl == rocblas_evect_original);
    bool wantvr = (jobvr == rocblas_evect_original);

    // 1. invalid/non-supported values
    if(jobvl != rocblas_evect_none && jobvl != rocblas_evect_original)
        return rocblas_status_invalid_value;
    if(jobvr != rocblas_evect_none && jobvr != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < std::max<I>(1, n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(wantvl && ldvl < std::max<I>(1, n))
        return rocblas_status_invalid_size;
    if(wantvr && ldvr < std::max<I>(1, n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    // for complex T, wi is unused (eigenvalues stored in wr as complex array)
    if((n && !A) || (n && !wr) || (n && !rocblas_is_complex<T> && !wi)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;
    if(wantvl && n && !VL)
        return rocblas_status_invalid_pointer;
    if(wantvr && n && !VR)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_geev_getMemorySize(const I n, const I batch_count, size_t* size_work)
{
    (void)n;
    (void)batch_count;
    *size_work = 0;
}

// Type-dispatched wrapper for rocSOLVER ORGQR/UNGQR on GPU.
template <typename T>
inline rocblas_status rocsolver_orgqr_ungqr(rocblas_handle handle, rocblas_int m,
                                             rocblas_int n, rocblas_int k,
                                             T* A, rocblas_int lda, T* ipiv);

template <>
inline rocblas_status rocsolver_orgqr_ungqr<float>(rocblas_handle h, rocblas_int m,
                                                    rocblas_int n, rocblas_int k,
                                                    float* A, rocblas_int lda, float* ipiv)
{
    return ::rocsolver_sorgqr(h, m, n, k, A, lda, ipiv);
}

template <>
inline rocblas_status rocsolver_orgqr_ungqr<double>(rocblas_handle h, rocblas_int m,
                                                     rocblas_int n, rocblas_int k,
                                                     double* A, rocblas_int lda, double* ipiv)
{
    return ::rocsolver_dorgqr(h, m, n, k, A, lda, ipiv);
}

// GPU ORGHR: generate Q from GEHRD reflectors using rocSOLVER ORGQR on device.
// All work stays on device — no host allocations or host-device transfers.
// dA (n×n, lda): Hessenberg form from GEHRD (reflectors stored below diagonal)
// dTau: (n-1) Householder scalars from GEHRD
// dQ (n×n, ldq): output orthogonal matrix Q
// ilo, ihi: 1-based LAPACK convention from GEBAL
template <typename T>
static void gpu_orghr(rocblas_handle handle, hipStream_t stream,
                       rocblas_int n, rocblas_int ilo, rocblas_int ihi,
                       const T* dA, rocblas_int lda, T* dTau,
                       T* dQ, rocblas_int ldq)
{
    const rocblas_int nh = ihi - ilo;

    // Initialize dQ to identity entirely on device using existing kernel
    {
        constexpr rocblas_int BS = 32;
        dim3 threads(BS, BS, 1);
        dim3 blocks((n + BS - 1) / BS, (n + BS - 1) / BS, 1);
        ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, blocks, threads, 0, stream,
                                n, n, dQ, 0, ldq, 0);
    }

    if(nh <= 1)
    {
        (void)hipStreamSynchronize(stream);
        return;
    }

    // LAPACK orghr shifts reflectors one column right then calls orgqr.
    // GEHRD stores reflectors in A(ilo+1:ihi, ilo:ihi-1) [1-based].
    // In 0-based: A[ilo..ihi-1, ilo-1..ihi-2], i.e. rows ilo..ihi-1, cols (ilo-1)..(ihi-2).
    // After the shift, reflectors land in Q(ilo+1:ihi, ilo+1:ihi) [1-based]
    // = Q[ilo..ihi-1, ilo..ihi-1] [0-based], in the lower triangle.
    // Then orgqr is called on submatrix Q(ilo+1:ihi, ilo+1:ihi) [1-based]
    // = Q[ilo..ihi-1, ilo..ihi-1] [0-based] with tau(ilo) [1-based] = tau[ilo-1] [0-based].
    //
    // We implement the column shift as a 2D device-to-device copy:
    //   src col j: dA[ilo..ihi-1, (ilo-1)+j]  for j=0..nh-2
    //   dst col j: dQ[ilo..ihi-1, ilo+j]       (shifted one column right)
    const rocblas_int ilo0 = ilo - 1; // 0-based ilo
    (void)hipMemcpy2DAsync(
        dQ + ilo + static_cast<size_t>(ilo) * ldq,           // dest: Q[ilo, ilo] (0-based)
        ldq * sizeof(T),                                      // dest pitch
        dA + ilo + static_cast<size_t>(ilo0) * lda,           // src:  A[ilo, ilo-1] (0-based)
        lda * sizeof(T),                                      // src pitch
        nh * sizeof(T),                                       // width: nh rows
        nh - 1,                                               // height: nh-1 columns
        hipMemcpyDeviceToDevice, stream);
    (void)hipStreamSynchronize(stream);

    // Call ORGQR on the nh×nh submatrix at Q[ilo, ilo] (0-based)
    rocblas_set_stream(handle, stream);
    rocsolver_orgqr_ungqr<T>(handle,
                              nh, nh, nh - 1,
                              dQ + ilo + static_cast<size_t>(ilo) * ldq,
                              ldq, dTau + ilo0);
    (void)hipStreamSynchronize(stream);
}

/**
 * GEEV driver following LAPACK dgeev algorithm:
 *   GEBAL -> GEHRD -> ORGHR -> HSEQR -> TREVC -> GEBAK
 *
 * Matches LAPACK API: JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, LDVR, INFO
 *
 * GPU-accelerated GEHRD + ORGHR fusion for large matrices.
 * HSEQR uses multi-shift QR with BLAS-3 accumulation (host).
 * TREVC with GPU GEMM, GEBAL/GEBAK on host.
 */
template <bool BATCHED, bool STRIDED, typename T, typename I, typename U,
          typename UVL = U, typename UVR = U>
rocblas_status rocsolver_geev_template(rocblas_handle handle,
                                       const rocblas_evect jobvl,
                                       const rocblas_evect jobvr,
                                       const I n,
                                       U A,
                                       const rocblas_stride shiftA,
                                       const I lda,
                                       const rocblas_stride strideA,
                                       T* wr,
                                       const rocblas_stride strideWr,
                                       T* wi,
                                       const rocblas_stride strideWi,
                                       UVL VL,
                                       const rocblas_stride shiftVL,
                                       const I ldvl,
                                       const rocblas_stride strideVL,
                                       UVR VR,
                                       const rocblas_stride shiftVR,
                                       const I ldvr,
                                       const rocblas_stride strideVR,
                                       rocblas_int* info,
                                       const I batch_count)
{
    ROCSOLVER_ENTER("geev", "jobvl:", jobvl, "jobvr:", jobvr,
                    "n:", n, "lda:", lda, "ldvl:", ldvl, "ldvr:", ldvr, "bc:", batch_count);

    const bool wantvl = (jobvl == rocblas_evect_original);
    const bool wantvr = (jobvr == rocblas_evect_original);
    const bool wantvec = wantvl || wantvr;

    if(!batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const rocblas_int blocksReset
        = static_cast<rocblas_int>((static_cast<size_t>(batch_count) + BS1 - 1) / BS1);
    ROCSOLVER_LAUNCH_KERNEL(reset_info,
                            dim3(blocksReset, 1, 1),
                            dim3(BS1, 1, 1),
                            0, stream, info, batch_count, 0);
    HIP_CHECK(hipStreamSynchronize(stream));

    if(n == 0)
        return rocblas_status_success;

    const size_t a_elems = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const size_t a_bytes = a_elems * sizeof(T);
    const size_t n_sz = static_cast<size_t>(n);

    const rocblas_stride effStrideWr
        = strideWr ? strideWr : std::max<rocblas_stride>(1, static_cast<rocblas_stride>(n));
    const rocblas_stride effStrideWi
        = strideWi ? strideWi : std::max<rocblas_stride>(1, static_cast<rocblas_stride>(n));

    // For BATCHED mode: fetch array-of-pointers from device
    std::vector<T*> hBatchPtrsA, hBatchPtrsVL, hBatchPtrsVR;
    if constexpr(BATCHED)
    {
        const size_t bc = static_cast<size_t>(batch_count);
        hBatchPtrsA.resize(bc);
        HIP_CHECK(hipMemcpyAsync(hBatchPtrsA.data(), reinterpret_cast<T* const*>(A),
                                 sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        if(wantvl)
        {
            hBatchPtrsVL.resize(bc);
            HIP_CHECK(hipMemcpyAsync(hBatchPtrsVL.data(), reinterpret_cast<T* const*>(VL),
                                     sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        }
        if(wantvr)
        {
            hBatchPtrsVR.resize(bc);
            HIP_CHECK(hipMemcpyAsync(hBatchPtrsVR.data(), reinterpret_cast<T* const*>(VR),
                                     sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Pre-allocate host buffers (reused across batches)
    const size_t tau_elems = n > 1 ? static_cast<size_t>(n - 1) : 0;
    const size_t vec_elems = n_sz * n_sz;
    const size_t vr_elems = static_cast<size_t>(ldvr) * n_sz;
    const size_t vl_elems = static_cast<size_t>(ldvl) * n_sz;

    // Adaptive GPU threshold based on GPU architecture
    const rocblas_int gpu_threshold = get_gehrd_gpu_threshold();

    // Device buffers for GPU GEHRD/ORGHR and TREVC — allocated lazily on first
    // use to avoid hipMalloc overhead when the GPU path is not needed (small N).
    T* dTau = nullptr;
    T* dQ = nullptr;
    T* dVR_trevc = nullptr;
    T* dVL_trevc = nullptr;
    T* dW_src_trevc = nullptr;
    T* dW_dst_trevc = nullptr;
    hipStream_t trevc_update_stream = nullptr;
    const size_t tau_bytes = tau_elems * sizeof(T);
    const bool trevc_gpu_eligible = (n >= TREVC3_GPU_GEMM_THRESHOLD);
    const int trevc_nb = (n > 0)
        ? std::min(TREVC3_NBMAX, std::max(TREVC3_NBMIN, static_cast<int>(n) / 2))
        : 0;
    bool gpu_bufs_allocated = false;
    auto ensure_gpu_bufs = [&]() {
        if(gpu_bufs_allocated) return;
        gpu_bufs_allocated = true;
        if(tau_elems > 0)
            (void)hipMalloc(&dTau, tau_bytes);
        if(wantvec && vec_elems > 0)
            (void)hipMalloc(&dQ, vec_elems * sizeof(T));
        if(trevc_gpu_eligible && wantvr && vr_elems > 0)
            (void)hipMalloc(&dVR_trevc, vr_elems * sizeof(T));
        if(trevc_gpu_eligible && wantvl && vl_elems > 0)
            (void)hipMalloc(&dVL_trevc, vl_elems * sizeof(T));
        if(trevc_gpu_eligible && wantvec && n_sz > 0 && trevc_nb > 0)
        {
            (void)hipMalloc(&dW_src_trevc, n_sz * static_cast<size_t>(trevc_nb) * sizeof(T));
            (void)hipMalloc(&dW_dst_trevc, n_sz * static_cast<size_t>(trevc_nb) * sizeof(T));
            (void)hipStreamCreateWithFlags(&trevc_update_stream, hipStreamNonBlocking);
        }
    };

    // Lambda: process a single batch entry. Used both for serial and OpenMP paths.
    // For the OpenMP path each thread gets its own host buffers.
    // For the GPU path, only one thread uses dTau/handle at a time.
    // Helper: check HIP status inside void-returning lambda (HIP_CHECK returns non-void)
    #define GEEV_HIP_CHECK(call) do { \
        hipError_t _s = (call); \
        if(_s != hipSuccess) { \
            fprintf(stderr, "HIP error %d in geev at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        } \
    } while(0)

    auto process_batch = [&](I b,
                             std::vector<T>& hA_local,
                             std::vector<T>& tau_local,
                             std::vector<T>& scale_local,
                             std::vector<T>& hWr_local,
                             std::vector<T>& hWi_local,
                             std::vector<T>& hSchurVec_local,
                             std::vector<T>& hVR_local,
                             std::vector<T>& hVL_local,
                             hipStream_t local_stream,
                             bool use_gpu_gehrd_allowed) -> void
    {
        T* dA_ptr = nullptr;
        T* dVL_ptr = nullptr;
        T* dVR_ptr = nullptr;

        if constexpr(BATCHED)
        {
            dA_ptr = hBatchPtrsA[static_cast<size_t>(b)] + shiftA;
            if(wantvl) dVL_ptr = hBatchPtrsVL[static_cast<size_t>(b)] + shiftVL;
            if(wantvr) dVR_ptr = hBatchPtrsVR[static_cast<size_t>(b)] + shiftVR;
        }
        else if constexpr(STRIDED)
        {
            dA_ptr = reinterpret_cast<T*>(A) + shiftA + b * strideA;
            if(wantvl) dVL_ptr = reinterpret_cast<T*>(VL) + shiftVL + b * strideVL;
            if(wantvr) dVR_ptr = reinterpret_cast<T*>(VR) + shiftVR + b * strideVR;
        }
        else
        {
            dA_ptr = reinterpret_cast<T*>(A) + shiftA;
            if(wantvl) dVL_ptr = reinterpret_cast<T*>(VL) + shiftVL;
            if(wantvr) dVR_ptr = reinterpret_cast<T*>(VR) + shiftVR;
        }

        // Bind the handle to this batch's stream so all rocBLAS BLAS ops
        // (GEMM/GEMV in GEHRD, ORGHR, HSEQR, TREVC) are ordered correctly
        // with hipMemcpy*Async calls on local_stream.
        if(use_gpu_gehrd_allowed)
            rocblas_set_stream(handle, local_stream);

        GEEV_HIP_CHECK(hipMemcpyAsync(hA_local.data(), dA_ptr, a_bytes,
                                       hipMemcpyDeviceToHost, local_stream));
        GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));

        rocblas_int batch_info = 0;

        // 0. Scale A if max element outside safe range [smlnum, bignum]
        // This matches LAPACK dgeev.f and MAGMA dgeev.cpp prescaling.
        const T eps_mach = std::numeric_limits<T>::epsilon();
        const T safe_min = std::numeric_limits<T>::min();
        T smlnum_scl = std::sqrt(safe_min) / eps_mach;
        T bignum_scl = T(1) / smlnum_scl;
        T anrm = T(0);
        for(I j = 0; j < n; ++j)
            for(I i = 0; i < n; ++i)
                anrm = std::max(anrm, std::abs(hA_local[i + j * static_cast<size_t>(lda)]));
        bool scalea = false;
        T cscale = T(1);
        if(anrm > T(0) && anrm < smlnum_scl)
        {
            scalea = true;
            cscale = smlnum_scl;
        }
        else if(anrm > bignum_scl)
        {
            scalea = true;
            cscale = bignum_scl;
        }
        if(scalea)
        {
            T ratio = cscale / anrm;
            for(I j = 0; j < n; ++j)
                for(I i = 0; i < n; ++i)
                    hA_local[i + j * static_cast<size_t>(lda)] *= ratio;
        }

        // 1. GEBAL
        I ilo_i = 0, ihi_i = 0, gebal_info_i = 0;
        run_gebal(n, hA_local.data(), lda, ilo_i, ihi_i, scale_local.data(), gebal_info_i);
        if(gebal_info_i != 0)
        {
            batch_info = static_cast<rocblas_int>(gebal_info_i);
            GEEV_HIP_CHECK(hipMemcpyAsync(info + b, &batch_info, sizeof(rocblas_int),
                                           hipMemcpyHostToDevice, local_stream));
            return;
        }
        const rocblas_int ilo = static_cast<rocblas_int>(ilo_i);
        const rocblas_int ihi = static_cast<rocblas_int>(ihi_i);
        const rocblas_int nh = ihi - ilo;

        // 2. GEHRD — use GPU path when nh exceeds adaptive threshold
        // 3. ORGHR — fused with GPU GEHRD when possible to avoid host round-trip
        bool used_gpu_gehrd = false;
        bool used_gpu_orghr = false;
        if(use_gpu_gehrd_allowed && nh > gpu_threshold)
        {
            used_gpu_gehrd = true;
            ensure_gpu_bufs();
            GEEV_HIP_CHECK(hipMemcpyAsync(dA_ptr, hA_local.data(), a_bytes,
                                           hipMemcpyHostToDevice, local_stream));
            GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));

            blocked_gehrd_gpu<T>(handle, local_stream, static_cast<rocblas_int>(n),
                                 ilo, ihi, dA_ptr, static_cast<rocblas_int>(lda), dTau);

            // GPU ORGHR fusion: keep data on device, run ORGHR on GPU via ORGQR
            // dA_ptr now has Hessenberg form; run gpu_orghr to produce Q in dQ
            if(wantvec && dQ != nullptr)
            {
                used_gpu_orghr = true;
                const rocblas_int n_ri = static_cast<rocblas_int>(n);

                gpu_orghr<T>(handle, local_stream, n_ri, ilo, ihi,
                             dA_ptr, static_cast<rocblas_int>(lda), dTau,
                             dQ, n_ri);

                // Download Hessenberg A (for HSEQR) and Schur vectors Q
                GEEV_HIP_CHECK(hipMemcpyAsync(hA_local.data(), dA_ptr, a_bytes,
                                               hipMemcpyDeviceToHost, local_stream));
                GEEV_HIP_CHECK(hipMemcpyAsync(hSchurVec_local.data(), dQ,
                                               vec_elems * sizeof(T),
                                               hipMemcpyDeviceToHost, local_stream));
                GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));
            }
            else
            {
                // No eigenvectors requested — just download Hessenberg A
                GEEV_HIP_CHECK(hipMemcpyAsync(hA_local.data(), dA_ptr, a_bytes,
                                               hipMemcpyDeviceToHost, local_stream));
                GEEV_HIP_CHECK(hipMemcpyAsync(tau_local.data(), dTau, tau_bytes,
                                               hipMemcpyDeviceToHost, local_stream));
                GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));
            }
        }
        if(!used_gpu_gehrd)
        {
            rocblas_int gehrd_info = 0;
            run_gehrd_lapack(n, static_cast<I>(ilo), static_cast<I>(ihi), hA_local.data(), lda,
                             tau_local.data(), gehrd_info);
            if(gehrd_info != 0)
            {
                batch_info = gehrd_info;
                GEEV_HIP_CHECK(hipMemcpyAsync(info + b, &batch_info, sizeof(rocblas_int),
                                               hipMemcpyHostToDevice, local_stream));
                return;
            }
        }

        // 3b. ORGHR host fallback (when GPU ORGHR not used)
        if(wantvec && !used_gpu_orghr)
        {
            for(size_t j = 0; j < n_sz; ++j)
                for(size_t i = 0; i < n_sz; ++i)
                    hSchurVec_local[i + j * n_sz] = hA_local[i + j * static_cast<size_t>(lda)];

            rocblas_int orghr_info = 0;
#ifdef ROCSOLVER_NATIVE_ORGHR
            run_orghr(n, ilo, ihi, hSchurVec_local.data(), static_cast<I>(n), tau_local.data(), orghr_info);
#else
            run_orghr_lapack(n, ilo, ihi, hSchurVec_local.data(), static_cast<I>(n), tau_local.data(), orghr_info);
#endif
            if(orghr_info != 0)
            {
                batch_info = orghr_info;
                GEEV_HIP_CHECK(hipMemcpyAsync(info + b, &batch_info, sizeof(rocblas_int),
                                               hipMemcpyHostToDevice, local_stream));
                return;
            }
        }

        // 4. HSEQR
        {
            rocblas_int hseqr_info = 0;
            const rocblas_int n_ri = static_cast<rocblas_int>(n);
            const rocblas_int lda_ri = static_cast<rocblas_int>(lda);
            T* Zptr = wantvec ? hSchurVec_local.data() : nullptr;
            run_hseqr(n_ri, ilo, ihi, true, wantvec, false,
                      hA_local.data(), lda_ri, hWr_local.data(), hWi_local.data(), Zptr, n_ri,
                      hseqr_info,
                      use_gpu_gehrd_allowed ? handle : nullptr, local_stream);
            if(hseqr_info != 0)
            {
                batch_info = hseqr_info;
                // Undo scaling on partial eigenvalues (LAPACK/MAGMA convention)
                if(scalea)
                {
                    T inv_ratio = anrm / cscale;
                    for(rocblas_int k = hseqr_info; k < static_cast<rocblas_int>(n); ++k)
                    {
                        hWr_local[k] *= inv_ratio;
                        hWi_local[k] *= inv_ratio;
                    }
                    for(rocblas_int k = 0; k < ilo - 1; ++k)
                    {
                        hWr_local[k] *= inv_ratio;
                        hWi_local[k] *= inv_ratio;
                    }
                }
                T* dwr = wr + static_cast<rocblas_stride>(b) * effStrideWr;
                T* dwi = wi + static_cast<rocblas_stride>(b) * effStrideWi;
                GEEV_HIP_CHECK(hipMemcpyAsync(dwr, hWr_local.data(), n_sz * sizeof(T),
                                               hipMemcpyHostToDevice, local_stream));
                GEEV_HIP_CHECK(hipMemcpyAsync(dwi, hWi_local.data(), n_sz * sizeof(T),
                                               hipMemcpyHostToDevice, local_stream));
                GEEV_HIP_CHECK(hipMemcpyAsync(info + b, &batch_info, sizeof(rocblas_int),
                                               hipMemcpyHostToDevice, local_stream));
                return;
            }
        }

        // 4b. Undo eigenvalue scaling (all eigenvalues converged)
        if(scalea)
        {
            T inv_ratio = anrm / cscale;
            for(I k = 0; k < n; ++k)
            {
                hWr_local[k] *= inv_ratio;
                hWi_local[k] *= inv_ratio;
            }
        }

        // 5. TREVC + GEBAK + normalize for right eigenvectors
        const bool trevc_gpu = use_gpu_gehrd_allowed && trevc_gpu_eligible;
        if(trevc_gpu && wantvec)
            ensure_gpu_bufs();
        if(wantvr)
        {
            for(size_t j = 0; j < n_sz; ++j)
                for(size_t i = 0; i < n_sz; ++i)
                    hVR_local[i + j * static_cast<size_t>(ldvr)] = hSchurVec_local[i + j * n_sz];

            if(trevc_gpu && dVR_trevc)
            {
                GEEV_HIP_CHECK(hipMemcpyAsync(dVR_trevc, hVR_local.data(), vr_elems * sizeof(T),
                                               hipMemcpyHostToDevice, local_stream));
                GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));
            }

            run_trevc("R", static_cast<int>(n), hA_local.data(), static_cast<int>(lda),
                      static_cast<T*>(nullptr), static_cast<int>(n),
                      hVR_local.data(), static_cast<int>(ldvr),
                      trevc_gpu ? handle : nullptr, local_stream,
                      static_cast<T*>(nullptr), dVR_trevc,
                      dW_src_trevc, dW_dst_trevc, trevc_update_stream);

            rocblas_int gebak_info = 0;
            run_gebak("R", n, ilo, ihi, scale_local.data(), static_cast<I>(n), hVR_local.data(), ldvr, gebak_info);
            normalize_eigvecs(static_cast<rocblas_int>(n), hWi_local.data(), hVR_local.data(), static_cast<rocblas_int>(ldvr));

            GEEV_HIP_CHECK(hipMemcpyAsync(dVR_ptr, hVR_local.data(), vr_elems * sizeof(T),
                                           hipMemcpyHostToDevice, local_stream));
        }

        // 6. TREVC + GEBAK + normalize for left eigenvectors
        if(wantvl)
        {
            for(size_t j = 0; j < n_sz; ++j)
                for(size_t i = 0; i < n_sz; ++i)
                    hVL_local[i + j * static_cast<size_t>(ldvl)] = hSchurVec_local[i + j * n_sz];

            if(trevc_gpu && dVL_trevc)
            {
                GEEV_HIP_CHECK(hipMemcpyAsync(dVL_trevc, hVL_local.data(), vl_elems * sizeof(T),
                                               hipMemcpyHostToDevice, local_stream));
                GEEV_HIP_CHECK(hipStreamSynchronize(local_stream));
            }

            run_trevc("L", static_cast<int>(n), hA_local.data(), static_cast<int>(lda),
                      hVL_local.data(), static_cast<int>(ldvl),
                      static_cast<T*>(nullptr), static_cast<int>(n),
                      trevc_gpu ? handle : nullptr, local_stream,
                      dVL_trevc, static_cast<T*>(nullptr),
                      dW_src_trevc, dW_dst_trevc, trevc_update_stream);

            rocblas_int gebak_info = 0;
            run_gebak("L", n, ilo, ihi, scale_local.data(), static_cast<I>(n), hVL_local.data(), ldvl, gebak_info);
            normalize_eigvecs(static_cast<rocblas_int>(n), hWi_local.data(), hVL_local.data(), static_cast<rocblas_int>(ldvl));

            GEEV_HIP_CHECK(hipMemcpyAsync(dVL_ptr, hVL_local.data(), vl_elems * sizeof(T),
                                           hipMemcpyHostToDevice, local_stream));
        }

        // Upload eigenvalues and info
        T* dwr = wr + static_cast<rocblas_stride>(b) * effStrideWr;
        T* dwi = wi + static_cast<rocblas_stride>(b) * effStrideWi;
        GEEV_HIP_CHECK(hipMemcpyAsync(dwr, hWr_local.data(), n_sz * sizeof(T), hipMemcpyHostToDevice, local_stream));
        GEEV_HIP_CHECK(hipMemcpyAsync(dwi, hWi_local.data(), n_sz * sizeof(T), hipMemcpyHostToDevice, local_stream));
        GEEV_HIP_CHECK(hipMemcpyAsync(info + b, &batch_info, sizeof(rocblas_int),
                                       hipMemcpyHostToDevice, local_stream));
    }; // end process_batch lambda

    #undef GEEV_HIP_CHECK

    // Decide execution strategy based on batch_count and matrix size:
    //   - batch_count > 1 && n <= gpu_threshold: use OpenMP for CPU-parallel batches
    //   - batch_count > 1 && n > gpu_threshold: use multi-stream GPU pipelining
    //   - batch_count == 1: serial (original path)

    const bool use_openmp_batched = (batch_count > 1 && n <= gpu_threshold);

#ifdef _OPENMP
    if(use_openmp_batched)
    {
        // OpenMP parallel batch processing for small matrices (CPU-bound pipeline)
        const int max_threads = std::min(static_cast<int>(batch_count), omp_get_max_threads());

        #pragma omp parallel num_threads(max_threads)
        {
            // Per-thread host buffers
            std::vector<T> hA_thr(a_elems);
            std::vector<T> tau_thr(tau_elems);
            std::vector<T> scale_thr(n_sz);
            std::vector<T> hWr_thr(n_sz), hWi_thr(n_sz);
            std::vector<T> hSchurVec_thr(wantvec ? vec_elems : 0);
            std::vector<T> hVR_thr(wantvr ? vr_elems : 0);
            std::vector<T> hVL_thr(wantvl ? vl_elems : 0);

            // Per-thread HIP stream for async memory transfers
            hipStream_t thr_stream;
            (void)hipStreamCreateWithFlags(&thr_stream, hipStreamNonBlocking);

            #pragma omp for schedule(dynamic)
            for(I b = 0; b < batch_count; ++b)
            {
                process_batch(b, hA_thr, tau_thr, scale_thr, hWr_thr, hWi_thr,
                              hSchurVec_thr, hVR_thr, hVL_thr,
                              thr_stream, false /* no GPU GEHRD in OpenMP path */);
            }

            (void)hipStreamSynchronize(thr_stream);
            (void)hipStreamDestroy(thr_stream);
        }
    }
    else
#else
    (void)use_openmp_batched;
#endif
    if(batch_count > 1 && n > gpu_threshold)
    {
        // Multi-stream GPU pipelining: overlap data transfers with GPU GEHRD compute.
        // Use 2 streams to pipeline: while one batch does GPU GEHRD, the next
        // does host download / GEBAL.
        constexpr int NUM_PIPE_STREAMS = 2;
        hipStream_t pipe_streams[NUM_PIPE_STREAMS];
        for(int s = 0; s < NUM_PIPE_STREAMS; ++s)
            HIP_CHECK(hipStreamCreateWithFlags(&pipe_streams[s], hipStreamNonBlocking));

        // Per-stream host buffers
        std::vector<std::vector<T>> hA_pipe(NUM_PIPE_STREAMS, std::vector<T>(a_elems));
        std::vector<std::vector<T>> tau_pipe(NUM_PIPE_STREAMS, std::vector<T>(tau_elems));
        std::vector<std::vector<T>> scale_pipe(NUM_PIPE_STREAMS, std::vector<T>(n_sz));
        std::vector<std::vector<T>> hWr_pipe(NUM_PIPE_STREAMS, std::vector<T>(n_sz));
        std::vector<std::vector<T>> hWi_pipe(NUM_PIPE_STREAMS, std::vector<T>(n_sz));
        std::vector<std::vector<T>> hSchurVec_pipe(NUM_PIPE_STREAMS, std::vector<T>(wantvec ? vec_elems : 0));
        std::vector<std::vector<T>> hVR_pipe(NUM_PIPE_STREAMS, std::vector<T>(wantvr ? vr_elems : 0));
        std::vector<std::vector<T>> hVL_pipe(NUM_PIPE_STREAMS, std::vector<T>(wantvl ? vl_elems : 0));

        for(I b = 0; b < batch_count; ++b)
        {
            int si = static_cast<int>(b) % NUM_PIPE_STREAMS;
            // Wait for the previous batch on this stream to finish
            HIP_CHECK(hipStreamSynchronize(pipe_streams[si]));

            process_batch(b, hA_pipe[si], tau_pipe[si], scale_pipe[si],
                          hWr_pipe[si], hWi_pipe[si],
                          hSchurVec_pipe[si], hVR_pipe[si], hVL_pipe[si],
                          pipe_streams[si], true /* allow GPU GEHRD */);
        }

        for(int s = 0; s < NUM_PIPE_STREAMS; ++s)
        {
            HIP_CHECK(hipStreamSynchronize(pipe_streams[s]));
            HIP_CHECK(hipStreamDestroy(pipe_streams[s]));
        }
    }
    else
    {
        // Serial path: single batch or fallback
        std::vector<T> hA_ser(a_elems);
        std::vector<T> tau_ser(tau_elems);
        std::vector<T> scale_ser(n_sz);
        std::vector<T> hWr_ser(n_sz), hWi_ser(n_sz);
        std::vector<T> hSchurVec_ser(wantvec ? vec_elems : 0);
        std::vector<T> hVR_ser(wantvr ? vr_elems : 0);
        std::vector<T> hVL_ser(wantvl ? vl_elems : 0);

        for(I b = 0; b < batch_count; ++b)
        {
            process_batch(b, hA_ser, tau_ser, scale_ser, hWr_ser, hWi_ser,
                          hSchurVec_ser, hVR_ser, hVL_ser,
                          stream, true /* allow GPU GEHRD */);
        }
    }

    if(dTau)
        (void)hipFree(dTau);
    if(dQ)
        (void)hipFree(dQ);
    if(dVR_trevc)
        (void)hipFree(dVR_trevc);
    if(dVL_trevc)
        (void)hipFree(dVL_trevc);
    if(dW_src_trevc)
        (void)hipFree(dW_src_trevc);
    if(dW_dst_trevc)
        (void)hipFree(dW_dst_trevc);
    if(trevc_update_stream)
        (void)hipStreamDestroy(trevc_update_stream);

    HIP_CHECK(hipStreamSynchronize(stream));
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
