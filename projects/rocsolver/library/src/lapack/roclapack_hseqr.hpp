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

#ifndef HSEQR_STANDALONE_BENCH
#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HSEQR_STANDALONE_BENCH
extern "C"
{
    void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
                const double*, const int*, const double*, const int*, const double*, double*,
                const int*);
    void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*,
                const float*, const int*, const float*, const int*, const float*, float*,
                const int*);
    void dgehrd_(const int*, const int*, const int*, double*, const int*, double*, double*,
                 const int*, int*);
    void sgehrd_(const int*, const int*, const int*, float*, const int*, float*, float*,
                 const int*, int*);
    void dormhr_(const char*, const char*, const int*, const int*, const int*, const int*,
                 const double*, const int*, const double*, double*, const int*, double*,
                 const int*, int*);
    void sormhr_(const char*, const char*, const int*, const int*, const int*, const int*,
                 const float*, const int*, const float*, float*, const int*, float*,
                 const int*, int*);
    void dlarfg_(const int*, double*, double*, const int*, double*);
    void slarfg_(const int*, float*, float*, const int*, float*);
    void dlarf_(const char*, const int*, const int*, const double*, const int*, const double*,
                double*, const int*, double*);
    void slarf_(const char*, const int*, const int*, const float*, const int*, const float*,
                float*, const int*, float*);
    void dlacpy_(const char*, const int*, const int*, const double*, const int*, double*,
                 const int*);
    void slacpy_(const char*, const int*, const int*, const float*, const int*, float*,
                 const int*);
    void dlaset_(const char*, const int*, const int*, const double*, const double*, double*,
                 const int*);
    void slaset_(const char*, const int*, const int*, const float*, const float*, float*,
                 const int*);
    void dtrexc_(const char*, const int*, double*, const int*, double*, const int*, int*,
                 int*, double*, int*);
    void strexc_(const char*, const int*, float*, const int*, float*, const int*, int*,
                 int*, float*, int*);
    int iparmq_(const int*, const int*, const int*, const int*);

    void dgebal_(const char*, const int*, double*, const int*, int*, int*, double*, int*);
    void sgebal_(const char*, const int*, float*, const int*, int*, int*, float*, int*);
    void dgebak_(const char*, const char*, const int*, const int*, const int*,
                 const double*, const int*, double*, const int*, int*);
    void sgebak_(const char*, const char*, const int*, const int*, const int*,
                 const float*, const int*, float*, const int*, int*);
    void dorghr_(const int*, const int*, const int*, double*, const int*,
                 const double*, double*, const int*, int*);
    void sorghr_(const int*, const int*, const int*, float*, const int*,
                 const float*, float*, const int*, int*);
    void dtrevc_(const char*, const char*, const int*, const int*,
                 const double*, const int*, double*, const int*,
                 double*, const int*, const int*, int*, double*, int*);
    void strevc_(const char*, const char*, const int*, const int*,
                 const float*, const int*, float*, const int*,
                 float*, const int*, const int*, int*, float*, int*);
    // Blocked BLAS-3 TREVC (LAPACK 3.7+)
    void dtrevc3_(const char*, const char*, const int*, const int*,
                  const double*, const int*, double*, const int*,
                  double*, const int*, const int*, int*, double*, const int*, int*);
    void strevc3_(const char*, const char*, const int*, const int*,
                  const float*, const int*, float*, const int*,
                  float*, const int*, const int*, int*, float*, const int*, int*);
    // Complex LAPACK routines for native complex GEEV pipeline
    void cgebal_(const char*, const int*, void*, const int*, int*, int*, float*, int*);
    void zgebal_(const char*, const int*, void*, const int*, int*, int*, double*, int*);
    void cgehrd_(const int*, const int*, const int*, void*, const int*,
                 void*, void*, const int*, int*);
    void zgehrd_(const int*, const int*, const int*, void*, const int*,
                 void*, void*, const int*, int*);
    void cunghr_(const int*, const int*, const int*, void*, const int*,
                 const void*, void*, const int*, int*);
    void zunghr_(const int*, const int*, const int*, void*, const int*,
                 const void*, void*, const int*, int*);
    void chseqr_(const char*, const char*, const int*, const int*, const int*,
                 void*, const int*, void*, void*, const int*, void*, const int*, int*);
    void zhseqr_(const char*, const char*, const int*, const int*, const int*,
                 void*, const int*, void*, void*, const int*, void*, const int*, int*);
    void ctrevc3_(const char*, const char*, const int*, const int*,
                  void*, const int*, void*, const int*, void*, const int*,
                  const int*, int*, void*, const int*, float*, int*);
    void ztrevc3_(const char*, const char*, const int*, const int*,
                  void*, const int*, void*, const int*, void*, const int*,
                  const int*, int*, void*, const int*, double*, int*);
    void ctrevc_(const char*, const char*, const int*, const int*,
                 void*, const int*, void*, const int*, void*, const int*,
                 const int*, int*, void*, float*, int*);
    void ztrevc_(const char*, const char*, const int*, const int*,
                 void*, const int*, void*, const int*, void*, const int*,
                 const int*, int*, void*, double*, int*);
    void cgebak_(const char*, const char*, const int*, const int*, const int*,
                 const float*, const int*, void*, const int*, int*);
    void zgebak_(const char*, const char*, const int*, const int*, const int*,
                 const double*, const int*, void*, const int*, int*);
    void dhseqr_(const char*, const char*, const int*, const int*, const int*,
                 double*, const int*, double*, double*, double*, const int*,
                 double*, const int*, int*);
    void shseqr_(const char*, const char*, const int*, const int*, const int*,
                 float*, const int*, float*, float*, float*, const int*,
                 float*, const int*, int*);
}
#endif

ROCSOLVER_BEGIN_NAMESPACE

// Threshold for GPU-accelerated slab GEMMs in HSEQR.
// Below this, the GEMM dimensions are too small for GPU to outperform host BLAS.
static constexpr int HSEQR_GPU_GEMM_THRESHOLD = 512;

// GPU context for HSEQR slab GEMMs. Holds device mirrors of H and Z
// plus scratch buffers for U, WH, WV. The bulge-chasing and AED window
// work stays on the host; only the BLAS-3 far-from-diagonal slab updates
// are offloaded to the GPU.
template <typename T>
struct hseqr_gpu_ctx
{
    rocblas_handle handle = nullptr;
    hipStream_t stream = nullptr;
    T* dH = nullptr;
    T* dZ = nullptr;
    T* dU = nullptr;
    T* dWH = nullptr;
    T* dWV = nullptr;
    int n = 0;
    int ldh = 0;
    int ldz = 0;
    bool active = false;

    void init(rocblas_handle h, hipStream_t s, int n_, int ldh_, int ldz_,
              T* hostH, T* hostZ, bool wantz)
    {
        handle = h;
        stream = s;
        n = n_;
        ldh = ldh_;
        ldz = ldz_;

        (void)hipMalloc(&dH, static_cast<size_t>(ldh) * n * sizeof(T));
        if(wantz && hostZ)
            (void)hipMalloc(&dZ, static_cast<size_t>(ldz) * n * sizeof(T));
        // U, WH, WV allocated per-sweep call since dimensions vary
        dU = nullptr;
        dWH = nullptr;
        dWV = nullptr;

        // Upload full H, Z to device
        for(int j = 0; j < n; ++j)
            (void)hipMemcpyAsync(dH + static_cast<size_t>(j) * ldh,
                                 hostH + static_cast<size_t>(j) * ldh,
                                 static_cast<size_t>(n) * sizeof(T),
                                 hipMemcpyHostToDevice, stream);
        if(dZ && hostZ)
            for(int j = 0; j < n; ++j)
                (void)hipMemcpyAsync(dZ + static_cast<size_t>(j) * ldz,
                                     hostZ + static_cast<size_t>(j) * ldz,
                                     static_cast<size_t>(n) * sizeof(T),
                                     hipMemcpyHostToDevice, stream);
        (void)hipStreamSynchronize(stream);
        active = true;
    }

    void sync_h_to_device(const T* hostH, int row_start, int row_end, int col_start, int col_end)
    {
        for(int j = col_start; j <= col_end; ++j)
            (void)hipMemcpyAsync(dH + static_cast<size_t>(j) * ldh + row_start,
                                 hostH + static_cast<size_t>(j) * ldh + row_start,
                                 static_cast<size_t>(row_end - row_start + 1) * sizeof(T),
                                 hipMemcpyHostToDevice, stream);
    }

    void sync_h_from_device(T* hostH, int row_start, int row_end, int col_start, int col_end)
    {
        for(int j = col_start; j <= col_end; ++j)
            (void)hipMemcpyAsync(hostH + static_cast<size_t>(j) * ldh + row_start,
                                 dH + static_cast<size_t>(j) * ldh + row_start,
                                 static_cast<size_t>(row_end - row_start + 1) * sizeof(T),
                                 hipMemcpyDeviceToHost, stream);
    }

    void sync_z_to_device(const T* hostZ, int row_start, int row_end, int col_start, int col_end)
    {
        if(!dZ) return;
        for(int j = col_start; j <= col_end; ++j)
            (void)hipMemcpyAsync(dZ + static_cast<size_t>(j) * ldz + row_start,
                                 hostZ + static_cast<size_t>(j) * ldz + row_start,
                                 static_cast<size_t>(row_end - row_start + 1) * sizeof(T),
                                 hipMemcpyHostToDevice, stream);
    }

    void sync_z_from_device(T* hostZ, int row_start, int row_end, int col_start, int col_end)
    {
        if(!dZ) return;
        for(int j = col_start; j <= col_end; ++j)
            (void)hipMemcpyAsync(hostZ + static_cast<size_t>(j) * ldz + row_start,
                                 dZ + static_cast<size_t>(j) * ldz + row_start,
                                 static_cast<size_t>(row_end - row_start + 1) * sizeof(T),
                                 hipMemcpyDeviceToHost, stream);
    }

    void sync() { if(active) (void)hipStreamSynchronize(stream); }

    void download_all(T* hostH, T* hostZ, bool wantz)
    {
        for(int j = 0; j < n; ++j)
            (void)hipMemcpyAsync(hostH + static_cast<size_t>(j) * ldh,
                                 dH + static_cast<size_t>(j) * ldh,
                                 static_cast<size_t>(n) * sizeof(T),
                                 hipMemcpyDeviceToHost, stream);
        if(dZ && hostZ && wantz)
            for(int j = 0; j < n; ++j)
                (void)hipMemcpyAsync(hostZ + static_cast<size_t>(j) * ldz,
                                     dZ + static_cast<size_t>(j) * ldz,
                                     static_cast<size_t>(n) * sizeof(T),
                                     hipMemcpyDeviceToHost, stream);
        (void)hipStreamSynchronize(stream);
    }

    void destroy()
    {
        if(dH) { (void)hipFree(dH); dH = nullptr; }
        if(dZ) { (void)hipFree(dZ); dZ = nullptr; }
        if(dU) { (void)hipFree(dU); dU = nullptr; }
        if(dWH) { (void)hipFree(dWH); dWH = nullptr; }
        if(dWV) { (void)hipFree(dWV); dWV = nullptr; }
        active = false;
    }
};

namespace hseqr_host_detail
{

// ===== LAPACK/BLAS host wrappers (type-dispatched) =====

#ifndef HSEQR_STANDALONE_BENCH

inline void host_gemm(const char* ta, const char* tb,
                       int m, int n, int k,
                       double alpha, const double* A, int lda,
                       const double* B, int ldb,
                       double beta, double* C, int ldc)
{
    dgemm_(ta, tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
inline void host_gemm(const char* ta, const char* tb,
                       int m, int n, int k,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc)
{
    sgemm_(ta, tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void host_gehrd(int n, int ilo, int ihi, double* A, int lda,
                        double* tau, double* work, int lwork, int& info)
{
    dgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_gehrd(int n, int ilo, int ihi, float* A, int lda,
                        float* tau, float* work, int lwork, int& info)
{
    sgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

inline void host_ormhr(const char* side, const char* trans,
                        int m, int n, int ilo, int ihi,
                        const double* A, int lda, const double* tau,
                        double* C, int ldc, double* work, int lwork, int& info)
{
    dormhr_(side, trans, &m, &n, &ilo, &ihi, A, &lda, tau, C, &ldc, work, &lwork, &info);
}
inline void host_ormhr(const char* side, const char* trans,
                        int m, int n, int ilo, int ihi,
                        const float* A, int lda, const float* tau,
                        float* C, int ldc, float* work, int lwork, int& info)
{
    sormhr_(side, trans, &m, &n, &ilo, &ihi, A, &lda, tau, C, &ldc, work, &lwork, &info);
}

inline void host_larfg(int n, double& alpha, double* x, int incx, double& tau)
{
    dlarfg_(&n, &alpha, x, &incx, &tau);
}
inline void host_larfg(int n, float& alpha, float* x, int incx, float& tau)
{
    slarfg_(&n, &alpha, x, &incx, &tau);
}

inline void host_larf(const char* side, int m, int n, const double* v, int incv,
                       double tau, double* C, int ldc, double* work)
{
    dlarf_(side, &m, &n, v, &incv, &tau, C, &ldc, work);
}
inline void host_larf(const char* side, int m, int n, const float* v, int incv,
                       float tau, float* C, int ldc, float* work)
{
    slarf_(side, &m, &n, v, &incv, &tau, C, &ldc, work);
}

inline void host_lacpy(const char* uplo, int m, int n,
                        const double* A, int lda, double* B, int ldb)
{
    dlacpy_(uplo, &m, &n, A, &lda, B, &ldb);
}
inline void host_lacpy(const char* uplo, int m, int n,
                        const float* A, int lda, float* B, int ldb)
{
    slacpy_(uplo, &m, &n, A, &lda, B, &ldb);
}

inline void host_laset(const char* uplo, int m, int n,
                        double alpha, double beta, double* A, int lda)
{
    dlaset_(uplo, &m, &n, &alpha, &beta, A, &lda);
}
inline void host_laset(const char* uplo, int m, int n,
                        float alpha, float beta, float* A, int lda)
{
    slaset_(uplo, &m, &n, &alpha, &beta, A, &lda);
}

inline void host_trexc(const char* compq, int n, double* T, int ldt,
                        double* Q, int ldq, int& ifst, int& ilst, double* work, int& info)
{
    dtrexc_(compq, &n, T, &ldt, Q, &ldq, &ifst, &ilst, work, &info);
}
inline void host_trexc(const char* compq, int n, float* T, int ldt,
                        float* Q, int ldq, int& ifst, int& ilst, float* work, int& info)
{
    strexc_(compq, &n, T, &ldt, Q, &ldq, &ifst, &ilst, work, &info);
}

inline int host_iparmq(int ispec, int name, int ilo, int ihi)
{
    return iparmq_(&ispec, &name, &ilo, &ihi);
}

inline void host_gebal(const char* job, int n, double* A, int lda,
                        int& ilo, int& ihi, double* scale, int& info)
{
    dgebal_(job, &n, A, &lda, &ilo, &ihi, scale, &info);
}
inline void host_gebal(const char* job, int n, float* A, int lda,
                        int& ilo, int& ihi, float* scale, int& info)
{
    sgebal_(job, &n, A, &lda, &ilo, &ihi, scale, &info);
}

inline void host_gebak(const char* job, const char* side, int n, int ilo, int ihi,
                        const double* scale, int m, double* V, int ldv, int& info)
{
    dgebak_(job, side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}
inline void host_gebak(const char* job, const char* side, int n, int ilo, int ihi,
                        const float* scale, int m, float* V, int ldv, int& info)
{
    sgebak_(job, side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}

inline void host_orghr(int n, int ilo, int ihi, double* A, int lda,
                        const double* tau, double* work, int lwork, int& info)
{
    dorghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_orghr(int n, int ilo, int ihi, float* A, int lda,
                        const float* tau, float* work, int lwork, int& info)
{
    sorghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

inline void host_hseqr(const char* job, const char* compz,
                        int n, int ilo, int ihi,
                        double* H, int ldh, double* wr, double* wi,
                        double* Z, int ldz, double* work, int lwork, int& info)
{
    dhseqr_(job, compz, &n, &ilo, &ihi, H, &ldh, wr, wi, Z, &ldz, work, &lwork, &info);
}
inline void host_hseqr(const char* job, const char* compz,
                        int n, int ilo, int ihi,
                        float* H, int ldh, float* wr, float* wi,
                        float* Z, int ldz, float* work, int lwork, int& info)
{
    shseqr_(job, compz, &n, &ilo, &ihi, H, &ldh, wr, wi, Z, &ldz, work, &lwork, &info);
}

#endif // HSEQR_STANDALONE_BENCH
/** Column-major access with 1-based (i,j) like Fortran. */
template <typename T>
inline T& hf(T* H, rocblas_int ldh, rocblas_int i1, rocblas_int j1)
{
    return H[(i1 - 1) + (j1 - 1) * ldh];
}

template <typename T>
inline T dlapy2(T x, T y)
{
    x = std::abs(x);
    y = std::abs(y);
    if(x < y)
        std::swap(x, y);
    if(y == T(0))
        return x;
    return x * std::sqrt(T(1) + (y / x) * (y / x));
}

/** LAPACK DLANV2 / SLANV2 — Schur form of a real 2-by-2. */
template <typename T>
void dlanv2(T& a,
            T& b,
            T& c,
            T& d,
            T& rt1r,
            T& rt1i,
            T& rt2r,
            T& rt2i,
            T& cs,
            T& sn)
{
    const T zero = 0;
    const T half = T(0.5);
    const T one = 1;
    const T two = 2;
    const T multpl = 4;

    T aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab, sac, scale, sigma, sn1, tau, temp, z,
        safmin, safmn2, safmx2;
    int count;

    safmin = std::numeric_limits<T>::min();
    eps = std::numeric_limits<T>::epsilon();
    const T flt_base = T(std::numeric_limits<T>::radix);
    safmn2 = std::pow(flt_base, int(std::log(safmin / eps) / std::log(flt_base) / two));
    safmx2 = one / safmn2;

    if(c == zero)
    {
        cs = one;
        sn = zero;
    }
    else if(b == zero)
    {
        cs = zero;
        sn = one;
        temp = d;
        d = a;
        a = temp;
        b = -c;
        c = zero;
    }
    else if((a - d) == zero && std::copysign(one, b) != std::copysign(one, c))
    {
        cs = one;
        sn = zero;
    }
    else
    {
        temp = a - d;
        p = half * temp;
        bcmax = std::max(std::abs(b), std::abs(c));
        bcmis = std::min(std::abs(b), std::abs(c)) * std::copysign(one, b) * std::copysign(one, c);
        scale = std::max(std::abs(p), bcmax);
        z = (p / scale) * p + (bcmax / scale) * bcmis;

        if(z >= multpl * eps)
        {
            z = p + std::copysign(std::sqrt(scale) * std::sqrt(z), p);
            a = d + z;
            d = d - (bcmax / z) * bcmis;
            tau = dlapy2(c, z);
            cs = z / tau;
            sn = c / tau;
            b = b - c;
            c = zero;
        }
        else
        {
            count = 0;
            sigma = b + c;
        L10:
            count = count + 1;
            scale = std::max(std::abs(temp), std::abs(sigma));
            if(scale >= safmx2)
            {
                sigma *= safmn2;
                temp *= safmn2;
                if(count <= 20)
                    goto L10;
            }
            if(scale <= safmn2)
            {
                sigma *= safmx2;
                temp *= safmx2;
                if(count <= 20)
                    goto L10;
            }
            p = half * temp;
            tau = dlapy2(sigma, temp);
            cs = std::sqrt(half * (one + std::abs(sigma) / tau));
            sn = -(p / (tau * cs)) * std::copysign(one, sigma);

            aa = a * cs + b * sn;
            bb = -a * sn + b * cs;
            cc = c * cs + d * sn;
            dd = -c * sn + d * cs;

            a = aa * cs + cc * sn;
            b = (bb * cs) + (dd * sn);
            c = -(aa * sn) + (cc * cs);
            d = -bb * sn + dd * cs;

            temp = half * (a + d);
            a = temp;
            d = temp;

            if(c != zero)
            {
                if(b != zero)
                {
                    if(std::copysign(one, b) == std::copysign(one, c))
                    {
                        sab = std::sqrt(std::abs(b));
                        sac = std::sqrt(std::abs(c));
                        p = std::copysign(sab * sac, c);
                        tau = one / std::sqrt(std::abs(b + c));
                        a = temp + p;
                        d = temp - p;
                        b = b - c;
                        c = zero;
                        cs1 = sab * tau;
                        sn1 = sac * tau;
                        temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                }
                else
                {
                    b = -c;
                    c = zero;
                    temp = cs;
                    cs = -sn;
                    sn = temp;
                }
            }
        }
    }

    rt1r = a;
    rt2r = d;
    if(c == zero)
    {
        rt1i = zero;
        rt2i = zero;
    }
    else
    {
        rt1i = std::sqrt(std::abs(b)) * std::sqrt(std::abs(c));
        rt2i = -rt1i;
    }
}

/** LAPACK DLARFG — elementary reflector (n <= 3). */
template <typename T>
void dlarfg_small(rocblas_int n, T& alpha, T* x, rocblas_int incx, T& tau)
{
    const T zero = 0;
    const T one = 1;
    if(n <= 1)
    {
        tau = zero;
        return;
    }
    T xnorm = zero;
    for(rocblas_int i = 0; i < n - 1; i++)
        xnorm += (x[i * incx]) * (x[i * incx]);
    xnorm = std::sqrt(xnorm);
    if(xnorm == zero)
    {
        tau = zero;
        return;
    }
    T beta = -std::copysign(dlapy2(alpha, xnorm), alpha);
    tau = (beta - alpha) / beta;
    T scl = one / (alpha - beta);
    alpha = beta;
    for(rocblas_int i = 0; i < n - 1; i++)
        x[i * incx] *= scl;
}

/** LAPACK DROT — Givens rotation on two vectors. */
template <typename T>
void drot(rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy, T c, T s)
{
    for(rocblas_int k = 0; k < n; k++)
    {
        T xk = x[k * incx];
        T yk = y[k * incy];
        x[k * incx] = c * xk + s * yk;
        y[k * incy] = -s * xk + c * yk;
    }
}

// Forward declaration — dlahqr is defined later but used by dlaqr3_aed and dlaqr0.
template <typename T>
rocblas_int dlahqr(bool wantt, bool wantz, rocblas_int n,
                   rocblas_int ilo, rocblas_int ihi,
                   T* H, rocblas_int ldh, T* wr, T* wi,
                   rocblas_int iloz, rocblas_int ihiz, T* Z, rocblas_int ldz);

/**
 * DLARF1F — apply elementary reflector H = I - tau*v*v^T assuming v(1) = 1.
 * Native C++ implementation of LAPACK 3.12 DLARF1F using BLAS-2.
 * SIDE='L': C := H*C,  SIDE='R': C := C*H.
 * v(1) is assumed 1 and NOT stored/referenced; v(2:) holds rest.
 */
template <typename T>
void dlarf1f(char side, int m, int n, const T* v, int incv, T tau,
             T* C, int ldc, T* work)
{
    const T zero = 0;
    const T one = 1;
    if(tau == zero || m == 0 || n == 0)
        return;

    bool left = (side == 'L' || side == 'l');

    if(left)
    {
        if(m == 1)
        {
            T scl = one - tau;
            for(int j = 0; j < n; j++)
                C[j * ldc] *= scl;
        }
        else
        {
            // w = C(2:m,:)^T * v(2:m) + C(1,:)^T
            for(int j = 0; j < n; j++)
                work[j] = zero;
            for(int j = 0; j < n; j++)
            {
                T s = C[0 + j * ldc]; // C(1,j) * v(1) = C(1,j)
                for(int i = 1; i < m; i++)
                    s += C[i + j * ldc] * v[1 + (i - 1) * incv];
                work[j] = s;
            }
            // C(1,:) -= tau * w^T
            for(int j = 0; j < n; j++)
                C[0 + j * ldc] -= tau * work[j];
            // C(2:m,:) -= tau * v(2:m) * w^T
            for(int j = 0; j < n; j++)
            {
                T tw = tau * work[j];
                for(int i = 1; i < m; i++)
                    C[i + j * ldc] -= v[1 + (i - 1) * incv] * tw;
            }
        }
    }
    else
    {
        if(n == 1)
        {
            T scl = one - tau;
            for(int i = 0; i < m; i++)
                C[i] *= scl;
        }
        else
        {
            // w = C(:,2:n) * v(2:n) + C(:,1)
            for(int i = 0; i < m; i++)
                work[i] = C[i + 0 * ldc]; // C(:,1) * v(1) = C(:,1)
            for(int j = 1; j < n; j++)
            {
                T vj = v[1 + (j - 1) * incv];
                for(int i = 0; i < m; i++)
                    work[i] += C[i + j * ldc] * vj;
            }
            // C(:,1) -= tau * w
            for(int i = 0; i < m; i++)
                C[i + 0 * ldc] -= tau * work[i];
            // C(:,2:n) -= tau * w * v(2:n)^T
            for(int j = 1; j < n; j++)
            {
                T tv = tau * v[1 + (j - 1) * incv];
                for(int i = 0; i < m; i++)
                    C[i + j * ldc] -= work[i] * tv;
            }
        }
    }
}

/**
 * DLAQR1 — set v to a scalar multiple of the first column of
 * (H - (sr1+i*si1)*I)*(H - (sr2+i*si2)*I). N must be 2 or 3.
 * All indices are 0-based; H is column-major with leading dim ldh.
 */
template <typename T>
void dlaqr1(int n, const T* H, int ldh, T sr1, T si1, T sr2, T si2, T* v)
{
    const T zero = 0;
    if(n != 2 && n != 3)
        return;

    auto HH = [&](int i, int j) -> T { return H[i + j * ldh]; };

    if(n == 2)
    {
        T s = std::abs(HH(0, 0) - sr2) + std::abs(si2) + std::abs(HH(1, 0));
        if(s == zero)
        {
            v[0] = zero;
            v[1] = zero;
        }
        else
        {
            T h21s = HH(1, 0) / s;
            v[0] = h21s * HH(0, 1) + (HH(0, 0) - sr1) * ((HH(0, 0) - sr2) / s)
                   - si1 * (si2 / s);
            v[1] = h21s * (HH(0, 0) + HH(1, 1) - sr1 - sr2);
        }
    }
    else
    {
        T s = std::abs(HH(0, 0) - sr2) + std::abs(si2)
              + std::abs(HH(1, 0)) + std::abs(HH(2, 0));
        if(s == zero)
        {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        }
        else
        {
            T h21s = HH(1, 0) / s;
            T h31s = HH(2, 0) / s;
            v[0] = (HH(0, 0) - sr1) * ((HH(0, 0) - sr2) / s)
                   - si1 * (si2 / s) + HH(0, 1) * h21s + HH(0, 2) * h31s;
            v[1] = h21s * (HH(0, 0) + HH(1, 1) - sr1 - sr2) + HH(1, 2) * h31s;
            v[2] = h31s * (HH(0, 0) + HH(2, 2) - sr1 - sr2) + h21s * HH(2, 1);
        }
    }
}

/**
 * DLAQR5 — single small-bulge multi-shift QR sweep with BLAS-3 accumulation.
 * Faithful C++ translation of LAPACK 3.12 dlaqr5.f.
 * All indices are 1-based (Fortran convention) via the hf() accessor.
 * KACC22=1 enables BLAS-3 far-from-diagonal updates.
 */
template <typename T>
void dlaqr5(bool wantt, bool wantz, int kacc22, int n, int ktop, int kbot,
            int nshfts, T* sr, T* si, T* H, int ldh, int iloz, int ihiz,
            T* Z, int ldz, T* V, int ldv, T* U, int ldu,
            int nv, T* WV, int ldwv, int nh, T* WH, int ldwh,
            hseqr_gpu_ctx<T>* gpu = nullptr)
{
    const T zero = 0;
    const T one = 1;

    if(nshfts < 2)
        return;
    if(ktop >= kbot)
        return;

    // Shuffle shifts into conjugate pairs
    for(int i = 1; i <= nshfts - 2; i += 2)
    {
        if(si[i - 1] != -si[i])
        {
            std::swap(sr[i - 1], sr[i]);
            std::swap(sr[i], sr[i + 1]);
            std::swap(si[i - 1], si[i]);
            std::swap(si[i], si[i + 1]);
        }
    }

    int ns = nshfts - (nshfts % 2);

    T safmin = std::numeric_limits<T>::min();
    T ulp = std::numeric_limits<T>::epsilon();
    T smlnum = safmin * (T(n) / ulp);

    bool accum = (kacc22 == 1 || kacc22 == 2);

    if(ktop + 2 <= kbot)
        hf(H, ldh, ktop + 2, ktop) = zero;

    int nbmps = ns / 2;
    int kdu = 4 * nbmps;

    for(int incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps)
    {
        int jtop;
        if(accum)
            jtop = std::max(ktop, incol);
        else if(wantt)
            jtop = 1;
        else
            jtop = ktop;

        int ndcol = incol + kdu;
        if(accum)
            host_laset("A", kdu, kdu, zero, one, U, ldu);

        for(int krcol = incol; krcol <= std::min(incol + 2 * nbmps - 1, kbot - 2); krcol++)
        {
            int mtop = std::max(1, (ktop - krcol) / 2 + 1);
            int mbot = std::min(nbmps, (kbot - krcol - 1) / 2);
            int m22 = mbot + 1;
            bool bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1)) == (kbot - 2);

            if(bmp22)
            {
                int k = krcol + 2 * (m22 - 1);
                if(k == ktop - 1)
                {
                    dlaqr1(2, &hf(H, ldh, k + 1, k + 1), ldh,
                            sr[2 * m22 - 2], si[2 * m22 - 2],
                            sr[2 * m22 - 1], si[2 * m22 - 1],
                            &hf(V, ldv, 1, m22));
                    T beta = hf(V, ldv, 1, m22);
                    host_larfg(2, beta, &hf(V, ldv, 2, m22), 1, hf(V, ldv, 1, m22));
                }
                else
                {
                    T beta = hf(H, ldh, k + 1, k);
                    hf(V, ldv, 2, m22) = hf(H, ldh, k + 2, k);
                    host_larfg(2, beta, &hf(V, ldv, 2, m22), 1, hf(V, ldv, 1, m22));
                    hf(H, ldh, k + 1, k) = beta;
                    hf(H, ldh, k + 2, k) = zero;
                }

                T t1 = hf(V, ldv, 1, m22);
                T t2 = t1 * hf(V, ldv, 2, m22);
                for(int j = jtop; j <= std::min(kbot, k + 3); j++)
                {
                    T refsum = hf(H, ldh, j, k + 1) + hf(V, ldv, 2, m22) * hf(H, ldh, j, k + 2);
                    hf(H, ldh, j, k + 1) -= refsum * t1;
                    hf(H, ldh, j, k + 2) -= refsum * t2;
                }

                int jbot;
                if(accum)
                    jbot = std::min(ndcol, kbot);
                else if(wantt)
                    jbot = n;
                else
                    jbot = kbot;
                t1 = hf(V, ldv, 1, m22);
                t2 = t1 * hf(V, ldv, 2, m22);
                for(int j = k + 1; j <= jbot; j++)
                {
                    T refsum = hf(H, ldh, k + 1, j) + hf(V, ldv, 2, m22) * hf(H, ldh, k + 2, j);
                    hf(H, ldh, k + 1, j) -= refsum * t1;
                    hf(H, ldh, k + 2, j) -= refsum * t2;
                }

                if(k >= ktop)
                {
                    if(hf(H, ldh, k + 1, k) != zero)
                    {
                        T tst1 = std::abs(hf(H, ldh, k, k)) + std::abs(hf(H, ldh, k + 1, k + 1));
                        if(tst1 == zero)
                        {
                            if(k >= ktop + 1) tst1 += std::abs(hf(H, ldh, k, k - 1));
                            if(k >= ktop + 2) tst1 += std::abs(hf(H, ldh, k, k - 2));
                            if(k >= ktop + 3) tst1 += std::abs(hf(H, ldh, k, k - 3));
                            if(k <= kbot - 2) tst1 += std::abs(hf(H, ldh, k + 2, k + 1));
                            if(k <= kbot - 3) tst1 += std::abs(hf(H, ldh, k + 3, k + 1));
                            if(k <= kbot - 4) tst1 += std::abs(hf(H, ldh, k + 4, k + 1));
                        }
                        if(std::abs(hf(H, ldh, k + 1, k)) <= std::max(smlnum, ulp * tst1))
                        {
                            T h12 = std::max(std::abs(hf(H, ldh, k + 1, k)), std::abs(hf(H, ldh, k, k + 1)));
                            T h21 = std::min(std::abs(hf(H, ldh, k + 1, k)), std::abs(hf(H, ldh, k, k + 1)));
                            T h11 = std::max(std::abs(hf(H, ldh, k + 1, k + 1)),
                                             std::abs(hf(H, ldh, k, k) - hf(H, ldh, k + 1, k + 1)));
                            T h22 = std::min(std::abs(hf(H, ldh, k + 1, k + 1)),
                                             std::abs(hf(H, ldh, k, k) - hf(H, ldh, k + 1, k + 1)));
                            T scl = h11 + h12;
                            T tst2 = h22 * (h11 / scl);
                            if(tst2 == zero || h21 * (h12 / scl) <= std::max(smlnum, ulp * tst2))
                                hf(H, ldh, k + 1, k) = zero;
                        }
                    }
                }

                if(accum)
                {
                    int kms = k - incol;
                    t1 = hf(V, ldv, 1, m22);
                    t2 = t1 * hf(V, ldv, 2, m22);
                    for(int j = std::max(1, ktop - incol); j <= kdu; j++)
                    {
                        T refsum = hf(U, ldu, j, kms + 1) + hf(V, ldv, 2, m22) * hf(U, ldu, j, kms + 2);
                        hf(U, ldu, j, kms + 1) -= refsum * t1;
                        hf(U, ldu, j, kms + 2) -= refsum * t2;
                    }
                }
                else if(wantz)
                {
                    t1 = hf(V, ldv, 1, m22);
                    t2 = t1 * hf(V, ldv, 2, m22);
                    for(int j = iloz; j <= ihiz; j++)
                    {
                        T refsum = hf(Z, ldz, j, k + 1) + hf(V, ldv, 2, m22) * hf(Z, ldz, j, k + 2);
                        hf(Z, ldz, j, k + 1) -= refsum * t1;
                        hf(Z, ldz, j, k + 2) -= refsum * t2;
                    }
                }
            }

            // Normal case: chain of 3-by-3 reflections
            for(int m = mbot; m >= mtop; m--)
            {
                int k = krcol + 2 * (m - 1);
                if(k == ktop - 1)
                {
                    dlaqr1(3, &hf(H, ldh, ktop, ktop), ldh,
                            sr[2 * m - 2], si[2 * m - 2],
                            sr[2 * m - 1], si[2 * m - 1],
                            &hf(V, ldv, 1, m));
                    T alpha = hf(V, ldv, 1, m);
                    host_larfg(3, alpha, &hf(V, ldv, 2, m), 1, hf(V, ldv, 1, m));
                }
                else
                {
                    T t1 = hf(V, ldv, 1, m);
                    T t2 = t1 * hf(V, ldv, 2, m);
                    T t3 = t1 * hf(V, ldv, 3, m);
                    T refsum = hf(V, ldv, 3, m) * hf(H, ldh, k + 3, k + 2);
                    hf(H, ldh, k + 3, k) = -refsum * t1;
                    hf(H, ldh, k + 3, k + 1) = -refsum * t2;
                    hf(H, ldh, k + 3, k + 2) -= refsum * t3;

                    T beta = hf(H, ldh, k + 1, k);
                    hf(V, ldv, 2, m) = hf(H, ldh, k + 2, k);
                    hf(V, ldv, 3, m) = hf(H, ldh, k + 3, k);
                    host_larfg(3, beta, &hf(V, ldv, 2, m), 1, hf(V, ldv, 1, m));

                    if(hf(H, ldh, k + 3, k) != zero || hf(H, ldh, k + 3, k + 1) != zero
                       || hf(H, ldh, k + 3, k + 2) == zero)
                    {
                        hf(H, ldh, k + 1, k) = beta;
                        hf(H, ldh, k + 2, k) = zero;
                        hf(H, ldh, k + 3, k) = zero;
                    }
                    else
                    {
                        T vt[3];
                        dlaqr1(3, &hf(H, ldh, k + 1, k + 1), ldh,
                                sr[2 * m - 2], si[2 * m - 2],
                                sr[2 * m - 1], si[2 * m - 1], vt);
                        T alpha2 = vt[0];
                        host_larfg(3, alpha2, &vt[1], 1, vt[0]);
                        T tt1 = vt[0], tt2 = tt1 * vt[1], tt3 = tt1 * vt[2];
                        refsum = hf(H, ldh, k + 1, k) + vt[1] * hf(H, ldh, k + 2, k);
                        if(std::abs(hf(H, ldh, k + 2, k) - refsum * tt2) + std::abs(refsum * tt3)
                           > ulp * (std::abs(hf(H, ldh, k, k)) + std::abs(hf(H, ldh, k + 1, k + 1))
                                    + std::abs(hf(H, ldh, k + 2, k + 2))))
                        {
                            hf(H, ldh, k + 1, k) = beta;
                            hf(H, ldh, k + 2, k) = zero;
                            hf(H, ldh, k + 3, k) = zero;
                        }
                        else
                        {
                            hf(H, ldh, k + 1, k) -= refsum * tt1;
                            hf(H, ldh, k + 2, k) = zero;
                            hf(H, ldh, k + 3, k) = zero;
                            hf(V, ldv, 1, m) = vt[0];
                            hf(V, ldv, 2, m) = vt[1];
                            hf(V, ldv, 3, m) = vt[2];
                        }
                    }
                }

                T t1 = hf(V, ldv, 1, m);
                T t2 = t1 * hf(V, ldv, 2, m);
                T t3 = t1 * hf(V, ldv, 3, m);
                for(int j = jtop; j <= std::min(kbot, k + 3); j++)
                {
                    T refsum = hf(H, ldh, j, k + 1)
                               + hf(V, ldv, 2, m) * hf(H, ldh, j, k + 2)
                               + hf(V, ldv, 3, m) * hf(H, ldh, j, k + 3);
                    hf(H, ldh, j, k + 1) -= refsum * t1;
                    hf(H, ldh, j, k + 2) -= refsum * t2;
                    hf(H, ldh, j, k + 3) -= refsum * t3;
                }

                {
                    T refsum = hf(H, ldh, k + 1, k + 1)
                               + hf(V, ldv, 2, m) * hf(H, ldh, k + 2, k + 1)
                               + hf(V, ldv, 3, m) * hf(H, ldh, k + 3, k + 1);
                    hf(H, ldh, k + 1, k + 1) -= refsum * t1;
                    hf(H, ldh, k + 2, k + 1) -= refsum * t2;
                    hf(H, ldh, k + 3, k + 1) -= refsum * t3;
                }

                if(k < ktop)
                    continue;
                if(hf(H, ldh, k + 1, k) != zero)
                {
                    T tst1 = std::abs(hf(H, ldh, k, k)) + std::abs(hf(H, ldh, k + 1, k + 1));
                    if(tst1 == zero)
                    {
                        if(k >= ktop + 1) tst1 += std::abs(hf(H, ldh, k, k - 1));
                        if(k >= ktop + 2) tst1 += std::abs(hf(H, ldh, k, k - 2));
                        if(k >= ktop + 3) tst1 += std::abs(hf(H, ldh, k, k - 3));
                        if(k <= kbot - 2) tst1 += std::abs(hf(H, ldh, k + 2, k + 1));
                        if(k <= kbot - 3) tst1 += std::abs(hf(H, ldh, k + 3, k + 1));
                        if(k <= kbot - 4) tst1 += std::abs(hf(H, ldh, k + 4, k + 1));
                    }
                    if(std::abs(hf(H, ldh, k + 1, k)) <= std::max(smlnum, ulp * tst1))
                    {
                        T h12 = std::max(std::abs(hf(H, ldh, k + 1, k)), std::abs(hf(H, ldh, k, k + 1)));
                        T h21 = std::min(std::abs(hf(H, ldh, k + 1, k)), std::abs(hf(H, ldh, k, k + 1)));
                        T h11 = std::max(std::abs(hf(H, ldh, k + 1, k + 1)),
                                         std::abs(hf(H, ldh, k, k) - hf(H, ldh, k + 1, k + 1)));
                        T h22 = std::min(std::abs(hf(H, ldh, k + 1, k + 1)),
                                         std::abs(hf(H, ldh, k, k) - hf(H, ldh, k + 1, k + 1)));
                        T scl = h11 + h12;
                        T tst2 = h22 * (h11 / scl);
                        if(tst2 == zero || h21 * (h12 / scl) <= std::max(smlnum, ulp * tst2))
                            hf(H, ldh, k + 1, k) = zero;
                    }
                }
            }

            // Left updates
            int jbot;
            if(accum)
                jbot = std::min(ndcol, kbot);
            else if(wantt)
                jbot = n;
            else
                jbot = kbot;

            for(int m = mbot; m >= mtop; m--)
            {
                int k = krcol + 2 * (m - 1);
                T t1 = hf(V, ldv, 1, m);
                T t2 = t1 * hf(V, ldv, 2, m);
                T t3 = t1 * hf(V, ldv, 3, m);
                for(int j = std::max(ktop, krcol + 2 * m); j <= jbot; j++)
                {
                    T refsum = hf(H, ldh, k + 1, j)
                               + hf(V, ldv, 2, m) * hf(H, ldh, k + 2, j)
                               + hf(V, ldv, 3, m) * hf(H, ldh, k + 3, j);
                    hf(H, ldh, k + 1, j) -= refsum * t1;
                    hf(H, ldh, k + 2, j) -= refsum * t2;
                    hf(H, ldh, k + 3, j) -= refsum * t3;
                }
            }

            // Accumulate U or update Z
            if(accum)
            {
                for(int m = mbot; m >= mtop; m--)
                {
                    int k = krcol + 2 * (m - 1);
                    int kms = k - incol;
                    int i2 = std::max(1, ktop - incol);
                    i2 = std::max(i2, kms - (krcol - incol) + 1);
                    int i4 = std::min(kdu, krcol + 2 * (mbot - 1) - incol + 5);
                    T t1 = hf(V, ldv, 1, m);
                    T t2 = t1 * hf(V, ldv, 2, m);
                    T t3 = t1 * hf(V, ldv, 3, m);
                    for(int j = i2; j <= i4; j++)
                    {
                        T refsum = hf(U, ldu, j, kms + 1)
                                   + hf(V, ldv, 2, m) * hf(U, ldu, j, kms + 2)
                                   + hf(V, ldv, 3, m) * hf(U, ldu, j, kms + 3);
                        hf(U, ldu, j, kms + 1) -= refsum * t1;
                        hf(U, ldu, j, kms + 2) -= refsum * t2;
                        hf(U, ldu, j, kms + 3) -= refsum * t3;
                    }
                }
            }
            else if(wantz)
            {
                for(int m = mbot; m >= mtop; m--)
                {
                    int k = krcol + 2 * (m - 1);
                    T t1 = hf(V, ldv, 1, m);
                    T t2 = t1 * hf(V, ldv, 2, m);
                    T t3 = t1 * hf(V, ldv, 3, m);
                    for(int j = iloz; j <= ihiz; j++)
                    {
                        T refsum = hf(Z, ldz, j, k + 1)
                                   + hf(V, ldv, 2, m) * hf(Z, ldz, j, k + 2)
                                   + hf(V, ldv, 3, m) * hf(Z, ldz, j, k + 3);
                        hf(Z, ldz, j, k + 1) -= refsum * t1;
                        hf(Z, ldz, j, k + 2) -= refsum * t2;
                        hf(Z, ldz, j, k + 3) -= refsum * t3;
                    }
                }
            }
        } // krcol

        // BLAS-3 far-from-diagonal updates
        if(accum)
        {
            int jtop2, jbot2;
            if(wantt) { jtop2 = 1; jbot2 = n; }
            else { jtop2 = ktop; jbot2 = kbot; }

            int k1 = std::max(1, ktop - incol);
            int nu = (kdu - std::max(0, ndcol - kbot)) - k1 + 1;

            if(gpu && gpu->active && nu >= 64)
            {
                // Sync the bulge-chased band region of H to device
                // The band modified by bulge chasing spans rows/cols [incol+k1-1..ndcol-1] (0-based)
                int band_r0 = incol + k1 - 1;
                int band_r1 = std::min(n - 1, ndcol + kdu - 1);
                int band_c0 = incol + k1 - 1;
                int band_c1 = std::min(n - 1, ndcol + kdu - 1);
                gpu->sync_h_to_device(H, band_r0, band_r1, band_c0, band_c1);

                // Upload U (accumulator) to device
                size_t u_bytes = static_cast<size_t>(ldu) * static_cast<size_t>(kdu) * sizeof(T);
                if(!gpu->dU) (void)hipMalloc(&gpu->dU, u_bytes);
                (void)hipMemcpyAsync(gpu->dU, U, u_bytes, hipMemcpyHostToDevice, gpu->stream);

                // Allocate scratch on device
                size_t wh_bytes = static_cast<size_t>(nu) * static_cast<size_t>(nh > 0 ? nh : 1) * sizeof(T);
                size_t wv_bytes = static_cast<size_t>(nv > 0 ? nv : 1) * static_cast<size_t>(nu) * sizeof(T);
                if(!gpu->dWH) (void)hipMalloc(&gpu->dWH, wh_bytes);
                if(!gpu->dWV) (void)hipMalloc(&gpu->dWV, wv_bytes);
                (void)hipStreamSynchronize(gpu->stream);

                T alpha_one = one;
                T beta_zero = zero;

                // Horizontal multiply: WH = U^T * H_strip, then copy back
                for(int jcol = std::min(ndcol, kbot) + 1; jcol <= jbot2; jcol += nh)
                {
                    int jlen = std::min(nh, jbot2 - jcol + 1);
                    T* dU_k1 = gpu->dU + (k1 - 1) + static_cast<size_t>(k1 - 1) * ldu;
                    T* dH_src = gpu->dH + (incol + k1 - 1) + static_cast<size_t>(jcol - 1) * gpu->ldh;
                    rocblas_gemm_dispatch<T>(gpu->handle,
                        rocblas_operation_transpose, rocblas_operation_none,
                        nu, jlen, nu, &alpha_one, dU_k1, ldu, dH_src, gpu->ldh,
                        &beta_zero, gpu->dWH, nu);
                    // Copy result back to dH
                    (void)hipMemcpy2DAsync(dH_src, static_cast<size_t>(gpu->ldh) * sizeof(T),
                                           gpu->dWH, static_cast<size_t>(nu) * sizeof(T),
                                           static_cast<size_t>(nu) * sizeof(T), jlen,
                                           hipMemcpyDeviceToDevice, gpu->stream);
                }

                // Vertical multiply: WV = H_strip * U, then copy back
                for(int jrow = jtop2; jrow <= std::max(ktop, incol) - 1; jrow += nv)
                {
                    int jlen = std::min(nv, std::max(ktop, incol) - jrow);
                    T* dU_k1 = gpu->dU + (k1 - 1) + static_cast<size_t>(k1 - 1) * ldu;
                    T* dH_src = gpu->dH + (jrow - 1) + static_cast<size_t>(incol + k1 - 1) * gpu->ldh;
                    rocblas_gemm_dispatch<T>(gpu->handle,
                        rocblas_operation_none, rocblas_operation_none,
                        jlen, nu, nu, &alpha_one, dH_src, gpu->ldh, dU_k1, ldu,
                        &beta_zero, gpu->dWV, jlen);
                    (void)hipMemcpy2DAsync(dH_src, static_cast<size_t>(gpu->ldh) * sizeof(T),
                                           gpu->dWV, static_cast<size_t>(jlen) * sizeof(T),
                                           static_cast<size_t>(jlen) * sizeof(T), nu,
                                           hipMemcpyDeviceToDevice, gpu->stream);
                }

                // Z multiply
                if(wantz && gpu->dZ)
                {
                    gpu->sync_z_to_device(Z, iloz - 1, ihiz - 1, incol + k1 - 1,
                                          std::min(n - 1, incol + k1 - 1 + nu - 1));
                    for(int jrow = iloz; jrow <= ihiz; jrow += nv)
                    {
                        int jlen = std::min(nv, ihiz - jrow + 1);
                        T* dU_k1 = gpu->dU + (k1 - 1) + static_cast<size_t>(k1 - 1) * ldu;
                        T* dZ_src = gpu->dZ + (jrow - 1) + static_cast<size_t>(incol + k1 - 1) * gpu->ldz;
                        rocblas_gemm_dispatch<T>(gpu->handle,
                            rocblas_operation_none, rocblas_operation_none,
                            jlen, nu, nu, &alpha_one, dZ_src, gpu->ldz, dU_k1, ldu,
                            &beta_zero, gpu->dWV, jlen);
                        (void)hipMemcpy2DAsync(dZ_src, static_cast<size_t>(gpu->ldz) * sizeof(T),
                                               gpu->dWV, static_cast<size_t>(jlen) * sizeof(T),
                                               static_cast<size_t>(jlen) * sizeof(T), nu,
                                               hipMemcpyDeviceToDevice, gpu->stream);
                    }
                }

                (void)hipStreamSynchronize(gpu->stream);

                // Download updated H columns and Z columns back to host
                gpu->sync_h_from_device(H, 0, n - 1, incol + k1 - 1,
                                        std::min(n - 1, incol + k1 - 1 + nu - 1));
                if(wantt)
                {
                    int hcol_start = std::min(ndcol, kbot);
                    if(hcol_start < n - 1)
                        gpu->sync_h_from_device(H, incol + k1 - 1, incol + k1 - 1 + nu - 1,
                                                hcol_start, n - 1);
                }
                if(wantz && gpu->dZ)
                    gpu->sync_z_from_device(Z, iloz - 1, ihiz - 1, incol + k1 - 1,
                                            std::min(n - 1, incol + k1 - 1 + nu - 1));
                gpu->sync();
            }
            else
            {
                // Host GEMM path (original)
                for(int jcol = std::min(ndcol, kbot) + 1; jcol <= jbot2; jcol += nh)
                {
                    int jlen = std::min(nh, jbot2 - jcol + 1);
                    host_gemm("C", "N", nu, jlen, nu, one,
                               &hf(U, ldu, k1, k1), ldu,
                               &hf(H, ldh, incol + k1, jcol), ldh,
                               zero, WH, ldwh);
                    host_lacpy("A", nu, jlen, WH, ldwh, &hf(H, ldh, incol + k1, jcol), ldh);
                }

                for(int jrow = jtop2; jrow <= std::max(ktop, incol) - 1; jrow += nv)
                {
                    int jlen = std::min(nv, std::max(ktop, incol) - jrow);
                    host_gemm("N", "N", jlen, nu, nu, one,
                               &hf(H, ldh, jrow, incol + k1), ldh,
                               &hf(U, ldu, k1, k1), ldu,
                               zero, WV, ldwv);
                    host_lacpy("A", jlen, nu, WV, ldwv, &hf(H, ldh, jrow, incol + k1), ldh);
                }

                if(wantz)
                {
                    for(int jrow = iloz; jrow <= ihiz; jrow += nv)
                    {
                        int jlen = std::min(nv, ihiz - jrow + 1);
                        host_gemm("N", "N", jlen, nu, nu, one,
                                   &hf(Z, ldz, jrow, incol + k1), ldz,
                                   &hf(U, ldu, k1, k1), ldu,
                                   zero, WV, ldwv);
                        host_lacpy("A", jlen, nu, WV, ldwv, &hf(Z, ldz, jrow, incol + k1), ldz);
                    }
                }
            }
        }
    } // incol
}

/**
 * DLAQR3_AED — aggressive early deflation.
 * Faithful C++ translation of LAPACK 3.12 dlaqr2.f (non-recursive variant)
 * using native dlarf1f (v(1)=1 convention).
 * All indices are 1-based (Fortran convention).
 */
template <typename T>
void dlaqr3_aed(bool wantt, bool wantz, int n, int ktop, int kbot, int nw,
                T* H, int ldh, int iloz, int ihiz, T* Z, int ldz,
                int& ns, int& nd, T* sr, T* si,
                T* V, int ldv, int nho, T* Tbuf, int ldt,
                int nve, T* WV, int ldwv, T* work, int lwork,
                hseqr_gpu_ctx<T>* gpu = nullptr)
{
    const T zero = 0;
    const T one = 1;

    int jw = std::min(nw, kbot - ktop + 1);
    if(jw <= 2)
    {
        // workspace query
        if(lwork == -1) { work[0] = one; return; }
    }
    else if(lwork == -1)
    {
        int lwk1_info = 0;
        host_gehrd(jw, 1, jw - 1, Tbuf, ldt, work, work, -1, lwk1_info);
        int lwk1 = (int)work[0];
        int lwk2_info = 0;
        host_ormhr("R", "N", jw, jw, 1, jw - 1, Tbuf, ldt, work, V, ldv, work, -1, lwk2_info);
        int lwk2 = (int)work[0];
        work[0] = T(jw + std::max(lwk1, lwk2));
        return;
    }

    ns = 0;
    nd = 0;
    work[0] = one;
    if(ktop > kbot)
        return;
    if(nw < 1)
        return;

    T safmin = std::numeric_limits<T>::min();
    T ulp = std::numeric_limits<T>::epsilon();
    T smlnum = safmin * (T(n) / ulp);

    jw = std::min(nw, kbot - ktop + 1);
    int kwtop = kbot - jw + 1;
    T s;
    if(kwtop == ktop)
        s = zero;
    else
        s = hf(H, ldh, kwtop, kwtop - 1);

    if(kbot == kwtop)
    {
        sr[kwtop - 1] = hf(H, ldh, kwtop, kwtop);
        si[kwtop - 1] = zero;
        ns = 1;
        nd = 0;
        if(std::abs(s) <= std::max(smlnum, ulp * std::abs(hf(H, ldh, kwtop, kwtop))))
        {
            ns = 0;
            nd = 1;
            if(kwtop > ktop)
                hf(H, ldh, kwtop, kwtop - 1) = zero;
        }
        work[0] = one;
        return;
    }

    // Copy trailing window into T, including sub-diagonal
    host_lacpy("U", jw, jw, &hf(H, ldh, kwtop, kwtop), ldh, Tbuf, ldt);
    for(int j = 0; j < jw - 1; j++)
        Tbuf[(j + 1) + j * ldt] = hf(H, ldh, kwtop + j + 1, kwtop + j);

    host_laset("A", jw, jw, zero, one, V, ldv);

    int infqr = dlahqr(true, true, jw, 1, jw, Tbuf, ldt,
                        &sr[kwtop - 1], &si[kwtop - 1], 1, jw, V, ldv);

    // Clean margin for DTREXC
    for(int j = 1; j <= jw - 3; j++)
    {
        Tbuf[(j + 1) + (j - 1) * ldt] = zero;
        Tbuf[(j + 2) + (j - 1) * ldt] = zero;
    }
    if(jw > 2)
        Tbuf[(jw - 1) + (jw - 3) * ldt] = zero;

    // Deflation detection loop
    ns = jw;
    int ilst = infqr + 1;
    while(ilst <= ns)
    {
        bool bulge = false;
        if(ns > 1)
            bulge = (Tbuf[(ns - 1) + (ns - 2) * ldt] != zero);

        if(!bulge)
        {
            T foo = std::abs(Tbuf[(ns - 1) + (ns - 1) * ldt]);
            if(foo == zero)
                foo = std::abs(s);
            if(std::abs(s * V[(ns - 1) * ldv]) <= std::max(smlnum, ulp * foo))
            {
                ns--;
            }
            else
            {
                int ifst = ns;
                host_trexc("V", jw, Tbuf, ldt, V, ldv, ifst, ilst, work, infqr);
                ilst++;
            }
        }
        else
        {
            T foo = std::abs(Tbuf[(ns - 1) + (ns - 1) * ldt])
                    + std::sqrt(std::abs(Tbuf[(ns - 1) + (ns - 2) * ldt]))
                      * std::sqrt(std::abs(Tbuf[(ns - 2) + (ns - 1) * ldt]));
            if(foo == zero)
                foo = std::abs(s);
            if(std::max(std::abs(s * V[(ns - 1) * ldv]),
                        std::abs(s * V[(ns - 2) * ldv]))
               <= std::max(smlnum, ulp * foo))
            {
                ns -= 2;
            }
            else
            {
                int ifst = ns;
                host_trexc("V", jw, Tbuf, ldt, V, ldv, ifst, ilst, work, infqr);
                ilst += 2;
            }
        }
    }

    if(ns == 0)
        s = zero;

    if(ns < jw)
    {
        // Sort diagonal blocks by eigenvalue magnitude (bubble sort)
        bool sorted = false;
        int isrt = ns + 1;
        while(!sorted)
        {
            sorted = true;
            int kend = isrt - 1;
            isrt = infqr + 1;
            int kk;
            if(isrt == ns)
                kk = isrt + 1;
            else if(Tbuf[isrt + (isrt - 1) * ldt] == zero)
                kk = isrt + 1;
            else
                kk = isrt + 2;

            while(kk <= kend)
            {
                T evi, evk;
                if(kk == isrt + 1)
                    evi = std::abs(Tbuf[(isrt - 1) + (isrt - 1) * ldt]);
                else
                    evi = std::abs(Tbuf[(isrt - 1) + (isrt - 1) * ldt])
                          + std::sqrt(std::abs(Tbuf[isrt + (isrt - 1) * ldt]))
                            * std::sqrt(std::abs(Tbuf[(isrt - 1) + isrt * ldt]));

                if(kk == kend)
                    evk = std::abs(Tbuf[(kk - 1) + (kk - 1) * ldt]);
                else if(Tbuf[kk + (kk - 1) * ldt] == zero)
                    evk = std::abs(Tbuf[(kk - 1) + (kk - 1) * ldt]);
                else
                    evk = std::abs(Tbuf[(kk - 1) + (kk - 1) * ldt])
                          + std::sqrt(std::abs(Tbuf[kk + (kk - 1) * ldt]))
                            * std::sqrt(std::abs(Tbuf[(kk - 1) + kk * ldt]));

                if(evi >= evk)
                {
                    isrt = kk;
                }
                else
                {
                    sorted = false;
                    int ifst = isrt;
                    int ilst2 = kk;
                    int info2 = 0;
                    host_trexc("V", jw, Tbuf, ldt, V, ldv, ifst, ilst2, work, info2);
                    if(info2 == 0)
                        isrt = ilst2;
                    else
                        isrt = kk;
                }

                if(isrt == kend)
                    kk = isrt + 1;
                else if(Tbuf[isrt + (isrt - 1) * ldt] == zero)
                    kk = isrt + 1;
                else
                    kk = isrt + 2;
            }
        }
    }

    // Restore eigenvalues from T
    {
        int ii = jw;
        while(ii >= infqr + 1)
        {
            if(ii == infqr + 1 || Tbuf[(ii - 1) + (ii - 2) * ldt] == zero)
            {
                sr[kwtop + ii - 2] = Tbuf[(ii - 1) + (ii - 1) * ldt];
                si[kwtop + ii - 2] = zero;
                ii--;
            }
            else
            {
                T aa = Tbuf[(ii - 2) + (ii - 2) * ldt];
                T cc = Tbuf[(ii - 1) + (ii - 2) * ldt];
                T bb = Tbuf[(ii - 2) + (ii - 1) * ldt];
                T dd = Tbuf[(ii - 1) + (ii - 1) * ldt];
                T cs_loc, sn_loc;
                dlanv2(aa, bb, cc, dd,
                       sr[kwtop + ii - 3], si[kwtop + ii - 3],
                       sr[kwtop + ii - 2], si[kwtop + ii - 2],
                       cs_loc, sn_loc);
                ii -= 2;
            }
        }
    }

    if(ns < jw || s == zero)
    {
        if(ns > 1 && s != zero)
        {
            // Reflect spike back using dlarf1f (v(1)=1 convention).
            // Copy first row of V into work (spike vector with stride ldv).
            for(int i = 0; i < ns; i++)
                work[i] = V[i * ldv]; // first ROW of V

            T beta_loc = work[0];
            T tau_loc;
            host_larfg(ns, beta_loc, &work[1], 1, tau_loc);

            host_laset("L", jw - 2, jw - 2, zero, zero, &Tbuf[2 + 0 * ldt], ldt);

            dlarf1f('L', ns, jw, work, 1, tau_loc, Tbuf, ldt, &work[jw]);
            dlarf1f('R', ns, ns, work, 1, tau_loc, Tbuf, ldt, &work[jw]);
            dlarf1f('R', jw, ns, work, 1, tau_loc, V, ldv, &work[jw]);

            int gehrd_info = 0;
            host_gehrd(jw, 1, ns, Tbuf, ldt, work, &work[jw], lwork - jw, gehrd_info);
        }

        // Copy updated window back
        if(kwtop > 1)
            hf(H, ldh, kwtop, kwtop - 1) = s * V[0]; // V(1,1)
        host_lacpy("U", jw, jw, Tbuf, ldt, &hf(H, ldh, kwtop, kwtop), ldh);
        for(int j = 0; j < jw - 1; j++)
            hf(H, ldh, kwtop + j + 1, kwtop + j) = Tbuf[(j + 1) + j * ldt];

        // Accumulate orthogonal matrix via DORMHR
        if(ns > 1 && s != zero)
        {
            int ormhr_info = 0;
            host_ormhr("R", "N", jw, ns, 1, ns, Tbuf, ldt, work,
                        V, ldv, &work[jw], lwork - jw, ormhr_info);
        }

        // AED slab updates (vertical, horizontal, Z)
        int ltop = wantt ? 1 : ktop;

        if(gpu && gpu->active && jw >= 64)
        {
            gpu->sync_h_to_device(H, kwtop - 1, kbot - 1, kwtop - 1, kbot - 1);

            T* dV_aed = nullptr;
            size_t v_bytes = static_cast<size_t>(ldv) * static_cast<size_t>(jw) * sizeof(T);
            (void)hipMalloc(&dV_aed, v_bytes);
            (void)hipMemcpyAsync(dV_aed, V, v_bytes, hipMemcpyHostToDevice, gpu->stream);

            T* dWV_aed = nullptr;
            size_t wv_bytes = static_cast<size_t>(nve > 0 ? nve : 1) * static_cast<size_t>(jw) * sizeof(T);
            (void)hipMalloc(&dWV_aed, wv_bytes);

            T* dTbuf_aed = nullptr;
            size_t tbuf_bytes = static_cast<size_t>(jw) * static_cast<size_t>(nho > 0 ? nho : 1) * sizeof(T);
            (void)hipMalloc(&dTbuf_aed, tbuf_bytes);
            (void)hipStreamSynchronize(gpu->stream);

            T alpha_one = one;
            T beta_zero = zero;

            for(int krow = ltop; krow <= kwtop - 1; krow += nve)
            {
                int kln = std::min(nve, kwtop - krow);
                T* dH_src = gpu->dH + (krow - 1) + static_cast<size_t>(kwtop - 1) * gpu->ldh;
                rocblas_gemm_dispatch<T>(gpu->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    kln, jw, jw, &alpha_one, dH_src, gpu->ldh, dV_aed, ldv,
                    &beta_zero, dWV_aed, kln);
                (void)hipMemcpy2DAsync(dH_src, static_cast<size_t>(gpu->ldh) * sizeof(T),
                                       dWV_aed, static_cast<size_t>(kln) * sizeof(T),
                                       static_cast<size_t>(kln) * sizeof(T), jw,
                                       hipMemcpyDeviceToDevice, gpu->stream);
            }

            if(wantt)
            {
                for(int kcol = kbot + 1; kcol <= n; kcol += nho)
                {
                    int kln = std::min(nho, n - kcol + 1);
                    T* dH_src = gpu->dH + (kwtop - 1) + static_cast<size_t>(kcol - 1) * gpu->ldh;
                    rocblas_gemm_dispatch<T>(gpu->handle,
                        rocblas_operation_transpose, rocblas_operation_none,
                        jw, kln, jw, &alpha_one, dV_aed, ldv, dH_src, gpu->ldh,
                        &beta_zero, dTbuf_aed, jw);
                    (void)hipMemcpy2DAsync(dH_src, static_cast<size_t>(gpu->ldh) * sizeof(T),
                                           dTbuf_aed, static_cast<size_t>(jw) * sizeof(T),
                                           static_cast<size_t>(jw) * sizeof(T), kln,
                                           hipMemcpyDeviceToDevice, gpu->stream);
                }
            }

            if(wantz && gpu->dZ)
            {
                gpu->sync_z_to_device(Z, iloz - 1, ihiz - 1, kwtop - 1, kbot - 1);
                for(int krow = iloz; krow <= ihiz; krow += nve)
                {
                    int kln = std::min(nve, ihiz - krow + 1);
                    T* dZ_src = gpu->dZ + (krow - 1) + static_cast<size_t>(kwtop - 1) * gpu->ldz;
                    rocblas_gemm_dispatch<T>(gpu->handle,
                        rocblas_operation_none, rocblas_operation_none,
                        kln, jw, jw, &alpha_one, dZ_src, gpu->ldz, dV_aed, ldv,
                        &beta_zero, dWV_aed, kln);
                    (void)hipMemcpy2DAsync(dZ_src, static_cast<size_t>(gpu->ldz) * sizeof(T),
                                           dWV_aed, static_cast<size_t>(kln) * sizeof(T),
                                           static_cast<size_t>(kln) * sizeof(T), jw,
                                           hipMemcpyDeviceToDevice, gpu->stream);
                }
            }

            (void)hipStreamSynchronize(gpu->stream);
            gpu->sync_h_from_device(H, 0, n - 1, kwtop - 1, kbot - 1);
            if(wantt && kbot < n)
                gpu->sync_h_from_device(H, kwtop - 1, kwtop - 1 + jw - 1, kbot, n - 1);
            if(wantz && gpu->dZ)
                gpu->sync_z_from_device(Z, iloz - 1, ihiz - 1, kwtop - 1, kbot - 1);
            gpu->sync();

            if(dV_aed) (void)hipFree(dV_aed);
            if(dWV_aed) (void)hipFree(dWV_aed);
            if(dTbuf_aed) (void)hipFree(dTbuf_aed);
        }
        else
        {
            for(int krow = ltop; krow <= kwtop - 1; krow += nve)
            {
                int kln = std::min(nve, kwtop - krow);
                host_gemm("N", "N", kln, jw, jw, one,
                           &hf(H, ldh, krow, kwtop), ldh, V, ldv,
                           zero, WV, ldwv);
                host_lacpy("A", kln, jw, WV, ldwv, &hf(H, ldh, krow, kwtop), ldh);
            }

            if(wantt)
            {
                for(int kcol = kbot + 1; kcol <= n; kcol += nho)
                {
                    int kln = std::min(nho, n - kcol + 1);
                    host_gemm("C", "N", jw, kln, jw, one, V, ldv,
                               &hf(H, ldh, kwtop, kcol), ldh,
                               zero, Tbuf, ldt);
                    host_lacpy("A", jw, kln, Tbuf, ldt, &hf(H, ldh, kwtop, kcol), ldh);
                }
            }

            if(wantz)
            {
                for(int krow = iloz; krow <= ihiz; krow += nve)
                {
                    int kln = std::min(nve, ihiz - krow + 1);
                    host_gemm("N", "N", kln, jw, jw, one,
                               &hf(Z, ldz, krow, kwtop), ldz, V, ldv,
                               zero, WV, ldwv);
                    host_lacpy("A", kln, jw, WV, ldwv, &hf(Z, ldz, krow, kwtop), ldz);
                }
            }
        }
    }

    nd = jw - ns;
    ns = ns - infqr;
    work[0] = T(jw + std::max(1, jw));
}

/**
 * DLAQR0 — multi-shift QR with aggressive early deflation.
 * Faithful C++ translation of LAPACK 3.12 dlaqr0.f.
 * Uses dlaqr5 for the multi-shift sweep and dlaqr3_aed for AED.
 * All indices are 1-based (Fortran convention).
 */
template <typename T>
int dlaqr0(bool wantt, bool wantz, int n, int ilo, int ihi,
           T* H, int ldh, T* wr, T* wi,
           int iloz, int ihiz, T* Z, int ldz,
           T* work, int lwork,
           hseqr_gpu_ctx<T>* gpu = nullptr)
{
    const int NTINY = 15;
    const int KEXNW = 5;
    const int KEXSH = 6;
    const T WILK1 = T(0.75);
    const T WILK2 = T(-0.4375);
    const T zero = 0;
    const T one = 1;

    int info = 0;

    if(n == 0)
    {
        work[0] = one;
        return 0;
    }

    if(n <= NTINY)
    {
        if(lwork != -1)
            info = dlahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, iloz, ihiz, Z, ldz);
        work[0] = one;
        return info;
    }

    info = 0;

    int nwr = host_iparmq(13, 0, ilo, ihi);
    nwr = std::max(2, nwr);
    nwr = std::min({ihi - ilo + 1, (n - 1) / 3, nwr});

    int nsr = host_iparmq(15, 0, ilo, ihi);
    nsr = std::min({nsr, (n - 3) / 6, ihi - ilo});
    nsr = std::max(2, nsr - (nsr % 2));

    // Workspace query for dlaqr3_aed
    int ls_dummy = 0, ld_dummy = 0;
    dlaqr3_aed(wantt, wantz, n, ilo, ihi, nwr + 1, H, ldh, iloz, ihiz, Z, ldz,
               ls_dummy, ld_dummy, wr, wi, H, ldh, n, H, ldh, n, H, ldh, work, -1);
    int lwkopt = std::max(3 * nsr / 2, (int)work[0]);

    if(lwork == -1)
    {
        work[0] = T(lwkopt);
        return 0;
    }

    int nmin = host_iparmq(12, 0, ilo, ihi);
    nmin = std::max(NTINY, nmin);

    int nibble = host_iparmq(14, 0, ilo, ihi);
    nibble = std::max(0, nibble);

    int kacc22 = host_iparmq(16, 0, ilo, ihi);
    kacc22 = std::max(0, std::min(2, kacc22));

    int nwmax = std::min((n - 1) / 3, lwork / 2);
    int nw_loc = nwmax;

    int nsmax = std::min((n - 3) / 6, 2 * lwork / 3);
    nsmax = nsmax - (nsmax % 2);

    int ndfl = 1;
    int ndec = 0;
    int itmax = std::max(30, 2 * KEXSH) * std::max(10, ihi - ilo + 1);
    int kbot = ihi;

    for(int it = 1; it <= itmax; it++)
    {
        if(kbot < ilo)
            goto L90;

        // Locate active block
        int ktop_loc;
        for(int k = kbot; k >= ilo + 1; k--)
        {
            if(hf(H, ldh, k, k - 1) == zero)
            {
                ktop_loc = k;
                goto L20;
            }
        }
        ktop_loc = ilo;
    L20:

        // Select deflation window size
        int nh = kbot - ktop_loc + 1;
        int nwupbd = std::min(nh, nwmax);
        if(ndfl < KEXNW)
            nw_loc = std::min(nwupbd, nwr);
        else
            nw_loc = std::min(nwupbd, 2 * nw_loc);

        if(nw_loc < nwmax)
        {
            if(nw_loc >= nh - 1)
                nw_loc = nh;
            else
            {
                int kwtop = kbot - nw_loc + 1;
                if(std::abs(hf(H, ldh, kwtop, kwtop - 1))
                   > std::abs(hf(H, ldh, kwtop - 1, kwtop - 2)))
                    nw_loc++;
            }
        }
        if(ndfl < KEXNW)
            ndec = -1;
        else if(ndec >= 0 || nw_loc >= nwupbd)
        {
            ndec++;
            if(nw_loc - ndec < 2)
                ndec = 0;
            nw_loc -= ndec;
        }

        // AED workspace
        int kv = n - nw_loc + 1;
        int kt_loc = nw_loc + 1;
        int nho_loc = (n - nw_loc - 1) - kt_loc + 1;
        int kwv = nw_loc + 2;
        int nve_loc = (n - nw_loc) - kwv + 1;

        int ls = 0, ld = 0;
        dlaqr3_aed(wantt, wantz, n, ktop_loc, kbot, nw_loc, H, ldh,
                   iloz, ihiz, Z, ldz, ls, ld, wr, wi,
                   &hf(H, ldh, kv, 1), ldh, nho_loc, &hf(H, ldh, kv, kt_loc), ldh,
                   nve_loc, &hf(H, ldh, kwv, 1), ldh, work, lwork, gpu);

        kbot -= ld;
        int ks = kbot - ls + 1;

        if((ld == 0) || ((100 * ld <= nw_loc * nibble) && (kbot - ktop_loc + 1 > std::min(nmin, nwmax))))
        {
            int ns_loc = std::min({nsmax, nsr, std::max(2, kbot - ktop_loc)});
            ns_loc = ns_loc - (ns_loc % 2);

            if(ndfl % KEXSH == 0)
            {
                ks = kbot - ns_loc + 1;
                for(int i = kbot; i >= std::max(ks + 1, ktop_loc + 2); i -= 2)
                {
                    T ss = std::abs(hf(H, ldh, i, i - 1)) + std::abs(hf(H, ldh, i - 1, i - 2));
                    T aa = WILK1 * ss + hf(H, ldh, i, i);
                    T bb = ss;
                    T cc = WILK2 * ss;
                    T dd = aa;
                    T cs_loc, sn_loc;
                    dlanv2(aa, bb, cc, dd, wr[i - 2], wi[i - 2], wr[i - 1], wi[i - 1], cs_loc, sn_loc);
                }
                if(ks == ktop_loc)
                {
                    wr[ks] = hf(H, ldh, ks + 1, ks + 1);
                    wi[ks] = zero;
                    wr[ks - 1] = wr[ks];
                    wi[ks - 1] = wi[ks];
                }
            }
            else
            {
                if(kbot - ks + 1 <= ns_loc / 2)
                {
                    ks = kbot - ns_loc + 1;
                    int kt2 = n - ns_loc + 1;
                    host_lacpy("A", ns_loc, ns_loc, &hf(H, ldh, ks, ks), ldh,
                                &hf(H, ldh, kt2, 1), ldh);

                    T zdum_arr[1] = {zero};
                    if(ns_loc > nmin)
                    {
                        int inf2 = dlaqr0(false, false, ns_loc, 1, ns_loc,
                                          &hf(H, ldh, kt2, 1), ldh,
                                          &wr[ks - 1], &wi[ks - 1],
                                          1, 1, zdum_arr, 1, work, lwork);
                        ks += inf2;
                    }
                    else
                    {
                        int inf2 = dlahqr(false, false, ns_loc, 1, ns_loc,
                                          &hf(H, ldh, kt2, 1), ldh,
                                          &wr[ks - 1], &wi[ks - 1],
                                          1, 1, zdum_arr, 1);
                        ks += inf2;
                    }

                    if(ks >= kbot)
                    {
                        T aa = hf(H, ldh, kbot - 1, kbot - 1);
                        T cc = hf(H, ldh, kbot, kbot - 1);
                        T bb = hf(H, ldh, kbot - 1, kbot);
                        T dd = hf(H, ldh, kbot, kbot);
                        T cs_loc, sn_loc;
                        dlanv2(aa, bb, cc, dd, wr[kbot - 2], wi[kbot - 2],
                               wr[kbot - 1], wi[kbot - 1], cs_loc, sn_loc);
                        ks = kbot - 1;
                    }
                }

                if(kbot - ks + 1 > ns_loc)
                {
                    bool sorted2 = false;
                    for(int k = kbot; k >= ks + 1 && !sorted2; k--)
                    {
                        sorted2 = true;
                        for(int i = ks; i <= k - 1; i++)
                        {
                            if(std::abs(wr[i - 1]) + std::abs(wi[i - 1])
                               < std::abs(wr[i]) + std::abs(wi[i]))
                            {
                                sorted2 = false;
                                std::swap(wr[i - 1], wr[i]);
                                std::swap(wi[i - 1], wi[i]);
                            }
                        }
                    }
                }

                for(int i = kbot; i >= ks + 2; i -= 2)
                {
                    if(wi[i - 1] != -wi[i - 2])
                    {
                        std::swap(wr[i - 1], wr[i - 2]);
                        std::swap(wr[i - 2], wr[i - 3]);
                        std::swap(wi[i - 1], wi[i - 2]);
                        std::swap(wi[i - 2], wi[i - 3]);
                    }
                }
            }

            if(kbot - ks + 1 == 2 && wi[kbot - 1] == zero)
            {
                if(std::abs(wr[kbot - 1] - hf(H, ldh, kbot, kbot))
                   < std::abs(wr[kbot - 2] - hf(H, ldh, kbot, kbot)))
                    wr[kbot - 2] = wr[kbot - 1];
                else
                    wr[kbot - 1] = wr[kbot - 2];
            }

            ns_loc = std::min(ns_loc, kbot - ks + 1);
            ns_loc = ns_loc - (ns_loc % 2);
            ks = kbot - ns_loc + 1;

            // Multi-shift QR sweep workspace
            int kdu2 = 2 * ns_loc;
            int ku = n - kdu2 + 1;
            int kwh2 = kdu2 + 1;
            int nho2 = (n - kdu2 + 1 - 4) - (kdu2 + 1) + 1;
            int kwv2 = kdu2 + 4;
            int nve2 = n - kdu2 - kwv2 + 1;

            dlaqr5(wantt, wantz, kacc22, n, ktop_loc, kbot, ns_loc,
                   &wr[ks - 1], &wi[ks - 1], H, ldh, iloz, ihiz, Z, ldz,
                   work, 3, &hf(H, ldh, ku, 1), ldh, nve2,
                   &hf(H, ldh, kwv2, 1), ldh, nho2, &hf(H, ldh, ku, kwh2), ldh,
                   gpu);
        }

        if(ld > 0)
            ndfl = 1;
        else
            ndfl++;
    }

    info = kbot;
L90:
    work[0] = T(lwkopt);
    return info;
}

/**
 * DLAHQR — implicit double-shift QR on an upper Hessenberg block.
 * Indices ILO, IHI, ILOZ, IHIZ are 1-based (Fortran).
 */
template <typename T>
rocblas_int dlahqr(bool wantt,
                   bool wantz,
                   rocblas_int n,
                   rocblas_int ilo,
                   rocblas_int ihi,
                   T* H,
                   rocblas_int ldh,
                   T* wr,
                   T* wi,
                   rocblas_int iloz,
                   rocblas_int ihiz,
                   T* Z,
                   rocblas_int ldz)
{
    const T zero = 0;
    const T one = 1;
    const T two = 2;
    const T dat1 = T(3) / T(4);
    const T dat2 = T(-0.4375);
    const rocblas_int kexsh = 10;

    rocblas_int info = 0;
    if(n == 0)
        return info;
    if(ilo == ihi)
    {
        wr[ilo - 1] = hf(H, ldh, ilo, ilo);
        wi[ilo - 1] = zero;
        return info;
    }

    for(rocblas_int j = ilo; j <= ihi - 3; j++)
    {
        hf(H, ldh, j + 2, j) = zero;
        hf(H, ldh, j + 3, j) = zero;
    }
    if(ilo <= ihi - 2)
        hf(H, ldh, ihi, ihi - 2) = zero;

    rocblas_int nh = ihi - ilo + 1;
    rocblas_int nz = ihiz - iloz + 1;

    T safmin = std::numeric_limits<T>::min();
    T ulp = std::numeric_limits<T>::epsilon();
    T smlnum = safmin * (T(nh) / ulp);

    rocblas_int i1 = 1;
    rocblas_int i2 = n;
    if(wantt)
    {
        i1 = 1;
        i2 = n;
    }

    rocblas_int itmax = 30 * std::max(rocblas_int(10), nh);
    rocblas_int kdefl = 0;

    rocblas_int i = ihi;
    while(true)
    {
        if(i < ilo)
            return info;

        rocblas_int l = ilo;
        bool split = false;

        for(rocblas_int its = 0; its <= itmax; ++its)
        {
            rocblas_int k = i;
            bool early = false;
            for(; k >= l + 1; k--)
            {
                if(std::abs(hf(H, ldh, k, k - 1)) <= smlnum)
                {
                    early = true;
                    break;
                }
                T tst = std::abs(hf(H, ldh, k - 1, k - 1)) + std::abs(hf(H, ldh, k, k));
                if(tst == zero)
                {
                    if(k - 2 >= ilo)
                        tst = tst + std::abs(hf(H, ldh, k - 1, k - 2));
                    if(k + 1 <= ihi)
                        tst = tst + std::abs(hf(H, ldh, k + 1, k));
                }
                if(std::abs(hf(H, ldh, k, k - 1)) <= ulp * tst)
                {
                    T ab = std::max(std::abs(hf(H, ldh, k, k - 1)), std::abs(hf(H, ldh, k - 1, k)));
                    T ba = std::min(std::abs(hf(H, ldh, k, k - 1)), std::abs(hf(H, ldh, k - 1, k)));
                    T aa = std::max(std::abs(hf(H, ldh, k, k)),
                                      std::abs(hf(H, ldh, k - 1, k - 1) - hf(H, ldh, k, k)));
                    T bb = std::min(std::abs(hf(H, ldh, k, k)),
                                    std::abs(hf(H, ldh, k - 1, k - 1) - hf(H, ldh, k, k)));
                    T s = aa + ab;
                    if(ba * (ab / s) <= std::max(smlnum, ulp * (bb * (aa / s))))
                    {
                        early = true;
                        break;
                    }
                }
            }
            if(!early)
                k = l;

            l = k;
            if(l > ilo)
                hf(H, ldh, l, l - 1) = zero;

            if(l >= i - 1)
            {
                split = true;
                break;
            }

            kdefl = kdefl + 1;

            if(!wantt)
            {
                i1 = l;
                i2 = i;
            }

            T h11, h12, h21, h22;
            T rt1r, rt1i, rt2r, rt2i;
            T tr, det, rtdisc, s, h21s;
            T v1 = zero, v2 = zero, v3 = zero;
            T sum, t1 = zero, t2, t3;

            if(kdefl % (2 * kexsh) == 0)
            {
                s = std::abs(hf(H, ldh, i, i - 1)) + std::abs(hf(H, ldh, i - 1, i - 2));
                h11 = dat1 * s + hf(H, ldh, i, i);
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            }
            else if(kdefl % kexsh == 0)
            {
                s = std::abs(hf(H, ldh, l + 1, l)) + std::abs(hf(H, ldh, l + 2, l + 1));
                h11 = dat1 * s + hf(H, ldh, l, l);
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            }
            else
            {
                h11 = hf(H, ldh, i - 1, i - 1);
                h21 = hf(H, ldh, i, i - 1);
                h12 = hf(H, ldh, i - 1, i);
                h22 = hf(H, ldh, i, i);
            }

            s = std::abs(h11) + std::abs(h12) + std::abs(h21) + std::abs(h22);
            if(s == zero)
            {
                rt1r = zero;
                rt1i = zero;
                rt2r = zero;
                rt2i = zero;
            }
            else
            {
                h11 = h11 / s;
                h21 = h21 / s;
                h12 = h12 / s;
                h22 = h22 / s;
                tr = (h11 + h22) / two;
                det = (h11 - tr) * (h22 - tr) - h12 * h21;
                rtdisc = std::sqrt(std::abs(det));
                if(det >= zero)
                {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                }
                else
                {
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if(std::abs(rt1r - h22) <= std::abs(rt2r - h22))
                    {
                        rt1r = rt1r * s;
                        rt2r = rt1r;
                    }
                    else
                    {
                        rt2r = rt2r * s;
                        rt1r = rt2r;
                    }
                    rt1i = zero;
                    rt2i = zero;
                }
            }

            rocblas_int m = l;
            bool mfound = false;
            for(m = i - 2; m >= l; m--)
            {
                h21s = hf(H, ldh, m + 1, m);
                s = std::abs(hf(H, ldh, m, m) - rt2r) + std::abs(rt2i) + std::abs(h21s);
                h21s = hf(H, ldh, m + 1, m) / s;
                v1 = h21s * hf(H, ldh, m, m + 1)
                     + (hf(H, ldh, m, m) - rt1r) * ((hf(H, ldh, m, m) - rt2r) / s)
                     - rt1i * (rt2i / s);
                v2 = h21s * (hf(H, ldh, m, m) + hf(H, ldh, m + 1, m + 1) - rt1r - rt2r);
                v3 = h21s * hf(H, ldh, m + 2, m + 1);
                s = std::abs(v1) + std::abs(v2) + std::abs(v3);
                v1 = v1 / s;
                v2 = v2 / s;
                v3 = v3 / s;
                if(m == l)
                {
                    mfound = true;
                    break;
                }
                if(std::abs(hf(H, ldh, m, m - 1)) * (std::abs(v2) + std::abs(v3))
                   <= ulp * std::abs(v1)
                          * (std::abs(hf(H, ldh, m - 1, m - 1)) + std::abs(hf(H, ldh, m, m))
                             + std::abs(hf(H, ldh, m + 1, m + 1))))
                {
                    mfound = true;
                    break;
                }
            }
            if(!mfound)
                m = l;

            T v[3];
            for(rocblas_int k2 = m; k2 <= i - 1; k2++)
            {
                rocblas_int nr = std::min(rocblas_int(3), i - k2 + 1);
                if(k2 > m)
                {
                    v[0] = hf(H, ldh, k2, k2 - 1);
                    v[1] = hf(H, ldh, k2 + 1, k2 - 1);
                    if(nr == 3)
                        v[2] = hf(H, ldh, k2 + 2, k2 - 1);
                }
                else
                {
                    v[0] = v1;
                    v[1] = v2;
                    v[2] = v3;
                }
                dlarfg_small(nr, v[0], &v[1], 1, t1);
                if(k2 > m)
                {
                    hf(H, ldh, k2, k2 - 1) = v[0];
                    hf(H, ldh, k2 + 1, k2 - 1) = zero;
                    if(k2 < i - 1)
                        hf(H, ldh, k2 + 2, k2 - 1) = zero;
                }
                else if(m > l)
                {
                    hf(H, ldh, k2, k2 - 1) = hf(H, ldh, k2, k2 - 1) * (one - t1);
                }
                v2 = v[1];
                t2 = t1 * v2;
                if(nr == 3)
                {
                    v3 = v[2];
                    t3 = t1 * v3;
                    for(rocblas_int j = k2; j <= i2; j++)
                    {
                        sum = hf(H, ldh, k2, j) + v2 * hf(H, ldh, k2 + 1, j)
                            + v3 * hf(H, ldh, k2 + 2, j);
                        hf(H, ldh, k2, j) -= sum * t1;
                        hf(H, ldh, k2 + 1, j) -= sum * t2;
                        hf(H, ldh, k2 + 2, j) -= sum * t3;
                    }
                    rocblas_int jlim = std::min(k2 + 3, i);
                    for(rocblas_int j = i1; j <= jlim; j++)
                    {
                        sum = hf(H, ldh, j, k2) + v2 * hf(H, ldh, j, k2 + 1)
                            + v3 * hf(H, ldh, j, k2 + 2);
                        hf(H, ldh, j, k2) -= sum * t1;
                        hf(H, ldh, j, k2 + 1) -= sum * t2;
                        hf(H, ldh, j, k2 + 2) -= sum * t3;
                    }
                    if(wantz)
                    {
                        for(rocblas_int j = iloz; j <= ihiz; j++)
                        {
                            sum = hf(Z, ldz, j, k2) + v2 * hf(Z, ldz, j, k2 + 1)
                                + v3 * hf(Z, ldz, j, k2 + 2);
                            hf(Z, ldz, j, k2) -= sum * t1;
                            hf(Z, ldz, j, k2 + 1) -= sum * t2;
                            hf(Z, ldz, j, k2 + 2) -= sum * t3;
                        }
                    }
                }
                else if(nr == 2)
                {
                    for(rocblas_int j = k2; j <= i2; j++)
                    {
                        sum = hf(H, ldh, k2, j) + v2 * hf(H, ldh, k2 + 1, j);
                        hf(H, ldh, k2, j) -= sum * t1;
                        hf(H, ldh, k2 + 1, j) -= sum * t2;
                    }
                    for(rocblas_int j = i1; j <= i; j++)
                    {
                        sum = hf(H, ldh, j, k2) + v2 * hf(H, ldh, j, k2 + 1);
                        hf(H, ldh, j, k2) -= sum * t1;
                        hf(H, ldh, j, k2 + 1) -= sum * t2;
                    }
                    if(wantz)
                    {
                        for(rocblas_int j = iloz; j <= ihiz; j++)
                        {
                            sum = hf(Z, ldz, j, k2) + v2 * hf(Z, ldz, j, k2 + 1);
                            hf(Z, ldz, j, k2) -= sum * t1;
                            hf(Z, ldz, j, k2 + 1) -= sum * t2;
                        }
                    }
                }
            }
        }

        if(!split)
            return i;

        if(l == i)
        {
            wr[i - 1] = hf(H, ldh, i, i);
            wi[i - 1] = zero;
        }
        else if(l == i - 1)
        {
            T aa = hf(H, ldh, i - 1, i - 1);
            T bb = hf(H, ldh, i - 1, i);
            T cc = hf(H, ldh, i, i - 1);
            T dd = hf(H, ldh, i, i);
            T cs, sn;
            dlanv2(aa, bb, cc, dd, wr[i - 2], wi[i - 2], wr[i - 1], wi[i - 1], cs, sn);
            hf(H, ldh, i - 1, i - 1) = aa;
            hf(H, ldh, i - 1, i) = bb;
            hf(H, ldh, i, i - 1) = cc;
            hf(H, ldh, i, i) = dd;
            if(wantt)
            {
                if(i2 > i)
                    drot(i2 - i, &hf(H, ldh, i - 1, i + 1), ldh, &hf(H, ldh, i, i + 1), ldh, cs,
                         sn);
                drot(i - i1 - 1, &hf(H, ldh, i1, i - 1), 1, &hf(H, ldh, i1, i), 1, cs, sn);
            }
            if(wantz)
            {
                drot(nz, &hf(Z, ldz, iloz, i - 1), 1, &hf(Z, ldz, iloz, i), 1, cs, sn);
            }
        }
        kdefl = 0;
        i = l - 1;
    }
}

/** DHSEQR host path: native multi-shift QR (dlaqr0) for large N, dlahqr for small. */
template <typename T>
void run_hseqr_native(rocblas_int n,
                      rocblas_int ilo,
                      rocblas_int ihi,
                      bool wantt,
                      bool wantz,
                      bool initz,
                      T* H,
                      rocblas_int ldh,
                      T* wr,
                      T* wi,
                      T* Z,
                      rocblas_int ldz,
                      rocblas_int& info,
                      rocblas_handle gpu_handle = nullptr,
                      hipStream_t gpu_stream = nullptr)
{
    info = 0;
    if(n == 0)
        return;

    const T zero = 0, one = 1;
    for(rocblas_int ii = 1; ii <= ilo - 1; ii++) { wr[ii-1] = hf(H,ldh,ii,ii); wi[ii-1] = zero; }
    for(rocblas_int ii = ihi + 1; ii <= n; ii++) { wr[ii-1] = hf(H,ldh,ii,ii); wi[ii-1] = zero; }
    if(initz && wantz)
        for(rocblas_int j = 1; j <= n; j++)
            for(rocblas_int i = 1; i <= n; i++)
                hf(Z,ldz,i,j) = (i==j) ? one : zero;
    if(ilo == ihi) { wr[ilo-1] = hf(H,ldh,ilo,ilo); wi[ilo-1] = zero; return; }
    for(rocblas_int j = ilo; j <= ihi; ++j)
        for(rocblas_int i = j + 2; i <= ihi; ++i) hf(H,ldh,i,j) = zero;

    {
        hseqr_gpu_ctx<T> gpu_ctx;
        hseqr_gpu_ctx<T>* gpu_ptr = nullptr;
        if(gpu_handle && gpu_stream && n >= HSEQR_GPU_GEMM_THRESHOLD)
        {
            gpu_ctx.init(gpu_handle, gpu_stream, (int)n, (int)ldh, (int)ldz, H, Z, wantz);
            gpu_ptr = &gpu_ctx;
        }

        int lwork = -1;
        T wq;
        dlaqr0(wantt, wantz && (Z != nullptr), (int)n, (int)ilo, (int)ihi,
               H, (int)ldh, wr, wi, 1, (int)n, Z, (int)ldz, &wq, lwork);
        lwork = std::max(1, (int)wq);
        std::vector<T> work(lwork);
        info = (rocblas_int)dlaqr0(wantt, wantz && (Z != nullptr), (int)n, (int)ilo, (int)ihi,
                                    H, (int)ldh, wr, wi, 1, (int)n, Z, (int)ldz,
                                    work.data(), lwork, gpu_ptr);

        if(gpu_ptr)
            gpu_ctx.destroy();
    }
}

#ifndef HSEQR_STANDALONE_BENCH
/** DHSEQR via optimized LAPACK for best performance. */
template <typename T>
void run_hseqr_lapack(rocblas_int n,
                      rocblas_int ilo,
                      rocblas_int ihi,
                      bool wantt,
                      bool wantz,
                      bool initz,
                      T* H,
                      rocblas_int ldh,
                      T* wr,
                      T* wi,
                      T* Z,
                      rocblas_int ldz,
                      rocblas_int& info)
{
    info = 0;
    if(n == 0)
        return;

    const char* job = wantt ? "S" : "E";
    const char* compz = wantz ? (initz ? "I" : "V") : "N";
    int n_i = (int)n, ilo_i = (int)ilo, ihi_i = (int)ihi;
    int ldh_i = (int)ldh, ldz_i = (int)ldz;
    int info_i = 0;

    int lwork = -1;
    T wq;
    host_hseqr(job, compz, n_i, ilo_i, ihi_i, H, ldh_i, wr, wi, Z, ldz_i, &wq, lwork, info_i);
    lwork = std::max(1, (int)wq);
    std::vector<T> work(lwork);
    host_hseqr(job, compz, n_i, ilo_i, ihi_i, H, ldh_i, wr, wi, Z, ldz_i, work.data(), lwork, info_i);
    info = (rocblas_int)info_i;
}
#endif

} // namespace hseqr_host_detail

template <typename T>
void run_hseqr(rocblas_int n,
               rocblas_int ilo,
               rocblas_int ihi,
               bool wantt,
               bool wantz,
               bool initz,
               T* H,
               rocblas_int ldh,
               T* wr,
               T* wi,
               T* Z,
               rocblas_int ldz,
               rocblas_int& info,
               rocblas_handle gpu_handle = nullptr,
               hipStream_t gpu_stream = nullptr)
{
#ifdef ROCSOLVER_NATIVE_HSEQR
    hseqr_host_detail::run_hseqr_native(n, ilo, ihi, wantt, wantz, initz, H, ldh, wr, wi, Z, ldz,
                                          info, gpu_handle, gpu_stream);
#else
    (void)gpu_handle; (void)gpu_stream;
    hseqr_host_detail::run_hseqr_lapack(n, ilo, ihi, wantt, wantz, initz, H, ldh, wr, wi, Z, ldz, info);
#endif
}

#ifndef HSEQR_STANDALONE_BENCH
inline bool rocsolver_hseqr_decode_job(rocblas_evect job, bool& want_schur)
{
    if(job == rocblas_evect_none)
    {
        want_schur = false;
        return true;
    }
    if(job == rocblas_evect_original)
    {
        want_schur = true;
        return true;
    }
    return false;
}

inline bool rocsolver_hseqr_decode_compz(rocblas_evect compz, bool& wantz, bool& initz)
{
    if(compz == rocblas_evect_none)
    {
        wantz = false;
        initz = false;
        return true;
    }
    if(compz == rocblas_evect_original)
    {
        wantz = true;
        initz = false;
        return true;
    }
    if(compz == rocblas_evect_tridiagonal)
    {
        wantz = true;
        initz = true;
        return true;
    }
    return false;
}

template <typename T, typename I, typename UH, typename UZ>
rocblas_status rocsolver_hseqr_argCheck(rocblas_handle handle,
                                        const rocblas_evect job,
                                        const rocblas_evect compz,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        UH H,
                                        const I ldh,
                                        T* wr,
                                        T* wi,
                                        UZ Z,
                                        const I ldz,
                                        rocblas_int* info,
                                        const I batch_count = 1)
{
    (void)handle;
    bool want_schur, wantz, initz;
    if(!rocsolver_hseqr_decode_job(job, want_schur))
        return rocblas_status_invalid_value;
    if(!rocsolver_hseqr_decode_compz(compz, wantz, initz))
        return rocblas_status_invalid_value;
    (void)want_schur;
    (void)initz;

    if(n < 0 || ldh < std::max(I(1), n) || batch_count < 0)
        return rocblas_status_invalid_size;

    if(n > 0)
    {
        if(ilo < 1 || ilo > std::max(I(1), n))
            return rocblas_status_invalid_size;
        if(ihi < std::min(ilo, n) || ihi > n)
            return rocblas_status_invalid_size;
    }

    if(wantz && ldz < std::max(I(1), n))
        return rocblas_status_invalid_size;
    if(!wantz && ldz < 1)
        return rocblas_status_invalid_size;

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    if(!info)
        return rocblas_status_invalid_pointer;
    if(!H || !wr || !wi)
        return rocblas_status_invalid_pointer;
    if(wantz && !Z)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void rocsolver_hseqr_getMemorySize(const I n, const I batch_count, size_t* size_work)
{
    (void)n;
    (void)batch_count;
    *size_work = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename UH, typename UZ>
rocblas_status rocsolver_hseqr_template(rocblas_handle handle,
                                        const rocblas_evect job,
                                        const rocblas_evect compz,
                                        const I n,
                                        const I ilo,
                                        const I ihi,
                                        UH H,
                                        const rocblas_stride shiftH,
                                        const I ldh,
                                        const rocblas_stride strideH,
                                        T* wr,
                                        const rocblas_stride strideWr,
                                        T* wi,
                                        const rocblas_stride strideWi,
                                        UZ Z,
                                        const rocblas_stride shiftZ,
                                        const I ldz,
                                        const rocblas_stride strideZ,
                                        rocblas_int* info,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("hseqr", "n:", n, "ilo:", ilo, "ihi:", ihi, "ldh:", ldh, "bc:", batch_count);

    bool want_schur, wantz, initz;
    rocsolver_hseqr_decode_job(job, want_schur);
    rocsolver_hseqr_decode_compz(compz, wantz, initz);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(batch_count == 0)
        return rocblas_status_success;

    rocblas_int blocksReset = (rocblas_int)((batch_count - 1) / BS1 + 1);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocksReset), dim3(BS1), 0, stream, info,
                            (rocblas_int)batch_count, 0);

    const rocblas_stride effStrideWr = strideWr ? strideWr : std::max(I(1), n);
    const rocblas_stride effStrideWi = strideWi ? strideWi : std::max(I(1), n);

    const size_t bc = size_t(batch_count);

    if(n == 0)
    {
        HIP_CHECK(hipStreamSynchronize(stream));
        return rocblas_status_success;
    }

    const size_t sizeH = size_t(ldh) * size_t(n);
    const size_t sizeZ = size_t(ldz) * size_t(n);

    std::vector<T> hH(bc * sizeH);
    std::vector<T> hZ(bc * sizeZ);
    std::vector<T> hWr(bc * size_t(n));
    std::vector<T> hWi(bc * size_t(n));
    std::vector<rocblas_int> hInfo(bc);

    if constexpr(BATCHED)
    {
        std::vector<T*> ptrsH(bc);
        std::vector<T*> ptrsZ(bc);
        HIP_CHECK(hipMemcpyAsync(ptrsH.data(), H, sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        if(wantz)
            HIP_CHECK(hipMemcpyAsync(ptrsZ.data(), Z, sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for(I b = 0; b < batch_count; b++)
        {
            HIP_CHECK(hipMemcpyAsync(hH.data() + size_t(b) * sizeH, ptrsH[size_t(b)] + shiftH,
                                     sizeof(T) * sizeH, hipMemcpyDeviceToHost, stream));
            if(wantz)
                HIP_CHECK(hipMemcpyAsync(hZ.data() + size_t(b) * sizeZ, ptrsZ[size_t(b)] + shiftZ,
                                         sizeof(T) * sizeZ, hipMemcpyDeviceToHost, stream));
        }
    }
    else
    {
        for(I b = 0; b < batch_count; b++)
        {
            HIP_CHECK(hipMemcpyAsync(hH.data() + size_t(b) * sizeH,
                                     (const T*)H + shiftH + b * strideH, sizeof(T) * sizeH,
                                     hipMemcpyDeviceToHost, stream));
            if(wantz)
                HIP_CHECK(hipMemcpyAsync(hZ.data() + size_t(b) * sizeZ,
                                         (const T*)Z + shiftZ + b * strideZ, sizeof(T) * sizeZ,
                                         hipMemcpyDeviceToHost, stream));
        }
    }

    for(I b = 0; b < batch_count; b++)
    {
        HIP_CHECK(hipMemcpyAsync(hWr.data() + size_t(b) * size_t(n), wr + b * effStrideWr,
                                 sizeof(T) * size_t(n), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(hWi.data() + size_t(b) * size_t(n), wi + b * effStrideWi,
                                 sizeof(T) * size_t(n), hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    auto run_batch = [&](I b) {
        T* pH = hH.data() + size_t(b) * sizeH;
        T* pZ = wantz ? (hZ.data() + size_t(b) * sizeZ) : nullptr;
        T* pWr = hWr.data() + size_t(b) * size_t(n);
        T* pWi = hWi.data() + size_t(b) * size_t(n);
        rocblas_int loc_info = 0;
        run_hseqr((rocblas_int)n, (rocblas_int)ilo, (rocblas_int)ihi, want_schur,
                  wantz, initz, pH, (rocblas_int)ldh, pWr, pWi, pZ, (rocblas_int)ldz,
                  loc_info);
        hInfo[size_t(b)] = loc_info;
    };

    (void)STRIDED;
#if defined(_OPENMP)
#pragma omp parallel for if(batch_count > 1)
#endif
    for(I b = 0; b < batch_count; b++)
    {
        run_batch(b);
    }

    if constexpr(BATCHED)
    {
        std::vector<T*> ptrsH(bc);
        std::vector<T*> ptrsZ(bc);
        HIP_CHECK(hipMemcpyAsync(ptrsH.data(), H, sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        if(wantz)
            HIP_CHECK(hipMemcpyAsync(ptrsZ.data(), Z, sizeof(T*) * bc, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for(I b = 0; b < batch_count; b++)
        {
            HIP_CHECK(hipMemcpyAsync(ptrsH[size_t(b)] + shiftH, hH.data() + size_t(b) * sizeH,
                                     sizeof(T) * sizeH, hipMemcpyHostToDevice, stream));
            if(wantz)
                HIP_CHECK(hipMemcpyAsync(ptrsZ[size_t(b)] + shiftZ, hZ.data() + size_t(b) * sizeZ,
                                         sizeof(T) * sizeZ, hipMemcpyHostToDevice, stream));
        }
    }
    else
    {
        for(I b = 0; b < batch_count; b++)
        {
            HIP_CHECK(hipMemcpyAsync((T*)H + shiftH + b * strideH, hH.data() + size_t(b) * sizeH,
                                     sizeof(T) * sizeH, hipMemcpyHostToDevice, stream));
            if(wantz)
                HIP_CHECK(hipMemcpyAsync((T*)Z + shiftZ + b * strideZ, hZ.data() + size_t(b) * sizeZ,
                                         sizeof(T) * sizeZ, hipMemcpyHostToDevice, stream));
        }
    }

    for(I b = 0; b < batch_count; b++)
    {
        HIP_CHECK(hipMemcpyAsync(wr + b * effStrideWr, hWr.data() + size_t(b) * size_t(n),
                                 sizeof(T) * size_t(n), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(wi + b * effStrideWi, hWi.data() + size_t(b) * size_t(n),
                                 sizeof(T) * size_t(n), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(info + b, &hInfo[size_t(b)], sizeof(rocblas_int),
                                 hipMemcpyHostToDevice, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    return rocblas_status_success;
}
#endif // !HSEQR_STANDALONE_BENCH

ROCSOLVER_END_NAMESPACE
