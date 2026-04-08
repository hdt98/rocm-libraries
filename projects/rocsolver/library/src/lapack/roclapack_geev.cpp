/* **************************************************************************
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
 * *************************************************************************/

#include "roclapack_geev.hpp"

#include <climits>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>

ROCSOLVER_BEGIN_NAMESPACE

// ============================================================================
// Native complex GEEV pipeline: GEBAL → GEHRD → UNGHR → HSEQR → TREVC → GEBAK
// Replaces the previous monolithic LAPACK cgeev_/zgeev_ delegation with
// individual component calls, enabling future GPU acceleration of each stage
// and consistent batched/strided-batched support.
// ============================================================================

namespace complex_geev_detail
{

// ---- Type-dispatched host wrappers for complex LAPACK component routines ----

// GEBAL (balance)
inline void host_gebal(int n, rocblas_float_complex* A, int lda,
                       int& ilo, int& ihi, float* scale, int& info)
{
    cgebal_("B", &n, A, &lda, &ilo, &ihi, scale, &info);
}
inline void host_gebal(int n, rocblas_double_complex* A, int lda,
                       int& ilo, int& ihi, double* scale, int& info)
{
    zgebal_("B", &n, A, &lda, &ilo, &ihi, scale, &info);
}

// GEHRD (Hessenberg reduction)
inline void host_gehrd(int n, int ilo, int ihi, rocblas_float_complex* A, int lda,
                       rocblas_float_complex* tau, rocblas_float_complex* work, int lwork, int& info)
{
    cgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_gehrd(int n, int ilo, int ihi, rocblas_double_complex* A, int lda,
                       rocblas_double_complex* tau, rocblas_double_complex* work, int lwork, int& info)
{
    zgehrd_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

// UNGHR (generate unitary Q from Hessenberg reduction)
inline void host_unghr(int n, int ilo, int ihi, rocblas_float_complex* A, int lda,
                       const rocblas_float_complex* tau, rocblas_float_complex* work, int lwork, int& info)
{
    cunghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}
inline void host_unghr(int n, int ilo, int ihi, rocblas_double_complex* A, int lda,
                       const rocblas_double_complex* tau, rocblas_double_complex* work, int lwork, int& info)
{
    zunghr_(&n, &ilo, &ihi, A, &lda, tau, work, &lwork, &info);
}

// HSEQR (Schur form via QR iteration)
inline void host_hseqr(const char* job, const char* compz, int n, int ilo, int ihi,
                       rocblas_float_complex* H, int ldh, rocblas_float_complex* W,
                       rocblas_float_complex* Z, int ldz, rocblas_float_complex* work,
                       int lwork, int& info)
{
    chseqr_(job, compz, &n, &ilo, &ihi, H, &ldh, W, Z, &ldz, work, &lwork, &info);
}
inline void host_hseqr(const char* job, const char* compz, int n, int ilo, int ihi,
                       rocblas_double_complex* H, int ldh, rocblas_double_complex* W,
                       rocblas_double_complex* Z, int ldz, rocblas_double_complex* work,
                       int lwork, int& info)
{
    zhseqr_(job, compz, &n, &ilo, &ihi, H, &ldh, W, Z, &ldz, work, &lwork, &info);
}

// TREVC3 (blocked BLAS-3 eigenvector computation)
inline void host_trevc3(const char* side, int n, rocblas_float_complex* T_mat, int ldt,
                        rocblas_float_complex* VL, int ldvl, rocblas_float_complex* VR, int ldvr,
                        int mm, int& m, rocblas_float_complex* work, int lwork,
                        float* rwork, int& info)
{
    ctrevc3_(side, "B", nullptr, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr,
             &mm, &m, work, &lwork, rwork, &info);
}
inline void host_trevc3(const char* side, int n, rocblas_double_complex* T_mat, int ldt,
                        rocblas_double_complex* VL, int ldvl, rocblas_double_complex* VR, int ldvr,
                        int mm, int& m, rocblas_double_complex* work, int lwork,
                        double* rwork, int& info)
{
    ztrevc3_(side, "B", nullptr, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr,
             &mm, &m, work, &lwork, rwork, &info);
}

// TREVC (unblocked eigenvector computation, for small N)
inline void host_trevc(const char* side, int n, rocblas_float_complex* T_mat, int ldt,
                       rocblas_float_complex* VL, int ldvl, rocblas_float_complex* VR, int ldvr,
                       int mm, int& m, rocblas_float_complex* work, float* rwork, int& info)
{
    ctrevc_(side, "B", nullptr, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr,
            &mm, &m, work, rwork, &info);
}
inline void host_trevc(const char* side, int n, rocblas_double_complex* T_mat, int ldt,
                       rocblas_double_complex* VL, int ldvl, rocblas_double_complex* VR, int ldvr,
                       int mm, int& m, rocblas_double_complex* work, double* rwork, int& info)
{
    ztrevc_(side, "B", nullptr, &n, T_mat, &ldt, VL, &ldvl, VR, &ldvr,
            &mm, &m, work, rwork, &info);
}

// GEBAK (back-transform eigenvectors)
inline void host_gebak(const char* side, int n, int ilo, int ihi,
                       const float* scale, int m, rocblas_float_complex* V, int ldv, int& info)
{
    cgebak_("B", side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}
inline void host_gebak(const char* side, int n, int ilo, int ihi,
                       const double* scale, int m, rocblas_double_complex* V, int ldv, int& info)
{
    zgebak_("B", side, &n, &ilo, &ihi, scale, &m, V, &ldv, &info);
}

// Complex eigenvector normalization: Euclidean norm 1, largest component real.
// Matches LAPACK zgeev.f and MAGMA zgeev.cpp exactly.
template <typename Cpx>
void normalize_complex_eigvecs(int n, Cpx* V, int ldv)
{
    using Real = decltype(std::abs(Cpx{}));
    for(int j = 0; j < n; ++j)
    {
        // 1. Scale column to Euclidean norm 1
        Real nrm = 0;
        for(int i = 0; i < n; ++i)
        {
            Real a = std::abs(V[i + static_cast<size_t>(j) * ldv]);
            nrm += a * a;
        }
        nrm = std::sqrt(nrm);
        if(nrm > 0)
        {
            Real scl = Real(1) / nrm;
            for(int i = 0; i < n; ++i)
                V[i + static_cast<size_t>(j) * ldv] *= Cpx(scl);
        }

        // 2. Find element with largest squared modulus
        int k = 0;
        Real maxmod2 = 0;
        for(int i = 0; i < n; ++i)
        {
            Real re = std::real(V[i + static_cast<size_t>(j) * ldv]);
            Real im = std::imag(V[i + static_cast<size_t>(j) * ldv]);
            Real mod2 = re * re + im * im;
            if(mod2 > maxmod2)
            {
                maxmod2 = mod2;
                k = i;
            }
        }

        // 3. Rotate so that V(k,j) is real and positive
        if(maxmod2 > 0)
        {
            Cpx vk = V[k + static_cast<size_t>(j) * ldv];
            Cpx phase = std::conj(vk) / Cpx(std::sqrt(maxmod2));
            for(int i = 0; i < n; ++i)
                V[i + static_cast<size_t>(j) * ldv] *= phase;
            // Ensure imaginary part of V(k,j) is exactly zero
            V[k + static_cast<size_t>(j) * ldv] = Cpx(std::real(V[k + static_cast<size_t>(j) * ldv]), Real(0));
        }
    }
}

// Optimal workspace query helper
template <typename Cpx>
int query_lwork(Cpx* wq)
{
    using StdCpx = std::complex<decltype(std::abs(Cpx{}))>;
    return std::max(1, static_cast<int>(
        std::real(*reinterpret_cast<const StdCpx*>(wq))));
}

} // namespace complex_geev_detail

// ctrevc3_/ztrevc3_ require extra RWORK workspace and Fortran hidden-length
// args that can cause ABI issues. Use ctrevc_/ztrevc_ (unblocked) for safety;
// the real TREVC3 path handles BLAS-3 blocking for the real case.
static constexpr int COMPLEX_TREVC3_THRESHOLD = 100000;

// Native complex GEEV: GEBAL → GEHRD → UNGHR → HSEQR → TREVC → GEBAK
template <typename Cpx, typename Real>
rocblas_status rocsolver_geev_complex_impl(
    rocblas_handle handle,
    const rocblas_evect jobvl,
    const rocblas_evect jobvr,
    const rocblas_int n,
    Cpx* A,
    const rocblas_int lda,
    Cpx* W,
    Cpx* VL,
    const rocblas_int ldvl,
    Cpx* VR,
    const rocblas_int ldvr,
    rocblas_int* info,
    void (*)(char*, char*, const rocblas_int*, Cpx*, const rocblas_int*, Cpx*,
             Cpx*, const rocblas_int*, Cpx*, const rocblas_int*, Cpx*,
             const rocblas_int*, Real*, rocblas_int*, std::size_t, std::size_t))
{
    using namespace complex_geev_detail;
    using T = Cpx;

    ROCSOLVER_ENTER_TOP("geev", "--jobvl", jobvl, "--jobvr", jobvr, "-n", n, "--lda", lda,
                        "--ldvl", ldvl, "--ldvr", ldvr);

    if(!handle)
        return rocblas_status_invalid_handle;

    rocblas_status st = rocsolver_geev_argCheck<Cpx, rocblas_int>(
        handle, jobvl, jobvr, n, A, lda, W, (Cpx*)nullptr, VL, ldvl, VR, ldvr, info);
    if(st != rocblas_status_continue)
        return st;

    const bool wantvl = (jobvl == rocblas_evect_original);
    const bool wantvr = (jobvr == rocblas_evect_original);
    const bool wantvec = wantvl || wantvr;

    size_t size_work;
    rocsolver_geev_getMemorySize<false, false, float, rocblas_int>(n, 1, &size_work);
    (void)size_work;

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, 0);

    hipStream_t stream{};
    rocblas_get_stream(handle, &stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    if(n == 0)
    {
        rocblas_int zero = 0;
        HIP_CHECK(hipMemcpy(info, &zero, sizeof(rocblas_int), hipMemcpyHostToDevice));
        return rocblas_status_success;
    }

    const int n_i = static_cast<int>(n);
    const int lda_i = static_cast<int>(lda);
    const size_t n_sz = static_cast<size_t>(n);
    const size_t a_elems = static_cast<size_t>(lda) * n_sz;
    const size_t vl_elems = wantvl ? static_cast<size_t>(ldvl) * n_sz : 0;
    const size_t vr_elems = wantvr ? static_cast<size_t>(ldvr) * n_sz : 0;

    std::vector<Cpx> hA(a_elems);
    HIP_CHECK(hipMemcpy(hA.data(), A, a_elems * sizeof(Cpx), hipMemcpyDeviceToHost));

    // 0. Scale A if max element outside safe range [smlnum, bignum]
    const Real eps_mach = std::numeric_limits<Real>::epsilon();
    const Real safe_min = std::numeric_limits<Real>::min();
    Real smlnum_scl = std::sqrt(safe_min) / eps_mach;
    Real bignum_scl = Real(1) / smlnum_scl;
    Real anrm = Real(0);
    for(int j = 0; j < n_i; ++j)
        for(int i = 0; i < n_i; ++i)
            anrm = std::max(anrm, std::abs(hA[i + j * static_cast<size_t>(lda_i)]));
    bool scalea = false;
    Real cscale = Real(1);
    if(anrm > Real(0) && anrm < smlnum_scl)
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
        Real ratio = cscale / anrm;
        for(int j = 0; j < n_i; ++j)
            for(int i = 0; i < n_i; ++i)
                hA[i + j * static_cast<size_t>(lda_i)] *= Cpx(ratio);
    }

    // 1. GEBAL: balance the matrix
    int ilo = 1, ihi = n_i, info_i = 0;
    std::vector<Real> scale(n_sz);
    host_gebal(n_i, hA.data(), lda_i, ilo, ihi, scale.data(), info_i);
    if(info_i != 0)
    {
        rocblas_int info_h = info_i;
        HIP_CHECK(hipMemcpy(info, &info_h, sizeof(rocblas_int), hipMemcpyHostToDevice));
        return rocblas_status_success;
    }

    // 2. GEHRD: reduce to upper Hessenberg form
    std::vector<Cpx> tau(std::max(n_sz, size_t(1)));
    {
        int lwork = -1;
        Cpx wq;
        host_gehrd(n_i, ilo, ihi, hA.data(), lda_i, tau.data(), &wq, lwork, info_i);
        lwork = query_lwork(&wq);
        std::vector<Cpx> work(lwork);
        host_gehrd(n_i, ilo, ihi, hA.data(), lda_i, tau.data(), work.data(), lwork, info_i);
        if(info_i != 0)
        {
            rocblas_int info_h = info_i;
            HIP_CHECK(hipMemcpy(info, &info_h, sizeof(rocblas_int), hipMemcpyHostToDevice));
            return rocblas_status_success;
        }
    }

    // 3. UNGHR: generate unitary Q (only when eigenvectors requested)
    std::vector<Cpx> hZ;
    if(wantvec)
    {
        hZ.resize(n_sz * n_sz);
        for(size_t j = 0; j < n_sz; ++j)
            for(size_t i = 0; i < n_sz; ++i)
                hZ[i + j * n_sz] = hA[i + j * static_cast<size_t>(lda_i)];

        int lwork = -1;
        Cpx wq;
        host_unghr(n_i, ilo, ihi, hZ.data(), n_i, tau.data(), &wq, lwork, info_i);
        lwork = query_lwork(&wq);
        std::vector<Cpx> work(lwork);
        host_unghr(n_i, ilo, ihi, hZ.data(), n_i, tau.data(), work.data(), lwork, info_i);
        if(info_i != 0)
        {
            rocblas_int info_h = info_i;
            HIP_CHECK(hipMemcpy(info, &info_h, sizeof(rocblas_int), hipMemcpyHostToDevice));
            return rocblas_status_success;
        }
    }

    // 4. HSEQR: QR iteration to get Schur form and eigenvalues
    std::vector<Cpx> hW(n_sz);
    {
        const char* job = "S";
        const char* compz = wantvec ? "V" : "N";
        int lwork = -1;
        Cpx wq;
        host_hseqr(job, compz, n_i, ilo, ihi, hA.data(), lda_i, hW.data(),
                   wantvec ? hZ.data() : nullptr, wantvec ? n_i : 1, &wq, lwork, info_i);
        lwork = query_lwork(&wq);
        std::vector<Cpx> work(lwork);
        host_hseqr(job, compz, n_i, ilo, ihi, hA.data(), lda_i, hW.data(),
                   wantvec ? hZ.data() : nullptr, wantvec ? n_i : 1,
                   work.data(), lwork, info_i);
        if(info_i != 0)
        {
            if(scalea)
            {
                Real inv_ratio = anrm / cscale;
                int nval = n_i - info_i;
                for(int k = info_i; k < n_i; ++k)
                    hW[k] *= Cpx(inv_ratio);
                for(int k = 0; k < ilo - 1; ++k)
                    hW[k] *= Cpx(inv_ratio);
            }
            HIP_CHECK(hipMemcpyAsync(W, hW.data(), n_sz * sizeof(Cpx), hipMemcpyHostToDevice, stream));
            rocblas_int info_h = info_i;
            HIP_CHECK(hipMemcpyAsync(info, &info_h, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            return rocblas_status_success;
        }
    }

    // 4b. Undo eigenvalue scaling (all eigenvalues converged)
    if(scalea)
    {
        Real inv_ratio = anrm / cscale;
        for(int k = 0; k < n_i; ++k)
            hW[k] *= Cpx(inv_ratio);
    }

    // 5. TREVC + GEBAK for right eigenvectors
    if(wantvr)
    {
        std::vector<Cpx> hVR(vr_elems);
        for(size_t j = 0; j < n_sz; ++j)
            for(size_t i = 0; i < n_sz; ++i)
                hVR[i + j * static_cast<size_t>(ldvr)] = hZ[i + j * n_sz];

        int m = 0;
        if(n_i >= COMPLEX_TREVC3_THRESHOLD)
        {
            int lwork = -1;
            Cpx wq;
            Real rwork_query;
            host_trevc3("R", n_i, hA.data(), lda_i, nullptr, 1, hVR.data(), static_cast<int>(ldvr),
                        n_i, m, &wq, lwork, &rwork_query, info_i);
            lwork = query_lwork(&wq);
            std::vector<Cpx> work(lwork);
            std::vector<Real> rwork(std::max(size_t(1), static_cast<size_t>(lwork)));
            info_i = 0; m = 0;
            host_trevc3("R", n_i, hA.data(), lda_i, nullptr, 1, hVR.data(), static_cast<int>(ldvr),
                        n_i, m, work.data(), lwork, rwork.data(), info_i);
        }
        else
        {
            std::vector<Cpx> work(2 * n_sz);
            std::vector<Real> rwork(n_sz);
            host_trevc("R", n_i, hA.data(), lda_i, nullptr, 1, hVR.data(), static_cast<int>(ldvr),
                       n_i, m, work.data(), rwork.data(), info_i);
        }

        info_i = 0;
        host_gebak("R", n_i, ilo, ihi, scale.data(), n_i, hVR.data(), static_cast<int>(ldvr), info_i);
        normalize_complex_eigvecs(n_i, hVR.data(), static_cast<int>(ldvr));

        HIP_CHECK(hipMemcpyAsync(VR, hVR.data(), vr_elems * sizeof(Cpx), hipMemcpyHostToDevice, stream));
    }

    // 6. TREVC + GEBAK for left eigenvectors
    if(wantvl)
    {
        std::vector<Cpx> hVL(vl_elems);
        for(size_t j = 0; j < n_sz; ++j)
            for(size_t i = 0; i < n_sz; ++i)
                hVL[i + j * static_cast<size_t>(ldvl)] = hZ[i + j * n_sz];

        int m = 0;
        if(n_i >= COMPLEX_TREVC3_THRESHOLD)
        {
            int lwork = -1;
            Cpx wq;
            Real rwork_query;
            host_trevc3("L", n_i, hA.data(), lda_i, hVL.data(), static_cast<int>(ldvl), nullptr, 1,
                        n_i, m, &wq, lwork, &rwork_query, info_i);
            lwork = query_lwork(&wq);
            std::vector<Cpx> work(lwork);
            std::vector<Real> rwork(std::max(size_t(1), static_cast<size_t>(lwork)));
            info_i = 0; m = 0;
            host_trevc3("L", n_i, hA.data(), lda_i, hVL.data(), static_cast<int>(ldvl), nullptr, 1,
                        n_i, m, work.data(), lwork, rwork.data(), info_i);
        }
        else
        {
            std::vector<Cpx> work(2 * n_sz);
            std::vector<Real> rwork(n_sz);
            host_trevc("L", n_i, hA.data(), lda_i, hVL.data(), static_cast<int>(ldvl), nullptr, 1,
                       n_i, m, work.data(), rwork.data(), info_i);
        }

        info_i = 0;
        host_gebak("L", n_i, ilo, ihi, scale.data(), n_i, hVL.data(), static_cast<int>(ldvl), info_i);
        normalize_complex_eigvecs(n_i, hVL.data(), static_cast<int>(ldvl));

        HIP_CHECK(hipMemcpyAsync(VL, hVL.data(), vl_elems * sizeof(Cpx), hipMemcpyHostToDevice, stream));
    }

    // Upload eigenvalues and info
    HIP_CHECK(hipMemcpyAsync(W, hW.data(), n_sz * sizeof(Cpx), hipMemcpyHostToDevice, stream));
    rocblas_int info_h = 0;
    HIP_CHECK(hipMemcpyAsync(info, &info_h, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    return rocblas_status_success;
}

template <typename T>
rocblas_status rocsolver_geev_impl(rocblas_handle handle,
                                   const rocblas_evect jobvl,
                                   const rocblas_evect jobvr,
                                   const rocblas_int n,
                                   T* A,
                                   const rocblas_int lda,
                                   T* wr,
                                   T* wi,
                                   T* VL,
                                   const rocblas_int ldvl,
                                   T* VR,
                                   const rocblas_int ldvr,
                                   rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("geev", "--jobvl", jobvl, "--jobvr", jobvr,
                        "-n", n, "--lda", lda, "--ldvl", ldvl, "--ldvr", ldvr);

    if(!handle)
        return rocblas_status_invalid_handle;

    rocblas_status st = rocsolver_geev_argCheck<T, rocblas_int>(
        handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, info);
    if(st != rocblas_status_continue)
        return st;

    size_t size_work;
    rocsolver_geev_getMemorySize<false, false, T, rocblas_int>(n, 1, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    return rocsolver_geev_template<false, false, T, rocblas_int>(
        handle, jobvl, jobvr, n,
        A, rocblas_stride(0), lda, rocblas_stride(0),
        wr, rocblas_stride(0), wi, rocblas_stride(0),
        VL, rocblas_stride(0), ldvl, rocblas_stride(0),
        VR, rocblas_stride(0), ldvr, rocblas_stride(0),
        info, rocblas_int(1));
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper — matches LAPACK DGEEV/SGEEV signature
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               float* A,
                               const rocblas_int lda,
                               float* wr,
                               float* wi,
                               float* VL,
                               const rocblas_int ldvl,
                               float* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<float>(
        handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, info);
}

rocblas_status rocsolver_dgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               double* A,
                               const rocblas_int lda,
                               double* wr,
                               double* wi,
                               double* VL,
                               const rocblas_int ldvl,
                               double* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_impl<double>(
        handle, jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, info);
}

rocblas_status rocsolver_cgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_float_complex* A,
                               const rocblas_int lda,
                               rocblas_float_complex* W,
                               rocblas_float_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_float_complex* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_complex_impl<rocblas_float_complex, float>(
        handle, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr, info, nullptr);
}

rocblas_status rocsolver_zgeev(rocblas_handle handle,
                               const rocblas_evect jobvl,
                               const rocblas_evect jobvr,
                               const rocblas_int n,
                               rocblas_double_complex* A,
                               const rocblas_int lda,
                               rocblas_double_complex* W,
                               rocblas_double_complex* VL,
                               const rocblas_int ldvl,
                               rocblas_double_complex* VR,
                               const rocblas_int ldvr,
                               rocblas_int* info)
{
    return rocsolver::rocsolver_geev_complex_impl<rocblas_double_complex, double>(
        handle, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr, info, nullptr);
}

rocblas_status rocsolver_sgeev_64(rocblas_handle handle,
                                  const rocblas_evect jobvl,
                                  const rocblas_evect jobvr,
                                  const int64_t n,
                                  float* A,
                                  const int64_t lda,
                                  float* wr,
                                  float* wi,
                                  float* VL,
                                  const int64_t ldvl,
                                  float* VR,
                                  const int64_t ldvr,
                                  rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    if(n > INT32_MAX || lda > INT32_MAX || ldvl > INT32_MAX || ldvr > INT32_MAX)
        return rocblas_status_invalid_size;
    return rocsolver::rocsolver_geev_impl<float>(
        handle, jobvl, jobvr, (rocblas_int)n, A, (rocblas_int)lda,
        wr, wi, VL, (rocblas_int)ldvl, VR, (rocblas_int)ldvr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dgeev_64(rocblas_handle handle,
                                  const rocblas_evect jobvl,
                                  const rocblas_evect jobvr,
                                  const int64_t n,
                                  double* A,
                                  const int64_t lda,
                                  double* wr,
                                  double* wi,
                                  double* VL,
                                  const int64_t ldvl,
                                  double* VR,
                                  const int64_t ldvr,
                                  rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    if(n > INT32_MAX || lda > INT32_MAX || ldvl > INT32_MAX || ldvr > INT32_MAX)
        return rocblas_status_invalid_size;
    return rocsolver::rocsolver_geev_impl<double>(
        handle, jobvl, jobvr, (rocblas_int)n, A, (rocblas_int)lda,
        wr, wi, VL, (rocblas_int)ldvl, VR, (rocblas_int)ldvr, info);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_cgeev_64(rocblas_handle handle,
                                  const rocblas_evect jobvl,
                                  const rocblas_evect jobvr,
                                  const int64_t n,
                                  rocblas_float_complex* A,
                                  const int64_t lda,
                                  rocblas_float_complex* W,
                                  rocblas_float_complex* VL,
                                  const int64_t ldvl,
                                  rocblas_float_complex* VR,
                                  const int64_t ldvr,
                                  rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    if(n > INT32_MAX || lda > INT32_MAX || ldvl > INT32_MAX || ldvr > INT32_MAX)
        return rocblas_status_invalid_size;
    return rocsolver::rocsolver_geev_complex_impl<rocblas_float_complex, float>(
        handle, jobvl, jobvr, static_cast<rocblas_int>(n), A, static_cast<rocblas_int>(lda), W, VL,
        static_cast<rocblas_int>(ldvl), VR, static_cast<rocblas_int>(ldvr), info, nullptr);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zgeev_64(rocblas_handle handle,
                                  const rocblas_evect jobvl,
                                  const rocblas_evect jobvr,
                                  const int64_t n,
                                  rocblas_double_complex* A,
                                  const int64_t lda,
                                  rocblas_double_complex* W,
                                  rocblas_double_complex* VL,
                                  const int64_t ldvl,
                                  rocblas_double_complex* VR,
                                  const int64_t ldvr,
                                  rocblas_int* info)
{
#ifdef HAVE_ROCBLAS_64
    if(n > INT32_MAX || lda > INT32_MAX || ldvl > INT32_MAX || ldvr > INT32_MAX)
        return rocblas_status_invalid_size;
    return rocsolver::rocsolver_geev_complex_impl<rocblas_double_complex, double>(
        handle, jobvl, jobvr, static_cast<rocblas_int>(n), A, static_cast<rocblas_int>(lda), W, VL,
        static_cast<rocblas_int>(ldvl), VR, static_cast<rocblas_int>(ldvr), info, nullptr);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern "C"
