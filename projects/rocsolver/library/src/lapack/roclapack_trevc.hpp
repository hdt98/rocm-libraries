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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

namespace trevc_detail
{
    template <typename T>
    inline T safe_den(T x, T smin)
    {
        if(std::abs(x) < smin)
            return (x >= T(0) ? smin : -smin);
        return x;
    }

    /** Solve 4x4 Ax=b with partial pivoting; A stored row-major 4x4. */
    template <typename T>
    inline void solve4x4(T A[4][4], T b[4])
    {
        for(int k = 0; k < 4; ++k)
        {
            int piv = k;
            T amax = std::abs(A[k][k]);
            for(int r = k + 1; r < 4; ++r)
            {
                const T t = std::abs(A[r][k]);
                if(t > amax)
                {
                    amax = t;
                    piv = r;
                }
            }
            if(piv != k)
            {
                for(int c = k; c < 4; ++c)
                    std::swap(A[k][c], A[piv][c]);
                std::swap(b[k], b[piv]);
            }
            T akk = A[k][k];
            const T tol = std::numeric_limits<T>::min();
            if(std::abs(akk) < tol)
                akk = (akk >= T(0) ? tol : -tol);
            const T inv = T(1) / akk;
            for(int r = k + 1; r < 4; ++r)
            {
                const T f = A[r][k] * inv;
                b[r] -= f * b[k];
                for(int c = k + 1; c < 4; ++c)
                    A[r][c] -= f * A[k][c];
            }
        }
        for(int r = 3; r >= 0; --r)
        {
            T sum = b[r];
            for(int c = r + 1; c < 4; ++c)
                sum -= A[r][c] * b[c];
            const T tol = std::numeric_limits<T>::min();
            b[r] = sum / safe_den(A[r][r], tol);
        }
    }

    template <typename T, typename I>
    inline void trevc_dlaln2_1x1(bool /*onebyone*/,
                                 I nw,
                                 T smin,
                                 const T* Tblk,
                                 I /*ldt*/,
                                 T wr,
                                 T wi,
                                 T rhsr,
                                 T rhsi,
                                 T x[2][2],
                                 T& scale,
                                 T& xnorm)
    {
        (void)Tblk;
        scale = T(1);
        if(nw == 1)
        {
            const T na = safe_den(Tblk[0] - wr, smin);
            x[0][0] = rhsr / na;
            xnorm = std::abs(x[0][0]);
        }
        else
        {
            const T na = Tblk[0] - wr;
            const T det = na * na + wi * wi;
            const T sdet = std::max(std::abs(det), smin * smin);
            x[0][0] = (na * rhsr - wi * rhsi) / sdet;
            x[0][1] = (na * rhsi + wi * rhsr) / sdet;
            xnorm = std::abs(x[0][0]) + std::abs(x[0][1]);
        }
    }

    template <typename T, typename I>
    inline void trevc_dlaln2_2x2(I nw,
                                T smin,
                                const T* T22,
                                I /*ldt*/,
                                T wr,
                                T wi,
                                const T rhsr[2],
                                const T rhsi[2],
                                T x[2][2],
                                T& scale,
                                T& xnorm)
    {
        scale = T(1);
        const T a = T22[0];
        const T b = T22[1];
        const T c = T22[2];
        const T d = T22[3];

        if(nw == 1)
        {
            const T a11 = a - wr;
            const T a12 = b;
            const T a21 = c;
            const T a22 = d - wr;
            T det = a11 * a22 - a12 * a21;
            det = safe_den(det, smin * smin);
            x[0][0] = (a22 * rhsr[0] - a12 * rhsr[1]) / det;
            x[1][0] = (a11 * rhsr[1] - a21 * rhsr[0]) / det;
            xnorm = std::abs(x[0][0]) + std::abs(x[1][0]);
        }
        else
        {
            T M[4][4];
            T br[4];
            M[0][0] = a - wr;
            M[0][1] = wi;
            M[0][2] = b;
            M[0][3] = T(0);
            M[1][0] = -wi;
            M[1][1] = a - wr;
            M[1][2] = T(0);
            M[1][3] = b;
            M[2][0] = c;
            M[2][1] = T(0);
            M[2][2] = d - wr;
            M[2][3] = wi;
            M[3][0] = T(0);
            M[3][1] = c;
            M[3][2] = -wi;
            M[3][3] = d - wr;
            br[0] = rhsr[0];
            br[1] = rhsi[0];
            br[2] = rhsr[1];
            br[3] = rhsi[1];
            solve4x4(M, br);
            x[0][0] = br[0];
            x[0][1] = br[1];
            x[1][0] = br[2];
            x[1][1] = br[3];
            xnorm = std::abs(x[0][0]) + std::abs(x[0][1]) + std::abs(x[1][0]) + std::abs(x[1][1]);
        }
    }
} // namespace trevc_detail

/**
 * DTREVC with SIDE='R', HOWMNY='B': right eigenvectors with Schur-vector back-transform.
 * T is upper quasi-triangular (unchanged). VR input = Schur vectors Z; output = eigenvectors.
 */
template <typename T, typename I>
void run_trevc_right_howmny_b(const I n, const T* TT, const I ldt, T* VR, const I ldvr)
{
    using namespace trevc_detail;

    auto t = [&](I i, I j) -> const T& { return TT[i + j * ldt]; };
    auto vr = [&](I i, I j) -> T& { return VR[i + j * ldvr]; };

    const T ulp = std::numeric_limits<T>::epsilon();
    const T unfl = std::numeric_limits<T>::min();
    const T smlnum = std::max(unfl * static_cast<T>(n) / ulp, ulp);
    const T bignum = (T(1) - ulp) / smlnum;

    std::vector<T> work(static_cast<size_t>(3 * static_cast<size_t>(n)));
    T* const wrk = work.data();
    T* const wim = wrk + static_cast<size_t>(n);
    T* const coln = wim + static_cast<size_t>(n);

    for(I j = 0; j < n; ++j)
    {
        T s = T(0);
        for(I i = 0; i < j; ++i)
            s += std::abs(t(i, j));
        coln[j] = s;
    }

    I ip = 0;

    for(I ki = n - 1; ki >= 0; --ki)
    {
        bool skip_body = false;
        if(ip == 1)
            skip_body = true;
        else
        {
            if(ki > 0 && t(ki, ki - 1) != T(0))
                ip = -1;
        }

        if(!skip_body)
        {
            const T wr_val = t(ki, ki);
            T wi_val = T(0);
            if(ip != 0)
                wi_val = std::sqrt(std::abs(t(ki, ki - 1) * t(ki - 1, ki)));
            const T smin = std::max(ulp * (std::abs(wr_val) + std::abs(wi_val)), smlnum);

            if(ip == 0)
            {
                const I col = ki;
                for(I k = 0; k <= col - 1; ++k)
                    wrk[k] = -t(k, col);

                I jnxt = col - 1;
                for(I j = col - 1; j >= 0; --j)
                {
                    if(j > jnxt)
                        continue;
                    I j1 = j;
                    I j2 = j;
                    jnxt = j - 1;
                    if(j > 0 && t(j, j - 1) != T(0))
                    {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }

                    T x[2][2];
                    T scale;
                    T xnorm;

                    if(j1 == j2)
                    {
                        trevc_dlaln2_1x1<T, I>(true,
                                                1,
                                                smin,
                                                &t(j, j),
                                                ldt,
                                                wr_val,
                                                T(0),
                                                wrk[j],
                                                T(0),
                                                x,
                                                scale,
                                                xnorm);
                        if(xnorm > T(1))
                        {
                            if(coln[static_cast<size_t>(j)] > bignum / xnorm)
                            {
                                x[0][0] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = 0; kk <= col - 1; ++kk)
                                wrk[kk] *= scale;
                        }
                        wrk[j] = x[0][0];
                        for(I i = 0; i < j; ++i)
                            wrk[i] -= x[0][0] * t(i, j);
                    }
                    else
                    {
                        T t22[4] = {t(j1, j1), t(j1, j2), t(j2, j1), t(j2, j2)};
                        const T rhsr[2] = {wrk[j1], wrk[j2]};
                        trevc_dlaln2_2x2<T, I>(1,
                                              smin,
                                              t22,
                                              2,
                                              wr_val,
                                              T(0),
                                              rhsr,
                                              rhsr,
                                              x,
                                              scale,
                                              xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta
                                = std::max(coln[static_cast<size_t>(j1)], coln[static_cast<size_t>(j2)]);
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[1][0] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = 0; kk <= col - 1; ++kk)
                                wrk[kk] *= scale;
                        }
                        wrk[j1] = x[0][0];
                        wrk[j2] = x[1][0];
                        for(I i = 0; i < j1; ++i)
                        {
                            wrk[i] -= x[0][0] * t(i, j1);
                            wrk[i] -= x[1][0] * t(i, j2);
                        }
                    }
                }

                for(I i = 0; i < n; ++i)
                {
                    T sum = vr(i, col);
                    for(I j = 0; j <= col - 1; ++j)
                        sum += vr(i, j) * wrk[j];
                    vr(i, col) = sum;
                }
                {
                    T amax = T(0);
                    for(I i = 0; i < n; ++i)
                        amax = std::max(amax, std::abs(vr(i, col)));
                    if(amax > T(0))
                    {
                        const T scl = T(1) / amax;
                        for(I i = 0; i < n; ++i)
                            vr(i, col) *= scl;
                    }
                }
            }
            else
            {
                const I k2 = ki;
                const I k1 = ki - 1;

                if(std::abs(t(k1, k2)) >= std::abs(t(k2, k1)))
                {
                    wrk[k1] = T(1);
                    wim[k2] = wi_val / safe_den(t(k1, k2), smin);
                }
                else
                {
                    wrk[k1] = -wi_val / safe_den(t(k2, k1), smin);
                    wim[k2] = T(1);
                }
                wrk[k2] = T(0);
                wim[k1] = T(0);

                for(I k = 0; k < k1; ++k)
                {
                    wrk[k] = -wrk[k1] * t(k, k1);
                    wim[k] = -wim[k2] * t(k, k2);
                }

                I jnxt = k1 - 1;
                for(I j = k1 - 1; j >= 0; --j)
                {
                    if(j > jnxt)
                        continue;
                    I j1 = j;
                    I j2 = j;
                    jnxt = j - 1;
                    if(j > 0 && t(j, j - 1) != T(0))
                    {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }

                    T x[2][2];
                    T scale;
                    T xnorm;

                    if(j1 == j2)
                    {
                        trevc_dlaln2_1x1<T, I>(true,
                                                2,
                                                smin,
                                                &t(j, j),
                                                ldt,
                                                wr_val,
                                                wi_val,
                                                wrk[j],
                                                wim[j],
                                                x,
                                                scale,
                                                xnorm);
                        if(xnorm > T(1))
                        {
                            if(coln[static_cast<size_t>(j)] > bignum / xnorm)
                            {
                                x[0][0] /= xnorm;
                                x[0][1] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = 0; kk < k2; ++kk)
                            {
                                wrk[kk] *= scale;
                                wim[kk] *= scale;
                            }
                        }
                        wrk[j] = x[0][0];
                        wim[j] = x[0][1];
                        for(I i = 0; i < j; ++i)
                        {
                            wrk[i] -= x[0][0] * t(i, j);
                            wim[i] -= x[0][1] * t(i, j);
                        }
                    }
                    else
                    {
                        T t22[4] = {t(j1, j1), t(j1, j2), t(j2, j1), t(j2, j2)};
                        const T rhsr[2] = {wrk[j1], wrk[j2]};
                        const T rhsi[2] = {wim[j1], wim[j2]};
                        trevc_dlaln2_2x2<T, I>(2,
                                              smin,
                                              t22,
                                              2,
                                              wr_val,
                                              wi_val,
                                              rhsr,
                                              rhsi,
                                              x,
                                              scale,
                                              xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta
                                = std::max(coln[static_cast<size_t>(j1)], coln[static_cast<size_t>(j2)]);
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
                        {
                            for(I kk = 0; kk < k2; ++kk)
                            {
                                wrk[kk] *= scale;
                                wim[kk] *= scale;
                            }
                        }
                        wrk[j1] = x[0][0];
                        wrk[j2] = x[1][0];
                        wim[j1] = x[0][1];
                        wim[j2] = x[1][1];
                        for(I i = 0; i < j1; ++i)
                        {
                            wrk[i] -= x[0][0] * t(i, j1) + x[1][0] * t(i, j2);
                            wim[i] -= x[0][1] * t(i, j1) + x[1][1] * t(i, j2);
                        }
                    }
                }

                if(k2 > 1)
                {
                    for(I i = 0; i < n; ++i)
                    {
                        T sumr = wrk[k1] * vr(i, k1);
                        T sumi = wim[k2] * vr(i, k2);
                        for(I j = 0; j <= k2 - 2; ++j)
                        {
                            sumr += vr(i, j) * wrk[j];
                            sumi += vr(i, j) * wim[j];
                        }
                        vr(i, k1) = sumr;
                        vr(i, k2) = sumi;
                    }
                }
                else
                {
                    for(I i = 0; i < n; ++i)
                    {
                        vr(i, 0) *= wrk[0];
                        vr(i, 1) *= wim[1];
                    }
                }

                T emax = T(0);
                for(I k = 0; k < n; ++k)
                    emax = std::max(emax, std::abs(vr(k, k1)) + std::abs(vr(k, k2)));
                if(emax > T(0))
                {
                    const T remax = T(1) / emax;
                    for(I i = 0; i < n; ++i)
                    {
                        vr(i, k1) *= remax;
                        vr(i, k2) *= remax;
                    }
                }
            }
        }

        if(ip == 1)
            ip = 0;
        else if(ip == -1)
            ip = 1;
    }
}

/**
 * DTREVC with SIDE='L', HOWMNY='B': left eigenvectors with Schur-vector back-transform.
 * T is upper quasi-triangular (unchanged). VL input = Schur vectors Z; output = left eigenvectors.
 * Left eigenvectors satisfy u^T T = lambda u^T (equivalently T^T u = lambda u).
 *
 * Matches LAPACK xTREVC (LEFTV, HOWMNY='B', OVERWRITE): forward sweep ki = 0..n-1,
 * forward substitution for (T^T - lambda I) y = b using only column-wise accumulation
 * (DDOT on T(ki+1:j-1, j)), DLALN2 for diagonal blocks; for 2x2 left blocks LAPACK uses
 * LTRANS=.TRUE., implemented by passing the transposed 2x2 block to trevc_dlaln2_2x2.
 */
template <typename T, typename I>
void run_trevc_left_howmny_b(const I n, const T* TT, const I ldt, T* VL, const I ldvl)
{
    using namespace trevc_detail;

    auto t = [&](I i, I j) -> const T& { return TT[i + j * ldt]; };
    auto vl = [&](I i, I j) -> T& { return VL[i + j * ldvl]; };

    const T ulp = std::numeric_limits<T>::epsilon();
    const T unfl = std::numeric_limits<T>::min();
    const T smlnum = std::max(unfl * static_cast<T>(n) / ulp, ulp);
    const T bignum = (T(1) - ulp) / smlnum;

    std::vector<T> work(static_cast<size_t>(3 * static_cast<size_t>(n)));
    T* const wrk = work.data();
    T* const wim = wrk + static_cast<size_t>(n);
    T* const rown = wim + static_cast<size_t>(n);

    for(I i = 0; i < n; ++i)
    {
        T s = T(0);
        for(I j = i + 1; j < n; ++j)
            s += std::abs(t(i, j));
        rown[i] = s;
    }

    I ip = 0;

    for(I ki = 0; ki < n; ++ki)
    {
        bool skip_body = false;
        if(ip == 1)
            skip_body = true;
        else
        {
            if(ki + 1 < n && t(ki + 1, ki) != T(0))
                ip = -1;
        }

        if(!skip_body)
        {
            const T wr_val = t(ki, ki);
            T wi_val = T(0);
            if(ip != 0)
                wi_val = std::sqrt(std::abs(t(ki, ki + 1) * t(ki + 1, ki)));
            const T smin = std::max(ulp * (std::abs(wr_val) + std::abs(wi_val)), smlnum);

            if(ip == 0)
            {
                const I row = ki;
                wrk[row] = T(1);
                for(I k = row + 1; k < n; ++k)
                    wrk[k] = -t(row, k);

                I jnxt = row + 1;
                for(I j = row + 1; j < n; ++j)
                {
                    if(j < jnxt)
                        continue;
                    I j1 = j;
                    I j2 = j;
                    jnxt = j + 1;
                    if(j + 1 < n && t(j + 1, j) != T(0))
                    {
                        j1 = j;
                        j2 = j + 1;
                        jnxt = j + 2;
                    }

                    T x[2][2];
                    T scale;
                    T xnorm;

                    if(j1 == j2)
                    {
                        T dot = T(0);
                        for(I ii = row + 1; ii < j; ++ii)
                            dot += t(ii, j) * wrk[ii];
                        wrk[j] -= dot;

                        trevc_dlaln2_1x1<T, I>(true,
                                                1,
                                                smin,
                                                &t(j, j),
                                                ldt,
                                                wr_val,
                                                T(0),
                                                wrk[j],
                                                T(0),
                                                x,
                                                scale,
                                                xnorm);
                        if(xnorm > T(1))
                        {
                            if(rown[static_cast<size_t>(j)] > bignum / xnorm)
                            {
                                x[0][0] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = row; kk < n; ++kk)
                                wrk[kk] *= scale;
                        }
                        wrk[j] = x[0][0];
                    }
                    else
                    {
                        T dot1 = T(0);
                        T dot2 = T(0);
                        for(I ii = row + 1; ii < j; ++ii)
                        {
                            dot1 += t(ii, j1) * wrk[ii];
                            dot2 += t(ii, j2) * wrk[ii];
                        }
                        wrk[j1] -= dot1;
                        wrk[j2] -= dot2;

                        /* DLALN2(LTRANS=.TRUE.): pass A^T, row-major [a c; b d]. */
                        T t22[4]
                            = {t(j1, j1), t(j2, j1), t(j1, j2), t(j2, j2)};
                        const T rhsr[2] = {wrk[j1], wrk[j2]};
                        trevc_dlaln2_2x2<T, I>(1,
                                              smin,
                                              t22,
                                              2,
                                              wr_val,
                                              T(0),
                                              rhsr,
                                              rhsr,
                                              x,
                                              scale,
                                              xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta
                                = std::max(rown[static_cast<size_t>(j1)], rown[static_cast<size_t>(j2)]);
                            if(beta > bignum / xnorm)
                            {
                                const T rec = T(1) / xnorm;
                                x[0][0] *= rec;
                                x[1][0] *= rec;
                                scale *= rec;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = row; kk < n; ++kk)
                                wrk[kk] *= scale;
                        }
                        wrk[j1] = x[0][0];
                        wrk[j2] = x[1][0];
                    }
                }

                for(I i = 0; i < n; ++i)
                {
                    T sum = wrk[row] * vl(i, row);
                    for(I j = row + 1; j < n; ++j)
                        sum += vl(i, j) * wrk[j];
                    vl(i, row) = sum;
                }
                {
                    T amax = T(0);
                    for(I i = 0; i < n; ++i)
                        amax = std::max(amax, std::abs(vl(i, row)));
                    if(amax > T(0))
                    {
                        const T scl = T(1) / amax;
                        for(I i = 0; i < n; ++i)
                            vl(i, row) *= scl;
                    }
                }
            }
            else
            {
                const I k1 = ki;
                const I k2 = ki + 1;

                if(std::abs(t(k1, k2)) >= std::abs(t(k2, k1)))
                {
                    wrk[k1] = wi_val / safe_den(t(k1, k2), smin);
                    wim[k2] = T(1);
                }
                else
                {
                    wrk[k1] = T(1);
                    wim[k2] = -wi_val / safe_den(t(k2, k1), smin);
                }
                wrk[k2] = T(0);
                wim[k1] = T(0);

                for(I k = k1 + 2; k < n; ++k)
                {
                    wrk[k] = -wrk[k1] * t(k1, k);
                    wim[k] = -wim[k2] * t(k2, k);
                }

                I jnxt = k1 + 2;
                for(I j = k1 + 2; j < n; ++j)
                {
                    if(j < jnxt)
                        continue;
                    I j1 = j;
                    I j2 = j;
                    jnxt = j + 1;
                    if(j + 1 < n && t(j + 1, j) != T(0))
                    {
                        j1 = j;
                        j2 = j + 1;
                        jnxt = j + 2;
                    }

                    T x[2][2];
                    T scale;
                    T xnorm;

                    if(j1 == j2)
                    {
                        T dotr = T(0);
                        T doti = T(0);
                        for(I ii = k1 + 2; ii < j; ++ii)
                        {
                            dotr += t(ii, j) * wrk[ii];
                            doti += t(ii, j) * wim[ii];
                        }
                        wrk[j] -= dotr;
                        wim[j] -= doti;

                        trevc_dlaln2_1x1<T, I>(true,
                                                2,
                                                smin,
                                                &t(j, j),
                                                ldt,
                                                wr_val,
                                                -wi_val,
                                                wrk[j],
                                                wim[j],
                                                x,
                                                scale,
                                                xnorm);
                        if(xnorm > T(1))
                        {
                            if(rown[static_cast<size_t>(j)] > bignum / xnorm)
                            {
                                x[0][0] /= xnorm;
                                x[0][1] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if(scale != T(1))
                        {
                            for(I kk = k1; kk < n; ++kk)
                            {
                                wrk[kk] *= scale;
                                wim[kk] *= scale;
                            }
                        }
                        wrk[j] = x[0][0];
                        wim[j] = x[0][1];
                    }
                    else
                    {
                        T dot1r = T(0);
                        T dot1i = T(0);
                        T dot2r = T(0);
                        T dot2i = T(0);
                        for(I ii = k1 + 2; ii < j; ++ii)
                        {
                            dot1r += t(ii, j1) * wrk[ii];
                            dot1i += t(ii, j1) * wim[ii];
                            dot2r += t(ii, j2) * wrk[ii];
                            dot2i += t(ii, j2) * wim[ii];
                        }
                        wrk[j1] -= dot1r;
                        wim[j1] -= dot1i;
                        wrk[j2] -= dot2r;
                        wim[j2] -= dot2i;

                        T t22[4]
                            = {t(j1, j1), t(j2, j1), t(j1, j2), t(j2, j2)};
                        const T rhsr[2] = {wrk[j1], wrk[j2]};
                        const T rhsi[2] = {wim[j1], wim[j2]};
                        trevc_dlaln2_2x2<T, I>(2,
                                              smin,
                                              t22,
                                              2,
                                              wr_val,
                                              -wi_val,
                                              rhsr,
                                              rhsi,
                                              x,
                                              scale,
                                              xnorm);
                        if(xnorm > T(1))
                        {
                            const T beta
                                = std::max(rown[static_cast<size_t>(j1)], rown[static_cast<size_t>(j2)]);
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
                        {
                            for(I kk = k1; kk < n; ++kk)
                            {
                                wrk[kk] *= scale;
                                wim[kk] *= scale;
                            }
                        }
                        wrk[j1] = x[0][0];
                        wrk[j2] = x[1][0];
                        wim[j1] = x[0][1];
                        wim[j2] = x[1][1];
                    }
                }

                if(ki + 2 < n)
                {
                    for(I i = 0; i < n; ++i)
                    {
                        T sumr = wrk[k1] * vl(i, k1);
                        T sumi = wim[k2] * vl(i, k2);
                        for(I j = k1 + 2; j < n; ++j)
                        {
                            sumr += vl(i, j) * wrk[j];
                            sumi += vl(i, j) * wim[j];
                        }
                        vl(i, k1) = sumr;
                        vl(i, k2) = sumi;
                    }
                }
                else
                {
                    for(I i = 0; i < n; ++i)
                    {
                        vl(i, k1) *= wrk[k1];
                        vl(i, k2) *= wim[k2];
                    }
                }

                T emax = T(0);
                for(I k = 0; k < n; ++k)
                    emax = std::max(emax, std::abs(vl(k, k1)) + std::abs(vl(k, k2)));
                if(emax > T(0))
                {
                    const T remax = T(1) / emax;
                    for(I i = 0; i < n; ++i)
                    {
                        vl(i, k1) *= remax;
                        vl(i, k2) *= remax;
                    }
                }
            }
        }

        if(ip == 1)
            ip = 0;
        else if(ip == -1)
            ip = 1;
    }
}

template <typename T, typename I, typename UT, typename UVR>
rocblas_status rocsolver_trevc_argCheck(rocblas_handle handle,
                                        const rocblas_side side,
                                        const I n,
                                        const I ldt,
                                        const I ldvl,
                                        const I ldvr,
                                        UT dT,
                                        UVR dVR,
                                        rocblas_int* m,
                                        rocblas_int* info,
                                        const I batch_count)
{
    (void)dT;
    (void)dVR;
    if(side != rocblas_side_right)
        return rocblas_status_invalid_value;
    if(n < 0 || ldt < std::max<I>(1, n) || ldvr < std::max<I>(1, n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(ldvl < 1)
        return rocblas_status_invalid_size;

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    if(!info || !m || !dT || !dVR)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename UT, typename UVR>
void rocsolver_trevc_getMemorySize(const I n, const I batch_count, size_t* size_work)
{
    (void)BATCHED;
    (void)STRIDED;
    (void)n;
    (void)batch_count;
    *size_work = 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename UT, typename UVR>
rocblas_status rocsolver_trevc_template(rocblas_handle handle,
                                        const rocblas_side side,
                                        const I n,
                                        UT dT_in,
                                        const rocblas_stride shiftT,
                                        const I ldt,
                                        const rocblas_stride strideT,
                                        UVR dVR_in,
                                        const rocblas_stride shiftVR,
                                        const I ldvr,
                                        const rocblas_stride strideVR,
                                        rocblas_int* m,
                                        const rocblas_stride strideM,
                                        rocblas_int* info,
                                        const rocblas_stride strideInfo,
                                        const I batch_count)
{
    (void)side;
    ROCSOLVER_ENTER("trevc", "n:", n, "ldt:", ldt, "ldvr:", ldvr, "bc:", batch_count);

    if(!batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const rocblas_stride stride_t = strideT ? strideT : rocblas_stride(ldt) * rocblas_stride(n);
    const rocblas_stride stride_vr = strideVR ? strideVR : rocblas_stride(ldvr) * rocblas_stride(n);
    const rocblas_stride stride_m = strideM ? strideM : 1;
    const rocblas_stride stride_info = strideInfo ? strideInfo : 1;

    const rocblas_int blocksReset
        = static_cast<rocblas_int>((static_cast<size_t>(batch_count) + BS1 - 1) / BS1);
    ROCSOLVER_LAUNCH_KERNEL(reset_info,
                            dim3(blocksReset, 1, 1),
                            dim3(BS1, 1, 1),
                            0,
                            stream,
                            info,
                            batch_count,
                            0);
    HIP_CHECK(hipStreamSynchronize(stream));

    if(n == 0)
        return rocblas_status_success;

    const size_t t_elems = static_cast<size_t>(ldt) * static_cast<size_t>(n);
    const size_t vr_elems = static_cast<size_t>(ldvr) * static_cast<size_t>(n);
    const size_t t_bytes = t_elems * sizeof(T);
    const size_t vr_bytes = vr_elems * sizeof(T);

    std::vector<T*> hPtrT;
    std::vector<T*> hPtrVR;

    if constexpr(BATCHED)
    {
        hPtrT.resize(static_cast<size_t>(batch_count));
        hPtrVR.resize(static_cast<size_t>(batch_count));
        HIP_CHECK(hipMemcpyAsync(hPtrT.data(),
                                 reinterpret_cast<T* const*>(dT_in),
                                 sizeof(T*) * static_cast<size_t>(batch_count),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(hPtrVR.data(),
                                 reinterpret_cast<T* const*>(dVR_in),
                                 sizeof(T*) * static_cast<size_t>(batch_count),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        std::vector<std::vector<T>> hT_batches(static_cast<size_t>(batch_count));
        std::vector<std::vector<T>> hVR_batches(static_cast<size_t>(batch_count));
        for(I b = 0; b < batch_count; ++b)
        {
            hT_batches[static_cast<size_t>(b)].resize(t_elems);
            hVR_batches[static_cast<size_t>(b)].resize(vr_elems);
            T* const dTb = hPtrT[static_cast<size_t>(b)] + shiftT;
            T* const dVRb = hPtrVR[static_cast<size_t>(b)] + shiftVR;
            HIP_CHECK(hipMemcpyAsync(hT_batches[static_cast<size_t>(b)].data(),
                                     dTb,
                                     t_bytes,
                                     hipMemcpyDeviceToHost,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(hVR_batches[static_cast<size_t>(b)].data(),
                                     dVRb,
                                     vr_bytes,
                                     hipMemcpyDeviceToHost,
                                     stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_count > 1)
#endif
        for(I b = 0; b < batch_count; ++b)
        {
            run_trevc_right_howmny_b<T, I>(n,
                                           hT_batches[static_cast<size_t>(b)].data(),
                                           ldt,
                                           hVR_batches[static_cast<size_t>(b)].data(),
                                           ldvr);
        }

        for(I b = 0; b < batch_count; ++b)
        {
            T* const dVRb = hPtrVR[static_cast<size_t>(b)] + shiftVR;
            HIP_CHECK(hipMemcpyAsync(dVRb,
                                     hVR_batches[static_cast<size_t>(b)].data(),
                                     vr_bytes,
                                     hipMemcpyHostToDevice,
                                     stream));
            const rocblas_int hInfo = 0;
            const rocblas_int hm = static_cast<rocblas_int>(n);
            HIP_CHECK(hipMemcpyAsync(m + b * stride_m,
                                     &hm,
                                     sizeof(rocblas_int),
                                     hipMemcpyHostToDevice,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(info + b * stride_info,
                                     &hInfo,
                                     sizeof(rocblas_int),
                                     hipMemcpyHostToDevice,
                                     stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    else if constexpr(STRIDED)
    {
        if(batch_count > 1)
        {
            std::vector<std::vector<T>> hT_batches(static_cast<size_t>(batch_count));
            std::vector<std::vector<T>> hVR_batches(static_cast<size_t>(batch_count));
            for(I b = 0; b < batch_count; ++b)
            {
                hT_batches[static_cast<size_t>(b)].resize(t_elems);
                hVR_batches[static_cast<size_t>(b)].resize(vr_elems);
                T* const dTb = reinterpret_cast<T*>(dT_in) + shiftT + b * stride_t;
                T* const dVRb = reinterpret_cast<T*>(dVR_in) + shiftVR + b * stride_vr;
                HIP_CHECK(hipMemcpyAsync(hT_batches[static_cast<size_t>(b)].data(),
                                         dTb,
                                         t_bytes,
                                         hipMemcpyDeviceToHost,
                                         stream));
                HIP_CHECK(hipMemcpyAsync(hVR_batches[static_cast<size_t>(b)].data(),
                                         dVRb,
                                         vr_bytes,
                                         hipMemcpyDeviceToHost,
                                         stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(I b = 0; b < batch_count; ++b)
            {
                run_trevc_right_howmny_b<T, I>(n,
                                               hT_batches[static_cast<size_t>(b)].data(),
                                               ldt,
                                               hVR_batches[static_cast<size_t>(b)].data(),
                                               ldvr);
            }

            for(I b = 0; b < batch_count; ++b)
            {
                T* const dVRb = reinterpret_cast<T*>(dVR_in) + shiftVR + b * stride_vr;
                HIP_CHECK(hipMemcpyAsync(dVRb,
                                         hVR_batches[static_cast<size_t>(b)].data(),
                                         vr_bytes,
                                         hipMemcpyHostToDevice,
                                         stream));
                const rocblas_int hInfo = 0;
                const rocblas_int hm = static_cast<rocblas_int>(n);
                HIP_CHECK(hipMemcpyAsync(m + b * stride_m,
                                         &hm,
                                         sizeof(rocblas_int),
                                         hipMemcpyHostToDevice,
                                         stream));
                HIP_CHECK(hipMemcpyAsync(info + b * stride_info,
                                         &hInfo,
                                         sizeof(rocblas_int),
                                         hipMemcpyHostToDevice,
                                         stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        else
        {
            std::vector<T> hT(t_elems);
            std::vector<T> hVR(vr_elems);
            T* dT = reinterpret_cast<T*>(dT_in) + shiftT;
            T* dVR = reinterpret_cast<T*>(dVR_in) + shiftVR;

            HIP_CHECK(hipMemcpyAsync(hT.data(), dT, t_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(hVR.data(), dVR, vr_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            const rocblas_int hInfo = 0;
            run_trevc_right_howmny_b<T, I>(n, hT.data(), ldt, hVR.data(), ldvr);

            HIP_CHECK(hipMemcpyAsync(dVR, hVR.data(), vr_bytes, hipMemcpyHostToDevice, stream));
            const rocblas_int hm = static_cast<rocblas_int>(n);
            HIP_CHECK(hipMemcpyAsync(m, &hm, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
            HIP_CHECK(
                hipMemcpyAsync(info, &hInfo, sizeof(rocblas_int), hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    else
    {
        std::vector<T> hT(t_elems);
        std::vector<T> hVR(vr_elems);

        for(I b = 0; b < batch_count; ++b)
        {
            T* dT = reinterpret_cast<T*>(dT_in) + shiftT + b * stride_t;
            T* dVR = reinterpret_cast<T*>(dVR_in) + shiftVR + b * stride_vr;

            HIP_CHECK(hipMemcpyAsync(hT.data(), dT, t_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(hVR.data(), dVR, vr_bytes, hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            const rocblas_int hInfo = 0;
            run_trevc_right_howmny_b<T, I>(n, hT.data(), ldt, hVR.data(), ldvr);

            HIP_CHECK(hipMemcpyAsync(dVR, hVR.data(), vr_bytes, hipMemcpyHostToDevice, stream));
            const rocblas_int hm = static_cast<rocblas_int>(n);
            HIP_CHECK(hipMemcpyAsync(m + b * stride_m,
                                     &hm,
                                     sizeof(rocblas_int),
                                     hipMemcpyHostToDevice,
                                     stream));
            HIP_CHECK(hipMemcpyAsync(info + b * stride_info,
                                     &hInfo,
                                     sizeof(rocblas_int),
                                     hipMemcpyHostToDevice,
                                     stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
