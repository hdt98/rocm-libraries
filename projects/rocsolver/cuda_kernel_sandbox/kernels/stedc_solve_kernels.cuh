/* **************************************************************************
 * STEDC Solve Kernels - Secular equation solvers for divide-and-conquer
 *
 * Based on rocSOLVER's rocauxiliary_stedc.hpp and lapack_device_functions.hpp
 *
 * These kernels solve secular equations that arise during the merge phase
 * of the divide-and-conquer algorithm for computing eigenvalues/eigenvectors
 * of symmetric tridiagonal matrices.
 *
 * Contains:
 * - slaed6: Solves a modified secular equation using 3 poles
 * - slaed4: Single-threaded LAPACK-style solver
 * - laed4_alt: Multi-threaded parallel solver (uses BDIM threads per eigenvalue)
 * - laed6 (single-thread version for laed4_alt)
 * - reduce_wave_sum / reduce_block_sum: Warp/block reduction helpers
 * - stedc_mergeValues_Solve_kernel: Main kernel that calls the solvers
 *
 * IMPORTANT: laed4_alt has a potential race condition bug when BDIM is a
 * multiple of 32 (warp size). This is intentionally included to test
 * CUDA Compute Sanitizer's ability to detect such race conditions.
 *
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
 * *************************************************************************/

#pragma once

#include "../cuda_compat.cuh"
#include "../rocsolver_types.cuh"
#include "../device_helpers.cuh"

ROCSOLVER_BEGIN_NAMESPACE

// Max number of iterations for root finding methods
#define MAXITERS 50

/************** Warp/Block Reduction Helpers *********************************/

/** Single-value warp-level reduction using shuffle instructions */
template <std::int32_t BDIM = 0, typename S>
__device__ inline void reduce_wave_sum(S& val)
{
    // BDIM should be <= warpSize
#pragma unroll
    for(rocblas_int r = warpSize / 2; r >= 1; r /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, r);
    }

    val = __shfl_sync(0xffffffff, val, 0);
}

/** Single-value block-level reduction using shared memory + shuffles */
template <std::int32_t BDIM, typename S>
__device__ inline void reduce_block_sum(S& val)
{
    if(BDIM > warpSize)
    {
        __shared__ S lds[BDIM];
        rocblas_int tid = threadIdx.x;

        lds[tid] = val;
        __syncthreads();

#pragma unroll
        for(rocblas_int r = BDIM / 2; r >= warpSize; r /= 2)
        {
            if(tid < r)
            {
                lds[tid] += lds[tid + r];
            }
            __syncthreads();
        }

        // Only threads in first warp participate in warp-level reduction
        if(tid < warpSize)
        {
            val = lds[tid];
        }
        __syncthreads();

        if(tid < warpSize)
        {
#pragma unroll
            for(rocblas_int r = warpSize / 2; r >= 1; r /= 2)
            {
                val += __shfl_down_sync(0xffffffff, val, r);
            }

            if(threadIdx.x == 0)
            {
                lds[0] = val;
            }
        }
        __syncthreads();

        val = lds[0];
        __syncthreads();
    }
    else
    {
        reduce_wave_sum<BDIM>(val);
    }
}

/** Three-value warp-level reduction using shuffle instructions */
template <std::int32_t BDIM = 0, typename S>
__device__ inline void reduce_wave_sum(S& val1, S& val2, S& val3)
{
#pragma unroll
    for(rocblas_int r = warpSize / 2; r >= 1; r /= 2)
    {
        val1 += __shfl_down_sync(0xffffffff, val1, r);
        val2 += __shfl_down_sync(0xffffffff, val2, r);
        val3 += __shfl_down_sync(0xffffffff, val3, r);
    }

    val1 = __shfl_sync(0xffffffff, val1, 0);
    val2 = __shfl_sync(0xffffffff, val2, 0);
    val3 = __shfl_sync(0xffffffff, val3, 0);
}

/** Three-value block-level reduction using shared memory + shuffles */
template <std::int32_t BDIM, typename S>
__device__ inline void reduce_block_sum(S& val1, S& val2, S& val3)
{
    if(BDIM > warpSize)
    {
        __shared__ S lds1[BDIM];
        __shared__ S lds2[BDIM];
        __shared__ S lds3[BDIM];
        rocblas_int tid = threadIdx.x;

        lds1[tid] = val1;
        lds2[tid] = val2;
        lds3[tid] = val3;
        __syncthreads();

#pragma unroll
        for(rocblas_int r = BDIM / 2; r >= warpSize; r /= 2)
        {
            if(tid < r)
            {
                lds1[tid] += lds1[tid + r];
                lds2[tid] += lds2[tid + r];
                lds3[tid] += lds3[tid + r];
            }
            __syncthreads();
        }

        // Only threads in first warp participate in warp-level reduction
        if(tid < warpSize)
        {
            val1 = lds1[tid];
            val2 = lds2[tid];
            val3 = lds3[tid];
        }
        __syncthreads();

        if(tid < warpSize)
        {
#pragma unroll
            for(rocblas_int r = warpSize / 2; r >= 1; r /= 2)
            {
                val1 += __shfl_down_sync(0xffffffff, val1, r);
                val2 += __shfl_down_sync(0xffffffff, val2, r);
                val3 += __shfl_down_sync(0xffffffff, val3, r);
            }

            if(threadIdx.x == 0)
            {
                lds1[0] = val1;
                lds2[0] = val2;
                lds3[0] = val3;
            }
        }
        __syncthreads();

        val1 = lds1[0];
        val2 = lds2[0];
        val3 = lds3[0];
        __syncthreads();
    }
    else
    {
        reduce_wave_sum<BDIM>(val1, val2, val3);
    }
}

/************** Single-threaded slaed6 for laed4_alt *************************/

/** LAED6 for 3-pole interpolation (single-threaded, called from laed4_alt) */
template <typename S, typename I>
__device__ I laed6(I kniter,
                   bool orgati,
                   S rho,
                   S* d,
                   S* z,
                   S finit,
                   S& tau,
                   S eps,
                   S ssfmin,
                   I MAXIT = 50)
{
    auto lam_abs = [](auto x) -> auto { return device_abs(x); };
    auto lam_sqrt = [](auto x) -> auto { return sqrt(x); };
    auto lam_max = [](auto x, auto y, auto zz) -> auto { return device_max(device_max(x, y), zz); };
    auto lam_min = [](auto x, auto y) -> auto { return device_min(x, y); };

    S dscale[3]{}, zscale[3]{};

    struct X_t
    {
        S* x_;
        __device__ X_t(S* x) : x_(x) {}
        __device__ S& operator()(int j) { return x_[j - 1]; }
    } D(d), Z(z), DSCALE(dscale), ZSCALE(zscale);

    bool scale;
    S a, b, c, ddf, df, erretm, eta, f, fc, sclfac, sclinv, temp, temp1, temp2, temp3, temp4, lbd, ubd;
    I iter, niter;
    I info = 0;

    if(orgati)
    {
        lbd = D(2);
        ubd = D(3);
    }
    else
    {
        lbd = D(1);
        ubd = D(2);
    }

    if(finit < S(0.))
        lbd = S(0.);
    else
        ubd = S(0.);

    niter = 1;
    tau = S(0.);

    if(kniter == 2)
    {
        if(orgati)
        {
            temp = (D(3) - D(2)) / S(2.);
            c = rho + Z(1) / ((D(1) - D(2)) - temp);
            a = c * (D(2) + D(3)) + Z(2) + Z(3);
            b = c * D(2) * D(3) + Z(2) * D(3) + Z(3) * D(2);
        }
        else
        {
            temp = (D(1) - D(2)) / S(2.);
            c = rho + Z(3) / ((D(3) - D(2)) - temp);
            a = c * (D(1) + D(2)) + Z(1) + Z(2);
            b = c * D(1) * D(2) + Z(1) * D(2) + Z(2) * D(1);
        }

        temp = lam_max(lam_abs(a), lam_abs(b), lam_abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if(c == S(0.))
            tau = b / a;
        else if(a <= S(0.))
            tau = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        else
            tau = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));

        if(tau < lbd || tau > ubd)
            tau = (lbd + ubd) / S(2.);

        if(D(1) == tau || D(2) == tau || D(3) == tau)
            tau = S(0.);
        else
        {
            temp = finit + tau * Z(1) / (D(1) * (D(1) - tau))
                 + tau * Z(2) / (D(2) * (D(2) - tau))
                 + tau * Z(3) / (D(3) * (D(3) - tau));
            if(temp <= S(0.))
                lbd = tau;
            else
                ubd = tau;

            if(lam_abs(finit) <= lam_abs(temp))
                tau = S(0.);
        }
    }

    const S small1 = pow(S(2.), (device_log(ssfmin) / device_log(S(2.))) / S(3.));
    const S sminv1 = S(1.) / small1;
    const S small2 = small1 * small1;
    const S sminv2 = sminv1 * sminv1;

    if(orgati)
        temp = lam_min(lam_abs(D(2) - tau), lam_abs(D(3) - tau));
    else
        temp = lam_min(lam_abs(D(1) - tau), lam_abs(D(2) - tau));

    scale = false;
    if(temp <= small1)
    {
        scale = true;
        if(temp <= small2)
        {
            sclfac = sminv2;
            sclinv = small2;
        }
        else
        {
            sclfac = sminv1;
            sclinv = small1;
        }
        for(int i = 1; i <= 3; ++i)
        {
            DSCALE(i) = D(i) * sclfac;
            ZSCALE(i) = Z(i) * sclfac;
        }
        tau = tau * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    }
    else
    {
        for(int i = 1; i <= 3; ++i)
        {
            DSCALE(i) = D(i);
            ZSCALE(i) = Z(i);
        }
    }

    fc = S(0.);
    df = S(0.);
    ddf = S(0.);
    for(int i = 1; i <= 3; ++i)
    {
        temp = S(1.) / (DSCALE(i) - tau);
        temp1 = ZSCALE(i) * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        fc = fc + temp1 / DSCALE(i);
        df = df + temp2;
        ddf = ddf + temp3;
    }
    f = finit + tau * fc;

    if(lam_abs(f) <= S(0.))
    {
        if(scale)
            tau = tau * sclinv;
        return info;
    }

    if(f <= S(0.))
        lbd = tau;
    else
        ubd = tau;

    iter = niter + 1;
    for(int niter = iter; niter <= MAXIT; ++niter)
    {
        if(orgati)
        {
            temp1 = DSCALE(2) - tau;
            temp2 = DSCALE(3) - tau;
        }
        else
        {
            temp1 = DSCALE(1) - tau;
            temp2 = DSCALE(2) - tau;
        }

        a = (temp1 + temp2) * f - temp1 * temp2 * df;
        b = temp1 * temp2 * f;
        c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
        temp = lam_max(lam_abs(a), lam_abs(b), lam_abs(c));
        a = a / temp;
        b = b / temp;
        c = c / temp;

        if(c == S(0.))
            eta = b / a;
        else if(a <= S(0.))
            eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        else
            eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));

        if(f * eta >= S(0.))
            eta = -f / df;

        tau = tau + eta;
        if(tau < lbd || tau > ubd)
            tau = (lbd + ubd) / S(2.);

        fc = S(0.);
        erretm = S(0.);
        df = S(0.);
        ddf = S(0.);
        for(int i = 1; i <= 3; ++i)
        {
            if((DSCALE(i) - tau) != S(0.))
            {
                temp = S(1.) / (DSCALE(i) - tau);
                temp1 = ZSCALE(i) * temp;
                temp2 = temp1 * temp;
                temp3 = temp2 * temp;
                temp4 = temp1 / DSCALE(i);
                fc = fc + temp4;
                erretm = erretm + lam_abs(temp4);
                df = df + temp2;
                ddf = ddf + temp3;
            }
            else
            {
                if(scale)
                    tau = tau * sclinv;
                return info;
            }
        }
        f = finit + tau * fc;
        erretm = S(8.) * (lam_abs(finit) + lam_abs(tau) * erretm) + lam_abs(tau) * df;

        if((lam_abs(f) <= S(4.) * eps * erretm) || ((ubd - lbd) <= S(4.) * eps * lam_abs(tau)))
        {
            if(scale)
                tau = tau * sclinv;
            return info;
        }

        if(f <= S(0.))
            lbd = tau;
        else
            ubd = tau;
    }

    info = 1;
    if(scale)
        tau = tau * sclinv;

    return info;
}

/************** LAED4_ALT: Multi-threaded parallel solver *******************/

/** LAED4_ALT computes eigenvalue i of D + rho * z * z^T using BDIM threads
 *
 * This is a parallelized version of slaed4 that uses warp/block-level
 * reductions to compute the secular equation terms.
 *
 * KNOWN BUG: When BDIM > warpSize (i.e., BDIM is a multiple of 32 like 64),
 * there may be race conditions due to missing or improperly placed
 * __syncthreads() calls. This is intentionally included to test
 * CUDA Compute Sanitizer's race detection capability.
 *
 * @param BDIM Template parameter: number of threads per eigenvalue
 * @param OVERRIDE_3RD_ORDER_SCHEME If true, skip 3-pole interpolation
 */
template <typename S, typename I, I BDIM, bool OVERRIDE_3RD_ORDER_SCHEME = false>
__device__ I laed4_alt(I n,
                       I i,
                       S* delta,
                       S* z,
                       S rho,
                       S& dlam,
                       S eps = std::numeric_limits<S>::epsilon() / S(2.),
                       S ssfmin = std::numeric_limits<S>::min(),
                       I MAXIT = 50)
{
    i = i + 1;  // Convert to 1-indexed
    auto lam_abs = [](auto x) -> auto { return device_abs(x); };
    auto lam_sqr = [](auto x) -> auto { return x * x; };
    auto lam_sqrt = [](auto x) -> auto { return sqrt(x); };
    auto lam_max = [](auto x, auto y) -> auto { return device_max(x, y); };
    auto lam_min = [](auto x, auto y) -> auto { return device_min(x, y); };

    S zz[3]{};
    struct X_t
    {
        S* x_;
        __device__ X_t(S* x) : x_(x) {}
        __device__ S& operator()(int j) { return x_[j - 1]; }
    } Z(z), ZZ(zz), DELTA(delta);

    S tau, eta = S(0.), dltlb, dltub;
    S psi, dpsi, phi, dphi, rhoinv, midpt;
    S del, a, b, c, w, erretm, erretm2, temp, dw, temp1, prew;
    I ii, niter, iter, orgati, iim1, iip1;
    bool swtch3, swtch;

    S d1 = DELTA(1);
    S di = DELTA(i);
    S dnm1 = DELTA(n - 1);
    S dn = DELTA(n);

    rhoinv = S(1.) / rho;
    I info = 0;

    if(n == 1)
    {
        dlam = d1 + rho * Z(1) * Z(1);
        DELTA(1) = S(1.);
    }
    else if(i == n)
    {
        ii = n - 1;
        niter = 1;
        midpt = rho / S(2.);

        psi = S(0.);
        for(int j = 1 + threadIdx.x; j <= n - 2; j += blockDim.x)
        {
            S dj = (DELTA(j) - di) - midpt;
            psi = psi + Z(j) * Z(j) / ((DELTA(j) - di) - midpt);
        }
        reduce_block_sum<BDIM>(psi);

        c = rhoinv + psi;
        w = c + Z(ii) * Z(ii) / ((DELTA(ii) - di) - midpt) + Z(n) * Z(n) / ((dn - di) - midpt);
        if(w <= S(0.))
        {
            temp = Z(n - 1) * Z(n - 1) / (dn - dnm1 + rho) + Z(n) * Z(n) / rho;
            if(c <= temp)
            {
                tau = rho;
            }
            else
            {
                del = dn - dnm1;
                a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
                b = Z(n) * Z(n) * del;
                if(a < S(0.))
                {
                    tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
                }
                else
                {
                    tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
                }
            }
            dltlb = midpt;
            dltub = rho;
        }
        else
        {
            del = dn - dnm1;
            a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
            b = Z(n) * Z(n) * del;
            if(a < S(0.))
            {
                tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
            }
            else
            {
                tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
            }
            dltlb = S(0.);
            dltub = midpt;
        }
        for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
        {
            DELTA(j) = (DELTA(j) - di) - tau;
        }
        __syncthreads();
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1 + threadIdx.x; j <= ii; j += blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        reduce_block_sum<BDIM>(psi, dpsi, erretm);
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;
        //
        //        Test for convergence
        //
        if(lam_abs(w) <= eps * erretm)
        {
            dlam = di + tau;
            return info;
        }
        if(w <= S(0.))
        {
            dltlb = lam_max(dltlb, tau);
        }
        else
        {
            dltub = lam_min(dltub, tau);
        }
        //
        //        Calculate the new step
        //
        niter = niter + 1;
        c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
        a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
        b = DELTA(n - 1) * DELTA(n) * w;
        // REVIEW
        if(c < S(0.))
        {
            c = lam_abs(c);
        }
        if(c <= S(0.))
        {
            eta = -w / (dpsi + dphi);
        }
        else if(a >= S(0.))
        {
            eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        }
        else
        {
            eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
        }
        //
        //        Note, eta should be positive if w is negative, and
        //        eta should be negative otherwise. However,
        //        if for some reason caused by roundoff, eta*w > 0,
        //        we simply use one Newton step instead. This way
        //        will guarantee eta*w < 0.
        //
        if(w * eta > S(0.))
        {
            eta = -w / (dpsi + dphi);
        }
        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
            {
                eta = (dltub - tau) / S(2.);
            }
            else
            {
                eta = (dltlb - tau) / S(2.);
            }
        }
        for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
        {
            DELTA(j) = DELTA(j) - eta;
        }
        __syncthreads();

        tau = tau + eta;
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1 + threadIdx.x; j <= ii; j += blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        reduce_block_sum<BDIM>(psi, dpsi, erretm);
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;
        //
        //        Main loop to update the values of the array DELTA
        //
        iter = niter + 1;
        for(niter = iter; niter <= MAXIT; ++niter)
        {
            //
            //           Test for convergence
            //
            if(lam_abs(w) <= eps * erretm)
            {
                dlam = di + tau;
                return info;
            }
            if(w <= S(0.))
            {
                dltlb = lam_max(dltlb, tau);
            }
            else
            {
                dltub = lam_min(dltub, tau);
            }
            //
            //           Calculate the new step
            //
            c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
            a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
            b = DELTA(n - 1) * DELTA(n) * w;
            if(a >= S(0.))
            {
                eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            else
            {
                eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
            //
            //           Note, eta should be positive if w is negative, and
            //           eta should be negative otherwise. However,
            //           if for some reason caused by roundoff, eta*w > 0,
            //           we simply use one Newton step instead. This way
            //           will guarantee eta*w < 0.
            //
            if(w * eta > S(0.))
            {
                eta = -w / (dpsi + dphi);
            }
            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                {
                    eta = (dltub - tau) / S(2.);
                }
                else
                {
                    eta = (dltlb - tau) / S(2.);
                }
            }
            for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
            {
                DELTA(j) = DELTA(j) - eta;
            }
            __syncthreads();
            tau = tau + eta;
            //
            //           Evaluate psi and the derivative dpsi
            //
            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1 + threadIdx.x; j <= ii; j += blockDim.x)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            reduce_block_sum<BDIM>(psi, dpsi, erretm);
            erretm = lam_abs(erretm);
            //
            //           Evaluate phi and the derivative dphi
            //
            temp = Z(n) / DELTA(n);
            phi = Z(n) * temp;
            dphi = temp * temp;
            erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
            w = rhoinv + phi + psi;
        }
        //
        //        Return with info = 1, niter = MAXIT and not converged
        //
        info = 1;
        dlam = di + tau;
    }
    else
    {
        //
        //        The case for i < n
        //
        niter = 1;
        I ip1 = i + 1;
        S dip1 = DELTA(ip1);
        //
        //        Calculate initial guess
        //
        del = dip1 - di;
        midpt = del / S(2.);
        psi = S(0.);
        for(int j = 1 + threadIdx.x; j <= i - 1; j += blockDim.x)
        {
            S dj = (DELTA(j) - di) - midpt;
            psi = psi + Z(j) * Z(j) / dj;
        }
        reduce_block_sum<BDIM>(psi);

        phi = S(0.);
        for(int j = n - threadIdx.x; j >= i + 2; j -= blockDim.x)
        {
            S dj = (DELTA(j) - di) - midpt;
            phi = phi + Z(j) * Z(j) / dj;
        }
        reduce_block_sum<BDIM>(phi);

        c = rhoinv + psi + phi;
        w = c + Z(i) * Z(i) / (-midpt) + Z(ip1) * Z(ip1) / ((dip1 - di) - midpt);
        if(w > S(0.))
        {
            //
            //           d(i) < the ith eigenvalue < (d(i)+d(i+1))/2
            //
            //           We choose d(i) as origin.
            //
            orgati = true;
            a = c * del + Z(i) * Z(i) + Z(ip1) * Z(ip1);
            b = Z(i) * Z(i) * del;
            if(a > S(0.))
            {
                tau = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
            else
            {
                tau = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            dltlb = S(0.);
            dltub = midpt;
        }
        else
        {
            //
            //           (d(i)+d(i+1))/2 <= the ith eigenvalue < d(i+1)
            //
            //           We choose d(i+1) as origin.
            //
            orgati = false;
            a = c * del - Z(i) * Z(i) - Z(ip1) * Z(ip1);
            b = Z(ip1) * Z(ip1) * del;
            if(a < S(0.))
            {
                tau = S(2.) * b / (a - lam_sqrt(lam_abs(a * a + S(4.) * b * c)));
            }
            else
            {
                tau = -(a + lam_sqrt(lam_abs(a * a + S(4.) * b * c))) / (S(2.) * c);
            }
            dltlb = -midpt;
            dltub = S(0.);
        }
        if(orgati)
        {
            ii = i;
        }
        else
        {
            ii = i + 1;
        }
        iim1 = ii - 1;
        iip1 = ii + 1;

        // Handle i=1 edge case separately to avoid out-of-bounds access
        S diim1, diip1;
        if(i == 1)
        {
            // When i=1 and orgati=true, ii=1, so iim1=0 would cause delta[-1] access
            // Set diim1 to 0 since it won't be used in the psi loop (loop from 1 to iim1=0 is empty)
            diim1 = S(0.);
            diip1 = (iip1 <= n) ? DELTA(iip1) : S(0.);
        }
        else
        {
            diim1 = DELTA(iim1);
            diip1 = (iip1 <= n) ? DELTA(iip1) : S(0.);
        }

        if(orgati)
        {
            for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
            {
                DELTA(j) = (DELTA(j) - di) - tau;
            }
        }
        else
        {
            for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
            {
                DELTA(j) = (DELTA(j) - dip1) - tau;
            }
        }
        __syncthreads();
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1 + threadIdx.x; j <= iim1; j += blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        reduce_block_sum<BDIM>(psi, dpsi, erretm);
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        dphi = S(0.);
        phi = S(0.);
        erretm2 = S(0.);
        for(int j = n - threadIdx.x; j >= iip1; j -= blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm2 = erretm2 + phi;
        }
        reduce_block_sum<BDIM>(phi, dphi, erretm2);
        erretm += erretm2;
        w = rhoinv + phi + psi;
        //
        //        w is the value of the secular function with
        //        its ii-th element removed.
        //
        swtch3 = false;
        if(!OVERRIDE_3RD_ORDER_SCHEME)
        {
            if(orgati)
            {
                if(w < S(0.))
                {
                    swtch3 = true;
                }
            }
            else
            {
                if(w > S(0.))
                {
                    swtch3 = true;
                }
            }
            if(ii == 1 || ii == n)
            {
                swtch3 = false;
            }
        }
        temp = Z(ii) / DELTA(ii);
        dw = dpsi + dphi + temp * temp;
        temp = Z(ii) * temp;
        w = w + temp;
        erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
            + lam_abs(tau) * dw;
        //
        //        Test for convergence
        //
        if(lam_abs(w) <= eps * erretm)
        {
            if(orgati)
            {
                dlam = di + tau;
            }
            else
            {
                dlam = dip1 + tau;
            }

            return info;
        }
        if(w <= S(0.))
        {
            dltlb = lam_max(dltlb, tau);
        }
        else
        {
            dltub = lam_min(dltub, tau);
        }
        //
        //        Calculate the new step
        //
        niter = niter + 1;
        if(OVERRIDE_3RD_ORDER_SCHEME || !swtch3)
        {
            if(orgati)
            {
                c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
            }
            else
            {
                c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));
            }
            a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
            b = DELTA(i) * DELTA(ip1) * w;
            if(c == S(0.))
            {
                if(a == S(0.))
                {
                    if(orgati)
                    {
                        a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                    }
                    else
                    {
                        a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                    }
                }
                eta = b / a;
            }
            else if(a <= S(0.))
            {
                eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            }
            else
            {
                eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
        }
        else
        {
            //
            //           Interpolation using THREE most relevant poles
            //
            temp = rhoinv + psi + phi;
            if(orgati)
            {
                temp1 = Z(iim1) / DELTA(iim1);
                temp1 = temp1 * temp1;
                c = temp - DELTA(iip1) * (dpsi + dphi) - (diim1 - diip1) * temp1;
                ZZ(1) = Z(iim1) * Z(iim1);
                ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
            }
            else
            {
                temp1 = Z(iip1) / DELTA(iip1);
                temp1 = temp1 * temp1;
                c = temp - DELTA(iim1) * (dpsi + dphi) - (diip1 - diim1) * temp1;
                ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
                ZZ(3) = Z(iip1) * Z(iip1);
            }
            ZZ(2) = Z(ii) * Z(ii);
            info = laed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin, MAXIT);
            if(info != 0)
            {
                return info;
            }
        }
        //
        //        Note, eta should be positive if w is negative, and
        //        eta should be negative otherwise. However,
        //        if for some reason caused by roundoff, eta*w > 0,
        //        we simply use one Newton step instead. This way
        //        will guarantee eta*w < 0.
        //
        if(w * eta >= S(0.))
        {
            eta = -w / dw;
        }
        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
            {
                eta = (dltub - tau) / S(2.);
            }
            else
            {
                eta = (dltlb - tau) / S(2.);
            }
        }
        prew = w;
        for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
        {
            DELTA(j) = DELTA(j) - eta;
        }
        __syncthreads();
        //
        //        Evaluate psi and the derivative dpsi
        //
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1 + threadIdx.x; j <= iim1; j += blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        reduce_block_sum<BDIM>(psi, dpsi, erretm);
        erretm = lam_abs(erretm);
        //
        //        Evaluate phi and the derivative dphi
        //
        dphi = S(0.);
        phi = S(0.);
        erretm2 = S(0.);
        for(int j = n - threadIdx.x; j >= iip1; j -= blockDim.x)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm2 = erretm2 + phi;
        }
        reduce_block_sum<BDIM>(phi, dphi, erretm2);
        erretm += erretm2;
        temp = Z(ii) / DELTA(ii);
        dw = dpsi + dphi + temp * temp;
        temp = Z(ii) * temp;
        w = rhoinv + phi + psi + temp;
        erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
            + lam_abs(tau + eta) * dw;
        swtch = false;
        if(orgati)
        {
            if(-w > lam_abs(prew) / S(10.))
            {
                swtch = true;
            }
        }
        else
        {
            if(w > lam_abs(prew) / S(10.))
            {
                swtch = true;
            }
        }
        tau = tau + eta;
        //
        //        Main loop to update the values of the array   DELTA
        //
        iter = niter + 1;
        for(niter = iter; niter < MAXIT; ++niter)
        {
            //
            //           Test for convergence
            //
            if(lam_abs(w) <= eps * erretm)
            {
                if(orgati)
                {
                    dlam = di + tau;
                }
                else
                {
                    dlam = dip1 + tau;
                }

                return info;
            }
            if(w <= S(0.))
            {
                dltlb = lam_max(dltlb, tau);
            }
            else
            {
                dltub = lam_min(dltub, tau);
            }
            //
            //           Calculate the new step
            //
            if(OVERRIDE_3RD_ORDER_SCHEME || !swtch3)
            {
                if(!swtch)
                {
                    if(orgati)
                    {
                        c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
                    }
                    else
                    {
                        c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));
                    }
                }
                else
                {
                    temp = Z(ii) / DELTA(ii);
                    if(orgati)
                    {
                        dpsi = dpsi + temp * temp;
                    }
                    else
                    {
                        dphi = dphi + temp * temp;
                    }
                    c = w - DELTA(i) * dpsi - DELTA(ip1) * dphi;
                }
                a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
                b = DELTA(i) * DELTA(ip1) * w;
                if(c == S(0.))
                {
                    if(a == S(0.))
                    {
                        if(!swtch)
                        {
                            if(orgati)
                            {
                                a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                            }
                            else
                            {
                                a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                            }
                        }
                        else
                        {
                            a = DELTA(i) * DELTA(i) * dpsi + DELTA(ip1) * DELTA(ip1) * dphi;
                        }
                    }
                    eta = b / a;
                }
                else if(a <= S(0.))
                {
                    eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
                }
                else
                {
                    eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
                }
            }
            else
            {
                //
                //              Interpolation using 3 most relevant poles
                //
                temp = rhoinv + psi + phi;
                if(swtch)
                {
                    c = temp - DELTA(iim1) * dpsi - DELTA(iip1) * dphi;
                    ZZ(1) = DELTA(iim1) * DELTA(iim1) * dpsi;
                    ZZ(3) = DELTA(iip1) * DELTA(iip1) * dphi;
                }
                else
                {
                    if(orgati)
                    {
                        temp1 = Z(iim1) / DELTA(iim1);
                        temp1 = temp1 * temp1;
                        c = temp - DELTA(iip1) * (dpsi + dphi) - (diim1 - diip1) * temp1;
                        ZZ(1) = Z(iim1) * Z(iim1);
                        ZZ(3) = DELTA(iip1) * DELTA(iip1) * ((dpsi - temp1) + dphi);
                    }
                    else
                    {
                        temp1 = Z(iip1) / DELTA(iip1);
                        temp1 = temp1 * temp1;
                        c = temp - DELTA(iim1) * (dpsi + dphi) - (diip1 - diim1) * temp1;
                        ZZ(1) = DELTA(iim1) * DELTA(iim1) * (dpsi + (dphi - temp1));
                        ZZ(3) = Z(iip1) * Z(iip1);
                    }
                }
                info = laed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin,
                             MAXIT);
                if(info != 0)
                {
                    return info;
                }
            }
            //
            //           Note, eta should be positive if w is negative, and
            //           eta should be negative otherwise. However,
            //           if for some reason caused by roundoff, eta*w > 0,
            //           we simply use one Newton step instead. This way
            //           will guarantee eta*w < 0.
            //
            if(w * eta >= S(0.))
            {
                eta = -w / dw;
            }
            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                {
                    eta = (dltub - tau) / S(2.);
                }
                else
                {
                    eta = (dltlb - tau) / S(2.);
                }
            }
            /* * */
            for(int j = 1 + threadIdx.x; j <= n; j += blockDim.x)
            {
                DELTA(j) = DELTA(j) - eta;
            }
            __syncthreads();
            tau = tau + eta;
            prew = w;
            //
            //           Evaluate psi and the derivative dpsi
            //
            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1 + threadIdx.x; j <= iim1; j += blockDim.x)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            reduce_block_sum<BDIM>(psi, dpsi, erretm);
            erretm = lam_abs(erretm);
            //
            //           Evaluate phi and the derivative dphi
            //
            dphi = S(0.);
            phi = S(0.);
            erretm2 = S(0.);
            for(int j = n - threadIdx.x; j >= iip1; j -= blockDim.x)
            {
                temp = Z(j) / DELTA(j);
                phi = phi + Z(j) * temp;
                dphi = dphi + temp * temp;
                erretm2 = erretm2 + phi;
            }
            reduce_block_sum<BDIM>(phi, dphi, erretm2);
            erretm += erretm2;
            temp = Z(ii) / DELTA(ii);
            dw = dpsi + dphi + temp * temp;
            temp = Z(ii) * temp;
            w = rhoinv + phi + psi + temp;
            erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
                + lam_abs(tau) * dw;
            if(w * prew > S(0.) && lam_abs(w) > lam_abs(prew) / S(10.))
            {
                swtch = !swtch;
            }
        }
        //
        //        Return with info = 1, niter = MAXIT and not converged
        //
        info = 1;
        if(orgati)
        {
            dlam = di + tau;
        }
        else
        {
            dlam = dip1 + tau;
        }
    }

    return info;
}

/************** Main STEDC solve kernel using laed4_alt *********************/

// Thread block size for solve kernel - MUST match BDIM template param
// When this is a multiple of 32 (e.g., 32, 64, 128), the laed4_alt
// function will use block-level reductions which may have race conditions
#define STEDC_SOLVE_BDIM 64

/** STEDC_MERGEVALUES_SOLVE_KERNEL solves the secular equation for every pair
    of sub-blocks that need to be merged.
    - Call with batch_count blocks in y, and as many blocks in x as needed
    - Uses laed4_alt which parallelizes work across BDIM threads per eigenvalue
    - When use_reference_solver=true, only thread 0 does the work (for comparison) */
template <rocblas_int BDIM /* = STEDC_SOLVE_BDIM */, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_SOLVE_BDIM)
    stedc_mergeValues_Solve_kernel(const rocblas_int n,
                                   S* DD,
                                   const rocblas_stride strideD,
                                   S* evs,        // eigenvalue output array
                                   S* etmpd,      // delta/poles array (n*n)
                                   const S* z,    // rank-1 modification vector
                                   const S* r1p,  // p values for rank-1 modification
                                   const rocblas_int* nps,   // starting positions
                                   const rocblas_int* ndd,   // degrees (non-deflated count)
                                   const S eps,
                                   const S ssfmin,
                                   const S ssfmax,
                                   const bool use_reference_solver = false)
{
    // Thread and block indices
    rocblas_int bid = blockIdx.y;  // batch instance
    rocblas_int i = blockIdx.x;    // eigenvalue index (one block per eigenvalue)

    if(i >= n)
        return;

    // Get work parameters for this eigenvalue
    S p = r1p[i];
    rocblas_int p1 = nps[i];
    rocblas_int dd = ndd[i];

    /* ----------------------------------------------------------------- */

    // 2. Solve secular eqns, i.e. find the dd zeros
    // corresponding to non-deflated new eigenvalues of the merged block
    /* ----------------------------------------------------------------- */
    // Each thread block finds one eigenvalue
    if((i - p1) < dd)
    {
        int cc = i - p1;

        // computed zero will overwrite 'ev' at the corresponding position.
        // 'etmpd' will be updated with the distances D - lambda_i.
        // deflated values are not changed.
        rocblas_int linfo;
        S dlam{};

        if(use_reference_solver)
        {
            // Reference: Only thread 0 does the computation
            // This simulates the single-threaded slaed4 behavior
            if(threadIdx.x == 0)
            {
                // Note: We still call laed4_alt but with single-thread behavior
                // by having only thread 0 do any work. This is for comparison.
                linfo = laed4_alt<S, rocblas_int, BDIM>(dd, cc, etmpd + i * n, const_cast<S*>(z) + p1, device_abs(p), dlam, eps, ssfmin);
                evs[i] = (p < 0) ? -dlam : dlam;
            }
        }
        else
        {
            // Use parallelized laed4_alt with all BDIM threads
            linfo = laed4_alt<S, rocblas_int, BDIM>(dd, cc, etmpd + i * n, const_cast<S*>(z) + p1, device_abs(p), dlam, eps, ssfmin);
            if(threadIdx.x == 0)
            {
                evs[i] = (p < 0) ? -dlam : dlam;
            }
        }
    }
}

ROCSOLVER_END_NAMESPACE
