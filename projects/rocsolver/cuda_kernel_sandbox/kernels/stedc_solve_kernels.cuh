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
 * - slaed4: Computes the i-th eigenvalue of a positive symmetric rank-1 modification
 *           of a 2x2 diagonal matrix (calls slaed6 for 3-pole interpolation)
 * - seq_eval: Evaluates secular equation at a given point
 * - seq_solve: Solves secular equation for internal eigenvalues
 * - seq_solve_ext: Solves secular equation for the last eigenvalue
 * - stedc_mergeValues_Solve_kernel: Main kernel that calls the solvers
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

/************** Device functions for solving secular equations ****************/

/** SLAED6 solves the modified secular equation using 3 poles
    This is based on the Gragg-Thornton-Warner cubic convergent scheme **/
template <typename S, typename I>
__device__ I slaed6(I kniter,
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
    // Lambda functions for device-compatible math operations
    auto lam_abs = [](auto x) -> auto { return device_abs(x); };
    auto lam_sqrt = [](auto x) -> auto { return sqrt(x); };
    auto lam_max = [](auto x, auto y, auto zz) -> auto { return device_max(device_max(x, y), zz); };
    auto lam_min = [](auto x, auto y) -> auto { return device_min(x, y); };

    S dscale[3]{}, zscale[3]{};

    // 1-indexed accessor struct
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

    // Get machine parameters for possible scaling to avoid overflow
    const S small1 = pow(S(2.), (device_log(ssfmin) / device_log(S(2.))) / S(3.));
    const S sminv1 = S(1.) / small1;
    const S small2 = small1 * small1;
    const S sminv2 = sminv1 * sminv1;

    // Determine if scaling of inputs necessary to avoid overflow
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

    // Iteration using Gragg-Thornton-Warner cubic convergent scheme
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

/** SLAED4 computes the i-th eigenvalue of a positive symmetric rank-1 modification
    of a 2x2 diagonal matrix: D + rho * z * z^T
    This implementation uses 1-indexed arrays internally **/
template <typename S, typename I>
__device__ I slaed4(I n,
                    I i,
                    S* delta,
                    S* z,
                    S rho,
                    S& dlam,
                    S eps,
                    S ssfmin,
                    I MAXIT = 50)
{
    auto lam_abs = [](auto x) -> auto { return device_abs(x); };
    auto lam_sqr = [](auto x) -> auto { return x * x; };
    auto lam_sqrt = [](auto x) -> auto { return sqrt(x); };
    auto lam_max = [](auto x, auto y) -> auto { return device_max(x, y); };
    auto lam_min = [](auto x, auto y) -> auto { return device_min(x, y); };

    i = i + 1;  // Convert to 1-indexed
    S zz[3]{};

    struct X_t
    {
        S* x_;
        __device__ X_t(S* x) : x_(x) {}
        __device__ S& operator()(int j) { return x_[j - 1]; }
    } Z(z), ZZ(zz), DELTA(delta);

    S tau, eta = S(0.), dltlb, dltub;
    S psi, dpsi, phi, dphi, rhoinv, midpt;
    S del, a, b, c, w, erretm, temp, dw, temp1, prew;
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
        // Case for last eigenvalue
        ii = n - 1;
        niter = 1;
        midpt = rho / S(2.);

        psi = S(0.);
        for(int j = 1; j <= n - 2; ++j)
            psi = psi + Z(j) * Z(j) / ((DELTA(j) - di) - midpt);

        c = rhoinv + psi;
        w = c + Z(ii) * Z(ii) / ((DELTA(ii) - di) - midpt) + Z(n) * Z(n) / ((dn - di) - midpt);

        if(w <= S(0.))
        {
            temp = Z(n - 1) * Z(n - 1) / (dn - dnm1 + rho) + Z(n) * Z(n) / rho;
            if(c <= temp)
                tau = rho;
            else
            {
                del = dn - dnm1;
                a = -c * del + Z(n - 1) * Z(n - 1) + Z(n) * Z(n);
                b = Z(n) * Z(n) * del;
                if(a < S(0.))
                    tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
                else
                    tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
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
                tau = S(2.) * b / (lam_sqrt(a * a + S(4.) * b * c) - a);
            else
                tau = (a + lam_sqrt(a * a + S(4.) * b * c)) / (S(2.) * c);
            dltlb = S(0.);
            dltub = midpt;
        }

        for(int j = 1; j <= n; ++j)
            DELTA(j) = (DELTA(j) - di) - tau;

        // Evaluate psi and derivative
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= ii; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);

        // Evaluate phi and derivative
        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;

        if(lam_abs(w) <= eps * erretm)
        {
            dlam = di + tau;
            return info;
        }

        if(w <= S(0.))
            dltlb = lam_max(dltlb, tau);
        else
            dltub = lam_min(dltub, tau);

        // Calculate new step
        niter = niter + 1;
        c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
        a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
        b = DELTA(n - 1) * DELTA(n) * w;

        if(c < S(0.))
            c = lam_abs(c);

        if(c <= S(0.))
            eta = -w / (dpsi + dphi);
        else if(a >= S(0.))
            eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
        else
            eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));

        if(w * eta > S(0.))
            eta = -w / (dpsi + dphi);

        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
                eta = (dltub - tau) / S(2.);
            else
                eta = (dltlb - tau) / S(2.);
        }

        for(int j = 1; j <= n; ++j)
            DELTA(j) = DELTA(j) - eta;
        tau = tau + eta;

        // Evaluate psi and derivative again
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= ii; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);

        temp = Z(n) / DELTA(n);
        phi = Z(n) * temp;
        dphi = temp * temp;
        erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
        w = rhoinv + phi + psi;

        // Main iteration loop
        iter = niter + 1;
        for(niter = iter; niter <= MAXIT; ++niter)
        {
            if(lam_abs(w) <= eps * erretm)
            {
                dlam = di + tau;
                return info;
            }
            if(w <= S(0.))
                dltlb = lam_max(dltlb, tau);
            else
                dltub = lam_min(dltub, tau);

            c = w - DELTA(n - 1) * dpsi - DELTA(n) * dphi;
            a = (DELTA(n - 1) + DELTA(n)) * w - DELTA(n - 1) * DELTA(n) * (dpsi + dphi);
            b = DELTA(n - 1) * DELTA(n) * w;

            if(a >= S(0.))
                eta = (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            else
                eta = S(2.) * b / (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c)));

            if(w * eta > S(0.))
                eta = -w / (dpsi + dphi);

            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                    eta = (dltub - tau) / S(2.);
                else
                    eta = (dltlb - tau) / S(2.);
            }

            for(int j = 1; j <= n; ++j)
                DELTA(j) = DELTA(j) - eta;
            tau = tau + eta;

            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1; j <= ii; ++j)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = lam_abs(erretm);

            temp = Z(n) / DELTA(n);
            phi = Z(n) * temp;
            dphi = temp * temp;
            erretm = S(8.) * (-phi - psi) + erretm - phi + rhoinv + lam_abs(tau) * (dpsi + dphi);
            w = rhoinv + phi + psi;
        }

        info = 1;
        dlam = di + tau;
    }
    else
    {
        // Case for i < n (interior eigenvalue)
        niter = 1;
        I ip1 = i + 1;
        S dip1 = DELTA(ip1);

        del = dip1 - di;
        midpt = del / S(2.);
        psi = S(0.);
        for(int j = 1; j <= i - 1; ++j)
        {
            S dj = (DELTA(j) - di) - midpt;
            psi = psi + Z(j) * Z(j) / dj;
        }
        phi = S(0.);
        for(int j = n; j >= i + 2; --j)
        {
            S dj = (DELTA(j) - di) - midpt;
            phi = phi + Z(j) * Z(j) / dj;
        }
        c = rhoinv + psi + phi;
        w = c + Z(i) * Z(i) / (-midpt) + Z(ip1) * Z(ip1) / ((dip1 - di) - midpt);

        if(w > S(0.))
        {
            orgati = true;
            a = c * del + Z(i) * Z(i) + Z(ip1) * Z(ip1);
            b = Z(i) * Z(i) * del;
            if(a > S(0.))
                tau = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            else
                tau = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            dltlb = S(0.);
            dltub = midpt;
        }
        else
        {
            orgati = false;
            a = c * del - Z(i) * Z(i) - Z(ip1) * Z(ip1);
            b = Z(ip1) * Z(ip1) * del;
            if(a < S(0.))
                tau = S(2.) * b / (a - lam_sqrt(lam_abs(a * a + S(4.) * b * c)));
            else
                tau = -(a + lam_sqrt(lam_abs(a * a + S(4.) * b * c))) / (S(2.) * c);
            dltlb = -midpt;
            dltub = S(0.);
        }

        if(orgati)
            ii = i;
        else
            ii = i + 1;

        iim1 = ii - 1;
        iip1 = ii + 1;
        S diim1 = DELTA(iim1);
        S diip1 = DELTA(iip1);

        if(orgati)
        {
            for(int j = 1; j <= n; ++j)
                DELTA(j) = (DELTA(j) - di) - tau;
        }
        else
        {
            for(int j = 1; j <= n; ++j)
                DELTA(j) = (DELTA(j) - dip1) - tau;
        }

        // Evaluate psi and derivative
        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= iim1; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);

        // Evaluate phi and derivative
        dphi = S(0.);
        phi = S(0.);
        for(int j = n; j >= iip1; --j)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }
        w = rhoinv + phi + psi;

        swtch3 = false;
        if(orgati)
        {
            if(w < S(0.))
                swtch3 = true;
        }
        else
        {
            if(w > S(0.))
                swtch3 = true;
        }
        if(ii == 1 || ii == n)
            swtch3 = false;

        temp = Z(ii) / DELTA(ii);
        dw = dpsi + dphi + temp * temp;
        temp = Z(ii) * temp;
        w = w + temp;
        erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
            + lam_abs(tau) * dw;

        if(lam_abs(w) <= eps * erretm)
        {
            if(orgati)
                dlam = di + tau;
            else
                dlam = dip1 + tau;
            return info;
        }

        if(w <= S(0.))
            dltlb = lam_max(dltlb, tau);
        else
            dltub = lam_min(dltub, tau);

        // Calculate new step
        niter = niter + 1;
        if(!swtch3)
        {
            if(orgati)
                c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
            else
                c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));

            a = (DELTA(i) + DELTA(ip1)) * w - DELTA(i) * DELTA(ip1) * dw;
            b = DELTA(i) * DELTA(ip1) * w;

            if(c == S(0.))
            {
                if(a == S(0.))
                {
                    if(orgati)
                        a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                    else
                        a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                }
                eta = b / a;
            }
            else if(a <= S(0.))
                eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
            else
                eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
        }
        else
        {
            // 3-pole interpolation
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
            info = slaed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin, MAXIT);
            if(info != 0)
                return info;
        }

        if(w * eta >= S(0.))
            eta = -w / dw;

        temp = tau + eta;
        if(temp > dltub || temp < dltlb)
        {
            if(w < S(0.))
                eta = (dltub - tau) / S(2.);
            else
                eta = (dltlb - tau) / S(2.);
        }

        prew = w;
        for(int j = 1; j <= n; ++j)
            DELTA(j) = DELTA(j) - eta;

        dpsi = S(0.);
        psi = S(0.);
        erretm = S(0.);
        for(int j = 1; j <= iim1; ++j)
        {
            temp = Z(j) / DELTA(j);
            psi = psi + Z(j) * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = lam_abs(erretm);

        dphi = S(0.);
        phi = S(0.);
        for(int j = n; j >= iip1; --j)
        {
            temp = Z(j) / DELTA(j);
            phi = phi + Z(j) * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }

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
                swtch = true;
        }
        else
        {
            if(w > lam_abs(prew) / S(10.))
                swtch = true;
        }
        tau = tau + eta;

        // Main iteration loop
        iter = niter + 1;
        for(niter = iter; niter < MAXIT; ++niter)
        {
            if(lam_abs(w) <= eps * erretm)
            {
                if(orgati)
                    dlam = di + tau;
                else
                    dlam = dip1 + tau;
                return info;
            }

            if(w <= S(0.))
                dltlb = lam_max(dltlb, tau);
            else
                dltub = lam_min(dltub, tau);

            if(!swtch3)
            {
                if(!swtch)
                {
                    if(orgati)
                        c = w - DELTA(ip1) * dw - (di - dip1) * lam_sqr(Z(i) / DELTA(i));
                    else
                        c = w - DELTA(i) * dw - (dip1 - di) * lam_sqr(Z(ip1) / DELTA(ip1));
                }
                else
                {
                    temp = Z(ii) / DELTA(ii);
                    if(orgati)
                        dpsi = dpsi + temp * temp;
                    else
                        dphi = dphi + temp * temp;
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
                                a = Z(i) * Z(i) + DELTA(ip1) * DELTA(ip1) * (dpsi + dphi);
                            else
                                a = Z(ip1) * Z(ip1) + DELTA(i) * DELTA(i) * (dpsi + dphi);
                        }
                        else
                            a = DELTA(i) * DELTA(i) * dpsi + DELTA(ip1) * DELTA(ip1) * dphi;
                    }
                    eta = b / a;
                }
                else if(a <= S(0.))
                    eta = (a - lam_sqrt(lam_abs(a * a - S(4.) * b * c))) / (S(2.) * c);
                else
                    eta = S(2.) * b / (a + lam_sqrt(lam_abs(a * a - S(4.) * b * c)));
            }
            else
            {
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
                ZZ(2) = Z(ii) * Z(ii);
                info = slaed6(niter, orgati, c, DELTA.x_ + iim1 - 1, ZZ.x_, w, eta, eps, ssfmin, MAXIT);
                if(info != 0)
                    return info;
            }

            if(w * eta >= S(0.))
                eta = -w / dw;

            temp = tau + eta;
            if(temp > dltub || temp < dltlb)
            {
                if(w < S(0.))
                    eta = (dltub - tau) / S(2.);
                else
                    eta = (dltlb - tau) / S(2.);
            }

            for(int j = 1; j <= n; ++j)
                DELTA(j) = DELTA(j) - eta;
            tau = tau + eta;
            prew = w;

            dpsi = S(0.);
            psi = S(0.);
            erretm = S(0.);
            for(int j = 1; j <= iim1; ++j)
            {
                temp = Z(j) / DELTA(j);
                psi = psi + Z(j) * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = lam_abs(erretm);

            dphi = S(0.);
            phi = S(0.);
            for(int j = n; j >= iip1; --j)
            {
                temp = Z(j) / DELTA(j);
                phi = phi + Z(j) * temp;
                dphi = dphi + temp * temp;
                erretm = erretm + phi;
            }

            temp = Z(ii) / DELTA(ii);
            dw = dpsi + dphi + temp * temp;
            temp = Z(ii) * temp;
            w = rhoinv + phi + psi + temp;
            erretm = S(8.) * (phi - psi) + erretm + S(2.) * rhoinv + S(3.) * lam_abs(temp)
                + lam_abs(tau) * dw;

            if(w * prew > S(0.) && lam_abs(w) > lam_abs(prew) / S(10.))
                swtch = !swtch;
        }

        info = 1;
        if(orgati)
            dlam = di + tau;
        else
            dlam = dip1 + tau;
    }

    return info;
}

/** SEQ_EVAL evaluates the secular equation at a given point **/
template <typename S>
__device__ void seq_eval(const rocblas_int type,
                         const rocblas_int k,
                         const rocblas_int dd,
                         S* D,
                         const S* z,
                         const S p,
                         const S cor,
                         S* pt_fx,
                         S* pt_fdx,
                         S* pt_gx,
                         S* pt_gdx,
                         S* pt_hx,
                         S* pt_hdx,
                         S* pt_er,
                         bool modif)
{
    S er, fx, gx, hx, fdx, gdx, hdx, zz, tmp;
    rocblas_int gout, hout;

    if(type == 0)
    {
        gout = k + 1;
        hout = k;
    }
    else if(type == 1)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
        }
        gout = k;
        hout = k;
    }
    else if(type == 2)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
            tmp = D[k + 1] - cor;
            D[k + 1] = tmp;
        }
        gout = k;
        hout = k + 1;
    }
    else
    {
        gout = k;
        hout = k;
    }

    gx = 0;
    gdx = 0;
    er = 0;
    for(int i = 0; i < gout; ++i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        gx += zz * tmp;
        gdx += tmp * tmp;
        er += gx;
    }
    er = device_abs(er);

    hx = 0;
    hdx = 0;
    for(int i = dd - 1; i > hout; --i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        hx += zz * tmp;
        hdx += tmp * tmp;
        er += hx;
    }

    fx = p + gx + hx;
    fdx = gdx + hdx;

    *pt_fx = fx;
    *pt_fdx = fdx;
    *pt_gx = gx;
    *pt_gdx = gdx;
    *pt_hx = hx;
    *pt_hdx = hdx;
    *pt_er = er;
}

/** SEQ_SOLVE solves secular equation at point k (internal eigenvalue) **/
template <typename S>
__device__ rocblas_int seq_solve(const rocblas_int dd,
                                 S* D,
                                 const S* z,
                                 const S p,
                                 rocblas_int k,
                                 S* ev,
                                 const S tol,
                                 const S ssfmin,
                                 const S ssfmax)
{
    bool converged = false;
    bool up, fixed;
    S lowb, uppb, aa, bb, cc, x;
    S nx, er, fx, fdx, gx, gdx, hx, hdx, oldfx;
    S tau, eta;
    S dk, dk1, ddk, ddk1;
    rocblas_int kk;
    rocblas_int k1 = k + 1;

    dk = D[k];
    dk1 = D[k1];
    x = (dk + dk1) / 2;
    tau = (dk1 - dk);
    S pinv = 1 / p;

    seq_eval(2, k, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[k] * z[k];
    hdx = z[k1] * z[k1];
    fx = cc + 2 * (hdx - gdx) / tau;

    if(fx > 0)
    {
        lowb = 0;
        uppb = tau / 2;
        up = true;
        kk = k;
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(device_abs(aa * aa - 4 * bb * cc));
        if(aa > 0)
            tau = 2 * bb / (aa + eta);
        else
            tau = (aa - eta) / (2 * cc);
        x = dk + tau;
    }
    else
    {
        lowb = -tau / 2;
        uppb = 0;
        up = false;
        kk = k + 1;
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(device_abs(aa * aa + 4 * bb * cc));
        if(aa < 0)
            tau = 2 * bb / (aa - eta);
        else
            tau = -(aa + eta) / (2 * cc);
        x = dk1 + tau;
    }

    seq_eval(0, kk, dd, D, z, pinv, (up ? dk : dk1), &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(1, kk, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    bb = z[kk];
    aa = bb / D[kk];
    fdx += aa * aa;
    bb *= aa;
    fx += bb;

    er += 8 * (hx - gx) + 2 * pinv + 3 * device_abs(bb) + device_abs(tau) * fdx;

    if(device_abs(fx) <= tol * er)
        converged = true;
    else
    {
        lowb = (fx <= 0) ? device_max(lowb, tau) : lowb;
        uppb = (fx > 0) ? device_min(uppb, tau) : uppb;

        ddk = D[k];
        ddk1 = D[k1];
        if(up)
            cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
        else
            cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;

        if(cc == 0)
        {
            if(aa == 0)
            {
                if(up)
                    aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                else
                    aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        }
        else
        {
            eta = sqrt(device_abs(aa * aa - 4 * bb * cc));
            if(aa <= 0)
                eta = (aa - eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa + eta);
        }

        if(fx * eta >= 0)
            eta = -fx / fdx;

        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        tau += eta;
        x = (up ? dk : dk1) + tau;

        oldfx = fx;
        seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
        bb = z[kk];
        aa = bb / D[kk];
        fdx += aa * aa;
        bb *= aa;
        fx += bb;

        er += 8 * (hx - gx) + 2 * pinv + 3 * device_abs(bb) + device_abs(tau) * fdx;

        cc = up ? -1 : 1;
        fixed = (cc * fx) > (device_abs(oldfx) / 10);

        // Main iteration loop
        for(int i = 1; i < MAXITERS; ++i)
        {
            if(device_abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            lowb = (fx <= 0) ? device_max(lowb, tau) : lowb;
            uppb = (fx > 0) ? device_min(uppb, tau) : uppb;

            ddk = D[k];
            ddk1 = D[k1];
            if(fixed)
            {
                if(up)
                    cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
                else
                    cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
            }
            else
            {
                if(up)
                    gdx += aa * aa;
                else
                    hdx += aa * aa;
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;

            if(cc == 0)
            {
                if(aa == 0)
                {
                    if(fixed)
                    {
                        if(up)
                            aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                        else
                            aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
                    }
                    else
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                }
                eta = bb / aa;
            }
            else
            {
                eta = sqrt(device_abs(aa * aa - 4 * bb * cc));
                if(aa <= 0)
                    eta = (aa - eta) / (2 * cc);
                else
                    eta = (2 * bb) / (aa + eta);
            }

            if(fx * eta >= 0)
                eta = -fx / fdx;

            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            tau += eta;
            x = (up ? dk : dk1) + tau;

            oldfx = fx;
            seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
            bb = z[kk];
            aa = bb / D[kk];
            fdx += aa * aa;
            bb *= aa;
            fx += bb;

            er += 8 * (hx - gx) + 2 * pinv + 3 * device_abs(bb) + device_abs(tau) * fdx;

            if(fx * oldfx > 0 && device_abs(fx) > device_abs(oldfx) / 10)
                fixed = !fixed;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

/** SEQ_SOLVE_EXT solves secular equation at point n (last eigenvalue) **/
template <typename S>
__device__ rocblas_int seq_solve_ext(const rocblas_int dd,
                                     S* D,
                                     const S* z,
                                     const S p,
                                     S* ev,
                                     const S tol,
                                     const S ssfmin,
                                     const S ssfmax)
{
    bool converged = false;
    S lowb, uppb, aa, bb, cc, x;
    S er, fx, fdx, gx, gdx, hx, hdx;
    S tau, eta;
    S dk, dkm1, ddk, ddkm1;
    rocblas_int k = dd - 1;
    rocblas_int km1 = dd - 2;

    dk = D[k];
    dkm1 = D[km1];
    x = dk + p / 2;
    S pinv = 1 / p;

    seq_eval(2, km1, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[km1] * z[km1];
    hdx = z[k] * z[k];
    fx = cc + gdx / (dkm1 - x) - 2 * hdx * pinv;

    if(fx > 0)
    {
        lowb = 0;
        uppb = p / 2;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = sqrt(aa * aa + 4 * bb * cc);
        if(aa < 0)
            tau = 2 * bb / (eta - aa);
        else
            tau = (aa + eta) / (2 * cc);
    }
    else
    {
        lowb = p / 2;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if(cc <= eta)
            tau = p;
        else
        {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = sqrt(aa * aa + 4 * bb * cc);
            if(aa < 0)
                tau = 2 * bb / (eta - aa);
            else
                tau = (aa + eta) / (2 * cc);
        }
    }
    x = dk + tau;

    seq_eval(0, km1, dd, D, z, pinv, dk, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(0, km1, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

    er += device_abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

    if(device_abs(fx) <= tol * er)
        converged = true;
    else
    {
        lowb = (fx <= 0) ? device_max(lowb, tau) : lowb;
        uppb = (fx > 0) ? device_min(uppb, tau) : uppb;

        ddk = D[k];
        ddkm1 = D[km1];
        cc = device_abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;

        if(cc == 0)
            eta = uppb - tau;
        else
        {
            eta = sqrt(device_abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);
        }

        if(fx * eta > 0)
            eta = -fx / (gdx + hdx);

        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        tau += eta;
        x = dk + tau;

        seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
        er += device_abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

        // Main iteration loop
        for(int i = 1; i < MAXITERS; ++i)
        {
            if(device_abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            lowb = (fx <= 0) ? device_max(lowb, tau) : lowb;
            uppb = (fx > 0) ? device_min(uppb, tau) : uppb;

            ddk = D[k];
            ddkm1 = D[km1];
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = sqrt(device_abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);

            if(fx * eta > 0)
                eta = -fx / (gdx + hdx);

            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            tau += eta;
            x = dk + tau;

            seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
            er += device_abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

/************** Main STEDC solve kernel ********************************************/

#define STEDC_SOLVE_BDIM 4  // Thread block size for solve kernel

/** STEDC_MERGEVALUES_SOLVE_KERNEL solves the secular equation for every pair
    of sub-blocks that need to be merged.
    - Each thread finds a different eigenvalue in parallel
    - Calls slaed4 (reference solver) or seq_solve/seq_solve_ext (optimized) **/
template <typename S>
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
    rocblas_int bid = hipBlockIdx_y;  // batch instance
    rocblas_int i = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;

    if(i >= n)
        return;

    // Select batch instance
    S* D = DD + bid * strideD;

    // Get work arrays for this eigenvalue
    S p = r1p[i];
    rocblas_int p1 = nps[i];
    rocblas_int dd = ndd[i];

    // Solve secular equation if this is a non-deflated eigenvalue
    rocblas_int cc = i - p1;
    if(cc < dd)
    {
        rocblas_int linfo;

        if(use_reference_solver)
        {
            // Use reference LAPACK-style solver (slaed4)
            linfo = slaed4(dd, cc, etmpd + i * n, const_cast<S*>(z) + p1,
                          device_abs(p), evs[i], eps, ssfmin, 50);
        }
        else
        {
            // Use optimized solver
            if(cc == dd - 1)
                linfo = seq_solve_ext(dd, etmpd + i * n, z + p1, device_abs(p),
                                      evs + i, eps, ssfmin, ssfmax);
            else
                linfo = seq_solve(dd, etmpd + i * n, z + p1, device_abs(p), cc,
                                  evs + i, eps, ssfmin, ssfmax);
        }

        // Apply sign correction
        if(p < 0)
            evs[i] *= -1;
    }
}

/** Simple test kernel that just calls slaed4 to verify it works **/
template <typename S>
ROCSOLVER_KERNEL void test_slaed4_kernel(const rocblas_int n,
                                         S* delta,
                                         S* z,
                                         S rho,
                                         S* dlam,
                                         rocblas_int* info,
                                         const S eps,
                                         const S ssfmin)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        // Each thread computes one eigenvalue
        S lambda;
        rocblas_int linfo = slaed4<S, rocblas_int>(n, tid, delta + tid * n, z, rho, lambda,
                                                    eps, ssfmin, 50);
        dlam[tid] = lambda;
        info[tid] = linfo;
    }
}

ROCSOLVER_END_NAMESPACE
