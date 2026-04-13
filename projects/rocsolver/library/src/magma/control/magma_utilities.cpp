/*
 * Minimal utility functions needed by the MAGMA geev implementation.
 * All BLAS operations are implemented inline in C — no external CBLAS dependency.
 *
 * Copyright (c) 2009-2023, The University of Tennessee (BSD-3-Clause)
 * See MAGMA_LICENSE for details.
 */

#include "magma_internal.h"
#include <math.h>

/* -------------------------------------------------------------------- */
/* magma_*make_lwork: convert integer workspace size to floating point   */
/* -------------------------------------------------------------------- */
extern "C"
double magma_dmake_lwork(magma_int_t n) {
    return (double)n;
}

extern "C"
float magma_smake_lwork(magma_int_t n) {
    return (float)n;
}

extern "C"
magmaDoubleComplex magma_zmake_lwork(magma_int_t n) {
    magmaDoubleComplex val;
    val.x = (double)n;
    val.y = 0.0;
    return val;
}

extern "C"
magmaFloatComplex magma_cmake_lwork(magma_int_t n) {
    magmaFloatComplex val;
    val.x = (float)n;
    val.y = 0.0f;
    return val;
}

/* -------------------------------------------------------------------- */
/* magma_cblas_*: Inline BLAS implementations (no external CBLAS)        */
/* -------------------------------------------------------------------- */

extern "C"
double magma_cblas_ddot(magma_int_t n, const double *x, magma_int_t incx,
                        const double *y, magma_int_t incy) {
    double s = 0.0;
    for(magma_int_t i = 0; i < n; i++)
        s += x[i * incx] * y[i * incy];
    return s;
}

extern "C"
float magma_cblas_sdot(magma_int_t n, const float *x, magma_int_t incx,
                       const float *y, magma_int_t incy) {
    float s = 0.0f;
    for(magma_int_t i = 0; i < n; i++)
        s += x[i * incx] * y[i * incy];
    return s;
}

extern "C"
double magma_cblas_dnrm2(magma_int_t n, const double *x, magma_int_t incx) {
    double s = 0.0;
    for(magma_int_t i = 0; i < n; i++)
        s += x[i * incx] * x[i * incx];
    return sqrt(s);
}

extern "C"
float magma_cblas_snrm2(magma_int_t n, const float *x, magma_int_t incx) {
    float s = 0.0f;
    for(magma_int_t i = 0; i < n; i++)
        s += x[i * incx] * x[i * incx];
    return sqrtf(s);
}

extern "C"
double magma_cblas_dznrm2(magma_int_t n, const magmaDoubleComplex *x, magma_int_t incx) {
    double s = 0.0;
    for(magma_int_t i = 0; i < n; i++) {
        double re = x[i * incx].x, im = x[i * incx].y;
        s += re * re + im * im;
    }
    return sqrt(s);
}

extern "C"
float magma_cblas_scnrm2(magma_int_t n, const magmaFloatComplex *x, magma_int_t incx) {
    float s = 0.0f;
    for(magma_int_t i = 0; i < n; i++) {
        float re = x[i * incx].x, im = x[i * incx].y;
        s += re * re + im * im;
    }
    return sqrtf(s);
}

extern "C"
double magma_cblas_dzasum(magma_int_t n, const magmaDoubleComplex *x, magma_int_t incx) {
    double s = 0.0;
    for(magma_int_t i = 0; i < n; i++)
        s += fabs(x[i * incx].x) + fabs(x[i * incx].y);
    return s;
}

extern "C"
float magma_cblas_scasum(magma_int_t n, const magmaFloatComplex *x, magma_int_t incx) {
    float s = 0.0f;
    for(magma_int_t i = 0; i < n; i++)
        s += fabsf(x[i * incx].x) + fabsf(x[i * incx].y);
    return s;
}

extern "C"
magmaDoubleComplex magma_cblas_zdotc(magma_int_t n, const magmaDoubleComplex *x,
                                      magma_int_t incx, const magmaDoubleComplex *y,
                                      magma_int_t incy) {
    magmaDoubleComplex result = {0.0, 0.0};
    for(magma_int_t i = 0; i < n; i++) {
        // dotc: conjugate(x) * y
        double xr = x[i * incx].x, xi = x[i * incx].y;
        double yr = y[i * incy].x, yi = y[i * incy].y;
        result.x += xr * yr + xi * yi;
        result.y += xr * yi - xi * yr;
    }
    return result;
}

extern "C"
magmaDoubleComplex magma_cblas_zdotu(magma_int_t n, const magmaDoubleComplex *x,
                                      magma_int_t incx, const magmaDoubleComplex *y,
                                      magma_int_t incy) {
    magmaDoubleComplex result = {0.0, 0.0};
    for(magma_int_t i = 0; i < n; i++) {
        double xr = x[i * incx].x, xi = x[i * incx].y;
        double yr = y[i * incy].x, yi = y[i * incy].y;
        result.x += xr * yr - xi * yi;
        result.y += xr * yi + xi * yr;
    }
    return result;
}

extern "C"
magmaFloatComplex magma_cblas_cdotc(magma_int_t n, const magmaFloatComplex *x,
                                     magma_int_t incx, const magmaFloatComplex *y,
                                     magma_int_t incy) {
    magmaFloatComplex result = {0.0f, 0.0f};
    for(magma_int_t i = 0; i < n; i++) {
        float xr = x[i * incx].x, xi = x[i * incx].y;
        float yr = y[i * incy].x, yi = y[i * incy].y;
        result.x += xr * yr + xi * yi;
        result.y += xr * yi - xi * yr;
    }
    return result;
}

extern "C"
magmaFloatComplex magma_cblas_cdotu(magma_int_t n, const magmaFloatComplex *x,
                                     magma_int_t incx, const magmaFloatComplex *y,
                                     magma_int_t incy) {
    magmaFloatComplex result = {0.0f, 0.0f};
    for(magma_int_t i = 0; i < n; i++) {
        float xr = x[i * incx].x, xi = x[i * incx].y;
        float yr = y[i * incy].x, yi = y[i * incy].y;
        result.x += xr * yr - xi * yi;
        result.y += xr * yi + xi * yr;
    }
    return result;
}
