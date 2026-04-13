/*
    Host-side LAPACK/BLAS implementations for MAGMA's geev pipeline.
    These replace external LAPACK/BLAS library dependencies entirely.

    Reference: LAPACK 3.x (BSD-3-Clause, Univ. of Tennessee)
    Translated to C++ for use in rocsolver's MAGMA integration.

    All functions use Fortran calling convention:
      - extern "C" with trailing underscore (ADD_ mangling)
      - pass-by-pointer for all scalar arguments
      - column-major storage

    Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: BSD-3-Clause
*/

#include <cmath>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <algorithm>
#include <vector>

/* ========================================================================
   Complex type definitions matching magmaFloatComplex / magmaDoubleComplex
   (structs with .x and .y fields)
   ======================================================================== */

struct lapack_float_complex {
    float x, y;
};

struct lapack_double_complex {
    double x, y;
};

/* Helpers for complex arithmetic */
static inline lapack_float_complex cf_make(float r, float i) { return {r, i}; }
static inline lapack_double_complex zf_make(double r, double i) { return {r, i}; }

static inline lapack_float_complex cf_add(lapack_float_complex a, lapack_float_complex b) { return {a.x+b.x, a.y+b.y}; }
static inline lapack_double_complex zf_add(lapack_double_complex a, lapack_double_complex b) { return {a.x+b.x, a.y+b.y}; }

static inline lapack_float_complex cf_sub(lapack_float_complex a, lapack_float_complex b) { return {a.x-b.x, a.y-b.y}; }
static inline lapack_double_complex zf_sub(lapack_double_complex a, lapack_double_complex b) { return {a.x-b.x, a.y-b.y}; }

static inline lapack_float_complex cf_mul(lapack_float_complex a, lapack_float_complex b) {
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
static inline lapack_double_complex zf_mul(lapack_double_complex a, lapack_double_complex b) {
    return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}

static inline lapack_float_complex cf_div(lapack_float_complex a, lapack_float_complex b) {
    float d = b.x*b.x + b.y*b.y;
    return {(a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d};
}
static inline lapack_double_complex zf_div(lapack_double_complex a, lapack_double_complex b) {
    double d = b.x*b.x + b.y*b.y;
    return {(a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d};
}

static inline lapack_float_complex cf_conj(lapack_float_complex a) { return {a.x, -a.y}; }
static inline lapack_double_complex zf_conj(lapack_double_complex a) { return {a.x, -a.y}; }

static inline float cf_abs1(lapack_float_complex a) { return fabsf(a.x) + fabsf(a.y); }
static inline double zf_abs1(lapack_double_complex a) { return fabs(a.x) + fabs(a.y); }

static inline float cf_abs(lapack_float_complex a) { return hypotf(a.x, a.y); }
static inline double zf_abs(lapack_double_complex a) { return hypot(a.x, a.y); }

static inline lapack_float_complex cf_scale(float s, lapack_float_complex a) { return {s*a.x, s*a.y}; }
static inline lapack_double_complex zf_scale(double s, lapack_double_complex a) { return {s*a.x, s*a.y}; }

/* Helper to check character arguments (case-insensitive) */
static inline bool lsame(char a, char b) {
    return (a == b) || (a == (b ^ 32)) || ((a ^ 32) == b);
}

static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }

/* ========================================================================
   BLAS Level 1
   ======================================================================== */

extern "C" {

/* ----- AXPY: y = alpha*x + y ----- */

void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy) {
    float a = *alpha;
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] += a * x[i * (*incx)];
}

void daxpy_(const int *n, const double *alpha, const double *x, const int *incx,
            double *y, const int *incy) {
    double a = *alpha;
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] += a * x[i * (*incx)];
}

void caxpy_(const int *n, const lapack_float_complex *alpha,
            const lapack_float_complex *x, const int *incx,
            lapack_float_complex *y, const int *incy) {
    lapack_float_complex a = *alpha;
    for (int i = 0; i < *n; i++) {
        lapack_float_complex t = cf_mul(a, x[i * (*incx)]);
        y[i * (*incy)] = cf_add(y[i * (*incy)], t);
    }
}

void zaxpy_(const int *n, const lapack_double_complex *alpha,
            const lapack_double_complex *x, const int *incx,
            lapack_double_complex *y, const int *incy) {
    lapack_double_complex a = *alpha;
    for (int i = 0; i < *n; i++) {
        lapack_double_complex t = zf_mul(a, x[i * (*incx)]);
        y[i * (*incy)] = zf_add(y[i * (*incy)], t);
    }
}

/* ----- COPY: y = x ----- */

void scopy_(const int *n, const float *x, const int *incx, float *y, const int *incy) {
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] = x[i * (*incx)];
}

void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy) {
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] = x[i * (*incx)];
}

void ccopy_(const int *n, const lapack_float_complex *x, const int *incx,
            lapack_float_complex *y, const int *incy) {
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] = x[i * (*incx)];
}

void zcopy_(const int *n, const lapack_double_complex *x, const int *incx,
            lapack_double_complex *y, const int *incy) {
    for (int i = 0; i < *n; i++)
        y[i * (*incy)] = x[i * (*incx)];
}

/* ----- SCAL: x = alpha*x ----- */

void sscal_(const int *n, const float *alpha, float *x, const int *incx) {
    float a = *alpha;
    for (int i = 0; i < *n; i++)
        x[i * (*incx)] *= a;
}

void dscal_(const int *n, const double *alpha, double *x, const int *incx) {
    double a = *alpha;
    for (int i = 0; i < *n; i++)
        x[i * (*incx)] *= a;
}

void cscal_(const int *n, const lapack_float_complex *alpha,
            lapack_float_complex *x, const int *incx) {
    lapack_float_complex a = *alpha;
    for (int i = 0; i < *n; i++)
        x[i * (*incx)] = cf_mul(a, x[i * (*incx)]);
}

void zscal_(const int *n, const lapack_double_complex *alpha,
            lapack_double_complex *x, const int *incx) {
    lapack_double_complex a = *alpha;
    for (int i = 0; i < *n; i++)
        x[i * (*incx)] = zf_mul(a, x[i * (*incx)]);
}

/* csscal: scale complex by real */
void csscal_(const int *n, const float *alpha, lapack_float_complex *x, const int *incx) {
    float a = *alpha;
    for (int i = 0; i < *n; i++) {
        x[i * (*incx)].x *= a;
        x[i * (*incx)].y *= a;
    }
}

/* zdscal: scale double complex by double */
void zdscal_(const int *n, const double *alpha, lapack_double_complex *x, const int *incx) {
    double a = *alpha;
    for (int i = 0; i < *n; i++) {
        x[i * (*incx)].x *= a;
        x[i * (*incx)].y *= a;
    }
}

/* ----- ROT: Givens rotation ----- */

void srot_(const int *n, float *x, const int *incx, float *y, const int *incy,
           const float *c, const float *s) {
    float cc = *c, ss = *s;
    for (int i = 0; i < *n; i++) {
        float xi = x[i * (*incx)];
        float yi = y[i * (*incy)];
        x[i * (*incx)] = cc * xi + ss * yi;
        y[i * (*incy)] = cc * yi - ss * xi;
    }
}

void drot_(const int *n, double *x, const int *incx, double *y, const int *incy,
           const double *c, const double *s) {
    double cc = *c, ss = *s;
    for (int i = 0; i < *n; i++) {
        double xi = x[i * (*incx)];
        double yi = y[i * (*incy)];
        x[i * (*incx)] = cc * xi + ss * yi;
        y[i * (*incy)] = cc * yi - ss * xi;
    }
}

/* ----- IAMAX: index of max abs element (1-indexed!) ----- */

int isamax_(const int *n, const float *x, const int *incx) {
    if (*n < 1) return 0;
    int idx = 1;
    float mx = fabsf(x[0]);
    for (int i = 1; i < *n; i++) {
        float v = fabsf(x[i * (*incx)]);
        if (v > mx) { mx = v; idx = i + 1; }
    }
    return idx;
}

int idamax_(const int *n, const double *x, const int *incx) {
    if (*n < 1) return 0;
    int idx = 1;
    double mx = fabs(x[0]);
    for (int i = 1; i < *n; i++) {
        double v = fabs(x[i * (*incx)]);
        if (v > mx) { mx = v; idx = i + 1; }
    }
    return idx;
}

int icamax_(const int *n, const lapack_float_complex *x, const int *incx) {
    if (*n < 1) return 0;
    int idx = 1;
    float mx = cf_abs1(x[0]);
    for (int i = 1; i < *n; i++) {
        float v = cf_abs1(x[i * (*incx)]);
        if (v > mx) { mx = v; idx = i + 1; }
    }
    return idx;
}

int izamax_(const int *n, const lapack_double_complex *x, const int *incx) {
    if (*n < 1) return 0;
    int idx = 1;
    double mx = zf_abs1(x[0]);
    for (int i = 1; i < *n; i++) {
        double v = zf_abs1(x[i * (*incx)]);
        if (v > mx) { mx = v; idx = i + 1; }
    }
    return idx;
}

/* ========================================================================
   BLAS Level 2 - GEMV
   ======================================================================== */

void sgemv_(const char *trans, const int *m, const int *n,
            const float *alpha, const float *A, const int *lda,
            const float *x, const int *incx,
            const float *beta, float *y, const int *incy) {
    int M = *m, N = *n, LDA = *lda;
    float a = *alpha, b = *beta;
    bool notrans = lsame(*trans, 'N');
    int leny = notrans ? M : N;
    int lenx = notrans ? N : M;

    /* Scale y by beta */
    if (b == 0.0f) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = 0.0f;
    } else if (b != 1.0f) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] *= b;
    }
    if (a == 0.0f) return;

    if (notrans) {
        /* y = alpha * A * x + y */
        for (int j = 0; j < N; j++) {
            float tmp = a * x[j * (*incx)];
            for (int i = 0; i < M; i++)
                y[i * (*incy)] += tmp * A[i + j * LDA];
        }
    } else {
        /* y = alpha * A^T * x + y */
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int i = 0; i < M; i++)
                tmp += A[i + j * LDA] * x[i * (*incx)];
            y[j * (*incy)] += a * tmp;
        }
    }
}

void dgemv_(const char *trans, const int *m, const int *n,
            const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx,
            const double *beta, double *y, const int *incy) {
    int M = *m, N = *n, LDA = *lda;
    double a = *alpha, b = *beta;
    bool notrans = lsame(*trans, 'N');
    int leny = notrans ? M : N;

    if (b == 0.0) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = 0.0;
    } else if (b != 1.0) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] *= b;
    }
    if (a == 0.0) return;

    if (notrans) {
        for (int j = 0; j < N; j++) {
            double tmp = a * x[j * (*incx)];
            for (int i = 0; i < M; i++)
                y[i * (*incy)] += tmp * A[i + j * LDA];
        }
    } else {
        for (int j = 0; j < N; j++) {
            double tmp = 0.0;
            for (int i = 0; i < M; i++)
                tmp += A[i + j * LDA] * x[i * (*incx)];
            y[j * (*incy)] += a * tmp;
        }
    }
}

void cgemv_(const char *trans, const int *m, const int *n,
            const lapack_float_complex *alpha, const lapack_float_complex *A, const int *lda,
            const lapack_float_complex *x, const int *incx,
            const lapack_float_complex *beta, lapack_float_complex *y, const int *incy) {
    int M = *m, N = *n, LDA = *lda;
    lapack_float_complex a = *alpha, b = *beta;
    bool notrans = lsame(*trans, 'N');
    bool conjtrans = lsame(*trans, 'C');
    int leny = notrans ? M : N;

    /* Scale y by beta */
    if (b.x == 0.0f && b.y == 0.0f) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = cf_make(0, 0);
    } else if (!(b.x == 1.0f && b.y == 0.0f)) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = cf_mul(b, y[i * (*incy)]);
    }
    if (a.x == 0.0f && a.y == 0.0f) return;

    if (notrans) {
        for (int j = 0; j < N; j++) {
            lapack_float_complex tmp = cf_mul(a, x[j * (*incx)]);
            for (int i = 0; i < M; i++)
                y[i * (*incy)] = cf_add(y[i * (*incy)], cf_mul(tmp, A[i + j * LDA]));
        }
    } else {
        for (int j = 0; j < N; j++) {
            lapack_float_complex tmp = cf_make(0, 0);
            for (int i = 0; i < M; i++) {
                lapack_float_complex aij = conjtrans ? cf_conj(A[i + j * LDA]) : A[i + j * LDA];
                tmp = cf_add(tmp, cf_mul(aij, x[i * (*incx)]));
            }
            y[j * (*incy)] = cf_add(y[j * (*incy)], cf_mul(a, tmp));
        }
    }
}

void zgemv_(const char *trans, const int *m, const int *n,
            const lapack_double_complex *alpha, const lapack_double_complex *A, const int *lda,
            const lapack_double_complex *x, const int *incx,
            const lapack_double_complex *beta, lapack_double_complex *y, const int *incy) {
    int M = *m, N = *n, LDA = *lda;
    lapack_double_complex a = *alpha, b = *beta;
    bool notrans = lsame(*trans, 'N');
    bool conjtrans = lsame(*trans, 'C');
    int leny = notrans ? M : N;

    if (b.x == 0.0 && b.y == 0.0) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = zf_make(0, 0);
    } else if (!(b.x == 1.0 && b.y == 0.0)) {
        for (int i = 0; i < leny; i++) y[i * (*incy)] = zf_mul(b, y[i * (*incy)]);
    }
    if (a.x == 0.0 && a.y == 0.0) return;

    if (notrans) {
        for (int j = 0; j < N; j++) {
            lapack_double_complex tmp = zf_mul(a, x[j * (*incx)]);
            for (int i = 0; i < M; i++)
                y[i * (*incy)] = zf_add(y[i * (*incy)], zf_mul(tmp, A[i + j * LDA]));
        }
    } else {
        for (int j = 0; j < N; j++) {
            lapack_double_complex tmp = zf_make(0, 0);
            for (int i = 0; i < M; i++) {
                lapack_double_complex aij = conjtrans ? zf_conj(A[i + j * LDA]) : A[i + j * LDA];
                tmp = zf_add(tmp, zf_mul(aij, x[i * (*incx)]));
            }
            y[j * (*incy)] = zf_add(y[j * (*incy)], zf_mul(a, tmp));
        }
    }
}

/* ========================================================================
   BLAS Level 2 - TRMV
   ======================================================================== */

void strmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const float *A, const int *lda,
            float *x, const int *incx) {
    int N = *n, LDA = *lda, IX = *incx;
    bool upper = lsame(*uplo, 'U');
    bool notrans = lsame(*trans, 'N');
    bool nounit = lsame(*diag, 'N');

    if (notrans) {
        if (upper) {
            for (int j = 0; j < N; j++) {
                float tmp = x[j * IX];
                if (tmp != 0.0f) {
                    for (int i = 0; i < j; i++)
                        x[i * IX] += tmp * A[i + j * LDA];
                }
                if (nounit) x[j * IX] = tmp * A[j + j * LDA];
            }
        } else {
            for (int j = N - 1; j >= 0; j--) {
                float tmp = x[j * IX];
                if (tmp != 0.0f) {
                    for (int i = N - 1; i > j; i--)
                        x[i * IX] += tmp * A[i + j * LDA];
                }
                if (nounit) x[j * IX] = tmp * A[j + j * LDA];
            }
        }
    } else {
        /* Trans */
        if (upper) {
            for (int j = N - 1; j >= 0; j--) {
                float tmp = x[j * IX];
                if (nounit) tmp *= A[j + j * LDA];
                for (int i = j - 1; i >= 0; i--)
                    tmp += A[i + j * LDA] * x[i * IX];
                x[j * IX] = tmp;
            }
        } else {
            for (int j = 0; j < N; j++) {
                float tmp = x[j * IX];
                if (nounit) tmp *= A[j + j * LDA];
                for (int i = j + 1; i < N; i++)
                    tmp += A[i + j * LDA] * x[i * IX];
                x[j * IX] = tmp;
            }
        }
    }
}

void dtrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const double *A, const int *lda,
            double *x, const int *incx) {
    int N = *n, LDA = *lda, IX = *incx;
    bool upper = lsame(*uplo, 'U');
    bool notrans = lsame(*trans, 'N');
    bool nounit = lsame(*diag, 'N');

    if (notrans) {
        if (upper) {
            for (int j = 0; j < N; j++) {
                double tmp = x[j * IX];
                if (tmp != 0.0) {
                    for (int i = 0; i < j; i++)
                        x[i * IX] += tmp * A[i + j * LDA];
                }
                if (nounit) x[j * IX] = tmp * A[j + j * LDA];
            }
        } else {
            for (int j = N - 1; j >= 0; j--) {
                double tmp = x[j * IX];
                if (tmp != 0.0) {
                    for (int i = N - 1; i > j; i--)
                        x[i * IX] += tmp * A[i + j * LDA];
                }
                if (nounit) x[j * IX] = tmp * A[j + j * LDA];
            }
        }
    } else {
        if (upper) {
            for (int j = N - 1; j >= 0; j--) {
                double tmp = x[j * IX];
                if (nounit) tmp *= A[j + j * LDA];
                for (int i = j - 1; i >= 0; i--)
                    tmp += A[i + j * LDA] * x[i * IX];
                x[j * IX] = tmp;
            }
        } else {
            for (int j = 0; j < N; j++) {
                double tmp = x[j * IX];
                if (nounit) tmp *= A[j + j * LDA];
                for (int i = j + 1; i < N; i++)
                    tmp += A[i + j * LDA] * x[i * IX];
                x[j * IX] = tmp;
            }
        }
    }
}

void ctrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const lapack_float_complex *A, const int *lda,
            lapack_float_complex *x, const int *incx) {
    int N = *n, LDA = *lda, IX = *incx;
    bool upper = lsame(*uplo, 'U');
    bool notrans = lsame(*trans, 'N');
    bool conjtrans = lsame(*trans, 'C');
    bool nounit = lsame(*diag, 'N');

    if (notrans) {
        if (upper) {
            for (int j = 0; j < N; j++) {
                lapack_float_complex tmp = x[j * IX];
                for (int i = 0; i < j; i++)
                    x[i * IX] = cf_add(x[i * IX], cf_mul(tmp, A[i + j * LDA]));
                if (nounit) x[j * IX] = cf_mul(tmp, A[j + j * LDA]);
            }
        } else {
            for (int j = N - 1; j >= 0; j--) {
                lapack_float_complex tmp = x[j * IX];
                for (int i = N - 1; i > j; i--)
                    x[i * IX] = cf_add(x[i * IX], cf_mul(tmp, A[i + j * LDA]));
                if (nounit) x[j * IX] = cf_mul(tmp, A[j + j * LDA]);
            }
        }
    } else {
        /* Trans or ConjTrans */
        if (upper) {
            for (int j = N - 1; j >= 0; j--) {
                lapack_float_complex tmp = x[j * IX];
                if (nounit) {
                    lapack_float_complex ajj = conjtrans ? cf_conj(A[j + j * LDA]) : A[j + j * LDA];
                    tmp = cf_mul(tmp, ajj);
                }
                for (int i = j - 1; i >= 0; i--) {
                    lapack_float_complex aij = conjtrans ? cf_conj(A[i + j * LDA]) : A[i + j * LDA];
                    tmp = cf_add(tmp, cf_mul(aij, x[i * IX]));
                }
                x[j * IX] = tmp;
            }
        } else {
            for (int j = 0; j < N; j++) {
                lapack_float_complex tmp = x[j * IX];
                if (nounit) {
                    lapack_float_complex ajj = conjtrans ? cf_conj(A[j + j * LDA]) : A[j + j * LDA];
                    tmp = cf_mul(tmp, ajj);
                }
                for (int i = j + 1; i < N; i++) {
                    lapack_float_complex aij = conjtrans ? cf_conj(A[i + j * LDA]) : A[i + j * LDA];
                    tmp = cf_add(tmp, cf_mul(aij, x[i * IX]));
                }
                x[j * IX] = tmp;
            }
        }
    }
}

void ztrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const lapack_double_complex *A, const int *lda,
            lapack_double_complex *x, const int *incx) {
    int N = *n, LDA = *lda, IX = *incx;
    bool upper = lsame(*uplo, 'U');
    bool notrans = lsame(*trans, 'N');
    bool conjtrans = lsame(*trans, 'C');
    bool nounit = lsame(*diag, 'N');

    if (notrans) {
        if (upper) {
            for (int j = 0; j < N; j++) {
                lapack_double_complex tmp = x[j * IX];
                for (int i = 0; i < j; i++)
                    x[i * IX] = zf_add(x[i * IX], zf_mul(tmp, A[i + j * LDA]));
                if (nounit) x[j * IX] = zf_mul(tmp, A[j + j * LDA]);
            }
        } else {
            for (int j = N - 1; j >= 0; j--) {
                lapack_double_complex tmp = x[j * IX];
                for (int i = N - 1; i > j; i--)
                    x[i * IX] = zf_add(x[i * IX], zf_mul(tmp, A[i + j * LDA]));
                if (nounit) x[j * IX] = zf_mul(tmp, A[j + j * LDA]);
            }
        }
    } else {
        if (upper) {
            for (int j = N - 1; j >= 0; j--) {
                lapack_double_complex tmp = x[j * IX];
                if (nounit) {
                    lapack_double_complex ajj = conjtrans ? zf_conj(A[j + j * LDA]) : A[j + j * LDA];
                    tmp = zf_mul(tmp, ajj);
                }
                for (int i = j - 1; i >= 0; i--) {
                    lapack_double_complex aij = conjtrans ? zf_conj(A[i + j * LDA]) : A[i + j * LDA];
                    tmp = zf_add(tmp, zf_mul(aij, x[i * IX]));
                }
                x[j * IX] = tmp;
            }
        } else {
            for (int j = 0; j < N; j++) {
                lapack_double_complex tmp = x[j * IX];
                if (nounit) {
                    lapack_double_complex ajj = conjtrans ? zf_conj(A[j + j * LDA]) : A[j + j * LDA];
                    tmp = zf_mul(tmp, ajj);
                }
                for (int i = j + 1; i < N; i++) {
                    lapack_double_complex aij = conjtrans ? zf_conj(A[i + j * LDA]) : A[i + j * LDA];
                    tmp = zf_add(tmp, zf_mul(aij, x[i * IX]));
                }
                x[j * IX] = tmp;
            }
        }
    }
}

/* ----- LADIV: complex division ----- */

void sladiv_(const float *a, const float *b,
             const float *c, const float *d,
             float *p, float *q) {
    float e, f;
    if (fabsf(*d) < fabsf(*c)) {
        e = *d / *c;
        f = *c + *d * e;
        *p = (*a + *b * e) / f;
        *q = (*b - *a * e) / f;
    } else {
        e = *c / *d;
        f = *d + *c * e;
        *p = (*b + *a * e) / f;
        *q = (-*a + *b * e) / f;
    }
}

void dladiv_(const double *a, const double *b,
             const double *c, const double *d,
             double *p, double *q) {
    double e, f;
    if (fabs(*d) < fabs(*c)) {
        e = *d / *c;
        f = *c + *d * e;
        *p = (*a + *b * e) / f;
        *q = (*b - *a * e) / f;
    } else {
        e = *c / *d;
        f = *d + *c * e;
        *p = (*b + *a * e) / f;
        *q = (-*a + *b * e) / f;
    }
}

void sgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const float *alpha, const float *A, const int *lda,
            const float *B, const int *ldb,
            const float *beta, float *C, const int *ldc) {
    int M = *m, N = *n, K = *k, LDA = *lda, LDB = *ldb, LDC = *ldc;
    float a = *alpha, b = *beta;
    bool ta = !lsame(*transa, 'N');
    bool tb = !lsame(*transb, 'N');

    /* Scale C by beta */
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            C[i + j * LDC] = (b == 0.0f) ? 0.0f : b * C[i + j * LDC];

    if (a == 0.0f) return;

    for (int j = 0; j < N; j++) {
        for (int l = 0; l < K; l++) {
            float bval = tb ? B[j + l * LDB] : B[l + j * LDB];
            float tmp = a * bval;
            for (int i = 0; i < M; i++) {
                float aval = ta ? A[l + i * LDA] : A[i + l * LDA];
                C[i + j * LDC] += tmp * aval;
            }
        }
    }
}

void dgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const double *alpha, const double *A, const int *lda,
            const double *B, const int *ldb,
            const double *beta, double *C, const int *ldc) {
    int M = *m, N = *n, K = *k, LDA = *lda, LDB = *ldb, LDC = *ldc;
    double a = *alpha, b = *beta;
    bool ta = !lsame(*transa, 'N');
    bool tb = !lsame(*transb, 'N');

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
            C[i + j * LDC] = (b == 0.0) ? 0.0 : b * C[i + j * LDC];

    if (a == 0.0) return;

    for (int j = 0; j < N; j++) {
        for (int l = 0; l < K; l++) {
            double bval = tb ? B[j + l * LDB] : B[l + j * LDB];
            double tmp = a * bval;
            for (int i = 0; i < M; i++) {
                double aval = ta ? A[l + i * LDA] : A[i + l * LDA];
                C[i + j * LDC] += tmp * aval;
            }
        }
    }
}

void cgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const lapack_float_complex *alpha, const lapack_float_complex *A, const int *lda,
            const lapack_float_complex *B, const int *ldb,
            const lapack_float_complex *beta, lapack_float_complex *C, const int *ldc) {
    int M = *m, N = *n, K = *k, LDA = *lda, LDB = *ldb, LDC = *ldc;
    lapack_float_complex a = *alpha, b = *beta;
    bool nota = lsame(*transa, 'N');
    bool conja = lsame(*transa, 'C');
    bool notb = lsame(*transb, 'N');
    bool conjb = lsame(*transb, 'C');

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            if (b.x == 0.0f && b.y == 0.0f)
                C[i + j * LDC] = cf_make(0, 0);
            else
                C[i + j * LDC] = cf_mul(b, C[i + j * LDC]);
        }

    if (a.x == 0.0f && a.y == 0.0f) return;

    for (int j = 0; j < N; j++) {
        for (int l = 0; l < K; l++) {
            lapack_float_complex bval;
            if (notb) bval = B[l + j * LDB];
            else if (conjb) bval = cf_conj(B[j + l * LDB]);
            else bval = B[j + l * LDB];
            lapack_float_complex tmp = cf_mul(a, bval);
            for (int i = 0; i < M; i++) {
                lapack_float_complex aval;
                if (nota) aval = A[i + l * LDA];
                else if (conja) aval = cf_conj(A[l + i * LDA]);
                else aval = A[l + i * LDA];
                C[i + j * LDC] = cf_add(C[i + j * LDC], cf_mul(tmp, aval));
            }
        }
    }
}

void zgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const lapack_double_complex *alpha, const lapack_double_complex *A, const int *lda,
            const lapack_double_complex *B, const int *ldb,
            const lapack_double_complex *beta, lapack_double_complex *C, const int *ldc) {
    int M = *m, N = *n, K = *k, LDA = *lda, LDB = *ldb, LDC = *ldc;
    lapack_double_complex a = *alpha, b = *beta;
    bool nota = lsame(*transa, 'N');
    bool conja = lsame(*transa, 'C');
    bool notb = lsame(*transb, 'N');
    bool conjb = lsame(*transb, 'C');

    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            if (b.x == 0.0 && b.y == 0.0)
                C[i + j * LDC] = zf_make(0, 0);
            else
                C[i + j * LDC] = zf_mul(b, C[i + j * LDC]);
        }

    if (a.x == 0.0 && a.y == 0.0) return;

    for (int j = 0; j < N; j++) {
        for (int l = 0; l < K; l++) {
            lapack_double_complex bval;
            if (notb) bval = B[l + j * LDB];
            else if (conjb) bval = zf_conj(B[j + l * LDB]);
            else bval = B[j + l * LDB];
            lapack_double_complex tmp = zf_mul(a, bval);
            for (int i = 0; i < M; i++) {
                lapack_double_complex aval;
                if (nota) aval = A[i + l * LDA];
                else if (conja) aval = zf_conj(A[l + i * LDA]);
                else aval = A[l + i * LDA];
                C[i + j * LDC] = zf_add(C[i + j * LDC], zf_mul(tmp, aval));
            }
        }
    }
}

/* ========================================================================
   LAPACK Auxiliary - Machine Parameters
   ======================================================================== */

float slamch_(const char *cmach) {
    if (lsame(*cmach, 'E') || lsame(*cmach, 'e')) return FLT_EPSILON / 2; /* epsilon */
    if (lsame(*cmach, 'S') || lsame(*cmach, 's')) return FLT_MIN;         /* safe minimum */
    if (lsame(*cmach, 'B') || lsame(*cmach, 'b')) return 2.0f;            /* base */
    if (lsame(*cmach, 'P') || lsame(*cmach, 'p')) return FLT_EPSILON;     /* precision = eps*base */
    if (lsame(*cmach, 'N') || lsame(*cmach, 'n')) return (float)FLT_MANT_DIG; /* # of digits */
    if (lsame(*cmach, 'R') || lsame(*cmach, 'r')) return 1.0f;            /* rounding mode */
    if (lsame(*cmach, 'M') || lsame(*cmach, 'm')) return (float)FLT_MIN_EXP;  /* min exponent */
    if (lsame(*cmach, 'U') || lsame(*cmach, 'u')) return FLT_MIN;         /* underflow threshold */
    if (lsame(*cmach, 'L') || lsame(*cmach, 'l')) return (float)FLT_MAX_EXP;  /* max exponent */
    if (lsame(*cmach, 'O') || lsame(*cmach, 'o')) return FLT_MAX;         /* overflow threshold */
    return 0.0f;
}

double dlamch_(const char *cmach) {
    if (lsame(*cmach, 'E') || lsame(*cmach, 'e')) return DBL_EPSILON / 2; /* epsilon */
    if (lsame(*cmach, 'S') || lsame(*cmach, 's')) return DBL_MIN;         /* safe minimum */
    if (lsame(*cmach, 'B') || lsame(*cmach, 'b')) return 2.0;             /* base */
    if (lsame(*cmach, 'P') || lsame(*cmach, 'p')) return DBL_EPSILON;     /* precision = eps*base */
    if (lsame(*cmach, 'N') || lsame(*cmach, 'n')) return (double)DBL_MANT_DIG;
    if (lsame(*cmach, 'R') || lsame(*cmach, 'r')) return 1.0;
    if (lsame(*cmach, 'M') || lsame(*cmach, 'm')) return (double)DBL_MIN_EXP;
    if (lsame(*cmach, 'U') || lsame(*cmach, 'u')) return DBL_MIN;
    if (lsame(*cmach, 'L') || lsame(*cmach, 'l')) return (double)DBL_MAX_EXP;
    if (lsame(*cmach, 'O') || lsame(*cmach, 'o')) return DBL_MAX;
    return 0.0;
}

/* ========================================================================
   LAPACK Auxiliary - slabad/dlabad
   ======================================================================== */

void slabad_(float *small_val, float *large_val) {
    /* If very small, replace by sqrt */
    if (logf(*small_val) < -0.5f * logf(FLT_MAX)) {
        *small_val = sqrtf(*small_val);
        *large_val = sqrtf(*large_val);
    }
}

void dlabad_(double *small_val, double *large_val) {
    if (log(*small_val) < -0.5 * log(DBL_MAX)) {
        *small_val = sqrt(*small_val);
        *large_val = sqrt(*large_val);
    }
}

/* ========================================================================
   LAPACK Auxiliary - LAPY2: sqrt(x^2 + y^2) safely
   ======================================================================== */

float slapy2_(const float *x, const float *y) {
    return hypotf(*x, *y);
}

double dlapy2_(const double *x, const double *y) {
    return hypot(*x, *y);
}

/* ========================================================================
   LAPACK Auxiliary - LADIV: complex division for real types
   For real types: computes (a+ib)/(c+id) = p+iq
   ======================================================================== */

void slacpy_(const char *uplo, const int *m, const int *n,
             const float *A, const int *lda,
             float *B, const int *ldb) {
    int M = *m, N = *n, LDA = *lda, LDB = *ldb;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i <= imin(j, M-1); i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < N; j++)
            for (int i = j; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    }
}

void dlacpy_(const char *uplo, const int *m, const int *n,
             const double *A, const int *lda,
             double *B, const int *ldb) {
    int M = *m, N = *n, LDA = *lda, LDB = *ldb;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i <= imin(j, M-1); i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < N; j++)
            for (int i = j; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    }
}

void clacpy_(const char *uplo, const int *m, const int *n,
             const lapack_float_complex *A, const int *lda,
             lapack_float_complex *B, const int *ldb) {
    int M = *m, N = *n, LDA = *lda, LDB = *ldb;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i <= imin(j, M-1); i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < N; j++)
            for (int i = j; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    }
}

void zlacpy_(const char *uplo, const int *m, const int *n,
             const lapack_double_complex *A, const int *lda,
             lapack_double_complex *B, const int *ldb) {
    int M = *m, N = *n, LDA = *lda, LDB = *ldb;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i <= imin(j, M-1); i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < N; j++)
            for (int i = j; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    } else {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                B[i + j * LDB] = A[i + j * LDA];
    }
}

/* ========================================================================
   LAPACK Auxiliary - LASET: set matrix to constants
   ======================================================================== */

void slaset_(const char *uplo, const int *m, const int *n,
             const float *alpha, const float *beta,
             float *A, const int *lda) {
    int M = *m, N = *n, LDA = *lda;
    float al = *alpha, be = *beta;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < imin(j, M); i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < imin(M, N); j++) {
            A[j + j * LDA] = be;
            for (int i = j + 1; i < M; i++)
                A[i + j * LDA] = al;
        }
    } else {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    }
}

void dlaset_(const char *uplo, const int *m, const int *n,
             const double *alpha, const double *beta,
             double *A, const int *lda) {
    int M = *m, N = *n, LDA = *lda;
    double al = *alpha, be = *beta;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < imin(j, M); i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < imin(M, N); j++) {
            A[j + j * LDA] = be;
            for (int i = j + 1; i < M; i++)
                A[i + j * LDA] = al;
        }
    } else {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    }
}

void claset_(const char *uplo, const int *m, const int *n,
             const lapack_float_complex *alpha, const lapack_float_complex *beta,
             lapack_float_complex *A, const int *lda) {
    int M = *m, N = *n, LDA = *lda;
    lapack_float_complex al = *alpha, be = *beta;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < imin(j, M); i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < imin(M, N); j++) {
            A[j + j * LDA] = be;
            for (int i = j + 1; i < M; i++)
                A[i + j * LDA] = al;
        }
    } else {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    }
}

void zlaset_(const char *uplo, const int *m, const int *n,
             const lapack_double_complex *alpha, const lapack_double_complex *beta,
             lapack_double_complex *A, const int *lda) {
    int M = *m, N = *n, LDA = *lda;
    lapack_double_complex al = *alpha, be = *beta;
    if (lsame(*uplo, 'U')) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < imin(j, M); i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    } else if (lsame(*uplo, 'L')) {
        for (int j = 0; j < imin(M, N); j++) {
            A[j + j * LDA] = be;
            for (int i = j + 1; i < M; i++)
                A[i + j * LDA] = al;
        }
    } else {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++)
                A[i + j * LDA] = al;
            if (j < M) A[j + j * LDA] = be;
        }
    }
}

/* ========================================================================
   LAPACK Auxiliary - LANGE: matrix norm
   ======================================================================== */

float slange_(const char *norm, const int *m, const int *n,
              const float *A, const int *lda, float *work) {
    int M = *m, N = *n, LDA = *lda;
    if (M == 0 || N == 0) return 0.0f;

    if (lsame(*norm, 'M')) {
        /* Max norm */
        float val = 0.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                val = std::max(val, fabsf(A[i + j * LDA]));
        return val;
    } else if (lsame(*norm, 'O') || *norm == '1') {
        /* One norm (max column sum) */
        float val = 0.0f;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++)
                sum += fabsf(A[i + j * LDA]);
            val = std::max(val, sum);
        }
        return val;
    } else if (lsame(*norm, 'I')) {
        /* Infinity norm (max row sum) */
        for (int i = 0; i < M; i++) work[i] = 0.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                work[i] += fabsf(A[i + j * LDA]);
        float val = 0.0f;
        for (int i = 0; i < M; i++)
            val = std::max(val, work[i]);
        return val;
    } else {
        /* Frobenius norm */
        float scale = 0.0f, ssq = 1.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                float v = fabsf(A[i + j * LDA]);
                if (v != 0.0f) {
                    if (scale < v) { ssq = 1.0f + ssq * (scale / v) * (scale / v); scale = v; }
                    else { ssq += (v / scale) * (v / scale); }
                }
            }
        return scale * sqrtf(ssq);
    }
}

double dlange_(const char *norm, const int *m, const int *n,
               const double *A, const int *lda, double *work) {
    int M = *m, N = *n, LDA = *lda;
    if (M == 0 || N == 0) return 0.0;

    if (lsame(*norm, 'M')) {
        double val = 0.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                val = std::max(val, fabs(A[i + j * LDA]));
        return val;
    } else if (lsame(*norm, 'O') || *norm == '1') {
        double val = 0.0;
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int i = 0; i < M; i++)
                sum += fabs(A[i + j * LDA]);
            val = std::max(val, sum);
        }
        return val;
    } else if (lsame(*norm, 'I')) {
        for (int i = 0; i < M; i++) work[i] = 0.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                work[i] += fabs(A[i + j * LDA]);
        double val = 0.0;
        for (int i = 0; i < M; i++)
            val = std::max(val, work[i]);
        return val;
    } else {
        double scale = 0.0, ssq = 1.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                double v = fabs(A[i + j * LDA]);
                if (v != 0.0) {
                    if (scale < v) { ssq = 1.0 + ssq * (scale / v) * (scale / v); scale = v; }
                    else { ssq += (v / scale) * (v / scale); }
                }
            }
        return scale * sqrt(ssq);
    }
}

float clange_(const char *norm, const int *m, const int *n,
              const lapack_float_complex *A, const int *lda, float *work) {
    int M = *m, N = *n, LDA = *lda;
    if (M == 0 || N == 0) return 0.0f;

    if (lsame(*norm, 'M')) {
        float val = 0.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                val = std::max(val, cf_abs(A[i + j * LDA]));
        return val;
    } else if (lsame(*norm, 'O') || *norm == '1') {
        float val = 0.0f;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++)
                sum += cf_abs(A[i + j * LDA]);
            val = std::max(val, sum);
        }
        return val;
    } else if (lsame(*norm, 'I')) {
        for (int i = 0; i < M; i++) work[i] = 0.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                work[i] += cf_abs(A[i + j * LDA]);
        float val = 0.0f;
        for (int i = 0; i < M; i++)
            val = std::max(val, work[i]);
        return val;
    } else {
        float scale = 0.0f, ssq = 0.0f;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                float v = cf_abs(A[i + j * LDA]);
                if (v != 0.0f) {
                    if (scale < v) { ssq = 1.0f + ssq * (scale / v) * (scale / v); scale = v; }
                    else { ssq += (v / scale) * (v / scale); }
                }
            }
        return scale * sqrtf(ssq);
    }
}

double zlange_(const char *norm, const int *m, const int *n,
               const lapack_double_complex *A, const int *lda, double *work) {
    int M = *m, N = *n, LDA = *lda;
    if (M == 0 || N == 0) return 0.0;

    if (lsame(*norm, 'M')) {
        double val = 0.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                val = std::max(val, zf_abs(A[i + j * LDA]));
        return val;
    } else if (lsame(*norm, 'O') || *norm == '1') {
        double val = 0.0;
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int i = 0; i < M; i++)
                sum += zf_abs(A[i + j * LDA]);
            val = std::max(val, sum);
        }
        return val;
    } else if (lsame(*norm, 'I')) {
        for (int i = 0; i < M; i++) work[i] = 0.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                work[i] += zf_abs(A[i + j * LDA]);
        double val = 0.0;
        for (int i = 0; i < M; i++)
            val = std::max(val, work[i]);
        return val;
    } else {
        double scale = 0.0, ssq = 0.0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                double v = zf_abs(A[i + j * LDA]);
                if (v != 0.0) {
                    if (scale < v) { ssq = 1.0 + ssq * (scale / v) * (scale / v); scale = v; }
                    else { ssq += (v / scale) * (v / scale); }
                }
            }
        return scale * sqrt(ssq);
    }
}

/* ========================================================================
   LAPACK Auxiliary - LASCL: scale matrix by ratio cfrom/cto
   ======================================================================== */

void slascl_(const char *type, const int *kl, const int *ku,
             const float *cfrom, const float *cto,
             const int *m, const int *n, float *A, const int *lda,
             int *info) {
    *info = 0;
    int M = *m, N = *n, LDA = *lda;
    float cf = *cfrom, ct = *cto;
    bool done = false;
    while (!done) {
        float cfromc = cf;
        float ctoc = ct;
        float cfrom1 = cfromc * FLT_MIN;
        float mul;
        if (cfrom1 == cfromc) {
            mul = ctoc / cfromc;
            done = true;
        } else {
            float cto1 = ctoc / FLT_MAX;
            if (fabsf(cto1) > fabsf(cfromc)) {
                mul = FLT_MAX;
                done = false;
                cfromc = cto1;
            } else if (fabsf(cfrom1) > fabsf(ctoc)) {
                mul = FLT_MIN;
                done = false;
                cfromc = cfrom1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        if (lsame(*type, 'G')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'U')) {
            /* Upper triangular */
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j, M-1); i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'L')) {
            /* Lower triangular */
            for (int j = 0; j < N; j++)
                for (int i = j; i < M; i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'H')) {
            /* Upper Hessenberg */
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j+1, M-1); i++)
                    A[i + j * LDA] *= mul;
        }
        cf = cfromc;
    }
}

void dlascl_(const char *type, const int *kl, const int *ku,
             const double *cfrom, const double *cto,
             const int *m, const int *n, double *A, const int *lda,
             int *info) {
    *info = 0;
    int M = *m, N = *n, LDA = *lda;
    double cf = *cfrom, ct = *cto;
    bool done = false;
    while (!done) {
        double cfromc = cf;
        double ctoc = ct;
        double cfrom1 = cfromc * DBL_MIN;
        double mul;
        if (cfrom1 == cfromc) {
            mul = ctoc / cfromc;
            done = true;
        } else {
            double cto1 = ctoc / DBL_MAX;
            if (fabs(cto1) > fabs(cfromc)) {
                mul = DBL_MAX;
                done = false;
                cfromc = cto1;
            } else if (fabs(cfrom1) > fabs(ctoc)) {
                mul = DBL_MIN;
                done = false;
                cfromc = cfrom1;
            } else {
                mul = ctoc / cfromc;
                done = true;
            }
        }
        if (lsame(*type, 'G')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'U')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j, M-1); i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'L')) {
            for (int j = 0; j < N; j++)
                for (int i = j; i < M; i++)
                    A[i + j * LDA] *= mul;
        } else if (lsame(*type, 'H')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j+1, M-1); i++)
                    A[i + j * LDA] *= mul;
        }
        cf = cfromc;
    }
}

void clascl_(const char *type, const int *kl, const int *ku,
             const float *cfrom, const float *cto,
             const int *m, const int *n, lapack_float_complex *A, const int *lda,
             int *info) {
    *info = 0;
    int M = *m, N = *n, LDA = *lda;
    float cf = *cfrom, ct = *cto;
    bool done = false;
    while (!done) {
        float cfromc = cf, ctoc = ct;
        float cfrom1 = cfromc * FLT_MIN;
        float mul;
        if (cfrom1 == cfromc) { mul = ctoc / cfromc; done = true; }
        else {
            float cto1 = ctoc / FLT_MAX;
            if (fabsf(cto1) > fabsf(cfromc)) { mul = FLT_MAX; done = false; cfromc = cto1; }
            else if (fabsf(cfrom1) > fabsf(ctoc)) { mul = FLT_MIN; done = false; cfromc = cfrom1; }
            else { mul = ctoc / cfromc; done = true; }
        }
        if (lsame(*type, 'G')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    A[i + j * LDA] = cf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'U')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j, M-1); i++)
                    A[i + j * LDA] = cf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'L')) {
            for (int j = 0; j < N; j++)
                for (int i = j; i < M; i++)
                    A[i + j * LDA] = cf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'H')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j+1, M-1); i++)
                    A[i + j * LDA] = cf_scale(mul, A[i + j * LDA]);
        }
        cf = cfromc;
    }
}

void zlascl_(const char *type, const int *kl, const int *ku,
             const double *cfrom, const double *cto,
             const int *m, const int *n, lapack_double_complex *A, const int *lda,
             int *info) {
    *info = 0;
    int M = *m, N = *n, LDA = *lda;
    double cf = *cfrom, ct = *cto;
    bool done = false;
    while (!done) {
        double cfromc = cf, ctoc = ct;
        double cfrom1 = cfromc * DBL_MIN;
        double mul;
        if (cfrom1 == cfromc) { mul = ctoc / cfromc; done = true; }
        else {
            double cto1 = ctoc / DBL_MAX;
            if (fabs(cto1) > fabs(cfromc)) { mul = DBL_MAX; done = false; cfromc = cto1; }
            else if (fabs(cfrom1) > fabs(ctoc)) { mul = DBL_MIN; done = false; cfromc = cfrom1; }
            else { mul = ctoc / cfromc; done = true; }
        }
        if (lsame(*type, 'G')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    A[i + j * LDA] = zf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'U')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j, M-1); i++)
                    A[i + j * LDA] = zf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'L')) {
            for (int j = 0; j < N; j++)
                for (int i = j; i < M; i++)
                    A[i + j * LDA] = zf_scale(mul, A[i + j * LDA]);
        } else if (lsame(*type, 'H')) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i <= imin(j+1, M-1); i++)
                    A[i + j * LDA] = zf_scale(mul, A[i + j * LDA]);
        }
        cf = cfromc;
    }
}

/* ========================================================================
   LAPACK Auxiliary - LACGV: conjugate vector (complex only)
   ======================================================================== */

void clacgv_(const int *n, lapack_float_complex *x, const int *incx) {
    for (int i = 0; i < *n; i++)
        x[i * (*incx)].y = -x[i * (*incx)].y;
}

void zlacgv_(const int *n, lapack_double_complex *x, const int *incx) {
    for (int i = 0; i < *n; i++)
        x[i * (*incx)].y = -x[i * (*incx)].y;
}

/* For real types, lacgv is a no-op but must exist as a symbol */
void dlacgv_(const int * /*n*/, double * /*x*/, const int * /*incx*/) {
    /* no-op for real */
}

void slacgv_(const int * /*n*/, float * /*x*/, const int * /*incx*/) {
    /* no-op for real */
}

/* ========================================================================
   LAPACK Auxiliary - LARTG: generate Givens rotation
   For real: given f,g, compute cs,sn,r such that
      [cs  sn] [f] = [r]
      [-sn cs] [g]   [0]
   ======================================================================== */

void slartg_(const float *f, const float *g, float *cs, float *sn, float *r) {
    float safmin = FLT_MIN;
    float ff = *f, gg = *g;
    if (gg == 0.0f) {
        *cs = 1.0f; *sn = 0.0f; *r = ff;
    } else if (ff == 0.0f) {
        *cs = 0.0f; *sn = copysignf(1.0f, gg); *r = fabsf(gg);
    } else {
        float scl = std::max(fabsf(ff), fabsf(gg));
        if (scl >= FLT_MAX * safmin) {
            /* Scale down */
            int count = 0;
            float safmn2 = 1.0f;
            float safmx2 = 1.0f / safmn2;
            while (scl >= safmx2) {
                count++;
                ff *= safmn2;
                gg *= safmn2;
                scl = std::max(fabsf(ff), fabsf(gg));
            }
            *r = hypotf(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
            for (int i = 0; i < count; i++)
                *r *= safmx2;
        } else if (scl <= safmin) {
            int count = 0;
            float safmn2 = 1.0f;
            float safmx2 = 1.0f / safmn2;
            while (scl <= safmn2) {
                count++;
                ff *= safmx2;
                gg *= safmx2;
                scl = std::max(fabsf(ff), fabsf(gg));
            }
            *r = hypotf(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
            for (int i = 0; i < count; i++)
                *r *= safmn2;
        } else {
            *r = hypotf(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
        }
        /* Ensure r >= 0 for consistency */
        if (fabsf(*f) > fabsf(*g) && *cs < 0.0f) {
            *cs = -*cs;
            *sn = -*sn;
            *r = -*r;
        }
    }
}

void dlartg_(const double *f, const double *g, double *cs, double *sn, double *r) {
    double ff = *f, gg = *g;
    if (gg == 0.0) {
        *cs = 1.0; *sn = 0.0; *r = ff;
    } else if (ff == 0.0) {
        *cs = 0.0; *sn = copysign(1.0, gg); *r = fabs(gg);
    } else {
        double scl = std::max(fabs(ff), fabs(gg));
        if (scl >= DBL_MAX * DBL_MIN) {
            int count = 0;
            double safmn2 = 1.0;
            double safmx2 = 1.0 / safmn2;
            while (scl >= safmx2) {
                count++;
                ff *= safmn2;
                gg *= safmn2;
                scl = std::max(fabs(ff), fabs(gg));
            }
            *r = hypot(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
            for (int i = 0; i < count; i++)
                *r *= safmx2;
        } else if (scl <= DBL_MIN) {
            int count = 0;
            double safmn2 = 1.0;
            double safmx2 = 1.0 / safmn2;
            while (scl <= safmn2) {
                count++;
                ff *= safmx2;
                gg *= safmx2;
                scl = std::max(fabs(ff), fabs(gg));
            }
            *r = hypot(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
            for (int i = 0; i < count; i++)
                *r *= safmn2;
        } else {
            *r = hypot(ff, gg);
            *cs = ff / *r;
            *sn = gg / *r;
        }
        if (fabs(*f) > fabs(*g) && *cs < 0.0) {
            *cs = -*cs;
            *sn = -*sn;
            *r = -*r;
        }
    }
}

/* ========================================================================
   LAPACK Auxiliary - LARFG: generate Householder reflector
   Given a vector [alpha; x], generate tau and v such that
   H = I - tau * v * v^T, H * [alpha; x] = [beta; 0]
   ======================================================================== */

void slarfg_(const int *n, float *alpha, float *x, const int *incx, float *tau) {
    int N = *n;
    if (N <= 1) { *tau = 0.0f; return; }

    float xnorm = 0.0f;
    for (int i = 0; i < N - 1; i++)
        xnorm += x[i * (*incx)] * x[i * (*incx)];
    xnorm = sqrtf(xnorm);

    if (xnorm == 0.0f) {
        *tau = 0.0f;
    } else {
        float beta = -copysignf(hypotf(*alpha, xnorm), *alpha);
        float safmin = FLT_MIN / FLT_EPSILON;
        int knt = 0;
        if (fabsf(beta) < safmin) {
            /* Need to scale */
            float rsafmn = 1.0f / safmin;
            while (fabsf(beta) < safmin) {
                knt++;
                for (int i = 0; i < N - 1; i++)
                    x[i * (*incx)] *= rsafmn;
                beta *= rsafmn;
                *alpha *= rsafmn;
            }
            xnorm = 0.0f;
            for (int i = 0; i < N - 1; i++)
                xnorm += x[i * (*incx)] * x[i * (*incx)];
            xnorm = sqrtf(xnorm);
            beta = -copysignf(hypotf(*alpha, xnorm), *alpha);
        }
        *tau = (beta - *alpha) / beta;
        float scl = 1.0f / (*alpha - beta);
        for (int i = 0; i < N - 1; i++)
            x[i * (*incx)] *= scl;
        for (int j = 0; j < knt; j++)
            beta *= safmin;
        *alpha = beta;
    }
}

void dlarfg_(const int *n, double *alpha, double *x, const int *incx, double *tau) {
    int N = *n;
    if (N <= 1) { *tau = 0.0; return; }

    double xnorm = 0.0;
    for (int i = 0; i < N - 1; i++)
        xnorm += x[i * (*incx)] * x[i * (*incx)];
    xnorm = sqrt(xnorm);

    if (xnorm == 0.0) {
        *tau = 0.0;
    } else {
        double beta = -copysign(hypot(*alpha, xnorm), *alpha);
        double safmin = DBL_MIN / DBL_EPSILON;
        int knt = 0;
        if (fabs(beta) < safmin) {
            double rsafmn = 1.0 / safmin;
            while (fabs(beta) < safmin) {
                knt++;
                for (int i = 0; i < N - 1; i++)
                    x[i * (*incx)] *= rsafmn;
                beta *= rsafmn;
                *alpha *= rsafmn;
            }
            xnorm = 0.0;
            for (int i = 0; i < N - 1; i++)
                xnorm += x[i * (*incx)] * x[i * (*incx)];
            xnorm = sqrt(xnorm);
            beta = -copysign(hypot(*alpha, xnorm), *alpha);
        }
        *tau = (beta - *alpha) / beta;
        double scl = 1.0 / (*alpha - beta);
        for (int i = 0; i < N - 1; i++)
            x[i * (*incx)] *= scl;
        for (int j = 0; j < knt; j++)
            beta *= safmin;
        *alpha = beta;
    }
}

void clarfg_(const int *n, lapack_float_complex *alpha,
             lapack_float_complex *x, const int *incx,
             lapack_float_complex *tau) {
    int N = *n;
    if (N <= 1) { *tau = cf_make(0, 0); return; }

    float xnorm = 0.0f;
    for (int i = 0; i < N - 1; i++) {
        float re = x[i * (*incx)].x, im = x[i * (*incx)].y;
        xnorm += re * re + im * im;
    }
    xnorm = sqrtf(xnorm);

    if (xnorm == 0.0f && alpha->y == 0.0f) {
        *tau = cf_make(0, 0);
    } else {
        float alphr = alpha->x, alphi = alpha->y;
        float beta = -copysignf(hypotf(cf_abs(*alpha), xnorm), alphr);
        float safmin = FLT_MIN / FLT_EPSILON;
        int knt = 0;
        if (fabsf(beta) < safmin) {
            float rsafmn = 1.0f / safmin;
            while (fabsf(beta) < safmin) {
                knt++;
                for (int i = 0; i < N - 1; i++)
                    x[i * (*incx)] = cf_scale(rsafmn, x[i * (*incx)]);
                beta *= rsafmn;
                alphr *= rsafmn;
                alphi *= rsafmn;
            }
            xnorm = 0.0f;
            for (int i = 0; i < N - 1; i++) {
                float re = x[i * (*incx)].x, im = x[i * (*incx)].y;
                xnorm += re * re + im * im;
            }
            xnorm = sqrtf(xnorm);
            *alpha = cf_make(alphr, alphi);
            beta = -copysignf(hypotf(cf_abs(*alpha), xnorm), alphr);
        }
        tau->x = (beta - alphr) / beta;
        tau->y = -alphi / beta;
        lapack_float_complex scl = cf_div(cf_make(1, 0), cf_sub(*alpha, cf_make(beta, 0)));
        for (int i = 0; i < N - 1; i++)
            x[i * (*incx)] = cf_mul(scl, x[i * (*incx)]);
        for (int j = 0; j < knt; j++)
            beta *= safmin;
        *alpha = cf_make(beta, 0);
    }
}

void zlarfg_(const int *n, lapack_double_complex *alpha,
             lapack_double_complex *x, const int *incx,
             lapack_double_complex *tau) {
    int N = *n;
    if (N <= 1) { *tau = zf_make(0, 0); return; }

    double xnorm = 0.0;
    for (int i = 0; i < N - 1; i++) {
        double re = x[i * (*incx)].x, im = x[i * (*incx)].y;
        xnorm += re * re + im * im;
    }
    xnorm = sqrt(xnorm);

    if (xnorm == 0.0 && alpha->y == 0.0) {
        *tau = zf_make(0, 0);
    } else {
        double alphr = alpha->x, alphi = alpha->y;
        double beta = -copysign(hypot(zf_abs(*alpha), xnorm), alphr);
        double safmin = DBL_MIN / DBL_EPSILON;
        int knt = 0;
        if (fabs(beta) < safmin) {
            double rsafmn = 1.0 / safmin;
            while (fabs(beta) < safmin) {
                knt++;
                for (int i = 0; i < N - 1; i++)
                    x[i * (*incx)] = zf_scale(rsafmn, x[i * (*incx)]);
                beta *= rsafmn;
                alphr *= rsafmn;
                alphi *= rsafmn;
            }
            xnorm = 0.0;
            for (int i = 0; i < N - 1; i++) {
                double re = x[i * (*incx)].x, im = x[i * (*incx)].y;
                xnorm += re * re + im * im;
            }
            xnorm = sqrt(xnorm);
            *alpha = zf_make(alphr, alphi);
            beta = -copysign(hypot(zf_abs(*alpha), xnorm), alphr);
        }
        tau->x = (beta - alphr) / beta;
        tau->y = -alphi / beta;
        lapack_double_complex scl = zf_div(zf_make(1, 0), zf_sub(*alpha, zf_make(beta, 0)));
        for (int i = 0; i < N - 1; i++)
            x[i * (*incx)] = zf_mul(scl, x[i * (*incx)]);
        for (int j = 0; j < knt; j++)
            beta *= safmin;
        *alpha = zf_make(beta, 0);
    }
}

/* ========================================================================
   LAPACK Auxiliary - LARF: apply Householder reflector
   H = I - tau * v * v^T
   Applies H from left or right to matrix C
   ======================================================================== */

void slarf_(const char *side, const int *m, const int *n,
            const float *v, const int *incv,
            const float *tau, float *C, const int *ldc,
            float *work) {
    int M = *m, N = *n, LDC = *ldc;
    float t = *tau;
    if (t == 0.0f) return;

    if (lsame(*side, 'L')) {
        /* H * C:  C = C - tau * v * (v^T * C) */
        /* work = v^T * C (1 x N) */
        for (int j = 0; j < N; j++) {
            work[j] = 0.0f;
            for (int i = 0; i < M; i++)
                work[j] += v[i * (*incv)] * C[i + j * LDC];
        }
        /* C = C - tau * v * work^T */
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] -= t * v[i * (*incv)] * work[j];
    } else {
        /* C * H:  C = C - tau * (C * v) * v^T */
        /* work = C * v (M x 1) */
        for (int i = 0; i < M; i++) {
            work[i] = 0.0f;
            for (int j = 0; j < N; j++)
                work[i] += C[i + j * LDC] * v[j * (*incv)];
        }
        /* C = C - tau * work * v^T */
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] -= t * work[i] * v[j * (*incv)];
    }
}

void dlarf_(const char *side, const int *m, const int *n,
            const double *v, const int *incv,
            const double *tau, double *C, const int *ldc,
            double *work) {
    int M = *m, N = *n, LDC = *ldc;
    double t = *tau;
    if (t == 0.0) return;

    if (lsame(*side, 'L')) {
        for (int j = 0; j < N; j++) {
            work[j] = 0.0;
            for (int i = 0; i < M; i++)
                work[j] += v[i * (*incv)] * C[i + j * LDC];
        }
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] -= t * v[i * (*incv)] * work[j];
    } else {
        for (int i = 0; i < M; i++) {
            work[i] = 0.0;
            for (int j = 0; j < N; j++)
                work[i] += C[i + j * LDC] * v[j * (*incv)];
        }
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] -= t * work[i] * v[j * (*incv)];
    }
}

void clarf_(const char *side, const int *m, const int *n,
            const lapack_float_complex *v, const int *incv,
            const lapack_float_complex *tau, lapack_float_complex *C, const int *ldc,
            lapack_float_complex *work) {
    int M = *m, N = *n, LDC = *ldc;
    lapack_float_complex t = *tau;
    if (t.x == 0.0f && t.y == 0.0f) return;

    if (lsame(*side, 'L')) {
        /* work = v^H * C */
        for (int j = 0; j < N; j++) {
            work[j] = cf_make(0, 0);
            for (int i = 0; i < M; i++)
                work[j] = cf_add(work[j], cf_mul(cf_conj(v[i * (*incv)]), C[i + j * LDC]));
        }
        /* C = C - tau * v * work^T */
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] = cf_sub(C[i + j * LDC], cf_mul(t, cf_mul(v[i * (*incv)], work[j])));
    } else {
        /* work = C * v */
        for (int i = 0; i < M; i++) {
            work[i] = cf_make(0, 0);
            for (int j = 0; j < N; j++)
                work[i] = cf_add(work[i], cf_mul(C[i + j * LDC], v[j * (*incv)]));
        }
        /* C = C - tau * work * v^H */
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] = cf_sub(C[i + j * LDC], cf_mul(t, cf_mul(work[i], cf_conj(v[j * (*incv)]))));
    }
}

void zlarf_(const char *side, const int *m, const int *n,
            const lapack_double_complex *v, const int *incv,
            const lapack_double_complex *tau, lapack_double_complex *C, const int *ldc,
            lapack_double_complex *work) {
    int M = *m, N = *n, LDC = *ldc;
    lapack_double_complex t = *tau;
    if (t.x == 0.0 && t.y == 0.0) return;

    if (lsame(*side, 'L')) {
        for (int j = 0; j < N; j++) {
            work[j] = zf_make(0, 0);
            for (int i = 0; i < M; i++)
                work[j] = zf_add(work[j], zf_mul(zf_conj(v[i * (*incv)]), C[i + j * LDC]));
        }
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] = zf_sub(C[i + j * LDC], zf_mul(t, zf_mul(v[i * (*incv)], work[j])));
    } else {
        for (int i = 0; i < M; i++) {
            work[i] = zf_make(0, 0);
            for (int j = 0; j < N; j++)
                work[i] = zf_add(work[i], zf_mul(C[i + j * LDC], v[j * (*incv)]));
        }
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                C[i + j * LDC] = zf_sub(C[i + j * LDC], zf_mul(t, zf_mul(work[i], zf_conj(v[j * (*incv)]))));
    }
}

/* ========================================================================
   LAPACK Auxiliary - LARFT: form T factor of block Householder reflector
   T is upper triangular K-by-K
   ======================================================================== */

void sgehd2_(const int *n, const int *ilo, const int *ihi,
             float *A, const int *lda, float *tau, float *work, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDA = *lda;
    /* ilo, ihi are 1-indexed from Fortran */
    for (int i = ILO - 1; i < IHI - 1; i++) {
        /* Generate Householder to annihilate A(i+2:ihi, i) */
        int nn = IHI - i - 1; /* length of reflector */
        int one = 1;
        slarfg_(&nn, &A[(i+1) + i * LDA], (nn > 1) ? &A[(i+2) + i * LDA] : &A[(i+1) + i * LDA], &one, &tau[i]);
        float aii = A[(i+1) + i * LDA];
        A[(i+1) + i * LDA] = 1.0f;

        /* Apply H from right: A(0:IHI-1, i+1:N-1) */
        int m2 = IHI;
        int n2 = N - i - 1;
        if (m2 > 0 && n2 > 0)
            slarf_("R", &m2, &n2, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[0 + (i+1) * LDA], &LDA, work);

        /* Apply H from left: A(i+1:IHI-1, i+1:N-1) */
        int m1 = IHI - i - 1;
        int n1 = N - i - 1;
        if (m1 > 0 && n1 > 0)
            slarf_("L", &m1, &n1, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[(i+1) + (i+1) * LDA], &LDA, work);

        A[(i+1) + i * LDA] = aii;
    }
}

void dgehd2_(const int *n, const int *ilo, const int *ihi,
             double *A, const int *lda, double *tau, double *work, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDA = *lda;
    // LAPACK: for I = ILO..IHI-1 (1-indexed), our i is 0-indexed = I-1
    for (int i = ILO - 1; i < IHI - 1; i++) {
        // n_larfg = IHI - I = IHI - (i+1) = IHI - i - 1
        int nn = IHI - i - 1;
        int one = 1;
        // alpha = A(i+1,i), x = A(i+2:IHI-1, i) (length nn-1)
        double *x_ptr = (nn > 1) ? &A[(i+2) + i * LDA] : &A[(i+1) + i * LDA];
        dlarfg_(&nn, &A[(i+1) + i * LDA], x_ptr, &one, &tau[i]);
        double aii = A[(i+1) + i * LDA];
        A[(i+1) + i * LDA] = 1.0;

        // Apply H from right: A(0:IHI-1, i+1:N-1)
        int m2 = IHI;
        int n2 = N - i - 1;
        if (m2 > 0 && n2 > 0)
            dlarf_("R", &m2, &n2, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[0 + (i+1) * LDA], &LDA, work);

        // Apply H from left: A(i+1:IHI-1, i+1:N-1)
        int m1 = IHI - i - 1;
        int n1 = N - i - 1;
        if (m1 > 0 && n1 > 0)
            dlarf_("L", &m1, &n1, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[(i+1) + (i+1) * LDA], &LDA, work);

        A[(i+1) + i * LDA] = aii;
    }
}

void cgehd2_(const int *n, const int *ilo, const int *ihi,
             lapack_float_complex *A, const int *lda,
             lapack_float_complex *tau, lapack_float_complex *work, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDA = *lda;
    for (int i = ILO - 1; i < IHI - 1; i++) {
        int nn = IHI - i - 1;
        int one = 1;
        clarfg_(&nn, &A[(i+1) + i * LDA], (nn > 1) ? &A[(i+2) + i * LDA] : &A[(i+1) + i * LDA], &one, &tau[i]);
        lapack_float_complex aii = A[(i+1) + i * LDA];
        A[(i+1) + i * LDA] = cf_make(1, 0);

        // Apply H(i) from the right to A(0:IHI-1, i+1:N-1)
        int m2 = IHI, n2 = N - i - 1;
        if (m2 > 0 && n2 > 0)
            clarf_("R", &m2, &n2, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[0 + (i+1) * LDA], &LDA, work);

        // Apply H(i)^H from the left to A(i+1:IHI-1, i+1:N-1)
        lapack_float_complex conj_tau = cf_conj(tau[i]);
        int m1 = IHI - i - 1, n1 = N - i - 1;
        if (m1 > 0 && n1 > 0)
            clarf_("L", &m1, &n1, &A[(i+1) + i * LDA], &one, &conj_tau,
                   &A[(i+1) + (i+1) * LDA], &LDA, work);

        A[(i+1) + i * LDA] = aii;
    }
}

void zgehd2_(const int *n, const int *ilo, const int *ihi,
             lapack_double_complex *A, const int *lda,
             lapack_double_complex *tau, lapack_double_complex *work, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDA = *lda;
    for (int i = ILO - 1; i < IHI - 1; i++) {
        int nn = IHI - i - 1;
        int one = 1;
        zlarfg_(&nn, &A[(i+1) + i * LDA], (nn > 1) ? &A[(i+2) + i * LDA] : &A[(i+1) + i * LDA], &one, &tau[i]);
        lapack_double_complex aii = A[(i+1) + i * LDA];
        A[(i+1) + i * LDA] = zf_make(1, 0);

        // Apply H(i) from the right to A(0:IHI-1, i+1:N-1)
        int m2 = IHI, n2 = N - i - 1;
        if (m2 > 0 && n2 > 0)
            zlarf_("R", &m2, &n2, &A[(i+1) + i * LDA], &one, &tau[i],
                   &A[0 + (i+1) * LDA], &LDA, work);

        // Apply H(i)^H from the left to A(i+1:IHI-1, i+1:N-1)
        lapack_double_complex conj_tau = zf_conj(tau[i]);
        int m1 = IHI - i - 1, n1 = N - i - 1;
        if (m1 > 0 && n1 > 0)
            zlarf_("L", &m1, &n1, &A[(i+1) + i * LDA], &one, &conj_tau,
                   &A[(i+1) + (i+1) * LDA], &LDA, work);

        A[(i+1) + i * LDA] = aii;
    }
}

/* ========================================================================
   LAPACK Computational - GEBAL: balance a general matrix
   ======================================================================== */

/* Helper: swap rows/cols for balancing */
static void s_swap_rowcol(int n, float *A, int lda, int i, int j) {
    if (i == j) return;
    for (int k = 0; k < n; k++) { float t = A[i + k * lda]; A[i + k * lda] = A[j + k * lda]; A[j + k * lda] = t; }
    for (int k = 0; k < n; k++) { float t = A[k + i * lda]; A[k + i * lda] = A[k + j * lda]; A[k + j * lda] = t; }
}
static void d_swap_rowcol(int n, double *A, int lda, int i, int j) {
    if (i == j) return;
    for (int k = 0; k < n; k++) { double t = A[i + k * lda]; A[i + k * lda] = A[j + k * lda]; A[j + k * lda] = t; }
    for (int k = 0; k < n; k++) { double t = A[k + i * lda]; A[k + i * lda] = A[k + j * lda]; A[k + j * lda] = t; }
}

/* Simplified GEBAL: for 'B' (both permute and scale), 'P' (permute only),
   'S' (scale only), 'N' (none) */

void sgebal_(const char *job, const int *n,
             float *A, const int *lda,
             int *ilo, int *ihi, float *scale, int *info) {
    *info = 0;
    int N = *n, LDA = *lda;
    if (N == 0) { *ilo = 1; *ihi = 0; return; }

    for (int i = 0; i < N; i++) scale[i] = 1.0f;

    if (lsame(*job, 'N')) {
        *ilo = 1; *ihi = N;
        return;
    }

    /* Permutation phase */
    int lo = 0, hi = N - 1;
    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        /* Search for rows/cols to swap to isolate eigenvalues */
        bool found;
        /* Search from bottom for rows with single nonzero off-diagonal */
        do {
            found = false;
            for (int i = hi; i >= lo; i--) {
                bool isolated = true;
                for (int j = lo; j <= hi; j++) {
                    if (j != i && A[i + j * LDA] != 0.0f) { isolated = false; break; }
                }
                if (isolated) {
                    scale[hi] = (float)(i + 1); /* 1-indexed permutation */
                    if (i != hi) s_swap_rowcol(N, A, LDA, i, hi);
                    hi--;
                    found = true;
                    break;
                }
            }
        } while (found && hi >= lo);

        /* Search from top for cols with single nonzero off-diagonal */
        do {
            found = false;
            for (int j = lo; j <= hi; j++) {
                bool isolated = true;
                for (int i = lo; i <= hi; i++) {
                    if (i != j && A[i + j * LDA] != 0.0f) { isolated = false; break; }
                }
                if (isolated) {
                    scale[lo] = (float)(j + 1);
                    if (j != lo) s_swap_rowcol(N, A, LDA, j, lo);
                    lo++;
                    found = true;
                    break;
                }
            }
        } while (found && lo <= hi);
    }

    *ilo = lo + 1; /* Convert to 1-indexed */
    *ihi = hi + 1;

    if (lsame(*job, 'P') || lo == hi) {
        for (int i = lo; i <= hi; i++) scale[i] = 1.0f;
        return;
    }

    /* Scaling phase */
    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        float sfmin1 = FLT_MIN / FLT_EPSILON;
        float sfmax1 = 1.0f / sfmin1;
        float base = 2.0f;
        bool noconv;
        do {
            noconv = false;
            for (int i = lo; i <= hi; i++) {
                float c = 0.0f, r = 0.0f;
                for (int j = lo; j <= hi; j++) {
                    if (j == i) continue;
                    c += fabsf(A[j + i * LDA]);
                    r += fabsf(A[i + j * LDA]);
                }
                if (c == 0.0f || r == 0.0f) continue;

                float g = r / base;
                float f = 1.0f;
                float s = c + r;
                while (c < g) { f *= base; c *= (base * base); }
                g = r * base;
                while (c >= g) { f /= base; c /= (base * base); }

                if ((c + r) / f >= 0.95f * s) continue;
                if (f < sfmin1 || f > sfmax1) continue;

                g = 1.0f / f;
                scale[i] *= f;
                for (int j = lo; j < N; j++) A[i + j * LDA] *= g;
                for (int j = 0; j <= hi; j++) A[j + i * LDA] *= f;
                noconv = true;
            }
        } while (noconv);
    }
}

void dgebal_(const char *job, const int *n,
             double *A, const int *lda,
             int *ilo, int *ihi, double *scale, int *info) {
    *info = 0;
    int N = *n, LDA = *lda;
    if (N == 0) { *ilo = 1; *ihi = 0; return; }

    for (int i = 0; i < N; i++) scale[i] = 1.0;

    if (lsame(*job, 'N')) {
        *ilo = 1; *ihi = N;
        return;
    }

    int lo = 0, hi = N - 1;
    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        bool found;
        do {
            found = false;
            for (int i = hi; i >= lo; i--) {
                bool isolated = true;
                for (int j = lo; j <= hi; j++) {
                    if (j != i && A[i + j * LDA] != 0.0) { isolated = false; break; }
                }
                if (isolated) {
                    scale[hi] = (double)(i + 1);
                    if (i != hi) d_swap_rowcol(N, A, LDA, i, hi);
                    hi--;
                    found = true;
                    break;
                }
            }
        } while (found && hi >= lo);

        do {
            found = false;
            for (int j = lo; j <= hi; j++) {
                bool isolated = true;
                for (int i = lo; i <= hi; i++) {
                    if (i != j && A[i + j * LDA] != 0.0) { isolated = false; break; }
                }
                if (isolated) {
                    scale[lo] = (double)(j + 1);
                    if (j != lo) d_swap_rowcol(N, A, LDA, j, lo);
                    lo++;
                    found = true;
                    break;
                }
            }
        } while (found && lo <= hi);
    }

    *ilo = lo + 1;
    *ihi = hi + 1;

    if (lsame(*job, 'P') || lo == hi) {
        for (int i = lo; i <= hi; i++) scale[i] = 1.0;
        return;
    }

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        double sfmin1 = DBL_MIN / DBL_EPSILON;
        double sfmax1 = 1.0 / sfmin1;
        double base = 2.0;
        bool noconv;
        do {
            noconv = false;
            for (int i = lo; i <= hi; i++) {
                double c = 0.0, r = 0.0;
                for (int j = lo; j <= hi; j++) {
                    if (j == i) continue;
                    c += fabs(A[j + i * LDA]);
                    r += fabs(A[i + j * LDA]);
                }
                if (c == 0.0 || r == 0.0) continue;

                double g = r / base;
                double f = 1.0;
                double s = c + r;
                while (c < g) { f *= base; c *= (base * base); }
                g = r * base;
                while (c >= g) { f /= base; c /= (base * base); }

                if ((c + r) / f >= 0.95 * s) continue;
                if (f < sfmin1 || f > sfmax1) continue;

                g = 1.0 / f;
                scale[i] *= f;
                for (int j = lo; j < N; j++) A[i + j * LDA] *= g;
                for (int j = 0; j <= hi; j++) A[j + i * LDA] *= f;
                noconv = true;
            }
        } while (noconv);
    }
}

/* Complex GEBAL: permutation uses real scale array, complex matrix */
static void c_swap_rowcol(int n, lapack_float_complex *A, int lda, int i, int j) {
    if (i == j) return;
    for (int k = 0; k < n; k++) { lapack_float_complex t = A[i + k * lda]; A[i + k * lda] = A[j + k * lda]; A[j + k * lda] = t; }
    for (int k = 0; k < n; k++) { lapack_float_complex t = A[k + i * lda]; A[k + i * lda] = A[k + j * lda]; A[k + j * lda] = t; }
}
static void z_swap_rowcol(int n, lapack_double_complex *A, int lda, int i, int j) {
    if (i == j) return;
    for (int k = 0; k < n; k++) { lapack_double_complex t = A[i + k * lda]; A[i + k * lda] = A[j + k * lda]; A[j + k * lda] = t; }
    for (int k = 0; k < n; k++) { lapack_double_complex t = A[k + i * lda]; A[k + i * lda] = A[k + j * lda]; A[k + j * lda] = t; }
}

void cgebal_(const char *job, const int *n,
             lapack_float_complex *A, const int *lda,
             int *ilo, int *ihi, float *scale, int *info) {
    *info = 0;
    int N = *n, LDA = *lda;
    if (N == 0) { *ilo = 1; *ihi = 0; return; }
    for (int i = 0; i < N; i++) scale[i] = 1.0f;
    if (lsame(*job, 'N')) { *ilo = 1; *ihi = N; return; }

    int lo = 0, hi = N - 1;
    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        bool found;
        do {
            found = false;
            for (int i = hi; i >= lo; i--) {
                bool isolated = true;
                for (int j = lo; j <= hi; j++) {
                    if (j != i && (A[i + j * LDA].x != 0.0f || A[i + j * LDA].y != 0.0f))
                    { isolated = false; break; }
                }
                if (isolated) {
                    scale[hi] = (float)(i + 1);
                    if (i != hi) c_swap_rowcol(N, A, LDA, i, hi);
                    hi--; found = true; break;
                }
            }
        } while (found && hi >= lo);
        do {
            found = false;
            for (int j = lo; j <= hi; j++) {
                bool isolated = true;
                for (int i = lo; i <= hi; i++) {
                    if (i != j && (A[i + j * LDA].x != 0.0f || A[i + j * LDA].y != 0.0f))
                    { isolated = false; break; }
                }
                if (isolated) {
                    scale[lo] = (float)(j + 1);
                    if (j != lo) c_swap_rowcol(N, A, LDA, j, lo);
                    lo++; found = true; break;
                }
            }
        } while (found && lo <= hi);
    }
    *ilo = lo + 1; *ihi = hi + 1;
    if (lsame(*job, 'P') || lo == hi) { for (int i = lo; i <= hi; i++) scale[i] = 1.0f; return; }

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        float sfmin1 = FLT_MIN / FLT_EPSILON;
        float sfmax1 = 1.0f / sfmin1;
        float base = 2.0f;
        bool noconv;
        do {
            noconv = false;
            for (int i = lo; i <= hi; i++) {
                float c = 0.0f, r = 0.0f;
                for (int j = lo; j <= hi; j++) {
                    if (j == i) continue;
                    c += cf_abs(A[j + i * LDA]);
                    r += cf_abs(A[i + j * LDA]);
                }
                if (c == 0.0f || r == 0.0f) continue;
                float g = r / base, f = 1.0f, s = c + r;
                while (c < g) { f *= base; c *= (base * base); }
                g = r * base;
                while (c >= g) { f /= base; c /= (base * base); }
                if ((c + r) / f >= 0.95f * s) continue;
                if (f < sfmin1 || f > sfmax1) continue;
                g = 1.0f / f;
                scale[i] *= f;
                for (int j = lo; j < N; j++) A[i + j * LDA] = cf_scale(g, A[i + j * LDA]);
                for (int j = 0; j <= hi; j++) A[j + i * LDA] = cf_scale(f, A[j + i * LDA]);
                noconv = true;
            }
        } while (noconv);
    }
}

void zgebal_(const char *job, const int *n,
             lapack_double_complex *A, const int *lda,
             int *ilo, int *ihi, double *scale, int *info) {
    *info = 0;
    int N = *n, LDA = *lda;
    if (N == 0) { *ilo = 1; *ihi = 0; return; }
    for (int i = 0; i < N; i++) scale[i] = 1.0;
    if (lsame(*job, 'N')) { *ilo = 1; *ihi = N; return; }

    int lo = 0, hi = N - 1;
    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        bool found;
        do {
            found = false;
            for (int i = hi; i >= lo; i--) {
                bool isolated = true;
                for (int j = lo; j <= hi; j++) {
                    if (j != i && (A[i + j * LDA].x != 0.0 || A[i + j * LDA].y != 0.0))
                    { isolated = false; break; }
                }
                if (isolated) {
                    scale[hi] = (double)(i + 1);
                    if (i != hi) z_swap_rowcol(N, A, LDA, i, hi);
                    hi--; found = true; break;
                }
            }
        } while (found && hi >= lo);
        do {
            found = false;
            for (int j = lo; j <= hi; j++) {
                bool isolated = true;
                for (int i = lo; i <= hi; i++) {
                    if (i != j && (A[i + j * LDA].x != 0.0 || A[i + j * LDA].y != 0.0))
                    { isolated = false; break; }
                }
                if (isolated) {
                    scale[lo] = (double)(j + 1);
                    if (j != lo) z_swap_rowcol(N, A, LDA, j, lo);
                    lo++; found = true; break;
                }
            }
        } while (found && lo <= hi);
    }
    *ilo = lo + 1; *ihi = hi + 1;
    if (lsame(*job, 'P') || lo == hi) { for (int i = lo; i <= hi; i++) scale[i] = 1.0; return; }

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        double sfmin1 = DBL_MIN / DBL_EPSILON;
        double sfmax1 = 1.0 / sfmin1;
        double base = 2.0;
        bool noconv;
        do {
            noconv = false;
            for (int i = lo; i <= hi; i++) {
                double c = 0.0, r = 0.0;
                for (int j = lo; j <= hi; j++) {
                    if (j == i) continue;
                    c += zf_abs(A[j + i * LDA]);
                    r += zf_abs(A[i + j * LDA]);
                }
                if (c == 0.0 || r == 0.0) continue;
                double g = r / base, f = 1.0, s = c + r;
                while (c < g) { f *= base; c *= (base * base); }
                g = r * base;
                while (c >= g) { f /= base; c /= (base * base); }
                if ((c + r) / f >= 0.95 * s) continue;
                if (f < sfmin1 || f > sfmax1) continue;
                g = 1.0 / f;
                scale[i] *= f;
                for (int j = lo; j < N; j++) A[i + j * LDA] = zf_scale(g, A[i + j * LDA]);
                for (int j = 0; j <= hi; j++) A[j + i * LDA] = zf_scale(f, A[j + i * LDA]);
                noconv = true;
            }
        } while (noconv);
    }
}

/* ========================================================================
   LAPACK Computational - GEBAK: back-transform eigenvectors after balancing
   ======================================================================== */

void sgebak_(const char *job, const char *side,
             const int *n, const int *ilo, const int *ihi,
             const float *scale, const int *m,
             float *V, const int *ldv, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, M = *m, LDV = *ldv;
    if (N == 0 || M == 0) return;
    if (lsame(*job, 'N')) return;

    /* Undo scaling */
    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        if (lsame(*side, 'R')) {
            for (int i = ILO - 1; i < IHI; i++) {
                float s = scale[i];
                for (int j = 0; j < M; j++)
                    V[i + j * LDV] *= s;
            }
        } else {
            for (int i = ILO - 1; i < IHI; i++) {
                float s = 1.0f / scale[i];
                for (int j = 0; j < M; j++)
                    V[i + j * LDV] *= s;
            }
        }
    }

    /* Undo permutation */
    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        /* Backward permutation for IHI+1..N */
        for (int i = IHI; i < N; i++) {
            int k = (int)scale[i] - 1; /* 0-indexed */
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    float t = V[i + j * LDV];
                    V[i + j * LDV] = V[k + j * LDV];
                    V[k + j * LDV] = t;
                }
            }
        }
        /* Forward permutation for 1..ILO-1 */
        for (int i = ILO - 2; i >= 0; i--) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    float t = V[i + j * LDV];
                    V[i + j * LDV] = V[k + j * LDV];
                    V[k + j * LDV] = t;
                }
            }
        }
    }
}

void dgebak_(const char *job, const char *side,
             const int *n, const int *ilo, const int *ihi,
             const double *scale, const int *m,
             double *V, const int *ldv, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, M = *m, LDV = *ldv;
    if (N == 0 || M == 0) return;
    if (lsame(*job, 'N')) return;

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        if (lsame(*side, 'R')) {
            for (int i = ILO - 1; i < IHI; i++) {
                double s = scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] *= s;
            }
        } else {
            for (int i = ILO - 1; i < IHI; i++) {
                double s = 1.0 / scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] *= s;
            }
        }
    }

    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        for (int i = IHI; i < N; i++) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    double t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
        for (int i = ILO - 2; i >= 0; i--) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    double t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
    }
}

void cgebak_(const char *job, const char *side,
             const int *n, const int *ilo, const int *ihi,
             const float *scale, const int *m,
             lapack_float_complex *V, const int *ldv, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, M = *m, LDV = *ldv;
    if (N == 0 || M == 0) return;
    if (lsame(*job, 'N')) return;

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        if (lsame(*side, 'R')) {
            for (int i = ILO - 1; i < IHI; i++) {
                float s = scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] = cf_scale(s, V[i + j * LDV]);
            }
        } else {
            for (int i = ILO - 1; i < IHI; i++) {
                float s = 1.0f / scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] = cf_scale(s, V[i + j * LDV]);
            }
        }
    }

    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        for (int i = IHI; i < N; i++) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    lapack_float_complex t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
        for (int i = ILO - 2; i >= 0; i--) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    lapack_float_complex t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
    }
}

void zgebak_(const char *job, const char *side,
             const int *n, const int *ilo, const int *ihi,
             const double *scale, const int *m,
             lapack_double_complex *V, const int *ldv, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, M = *m, LDV = *ldv;
    if (N == 0 || M == 0) return;
    if (lsame(*job, 'N')) return;

    if (lsame(*job, 'B') || lsame(*job, 'S')) {
        if (lsame(*side, 'R')) {
            for (int i = ILO - 1; i < IHI; i++) {
                double s = scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] = zf_scale(s, V[i + j * LDV]);
            }
        } else {
            for (int i = ILO - 1; i < IHI; i++) {
                double s = 1.0 / scale[i];
                for (int j = 0; j < M; j++) V[i + j * LDV] = zf_scale(s, V[i + j * LDV]);
            }
        }
    }

    if (lsame(*job, 'B') || lsame(*job, 'P')) {
        for (int i = IHI; i < N; i++) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    lapack_double_complex t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
        for (int i = ILO - 2; i >= 0; i--) {
            int k = (int)scale[i] - 1;
            if (k != i) {
                for (int j = 0; j < M; j++) {
                    lapack_double_complex t = V[i + j * LDV]; V[i + j * LDV] = V[k + j * LDV]; V[k + j * LDV] = t;
                }
            }
        }
    }
}

/* ========================================================================
   LAPACK Auxiliary - DLANV2: Schur factorization of 2x2 real nonsymmetric
   matrix in standard form.  Translated from LAPACK dlanv2.f.
   ======================================================================== */

static void dlanv2_impl(double *a, double *b, double *c, double *d,
                         double *rt1r, double *rt1i, double *rt2r, double *rt2i,
                         double *cs, double *sn) {
    const double zero = 0.0, half = 0.5, one = 1.0;
    const double multpl = 4.0;
    double eps = DBL_EPSILON;

    if (*c == zero) {
        *cs = one;
        *sn = zero;
    } else if (*b == zero) {
        /* Swap rows and columns */
        *cs = zero;
        *sn = one;
        double temp = *d;
        *d = *a;
        *a = temp;
        *b = -*c;
        *c = zero;
    } else if ((*a - *d) == zero && copysign(one, *b) != copysign(one, *c)) {
        *cs = one;
        *sn = zero;
    } else {
        double temp = *a - *d;
        double p = half * temp;
        double bcmax = std::max(fabs(*b), fabs(*c));
        double bcmis = std::min(fabs(*b), fabs(*c)) * copysign(one, *b) * copysign(one, *c);
        double scale = std::max(fabs(p), bcmax);
        double z = (p / scale) * p + (bcmax / scale) * bcmis;

        if (z >= multpl * eps) {
            /* Real eigenvalues. */
            z = p + copysign(sqrt(scale) * sqrt(z), p);
            *a = *d + z;
            *d = *d - (bcmax / z) * bcmis;
            double x_dlapy2 = *c, y_dlapy2 = z;
            double tau = dlapy2_(&x_dlapy2, &y_dlapy2);
            *cs = z / tau;
            *sn = *c / tau;
            *b = *b - *c;
            *c = zero;
        } else {
            /* Complex eigenvalues, or real (almost) equal eigenvalues. */
            double sigma = *b + *c;
            double x_dlapy2 = sigma, y_dlapy2 = temp;
            double tau = dlapy2_(&x_dlapy2, &y_dlapy2);
            *cs = sqrt(half * (one + fabs(sigma) / tau));
            *sn = -(p / (tau * (*cs))) * copysign(one, sigma);

            double aa = (*a) * (*cs) + (*b) * (*sn);
            double bb = -(*a) * (*sn) + (*b) * (*cs);
            double cc = (*c) * (*cs) + (*d) * (*sn);
            double dd = -(*c) * (*sn) + (*d) * (*cs);

            *a = aa * (*cs) + cc * (*sn);
            *b = bb * (*cs) + dd * (*sn);
            *c = -aa * (*sn) + cc * (*cs);
            *d = -bb * (*sn) + dd * (*cs);

            temp = half * (*a + *d);
            *a = temp;
            *d = temp;

            if (*c != zero) {
                if (*b != zero) {
                    if (copysign(one, *b) == copysign(one, *c)) {
                        /* Real eigenvalues: reduce to upper triangular form */
                        double sab = sqrt(fabs(*b));
                        double sac = sqrt(fabs(*c));
                        p = copysign(sab * sac, *c);
                        tau = one / sqrt(fabs(*b + *c));
                        *a = temp + p;
                        *d = temp - p;
                        *b = *b - *c;
                        *c = zero;
                        double cs1 = sab * tau;
                        double sn1 = sac * tau;
                        temp = (*cs) * cs1 - (*sn) * sn1;
                        *sn = (*cs) * sn1 + (*sn) * cs1;
                        *cs = temp;
                    }
                } else {
                    *b = -*c;
                    *c = zero;
                    temp = *cs;
                    *cs = -*sn;
                    *sn = temp;
                }
            }
        }
    }

    /* Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I). */
    *rt1r = *a;
    *rt2r = *d;
    if (*c == zero) {
        *rt1i = zero;
        *rt2i = zero;
    } else {
        *rt1i = sqrt(fabs(*b)) * sqrt(fabs(*c));
        *rt2i = -*rt1i;
    }
}

/* ========================================================================
   LAPACK Computational - DLAHQR: small matrix QR iteration (real)
   This is the core of HSEQR for small/medium Hessenberg matrices.
   Implements the double-shift Francis QR algorithm.
   Faithfully translated from LAPACK dlahqr.f.
   NOTE: ilo, ihi, iloz, ihiz are 0-indexed (our convention).
   ======================================================================== */

static void dlahqr_impl(int wantt, int wantz, int n, int ilo, int ihi,
                         double *H, int ldh, double *wr, double *wi,
                         int iloz, int ihiz, double *Z, int ldz, int *info) {
    const double zero = 0.0, one = 1.0, two = 2.0;
    const double dat1 = 3.0 / 4.0, dat2 = -0.4375;

    *info = 0;

    /* Quick return if possible */
    if (n == 0) return;
    if (ilo == ihi) {
        wr[ilo] = H[ilo + ilo * ldh];
        wi[ilo] = zero;
        return;
    }

    /* Clear out the trash below the subdiagonal */
    for (int j = ilo; j <= ihi - 2; j++) {
        H[(j + 2) + j * ldh] = zero;
        if (j <= ihi - 3)
            H[(j + 3) + j * ldh] = zero;
    }
    /* Note: the j=ihi-3 case in Fortran writes H(IHI,IHI-2).
       The loop above already covers it when j <= ihi-3, but we also need
       the separate check for the last element: */
    if (ilo <= ihi - 2)
        H[ihi + (ihi - 2) * ldh] = zero;

    int nh = ihi - ilo + 1;
    int nz = ihiz - iloz + 1;

    /* Set machine-dependent constants for the stopping criterion. */
    double safmin = DBL_MIN;
    double safmax = one / safmin;
    dlabad_(&safmin, &safmax);
    double ulp = DBL_EPSILON;
    double smlnum = safmin * ((double)nh / ulp);

    /* I1 and I2 are the indices of the first row and last column of H
       to which transformations must be applied. (0-indexed) */
    int i1 = 0, i2 = 0;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    }

    /* ITMAX is the total number of QR iterations allowed. */
    int itmax = 30 * std::max(10, nh);

    /* The main loop begins here. I is the loop index and decreases from
       IHI to ILO in steps of 1 or 2. */
    int i = ihi;

    while (i >= ilo) {
        int l = ilo;
        double s = zero;
        double v[3] = {zero, zero, zero};

        /* Perform QR iterations on rows and columns ILO to I until a
           submatrix of order 1 or 2 splits off at the bottom. */
        int converged = 0;
        for (int its = 0; its <= itmax; its++) {

            /* Look for a single small subdiagonal element. */
            int k;
            for (k = i; k >= l + 1; k--) {
                if (fabs(H[k + (k - 1) * ldh]) <= smlnum)
                    break;
                double tst = fabs(H[(k - 1) + (k - 1) * ldh]) + fabs(H[k + k * ldh]);
                if (tst == zero) {
                    if (k - 2 >= ilo)
                        tst += fabs(H[(k - 1) + (k - 2) * ldh]);
                    if (k + 1 <= ihi)
                        tst += fabs(H[(k + 1) + k * ldh]);
                }
                /* Conservative small subdiagonal deflation criterion
                   due to Ahues & Tisseur (LAWN 122, 1997). */
                if (fabs(H[k + (k - 1) * ldh]) <= ulp * tst) {
                    double ab = std::max(fabs(H[k + (k - 1) * ldh]), fabs(H[(k - 1) + k * ldh]));
                    double ba = std::min(fabs(H[k + (k - 1) * ldh]), fabs(H[(k - 1) + k * ldh]));
                    double aa = std::max(fabs(H[k + k * ldh]),
                                         fabs(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]));
                    double bb = std::min(fabs(H[k + k * ldh]),
                                         fabs(H[(k - 1) + (k - 1) * ldh] - H[k + k * ldh]));
                    double s = aa + ab;
                    if (ba * (ab / s) <= std::max(smlnum, ulp * (bb * (aa / s))))
                        break;
                }
            }
            l = k;
            if (l > ilo) {
                /* H(L,L-1) is negligible */
                H[l + (l - 1) * ldh] = zero;
            }

            /* Exit from loop if a submatrix of order 1 or 2 has split off. */
            if (l >= i - 1) {
                converged = 1;
                break;
            }

            /* Now the active submatrix is in rows and columns L to I. */
            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            double h11, h12, h21, h22;
            if (its == 10) {
                /* Exceptional shift. */
                s = fabs(H[(l + 1) + l * ldh]) + fabs(H[(l + 2) + (l + 1) * ldh]);
                h11 = dat1 * s + H[l + l * ldh];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else if (its == 20) {
                /* Exceptional shift. */
                s = fabs(H[i + (i - 1) * ldh]) + fabs(H[(i - 1) + (i - 2) * ldh]);
                h11 = dat1 * s + H[i + i * ldh];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else {
                /* Prepare to use Francis' double shift */
                h11 = H[(i - 1) + (i - 1) * ldh];
                h21 = H[i + (i - 1) * ldh];
                h12 = H[(i - 1) + i * ldh];
                h22 = H[i + i * ldh];
            }

            s = fabs(h11) + fabs(h12) + fabs(h21) + fabs(h22);
            double rt1r, rt1i, rt2r, rt2i;
            if (s == zero) {
                rt1r = zero; rt1i = zero;
                rt2r = zero; rt2i = zero;
            } else {
                h11 /= s; h21 /= s; h12 /= s; h22 /= s;
                double tr = (h11 + h22) / two;
                double det = (h11 - tr) * (h22 - tr) - h12 * h21;
                double rtdisc = sqrt(fabs(det));
                if (det >= zero) {
                    /* Complex conjugate shifts */
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    /* Real shifts (use only one of them) */
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (fabs(rt1r - h22) <= fabs(rt2r - h22)) {
                        rt1r *= s;
                        rt2r = rt1r;
                    } else {
                        rt2r *= s;
                        rt1r = rt2r;
                    }
                    rt1i = zero;
                    rt2i = zero;
                }
            }

            /* Look for two consecutive small subdiagonal elements. */
            int m;
            for (m = i - 2; m >= l; m--) {
                double h21s = H[(m + 1) + m * ldh];
                s = fabs(H[m + m * ldh] - rt2r) + fabs(rt2i) + fabs(h21s);
                h21s = H[(m + 1) + m * ldh] / s;
                v[0] = h21s * H[m + (m + 1) * ldh] + (H[m + m * ldh] - rt1r) *
                       ((H[m + m * ldh] - rt2r) / s) - rt1i * (rt2i / s);
                v[1] = h21s * (H[m + m * ldh] + H[(m + 1) + (m + 1) * ldh] - rt1r - rt2r);
                v[2] = h21s * H[(m + 2) + (m + 1) * ldh];
                s = fabs(v[0]) + fabs(v[1]) + fabs(v[2]);
                v[0] /= s;
                v[1] /= s;
                v[2] /= s;
                if (m == l)
                    break;
                if (fabs(H[m + (m - 1) * ldh]) * (fabs(v[1]) + fabs(v[2])) <=
                    ulp * fabs(v[0]) * (fabs(H[(m - 1) + (m - 1) * ldh]) +
                    fabs(H[m + m * ldh]) + fabs(H[(m + 1) + (m + 1) * ldh])))
                    break;
            }

            /* Double-shift QR step */
            for (int kk = m; kk <= i - 1; kk++) {
                int nr = std::min(3, i - kk + 1);
                if (kk > m) {
                    /* Copy column of H into v */
                    for (int jj = 0; jj < nr; jj++)
                        v[jj] = H[(kk + jj) + (kk - 1) * ldh];
                }
                double t1;
                int incv = 1;
                dlarfg_(&nr, &v[0], &v[1], &incv, &t1);
                if (kk > m) {
                    H[kk + (kk - 1) * ldh] = v[0];
                    H[(kk + 1) + (kk - 1) * ldh] = zero;
                    if (kk < i - 1)
                        H[(kk + 2) + (kk - 1) * ldh] = zero;
                } else if (m > l) {
                    /* Use the following instead of
                       H(K,K-1) = -H(K,K-1) to avoid a bug when
                       v[1] and v[2] underflow. */
                    H[kk + (kk - 1) * ldh] *= (one - t1);
                }
                double v2 = v[1];
                double t2 = t1 * v2;
                if (nr == 3) {
                    double v3 = v[2];
                    double t3 = t1 * v3;

                    /* Apply G from the left to transform rows of H
                       in columns KK to I2. */
                    for (int j = kk; j <= i2; j++) {
                        double sum = H[kk + j * ldh] + v2 * H[(kk + 1) + j * ldh]
                                     + v3 * H[(kk + 2) + j * ldh];
                        H[kk + j * ldh] -= sum * t1;
                        H[(kk + 1) + j * ldh] -= sum * t2;
                        H[(kk + 2) + j * ldh] -= sum * t3;
                    }

                    /* Apply G from the right to transform columns of H
                       in rows I1 to min(KK+3, I). */
                    for (int j = i1; j <= std::min(kk + 3, i); j++) {
                        double sum = H[j + kk * ldh] + v2 * H[j + (kk + 1) * ldh]
                                     + v3 * H[j + (kk + 2) * ldh];
                        H[j + kk * ldh] -= sum * t1;
                        H[j + (kk + 1) * ldh] -= sum * t2;
                        H[j + (kk + 2) * ldh] -= sum * t3;
                    }

                    if (wantz) {
                        /* Accumulate transformations in the matrix Z */
                        for (int j = iloz; j <= ihiz; j++) {
                            double sum = Z[j + kk * ldz] + v2 * Z[j + (kk + 1) * ldz]
                                         + v3 * Z[j + (kk + 2) * ldz];
                            Z[j + kk * ldz] -= sum * t1;
                            Z[j + (kk + 1) * ldz] -= sum * t2;
                            Z[j + (kk + 2) * ldz] -= sum * t3;
                        }
                    }
                } else if (nr == 2) {
                    /* Apply G from the left to transform rows of H
                       in columns KK to I2. */
                    for (int j = kk; j <= i2; j++) {
                        double sum = H[kk + j * ldh] + v2 * H[(kk + 1) + j * ldh];
                        H[kk + j * ldh] -= sum * t1;
                        H[(kk + 1) + j * ldh] -= sum * t2;
                    }

                    /* Apply G from the right to transform columns of H
                       in rows I1 to I. */
                    for (int j = i1; j <= i; j++) {
                        double sum = H[j + kk * ldh] + v2 * H[j + (kk + 1) * ldh];
                        H[j + kk * ldh] -= sum * t1;
                        H[j + (kk + 1) * ldh] -= sum * t2;
                    }

                    if (wantz) {
                        /* Accumulate transformations in the matrix Z */
                        for (int j = iloz; j <= ihiz; j++) {
                            double sum = Z[j + kk * ldz] + v2 * Z[j + (kk + 1) * ldz];
                            Z[j + kk * ldz] -= sum * t1;
                            Z[j + (kk + 1) * ldz] -= sum * t2;
                        }
                    }
                }
            }
        }

        if (!converged) {
            /* Failure to converge in remaining number of iterations */
            *info = i + 1;  /* 1-indexed failure point for caller */
            return;
        }

        if (l == i) {
            /* H(I,I-1) is negligible: one eigenvalue has converged. */
            wr[i] = H[i + i * ldh];
            wi[i] = zero;
        } else if (l == i - 1) {
            /* H(I-1,I-2) is negligible: a pair of eigenvalues have converged.
               Transform the 2-by-2 submatrix to standard Schur form,
               and compute and store the eigenvalues. */
            dlanv2_impl(&H[(i - 1) + (i - 1) * ldh], &H[(i - 1) + i * ldh],
                        &H[i + (i - 1) * ldh], &H[i + i * ldh],
                        &wr[i - 1], &wi[i - 1], &wr[i], &wi[i],
                        &s, &v[0]);

            if (wantt) {
                /* Apply the transformation to the rest of H. */
                if (i2 > i) {
                    int len = i2 - i;
                    drot_(&len, &H[(i - 1) + (i + 1) * ldh], &ldh,
                          &H[i + (i + 1) * ldh], &ldh, &s, &v[0]);
                }
                int len = i - i1 - 1;
                int one_int = 1;
                drot_(&len, &H[i1 + (i - 1) * ldh], &one_int,
                      &H[i1 + i * ldh], &one_int, &s, &v[0]);
            }
            if (wantz) {
                /* Apply the transformation to Z. */
                int one_int = 1;
                drot_(&nz, &Z[iloz + (i - 1) * ldz], &one_int,
                      &Z[iloz + i * ldz], &one_int, &s, &v[0]);
            }
        }

        /* Return to start of the main loop with new value of I. */
        i = l - 1;
    }
}

/* ========================================================================
   LAPACK Computational - HSEQR: Schur decomposition (real)
   For geev, the matrix is already Hessenberg.
   job='E' -> eigenvalues only, 'S' -> Schur form
   compz='N' -> no Z, 'I' -> initialize Z to I, 'V' -> use input Z
   ======================================================================== */

void shseqr_(const char *job, const char *compz,
             const int *n, const int *ilo, const int *ihi,
             float *H, const int *ldh,
             float *wr, float *wi,
             float *Z, const int *ldz,
             float *work, const int *lwork, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDH = *ldh, LDZ = *ldz;

    if (*lwork == -1) {
        work[0] = (float)std::max(1, N);
        return;
    }

    if (N == 0) return;
    if (N == 1) {
        wr[0] = H[0]; wi[0] = 0.0f;
        return;
    }

    bool wantt = lsame(*job, 'S');
    bool wantz = !lsame(*compz, 'N');

    if (lsame(*compz, 'I')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = (i == j) ? 1.0f : 0.0f;
    }

    /* Convert to double, call dlahqr, convert back */
    std::vector<double> dH(N * N), dZ(N * N), dwr(N), dwi(N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            dH[i + j * N] = H[i + j * LDH];

    if (wantz) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                dZ[i + j * N] = Z[i + j * LDZ];
    }

    dlahqr_impl(wantt ? 1 : 0, wantz ? 1 : 0, N, ILO - 1, IHI - 1,
                dH.data(), N, dwr.data(), dwi.data(),
                0, N - 1, dZ.data(), N, info);

    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            H[i + j * LDH] = (float)dH[i + j * N];

    if (wantz) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = (float)dZ[i + j * N];
    }

    for (int i = 0; i < N; i++) { wr[i] = (float)dwr[i]; wi[i] = (float)dwi[i]; }
}

void dhseqr_(const char *job, const char *compz,
             const int *n, const int *ilo, const int *ihi,
             double *H, const int *ldh,
             double *wr, double *wi,
             double *Z, const int *ldz,
             double *work, const int *lwork, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDH = *ldh, LDZ = *ldz;

    if (*lwork == -1) {
        work[0] = (double)std::max(1, N);
        return;
    }

    if (N == 0) return;
    if (N == 1) {
        wr[0] = H[0]; wi[0] = 0.0;
        return;
    }

    bool wantt = lsame(*job, 'S');
    bool wantz = !lsame(*compz, 'N');

    if (lsame(*compz, 'I')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = (i == j) ? 1.0 : 0.0;
    }

    dlahqr_impl(wantt ? 1 : 0, wantz ? 1 : 0, N, ILO - 1, IHI - 1,
                H, LDH, wr, wi, 0, N - 1, Z, LDZ, info);
}

/* ========================================================================
   LAPACK Computational - ZHSEQR/CHSEQR: complex Schur decomposition
   For complex Hessenberg QR iteration
   ======================================================================== */

/* Complex QR iteration (single shift) for upper Hessenberg matrix.
   Faithfully translated from LAPACK zlahqr.f.
   NOTE: ilo, ihi, iloz, ihiz are 0-indexed (our convention). */
static void zlahqr_impl(int wantt, int wantz, int n, int ilo, int ihi,
                         lapack_double_complex *H, int ldh,
                         lapack_double_complex *w,
                         int iloz, int ihiz,
                         lapack_double_complex *Z, int ldz, int *info) {
    const lapack_double_complex czero = {0.0, 0.0}, cone = {1.0, 0.0};
    const double rzero = 0.0, rone = 1.0, half = 0.5, dat1 = 3.0 / 4.0;

    *info = 0;

    /* Quick return if possible */
    if (n == 0) return;
    if (ilo == ihi) {
        w[ilo] = H[ilo + ilo * ldh];
        return;
    }

    /* Clear out the trash below the subdiagonal */
    for (int j = ilo; j <= ihi - 2; j++) {
        H[(j + 2) + j * ldh] = czero;
        if (j <= ihi - 3)
            H[(j + 3) + j * ldh] = czero;
    }
    if (ilo <= ihi - 2)
        H[ihi + (ihi - 2) * ldh] = czero;

    /* Ensure that subdiagonal entries are real */
    int jlo, jhi_sc;
    if (wantt) {
        jlo = 0;
        jhi_sc = n - 1;
    } else {
        jlo = ilo;
        jhi_sc = ihi;
    }
    for (int ii = ilo + 1; ii <= ihi; ii++) {
        if (H[ii + (ii - 1) * ldh].y != rzero) {
            lapack_double_complex sc = H[ii + (ii - 1) * ldh];
            double cabs1_sc = fabs(sc.x) + fabs(sc.y);
            sc = zf_scale(1.0 / cabs1_sc, sc);  /* sc = sc / cabs1(sc) */
            sc = zf_scale(1.0 / zf_abs(sc), zf_conj(sc));  /* sc = conj(sc) / |sc| */
            H[ii + (ii - 1) * ldh] = zf_make(zf_abs(H[ii + (ii - 1) * ldh]), 0.0);
            /* Scale row ii from column ii to jhi_sc */
            int len = jhi_sc - ii + 1;
            if (len > 0)
                zscal_(&len, &sc, &H[ii + ii * ldh], &ldh);
            /* Scale column ii from row jlo to min(jhi_sc, ii+1) */
            len = std::min(jhi_sc, ii + 1) - jlo + 1;
            lapack_double_complex sc_conj = zf_conj(sc);
            if (len > 0) {
                int one_int = 1;
                zscal_(&len, &sc_conj, &H[jlo + ii * ldh], &one_int);
            }
            if (wantz) {
                int nz = ihiz - iloz + 1;
                int one_int = 1;
                zscal_(&nz, &sc_conj, &Z[iloz + ii * ldz], &one_int);
            }
        }
    }

    int nh = ihi - ilo + 1;
    int nz = ihiz - iloz + 1;

    /* Set machine-dependent constants for the stopping criterion. */
    double safmin = DBL_MIN;
    double safmax = rone / safmin;
    dlabad_(&safmin, &safmax);
    double ulp = DBL_EPSILON;
    double smlnum = safmin * ((double)nh / ulp);

    /* I1 and I2 are the indices of the first row and last column of H
       to which transformations must be applied. (0-indexed) */
    int i1 = 0, i2 = 0;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    }

    /* ITMAX is the total number of QR iterations allowed. */
    int itmax = 30 * std::max(10, nh);

    /* Main loop */
    int i = ihi;

    while (i >= ilo) {
        int l = ilo;

        /* Perform QR iterations on rows and columns ILO to I until a
           submatrix of order 1 splits off at the bottom. */
        int converged = 0;
        for (int its = 0; its <= itmax; its++) {

            /* Look for a single small subdiagonal element. */
            int k;
            for (k = i; k >= l + 1; k--) {
                if (zf_abs1(H[k + (k - 1) * ldh]) <= smlnum)
                    break;
                double tst = zf_abs1(H[(k - 1) + (k - 1) * ldh]) + zf_abs1(H[k + k * ldh]);
                if (tst == rzero) {
                    if (k - 2 >= ilo)
                        tst += fabs(H[(k - 1) + (k - 2) * ldh].x);
                    if (k + 1 <= ihi)
                        tst += fabs(H[(k + 1) + k * ldh].x);
                }
                /* Ahues & Tisseur deflation criterion (LAWN 122, 1997). */
                if (fabs(H[k + (k - 1) * ldh].x) <= ulp * tst) {
                    double ab = std::max(zf_abs1(H[k + (k - 1) * ldh]),
                                         zf_abs1(H[(k - 1) + k * ldh]));
                    double ba = std::min(zf_abs1(H[k + (k - 1) * ldh]),
                                         zf_abs1(H[(k - 1) + k * ldh]));
                    double aa = std::max(zf_abs1(H[k + k * ldh]),
                                         zf_abs1(zf_sub(H[(k - 1) + (k - 1) * ldh], H[k + k * ldh])));
                    double bb = std::min(zf_abs1(H[k + k * ldh]),
                                         zf_abs1(zf_sub(H[(k - 1) + (k - 1) * ldh], H[k + k * ldh])));
                    double ss = aa + ab;
                    if (ba * (ab / ss) <= std::max(smlnum, ulp * (bb * (aa / ss))))
                        break;
                }
            }
            l = k;
            if (l > ilo) {
                H[l + (l - 1) * ldh] = czero;
            }

            /* Exit from loop if a submatrix of order 1 has split off. */
            if (l >= i) {
                converged = 1;
                break;
            }

            /* Now the active submatrix is in rows and columns L to I. */
            if (!wantt) {
                i1 = l;
                i2 = i;
            }

            lapack_double_complex t;
            double s;
            if (its == 10) {
                /* Exceptional shift. */
                s = dat1 * fabs(H[(l + 1) + l * ldh].x);
                t = zf_add(zf_make(s, 0), H[l + l * ldh]);
            } else if (its == 20) {
                /* Exceptional shift. */
                s = dat1 * fabs(H[i + (i - 1) * ldh].x);
                t = zf_add(zf_make(s, 0), H[i + i * ldh]);
            } else {
                /* Wilkinson's shift. */
                t = H[i + i * ldh];
                /* u = sqrt(H(I-1,I)) * sqrt(H(I,I-1))  (complex sqrt) */
                auto csqrt_impl = [](lapack_double_complex z) -> lapack_double_complex {
                    double r = hypot(z.x, z.y);
                    if (r == 0.0) return {0.0, 0.0};
                    double re = sqrt((r + z.x) / 2.0);
                    double im = (z.y >= 0 ? 1.0 : -1.0) * sqrt(std::max(0.0, (r - z.x) / 2.0));
                    return {re, im};
                };
                lapack_double_complex u = zf_mul(csqrt_impl(H[(i - 1) + i * ldh]),
                                                  csqrt_impl(H[i + (i - 1) * ldh]));

                s = zf_abs1(u);
                if (s != rzero) {
                    lapack_double_complex x = zf_scale(half, zf_sub(H[(i - 1) + (i - 1) * ldh], t));
                    double sx = zf_abs1(x);
                    s = std::max(s, zf_abs1(x));
                    /* y = s * sqrt((x/s)^2 + (u/s)^2) */
                    lapack_double_complex x_over_s = zf_scale(1.0 / s, x);
                    lapack_double_complex u_over_s = zf_scale(1.0 / s, u);
                    lapack_double_complex y_sq = zf_add(zf_mul(x_over_s, x_over_s),
                                                        zf_mul(u_over_s, u_over_s));
                    /* sqrt of y_sq */
                    double yr = y_sq.x, yi = y_sq.y;
                    double absy = hypot(yr, yi);
                    double sqrtr_y = sqrt((absy + yr) / 2.0);
                    double sqrti_y = (yi >= 0 ? 1.0 : -1.0) * sqrt(std::max(0.0, (absy - yr) / 2.0));
                    lapack_double_complex y = zf_scale(s, zf_make(sqrtr_y, sqrti_y));

                    if (sx > rzero) {
                        lapack_double_complex x_over_sx = zf_scale(1.0 / sx, x);
                        if (x_over_sx.x * y.x + x_over_sx.y * y.y < rzero)
                            y = zf_make(-y.x, -y.y);
                    }
                    t = zf_sub(t, zf_mul(u, zf_div(u, zf_add(x, y))));
                }
            }

            /* Look for two consecutive small subdiagonal elements. */
            int m;
            lapack_double_complex v_vec[2];
            lapack_double_complex h11, h22, h11s;
            double h21;
            for (m = i - 1; m >= l + 1; m--) {
                h11 = H[m + m * ldh];
                h22 = H[(m + 1) + (m + 1) * ldh];
                h11s = zf_sub(h11, t);
                h21 = H[(m + 1) + m * ldh].x;  /* real by construction */
                s = zf_abs1(h11s) + fabs(h21);
                h11s = zf_scale(1.0 / s, h11s);
                h21 = h21 / s;
                v_vec[0] = h11s;
                v_vec[1] = zf_make(h21, 0);
                double h10 = H[m + (m - 1) * ldh].x;
                if (fabs(h10) * fabs(h21) <= ulp *
                    (zf_abs1(h11s) * (zf_abs1(h11) + zf_abs1(h22))))
                    break;
            }
            if (m == l) {
                /* Didn't break from loop -- set up V for m = l */
                h11 = H[l + l * ldh];
                h22 = H[(l + 1) + (l + 1) * ldh];
                h11s = zf_sub(h11, t);
                h21 = H[(l + 1) + l * ldh].x;
                s = zf_abs1(h11s) + fabs(h21);
                h11s = zf_scale(1.0 / s, h11s);
                h21 = h21 / s;
                v_vec[0] = h11s;
                v_vec[1] = zf_make(h21, 0);
            }

            /* Single-shift QR step */
            for (int kk = m; kk <= i - 1; kk++) {
                if (kk > m) {
                    v_vec[0] = H[kk + (kk - 1) * ldh];
                    v_vec[1] = H[(kk + 1) + (kk - 1) * ldh];
                }
                lapack_double_complex t1_val;
                int two_int = 2, incv = 1;
                zlarfg_(&two_int, &v_vec[0], &v_vec[1], &incv, &t1_val);
                if (kk > m) {
                    H[kk + (kk - 1) * ldh] = v_vec[0];
                    H[(kk + 1) + (kk - 1) * ldh] = czero;
                }
                lapack_double_complex v2 = v_vec[1];
                double t2_val = (zf_mul(t1_val, v2)).x;  /* T2 = DBLE(T1*V2), real */

                /* Apply G from the left to transform rows of H in columns KK to I2. */
                for (int j = kk; j <= i2; j++) {
                    lapack_double_complex sum = zf_add(zf_mul(zf_conj(t1_val), H[kk + j * ldh]),
                                                       zf_scale(t2_val, H[(kk + 1) + j * ldh]));
                    H[kk + j * ldh] = zf_sub(H[kk + j * ldh], sum);
                    H[(kk + 1) + j * ldh] = zf_sub(H[(kk + 1) + j * ldh], zf_mul(sum, v2));
                }

                /* Apply G from the right to transform columns of H in rows I1 to min(KK+2, I). */
                for (int j = i1; j <= std::min(kk + 2, i); j++) {
                    lapack_double_complex sum = zf_add(zf_mul(t1_val, H[j + kk * ldh]),
                                                       zf_scale(t2_val, H[j + (kk + 1) * ldh]));
                    H[j + kk * ldh] = zf_sub(H[j + kk * ldh], sum);
                    H[j + (kk + 1) * ldh] = zf_sub(H[j + (kk + 1) * ldh], zf_mul(sum, zf_conj(v2)));
                }

                if (wantz) {
                    /* Accumulate transformations in the matrix Z */
                    for (int j = iloz; j <= ihiz; j++) {
                        lapack_double_complex sum = zf_add(zf_mul(t1_val, Z[j + kk * ldz]),
                                                           zf_scale(t2_val, Z[j + (kk + 1) * ldz]));
                        Z[j + kk * ldz] = zf_sub(Z[j + kk * ldz], sum);
                        Z[j + (kk + 1) * ldz] = zf_sub(Z[j + (kk + 1) * ldz], zf_mul(sum, zf_conj(v2)));
                    }
                }

                if (kk == m && m > l) {
                    /* Extra scaling to ensure H(M+1, M) remains real. */
                    lapack_double_complex temp_sc = zf_sub(cone, t1_val);
                    temp_sc = zf_scale(1.0 / zf_abs(temp_sc), temp_sc);
                    H[(m + 1) + m * ldh] = zf_mul(H[(m + 1) + m * ldh], zf_conj(temp_sc));
                    if (m + 2 <= i)
                        H[(m + 2) + (m + 1) * ldh] = zf_mul(H[(m + 2) + (m + 1) * ldh], temp_sc);
                    for (int j = m; j <= i; j++) {
                        if (j != m + 1) {
                            if (i2 > j) {
                                int len = i2 - j;
                                zscal_(&len, &temp_sc, &H[j + (j + 1) * ldh], &ldh);
                            }
                            int len = j - i1;
                            lapack_double_complex temp_conj = zf_conj(temp_sc);
                            int one_int = 1;
                            if (len > 0)
                                zscal_(&len, &temp_conj, &H[i1 + j * ldh], &one_int);
                            if (wantz) {
                                zscal_(&nz, &temp_conj, &Z[iloz + j * ldz], &one_int);
                            }
                        }
                    }
                }
            }

            /* Ensure that H(I, I-1) is real. */
            lapack_double_complex temp_h = H[i + (i - 1) * ldh];
            if (temp_h.y != rzero) {
                double rtemp = zf_abs(temp_h);
                H[i + (i - 1) * ldh] = zf_make(rtemp, 0);
                temp_h = zf_scale(1.0 / rtemp, temp_h);
                if (i2 > i) {
                    int len = i2 - i;
                    lapack_double_complex temp_conj = zf_conj(temp_h);
                    zscal_(&len, &temp_conj, &H[i + (i + 1) * ldh], &ldh);
                }
                {
                    int len = i - i1;
                    int one_int = 1;
                    if (len > 0)
                        zscal_(&len, &temp_h, &H[i1 + i * ldh], &one_int);
                }
                if (wantz) {
                    int one_int = 1;
                    zscal_(&nz, &temp_h, &Z[iloz + i * ldz], &one_int);
                }
            }
        }

        if (!converged) {
            /* Failure to converge */
            *info = i + 1;  /* 1-indexed failure point */
            return;
        }

        /* H(I,I-1) is negligible: one eigenvalue has converged. */
        w[i] = H[i + i * ldh];

        /* Return to start of the main loop with new value of I. */
        i = l - 1;
    }
}

void chseqr_(const char *job, const char *compz,
             const int *n, const int *ilo, const int *ihi,
             lapack_float_complex *H, const int *ldh,
             lapack_float_complex *w,
             lapack_float_complex *Z, const int *ldz,
             lapack_float_complex *work, const int *lwork, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDH = *ldh, LDZ = *ldz;

    if (*lwork == -1) { work[0] = cf_make((float)std::max(1, N), 0); return; }
    if (N == 0) return;
    if (N == 1) { w[0] = H[0]; return; }

    bool wantt = lsame(*job, 'S');
    bool wantz = !lsame(*compz, 'N');

    if (lsame(*compz, 'I')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = (i == j) ? cf_make(1, 0) : cf_make(0, 0);
    }

    /* Promote to double for computation */
    std::vector<lapack_double_complex> dH(N * N), dZ(N * N), dw(N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            dH[i + j * N] = zf_make(H[i + j * LDH].x, H[i + j * LDH].y);
    if (wantz) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                dZ[i + j * N] = zf_make(Z[i + j * LDZ].x, Z[i + j * LDZ].y);
    }

    zlahqr_impl(wantt ? 1 : 0, wantz ? 1 : 0, N, ILO - 1, IHI - 1,
                dH.data(), N, dw.data(), 0, N - 1, dZ.data(), N, info);

    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            H[i + j * LDH] = cf_make((float)dH[i + j * N].x, (float)dH[i + j * N].y);
    if (wantz) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = cf_make((float)dZ[i + j * N].x, (float)dZ[i + j * N].y);
    }
    for (int i = 0; i < N; i++)
        w[i] = cf_make((float)dw[i].x, (float)dw[i].y);
}

void zhseqr_(const char *job, const char *compz,
             const int *n, const int *ilo, const int *ihi,
             lapack_double_complex *H, const int *ldh,
             lapack_double_complex *w,
             lapack_double_complex *Z, const int *ldz,
             lapack_double_complex *work, const int *lwork, int *info) {
    *info = 0;
    int N = *n, ILO = *ilo, IHI = *ihi, LDH = *ldh, LDZ = *ldz;

    if (*lwork == -1) { work[0] = zf_make((double)std::max(1, N), 0); return; }
    if (N == 0) return;
    if (N == 1) { w[0] = H[0]; return; }

    bool wantt = lsame(*job, 'S');
    bool wantz = !lsame(*compz, 'N');

    if (lsame(*compz, 'I')) {
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                Z[i + j * LDZ] = (i == j) ? zf_make(1, 0) : zf_make(0, 0);
    }

    zlahqr_impl(wantt ? 1 : 0, wantz ? 1 : 0, N, ILO - 1, IHI - 1,
                H, LDH, w, 0, N - 1, Z, LDZ, info);
}

/* ========================================================================
   LAPACK Auxiliary - DLALN2: solve 1x1 or 2x2 system (ca*A - w*D)*X = s*B
   This is needed by the trevc routines via dlaqtrsd.
   ======================================================================== */

void slaln2_(const int *ltrans, const int *na, const int *nw,
             const float *smin, const float *ca,
             const float *a, const int *lda,
             const float *d1, const float *d2,
             const float *b, const int *ldb,
             const float *wr, const float *wi,
             float *x, const int *ldx,
             float *scale, float *xnorm, int *info) {
    /* Forward to double version for simplicity */
    *info = 0;
    *scale = 1.0f;
    *xnorm = 0.0f;
    int NA = *na, NW = *nw;

    if (NA == 1 && NW == 1) {
        /* 1x1 real system */
        float crv = (*ca) * a[0] - (*wr) * (*d1);
        float smn = *smin;
        if (fabsf(crv) < smn) crv = smn;
        *scale = 1.0f;
        if (fabsf(crv) < 1.0f && fabsf(b[0]) > fabsf(crv) * FLT_MAX) {
            *scale = 1.0f / fabsf(b[0]);
        }
        x[0] = (*scale * b[0]) / crv;
        *xnorm = fabsf(x[0]);
    } else if (NA == 1 && NW == 2) {
        /* 1x1 complex system */
        float crv = (*ca) * a[0] - (*wr) * (*d1);
        float civ = -(*wi) * (*d1);
        float smn = *smin;
        if (fabsf(crv) + fabsf(civ) < smn) { crv = smn; civ = 0.0f; }
        *scale = 1.0f;
        float denom = crv * crv + civ * civ;
        x[0]            = (*scale * b[0] * crv + *scale * b[0 + 1 * (*ldb)] * civ) / denom;
        x[0 + 1 * (*ldx)] = (*scale * b[0 + 1 * (*ldb)] * crv - *scale * b[0] * civ) / denom;
        *xnorm = fabsf(x[0]) + fabsf(x[0 + 1 * (*ldx)]);
    } else if (NA == 2 && NW == 1) {
        /* 2x2 real system */
        float smn = *smin;
        float cr11 = (*ca) * a[0] - (*wr) * (*d1);
        float cr22 = (*ca) * a[1 + 1 * (*lda)] - (*wr) * (*d2);
        float cr12 = (*ltrans) ? (*ca) * a[1] : (*ca) * a[0 + 1 * (*lda)];
        float cr21 = (*ltrans) ? (*ca) * a[0 + 1 * (*lda)] : (*ca) * a[1];

        float det = cr11 * cr22 - cr12 * cr21;
        if (fabsf(det) < smn * smn) det = smn * smn;
        *scale = 1.0f;
        x[0] = (cr22 * b[0] - cr12 * b[1]) / det;
        x[1] = (cr11 * b[1] - cr21 * b[0]) / det;
        *xnorm = fabsf(x[0]) + fabsf(x[1]);
    } else {
        /* 2x2 complex system - simplified */
        *scale = 1.0f;
        float smn = *smin;
        float cr11 = (*ca) * a[0] - (*wr) * (*d1);
        float ci11 = -(*wi) * (*d1);
        float cr22 = (*ca) * a[1 + 1 * (*lda)] - (*wr) * (*d2);
        float ci22 = -(*wi) * (*d2);
        float cr12, cr21;
        if (*ltrans) { cr12 = (*ca) * a[1]; cr21 = (*ca) * a[0 + 1 * (*lda)]; }
        else { cr12 = (*ca) * a[0 + 1 * (*lda)]; cr21 = (*ca) * a[1]; }

        /* Solve using Cramer's rule with 2x2 complex LU */
        /* For simplicity, solve via direct formula */
        float d11r = cr11, d11i = ci11;
        float d22r = cr22, d22i = ci22;
        /* det = d11 * d22 - cr12 * cr21 (cr12, cr21 are real) */
        float detr = d11r * d22r - d11i * d22i - cr12 * cr21;
        float deti = d11r * d22i + d11i * d22r;
        float detsq = detr * detr + deti * deti;
        if (detsq < smn * smn) detsq = smn * smn;

        /* RHS: b is 2 x 2, columns are real and imaginary parts */
        float b1r = b[0], b1i = b[0 + 1 * (*ldb)];
        float b2r = b[1], b2i = b[1 + 1 * (*ldb)];

        /* x1 = (d22 * b1 - cr12 * b2) / det */
        float t1r = d22r * b1r - d22i * b1i - cr12 * b2r;
        float t1i = d22r * b1i + d22i * b1r - cr12 * b2i;
        /* x2 = (d11 * b2 - cr21 * b1) / det */
        float t2r = d11r * b2r - d11i * b2i - cr21 * b1r;
        float t2i = d11r * b2i + d11i * b2r - cr21 * b1i;

        x[0]                = (t1r * detr + t1i * deti) / detsq;
        x[0 + 1 * (*ldx)]   = (t1i * detr - t1r * deti) / detsq;
        x[1]                = (t2r * detr + t2i * deti) / detsq;
        x[1 + 1 * (*ldx)]   = (t2i * detr - t2r * deti) / detsq;
        *xnorm = fabsf(x[0]) + fabsf(x[0 + 1 * (*ldx)]) + fabsf(x[1]) + fabsf(x[1 + 1 * (*ldx)]);
    }
}

void dlaln2_(const int *ltrans, const int *na, const int *nw,
             const double *smin, const double *ca,
             const double *a, const int *lda,
             const double *d1, const double *d2,
             const double *b, const int *ldb,
             const double *wr, const double *wi,
             double *x, const int *ldx,
             double *scale, double *xnorm, int *info) {
    *info = 0;
    *scale = 1.0;
    *xnorm = 0.0;
    int NA = *na, NW = *nw;

    if (NA == 1 && NW == 1) {
        double crv = (*ca) * a[0] - (*wr) * (*d1);
        double smn = *smin;
        if (fabs(crv) < smn) crv = copysign(smn, crv);
        *scale = 1.0;
        if (fabs(crv) < 1.0 && fabs(b[0]) > fabs(crv) * DBL_MAX) {
            *scale = 1.0 / fabs(b[0]);
        }
        x[0] = (*scale * b[0]) / crv;
        *xnorm = fabs(x[0]);
    } else if (NA == 1 && NW == 2) {
        double crv = (*ca) * a[0] - (*wr) * (*d1);
        double civ = -(*wi) * (*d1);
        double smn = *smin;
        if (fabs(crv) + fabs(civ) < smn) { crv = smn; civ = 0.0; }
        *scale = 1.0;
        double denom = crv * crv + civ * civ;
        x[0]              = (*scale * b[0] * crv + *scale * b[0 + 1 * (*ldb)] * civ) / denom;
        x[0 + 1 * (*ldx)] = (*scale * b[0 + 1 * (*ldb)] * crv - *scale * b[0] * civ) / denom;
        *xnorm = fabs(x[0]) + fabs(x[0 + 1 * (*ldx)]);
    } else if (NA == 2 && NW == 1) {
        double smn = *smin;
        double cr11 = (*ca) * a[0] - (*wr) * (*d1);
        double cr22 = (*ca) * a[1 + 1 * (*lda)] - (*wr) * (*d2);
        double cr12 = (*ltrans) ? (*ca) * a[1] : (*ca) * a[0 + 1 * (*lda)];
        double cr21 = (*ltrans) ? (*ca) * a[0 + 1 * (*lda)] : (*ca) * a[1];

        double det = cr11 * cr22 - cr12 * cr21;
        if (fabs(det) < smn * smn) det = copysign(smn * smn, det);
        *scale = 1.0;
        x[0] = (cr22 * b[0] - cr12 * b[1]) / det;
        x[1] = (cr11 * b[1] - cr21 * b[0]) / det;
        *xnorm = fabs(x[0]) + fabs(x[1]);
    } else {
        /* 2x2 complex system */
        *scale = 1.0;
        double smn = *smin;
        double cr11 = (*ca) * a[0] - (*wr) * (*d1);
        double ci11 = -(*wi) * (*d1);
        double cr22 = (*ca) * a[1 + 1 * (*lda)] - (*wr) * (*d2);
        double ci22 = -(*wi) * (*d2);
        double cr12, cr21;
        if (*ltrans) { cr12 = (*ca) * a[1]; cr21 = (*ca) * a[0 + 1 * (*lda)]; }
        else { cr12 = (*ca) * a[0 + 1 * (*lda)]; cr21 = (*ca) * a[1]; }

        double d11r = cr11, d11i = ci11;
        double d22r = cr22, d22i = ci22;
        double detr = d11r * d22r - d11i * d22i - cr12 * cr21;
        double deti = d11r * d22i + d11i * d22r;
        double detsq = detr * detr + deti * deti;
        if (detsq < smn * smn) detsq = smn * smn;

        double b1r = b[0], b1i = b[0 + 1 * (*ldb)];
        double b2r = b[1], b2i = b[1 + 1 * (*ldb)];

        double t1r = d22r * b1r - d22i * b1i - cr12 * b2r;
        double t1i = d22r * b1i + d22i * b1r - cr12 * b2i;
        double t2r = d11r * b2r - d11i * b2i - cr21 * b1r;
        double t2i = d11r * b2i + d11i * b2r - cr21 * b1i;

        x[0]                = (t1r * detr + t1i * deti) / detsq;
        x[0 + 1 * (*ldx)]   = (t1i * detr - t1r * deti) / detsq;
        x[1]                = (t2r * detr + t2i * deti) / detsq;
        x[1 + 1 * (*ldx)]   = (t2i * detr - t2r * deti) / detsq;
        *xnorm = fabs(x[0]) + fabs(x[0 + 1 * (*ldx)]) + fabs(x[1]) + fabs(x[1 + 1 * (*ldx)]);
    }
}

/* lsame - character comparison (Fortran convention) */
long lsame_(const char *ca, const char *cb) {
    return lsame(*ca, *cb) ? 1L : 0L;
}

extern "C"
void dorgqr_(const int *m, const int *n, const int *k,
             double *A, const int *lda, const double *tau,
             double *work, const int *lwork, int *info) {
    *info = 0;
    int M = *m, N = *n, K = *k, LDA = *lda;
    if (*lwork == -1) { work[0] = (double)std::max(1, N); return; }
    if (N <= 0) return;

    // Initialize columns k+1:n to identity columns
    for (int j = K; j < N; j++) {
        for (int i = 0; i < M; i++) A[i + j * LDA] = 0.0;
        if (j < M) A[j + j * LDA] = 1.0;
    }

    // Apply reflectors K-1, K-2, ..., 0 (backward)
    for (int i = K - 1; i >= 0; i--) {
        // Apply H(i) to A(i:m-1, i:n-1) from left
        if (i < N - 1) {
            A[i + i * LDA] = 1.0;
            int m_sub = M - i;
            int n_sub = N - i - 1;
            int one = 1;
            dlarf_("L", &m_sub, &n_sub, &A[i + i * LDA], &one, &tau[i],
                   &A[i + (i + 1) * LDA], &LDA, work);
        }
        // Scale column i below diagonal by -tau
        if (i < M - 1) {
            double neg_tau = -tau[i];
            int len = M - i - 1;
            int one = 1;
            dscal_(&len, &neg_tau, &A[(i + 1) + i * LDA], &one);
        }
        A[i + i * LDA] = 1.0 - tau[i];
        // Zero above
        for (int j = 0; j < i; j++)
            A[j + i * LDA] = 0.0;
    }
}


/* Forward declarations for orgqr/ungqr (defined later in file) */
extern "C" void sorgqr_(const int*, const int*, const int*, float*, const int*, const float*, float*, const int*, int*);
extern "C" void cungqr_(const int*, const int*, const int*, lapack_float_complex*, const int*, const lapack_float_complex*, lapack_float_complex*, const int*, int*);
extern "C" void zungqr_(const int*, const int*, const int*, lapack_double_complex*, const int*, const lapack_double_complex*, lapack_double_complex*, const int*, int*);

extern "C"
void sorgqr_(const int *m, const int *n, const int *k,
             float *A, const int *lda, const float *tau,
             float *work, const int *lwork, int *info) {
    *info = 0;
    int M = *m, N = *n, K = *k, LDA = *lda;
    if (*lwork == -1) { work[0] = (float)std::max(1, N); return; }
    if (N <= 0) return;
    for (int j = K; j < N; j++) {
        for (int i = 0; i < M; i++) A[i + j * LDA] = 0.0f;
        if (j < M) A[j + j * LDA] = 1.0f;
    }
    for (int i = K - 1; i >= 0; i--) {
        if (i < N - 1) {
            A[i + i * LDA] = 1.0f;
            int m_sub = M - i, n_sub = N - i - 1, one = 1;
            slarf_("L", &m_sub, &n_sub, &A[i + i * LDA], &one, &tau[i],
                   &A[i + (i + 1) * LDA], &LDA, work);
        }
        if (i < M - 1) {
            float neg_tau = -tau[i];
            int len = M - i - 1, one = 1;
            sscal_(&len, &neg_tau, &A[(i + 1) + i * LDA], &one);
        }
        A[i + i * LDA] = 1.0f - tau[i];
        for (int j = 0; j < i; j++) A[j + i * LDA] = 0.0f;
    }
}

/* CUNGQR - unblocked complex float */
extern "C"
void cungqr_(const int *m, const int *n, const int *k,
             lapack_float_complex *A, const int *lda, const lapack_float_complex *tau,
             lapack_float_complex *work, const int *lwork, int *info) {
    *info = 0;
    int M = *m, N = *n, K = *k, LDA = *lda;
    lapack_float_complex zero = {0,0}, one_c = {1,0};
    if (*lwork == -1) { work[0] = cf_make((float)std::max(1, N), 0); return; }
    if (N <= 0) return;
    for (int j = K; j < N; j++) {
        for (int i = 0; i < M; i++) A[i + j * LDA] = zero;
        if (j < M) A[j + j * LDA] = one_c;
    }
    for (int i = K - 1; i >= 0; i--) {
        if (i < N - 1) {
            A[i + i * LDA] = one_c;
            int m_sub = M - i, n_sub = N - i - 1, one = 1;
            clarf_("L", &m_sub, &n_sub, &A[i + i * LDA], &one, &tau[i],
                   &A[i + (i + 1) * LDA], &LDA, work);
        }
        if (i < M - 1) {
            lapack_float_complex neg_tau = {-tau[i].x, -tau[i].y};
            int len = M - i - 1, one = 1;
            cscal_(&len, &neg_tau, &A[(i + 1) + i * LDA], &one);
        }
        A[i + i * LDA] = cf_make(1.0f - tau[i].x, -tau[i].y);
        for (int j = 0; j < i; j++) A[j + i * LDA] = zero;
    }
}

/* ZUNGQR - unblocked complex double */
extern "C"
void zungqr_(const int *m, const int *n, const int *k,
             lapack_double_complex *A, const int *lda, const lapack_double_complex *tau,
             lapack_double_complex *work, const int *lwork, int *info) {
    *info = 0;
    int M = *m, N = *n, K = *k, LDA = *lda;
    lapack_double_complex zero = {0,0}, one_c = {1,0};
    if (*lwork == -1) { work[0] = zf_make((double)std::max(1, N), 0); return; }
    if (N <= 0) return;
    for (int j = K; j < N; j++) {
        for (int i = 0; i < M; i++) A[i + j * LDA] = zero;
        if (j < M) A[j + j * LDA] = one_c;
    }
    for (int i = K - 1; i >= 0; i--) {
        if (i < N - 1) {
            A[i + i * LDA] = one_c;
            int m_sub = M - i, n_sub = N - i - 1, one = 1;
            zlarf_("L", &m_sub, &n_sub, &A[i + i * LDA], &one, &tau[i],
                   &A[i + (i + 1) * LDA], &LDA, work);
        }
        if (i < M - 1) {
            lapack_double_complex neg_tau = {-tau[i].x, -tau[i].y};
            int len = M - i - 1, one = 1;
            zscal_(&len, &neg_tau, &A[(i + 1) + i * LDA], &one);
        }
        A[i + i * LDA] = zf_make(1.0 - tau[i].x, -tau[i].y);
        for (int j = 0; j < i; j++) A[j + i * LDA] = zero;
    }
}

} // extern "C"
