/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "auxiliary/rocauxiliary_latrd.hpp"
#include "rocblas.hpp"
#include "roclapack_sytd2_hetd2.hpp"
#include "rocsolver/rocsolver.h"

#include <hip/hip_runtime.h>
#include <map>
#include <tuple>

ROCSOLVER_BEGIN_NAMESPACE

// SYTRD-level graph caching (compile-time flag)
// SYTRD_GRAPH_MODE controls whether to cache entire SYTRD calls:
//   0 = disabled (default, use LATRD-level caching if enabled)
//   1 = enabled (cache entire SYTRD call as single graph)
#ifndef SYTRD_GRAPH_MODE
    #define SYTRD_GRAPH_MODE 0
#endif

#if SYTRD_GRAPH_MODE != 0

// Cache key for SYTRD-level graphs
struct SytrdGraphCacheKey {
    rocblas_fill uplo;
    rocblas_int n;
    rocblas_int batch_count;
    
    bool operator<(const SytrdGraphCacheKey& other) const {
        return std::tie(uplo, n, batch_count) < 
               std::tie(other.uplo, other.n, other.batch_count);
    }
    
    bool operator==(const SytrdGraphCacheKey& other) const {
        return uplo == other.uplo && n == other.n && batch_count == other.batch_count;
    }
};

// Cache entry: the executable graph ready for replay.
struct SytrdGraphCacheEntry {
    hipGraphExec_t graph_exec = nullptr;
};

// Global cache (thread-safe via static initialization)
static std::map<SytrdGraphCacheKey, SytrdGraphCacheEntry>& get_sytrd_graph_cache() {
    static std::map<SytrdGraphCacheKey, SytrdGraphCacheEntry> cache;
    return cache;
}

// Cache statistics — only compiled when ROCSOLVER_SYTRD_GRAPH_DEBUG is defined.
// When disabled, no counter struct, no static, and no increments exist in the binary.
#ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
struct SytrdCacheStats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
};

static SytrdCacheStats& get_sytrd_cache_stats() {
    static SytrdCacheStats stats;
    return stats;
}
#endif // ROCSOLVER_SYTRD_GRAPH_DEBUG

#endif // SYTRD_GRAPH_MODE != 0

template <bool BATCHED, typename T>
void rocsolver_sytrd_hetrd_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau_W,
                                         size_t* size_workArr,
                                         bool recover_A = true)
{
    *size_scalars = 0;
    *size_work = 0;
    *size_norms = 0;
    *size_tmptau_W = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
        return;

    size_t s1 = 0, s2 = 0;
    size_t w1 = 0, w2 = 0;
    size_t n1 = 0, n2 = 0;
    size_t na;

    // extra requirements to call SYTD2/HETD2
    rocsolver_sytd2_hetd2_getMemorySize<BATCHED, T>(n, batch_count, size_scalars, &w2, &n2, &s2,
                                                    size_workArr);

    if(n > xxTRD_xxTD2_SWITCHSIZE)
    {
        // size required to store temporary matrix W
        s1 = n * xxTRD_BLOCKSIZE;
        s1 *= sizeof(T) * batch_count;

        // extra requirements to call latrd_forsytrd
        rocsolver_latrd_forsytrd_getMemorySize<BATCHED, T>(n, xxTRD_BLOCKSIZE, batch_count,
                                                           size_scalars, &w1, &n1, size_workArr);
    }

    *size_tmptau_W = std::max(s1, s2);
    *size_work = std::max(w1, w2);
    *size_norms = std::max(n1, n2);

    // when recovering the non-referenced part of A is necessary,
    // add the required buffer size to hold a copy
    // TODO: Actually we only need to hold half of the elements (upper or lower part)
    // The methods to do the copies can be modified for this new data layout in the future
    if(recover_A)
        *size_work += sizeof(T) * n * n * batch_count;
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytrd_hetrd_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tau,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !D) || (n > 1 && !E) || (n > 1 && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// ---------------------------------------------------------------------------
// Capture-status probe helper (active only when SYTRD_CAPTURE_DEBUG is defined)
// ---------------------------------------------------------------------------
#ifdef SYTRD_CAPTURE_DEBUG
static const char* sytrd_capture_status_str(hipStreamCaptureStatus s)
{
    switch(s)
    {
    case hipStreamCaptureStatusNone:        return "None";
    case hipStreamCaptureStatusActive:      return "Active";
    case hipStreamCaptureStatusInvalidated: return "Invalidated";
    default:                                return "Unknown";
    }
}

// Probe the capture status of 'stream'.  If it has become Invalidated (or any
// non-Active state during an expected capture), print the stage label and
// return rocblas_status_internal_error so the caller can abort immediately.
// Pass j<0 to suppress the iteration field.
static rocblas_status sytrd_check_capture(hipStream_t stream,
                                          const char* stage,
                                          int         j = -1)
{
    hipStreamCaptureStatus cap_status;
    unsigned long long     cap_id = 0;
    hipError_t hip_err = hipStreamGetCaptureInfo(stream, &cap_status, &cap_id);
    if(hip_err != hipSuccess)
    {
        if(j >= 0)
            fprintf(stderr,
                    "[SYTRD_CAPTURE_DEBUG] hipStreamGetCaptureInfo failed (%d) at stage='%s' j=%d\n",
                    (int)hip_err, stage, j);
        else
            fprintf(stderr,
                    "[SYTRD_CAPTURE_DEBUG] hipStreamGetCaptureInfo failed (%d) at stage='%s'\n",
                    (int)hip_err, stage);
        return rocblas_status_internal_error;
    }
    if(j >= 0)
        fprintf(stderr,
                "[SYTRD_CAPTURE_DEBUG] stage='%s' j=%d  cap_status=%s(%d)  cap_id=%llu\n",
                stage, j, sytrd_capture_status_str(cap_status), (int)cap_status,
                (unsigned long long)cap_id);
    else
        fprintf(stderr,
                "[SYTRD_CAPTURE_DEBUG] stage='%s'  cap_status=%s(%d)  cap_id=%llu\n",
                stage, sytrd_capture_status_str(cap_status), (int)cap_status,
                (unsigned long long)cap_id);

    if(cap_status == hipStreamCaptureStatusInvalidated)
        return rocblas_status_internal_error;
    return rocblas_status_success;
}
// Convenience macro: calls the probe and returns from the enclosing function on failure.
#define SYTRD_PROBE(stream_, stage_, j_)                                    \
    do {                                                                    \
        rocblas_status _probe_st = sytrd_check_capture(stream_, stage_, j_); \
        if(_probe_st != rocblas_status_success) return _probe_st;           \
    } while(0)
#else
#define SYTRD_PROBE(stream_, stage_, j_) (void)0
#endif // SYTRD_CAPTURE_DEBUG
// ---------------------------------------------------------------------------

template <bool BATCHED, typename T, typename S, typename U>
rocblas_status rocsolver_sytrd_hetrd_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              T* tau,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work_Acpy,
                                              T* norms,
                                              T* tmptau_W,
                                              T** workArr,
                                              bool recover_A = true)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Time the non-graph (direct) execution path when SYTRD_GRAPH_TIMING is enabled.
    // Gated on SYTRD_GRAPH_MODE != 1 so it never fires inside a graph capture.
#if SYTRD_GRAPH_MODE != 1
#ifdef SYTRD_GRAPH_TIMING
    hipEvent_t launch_start, launch_end;
    hipEventCreate(&launch_start);
    hipEventCreate(&launch_end);
    hipEventRecord(launch_start, stream);
#endif
#endif

    ROCSOLVER_ENTER("sytrd_hetrd", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    
    rocblas_int k = xxTRD_BLOCKSIZE;
    rocblas_int kk = xxTRD_xxTD2_SWITCHSIZE;

    // if the matrix is too small, use the unblocked variant of the algorithm
    if(n <= kk)
        return rocsolver_sytd2_hetd2_template(handle, uplo, n, A, shiftA, lda, strideA, D, strideD,
                                              E, strideE, tau, strideP, batch_count, scalars,
                                              work_Acpy, norms, tmptau_W, workArr);

    // Use host-side alpha/beta so that rocblas_copy_alpha_beta_to_host_if_on_device
    // (which does hipMemcpy + hipStreamSynchronize when pointer_mode==device) is
    // never triggered.  This keeps every operation on the stream fully capturable
    // inside a HIP graph without requiring USE_INTERNAL_GEMM.
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    SYTRD_PROBE(stream, "set_pointer_mode_host", -1);

    // Host-resident scalars: alpha = -1, beta = 1.
    // These match scalars[0] and scalars[2] as initialized by init_scalars()
    // via iota_n(scalars, 3, -1) → {-1, 0, 1}.  Using stack variables with
    // pointer_mode_host avoids the hipMemcpy + hipStreamSynchronize that
    // rocblas_copy_alpha_beta_to_host_if_on_device would inject when
    // pointer_mode == device, keeping the stream fully capturable.
    const T minone_h = T(-1);
    const T one_h    = T(1);
    const T* minone  = &minone_h;
    const T* one     = &one_h;

    rocblas_int ldw = n;
    rocblas_stride strideW = n * k;
    rocblas_int j;

    // make A a general matrix by copying its upper (lower) part
    // also copy the non referenced part of A to recover it if necessary
    T* work;
    T* Acpy;
    rocblas_int blocks = (n - 1) / BS2 + 1;
    rocblas_fill uplo2 = (uplo == rocblas_fill_upper) ? rocblas_fill_lower : rocblas_fill_upper;

    if(recover_A)
    {
        Acpy = work_Acpy;
        work = Acpy + n * n * batch_count;

        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                                0, stream, copymat_to_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                                no_mask{}, uplo2, rocblas_diagonal_unit);
        SYTRD_PROBE(stream, "copy_mat_to_buffer", -1);
    }
    else
        work = work_Acpy;

    ROCSOLVER_LAUNCH_KERNEL((copy_trans_mat<T, T>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2, 1), 0, stream, rocblas_operation_conjugate_transpose, n,
                            n, A, shiftA, lda, strideA, A, shiftA, lda, strideA, no_mask{}, uplo,
                            rocblas_diagonal_unit);
    SYTRD_PROBE(stream, "copy_trans_mat", -1);

    if(uplo == rocblas_fill_lower)
    {
        // reduce the lower part of A
        // main loop running forwards (for each block of columns)
        // when the unreduced part is not large enough, switch to unblocked algorithm
        j = 0;
        while(j < n - kk)
        {
            // reduce columns j:j+k-1
            rocsolver_latrd_forsytrd_template<T>(handle, uplo, n - j, k, A,
                                                 shiftA + idx2D(j, j, lda), lda, strideA, (E + j),
                                                 strideE, (tau + j), strideP, tmptau_W, 0, ldw,
                                                 strideW, batch_count, scalars, work, norms, workArr);
            SYTRD_PROBE(stream, "lower_latrd", j);

            // update trailing matrix
            // A = A - V*W' - W*V'
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - j - k, n - j - k, k, minone, A, shiftA + idx2D(j + k, j, lda), lda,
                           strideA, tmptau_W, idx2D(k, 0, ldw), ldw, strideW, one, A,
                           shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);
            SYTRD_PROBE(stream, "lower_gemm1", j);

            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose,
                           n - j - k, n - j - k, k, minone, tmptau_W, idx2D(k, 0, ldw), ldw,
                           strideW, A, shiftA + idx2D(j + k, j, lda), lda, strideA, one, A,
                           shiftA + idx2D(j + k, j + k, lda), lda, strideA, batch_count, workArr);
            SYTRD_PROBE(stream, "lower_gemm2", j);

            j += k;
        }

        // reduce last columns of A
        rocsolver_sytd2_hetd2_template<T>(handle, uplo, n - j, A, shiftA + idx2D(j, j, lda), lda,
                                          strideA, (D + j), strideD, (E + j), strideE, (tau + j),
                                          strideP, batch_count, scalars, work, norms, tmptau_W,
                                          workArr);
        SYTRD_PROBE(stream, "lower_sytd2_tail", j);
    }

    else
    {
        // reduce the upper part of A
        // main loop running backwards (for each block of columns)
        // when the unreduced part is not large enough, switch to unblocked algorithm
        j = n - k;
        rocblas_int upkk = n - ((n - kk + k - 1) / k) * k;
        while(j >= upkk)
        {
            // reduce columns j:j+k-1
            rocsolver_latrd_forsytrd_template<T>(handle, uplo, j + k, k, A, shiftA, lda, strideA, E,
                                                 strideE, tau, strideP, tmptau_W, 0, ldw, strideW,
                                                 batch_count, scalars, work, norms, workArr);
            SYTRD_PROBE(stream, "upper_latrd", j);

            // update trailing matrix
            // A = A - V*W' - W*V'
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                           j, k, minone, A, shiftA + idx2D(0, j, lda), lda, strideA, tmptau_W, 0,
                           ldw, strideW, one, A, shiftA, lda, strideA, batch_count, workArr);
            SYTRD_PROBE(stream, "upper_gemm1", j);

            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, j,
                           j, k, minone, tmptau_W, 0, ldw, strideW, A, shiftA + idx2D(0, j, lda),
                           lda, strideA, one, A, shiftA, lda, strideA, batch_count, workArr);
            SYTRD_PROBE(stream, "upper_gemm2", j);

            j -= k;
        }

        // reduce first columns of A
        rocsolver_sytd2_hetd2_template<T>(handle, uplo, upkk, A, shiftA, lda, strideA, D, strideD,
                                          E, strideE, tau, strideP, batch_count, scalars, work,
                                          norms, tmptau_W, workArr);
        SYTRD_PROBE(stream, "upper_sytd2_tail", j);
    }

    // Copy results (set tridiagonal form in A)
    blocks = (n - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, dim3(blocks, batch_count), dim3(BS1), 0, stream, uplo,
                            n, A, shiftA, lda, strideA, D, strideD, E, strideE);
    SYTRD_PROBE(stream, "set_tridiag", -1);

    // recover non-referenced part of A if necessary
    if(recover_A)
    {
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocks, blocks, batch_count), dim3(BS2, BS2, 1),
                                0, stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, Acpy,
                                no_mask{}, uplo2, rocblas_diagonal_unit);
        SYTRD_PROBE(stream, "copy_mat_from_buffer", -1);
    }

    rocblas_set_pointer_mode(handle, old_mode);

#if SYTRD_GRAPH_MODE != 1
#ifdef SYTRD_GRAPH_TIMING
    hipEventRecord(launch_end, stream);
    hipEventSynchronize(launch_end);
    float direct_run_us;
    hipEventElapsedTime(&direct_run_us, launch_start, launch_end);
    direct_run_us *= 1000.0f;
    fprintf(stderr, "[SYTRD_GRAPH DISABLED] default run time: %10.2f us\n", direct_run_us);
    hipEventDestroy(launch_start);
    hipEventDestroy(launch_end);
#endif
#endif
    return rocblas_status_success;  
}

#if SYTRD_GRAPH_MODE != 0

// SYTRD-level cached wrapper. Must appear after rocsolver_sytrd_hetrd_template
// so it can call it directly without a forward declaration.
template <bool BATCHED, typename T, typename S, typename U>
rocblas_status rocsolver_sytrd_hetrd_template_cached(rocblas_handle handle,
                                                     const rocblas_fill uplo,
                                                     const rocblas_int n,
                                                     U A,
                                                     const rocblas_int shiftA,
                                                     const rocblas_int lda,
                                                     const rocblas_stride strideA,
                                                     S* D,
                                                     const rocblas_stride strideD,
                                                     S* E,
                                                     const rocblas_stride strideE,
                                                     T* tau,
                                                     const rocblas_stride strideP,
                                                     const rocblas_int batch_count,
                                                     T* scalars,
                                                     T* work_Acpy,
                                                     T* norms,
                                                     T* tmptau_W,
                                                     T** workArr,
                                                     bool recover_A = true)
{
    SytrdGraphCacheKey cache_key{uplo, n, batch_count};
    auto& cache = get_sytrd_graph_cache();
#ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
    auto& stats = get_sytrd_cache_stats();
    stats.lookups++;
#endif

    auto it = cache.find(cache_key);

    if(it != cache.end())
    {
        // Cache HIT - reuse existing graph
#ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
        stats.hits++;
#endif

        hipGraphExec_t graph_exec = it->second.graph_exec;
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        #ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
        fprintf(stderr, "[SYTRD CACHE HIT] uplo=%d, n=%d, batch=%d "
                "(hits=%lu, misses=%lu, hit_rate=%.1f%%)\n",
                (int)uplo, (int)n, (int)batch_count,
                stats.hits, stats.misses, 100.0 * stats.hits / stats.lookups);
        #endif

        // HOT PATH
#ifdef SYTRD_GRAPH_TIMING
        hipEvent_t launch_start, launch_end;
        hipEventCreate(&launch_start);
        hipEventCreate(&launch_end);
        hipEventRecord(launch_start, stream);
#endif

        hipError_t hip_status = hipGraphLaunch(graph_exec, stream);

#ifdef SYTRD_GRAPH_TIMING
        hipEventRecord(launch_end, stream);
        hipEventSynchronize(launch_end);
        float launch_hip_us;
        hipEventElapsedTime(&launch_hip_us, launch_start, launch_end);
        launch_hip_us *= 1000.0f;
        fprintf(stderr, "[SYTRD CACHE HIT] HIP graph launch time: %10.2f us\n", launch_hip_us);
        hipEventDestroy(launch_start);
        hipEventDestroy(launch_end);
#endif

        if(hip_status != hipSuccess)
        {
            fprintf(stderr, "[SYTRD CACHE ERROR] hipGraphLaunch failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }
        return rocblas_status_success;
    }
    else
    {
        // Cache MISS - capture new graph
#ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
        stats.misses++;
#endif

        #ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
        fprintf(stderr, "[SYTRD CACHE MISS] uplo=%d, n=%d, batch=%d - "
                "Capturing new graph (entry #%zu)\n",
                (int)uplo, (int)n, (int)batch_count, cache.size() + 1);
        #endif

        hipGraph_t     graph;
        hipGraphExec_t graph_exec;
        hipStream_t    original_stream;
        rocblas_get_stream(handle, &original_stream);

        // Time the full capture+instantiate sequence on original_stream.
        // Both events live on original_stream (which is never destroyed during
        // this scope), so hipEventElapsedTime is always valid.
#ifdef SYTRD_GRAPH_TIMING
        hipEvent_t cap_start, cap_end;
        hipEventCreate(&cap_start);
        hipEventCreate(&cap_end);
        hipEventRecord(cap_start, original_stream);
        // Synchronize so cap_start is committed before we touch capture_stream.
        hipEventSynchronize(cap_start);
#endif

        // === CAPTURE PHASE ===
        // Use hipStreamCaptureModeThreadLocal so that rocBLAS internal side-streams
        // are not monitored, preventing false invalidations.
        hipStream_t capture_stream;
        hipError_t hip_status = hipStreamCreate(&capture_stream);
        if(hip_status != hipSuccess)
        {
            fprintf(stderr, "[SYTRD CACHE ERROR] hipStreamCreate failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }

        // Ensure the stream is idle before beginning capture.
        hipStreamSynchronize(capture_stream);

        // Point the rocBLAS handle at the capture stream so all rocBLAS kernel
        // launches go to capture_stream and are recorded into the graph.
        rocblas_set_stream(handle, capture_stream);

        hip_status = hipStreamBeginCapture(capture_stream, hipStreamCaptureModeThreadLocal);
        if(hip_status != hipSuccess)
        {
            rocblas_set_stream(handle, original_stream);
            hipStreamDestroy(capture_stream);
            fprintf(stderr, "[SYTRD CACHE ERROR] hipStreamBeginCapture failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }

        rocblas_status status = rocsolver_sytrd_hetrd_template<BATCHED, T>(
            handle, uplo, n, A, shiftA, lda, strideA,
            D, strideD, E, strideE, tau, strideP,
            batch_count, scalars, work_Acpy, norms, tmptau_W, workArr, recover_A);

        if(status != rocblas_status_success)
        {
            (void)hipStreamEndCapture(capture_stream, &graph);
            rocblas_set_stream(handle, original_stream);
            hipStreamDestroy(capture_stream);
            fprintf(stderr, "[SYTRD CACHE ERROR] Template call failed during capture: %d\n", status);
            return status;
        }

        hip_status = hipStreamEndCapture(capture_stream, &graph);
        rocblas_set_stream(handle, original_stream);
        hipStreamDestroy(capture_stream);

        if(hip_status != hipSuccess)
        {
            fprintf(stderr, "[SYTRD CACHE ERROR] hipStreamEndCapture failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }

        hip_status = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
        hipGraphDestroy(graph); // graph_exec is all we need for replay
        if(hip_status != hipSuccess)
        {
            fprintf(stderr, "[SYTRD CACHE ERROR] hipGraphInstantiate failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }

#ifdef SYTRD_GRAPH_TIMING
        hipEventRecord(cap_end, original_stream);
        hipEventSynchronize(cap_end);
        float cap_us;
        hipEventElapsedTime(&cap_us, cap_start, cap_end);
        cap_us *= 1000.0f;
        fprintf(stderr, "[SYTRD CACHE MISS] Graph capture+instantiate time: %10.2f us\n", cap_us);
        hipEventDestroy(cap_start);
        hipEventDestroy(cap_end);
#endif

        // Store entry.
        SytrdGraphCacheEntry entry;
        entry.graph_exec = graph_exec;
        cache[cache_key] = entry;

        #ifdef ROCSOLVER_SYTRD_GRAPH_DEBUG
        fprintf(stderr, "[SYTRD CACHE] Stored graph for (uplo=%d, n=%d, batch=%d). "
                "Cache size: %zu\n",
                (int)uplo, (int)n, (int)batch_count, cache.size());
        #endif

        // First launch of the freshly captured graph.
#ifdef SYTRD_GRAPH_TIMING
        hipEvent_t launch_start, launch_end;
        hipEventCreate(&launch_start);
        hipEventCreate(&launch_end);
        hipEventRecord(launch_start, original_stream);
#endif

        hip_status = hipGraphLaunch(graph_exec, original_stream);

#ifdef SYTRD_GRAPH_TIMING
        hipEventRecord(launch_end, original_stream);
        hipEventSynchronize(launch_end);
        float first_launch_hip_us;
        hipEventElapsedTime(&first_launch_hip_us, launch_start, launch_end);
        first_launch_hip_us *= 1000.0f;
        fprintf(stderr, "[SYTRD CACHE MISS] HIP Initial graph launch time: %10.2f us\n", first_launch_hip_us);
        hipEventDestroy(launch_start);
        hipEventDestroy(launch_end);
#endif

        if(hip_status != hipSuccess)
        {
            fprintf(stderr, "[SYTRD CACHE ERROR] Initial hipGraphLaunch failed: %d\n", hip_status);
            return rocblas_status_internal_error;
        }
        return rocblas_status_success;
    }
}

#endif // SYTRD_GRAPH_MODE != 0

ROCSOLVER_END_NAMESPACE
