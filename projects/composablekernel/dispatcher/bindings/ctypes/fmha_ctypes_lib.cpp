// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FMHA Dispatcher ctypes library.
// Provides a C API for Python ctypes integration.
// Kernel header included via -include at compile time.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

#include "ck_tile/dispatcher.hpp"

#ifndef GFX_ARCH
#define GFX_ARCH "gfx950"
#endif

using namespace ck_tile::dispatcher;

static std::unique_ptr<FmhaRegistry> g_registry;
static std::unique_ptr<FmhaDispatcher> g_dispatcher;
static bool g_initialized = false;

// Safe HIP check that sets rc and jumps to cleanup on failure.
// All functions using this must have:  int rc = 0;  and a  cleanup:  label.
#define HIP_CHECK(call)           \
    do                            \
    {                             \
        hipError_t err_ = (call); \
        if(err_ != hipSuccess)    \
        {                         \
            rc = -1;              \
            goto cleanup;         \
        }                         \
    } while(0)

// Helper to free a device pointer if non-null
static inline void safe_hip_free(void*& ptr)
{
    if(ptr)
    {
        hipFree(ptr);
        ptr = nullptr;
    }
}

extern "C" {

int fmha_dispatcher_initialize(const char* arch)
{
    if(g_initialized)
        return 0;

    const std::string gfx_arch = arch ? arch : GFX_ARCH;

    g_registry = std::make_unique<FmhaRegistry>();
    g_registry->set_name("fmha_ctypes");
    REGISTER_GENERATED_KERNELS(*g_registry, gfx_arch);

    if(g_registry->size() == 0)
        return -1;

    g_dispatcher = std::make_unique<FmhaDispatcher>(g_registry.get());
    g_dispatcher->set_benchmarking(true);
    g_dispatcher->set_timing(1, 3);
    g_initialized = true;
    return 0;
}

int fmha_dispatcher_run_fwd(const void* q_host,
                            const void* k_host,
                            const void* v_host,
                            void* o_host,
                            int batch,
                            int nhead_q,
                            int nhead_k,
                            int seqlen_q,
                            int seqlen_k,
                            int hdim_q,
                            int hdim_v,
                            float scale,
                            int mask_type_int,
                            int bias_type_int,
                            int has_lse,
                            int has_dropout,
                            int traits_hdim_q,
                            int traits_hdim_v,
                            int is_v_rowmajor,
                            int perm,
                            const char* data_type_str,
                            int is_group_mode,
                            int window_left,
                            int window_right,
                            float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    int rc                   = 0;
    const int64_t q_bytes    = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * 2;
    const int64_t k_bytes    = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * 2;
    const int64_t v_bytes    = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * 2;
    const int64_t o_bytes    = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * 2;
    const int64_t bias_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * seqlen_k * 2;
    const int64_t lse_bytes  = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    float elapsed            = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *bias_dev = nullptr, *lse_dev_buf = nullptr;
    void *seqstart_q_dev = nullptr, *seqstart_k_dev = nullptr, *seqlen_k_dev = nullptr;

    fmha_fwd_traits traits{};
    traits.hdim_q        = (traits_hdim_q > 0) ? traits_hdim_q : hdim_q;
    traits.hdim_v        = (traits_hdim_v > 0) ? traits_hdim_v : hdim_v;
    traits.data_type     = data_type_str ? data_type_str : "fp16";
    traits.is_group_mode = (is_group_mode != 0);
    traits.is_v_rowmajor = (is_v_rowmajor != 0);
    traits.mask_type     = static_cast<mask_enum>(mask_type_int);
    traits.bias_type     = static_cast<bias_enum>(bias_type_int);
    traits.has_lse       = (has_lse != 0);
    traits.has_dropout   = (has_dropout != 0);
    traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));

    if(is_group_mode)
    {
        std::vector<int> sq_starts(batch + 1), sk_starts(batch + 1), sk_lens(batch);
        for(int b = 0; b <= batch; ++b)
        {
            sq_starts[b] = b * seqlen_q;
            sk_starts[b] = b * seqlen_k;
        }
        for(int b = 0; b < batch; ++b)
            sk_lens[b] = seqlen_k;

        HIP_CHECK(hipMalloc(&seqstart_q_dev, (batch + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc(&seqstart_k_dev, (batch + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc(&seqlen_k_dev, batch * sizeof(int)));
        HIP_CHECK(hipMemcpy(
            seqstart_q_dev, sq_starts.data(), (batch + 1) * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            seqstart_k_dev, sk_starts.data(), (batch + 1) * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(seqlen_k_dev, sk_lens.data(), batch * sizeof(int), hipMemcpyHostToDevice));
    }

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(o_dev, 0, o_bytes));

    if(bias_type_int > 0)
    {
        HIP_CHECK(hipMalloc(&bias_dev, bias_bytes));
        HIP_CHECK(hipMemset(bias_dev, 0, bias_bytes));
    }
    if(has_lse)
    {
        HIP_CHECK(hipMalloc(&lse_dev_buf, lse_bytes));
        HIP_CHECK(hipMemset(lse_dev_buf, 0, lse_bytes));
    }

    args.q_ptr                      = q_dev;
    args.k_ptr                      = k_dev;
    args.v_ptr                      = v_dev;
    args.o_ptr                      = o_dev;
    args.bias_ptr                   = bias_dev;
    args.q_descale_ptr              = nullptr;
    args.k_descale_ptr              = nullptr;
    args.v_descale_ptr              = nullptr;
    args.rand_val_ptr               = nullptr;
    args.lse_ptr                    = lse_dev_buf;
    args.seqstart_q_ptr             = seqstart_q_dev;
    args.seqstart_k_ptr             = seqstart_k_dev;
    args.seqlen_q_ptr               = nullptr;
    args.seqlen_k_ptr               = seqlen_k_dev;
    args.sink_ptr                   = nullptr;
    args.block_scale_seqstart_q_ptr = nullptr;
    args.block_scale_seqstart_k_ptr = nullptr;

    args.seqlen_q        = seqlen_q;
    args.seqlen_k        = seqlen_k;
    args.batch           = batch;
    args.max_seqlen_q    = seqlen_q;
    args.hdim_q          = hdim_q;
    args.hdim_v          = hdim_v;
    args.nhead_q         = nhead_q;
    args.nhead_k         = nhead_k;
    args.scale_s         = scale;
    args.logits_soft_cap = 0.0f;

    if(is_group_mode)
    {
        // Group mode: [total_tokens, nhead, hdim] -- batch via seqstart arrays
        args.stride_q       = nhead_q * hdim_q;
        args.stride_k       = nhead_k * hdim_q;
        args.stride_v       = nhead_k * hdim_v;
        args.stride_o       = nhead_q * hdim_v;
        args.nhead_stride_q = hdim_q;
        args.nhead_stride_k = hdim_q;
        args.nhead_stride_v = hdim_v;
        args.nhead_stride_o = hdim_v;
        args.batch_stride_q = 0;
        args.batch_stride_k = 0;
        args.batch_stride_v = 0;
        args.batch_stride_o = 0;
    }
    else if(perm == 1)
    {
        // BHSD layout: [batch, head, seq, dim]
        args.stride_q       = hdim_q;
        args.stride_k       = hdim_q;
        args.stride_v       = hdim_v;
        args.stride_o       = hdim_v;
        args.nhead_stride_q = seqlen_q * hdim_q;
        args.nhead_stride_k = seqlen_k * hdim_q;
        args.nhead_stride_v = seqlen_k * hdim_v;
        args.nhead_stride_o = seqlen_q * hdim_v;
        args.batch_stride_q = nhead_q * seqlen_q * hdim_q;
        args.batch_stride_k = nhead_k * seqlen_k * hdim_q;
        args.batch_stride_v = nhead_k * seqlen_k * hdim_v;
        args.batch_stride_o = nhead_q * seqlen_q * hdim_v;
    }
    else
    {
        // BSHD layout: [batch, seq, head, dim]
        args.stride_q       = nhead_q * hdim_q;
        args.stride_k       = nhead_k * hdim_q;
        args.stride_v       = nhead_k * hdim_v;
        args.stride_o       = nhead_q * hdim_v;
        args.nhead_stride_q = hdim_q;
        args.nhead_stride_k = hdim_q;
        args.nhead_stride_v = hdim_v;
        args.nhead_stride_o = hdim_v;
        args.batch_stride_q = seqlen_q * nhead_q * hdim_q;
        args.batch_stride_k = seqlen_k * nhead_k * hdim_q;
        args.batch_stride_v = seqlen_k * nhead_k * hdim_v;
        args.batch_stride_o = seqlen_q * nhead_q * hdim_v;
    }
    args.stride_bias            = (bias_type_int > 0) ? seqlen_k : 0;
    args.stride_randval         = 0;
    args.nhead_stride_bias      = (bias_type_int > 0) ? seqlen_q * seqlen_k : 0;
    args.nhead_stride_randval   = 0;
    args.nhead_stride_lse       = has_lse ? seqlen_q : 0;
    args.nhead_stride_q_descale = 0;
    args.nhead_stride_k_descale = 0;
    args.nhead_stride_v_descale = 0;
    args.batch_stride_bias      = (bias_type_int > 0) ? nhead_q * seqlen_q * seqlen_k : 0;
    args.batch_stride_randval   = 0;
    args.batch_stride_lse       = has_lse ? nhead_q * seqlen_q : 0;
    args.batch_stride_q_descale = 0;
    args.batch_stride_k_descale = 0;
    args.batch_stride_v_descale = 0;

    args.window_size_left    = window_left;
    args.window_size_right   = window_right;
    args.sink_size           = 0;
    args.mask_type           = mask_type_int;
    args.min_seqlen_q        = 0;
    args.p_drop              = has_dropout ? 0.2f : 0.0f;
    args.s_randval           = false;
    args.drop_seed_offset    = has_dropout ? std::make_pair(uint64_t(1), uint64_t(0))
                                           : std::make_pair(uint64_t(0), uint64_t(0));
    args.block_scale_size_q  = 0;
    args.block_scale_size_kv = 0;

    try
    {
        elapsed = g_dispatcher->run_fwd(traits, args, nullptr);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "FMHA_ERR: %s\n", e.what());
        rc = -2;
        goto cleanup;
    }
    catch(...)
    {
        fprintf(stderr, "FMHA_ERR: unknown\n");
        rc = -2;
        goto cleanup;
    }

    {
        hipError_t cpy_err = hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost);
        if(cpy_err != hipSuccess)
            rc = -1;
    }

    if(time_ms_out)
        *time_ms_out = elapsed;

cleanup:
    safe_hip_free(q_dev);
    safe_hip_free(k_dev);
    safe_hip_free(v_dev);
    safe_hip_free(o_dev);
    safe_hip_free(bias_dev);
    safe_hip_free(lse_dev_buf);
    safe_hip_free(seqstart_q_dev);
    safe_hip_free(seqstart_k_dev);
    safe_hip_free(seqlen_k_dev);

    return rc;
}

int fmha_dispatcher_run_bwd(const void* q_host,
                            const void* k_host,
                            const void* v_host,
                            const void* o_host,
                            const void* lse_host,
                            const void* do_host,
                            void* dq_host,
                            void* dk_host,
                            void* dv_host,
                            int batch,
                            int nhead_q,
                            int nhead_k,
                            int seqlen_q,
                            int seqlen_k,
                            int hdim_q,
                            int hdim_v,
                            float scale,
                            float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    int rc                     = 0;
    const int64_t q_bytes      = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * 2;
    const int64_t k_bytes      = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * 2;
    const int64_t v_bytes      = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * 2;
    const int64_t o_bytes      = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * 2;
    const int64_t do_bytes     = o_bytes;
    const int64_t dq_bytes     = q_bytes;
    const int64_t dk_bytes     = k_bytes;
    const int64_t dv_bytes     = v_bytes;
    const int64_t lse_bytes    = static_cast<int64_t>(batch) * nhead_q * seqlen_q * 4;
    const int64_t d_bytes      = static_cast<int64_t>(batch) * nhead_q * seqlen_q * 4;
    const int64_t dq_acc_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * 4;
    float elapsed              = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *lse_dev = nullptr, *do_dev = nullptr, *d_dev = nullptr;
    void *dq_dev = nullptr, *dk_dev = nullptr, *dv_dev = nullptr, *dq_acc_dev = nullptr;

    fmha_bwd_traits traits{};
    traits.hdim_q           = hdim_q;
    traits.hdim_v           = hdim_v;
    traits.data_type        = "fp16";
    traits.is_group_mode    = false;
    traits.mask_type        = mask_enum::no_mask;
    traits.bias_type        = bias_enum::no_bias;
    traits.has_dbias        = false;
    traits.has_dropout      = false;
    traits.is_store_randval = false;
    traits.is_deterministic = false;

    fmha_bwd_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));
    HIP_CHECK(hipMalloc(&lse_dev, lse_bytes));
    HIP_CHECK(hipMalloc(&do_dev, do_bytes));
    HIP_CHECK(hipMalloc(&d_dev, d_bytes));
    HIP_CHECK(hipMalloc(&dq_dev, dq_bytes));
    HIP_CHECK(hipMalloc(&dk_dev, dk_bytes));
    HIP_CHECK(hipMalloc(&dv_dev, dv_bytes));
    HIP_CHECK(hipMalloc(&dq_acc_dev, dq_acc_bytes));

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(o_dev, o_host, o_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lse_dev, lse_host, lse_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(do_dev, do_host, do_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_dev, 0, d_bytes));
    HIP_CHECK(hipMemset(dq_dev, 0, dq_bytes));
    HIP_CHECK(hipMemset(dk_dev, 0, dk_bytes));
    HIP_CHECK(hipMemset(dv_dev, 0, dv_bytes));
    HIP_CHECK(hipMemset(dq_acc_dev, 0, dq_acc_bytes));

    args.q_ptr        = q_dev;
    args.k_ptr        = k_dev;
    args.v_ptr        = v_dev;
    args.bias_ptr     = nullptr;
    args.o_ptr        = o_dev;
    args.lse_ptr      = lse_dev;
    args.do_ptr       = do_dev;
    args.d_ptr        = d_dev;
    args.rand_val_ptr = nullptr;
    args.dq_ptr       = dq_dev;
    args.dk_ptr       = dk_dev;
    args.dv_ptr       = dv_dev;
    args.dbias_ptr    = nullptr;
    args.dq_acc_ptr   = dq_acc_dev;

    args.seqlen_q     = seqlen_q;
    args.seqlen_k     = seqlen_k;
    args.batch        = batch;
    args.max_seqlen_q = seqlen_q;
    args.max_seqlen_k = seqlen_k;
    args.hdim_q       = hdim_q;
    args.hdim_v       = hdim_v;
    args.nhead_q      = nhead_q;
    args.nhead_k      = nhead_k;
    args.scale        = scale;

    // bhsd strides
    args.stride_q       = hdim_q;
    args.stride_k       = hdim_q;
    args.stride_v       = hdim_v;
    args.stride_bias    = 0;
    args.stride_o       = hdim_v;
    args.stride_randval = 0;
    args.stride_do      = hdim_v;
    args.stride_dq_acc  = hdim_q;
    args.stride_dq      = hdim_q;
    args.stride_dk      = hdim_q;
    args.stride_dv      = hdim_v;
    args.stride_dbias   = 0;

    args.nhead_stride_q       = seqlen_q * hdim_q;
    args.nhead_stride_k       = seqlen_k * hdim_q;
    args.nhead_stride_v       = seqlen_k * hdim_v;
    args.nhead_stride_bias    = 0;
    args.nhead_stride_o       = seqlen_q * hdim_v;
    args.nhead_stride_randval = 0;
    args.nhead_stride_do      = seqlen_q * hdim_v;
    args.nhead_stride_lsed    = seqlen_q;
    args.nhead_stride_dq_acc  = static_cast<ck_tile::long_index_t>(seqlen_q) * hdim_q;
    args.nhead_stride_dq      = seqlen_q * hdim_q;
    args.nhead_stride_dk      = seqlen_k * hdim_q;
    args.nhead_stride_dv      = seqlen_k * hdim_v;
    args.nhead_stride_dbias   = 0;

    args.batch_stride_q       = nhead_q * seqlen_q * hdim_q;
    args.batch_stride_k       = nhead_k * seqlen_k * hdim_q;
    args.batch_stride_v       = nhead_k * seqlen_k * hdim_v;
    args.batch_stride_bias    = 0;
    args.batch_stride_o       = nhead_q * seqlen_q * hdim_v;
    args.batch_stride_randval = 0;
    args.batch_stride_do      = nhead_q * seqlen_q * hdim_v;
    args.batch_stride_lsed    = nhead_q * seqlen_q;
    args.batch_stride_dq_acc  = static_cast<ck_tile::long_index_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_dq      = nhead_q * seqlen_q * hdim_q;
    args.batch_stride_dk      = nhead_k * seqlen_k * hdim_q;
    args.batch_stride_dv      = nhead_k * seqlen_k * hdim_v;
    args.batch_stride_dbias   = 0;
    args.split_stride_dq_acc  = 0;

    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = 0;
    args.p_drop            = 0.0f;
    args.p_undrop          = 1.0f;
    args.drop_seed_offset  = std::make_pair(uint64_t(0), uint64_t(0));

    try
    {
        elapsed = g_dispatcher->run_bwd(traits, args, nullptr);
    }
    catch(...)
    {
        rc = -2;
        goto cleanup;
    }

    {
        hipError_t e1 = hipMemcpy(dq_host, dq_dev, dq_bytes, hipMemcpyDeviceToHost);
        hipError_t e2 = hipMemcpy(dk_host, dk_dev, dk_bytes, hipMemcpyDeviceToHost);
        hipError_t e3 = hipMemcpy(dv_host, dv_dev, dv_bytes, hipMemcpyDeviceToHost);
        if(e1 != hipSuccess || e2 != hipSuccess || e3 != hipSuccess)
            rc = -1;
    }

    if(time_ms_out)
        *time_ms_out = elapsed;

cleanup:
    safe_hip_free(q_dev);
    safe_hip_free(k_dev);
    safe_hip_free(v_dev);
    safe_hip_free(o_dev);
    safe_hip_free(lse_dev);
    safe_hip_free(do_dev);
    safe_hip_free(d_dev);
    safe_hip_free(dq_dev);
    safe_hip_free(dk_dev);
    safe_hip_free(dv_dev);
    safe_hip_free(dq_acc_dev);

    return rc;
}

int fmha_dispatcher_kernel_count()
{
    return g_initialized ? static_cast<int>(g_registry->size()) : 0;
}

void fmha_dispatcher_cleanup()
{
    g_dispatcher.reset();
    g_registry.reset();
    g_initialized = false;
}

} // extern "C"
