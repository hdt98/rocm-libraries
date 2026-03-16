// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// FMHA Dispatcher ctypes library.
// Provides a C API for Python ctypes integration.
// Kernel header included via -include at compile time.
//
// Thread safety: NOT thread-safe. Python ctypes releases the GIL during
// foreign calls, so single-threaded usage must be enforced by the caller.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

#include "ck_tile/dispatcher.hpp"

#ifndef GFX_ARCH
#error "GFX_ARCH must be defined at compile time (e.g. -DGFX_ARCH=\"gfx950\")"
#endif

using namespace ck_tile::dispatcher;

static std::unique_ptr<FmhaRegistry> g_registry;
static std::unique_ptr<FmhaDispatcher> g_dispatcher;
static bool g_initialized = false;

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

static inline void safe_hip_free(void*& ptr)
{
    if(ptr)
    {
        hipFree(ptr);
        ptr = nullptr;
    }
}

static int dtype_input_bytes(const char* dtype)
{
    if(!dtype)
        return 2;
    if(std::strcmp(dtype, "fp32") == 0)
        return 4;
    if(std::strcmp(dtype, "fp8bf16") == 0 || std::strcmp(dtype, "fp8fp32") == 0 ||
       std::strcmp(dtype, "bf8") == 0 || std::strcmp(dtype, "fp8") == 0)
        return 1;
    return 2; // fp16, bf16
}

static int dtype_output_bytes(const char* dtype)
{
    if(!dtype)
        return 2;
    if(std::strcmp(dtype, "fp32") == 0 || std::strcmp(dtype, "fp8fp32") == 0)
        return 4;
    if(std::strcmp(dtype, "fp8") == 0 || std::strcmp(dtype, "bf8") == 0)
        return 1;
    return 2; // fp16, bf16, fp8bf16 (output is bf16)
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
                            int has_logits,
                            int has_sink,
                            int has_skip,
                            float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes  = dtype_input_bytes(data_type_str);
    const int out_bytes = dtype_output_bytes(data_type_str);

    int rc                = 0;
    const int64_t q_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t k_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * in_bytes;
    const int64_t v_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * in_bytes;
    const int64_t o_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * out_bytes;
    const int64_t bias_bytes =
        static_cast<int64_t>(batch) * nhead_q * seqlen_q * seqlen_k * out_bytes;
    const int64_t lse_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    float elapsed           = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *bias_dev = nullptr, *lse_dev_buf = nullptr;
    void *seqstart_q_dev = nullptr, *seqstart_k_dev = nullptr, *seqlen_k_dev = nullptr;

    fmha_fwd_traits traits{};
    traits.hdim_q              = (traits_hdim_q > 0) ? traits_hdim_q : hdim_q;
    traits.hdim_v              = (traits_hdim_v > 0) ? traits_hdim_v : hdim_v;
    traits.data_type           = data_type_str ? data_type_str : "fp16";
    traits.is_group_mode       = (is_group_mode != 0);
    traits.is_v_rowmajor       = (is_v_rowmajor != 0);
    traits.mask_type           = static_cast<mask_enum>(mask_type_int);
    traits.bias_type           = static_cast<bias_enum>(bias_type_int);
    traits.has_lse             = (has_lse != 0);
    traits.has_dropout         = (has_dropout != 0);
    traits.qscale_type         = quant_scale_enum::no_scale;
    traits.has_logits_soft_cap = (has_logits != 0);
    traits.skip_min_seqlen_q   = (has_skip != 0);
    traits.has_sink            = (has_sink != 0);

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
        // BHSD: [batch, head, seq, dim]
        args.stride_q       = hdim_q;
        args.stride_k       = hdim_q;
        args.stride_v       = hdim_v;
        args.stride_o       = hdim_v;
        args.nhead_stride_q = seqlen_q * hdim_q;
        args.nhead_stride_k = seqlen_k * hdim_q;
        args.nhead_stride_v = seqlen_k * hdim_v;
        args.nhead_stride_o = seqlen_q * hdim_v;
        args.batch_stride_q = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
        args.batch_stride_k = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
        args.batch_stride_v = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
        args.batch_stride_o = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    }
    else
    {
        // BSHD: [batch, seq, head, dim]
        args.stride_q       = nhead_q * hdim_q;
        args.stride_k       = nhead_k * hdim_q;
        args.stride_v       = nhead_k * hdim_v;
        args.stride_o       = nhead_q * hdim_v;
        args.nhead_stride_q = hdim_q;
        args.nhead_stride_k = hdim_q;
        args.nhead_stride_v = hdim_v;
        args.nhead_stride_o = hdim_v;
        args.batch_stride_q = static_cast<int64_t>(seqlen_q) * nhead_q * hdim_q;
        args.batch_stride_k = static_cast<int64_t>(seqlen_k) * nhead_k * hdim_q;
        args.batch_stride_v = static_cast<int64_t>(seqlen_k) * nhead_k * hdim_v;
        args.batch_stride_o = static_cast<int64_t>(seqlen_q) * nhead_q * hdim_v;
    }
    args.stride_bias          = (bias_type_int > 0) ? seqlen_k : 0;
    args.stride_randval       = 0;
    args.nhead_stride_bias    = (bias_type_int > 0) ? static_cast<int64_t>(seqlen_q) * seqlen_k : 0;
    args.nhead_stride_randval = 0;
    args.nhead_stride_lse     = has_lse ? seqlen_q : 0;
    args.nhead_stride_q_descale = 0;
    args.nhead_stride_k_descale = 0;
    args.nhead_stride_v_descale = 0;
    args.batch_stride_bias =
        (bias_type_int > 0) ? static_cast<int64_t>(nhead_q) * seqlen_q * seqlen_k : 0;
    args.batch_stride_randval   = 0;
    args.batch_stride_lse       = has_lse ? static_cast<int64_t>(nhead_q) * seqlen_q : 0;
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
                            const char* data_type_str,
                            float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes  = dtype_input_bytes(data_type_str);
    const int out_bytes = dtype_output_bytes(data_type_str);

    int rc                  = 0;
    const int64_t q_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t k_bytes   = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * in_bytes;
    const int64_t v_bytes   = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * in_bytes;
    const int64_t o_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * out_bytes;
    const int64_t do_bytes  = o_bytes;
    const int64_t dq_bytes  = q_bytes;
    const int64_t dk_bytes  = k_bytes;
    const int64_t dv_bytes  = v_bytes;
    const int64_t lse_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    const int64_t d_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    const int64_t dq_acc_bytes =
        static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * sizeof(float);
    float elapsed = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *lse_dev = nullptr, *do_dev = nullptr, *d_dev = nullptr;
    void *dq_dev = nullptr, *dk_dev = nullptr, *dv_dev = nullptr, *dq_acc_dev = nullptr;

    fmha_bwd_traits traits{};
    traits.hdim_q           = hdim_q;
    traits.hdim_v           = hdim_v;
    traits.data_type        = data_type_str ? data_type_str : "fp16";
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

    // bhsd strides (cast first operand to int64_t to prevent int32 overflow)
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

    args.nhead_stride_q       = static_cast<int64_t>(seqlen_q) * hdim_q;
    args.nhead_stride_k       = static_cast<int64_t>(seqlen_k) * hdim_q;
    args.nhead_stride_v       = static_cast<int64_t>(seqlen_k) * hdim_v;
    args.nhead_stride_bias    = 0;
    args.nhead_stride_o       = static_cast<int64_t>(seqlen_q) * hdim_v;
    args.nhead_stride_randval = 0;
    args.nhead_stride_do      = static_cast<int64_t>(seqlen_q) * hdim_v;
    args.nhead_stride_lsed    = seqlen_q;
    args.nhead_stride_dq_acc  = static_cast<ck_tile::long_index_t>(seqlen_q) * hdim_q;
    args.nhead_stride_dq      = static_cast<int64_t>(seqlen_q) * hdim_q;
    args.nhead_stride_dk      = static_cast<int64_t>(seqlen_k) * hdim_q;
    args.nhead_stride_dv      = static_cast<int64_t>(seqlen_k) * hdim_v;
    args.nhead_stride_dbias   = 0;

    args.batch_stride_q       = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_k       = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
    args.batch_stride_v       = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
    args.batch_stride_bias    = 0;
    args.batch_stride_o       = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    args.batch_stride_randval = 0;
    args.batch_stride_do      = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    args.batch_stride_lsed    = static_cast<int64_t>(nhead_q) * seqlen_q;
    args.batch_stride_dq_acc  = static_cast<ck_tile::long_index_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_dq      = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_dk      = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
    args.batch_stride_dv      = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
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

// ---------------------------------------------------------------------------
// Split-KV forward: 2-stage (split + combine)
// Allocates o_acc / lse_acc internally for the split stage.
// ---------------------------------------------------------------------------
int fmha_dispatcher_run_splitkv(const void* q_host,
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
                                int num_splits,
                                int is_v_rowmajor,
                                const char* data_type_str,
                                int has_lse,
                                float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes  = dtype_input_bytes(data_type_str);
    const int out_bytes = dtype_output_bytes(data_type_str);

    int rc                = 0;
    const int64_t q_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t k_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * in_bytes;
    const int64_t v_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * in_bytes;
    const int64_t o_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * out_bytes;
    const int64_t o_acc_bytes =
        static_cast<int64_t>(num_splits) * batch * nhead_q * seqlen_q * hdim_v * sizeof(float);
    const int64_t lse_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    const int64_t lse_acc_bytes =
        static_cast<int64_t>(num_splits) * batch * nhead_q * seqlen_q * sizeof(float);
    float elapsed = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *o_acc_dev = nullptr, *lse_dev = nullptr, *lse_acc_dev = nullptr;

    fmha_fwd_splitkv_traits traits{};
    traits.hdim_q              = hdim_q;
    traits.hdim_v              = hdim_v;
    traits.data_type           = data_type_str ? data_type_str : "fp16";
    traits.is_group_mode       = false;
    traits.is_v_rowmajor       = (is_v_rowmajor != 0);
    traits.has_logits_soft_cap = false;
    traits.mask_type           = static_cast<mask_enum>(mask_type_int);
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = (has_lse != 0);

    fmha_fwd_splitkv_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));
    HIP_CHECK(hipMalloc(&o_acc_dev, o_acc_bytes));
    HIP_CHECK(hipMalloc(&lse_dev, lse_bytes));
    HIP_CHECK(hipMalloc(&lse_acc_dev, lse_acc_bytes));

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(o_dev, 0, o_bytes));
    HIP_CHECK(hipMemset(o_acc_dev, 0, o_acc_bytes));
    HIP_CHECK(hipMemset(lse_dev, 0, lse_bytes));
    HIP_CHECK(hipMemset(lse_acc_dev, 0, lse_acc_bytes));

    args.q_ptr                    = q_dev;
    args.k_ptr                    = k_dev;
    args.v_ptr                    = v_dev;
    args.bias_ptr                 = nullptr;
    args.lse_acc_ptr              = lse_acc_dev;
    args.o_acc_ptr                = o_acc_dev;
    args.lse_ptr                  = lse_dev;
    args.o_ptr                    = o_dev;
    args.block_table_ptr          = nullptr;
    args.batch_stride_block_table = 0;
    args.page_block_size          = 0;
    args.is_gappy                 = false;
    args.cache_batch_idx          = nullptr;
    args.seqstart_q_ptr           = nullptr;
    args.seqstart_k_ptr           = nullptr;
    args.seqlen_k_ptr             = nullptr;
    args.sink_ptr                 = nullptr;
    args.seqlen_q                 = seqlen_q;
    args.seqlen_k                 = seqlen_k;
    args.batch                    = batch;
    args.max_seqlen_q             = seqlen_q;
    args.hdim_q                   = hdim_q;
    args.hdim_v                   = hdim_v;
    args.nhead_q                  = nhead_q;
    args.nhead_k                  = nhead_k;
    args.num_splits               = num_splits;
    args.scale_s                  = scale;
    args.scale_p                  = 1.0f;
    args.scale_o                  = 1.0f;
    args.logits_soft_cap          = 0.0f;

    // BHSD strides
    args.stride_q             = hdim_q;
    args.stride_k             = hdim_q;
    args.stride_v             = hdim_v;
    args.stride_bias          = 0;
    args.stride_o_acc         = hdim_v;
    args.stride_o             = hdim_v;
    args.nhead_stride_q       = static_cast<int64_t>(seqlen_q) * hdim_q;
    args.nhead_stride_k       = static_cast<int64_t>(seqlen_k) * hdim_q;
    args.nhead_stride_v       = static_cast<int64_t>(seqlen_k) * hdim_v;
    args.nhead_stride_bias    = 0;
    args.nhead_stride_lse     = seqlen_q;
    args.nhead_stride_lse_acc = seqlen_q;
    args.nhead_stride_o_acc   = static_cast<int64_t>(seqlen_q) * hdim_v;
    args.nhead_stride_o       = static_cast<int64_t>(seqlen_q) * hdim_v;
    args.batch_stride_q       = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_k       = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
    args.batch_stride_v       = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
    args.batch_stride_bias    = 0;
    args.batch_stride_lse     = static_cast<int64_t>(nhead_q) * seqlen_q;
    args.batch_stride_lse_acc = static_cast<int64_t>(nhead_q) * seqlen_q;
    args.batch_stride_o_acc   = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    args.batch_stride_o       = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    args.split_stride_lse_acc = static_cast<int64_t>(batch) * nhead_q * seqlen_q;
    args.split_stride_o_acc   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v;
    args.window_size_left     = -1;
    args.window_size_right    = -1;
    args.sink_size            = 0;
    args.mask_type            = mask_type_int;

    try
    {
        elapsed = g_dispatcher->run_fwd_splitkv(traits, args, nullptr);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "FMHA_ERR: %s\n", e.what());
        rc = -2;
        goto cleanup;
    }
    catch(...)
    {
        rc = -2;
        goto cleanup;
    }

    {
        hipError_t cpy = hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost);
        if(cpy != hipSuccess)
            rc = -1;
    }
    if(time_ms_out)
        *time_ms_out = elapsed;

cleanup:
    safe_hip_free(q_dev);
    safe_hip_free(k_dev);
    safe_hip_free(v_dev);
    safe_hip_free(o_dev);
    safe_hip_free(o_acc_dev);
    safe_hip_free(lse_dev);
    safe_hip_free(lse_acc_dev);
    return rc;
}

// ---------------------------------------------------------------------------
// Paged-KV forward: Q in standard layout, K/V in paged blocks
// Creates a trivial contiguous page table for benchmarking.
// ---------------------------------------------------------------------------
int fmha_dispatcher_run_pagedkv(const void* q_host,
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
                                int page_block_size,
                                int is_v_rowmajor,
                                const char* data_type_str,
                                int has_lse,
                                float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes  = dtype_input_bytes(data_type_str);
    const int out_bytes = dtype_output_bytes(data_type_str);

    int rc                  = 0;
    const int64_t q_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t k_bytes   = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * in_bytes;
    const int64_t v_bytes   = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * in_bytes;
    const int64_t o_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * out_bytes;
    const int64_t lse_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    float elapsed           = 0.0f;

    if(page_block_size <= 0)
        page_block_size = 64;
    const int pages_per_seq = (seqlen_k + page_block_size - 1) / page_block_size;
    const int total_pages   = batch * pages_per_seq;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *lse_dev = nullptr, *block_table_dev = nullptr;
    void* seqlen_k_dev = nullptr;

    // Declare vectors before any HIP_CHECK to avoid goto-over-init
    std::vector<int> block_table(total_pages);
    for(int i = 0; i < total_pages; ++i)
        block_table[i] = i;
    std::vector<int> seqlen_k_vec(batch, seqlen_k);

    fmha_fwd_pagedkv_traits traits{};
    traits.hdim_q              = hdim_q;
    traits.hdim_v              = hdim_v;
    traits.data_type           = data_type_str ? data_type_str : "fp16";
    traits.is_group_mode       = true;
    traits.is_v_rowmajor       = (is_v_rowmajor != 0);
    traits.has_logits_soft_cap = false;
    traits.mask_type           = static_cast<mask_enum>(mask_type_int);
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = (has_lse != 0);
    traits.use_pagedkv         = true;

    fmha_fwd_pagedkv_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));

    HIP_CHECK(hipMalloc(&block_table_dev, total_pages * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        block_table_dev, block_table.data(), total_pages * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&seqlen_k_dev, batch * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(seqlen_k_dev, seqlen_k_vec.data(), batch * sizeof(int), hipMemcpyHostToDevice));

    if(has_lse)
    {
        HIP_CHECK(hipMalloc(&lse_dev, lse_bytes));
        HIP_CHECK(hipMemset(lse_dev, 0, lse_bytes));
    }

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(o_dev, 0, o_bytes));

    args.q_ptr                    = q_dev;
    args.k_ptr                    = k_dev;
    args.v_ptr                    = v_dev;
    args.bias_ptr                 = nullptr;
    args.lse_ptr                  = lse_dev;
    args.o_ptr                    = o_dev;
    args.block_table_ptr          = block_table_dev;
    args.batch_stride_block_table = pages_per_seq;
    args.page_block_size          = page_block_size;
    args.is_gappy                 = false;
    args.cache_batch_idx          = nullptr;
    args.seqstart_q_ptr           = nullptr;
    args.seqstart_k_ptr           = nullptr;
    args.seqlen_k_ptr             = seqlen_k_dev;
    args.sink_ptr                 = nullptr;
    args.seqlen_q                 = seqlen_q;
    args.seqlen_k                 = seqlen_k;
    args.batch                    = batch;
    args.max_seqlen_q             = seqlen_q;
    args.hdim_q                   = hdim_q;
    args.hdim_v                   = hdim_v;
    args.nhead_q                  = nhead_q;
    args.nhead_k                  = nhead_k;
    args.scale_s                  = scale;
    args.scale_p                  = 1.0f;
    args.scale_o                  = 1.0f;
    args.logits_soft_cap          = 0.0f;

    // K/V stored in page table: [total_pages, nhead_k, page_block_size, hdim]
    args.stride_q          = hdim_q;
    args.stride_k          = hdim_q;
    args.stride_v          = hdim_v;
    args.stride_bias       = 0;
    args.stride_o          = hdim_v;
    args.nhead_stride_q    = static_cast<int64_t>(seqlen_q) * hdim_q;
    args.nhead_stride_k    = static_cast<int64_t>(page_block_size) * hdim_q;
    args.nhead_stride_v    = static_cast<int64_t>(page_block_size) * hdim_v;
    args.nhead_stride_bias = 0;
    args.nhead_stride_lse  = seqlen_q;
    args.nhead_stride_o    = static_cast<int64_t>(seqlen_q) * hdim_v;
    args.batch_stride_q    = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_k    = static_cast<int64_t>(nhead_k) * page_block_size * hdim_q;
    args.batch_stride_v    = static_cast<int64_t>(nhead_k) * page_block_size * hdim_v;
    args.batch_stride_bias = 0;
    args.batch_stride_lse  = static_cast<int64_t>(nhead_q) * seqlen_q;
    args.batch_stride_o    = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.sink_size         = 0;
    args.mask_type         = mask_type_int;
    args.min_seqlen_q      = 0;

    try
    {
        elapsed = g_dispatcher->run_fwd_pagedkv(traits, args, nullptr);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "FMHA_ERR: %s\n", e.what());
        rc = -2;
        goto cleanup;
    }
    catch(...)
    {
        rc = -2;
        goto cleanup;
    }

    {
        hipError_t cpy = hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost);
        if(cpy != hipSuccess)
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
    safe_hip_free(block_table_dev);
    safe_hip_free(seqlen_k_dev);
    return rc;
}

// ---------------------------------------------------------------------------
// Append-KV: appends knew/vnew into K/V cache, optionally with RoPE
// ---------------------------------------------------------------------------
int fmha_dispatcher_run_appendkv(const void* q_host,
                                 const void* knew_host,
                                 const void* vnew_host,
                                 int batch,
                                 int nhead_q,
                                 int nhead_k,
                                 int seqlen_q,
                                 int seqlen_knew,
                                 int hdim_q,
                                 int hdim_v,
                                 int is_v_rowmajor,
                                 const char* data_type_str,
                                 float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes = dtype_input_bytes(data_type_str);
    int rc             = 0;

    const int seqlen_k    = seqlen_q + seqlen_knew;
    const int64_t q_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t knew_bytes =
        static_cast<int64_t>(batch) * nhead_k * seqlen_knew * hdim_q * in_bytes;
    const int64_t vnew_bytes =
        static_cast<int64_t>(batch) * nhead_k * seqlen_knew * hdim_v * in_bytes;
    const int64_t k_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * in_bytes;
    const int64_t v_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * in_bytes;
    float elapsed         = 0.0f;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr;
    void *knew_dev = nullptr, *vnew_dev = nullptr;
    void* seqlen_k_dev = nullptr;

    fmha_fwd_appendkv_traits traits{};
    traits.hdim_q        = hdim_q;
    traits.hdim_v        = hdim_v;
    traits.data_type     = data_type_str ? data_type_str : "fp16";
    traits.is_v_rowmajor = (is_v_rowmajor != 0);
    traits.rope_type     = rope_enum::none;

    std::vector<int> sk_vec(batch, seqlen_k - seqlen_knew);

    fmha_fwd_appendkv_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&knew_dev, knew_bytes));
    HIP_CHECK(hipMalloc(&vnew_dev, vnew_bytes));

    HIP_CHECK(hipMalloc(&seqlen_k_dev, batch * sizeof(int)));
    HIP_CHECK(hipMemcpy(seqlen_k_dev, sk_vec.data(), batch * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(knew_dev, knew_host, knew_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(vnew_dev, vnew_host, vnew_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(k_dev, 0, k_bytes));
    HIP_CHECK(hipMemset(v_dev, 0, v_bytes));

    args.q_ptr                    = q_dev;
    args.k_ptr                    = k_dev;
    args.knew_ptr                 = knew_dev;
    args.v_ptr                    = v_dev;
    args.vnew_ptr                 = vnew_dev;
    args.seqlen_k_ptr             = seqlen_k_dev;
    args.seqlen_q                 = seqlen_q;
    args.seqlen_knew              = seqlen_knew;
    args.batch                    = batch;
    args.hdim_q                   = hdim_q;
    args.hdim_v                   = hdim_v;
    args.nhead_q                  = nhead_q;
    args.nhead_k                  = nhead_k;
    args.rotary_cos_ptr           = nullptr;
    args.rotary_sin_ptr           = nullptr;
    args.rotary_dim               = 0;
    args.has_mask                 = false;
    args.block_table_ptr          = nullptr;
    args.batch_stride_block_table = 0;
    args.page_block_size          = 0;
    args.cache_batch_idx          = nullptr;
    args.sink_ptr                 = nullptr;

    // BHSD strides
    args.stride_q          = hdim_q;
    args.stride_k          = hdim_q;
    args.stride_knew       = hdim_q;
    args.stride_v          = hdim_v;
    args.stride_vnew       = hdim_v;
    args.nhead_stride_q    = static_cast<int64_t>(seqlen_q) * hdim_q;
    args.nhead_stride_k    = static_cast<int64_t>(seqlen_k) * hdim_q;
    args.nhead_stride_knew = static_cast<int64_t>(seqlen_knew) * hdim_q;
    args.nhead_stride_v    = static_cast<int64_t>(seqlen_k) * hdim_v;
    args.nhead_stride_vnew = static_cast<int64_t>(seqlen_knew) * hdim_v;
    args.batch_stride_q    = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
    args.batch_stride_k    = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
    args.batch_stride_knew = static_cast<int64_t>(nhead_k) * seqlen_knew * hdim_q;
    args.batch_stride_v    = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
    args.batch_stride_vnew = static_cast<int64_t>(nhead_k) * seqlen_knew * hdim_v;

    try
    {
        elapsed = g_dispatcher->run_fwd_appendkv(traits, args, nullptr);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "FMHA_ERR: %s\n", e.what());
        rc = -2;
        goto cleanup;
    }
    catch(...)
    {
        rc = -2;
        goto cleanup;
    }

    if(time_ms_out)
        *time_ms_out = elapsed;

cleanup:
    safe_hip_free(q_dev);
    safe_hip_free(k_dev);
    safe_hip_free(v_dev);
    safe_hip_free(knew_dev);
    safe_hip_free(vnew_dev);
    safe_hip_free(seqlen_k_dev);
    return rc;
}

// ---------------------------------------------------------------------------
// Batch Prefill: group-mode forward with paged KV cache
// ---------------------------------------------------------------------------
int fmha_dispatcher_run_batch_prefill(const void* q_host,
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
                                      int page_block_size,
                                      int is_v_rowmajor,
                                      const char* data_type_str,
                                      int has_lse,
                                      float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int in_bytes  = dtype_input_bytes(data_type_str);
    const int out_bytes = dtype_output_bytes(data_type_str);

    int rc                  = 0;
    const int64_t q_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * in_bytes;
    const int64_t o_bytes   = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * out_bytes;
    const int64_t lse_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * sizeof(float);
    float elapsed           = 0.0f;

    if(page_block_size <= 0)
        page_block_size = 64;
    const int pages_per_seq     = (seqlen_k + page_block_size - 1) / page_block_size;
    const int total_pages       = batch * pages_per_seq;
    const int64_t kv_page_bytes = static_cast<int64_t>(total_pages) * nhead_k * page_block_size *
                                  std::max(hdim_q, hdim_v) * in_bytes;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    void *lse_dev = nullptr, *seqstart_q_dev = nullptr;
    void *kv_indptr_dev = nullptr, *kv_page_indices_dev = nullptr, *kv_last_page_dev = nullptr;
    void* seqlen_k_dev = nullptr;

    fmha_batch_prefill_traits traits{};
    traits.hdim_q              = hdim_q;
    traits.hdim_v              = hdim_v;
    traits.data_type           = data_type_str ? data_type_str : "fp16";
    traits.is_group_mode       = true;
    traits.is_v_rowmajor       = (is_v_rowmajor != 0);
    traits.mask_type           = static_cast<mask_enum>(mask_type_int);
    traits.bias_type           = bias_enum::no_bias;
    traits.has_lse             = (has_lse != 0);
    traits.has_dropout         = false;
    traits.has_logits_soft_cap = false;
    traits.skip_min_seqlen_q   = false;
    traits.has_sink            = false;
    traits.qscale_type         = quant_scale_enum::no_scale;
    traits.kv_memory_layout    = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    traits.kv_lookup_table = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    traits.page_size       = page_block_size;

    // Declare all vectors before HIP_CHECK to avoid goto-over-init
    std::vector<int> seqstart_q(batch + 1);
    for(int b = 0; b <= batch; ++b)
        seqstart_q[b] = b * seqlen_q;
    std::vector<int> kv_indptr(batch + 1);
    for(int b = 0; b <= batch; ++b)
        kv_indptr[b] = b * pages_per_seq;
    std::vector<int> kv_page_indices(total_pages);
    for(int i = 0; i < total_pages; ++i)
        kv_page_indices[i] = i;
    std::vector<int> last_page(batch);
    for(int b = 0; b < batch; ++b)
        last_page[b] = seqlen_k - (pages_per_seq - 1) * page_block_size;
    std::vector<int> sk_vec(batch, seqlen_k);

    fmha_batch_prefill_args args{};

    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, kv_page_bytes));
    HIP_CHECK(hipMalloc(&v_dev, kv_page_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));

    HIP_CHECK(hipMalloc(&seqstart_q_dev, (batch + 1) * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        seqstart_q_dev, seqstart_q.data(), (batch + 1) * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&kv_indptr_dev, (batch + 1) * sizeof(int)));
    HIP_CHECK(hipMemcpy(
        kv_indptr_dev, kv_indptr.data(), (batch + 1) * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&kv_page_indices_dev, total_pages * sizeof(int)));
    HIP_CHECK(hipMemcpy(kv_page_indices_dev,
                        kv_page_indices.data(),
                        total_pages * sizeof(int),
                        hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&kv_last_page_dev, batch * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(kv_last_page_dev, last_page.data(), batch * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&seqlen_k_dev, batch * sizeof(int)));
    HIP_CHECK(hipMemcpy(seqlen_k_dev, sk_vec.data(), batch * sizeof(int), hipMemcpyHostToDevice));

    if(has_lse)
    {
        HIP_CHECK(hipMalloc(&lse_dev, lse_bytes));
        HIP_CHECK(hipMemset(lse_dev, 0, lse_bytes));
    }

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(k_dev, 0, kv_page_bytes));
    HIP_CHECK(hipMemset(v_dev, 0, kv_page_bytes));
    HIP_CHECK(hipMemset(o_dev, 0, o_bytes));

    args.q_ptr             = q_dev;
    args.k_ptr             = k_dev;
    args.v_ptr             = v_dev;
    args.bias_ptr          = nullptr;
    args.q_descale_ptr     = nullptr;
    args.k_descale_ptr     = nullptr;
    args.v_descale_ptr     = nullptr;
    args.rand_val_ptr      = nullptr;
    args.lse_ptr           = lse_dev;
    args.o_ptr             = o_dev;
    args.seqstart_q_ptr    = seqstart_q_dev;
    args.sink_ptr          = nullptr;
    args.seqlen_q          = seqlen_q;
    args.seqlen_k          = seqlen_k;
    args.batch             = batch;
    args.max_seqlen_q      = seqlen_q;
    args.hdim_q            = hdim_q;
    args.hdim_v            = hdim_v;
    args.nhead_q           = nhead_q;
    args.nhead_k           = nhead_k;
    args.num_total_pages   = total_pages;
    args.page_block_size   = page_block_size;
    args.kv_memory_layout  = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    args.kv_lookup_table   = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    args.kv_indptr         = kv_indptr_dev;
    args.kv_page_indices   = kv_page_indices_dev;
    args.kv_last_page_lens = kv_last_page_dev;
    args.seqlen_k_ptr      = seqlen_k_dev;
    args.batch_stride_block_table = pages_per_seq;
    args.scale_s                  = scale;
    args.scale_p                  = 1.0f;
    args.scale_o                  = 1.0f;
    args.logits_soft_cap          = 0.0f;

    // Group-mode strides: [total_tokens, nhead, hdim]
    args.stride_q             = nhead_q * hdim_q;
    args.stride_k             = hdim_q;
    args.stride_v             = hdim_v;
    args.stride_bias          = 0;
    args.stride_randval       = 0;
    args.stride_o             = nhead_q * hdim_v;
    args.nhead_stride_q       = hdim_q;
    args.nhead_stride_k       = static_cast<int64_t>(page_block_size) * hdim_q;
    args.nhead_stride_v       = static_cast<int64_t>(page_block_size) * hdim_v;
    args.nhead_stride_bias    = 0;
    args.nhead_stride_randval = 0;
    args.nhead_stride_lse     = seqlen_q;
    args.nhead_stride_o       = hdim_v;
    args.batch_stride_q       = 0;
    args.batch_stride_k       = static_cast<int64_t>(nhead_k) * page_block_size * hdim_q;
    args.batch_stride_v       = static_cast<int64_t>(nhead_k) * page_block_size * hdim_v;
    args.batch_stride_bias    = 0;
    args.batch_stride_randval = 0;
    args.batch_stride_lse     = static_cast<int64_t>(nhead_q) * seqlen_q;
    args.batch_stride_o       = 0;
    args.window_size_left     = -1;
    args.window_size_right    = -1;
    args.sink_size            = 0;
    args.mask_type            = mask_type_int;

    try
    {
        elapsed = g_dispatcher->run_batch_prefill(traits, args, nullptr);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "FMHA_ERR: %s\n", e.what());
        rc = -2;
        goto cleanup;
    }
    catch(...)
    {
        rc = -2;
        goto cleanup;
    }

    {
        hipError_t cpy = hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost);
        if(cpy != hipSuccess)
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
    safe_hip_free(seqstart_q_dev);
    safe_hip_free(kv_indptr_dev);
    safe_hip_free(kv_page_indices_dev);
    safe_hip_free(kv_last_page_dev);
    safe_hip_free(seqlen_k_dev);
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
