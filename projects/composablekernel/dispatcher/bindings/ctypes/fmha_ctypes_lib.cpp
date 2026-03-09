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

#define HIP_CHECK(call)        \
    {                          \
        hipError_t err = call; \
        if(err != hipSuccess)  \
            return -1;         \
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
                            float* time_ms_out)
{
    if(!g_initialized)
        return -1;

    const int64_t q_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_q * 2;
    const int64_t k_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_q * 2;
    const int64_t v_bytes = static_cast<int64_t>(batch) * nhead_k * seqlen_k * hdim_v * 2;
    const int64_t o_bytes = static_cast<int64_t>(batch) * nhead_q * seqlen_q * hdim_v * 2;

    void *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *o_dev = nullptr;
    HIP_CHECK(hipMalloc(&q_dev, q_bytes));
    HIP_CHECK(hipMalloc(&k_dev, k_bytes));
    HIP_CHECK(hipMalloc(&v_dev, v_bytes));
    HIP_CHECK(hipMalloc(&o_dev, o_bytes));

    HIP_CHECK(hipMemcpy(q_dev, q_host, q_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev, k_host, k_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev, v_host, v_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(o_dev, 0, o_bytes));

    fmha_fwd_traits traits{};
    traits.hdim_q        = hdim_q;
    traits.hdim_v        = hdim_v;
    traits.data_type     = "fp16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;
    traits.bias_type     = bias_enum::no_bias;
    traits.has_lse       = false;
    traits.has_dropout   = false;
    traits.qscale_type   = quant_scale_enum::no_scale;

    fmha_fwd_args args{};
    args.q_ptr                      = q_dev;
    args.k_ptr                      = k_dev;
    args.v_ptr                      = v_dev;
    args.o_ptr                      = o_dev;
    args.bias_ptr                   = nullptr;
    args.q_descale_ptr              = nullptr;
    args.k_descale_ptr              = nullptr;
    args.v_descale_ptr              = nullptr;
    args.rand_val_ptr               = nullptr;
    args.lse_ptr                    = nullptr;
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

    args.stride_q               = hdim_q;
    args.stride_k               = hdim_q;
    args.stride_v               = hdim_v;
    args.stride_bias            = 0;
    args.stride_randval         = 0;
    args.stride_o               = hdim_v;
    args.nhead_stride_q         = seqlen_q * hdim_q;
    args.nhead_stride_k         = seqlen_k * hdim_q;
    args.nhead_stride_v         = seqlen_k * hdim_v;
    args.nhead_stride_bias      = 0;
    args.nhead_stride_randval   = 0;
    args.nhead_stride_lse       = 0;
    args.nhead_stride_o         = seqlen_q * hdim_v;
    args.nhead_stride_q_descale = 0;
    args.nhead_stride_k_descale = 0;
    args.nhead_stride_v_descale = 0;
    args.batch_stride_q         = nhead_q * seqlen_q * hdim_q;
    args.batch_stride_k         = nhead_k * seqlen_k * hdim_q;
    args.batch_stride_v         = nhead_k * seqlen_k * hdim_v;
    args.batch_stride_bias      = 0;
    args.batch_stride_randval   = 0;
    args.batch_stride_lse       = 0;
    args.batch_stride_o         = nhead_q * seqlen_q * hdim_v;
    args.batch_stride_q_descale = 0;
    args.batch_stride_k_descale = 0;
    args.batch_stride_v_descale = 0;

    args.window_size_left    = -1;
    args.window_size_right   = -1;
    args.sink_size           = 0;
    args.mask_type           = 0;
    args.min_seqlen_q        = 0;
    args.p_drop              = 0.0f;
    args.s_randval           = false;
    args.drop_seed_offset    = std::make_pair(uint64_t(0), uint64_t(0));
    args.block_scale_size_q  = 0;
    args.block_scale_size_kv = 0;

    float elapsed = 0.0f;
    try
    {
        elapsed = g_dispatcher->run_fwd(traits, args, nullptr);
    }
    catch(...)
    {
        hipFree(q_dev);
        hipFree(k_dev);
        hipFree(v_dev);
        hipFree(o_dev);
        return -2;
    }

    HIP_CHECK(hipMemcpy(o_host, o_dev, o_bytes, hipMemcpyDeviceToHost));

    hipFree(q_dev);
    hipFree(k_dev);
    hipFree(v_dev);
    hipFree(o_dev);

    if(time_ms_out)
        *time_ms_out = elapsed;

    return 0;
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
