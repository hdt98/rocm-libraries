// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaFwdPlan.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <stdexcept>

namespace ck_fmha_plugin {

CkFmhaFwdPlan::CkFmhaFwdPlan(ParsedFwdParams params,
                             ck_tile::dispatcher::FmhaExecutionPlan exec_plan)
    : params_(std::move(params)), exec_plan_(std::move(exec_plan)) {}

size_t CkFmhaFwdPlan::getWorkspaceSize(const CkFmhaHandle&) const {
    return 0;
}

void* CkFmhaFwdPlan::findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs,
                                uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (bufs[i].uid == uid) return bufs[i].ptr;
    }
    return nullptr;
}

void CkFmhaFwdPlan::execute(const CkFmhaHandle& handle,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers, void*) const {
    const int batch = static_cast<int>(params_.batch);
    const int nhead_q = static_cast<int>(params_.nhead_q);
    const int nhead_k = static_cast<int>(params_.nhead_k);
    const int seqlen_q = static_cast<int>(params_.seqlen_q);
    const int seqlen_k = static_cast<int>(params_.seqlen_k);
    const int hdim_q = static_cast<int>(params_.hdim_q);
    const int hdim_v = static_cast<int>(params_.hdim_v);

    fmha_fwd_traits traits{};
    traits.hdim_q = hdim_q;
    traits.hdim_v = hdim_v;
    traits.data_type = params_.data_type;
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type = static_cast<mask_enum>(params_.mask_type);
    traits.bias_type = static_cast<bias_enum>(params_.bias_type);
    traits.has_lse = params_.has_lse;
    traits.has_dropout = params_.has_dropout;
    traits.qscale_type = quant_scale_enum::no_scale;
    traits.skip_min_seqlen_q = false;
    traits.has_sink = false;

    fmha_fwd_args args{};

    args.q_ptr = findBuffer(params_.q_uid, deviceBuffers, numDeviceBuffers);
    args.k_ptr = findBuffer(params_.k_uid, deviceBuffers, numDeviceBuffers);
    args.v_ptr = findBuffer(params_.v_uid, deviceBuffers, numDeviceBuffers);
    args.o_ptr = findBuffer(params_.o_uid, deviceBuffers, numDeviceBuffers);

    if (params_.bias_uid > 0)
        args.bias_ptr = findBuffer(params_.bias_uid, deviceBuffers, numDeviceBuffers);
    if (params_.lse_uid > 0)
        args.lse_ptr = findBuffer(params_.lse_uid, deviceBuffers, numDeviceBuffers);

    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.batch = batch;
    args.max_seqlen_q = seqlen_q;
    args.hdim_q = hdim_q;
    args.hdim_v = hdim_v;
    args.nhead_q = nhead_q;
    args.nhead_k = nhead_k;
    args.scale_s = params_.scale;

    // Strides: exactly matching fmha_ctypes_lib.cpp reference implementation
    if (params_.is_bhsd_layout) {
        // BHSD: [batch, head, seq, dim]
        args.stride_q = hdim_q;
        args.stride_k = hdim_q;
        args.stride_v = hdim_v;
        args.stride_o = hdim_v;
        args.nhead_stride_q = seqlen_q * hdim_q;
        args.nhead_stride_k = seqlen_k * hdim_q;
        args.nhead_stride_v = seqlen_k * hdim_v;
        args.nhead_stride_o = seqlen_q * hdim_v;
        args.batch_stride_q = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_q;
        args.batch_stride_k = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_q;
        args.batch_stride_v = static_cast<int64_t>(nhead_k) * seqlen_k * hdim_v;
        args.batch_stride_o = static_cast<int64_t>(nhead_q) * seqlen_q * hdim_v;
    } else {
        // BSHD: [batch, seq, head, dim]
        args.stride_q = nhead_q * hdim_q;
        args.stride_k = nhead_k * hdim_q;
        args.stride_v = nhead_k * hdim_v;
        args.stride_o = nhead_q * hdim_v;
        args.nhead_stride_q = hdim_q;
        args.nhead_stride_k = hdim_q;
        args.nhead_stride_v = hdim_v;
        args.nhead_stride_o = hdim_v;
        args.batch_stride_q = static_cast<int64_t>(seqlen_q) * nhead_q * hdim_q;
        args.batch_stride_k = static_cast<int64_t>(seqlen_k) * nhead_k * hdim_q;
        args.batch_stride_v = static_cast<int64_t>(seqlen_k) * nhead_k * hdim_v;
        args.batch_stride_o = static_cast<int64_t>(seqlen_q) * nhead_q * hdim_v;
    }

    if (params_.has_lse) {
        args.nhead_stride_lse = seqlen_q;
        args.batch_stride_lse = static_cast<int64_t>(nhead_q) * seqlen_q;
    }

    if (params_.bias_uid > 0) {
        args.stride_bias = seqlen_k;
        args.nhead_stride_bias = static_cast<int64_t>(seqlen_q) * seqlen_k;
        args.batch_stride_bias = static_cast<int64_t>(nhead_q) * seqlen_q * seqlen_k;
    }

    args.window_size_left = static_cast<int>(params_.window_left);
    args.window_size_right = static_cast<int>(params_.window_right);

    auto invocation = ck_tile::dispatcher::FmhaInvocation::make(std::move(traits), std::move(args));

    handle.dispatcher()->run(invocation, handle.getStream());
}

}  // namespace ck_fmha_plugin
