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
    ck_tile::dispatcher::fmha_fwd_traits traits{};
    traits.hdim_q = static_cast<int>(params_.hdim_q);
    traits.hdim_v = static_cast<int>(params_.hdim_v);
    traits.data_type = params_.data_type;
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type = static_cast<decltype(traits.mask_type)>(params_.mask_type);
    traits.bias_type = static_cast<decltype(traits.bias_type)>(params_.bias_type);
    traits.has_lse = params_.has_lse;
    traits.has_dropout = params_.has_dropout;

    ck_tile::dispatcher::fmha_fwd_args args{};
    args.q_ptr = findBuffer(params_.q_uid, deviceBuffers, numDeviceBuffers);
    args.k_ptr = findBuffer(params_.k_uid, deviceBuffers, numDeviceBuffers);
    args.v_ptr = findBuffer(params_.v_uid, deviceBuffers, numDeviceBuffers);
    args.o_ptr = findBuffer(params_.o_uid, deviceBuffers, numDeviceBuffers);

    if (params_.bias_uid > 0)
        args.bias_ptr = findBuffer(params_.bias_uid, deviceBuffers, numDeviceBuffers);
    if (params_.lse_uid > 0)
        args.lse_ptr = findBuffer(params_.lse_uid, deviceBuffers, numDeviceBuffers);

    args.seqlen_q = static_cast<int>(params_.seqlen_q);
    args.seqlen_k = static_cast<int>(params_.seqlen_k);
    args.batch = static_cast<int>(params_.batch);
    args.max_seqlen_q = static_cast<int>(params_.seqlen_q);
    args.nhead_q = static_cast<int>(params_.nhead_q);
    args.nhead_k = static_cast<int>(params_.nhead_k);
    args.scale_s = params_.scale;

    // BHSD stride convention
    const auto dq = params_.hdim_q;
    const auto dv = params_.hdim_v;
    const auto sq = params_.seqlen_q;
    const auto sk = params_.seqlen_k;
    const auto hq = params_.nhead_q;
    const auto hk = params_.nhead_k;

    if (params_.is_bhsd_layout) {
        args.stride_q = dq;
        args.nhead_stride_q = sq * dq;
        args.batch_stride_q = hq * sq * dq;

        args.stride_k = dq;
        args.nhead_stride_k = sk * dq;
        args.batch_stride_k = hk * sk * dq;

        args.stride_v = dv;
        args.nhead_stride_v = sk * dv;
        args.batch_stride_v = hk * sk * dv;

        args.stride_o = dv;
        args.nhead_stride_o = sq * dv;
        args.batch_stride_o = hq * sq * dv;
    } else {
        // BSHD layout
        args.stride_q = dq;
        args.nhead_stride_q = dq;
        args.batch_stride_q = sq * hq * dq;

        args.stride_k = dq;
        args.nhead_stride_k = dq;
        args.batch_stride_k = sk * hk * dq;

        args.stride_v = dv;
        args.nhead_stride_v = dv;
        args.batch_stride_v = sk * hk * dv;

        args.stride_o = dv;
        args.nhead_stride_o = dv;
        args.batch_stride_o = sq * hq * dv;
    }

    if (params_.has_lse) {
        args.stride_lse = 1;
        args.nhead_stride_lse = sq;
        args.batch_stride_lse = hq * sq;
    }

    if (params_.bias_uid > 0) {
        args.stride_bias = sk;
        args.nhead_stride_bias = sq * sk;
        args.batch_stride_bias = hq * sq * sk;
    }

    args.window_size_left = static_cast<int>(params_.window_left);
    args.window_size_right = static_cast<int>(params_.window_right);

    auto invocation = ck_tile::dispatcher::FmhaInvocation::make(std::move(traits), std::move(args));

    handle.dispatcher()->run(invocation, handle.getStream());
}

}  // namespace ck_fmha_plugin
