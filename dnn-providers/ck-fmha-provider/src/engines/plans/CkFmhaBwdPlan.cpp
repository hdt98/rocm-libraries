// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaBwdPlan.hpp"

#include <cstring>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <stdexcept>

namespace ck_fmha_plugin {

CkFmhaBwdPlan::CkFmhaBwdPlan(ParsedBwdParams params,
                             ck_tile::dispatcher::FmhaExecutionPlan exec_plan,
                             ck_tile::dispatcher::FmhaBwdWorkspaceInfo ws_info)
    : params_(std::move(params)), exec_plan_(std::move(exec_plan)), ws_info_(ws_info) {}

size_t CkFmhaBwdPlan::getWorkspaceSize(const CkFmhaHandle&) const {
    return ws_info_.total_bytes;
}

void* CkFmhaBwdPlan::findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs,
                                uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (bufs[i].uid == uid) return bufs[i].ptr;
    }
    return nullptr;
}

void CkFmhaBwdPlan::execute(const CkFmhaHandle& handle,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers, void* workspace) const {
    if (workspace == nullptr && ws_info_.total_bytes > 0) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "CkFmhaBwdPlan: null workspace for backward pass");
    }

    ck_tile::dispatcher::fmha_bwd_traits traits{};
    traits.hdim_q = static_cast<int>(params_.hdim_q);
    traits.hdim_v = static_cast<int>(params_.hdim_v);
    traits.data_type = params_.data_type;
    traits.is_group_mode = false;
    traits.mask_type = static_cast<decltype(traits.mask_type)>(params_.mask_type);
    traits.bias_type = static_cast<decltype(traits.bias_type)>(params_.bias_type);
    traits.has_dbias = params_.has_dbias;
    traits.has_dropout = params_.has_dropout;
    traits.is_store_randval = false;
    traits.is_deterministic = false;

    // Populate runtime shape fields carried in bwd_traits
    traits.seqlen_q = static_cast<int>(params_.seqlen_q);
    traits.seqlen_k = static_cast<int>(params_.seqlen_k);
    traits.batch = static_cast<int>(params_.batch);
    traits.max_seqlen_q = static_cast<int>(params_.seqlen_q);
    traits.max_seqlen_k = static_cast<int>(params_.seqlen_k);
    traits.nhead_q = static_cast<int>(params_.nhead_q);
    traits.nhead_k = static_cast<int>(params_.nhead_k);

    ck_tile::dispatcher::fmha_bwd_args args{};
    args.q_ptr = findBuffer(params_.q_uid, deviceBuffers, numDeviceBuffers);
    args.k_ptr = findBuffer(params_.k_uid, deviceBuffers, numDeviceBuffers);
    args.v_ptr = findBuffer(params_.v_uid, deviceBuffers, numDeviceBuffers);
    args.o_ptr = findBuffer(params_.o_uid, deviceBuffers, numDeviceBuffers);
    args.do_ptr = findBuffer(params_.do_uid, deviceBuffers, numDeviceBuffers);
    args.lse_ptr = findBuffer(params_.stats_uid, deviceBuffers, numDeviceBuffers);
    args.dq_ptr = findBuffer(params_.dq_uid, deviceBuffers, numDeviceBuffers);
    args.dk_ptr = findBuffer(params_.dk_uid, deviceBuffers, numDeviceBuffers);
    args.dv_ptr = findBuffer(params_.dv_uid, deviceBuffers, numDeviceBuffers);

    if (params_.bias_uid > 0)
        args.bias_ptr = findBuffer(params_.bias_uid, deviceBuffers, numDeviceBuffers);
    if (params_.dbias_uid > 0)
        args.dbias_ptr = findBuffer(params_.dbias_uid, deviceBuffers, numDeviceBuffers);

    // Suballocate workspace into d_ptr, dq_acc_ptr, and optionally rand_val_ptr
    auto* ws_base = static_cast<char*>(workspace);
    args.d_ptr = ws_base + ws_info_.d_offset;
    args.dq_acc_ptr = ws_base + ws_info_.dq_acc_offset;
    if (ws_info_.rand_val_bytes > 0) args.rand_val_ptr = ws_base + ws_info_.rand_val_offset;

    args.seqlen_q = static_cast<int>(params_.seqlen_q);
    args.seqlen_k = static_cast<int>(params_.seqlen_k);
    args.batch = static_cast<int>(params_.batch);
    args.max_seqlen_q = static_cast<int>(params_.seqlen_q);
    args.max_seqlen_k = static_cast<int>(params_.seqlen_k);
    args.nhead_q = static_cast<int>(params_.nhead_q);
    args.nhead_k = static_cast<int>(params_.nhead_k);
    args.scale_s = params_.scale;

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

        args.stride_do = dv;
        args.nhead_stride_do = sq * dv;
        args.batch_stride_do = hq * sq * dv;

        args.stride_dq = dq;
        args.nhead_stride_dq = sq * dq;
        args.batch_stride_dq = hq * sq * dq;

        args.stride_dk = dq;
        args.nhead_stride_dk = sk * dq;
        args.batch_stride_dk = hk * sk * dq;

        args.stride_dv = dv;
        args.nhead_stride_dv = sk * dv;
        args.batch_stride_dv = hk * sk * dv;
    } else {
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

        args.stride_do = dv;
        args.nhead_stride_do = dv;
        args.batch_stride_do = sq * hq * dv;

        args.stride_dq = dq;
        args.nhead_stride_dq = dq;
        args.batch_stride_dq = sq * hq * dq;

        args.stride_dk = dq;
        args.nhead_stride_dk = dq;
        args.batch_stride_dk = sk * hk * dq;

        args.stride_dv = dv;
        args.nhead_stride_dv = dv;
        args.batch_stride_dv = sk * hk * dv;
    }

    // LSE strides (always dense 1D per head per batch)
    args.stride_lse = 1;
    args.nhead_stride_lse = sq;
    args.batch_stride_lse = hq * sq;

    if (params_.bias_uid > 0) {
        args.stride_bias = sk;
        args.nhead_stride_bias = sq * sk;
        args.batch_stride_bias = hq * sq * sk;
    }

    if (params_.has_dbias) {
        args.stride_dbias = sk;
        args.nhead_stride_dbias = sq * sk;
        args.batch_stride_dbias = hq * sq * sk;
    }

    args.window_size_left = static_cast<int>(params_.window_left);
    args.window_size_right = static_cast<int>(params_.window_right);

    // No split-K in dense backward
    args.split_stride_dq_acc = 0;

    // Dropout inverse scaling
    if (params_.has_dropout)
        args.p_undrop = 1.0f;  // p_drop is set from seed/offset at runtime
    else
        args.p_undrop = 1.0f;

    auto invocation = ck_tile::dispatcher::FmhaInvocation::make(std::move(traits), std::move(args));

    handle.dispatcher()->run(invocation, handle.getStream());
}

}  // namespace ck_fmha_plugin
