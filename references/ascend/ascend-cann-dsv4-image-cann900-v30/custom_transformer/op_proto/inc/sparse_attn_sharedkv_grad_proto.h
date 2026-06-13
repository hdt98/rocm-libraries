#ifndef SPARSE_ATTN_SHAREDKV_GRAD_PROTO_H_
#define SPARSE_ATTN_SHAREDKV_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseAttnSharedkvGrad)
    .INPUT(query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(d_out, ge::TensorType::ALL())
    .OPTIONAL_INPUT(out, ge::TensorType::ALL())
    .OPTIONAL_INPUT(lse, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sinks, ge::TensorType::ALL())
    .OUTPUT(d_query, ge::TensorType::ALL())
    .OUTPUT(d_ori_kv, ge::TensorType::ALL())
    .OUTPUT(d_cmp_kv, ge::TensorType::ALL())
    .OUTPUT(d_sinks, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(cmp_ratio, Int)
    .REQUIRED_ATTR(ori_mask_mode, Int)
    .REQUIRED_ATTR(cmp_mask_mode, Int)
    .ATTR(ori_win_left, Int, 128)
    .ATTR(ori_win_right, Int, 0)
    .ATTR(layout, String, "BSND")
    .OP_END_FACTORY_REG(SparseAttnSharedkvGrad);

}

#endif
