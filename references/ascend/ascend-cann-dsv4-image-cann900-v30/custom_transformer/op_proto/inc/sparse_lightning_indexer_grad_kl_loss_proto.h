#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_PROTO_H_
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseLightningIndexerGradKLLoss)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(query_index, ge::TensorType::ALL())
    .INPUT(key_index, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(softmax_max, ge::TensorType::ALL())
    .OPTIONAL_INPUT(softmax_sum, ge::TensorType::ALL())
    .OPTIONAL_INPUT(query_rope, ge::TensorType::ALL())
    .OPTIONAL_INPUT(key_rope, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_key, ge::TensorType::ALL())
    .OUTPUT(d_query_index, ge::TensorType::ALL())
    .OUTPUT(d_key_index, ge::TensorType::ALL())
    .OUTPUT(d_weight, ge::TensorType::ALL())
    .OUTPUT(loss, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(layout, String, "TND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(deterministic, Bool, false)
    .ATTR(cmp_ratio, Int, 0)
    .OP_END_FACTORY_REG(SparseLightningIndexerGradKLLoss);

}

#endif
