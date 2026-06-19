#!/usr/bin/env bash

if [ ${#COMPRESS_RATIOS[@]} -eq 0 ]; then
  COMPRESS_RATIOS=(0 0 4 128 4 128 4 128 4 128 4 128)
fi

MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
MOE_FLEX_DISPATCHER_BACKEND="${MOE_FLEX_DISPATCHER_BACKEND:-}"
MOE_DISPATCHER_ARGS=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}")
if [ "${MOE_TOKEN_DISPATCHER_TYPE}" = "flex" ] && [ -n "${MOE_FLEX_DISPATCHER_BACKEND}" ]; then
  MOE_DISPATCHER_ARGS+=(--moe-flex-dispatcher-backend "${MOE_FLEX_DISPATCHER_BACKEND}")
fi

read -r -a SPEC_ARGS <<< "${MEGATRON_DSV4_SPEC:-miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec}"
SPEC_ARGS=(--spec "${SPEC_ARGS[@]}")

MODEL_ARGS=(
  --disable-bias-linear
  --num-layers 12
  --hidden-size 4096
  --ffn-hidden-size 2048
  --num-attention-heads 64
  --normalization RMSNorm
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 129280
  --make-vocab-size-divisible-by 1
  --hidden-dropout 0.0
  --attention-dropout 0.0
  --multi-latent-attention
  --q-lora-rank 1024
  --kv-lora-rank 512
  --qk-head-dim 512
  --qk-pos-emb-head-dim 64
  --v-head-dim 512
  --qk-layernorm
  --rotary-scaling-factor "${ROTARY_SCALING_FACTOR:-16}"
  --rotary-base 10000
  --original-max-position-embeddings 65536
  --beta-fast 32
  --beta-slow 1
  --attention-softmax-in-fp32
  --no-rope-fusion
  --num-experts 256
  --moe-layer-freq "[1,1,1,1,1,1,1,1,1,1,1,1]"
  --moe-ffn-hidden-size 2048
  --moe-router-topk 6
  --moe-shared-expert-intermediate-size 2048
  --moe-router-pre-softmax
  --moe-router-score-function sqrtsoftplus
  --moe-router-enable-expert-bias
  --moe-router-load-balancing-type seq_aux_loss
  "${MOE_DISPATCHER_ARGS[@]}"
  --moe-aux-loss-coeff 0
  --moe-grouped-gemm
  --moe-router-topk-scaling-factor 1.5
  --moe-router-dtype fp32
  --experimental-attention-variant dsv4
  --dsv4-hc-mult "${DSV4_HC_MULT:-4}"
  --dsv4-hc-sinkhorn-iters "${DSV4_HC_SINKHORN_ITERS:-20}"
  --dsv4-compress-ratios "${COMPRESS_RATIOS[@]}"
  --dsv4-compress-rope-theta 160000
  --dsv4-o-groups 8
  --dsv4-o-lora-rank 1024
  --dsv4-n-hash-layers "${DSV4_N_HASH_LAYERS:-3}"
  --dsv4-window-size "${DSV4_WINDOW_SIZE:-128}"
  --dsa-indexer-n-heads 64
  --dsa-indexer-head-dim 128
  --dsa-indexer-topk "${DSA_INDEXER_TOPK:-512}"
  --activation-func-clamp-value 10
  --no-bias-swiglu-fusion
  --no-activation-func-clamp-shared-expert
  "${SPEC_ARGS[@]}"
)
