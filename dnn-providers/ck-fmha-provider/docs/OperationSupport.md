# CK FMHA Provider: Operation Support Matrix

## Forward SDPA (`SdpaAttributes`)

### Data Types

| Type | Status |
|------|--------|
| fp16 (HALF) | Supported |
| bf16 (BFLOAT16) | Supported |
| fp8 (FP8_E4M3) | Not yet (schema exists, kernel support varies by arch) |

### Layouts

| Layout | Detection Method | Status |
|--------|-----------------|--------|
| BHSD [Batch, Head, Seq, Dim] | `stride[1] > stride[2]` | Supported |
| BSHD [Batch, Seq, Head, Dim] | `stride[1] <= stride[2]` | Supported |

### Head Dimensions

| hdim_q | hdim_v | Status |
|--------|--------|--------|
| 32 | 32 | Supported |
| 64 | 64 | Supported |
| 128 | 128 | Supported |
| 256 | 256 | Supported |
| 128 | 64 | Supported (asymmetric) |
| 16 | * | Supported (small hdim_q, any hdim_v) |

### Attention Features

| Feature | hipDNN Schema Field | CK Mapping | Status |
|---------|-------------------|------------|--------|
| No mask | (default) | `mask_type=0` | Supported |
| Causal top-left | `causal_mask=true` | `mask_type=1` | Supported |
| Causal bottom-right | `causal_mask_bottom_right=true` | `mask_type=2` | Supported |
| Window mask | `left_bound` + `right_bound` | `mask_type=3` | Supported |
| No bias | (default) | `bias_type=0` | Supported |
| Elementwise bias | `attn_mask_tensor_uid` present | `bias_type=1` | Supported |
| ALiBi | `alibi_mask=true` | `bias_type=2` | Supported |
| Dropout | `seed_tensor_uid` + `offset_tensor_uid` | `has_dropout=true` | Supported |
| LSE output | `stats_tensor_uid` present | `has_lse=true` | Supported |
| GQA | `nhead_q > nhead_k` | Runtime via strides | Supported |

### Not Supported (Forward)

- Paged KV attention (`page_table_k/v_tensor_uid`)
- Split-KV attention
- Append-KV attention
- Batch prefill
- FP8 quantization (`descale_*` / `scale_*` tensor UIDs)
- `implementation=COMPOSITE` mode
- Block mask (`block_mask_tensor_uid`)
- Sink tokens (`sink_token_tensor_uid`)

## Backward SDPA (`SdpaBackwardAttributes`)

### Data Types

| Type | Status |
|------|--------|
| fp16 | Supported |
| bf16 | Supported |

### Attention Features (Backward)

| Feature | Status |
|---------|--------|
| Gradient dQ, dK, dV | Supported |
| Gradient dBias | Supported (`dbias_tensor_uid` present) |
| Causal masks | Supported (same as forward) |
| Window masks | Supported |
| Elementwise bias | Supported |
| ALiBi bias | Supported |
| Dropout | Supported |
| Deterministic mode | Defaulted to false (schema gap) |
| Store randval | Defaulted to false (schema gap) |
| Logits soft cap | Not supported (schema gap) |

### Backward Workspace

The backward pass requires scratch memory for intermediate computations:

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_ptr` | `B * Hq * Sq * 4` bytes | `d = rowsum(dO * O)` in fp32 |
| `dq_acc_ptr` | `B * Hq * Sq * Dq * 4` bytes | fp32 accumulator for dQ |
| `rand_val_ptr` | `B * Hq * Sq * Sk * 1` byte | Only when `is_store_randval=true` |

Suballocation uses 256-byte alignment between regions.

## Architecture Support

| Architecture | Forward | Backward | Notes |
|-------------|---------|----------|-------|
| gfx90a (MI200) | ✓ | ✓ | CDNA2 |
| gfx942 (MI300) | ✓ | ✓ | CDNA3, primary target |
| gfx950 | ✓ | ✓ | CDNA3+ |
| gfx1100 (Navi31) | ✓ | Limited | RDNA3 |
| gfx1201 | ✓ | Limited | RDNA4 |

## Schema Gap Summary

These CK fields have no hipDNN backward schema representation yet
(see RFC Appendix A):

| CK Field | Default Applied | Impact |
|----------|----------------|--------|
| `is_deterministic` | `false` | Non-deterministic backward accumulation |
| `is_store_randval` | `false` | Cannot debug dropout reproducibility |
| `logits_soft_cap` | `0.0` | Cannot use Gemma-2 style logit clamping |
| FP8 descale/scale | N/A | No FP8 backward |
