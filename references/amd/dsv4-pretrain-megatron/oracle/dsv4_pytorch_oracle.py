#!/usr/bin/env python3
"""Fixed-batch PyTorch logprob oracle for the reduced DeepSeek V4 shape.

The full 12-layer, 256-expert shape is too large for a normal eager checkpoint.
This oracle therefore uses deterministic virtual weights. Every tensor is
materialized from `(weight_seed, tensor_name, shape)` only when needed, and MoE
expert weights are generated only for experts selected by the fixed batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DSV4Config:
    layers: int = 12
    hidden_dim: int = 4096
    vocab_size: int = 129280
    attention_heads: int = 64
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 448
    v_head_dim: int = 512
    moe_experts: int = 256
    moe_hidden: int = 2048
    router_top_k: int = 6
    seq_len: int = 4096
    compress_ratios: tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128)
    router_topk_scaling_factor: float = 1.5
    shared_expert_hidden: int = 2048
    rotary_base: float = 10000.0
    weight_seed: int = 1234
    batch_seed: int = 2027

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def _seed(base_seed: int, name: str, shape: Iterable[int]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode())
    h.update(b":")
    h.update(name.encode())
    h.update(b":")
    h.update(",".join(str(dim) for dim in shape).encode())
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


class VirtualWeights:
    def __init__(self, seed: int, device: torch.device, dtype: torch.dtype):
        self.seed = seed
        self.device = device
        self.dtype = dtype

    def tensor(self, name: str, shape: tuple[int, ...], scale: float) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_seed(self.seed, name, shape))
        value = torch.randn(shape, generator=gen, dtype=torch.float32) * scale
        return value.to(device=self.device, dtype=self.dtype)

    def linear(self, x: torch.Tensor, name: str, out_features: int, scale: float | None = None) -> torch.Tensor:
        in_features = x.shape[-1]
        if scale is None:
            scale = in_features**-0.5
        weight = self.tensor(f"{name}.weight", (in_features, out_features), scale)
        return x @ weight

    def norm_weight(self, name: str, hidden_dim: int) -> torch.Tensor:
        return torch.ones(hidden_dim, device=self.device, dtype=self.dtype)

    def embedding_rows(self, token_ids: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        flat = token_ids.reshape(-1)
        unique, inverse = torch.unique(flat.cpu(), sorted=True, return_inverse=True)
        rows = []
        scale = hidden_dim**-0.5
        for token in unique.tolist():
            rows.append(self.tensor(f"tok_embeddings.{token}", (hidden_dim,), scale))
        table = torch.stack(rows, dim=0)
        embedded = table[inverse.to(self.device)]
        return embedded.reshape(*token_ids.shape, hidden_dim)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def make_fixed_batch(config: DSV4Config, batch_size: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(config.batch_seed)
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.seq_len),
        generator=gen,
        dtype=torch.long,
    )
    return tokens.to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    seq_len = x.shape[1]
    dim = x.shape[-1]
    positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.einsum("s,d->sd", positions, freqs)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).view(1, seq_len, 1, dim)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).view(1, seq_len, 1, dim)
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


def attention_layer(x: torch.Tensor, layer: int, config: DSV4Config, weights: VirtualWeights) -> torch.Tensor:
    heads = config.attention_heads
    q = weights.linear(x, f"layers.{layer}.attention.wq_a", config.q_lora_rank)
    q = rms_norm(q, weights.norm_weight(f"layers.{layer}.attention.q_norm", config.q_lora_rank))
    q = weights.linear(q, f"layers.{layer}.attention.wq_b", heads * config.qk_head_dim)
    q = q.view(x.shape[0], x.shape[1], heads, config.qk_head_dim)
    q_nope, q_pe = torch.split(q, [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1)
    q = torch.cat([q_nope, apply_rope(q_pe, config.rotary_base)], dim=-1)

    kv = weights.linear(
        x,
        f"layers.{layer}.attention.wkv_a",
        config.kv_lora_rank + config.qk_rope_head_dim,
    )
    kv_latent, k_pe = torch.split(kv, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)
    kv_latent = rms_norm(
        kv_latent,
        weights.norm_weight(f"layers.{layer}.attention.kv_norm", config.kv_lora_rank),
    )
    kv = weights.linear(
        kv_latent,
        f"layers.{layer}.attention.wkv_b",
        heads * (config.qk_nope_head_dim + config.v_head_dim),
    )
    kv = kv.view(x.shape[0], x.shape[1], heads, config.qk_nope_head_dim + config.v_head_dim)
    k_nope, v = torch.split(kv, [config.qk_nope_head_dim, config.v_head_dim], dim=-1)
    k_pe = apply_rope(k_pe.unsqueeze(2), config.rotary_base).expand(-1, -1, heads, -1)
    k = torch.cat([k_nope, k_pe], dim=-1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=config.qk_head_dim**-0.5)
    out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], heads * config.v_head_dim)
    return weights.linear(out, f"layers.{layer}.attention.wo", config.hidden_dim)


def expert_forward(x: torch.Tensor, layer: int, expert: int, config: DSV4Config, weights: VirtualWeights) -> torch.Tensor:
    h = weights.linear(x, f"layers.{layer}.moe.experts.{expert}.w1", config.moe_hidden)
    g = weights.linear(x, f"layers.{layer}.moe.experts.{expert}.w3", config.moe_hidden)
    h = F.silu(h.float()).to(x.dtype) * g
    return weights.linear(h, f"layers.{layer}.moe.experts.{expert}.w2", config.hidden_dim)


def shared_expert_forward(x: torch.Tensor, layer: int, config: DSV4Config, weights: VirtualWeights) -> torch.Tensor:
    h = weights.linear(x, f"layers.{layer}.moe.shared.w1", config.shared_expert_hidden)
    g = weights.linear(x, f"layers.{layer}.moe.shared.w3", config.shared_expert_hidden)
    h = F.silu(h.float()).to(x.dtype) * g
    return weights.linear(h, f"layers.{layer}.moe.shared.w2", config.hidden_dim)


def moe_layer(x: torch.Tensor, layer: int, config: DSV4Config, weights: VirtualWeights) -> torch.Tensor:
    logits = weights.linear(x, f"layers.{layer}.moe.router.gate", config.moe_experts)
    scores = F.softplus(logits.float()).sqrt()
    top_scores, top_experts = torch.topk(scores, k=config.router_top_k, dim=-1, sorted=False)
    top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
    top_scores = (top_scores * config.router_topk_scaling_factor).to(x.dtype)

    batch, seq_len, hidden = x.shape
    flat_x = x.reshape(batch * seq_len, hidden)
    flat_out = torch.zeros_like(flat_x)
    token_index = torch.arange(batch * seq_len, device=x.device).view(batch, seq_len, 1)
    token_index = token_index.expand_as(top_experts)

    for expert in torch.unique(top_experts).tolist():
        mask = top_experts == expert
        selected = flat_x[token_index[mask]]
        expert_out = expert_forward(selected, layer, int(expert), config, weights)
        expert_out = expert_out * top_scores[mask].unsqueeze(-1)
        flat_out.index_add_(0, token_index[mask], expert_out)

    shared = shared_expert_forward(x, layer, config, weights)
    return flat_out.view(batch, seq_len, hidden) + shared


def transformer_forward(tokens: torch.Tensor, config: DSV4Config, weights: VirtualWeights) -> torch.Tensor:
    x = weights.embedding_rows(tokens, config.hidden_dim)
    for layer in range(config.layers):
        x = x + attention_layer(
            rms_norm(x, weights.norm_weight(f"layers.{layer}.attention_norm", config.hidden_dim)),
            layer,
            config,
            weights,
        )
        x = x + moe_layer(
            rms_norm(x, weights.norm_weight(f"layers.{layer}.ffn_norm", config.hidden_dim)),
            layer,
            config,
            weights,
        )
    return rms_norm(x, weights.norm_weight("final_norm", config.hidden_dim))


def chunked_target_logprobs(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    config: DSV4Config,
    weights: VirtualWeights,
    vocab_chunk: int,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, config.hidden_dim)
    flat_labels = labels.reshape(-1)
    logsumexp = torch.full((flat_hidden.shape[0],), -float("inf"), device=hidden.device, dtype=torch.float32)
    target_logits = torch.empty_like(logsumexp)

    for start in range(0, config.vocab_size, vocab_chunk):
        end = min(start + vocab_chunk, config.vocab_size)
        out_features = end - start
        weight = weights.tensor(
            f"lm_head.weight.{start}:{end}",
            (config.hidden_dim, out_features),
            config.hidden_dim**-0.5,
        )
        logits = (flat_hidden @ weight).float()
        logsumexp = torch.logaddexp(logsumexp, torch.logsumexp(logits, dim=-1))
        mask = (flat_labels >= start) & (flat_labels < end)
        if mask.any():
            target_logits[mask] = logits[mask, flat_labels[mask] - start]

    return (target_logits - logsumexp).view_as(labels)


def run(args: argparse.Namespace) -> dict:
    config = DSV4Config(
        layers=args.layers,
        seq_len=args.seq_len,
        weight_seed=args.weight_seed,
        batch_seed=args.batch_seed,
    )
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    weights = VirtualWeights(config.weight_seed, device, dtype)
    tokens = make_fixed_batch(config, args.batch_size, device)
    start = time.perf_counter()
    with torch.no_grad():
        hidden = transformer_forward(tokens[:, :-1], config, weights)
        logprobs = chunked_target_logprobs(hidden, tokens[:, 1:], config, weights, args.vocab_chunk)
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": tokens.cpu(),
            "logprobs": logprobs.cpu(),
            "config": config.__dict__,
            "elapsed_sec": elapsed,
        },
        args.output,
    )
    metrics = {
        "implementation": "pytorch_oracle",
        "output": str(args.output),
        "batch_size": args.batch_size,
        "seq_len": config.seq_len,
        "layers": config.layers,
        "tokens_scored": int(logprobs.numel()),
        "elapsed_sec": elapsed,
        "mean_logprob": float(logprobs.float().mean().item()),
        "max_logprob": float(logprobs.float().max().item()),
        "min_logprob": float(logprobs.float().min().item()),
        "weight_seed": config.weight_seed,
        "batch_seed": config.batch_seed,
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("oracle_logprobs.pt"))
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--vocab-chunk", type=int, default=4096)
    parser.add_argument("--weight-seed", type=int, default=1234)
    parser.add_argument("--batch-seed", type=int, default=2027)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
