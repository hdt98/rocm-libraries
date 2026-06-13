# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@torch.library.custom_op("torchtitan::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return out.scatter_add(dim=0, index=index, src=src)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


@deterministic_scatter_add.register_fake
def _(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, torch.Tensor]:
    (index,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    grad_src = torch.gather(grad_output, dim=0, index=index)
    return grad_output, None, grad_src


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    _out, index, _src = inputs
    ctx.save_for_backward(index)


deterministic_scatter_add.register_autograd(
    _backward,
    setup_context=_setup_context,
)


if triton is not None:

    @triton.jit
    def _weighted_scatter_add_forward_kernel(
        out,
        index,
        src,
        scores,
        n_rows,
        out_rows,
        dim,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        stride_out_row: tl.constexpr,
        stride_out_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < dim
        token_rows = tl.load(index + rows, mask=row_mask, other=0)
        valid_token = row_mask & (token_rows >= 0) & (token_rows < out_rows)
        score = tl.load(scores + rows, mask=row_mask, other=0.0).to(tl.float32)
        src_vals = tl.load(
            src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        weighted = src_vals * score[:, None]
        tl.atomic_add(
            out
            + token_rows[:, None] * stride_out_row
            + cols[None, :] * stride_out_col,
            weighted,
            sem="relaxed",
            mask=valid_token[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _weighted_scatter_add_backward_kernel(
        grad_output,
        index,
        src,
        scores,
        grad_src,
        grad_scores,
        n_rows,
        grad_output_rows,
        dim,
        stride_grad_output_row: tl.constexpr,
        stride_grad_output_col: tl.constexpr,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        stride_grad_src_row: tl.constexpr,
        stride_grad_src_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        COMPUTE_GRAD_SCORES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < dim
        token_rows = tl.load(index + rows, mask=row_mask, other=0)
        valid_token = row_mask & (token_rows >= 0) & (token_rows < grad_output_rows)
        score = tl.load(scores + rows, mask=row_mask, other=0.0).to(tl.float32)
        grad_vals = tl.load(
            grad_output
            + token_rows[:, None] * stride_grad_output_row
            + cols[None, :] * stride_grad_output_col,
            mask=valid_token[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_src
            + rows[:, None] * stride_grad_src_row
            + cols[None, :] * stride_grad_src_col,
            grad_vals * score[:, None],
            mask=row_mask[:, None] & col_mask[None, :],
        )
        if COMPUTE_GRAD_SCORES:
            src_vals = tl.load(
                src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_score_part = tl.sum(grad_vals * src_vals, axis=1)
            tl.atomic_add(
                grad_scores + rows,
                grad_score_part,
                sem="relaxed",
                mask=valid_token,
            )


    @triton.jit
    def _weighted_scatter_score_partial_kernel(
        grad_output,
        index,
        src,
        partial_scores,
        n_rows,
        grad_output_rows,
        dim,
        n_col_blocks,
        stride_grad_output_row: tl.constexpr,
        stride_grad_output_col: tl.constexpr,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < dim
        token_rows = tl.load(index + rows, mask=row_mask, other=0)
        valid_token = row_mask & (token_rows >= 0) & (token_rows < grad_output_rows)
        grad_vals = tl.load(
            grad_output
            + token_rows[:, None] * stride_grad_output_row
            + cols[None, :] * stride_grad_output_col,
            mask=valid_token[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        src_vals = tl.load(
            src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.sum(grad_vals * src_vals, axis=1)
        tl.store(
            partial_scores + rows * n_col_blocks + pid_n,
            partial,
            mask=row_mask,
        )


    @triton.jit
    def _weighted_scatter_score_finalize_kernel(
        partial_scores,
        grad_scores,
        n_rows,
        n_col_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        parts = tl.arange(0, BLOCK_P)
        row_mask = rows < n_rows
        part_mask = parts < n_col_blocks
        values = tl.load(
            partial_scores + rows[:, None] * n_col_blocks + parts[None, :],
            mask=row_mask[:, None] & part_mask[None, :],
            other=0.0,
        )
        tl.store(grad_scores + rows, tl.sum(values, axis=1), mask=row_mask)


    @triton.jit
    def _weighted_scatter_score_row_loop_kernel(
        grad_output,
        index,
        src,
        grad_scores,
        n_rows,
        grad_output_rows,
        dim,
        stride_grad_output_row: tl.constexpr,
        stride_grad_output_col: tl.constexpr,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_COL_BLOCKS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        token_rows = tl.load(index + rows, mask=row_mask, other=0)
        valid_token = row_mask & (token_rows >= 0) & (token_rows < grad_output_rows)
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for block_n in tl.range(0, N_COL_BLOCKS):
            cols = block_n * BLOCK_N + offs_n
            col_mask = cols < dim
            grad_vals = tl.load(
                grad_output
                + token_rows[:, None] * stride_grad_output_row
                + cols[None, :] * stride_grad_output_col,
                mask=valid_token[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            src_vals = tl.load(
                src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(grad_vals * src_vals, axis=1)
        tl.store(grad_scores + rows, acc, mask=row_mask)


    @triton.jit
    def _weighted_scatter_add_flat_forward_kernel(
        out,
        flat_positions,
        src,
        top_scores_flat,
        n_rows,
        dim,
        flat_position_offset,
        top_k: tl.constexpr,
        top_scores_n,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        stride_out_row: tl.constexpr,
        stride_out_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < dim
        local_flat = tl.load(flat_positions + rows, mask=row_mask, other=0).to(
            tl.int64
        ) - flat_position_offset
        valid_flat = row_mask & (local_flat >= 0) & (local_flat < top_scores_n)
        token_rows = local_flat // top_k
        score = tl.load(top_scores_flat + local_flat, mask=valid_flat, other=0.0).to(
            tl.float32
        )
        src_vals = tl.load(
            src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        weighted = src_vals * score[:, None]
        tl.atomic_add(
            out
            + token_rows[:, None] * stride_out_row
            + cols[None, :] * stride_out_col,
            weighted,
            sem="relaxed",
            mask=valid_flat[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _weighted_scatter_add_flat_backward_kernel(
        grad_output,
        flat_positions,
        src,
        top_scores_flat,
        grad_src,
        grad_top_scores_flat,
        n_rows,
        dim,
        flat_position_offset,
        top_k: tl.constexpr,
        top_scores_n,
        stride_grad_output_row: tl.constexpr,
        stride_grad_output_col: tl.constexpr,
        stride_src_row: tl.constexpr,
        stride_src_col: tl.constexpr,
        stride_grad_src_row: tl.constexpr,
        stride_grad_src_col: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < dim
        local_flat = tl.load(flat_positions + rows, mask=row_mask, other=0).to(
            tl.int64
        ) - flat_position_offset
        valid_flat = row_mask & (local_flat >= 0) & (local_flat < top_scores_n)
        token_rows = local_flat // top_k
        score = tl.load(top_scores_flat + local_flat, mask=valid_flat, other=0.0).to(
            tl.float32
        )
        grad_vals = tl.load(
            grad_output
            + token_rows[:, None] * stride_grad_output_row
            + cols[None, :] * stride_grad_output_col,
            mask=valid_flat[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        src_vals = tl.load(
            src + rows[:, None] * stride_src_row + cols[None, :] * stride_src_col,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_src
            + rows[:, None] * stride_grad_src_row
            + cols[None, :] * stride_grad_src_col,
            grad_vals * score[:, None],
            mask=row_mask[:, None] & col_mask[None, :],
        )
        grad_score_part = tl.sum(grad_vals * src_vals, axis=1)
        tl.atomic_add(
            grad_top_scores_flat + local_flat,
            grad_score_part,
            sem="relaxed",
            mask=valid_flat,
        )


def _weighted_scatter_backward(
    index_1d: torch.Tensor,
    src: torch.Tensor,
    scores: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    prefer_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        prefer_triton
        and triton is not None
        and tl is not None
        and _weighted_scatter_add_backward_kernel is not None
        and grad_output.is_cuda
    ):
        grad_output_contig = grad_output.contiguous()
        index_contig = index_1d.contiguous()
        src_contig = src.contiguous()
        scores_contig = scores.contiguous()
        grad_src = torch.empty_like(src)
        grad_scores = torch.zeros((src.shape[0],), device=src.device, dtype=torch.float32)
        n_rows = src.shape[0]
        dim = src.shape[1]
        if n_rows == 0 or dim == 0:
            return grad_src, grad_scores
        block_m = int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_BLOCK_M", "16"))
        block_n = int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_BLOCK_N", "128"))
        score_mode = os.environ.get(
            "TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_MODE", "triton"
        ).lower()
        split_score_modes = {"triton_split", "split", "two_pass"}
        row_score_modes = {
            "triton_row",
            "row",
            "rowwise",
            "row_loop",
            "row_owned",
            "rowowned",
            "triton_row_bm4",
            "row_bm4",
            "row4",
        }
        compute_grad_scores_in_triton = score_mode not in {
            "torch",
            "pytorch",
            "zero",
            "none",
            "off",
        } | split_score_modes | row_score_modes
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(dim, block_n))
        _weighted_scatter_add_backward_kernel[grid](
            grad_output_contig,
            index_contig,
            src_contig,
            scores_contig,
            grad_src,
            grad_scores,
            n_rows,
            grad_output_contig.shape[0],
            dim,
            grad_output_contig.stride(0),
            grad_output_contig.stride(1),
            src_contig.stride(0),
            src_contig.stride(1),
            grad_src.stride(0),
            grad_src.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            COMPUTE_GRAD_SCORES=compute_grad_scores_in_triton,
            num_warps=int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_NUM_WARPS", "4")
            ),
        )
        if not compute_grad_scores_in_triton and score_mode in {"torch", "pytorch"}:
            gather_index = index_contig.reshape(-1, 1).expand(-1, grad_output.shape[-1])
            grad_at_src = torch.gather(grad_output_contig, dim=0, index=gather_index)
            grad_scores = (grad_at_src.to(torch.float32) * src_contig.to(torch.float32)).sum(
                dim=1
            )
        elif score_mode in split_score_modes:
            score_block_m = int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_BLOCK_M", "4")
            )
            score_block_n = int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_BLOCK_N", "1024")
            )
            score_final_block_m = int(
                os.environ.get(
                    "TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_FINAL_BLOCK_M", "64"
                )
            )
            n_col_blocks = triton.cdiv(dim, score_block_n)
            partial_scores = torch.empty(
                (n_rows, n_col_blocks), device=src.device, dtype=torch.float32
            )
            score_grid = (triton.cdiv(n_rows, score_block_m), n_col_blocks)
            _weighted_scatter_score_partial_kernel[score_grid](
                grad_output_contig,
                index_contig,
                src_contig,
                partial_scores,
                n_rows,
                grad_output_contig.shape[0],
                dim,
                n_col_blocks,
                grad_output_contig.stride(0),
                grad_output_contig.stride(1),
                src_contig.stride(0),
                src_contig.stride(1),
                BLOCK_M=score_block_m,
                BLOCK_N=score_block_n,
                num_warps=int(
                    os.environ.get(
                        "TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_NUM_WARPS", "4"
                    )
                ),
            )
            final_block_p = 1 << (int(n_col_blocks) - 1).bit_length()
            _weighted_scatter_score_finalize_kernel[
                (triton.cdiv(n_rows, score_final_block_m),)
            ](
                partial_scores,
                grad_scores,
                n_rows,
                n_col_blocks=n_col_blocks,
                BLOCK_M=score_final_block_m,
                BLOCK_P=final_block_p,
                num_warps=1,
            )
        elif score_mode in row_score_modes:
            default_score_block_m = (
                "4"
                if score_mode in {"triton_row_bm4", "row_bm4", "row4"}
                else "1"
            )
            score_block_m = int(
                os.environ.get(
                    "TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_BLOCK_M",
                    default_score_block_m,
                )
            )
            score_block_n = int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_BLOCK_N", "256")
            )
            n_col_blocks = triton.cdiv(dim, score_block_n)
            _weighted_scatter_score_row_loop_kernel[
                (triton.cdiv(n_rows, score_block_m),)
            ](
                grad_output_contig,
                index_contig,
                src_contig,
                grad_scores,
                n_rows,
                grad_output_contig.shape[0],
                dim,
                grad_output_contig.stride(0),
                grad_output_contig.stride(1),
                src_contig.stride(0),
                src_contig.stride(1),
                BLOCK_M=score_block_m,
                BLOCK_N=score_block_n,
                N_COL_BLOCKS=n_col_blocks,
                num_warps=int(
                    os.environ.get(
                        "TORCHTITAN_WEIGHTED_SCATTER_BWD_SCORE_NUM_WARPS", "4"
                    )
                ),
            )
        return grad_src, grad_scores

    gather_index = index_1d.reshape(-1, 1).expand(-1, grad_output.shape[-1])
    grad_at_src = torch.gather(grad_output, dim=0, index=gather_index)
    grad_src = (grad_at_src.to(torch.float32) * scores.reshape(-1, 1)).to(src.dtype)
    grad_scores = (grad_at_src.to(torch.float32) * src.to(torch.float32)).sum(dim=1)
    return grad_src, grad_scores


def _weighted_scatter_flat_backward(
    flat_positions: torch.Tensor,
    src: torch.Tensor,
    top_scores_flat: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    top_k: int,
    flat_position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        triton is not None
        and tl is not None
        and _weighted_scatter_add_flat_backward_kernel is not None
        and grad_output.is_cuda
    ):
        grad_output_contig = grad_output.contiguous()
        flat_positions_contig = flat_positions.contiguous()
        src_contig = src.contiguous()
        top_scores_flat_contig = top_scores_flat.contiguous()
        grad_src = torch.empty_like(src_contig)
        grad_top_scores_flat = torch.zeros_like(top_scores_flat_contig)
        n_rows = src_contig.shape[0]
        dim = src_contig.shape[1]
        block_m = int(
            os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_BWD_BLOCK_M", "16")
        )
        block_n = int(
            os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_BWD_BLOCK_N", "128")
        )
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(dim, block_n))
        _weighted_scatter_add_flat_backward_kernel[grid](
            grad_output_contig,
            flat_positions_contig,
            src_contig,
            top_scores_flat_contig,
            grad_src,
            grad_top_scores_flat,
            n_rows,
            dim,
            int(flat_position_offset),
            top_k=int(top_k),
            top_scores_n=top_scores_flat_contig.numel(),
            stride_grad_output_row=grad_output_contig.stride(0),
            stride_grad_output_col=grad_output_contig.stride(1),
            stride_src_row=src_contig.stride(0),
            stride_src_col=src_contig.stride(1),
            stride_grad_src_row=grad_src.stride(0),
            stride_grad_src_col=grad_src.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_BWD_NUM_WARPS", "4")
            ),
        )
        return grad_src, grad_top_scores_flat

    local_flat = flat_positions.to(torch.int64) - int(flat_position_offset)
    token_rows = local_flat // int(top_k)
    scores = top_scores_flat[local_flat]
    gather_index = token_rows.reshape(-1, 1).expand(-1, grad_output.shape[-1])
    grad_at_src = torch.gather(grad_output, dim=0, index=gather_index)
    grad_src = (grad_at_src.to(torch.float32) * scores.reshape(-1, 1)).to(src.dtype)
    grad_scores = (grad_at_src.to(torch.float32) * src.to(torch.float32)).sum(dim=1)
    grad_top_scores_flat = torch.zeros_like(top_scores_flat)
    grad_top_scores_flat.scatter_add_(0, local_flat, grad_scores.to(top_scores_flat.dtype))
    return grad_src, grad_top_scores_flat


class _TritonWeightedScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        out: torch.Tensor,
        index_1d: torch.Tensor,
        src: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if triton is None or tl is None or _weighted_scatter_add_forward_kernel is None:
            raise RuntimeError("Triton weighted scatter-add requested but Triton is unavailable")
        if out.dim() != 2 or src.dim() != 2:
            raise ValueError("weighted scatter-add expects 2D out and src tensors")
        if index_1d.dim() != 1 or scores.dim() != 1:
            raise ValueError("weighted scatter-add expects 1D index and scores tensors")
        if src.shape[0] != index_1d.numel() or src.shape[0] != scores.numel():
            raise ValueError("src, index, and scores must have the same row count")
        if src.shape[1] != out.shape[1]:
            raise ValueError("src and out hidden dimensions must match")
        if not out.is_cuda:
            raise RuntimeError("Triton weighted scatter-add expects CUDA/ROCm tensors")

        src_contig = src.contiguous()
        index_contig = index_1d.contiguous()
        scores_contig = scores.contiguous()
        result = torch.empty_like(out)
        result.copy_(out)
        n_rows = src_contig.shape[0]
        dim = src_contig.shape[1]
        block_m = int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BLOCK_M", "16"))
        block_n = int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_BLOCK_N", "128"))
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(dim, block_n))
        _weighted_scatter_add_forward_kernel[grid](
            result,
            index_contig,
            src_contig,
            scores_contig,
            n_rows,
            result.shape[0],
            dim,
            src_contig.stride(0),
            src_contig.stride(1),
            result.stride(0),
            result.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_NUM_WARPS", "4")),
        )
        ctx.save_for_backward(index_contig, src_contig, scores_contig)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        index_1d, src, scores = ctx.saved_tensors
        grad_src, grad_scores = _weighted_scatter_backward(
            index_1d,
            src,
            scores,
            grad_output,
            prefer_triton=True,
        )
        return grad_output, None, grad_src, grad_scores.to(scores.dtype)


class _TritonWeightedScatterAddFromFlatPositions(torch.autograd.Function):
    """Weighted scatter where token and score come from source flat positions.

    This avoids materializing two adapter tensors in native MORI combine:
    ``token = flat_position // top_k`` and ``score = top_scores_flat[flat_position]``.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        out: torch.Tensor,
        flat_positions: torch.Tensor,
        src: torch.Tensor,
        top_scores_flat: torch.Tensor,
        top_k: int,
        flat_position_offset: int,
    ) -> torch.Tensor:
        if (
            triton is None
            or tl is None
            or _weighted_scatter_add_flat_forward_kernel is None
        ):
            raise RuntimeError(
                "Triton flat-position weighted scatter requested but Triton is unavailable"
            )
        if out.dim() != 2 or src.dim() != 2:
            raise ValueError("weighted scatter-add expects 2D out and src tensors")
        if flat_positions.dim() != 1 or top_scores_flat.dim() != 1:
            raise ValueError(
                "flat-position weighted scatter expects 1D positions and scores"
            )
        if src.shape[0] != flat_positions.numel():
            raise ValueError("src and flat_positions must have the same row count")
        if src.shape[1] != out.shape[1]:
            raise ValueError("src and out hidden dimensions must match")
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        if not out.is_cuda:
            raise RuntimeError(
                "Triton flat-position weighted scatter expects CUDA/ROCm tensors"
            )

        src_contig = src.contiguous()
        flat_positions_contig = flat_positions.contiguous()
        top_scores_flat_contig = top_scores_flat.contiguous()
        result = torch.empty_like(out)
        result.copy_(out)
        n_rows = src_contig.shape[0]
        dim = src_contig.shape[1]
        block_m = int(
            os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_BLOCK_M", "16")
        )
        block_n = int(
            os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_BLOCK_N", "128")
        )
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(dim, block_n))
        _weighted_scatter_add_flat_forward_kernel[grid](
            result,
            flat_positions_contig,
            src_contig,
            top_scores_flat_contig,
            n_rows,
            dim,
            int(flat_position_offset),
            top_k=int(top_k),
            top_scores_n=top_scores_flat_contig.numel(),
            stride_src_row=src_contig.stride(0),
            stride_src_col=src_contig.stride(1),
            stride_out_row=result.stride(0),
            stride_out_col=result.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=int(
                os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_FLAT_NUM_WARPS", "4")
            ),
        )
        ctx.top_k = int(top_k)
        ctx.flat_position_offset = int(flat_position_offset)
        ctx.save_for_backward(flat_positions_contig, src_contig, top_scores_flat_contig)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None, None]:
        flat_positions, src, top_scores_flat = ctx.saved_tensors
        grad_src, grad_top_scores_flat = _weighted_scatter_flat_backward(
            flat_positions,
            src,
            top_scores_flat,
            grad_output,
            top_k=int(ctx.top_k),
            flat_position_offset=int(ctx.flat_position_offset),
        )
        return (
            grad_output,
            None,
            grad_src,
            grad_top_scores_flat.to(top_scores_flat.dtype),
            None,
            None,
        )


class _TorchForwardTritonBackwardWeightedScatterAdd(torch.autograd.Function):
    """Torch forward semantics with the Triton weighted-scatter backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        out: torch.Tensor,
        index_1d: torch.Tensor,
        src: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        src_contig = src.contiguous()
        index_contig = index_1d.contiguous()
        scores_contig = scores.contiguous()
        result = deterministic_weighted_scatter_add(
            out,
            index_contig,
            src_contig,
            scores_contig,
        )
        ctx.save_for_backward(index_contig, src_contig, scores_contig)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        index_1d, src, scores = ctx.saved_tensors
        grad_src, grad_scores = _weighted_scatter_backward(
            index_1d,
            src,
            scores,
            grad_output,
            prefer_triton=True,
        )
        return grad_output, None, grad_src, grad_scores.to(scores.dtype)


class _TorchWeightedScatterAdd(torch.autograd.Function):
    """Torch forward with a fused manual backward for weighted scatter-add."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        out: torch.Tensor,
        index_1d: torch.Tensor,
        src: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if out.dim() != 2 or src.dim() != 2:
            raise ValueError("weighted scatter-add expects 2D out and src tensors")
        if index_1d.dim() != 1 or scores.dim() != 1:
            raise ValueError("weighted scatter-add expects 1D index and scores tensors")
        if src.shape[0] != index_1d.numel() or src.shape[0] != scores.numel():
            raise ValueError("src, index, and scores must have the same row count")
        if src.shape[1] != out.shape[1]:
            raise ValueError("src and out hidden dimensions must match")

        index_contig = index_1d.contiguous()
        src_contig = src.contiguous()
        scores_contig = scores.contiguous()
        weighted = (src_contig.to(torch.float32) * scores_contig.reshape(-1, 1)).to(
            src_contig.dtype
        )
        result = deterministic_scatter_add(
            out,
            index_contig.reshape(-1, 1).expand(-1, out.shape[-1]),
            weighted,
        )
        ctx.save_for_backward(index_contig, src_contig, scores_contig)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        index_1d, src, scores = ctx.saved_tensors
        gather_index = index_1d.reshape(-1, 1).expand(-1, grad_output.shape[-1])
        grad_at_src = torch.gather(grad_output, dim=0, index=gather_index)
        grad_src = (grad_at_src.to(torch.float32) * scores.reshape(-1, 1)).to(src.dtype)
        grad_scores = (grad_at_src.to(torch.float32) * src.to(torch.float32)).sum(dim=1)
        return grad_output, None, grad_src, grad_scores.to(scores.dtype)


class _ChunkedTorchWeightedScatterAdd(torch.autograd.Function):
    """Exact torch weighted scatter-add with bounded forward workspace."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        out: torch.Tensor,
        index_1d: torch.Tensor,
        src: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        if out.dim() != 2 or src.dim() != 2:
            raise ValueError("weighted scatter-add expects 2D out and src tensors")
        if index_1d.dim() != 1 or scores.dim() != 1:
            raise ValueError("weighted scatter-add expects 1D index and scores tensors")
        if src.shape[0] != index_1d.numel() or src.shape[0] != scores.numel():
            raise ValueError("src, index, and scores must have the same row count")
        if src.shape[1] != out.shape[1]:
            raise ValueError("src and out hidden dimensions must match")

        index_contig = index_1d.contiguous()
        src_contig = src.contiguous()
        scores_contig = scores.contiguous()
        chunk_rows = max(
            1, int(os.environ.get("TORCHTITAN_WEIGHTED_SCATTER_CHUNK_ROWS", "65536"))
        )
        result = out.clone()
        dim = out.shape[-1]
        for start in range(0, src_contig.shape[0], chunk_rows):
            end = min(start + chunk_rows, src_contig.shape[0])
            chunk_src = src_contig[start:end]
            chunk_scores = scores_contig[start:end]
            weighted = (chunk_src.to(torch.float32) * chunk_scores.reshape(-1, 1)).to(
                chunk_src.dtype
            )
            result = deterministic_scatter_add(
                result,
                index_contig[start:end].reshape(-1, 1).expand(-1, dim),
                weighted,
            )
        ctx.save_for_backward(index_contig, src_contig, scores_contig)
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
        index_1d, src, scores = ctx.saved_tensors
        gather_index = index_1d.reshape(-1, 1).expand(-1, grad_output.shape[-1])
        grad_at_src = torch.gather(grad_output, dim=0, index=gather_index)
        grad_src = (grad_at_src.to(torch.float32) * scores.reshape(-1, 1)).to(src.dtype)
        grad_scores = (grad_at_src.to(torch.float32) * src.to(torch.float32)).sum(dim=1)
        return grad_output, None, grad_src, grad_scores.to(scores.dtype)


def deterministic_weighted_scatter_add(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    weighted = (src.to(torch.float32) * scores.reshape(-1, 1)).to(src.dtype)
    return deterministic_scatter_add(
        out,
        index_1d.reshape(-1, 1).expand(-1, out.shape[-1]),
        weighted,
    )


def weighted_scatter_add_from_flat_positions(
    out: torch.Tensor,
    flat_positions: torch.Tensor,
    src: torch.Tensor,
    top_scores_flat: torch.Tensor,
    *,
    top_k: int,
    flat_position_offset: int = 0,
    backend: str | None = None,
) -> torch.Tensor:
    selected_backend = (
        backend
        or os.environ.get("TORCHTITAN_STANDARD_EP_SOURCE_SCATTER_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_SOURCE_SCATTER_BACKEND")
        or "materialize"
    ).lower()
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if selected_backend in {"", "off", "torch", "materialize"}:
        local_flat = flat_positions.to(torch.int64) - int(flat_position_offset)
        token_indices = local_flat // int(top_k)
        scores = top_scores_flat[local_flat]
        return weighted_scatter_add(out, token_indices, src, scores)
    if selected_backend in {"triton", "triton_flat", "flat"}:
        if triton is None or not src.is_cuda:
            return weighted_scatter_add_from_flat_positions(
                out,
                flat_positions,
                src,
                top_scores_flat,
                top_k=int(top_k),
                flat_position_offset=int(flat_position_offset),
                backend="materialize",
            )
        return _TritonWeightedScatterAddFromFlatPositions.apply(
            out,
            flat_positions,
            src,
            top_scores_flat,
            int(top_k),
            int(flat_position_offset),
        )
    raise ValueError(
        f"Unknown flat-position weighted scatter-add backend: {selected_backend!r}"
    )


def weighted_scatter_add(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
    scores: torch.Tensor,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    selected_backend = (
        backend
        or os.environ.get("TORCHTITAN_STANDARD_EP_WEIGHTED_SCATTER_BACKEND")
        or os.environ.get("CANARY_STANDARD_EP_WEIGHTED_SCATTER_BACKEND")
        or "torch"
    ).lower()
    if selected_backend in {"", "torch", "deterministic", "off"}:
        return deterministic_weighted_scatter_add(out, index_1d, src, scores)
    if selected_backend in {"torch_custom", "custom"}:
        return _TorchWeightedScatterAdd.apply(out, index_1d, src, scores)
    if selected_backend in {"torch_chunked", "chunked"}:
        return _ChunkedTorchWeightedScatterAdd.apply(out, index_1d, src, scores)
    if selected_backend in {"triton_bwd", "torch_triton_bwd", "torch_forward_triton_backward"}:
        return _TorchForwardTritonBackwardWeightedScatterAdd.apply(
            out, index_1d, src, scores
        )
    if selected_backend == "triton":
        if triton is None or not src.is_cuda:
            return deterministic_weighted_scatter_add(out, index_1d, src, scores)
        return _TritonWeightedScatterAdd.apply(out, index_1d, src, scores)
    raise ValueError(f"Unknown weighted scatter-add backend: {selected_backend!r}")
