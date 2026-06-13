# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from mori import cpp as mori_cpp
from mori.tensor_utils import from_gpu_ptr, dtype_to_int
import logging
import os
from dataclasses import dataclass
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

TOPK_IDX_DTYPE = torch.int32
WARP_SIZE = 64


class EpDispatchCombineKernelType(mori_cpp.EpDispatchCombineKernelType):
    def __str__(self):
        return self.name


class EpDispatchCombineQuantType(mori_cpp.EpDispatchCombineQuantType):
    def __str__(self):
        return self.name


_QUANT_TYPE_MAP = {
    "none": EpDispatchCombineQuantType.None_,
    "fp8_direct_cast": EpDispatchCombineQuantType.Fp8DirectCast,
    "fp8_blockwise": EpDispatchCombineQuantType.Fp8BlockwiseQuant,
}


def _normalize_quant_type(quant_type):
    if isinstance(quant_type, EpDispatchCombineQuantType):
        return quant_type
    if isinstance(quant_type, str):
        key = quant_type.strip().lower()
        if key in _QUANT_TYPE_MAP:
            return _QUANT_TYPE_MAP[key]
    raise ValueError(
        f"invalid quant_type '{quant_type}', expected one of {list(_QUANT_TYPE_MAP.keys())}"
    )


def _current_stream():
    return torch.cuda.current_stream().cuda_stream


def _non_null_1d_ptr_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a same-device tensor with a non-null data pointer.

    Native compact dispatch/combine uses zero splits legitimately, but the C++
    launcher validates pointer arguments before row counts.  Keep the logical
    row counts unchanged and pass one-element scratch only when PyTorch gives a
    zero-sized tensor a null data_ptr().
    """

    if tensor.numel() > 0:
        return tensor
    return tensor.new_empty((1,))


def _non_null_2d_ptr_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() > 0:
        return tensor
    if tensor.dim() != 2:
        raise ValueError("expected a 2D tensor for compact row scratch")
    return tensor.new_empty((1, int(tensor.size(1))))


@dataclass
class EpDispatchCombineConfig:
    """Configuration for :class:`EpDispatchCombineOp`.

    Args:
        data_type: Deprecated. Tensor dtype kept only for backward
            compatibility with tests and examples. Kernel launch dtype is
            inferred from the runtime input tensor instead of this field.
        rank: Rank of the current process in the expert-parallel group.
        world_size: Total number of ranks participating in the dispatch/combine
            operation.
        hidden_dim: Hidden dimension of each token embedding.
        scale_dim: Number of scale values stored per token for quantized paths.
            Describes caller-provided dispatch scales (e.g. FP4 input).
            ``quant_type="fp8_blockwise"`` combine uses its own internal
            scale_dim driven by ``MORI_FP8_COMBINE_SCALE_DIM`` (default 56).
        scale_type_size: Size in bytes of each scale element.
        max_token_type_size: Maximum size in bytes for the token element type.
        max_num_inp_token_per_rank: Maximum number of input tokens each rank
            can process.
        num_experts_per_rank: Number of local experts hosted on each rank.
        num_experts_per_token: Number of experts selected for each token.
        warp_num_per_block: Number of warps per GPU block for the kernel launch.
        block_num: Number of GPU blocks to launch for the main kernel.
        max_total_recv_tokens: Optional cap used to derive the maximum number
            of tokens a rank can receive, which also affects memory
            consumption. A value of ``0`` disables the cap. If the actual
            received token count exceeds the derived limit, the kernel
            currently asserts.
        use_external_inp_buf: Whether the operator expects the input buffer to
            be managed externally.
        kernel_type: Dispatch/combine kernel implementation to use.
        gpu_per_node: Number of GPUs per node. This affects all kernel types.
        rdma_block_num: Number of RDMA blocks for inter-node kernels.
        num_qp_per_pe: Number of queue pairs per processing element.
        quant_type: Quantization mode. Supported string values are ``"none"``,
            ``"fp8_direct_cast"``, and ``"fp8_blockwise"``.
        standard_ep_compact_only: Allocate only the MORI shared buffers needed
            by the native standard-EP compact-row dispatch/combine ABI.
        standard_ep_compact_role: Native compact allocation role. ``0`` keeps
            the historical bidirectional compact buffers, ``1`` allocates only
            dispatch staging, and ``2`` allocates only combine staging.
    """

    data_type: (
        torch.dtype
    )  # Deprecated for kernel launch (runtime dtype inferred from input tensor); retained for test/example compatibility
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    max_total_recv_tokens: int = 0
    use_external_inp_buf: bool = True
    kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 1
    quant_type: str = "none"
    standard_ep_compact_only: bool = False
    standard_ep_compact_role: int = 0


@dataclass(frozen=True)
class BalancedMoeCompactDispatchOutput:
    """Output of MORI balanced-MoE compact dispatch.

    The normal/cold fields are the existing standard-MoE packed receive ABI.
    The hot fields are helper-executed rows in the source partition order,
    usually owner-compact presorted when the caller requests the retained
    hot-helper layout.
    """

    normal_packed_recv_x: torch.Tensor
    normal_packed_recv_count: torch.Tensor
    normal_packed_recv_src_info: torch.Tensor
    normal_packed_recv_layout_range: torch.Tensor
    hot_packed_x: torch.Tensor
    hot_packed_scores: torch.Tensor
    hot_packed_src_info: torch.Tensor
    hot_packed_count: torch.Tensor | None
    hot_group_ends: torch.Tensor | None
    hot_offsets_presorted_by: str

    @property
    def normal_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.normal_packed_recv_x,
            self.normal_packed_recv_count,
            self.normal_packed_recv_src_info,
            self.normal_packed_recv_layout_range,
        )

    @property
    def hot_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.hot_packed_x,
            self.hot_packed_scores,
            self.hot_packed_src_info,
        )


@dataclass(frozen=True)
class BalancedMoeCompactCombineOutput:
    """Output of MORI balanced-MoE compact combine.

    ``source_rank_rows`` and ``source_rank_flat_positions`` are the compact
    return rows for the normal/cold EP path.  ``token_output`` is present only
    when the native combine also emits the weighted token-level output.  Keeping
    this as a named ABI avoids framework-side tuple unpacking around the native
    compact return path.
    """

    source_rank_rows: torch.Tensor
    source_rank_flat_positions: torch.Tensor | None
    token_output: torch.Tensor | None = None

    @property
    def source_outputs(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (self.source_rank_rows, self.source_rank_flat_positions)

    @property
    def has_token_output(self) -> bool:
        return self.token_output is not None


def _cpp_dispatch_combine_factory(entity_name, allow_missing=False):
    if allow_missing:
        return getattr(mori_cpp, entity_name, None)
    return getattr(mori_cpp, entity_name)


def _require_cuda_contiguous_tensor(name, tensor, *, dtype=None, ndim=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if ndim is not None and tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_compact_split_tensors(input_splits, output_splits):
    _require_cuda_contiguous_tensor("input_splits", input_splits, dtype=torch.int64, ndim=1)
    _require_cuda_contiguous_tensor("output_splits", output_splits, dtype=torch.int64, ndim=1)
    if input_splits.numel() != output_splits.numel():
        raise ValueError(
            "input_splits and output_splits must have the same number of ranks"
        )


# ---------------------------------------------------------------------------
# Kernel type → .hsaco compilation unit mapping
# ---------------------------------------------------------------------------
_KERNEL_TYPE_TO_HIP = {
    EpDispatchCombineKernelType.IntraNode: "ep_intranode",
    EpDispatchCombineKernelType.InterNode: "ep_internode",
    EpDispatchCombineKernelType.InterNodeV1: "ep_internode_v1",
    EpDispatchCombineKernelType.InterNodeV1LL: "ep_internode_v1ll",
    EpDispatchCombineKernelType.AsyncLL: "ep_async_ll",
}

# dtype → kernel name suffix
_DTYPE_SUFFIX = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
}
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fn] = "fp8_ocp"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fnuz] = "fp8_fnuz"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float4_e2m1fn_x2] = "fp4"
except AttributeError:
    pass

# pointer size on device for shared memory calculation (sizeof(T**) and sizeof(float**))
_PTR_SIZE = 8


def warmup_jit_kernels(kernel_type):
    """Pre-compile kernels for a kernel_type. Call from main process before spawning workers."""
    from mori.ops._jit_loader import ensure_compiled, _compiled_hsaco

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    hip_name = _KERNEL_TYPE_TO_HIP[kernel_type]
    ensure_compiled(hip_name)
    return _compiled_hsaco.get(hip_name)


def _ensure_jit_kernels(kernel_type):
    """Ensure the required kernels for this kernel_type are JIT-compiled."""
    from mori.ops._jit_loader import ensure_compiled

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    ensure_compiled(_KERNEL_TYPE_TO_HIP[kernel_type])


def _load_hip_modules(kernel_type):
    """Load HipModule for the given kernel_type and init shmem gpu states."""
    from mori.ops._jit_loader import load_hip_module

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    return load_hip_module(_KERNEL_TYPE_TO_HIP[kernel_type], init_shmem=True)


class EpDispatchCombineOp:
    def __init__(self, config):
        self.config = config
        _ensure_jit_kernels(config.kernel_type)

        if dist.is_initialized():
            dist.barrier()

        handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
        self._cpp_config = mori_cpp.EpDispatchCombineConfig(
            rank=config.rank,
            world_size=config.world_size,
            hidden_dim=config.hidden_dim,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            max_token_type_size=config.max_token_type_size,
            max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
            num_experts_per_rank=config.num_experts_per_rank,
            num_experts_per_token=config.num_experts_per_token,
            warp_num_per_block=config.warp_num_per_block,
            block_num=config.block_num,
            use_external_inp_buf=config.use_external_inp_buf,
            kernel_type=config.kernel_type,
            gpu_per_node=config.gpu_per_node,
            rdma_block_num=config.rdma_block_num,
            num_qp_per_pe=config.num_qp_per_pe,
            quant_type=_normalize_quant_type(config.quant_type),
            standard_ep_compact_only=bool(config.standard_ep_compact_only),
            standard_ep_compact_role=int(config.standard_ep_compact_role),
            max_total_recv_tokens=config.max_total_recv_tokens,
        )

        self._handle = handle_class(self._cpp_config)
        self._hip_module = _load_hip_modules(config.kernel_type)
        self._handle_info = mori_cpp.get_handle_info(self._handle)

        self._fp8_blockwise_combine_scale_dim = self._handle_info[
            "fp8_blockwise_combine_scale_dim"
        ]
        self._fp8_blockwise_combine_scale_type_size = self._handle_info[
            "fp8_blockwise_combine_scale_type_size"
        ]

        self._dispatch_out_ptrs = mori_cpp.get_dispatch_output_ptrs(self._handle, True)
        self._combine_out_ptrs = mori_cpp.get_combine_output_ptrs(self._handle, True)

        self.local_expert_count = torch.zeros(
            config.num_experts_per_rank, dtype=torch.int32, device="cuda"
        )

        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            "get_dispatch_src_token_pos"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_factory(
            "get_cur_rank_num_token"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_sender_token_idx_map"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_receiver_token_idx_map"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_factory(
            "get_registered_combine_input_buffer"
        )

        self.launch_config_mode = os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "MANUAL")
        if self.launch_config_mode == "AUTO":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            try:
                from mori.ops.tuning_config import (
                    TuningConfigManager,
                    kernel_type_to_config_str,
                    quant_type_to_config_str,
                    detect_gpu_model,
                )
                from mori.jit.config import detect_gpu_arch

                gpu_arch = detect_gpu_arch()
                gpu_model = detect_gpu_model()
                kt_str = kernel_type_to_config_str(config.kernel_type)
                self._qt_str = quant_type_to_config_str(config.quant_type)
                mgr = TuningConfigManager.get_instance(
                    gpu_arch,
                    kt_str,
                    config.world_size,
                    gpu_model,
                )
                self._dispatch_rules = mgr.dispatch_rules or None
                self._combine_rules = mgr.combine_rules or None
                if logger.isEnabledFor(logging.DEBUG):
                    if self._dispatch_rules is None and self._combine_rules is None:
                        logger.debug(
                            "AUTO tuning: no config for %s_%s_%s_ep%d; "
                            "using hard-coded fallback.",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                        )
                    else:
                        d_dtypes = sorted(
                            {r["dtype"] for r in (self._dispatch_rules or [])}
                        )
                        c_dtypes = sorted(
                            {r["dtype"] for r in (self._combine_rules or [])}
                        )
                        logger.debug(
                            "AUTO tuning: %s_%s_%s_ep%d — "
                            "dispatch(%d rules, dtypes=%s) combine(%d rules, dtypes=%s)",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                            len(self._dispatch_rules or []),
                            d_dtypes,
                            len(self._combine_rules or []),
                            c_dtypes,
                        )
            except Exception as exc:
                logger.warning(
                    "AUTO tuning: failed to load config (%s); "
                    "using hard-coded fallback.",
                    exc,
                )

            if (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (96, 64, 8)
            elif (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1LL.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (256, 128, 8)
            else:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (128, 0, 16)
        elif self.launch_config_mode == "MANUAL":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            self.auto_block_num, self.auto_rdma_block_num, self.auto_warp_per_block = (
                None,
                None,
                None,
            )
        else:
            raise ValueError(
                f"invalid MORI_EP_LAUNCH_CONFIG_MODE, must be ['MANUAL', 'AUTO'], got '{self.launch_config_mode}'"
            )

    # ------------------------------------------------------------------
    # Kernel launch helpers
    # ------------------------------------------------------------------
    def _resolve_launch_params(
        self,
        block_num,
        rdma_block_num,
        warp_per_block,
        *,
        num_tokens=0,
        hidden_dim=0,
        dtype=None,
        tuning_rules=None,
        zero_copy=None,
        quant_type=None,
    ):
        if tuning_rules and dtype is not None:
            from mori.ops.tuning_config import TuningConfigManager

            params = TuningConfigManager.lookup(
                tuning_rules, dtype, num_tokens, hidden_dim, zero_copy, quant_type
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        bn = self.auto_block_num if self.auto_block_num else block_num
        rbn = self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num
        wpb = self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block
        actual_bn = self.config.block_num if bn <= 0 else bn
        actual_rbn = self.config.rdma_block_num if rbn <= 0 else rbn
        actual_wpb = self.config.warp_num_per_block if wpb <= 0 else wpb
        return actual_bn, actual_rbn, actual_wpb

    def _get_func(self, name):
        return self._hip_module.get_function(name)

    def _dispatch_shared_mem(self, warp_per_block):
        """Shared memory for dispatch kernels (worldSize + numExpertPerRank per warp + numExpertPerRank) * sizeof(index_t)."""
        return (
            self.config.world_size * warp_per_block
            + self.config.num_experts_per_rank * warp_per_block
            + self.config.num_experts_per_rank
        ) * 4  # sizeof(index_t)

    def _combine_shared_mem(self, warp_per_block, use_weights=True):
        """Shared memory for combine kernels."""
        quant_type = _normalize_quant_type(self.config.quant_type)
        num_ptr_arrays = 1 + int(bool(use_weights))
        if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
            num_ptr_arrays += 1
        return (
            warp_per_block
            * self.config.num_experts_per_token
            * num_ptr_arrays
            * _PTR_SIZE
        )

    def _launch(self, func_name, grid, block, shared_mem, stream, args_ptr):
        func = self._get_func(func_name)
        func.launch_struct(grid, block, shared_mem, stream, args_ptr)

    def _launch_multi(self, func_names, grids, blocks, shared_mems, stream, args_ptr):
        from mori.jit.hip_driver import launch_multi

        funcs = [self._get_func(name)._func for name in func_names]
        launch_multi(funcs, grids, blocks, shared_mems, stream, args_ptr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_launch_config(
        self, is_dispatch=True, block_num=-1, rdma_block_num=-1, warp_per_block=-1
    ):
        rules = self._dispatch_rules if is_dispatch else self._combine_rules
        if rules:
            from mori.ops.tuning_config import TuningConfigManager

            zc = not self.config.use_external_inp_buf if not is_dispatch else None
            qt = self._qt_str if not is_dispatch else None
            params = TuningConfigManager.lookup(
                rules,
                self.config.data_type,
                self.config.max_num_inp_token_per_rank,
                self.config.hidden_dim,
                zero_copy=zc,
                quant_type=qt,
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        return (
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def max_num_tokens_to_recv(self):
        return self._cpp_config.max_num_tokens_to_recv()

    def max_num_tokens_to_recv_per_rank(self):
        return self._cpp_config.max_num_tokens_to_recv_per_rank()

    def max_num_tokens_to_send(self):
        return self._cpp_config.max_num_tokens_to_send()

    def max_num_tokens_to_send_per_rank(self):
        return self._cpp_config.max_num_tokens_to_send_per_rank()

    def decode_send_flat_idx(self, flat_idx):
        """Decode a flat send index into (rank, local_token_id)."""
        stride = self.max_num_tokens_to_send()
        return int(flat_idx) // stride, int(flat_idx) % stride

    def get_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        ptr, shape0, shape1 = self._get_registered_combine_input_buffer(
            self._handle, hidden_dim
        )
        return from_gpu_ptr(ptr, (shape0, shape1), dtype)

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        route_mask: torch.Tensor | None = None,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        hidden_dim = input.size(1)
        weight_ptr = weights.data_ptr() if weights is not None else 0
        has_scales = scales is not None and self.config.scale_dim > 0
        scale_ptr = scales.data_ptr() if has_scales else 0
        route_mask_ptr = 0
        if route_mask is not None:
            if self.config.kernel_type.value != EpDispatchCombineKernelType.IntraNode.value:
                raise ValueError("route_mask is currently supported only for IntraNode MORI EP")
            if route_mask.dtype != torch.uint8:
                raise TypeError("route_mask must be a torch.uint8 tensor")
            if route_mask.shape != indices.shape:
                raise ValueError(
                    f"route_mask shape {tuple(route_mask.shape)} must match indices "
                    f"shape {tuple(indices.shape)}"
                )
            if route_mask.device != indices.device:
                raise ValueError("route_mask and indices must be on the same device")
            if not route_mask.is_contiguous():
                raise ValueError("route_mask must be contiguous")
            route_mask_ptr = route_mask.data_ptr()
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        self._cached_dispatch_launch = (actual_bn, actual_rbn, actual_wpb)
        stream = _current_stream()
        self._dispatch_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
            route_mask_ptr=route_mask_ptr,
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpDispatchInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1Kernel_{sfx}",
                ],
                [mp, actual_bn],
                [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1KernelLowLatency_{sfx}",
                ],
                [mp, actual_bn],
                [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpDispatchIntraNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = WARP_SIZE * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncSendCopySlotAssign_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendCopyMultiBlock_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendTransfer_{sfx}",
                ],
                [mp_aligned, mp_aligned, self.config.world_size],
                [mb_block, mb_block, WARP_SIZE * actual_wpb],
                [0, 0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(f"Unsupported dispatch kernel_type: {kt}")

        if call_local_expert_count and kt != EpDispatchCombineKernelType.AsyncLL.value:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

        out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr = self._dispatch_out_ptrs
        max_recv = self._cpp_config.max_num_tokens_to_recv()
        out = from_gpu_ptr(out_ptr, (max_recv, hidden_dim), input.dtype)
        out_weights = from_gpu_ptr(
            outW_ptr, (max_recv, self.config.num_experts_per_token), torch.float32
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = from_gpu_ptr(
                outS_ptr, (max_recv, self.config.scale_dim), scales.dtype
            )
        out_indices = from_gpu_ptr(
            outI_ptr, (max_recv, self.config.num_experts_per_token), TOPK_IDX_DTYPE
        )
        total_recv = from_gpu_ptr(total_ptr, (1,), TOPK_IDX_DTYPE)

        return (out, out_weights, out_scales, out_indices, total_recv)

    def dispatch_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        return self.dispatch(
            input,
            weights,
            scales,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def dispatch_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        if hasattr(self, "_cached_dispatch_launch"):
            _, _, actual_wpb = self._cached_dispatch_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_dispatch_dtype"
        ), "dispatch_recv requires a prior dispatch/dispatch_send call"
        sfx = _DTYPE_SUFFIX[self._dispatch_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding dispatch_send/dispatch call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = WARP_SIZE * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncRecvTransfer_{sfx}",
                    f"EpDispatchLowLatencyAsyncRecvCopyMultiBlock_{sfx}",
                ],
                [self.config.world_size, mp_aligned],
                [WARP_SIZE * actual_wpb, mb_block],
                [0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                f"dispatch_recv only supports AsyncLL, got kernel_type={kt}"
            )

        if call_local_expert_count:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
    ):
        hidden_dim = input.size(1)
        weight_ptr = (
            weights.data_ptr() if weights is not None and weights.size(0) != 0 else 0
        )
        actual_use_ext = (
            use_external_inp_buf
            if use_external_inp_buf >= 0
            else int(self.config.use_external_inp_buf)
        )
        is_zero_copy = not actual_use_ext
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=self._get_cur_rank_num_token(self._handle),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=is_zero_copy,
            quant_type=self._qt_str,
        )
        self._cached_combine_launch = (actual_bn, actual_rbn, actual_wpb)
        stream = _current_stream()
        self._combine_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=self._get_cur_rank_num_token(self._handle),
            weight_ptr=weight_ptr,
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
            use_external_inp_buf=use_external_inp_buf,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        kt = self.config.kernel_type.value
        quant_type = _normalize_quant_type(self.config.quant_type)
        shared_mem = self._combine_shared_mem(actual_wpb)

        if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
            if kt != EpDispatchCombineKernelType.IntraNode.value:
                raise ValueError(
                    "Fp8BlockwiseQuant currently only supports IntraNode combine"
                )
            if sfx != "bf16":
                raise ValueError(f"Fp8BlockwiseQuant only supports bf16, got {sfx}")
            if not actual_use_ext:
                raise ValueError(
                    "Fp8BlockwiseQuant currently requires --zero-copy 0 "
                    "(useExternalInpBuffer=True). P2P read path not yet implemented."
                )
            if self._fp8_blockwise_combine_scale_dim <= 0:
                raise ValueError(
                    "Fp8BlockwiseQuant requires internal combine scale_dim > 0"
                )

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpCombineInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = WARP_SIZE * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1Kernel_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, WARP_SIZE, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = WARP_SIZE * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1KernelLowLatency_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, WARP_SIZE, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                # Mirror of the AccumNum=8 + VecBytes=8 specialization gating in
                # LaunchCombine() / launch.cpp. Keep in sync.
                fp8_scale_dim = self._fp8_blockwise_combine_scale_dim
                block_elems = (hidden_dim + fp8_scale_dim - 1) // fp8_scale_dim
                base_vec8_top8_eligible = (
                    weight_ptr == 0
                    and (hidden_dim % 512) == 0
                    and self.config.num_experts_per_token == 8
                    and self.config.world_size > 4
                )
                kernel_name = "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq"
                use_vec8_top8 = False
                if base_vec8_top8_eligible:
                    if block_elems == 128:
                        kernel_name = "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block128_vec8"
                        use_vec8_top8 = True
                    elif block_elems == 256:
                        kernel_name = "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block256_vec8"
                        use_vec8_top8 = True
                shared_mem = self._combine_shared_mem(
                    actual_wpb, use_weights=not use_vec8_top8
                )
                self._launch(
                    kernel_name,
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
            elif actual_use_ext:
                if (
                    sfx == "bf16"
                    and quant_type == EpDispatchCombineQuantType.Fp8DirectCast
                ):
                    self._launch(
                        "EpCombineIntraNodeKernel_bf16_nop2p_fp8cast",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
                else:
                    self._launch(
                        f"EpCombineIntraNodeKernel_{sfx}_nop2p",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
            else:
                self._launch(
                    f"EpCombineIntraNodeKernel_{sfx}_p2p",
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncSendCopy_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncSendTransfer_bf16_fp8cast",
                    ],
                    [mp_aligned, self.config.world_size],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncSendCopy_{sfx}",
                        f"EpCombineLowLatencyAsyncSendTransfer_{sfx}",
                    ],
                    [mp_aligned, self.config.world_size],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(f"Unsupported combine kernel_type: {kt}")

        out_ptr, outW_ptr = self._combine_out_ptrs
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = from_gpu_ptr(
                outW_ptr,
                (
                    self.config.max_num_inp_token_per_rank,
                    self.config.num_experts_per_token,
                ),
                weights.dtype,
            )

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def combine_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.combine(
            input,
            weights,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def combine_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        if hasattr(self, "_cached_combine_launch"):
            _, _, actual_wpb = self._cached_combine_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_combine_dtype"
        ), "combine_recv requires a prior combine/combine_send call"
        sfx = _DTYPE_SUFFIX[self._combine_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding combine_send/combine call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        shared_mem = self._combine_shared_mem(actual_wpb)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            quant_type = _normalize_quant_type(self.config.quant_type)
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncRecvTransfer_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncRecvCopy_bf16_fp8cast",
                    ],
                    [self.config.world_size, mp_aligned],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncRecvTransfer_{sfx}",
                        f"EpCombineLowLatencyAsyncRecvCopy_{sfx}",
                    ],
                    [self.config.world_size, mp_aligned],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(
                f"combine_recv only supports AsyncLL, got kernel_type={kt}"
            )

    def dispatch_standard_ep_compact_native(
        self,
        local_rows: torch.Tensor,
        local_flat_positions: torch.Tensor | None,
        local_num_tokens_per_expert: torch.Tensor,
        recv_counts_rank_major: torch.Tensor,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
        num_output_rows: int,
        flat_position_rank_stride: int,
        block_num: int = -1,
        warp_per_block: int = -1,
        return_flat_positions: bool = True,
    ):
        """Native standard-EP compact dispatch ABI.

        This is deliberately not a wrapper around ``dispatch_standard_moe``.
        The caller has already built the hot/cold/helper partition and the
        compact cold/normal row stream; MORI must move that stream directly and
        carry flat source positions for combine.
        """

        launch_fn = _cpp_dispatch_combine_factory(
            "launch_standard_ep_compact_dispatch", allow_missing=True
        )
        if launch_fn is None:
            raise RuntimeError(
                "dispatch_standard_ep_compact_native is not available. "
                "Rebuild MORI with the native compact-row dispatch ABI."
            )
        _require_cuda_contiguous_tensor("local_rows", local_rows, ndim=2)
        local_flat_positions_ptr = 0
        if return_flat_positions:
            if local_flat_positions is None:
                raise ValueError(
                    "local_flat_positions is required when return_flat_positions=True"
                )
            _require_cuda_contiguous_tensor(
                "local_flat_positions", local_flat_positions, dtype=torch.int64, ndim=1
            )
            if local_rows.size(0) != local_flat_positions.numel():
                raise ValueError(
                    "local_rows and local_flat_positions must describe the same rows"
                )
            local_flat_positions_ptr = _non_null_1d_ptr_tensor(
                local_flat_positions
            ).data_ptr()
        _require_cuda_contiguous_tensor(
            "local_num_tokens_per_expert",
            local_num_tokens_per_expert,
            dtype=torch.int64,
            ndim=1,
        )
        _require_cuda_contiguous_tensor(
            "recv_counts_rank_major", recv_counts_rank_major, dtype=torch.int64, ndim=1
        )
        _require_compact_split_tensors(input_splits, output_splits)
        if input_splits.numel() != self.config.world_size:
            raise ValueError("input_splits length must match EP world_size")
        if output_splits.numel() != self.config.world_size:
            raise ValueError("output_splits length must match EP world_size")
        num_output_rows = int(num_output_rows)
        if num_output_rows < 0:
            raise ValueError("num_output_rows must be non-negative")
        flat_position_rank_stride = int(flat_position_rank_stride)
        if flat_position_rank_stride < 0:
            raise ValueError("flat_position_rank_stride must be non-negative")

        hidden_dim = int(local_rows.size(1))
        local_rows_for_ptr = _non_null_2d_ptr_tensor(local_rows)
        rank_major_rows = torch.empty(
            (max(1, num_output_rows), hidden_dim),
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        rank_major_flat_positions = None
        rank_major_flat_positions_ptr = 0
        if return_flat_positions:
            rank_major_flat_positions = torch.empty(
                (max(1, num_output_rows),),
                dtype=torch.int64,
                device=local_rows.device,
            )
            rank_major_flat_positions_ptr = rank_major_flat_positions.data_ptr()
        launch_fn(
            self._handle,
            dtype_to_int(local_rows.dtype),
            local_rows_for_ptr.data_ptr(),
            local_flat_positions_ptr,
            local_num_tokens_per_expert.data_ptr(),
            recv_counts_rank_major.data_ptr(),
            input_splits.data_ptr(),
            output_splits.data_ptr(),
            recv_counts_rank_major.numel(),
            local_rows.size(0),
            num_output_rows,
            rank_major_rows.data_ptr(),
            rank_major_flat_positions_ptr,
            flat_position_rank_stride,
            block_num,
            warp_per_block,
            _current_stream(),
            hidden_dim,
        )
        return rank_major_rows[:num_output_rows], (
            rank_major_flat_positions[:num_output_rows]
            if rank_major_flat_positions is not None
            else None
        )

    def combine_standard_ep_compact_native(
        self,
        expert_major_rows: torch.Tensor,
        expert_major_flat_positions: torch.Tensor | None,
        expert_major_to_rank_major_indices: torch.Tensor | None,
        recv_counts_rank_major: torch.Tensor | None,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
        num_output_rows: int,
        block_num: int = -1,
        warp_per_block: int = -1,
        return_flat_positions: bool = True,
        top_scores_flat: torch.Tensor | None = None,
        top_k: int = 0,
        flat_position_offset: int = 0,
        token_output_rows: int = 0,
        return_token_output: bool = False,
    ):
        """Native standard-EP compact combine ABI.

        The input rows are already expert-major after local expert compute. The
        native MORI combine returns source-rank compact rows plus the source
        flat positions; downstream weighted scatter can then reuse the existing
        standard-EP semantics without re-reading raw top-k routing.
        """

        launch_fn = _cpp_dispatch_combine_factory(
            "launch_standard_ep_compact_combine", allow_missing=True
        )
        if launch_fn is None:
            raise RuntimeError(
                "combine_standard_ep_compact_native is not available. "
                "Rebuild MORI with the native compact-row combine ABI."
            )
        _require_cuda_contiguous_tensor("expert_major_rows", expert_major_rows, ndim=2)
        expert_major_flat_positions_ptr = 0
        if return_flat_positions:
            if expert_major_flat_positions is None:
                raise ValueError(
                    "expert_major_flat_positions is required when "
                    "return_flat_positions=True"
                )
            _require_cuda_contiguous_tensor(
                "expert_major_flat_positions",
                expert_major_flat_positions,
                dtype=torch.int64,
                ndim=1,
            )
            if expert_major_rows.size(0) != expert_major_flat_positions.numel():
                raise ValueError(
                    "expert_major_rows and expert_major_flat_positions must describe the same rows"
                )
            expert_major_flat_positions_ptr = _non_null_1d_ptr_tensor(
                expert_major_flat_positions
            ).data_ptr()
        recv_counts_rank_major_ptr = 0
        num_segments = 0
        if recv_counts_rank_major is not None:
            _require_cuda_contiguous_tensor(
                "recv_counts_rank_major",
                recv_counts_rank_major,
                dtype=torch.int64,
                ndim=1,
            )
            recv_counts_rank_major_ptr = recv_counts_rank_major.data_ptr()
            num_segments = int(recv_counts_rank_major.numel())
            if num_segments <= 0:
                raise ValueError(
                    "recv_counts_rank_major must be non-empty when provided"
                )
        elif expert_major_to_rank_major_indices is None:
            raise ValueError(
                "expert_major_to_rank_major_indices is required when "
                "recv_counts_rank_major is not provided"
            )
        expert_major_to_rank_major_indices_ptr = 0
        if expert_major_to_rank_major_indices is not None:
            _require_cuda_contiguous_tensor(
                "expert_major_to_rank_major_indices",
                expert_major_to_rank_major_indices,
                dtype=torch.int64,
                ndim=1,
            )
            expert_major_to_rank_major_indices_ptr = (
                expert_major_to_rank_major_indices.data_ptr()
            )
        _require_compact_split_tensors(input_splits, output_splits)
        if (
            expert_major_to_rank_major_indices is not None
            and expert_major_rows.size(0) != expert_major_to_rank_major_indices.numel()
        ):
            raise ValueError(
                "expert_major_rows and expert_major_to_rank_major_indices must describe the same rows"
            )
        if input_splits.numel() != self.config.world_size:
            raise ValueError("input_splits length must match EP world_size")
        if output_splits.numel() != self.config.world_size:
            raise ValueError("output_splits length must match EP world_size")
        num_output_rows = int(num_output_rows)
        if num_output_rows < 0:
            raise ValueError("num_output_rows must be non-negative")

        hidden_dim = int(expert_major_rows.size(1))
        top_scores_flat_ptr = 0
        top_scores_flat_size = 0
        token_output_ptr = 0
        token_output = None
        if return_token_output:
            if not return_flat_positions:
                raise ValueError(
                    "return_token_output requires return_flat_positions=True"
                )
            if top_scores_flat is None:
                raise ValueError("return_token_output requires top_scores_flat")
            _require_cuda_contiguous_tensor(
                "top_scores_flat",
                top_scores_flat,
                dtype=torch.float32,
                ndim=1,
            )
            top_k = int(top_k)
            flat_position_offset = int(flat_position_offset)
            token_output_rows = int(token_output_rows)
            if top_k <= 0:
                raise ValueError("return_token_output requires top_k > 0")
            if token_output_rows <= 0:
                raise ValueError(
                    "return_token_output requires token_output_rows > 0"
                )
            top_scores_flat_ptr = top_scores_flat.data_ptr()
            top_scores_flat_size = int(top_scores_flat.numel())
            token_output = torch.empty(
                (max(1, token_output_rows), hidden_dim),
                dtype=expert_major_rows.dtype,
                device=expert_major_rows.device,
            )
            token_output_ptr = token_output.data_ptr()
        expert_major_rows_for_ptr = _non_null_2d_ptr_tensor(expert_major_rows)
        source_rank_rows = torch.empty(
            (max(1, num_output_rows), hidden_dim),
            dtype=expert_major_rows.dtype,
            device=expert_major_rows.device,
        )
        source_rank_flat_positions = None
        source_rank_flat_positions_ptr = 0
        if return_flat_positions:
            source_rank_flat_positions = torch.empty(
                (max(1, num_output_rows),),
                dtype=torch.int64,
                device=expert_major_rows.device,
            )
            source_rank_flat_positions_ptr = source_rank_flat_positions.data_ptr()
        launch_fn(
            self._handle,
            dtype_to_int(expert_major_rows.dtype),
            expert_major_rows_for_ptr.data_ptr(),
            expert_major_flat_positions_ptr,
            expert_major_to_rank_major_indices_ptr,
            recv_counts_rank_major_ptr,
            input_splits.data_ptr(),
            output_splits.data_ptr(),
            num_segments,
            expert_major_rows.size(0),
            num_output_rows,
            source_rank_rows.data_ptr(),
            source_rank_flat_positions_ptr,
            top_scores_flat_ptr,
            top_scores_flat_size,
            int(top_k),
            int(flat_position_offset),
            token_output_ptr,
            int(token_output_rows),
            block_num,
            warp_per_block,
            _current_stream(),
            hidden_dim,
        )
        result = source_rank_rows[:num_output_rows], (
            source_rank_flat_positions[:num_output_rows]
            if source_rank_flat_positions is not None
            else None
        )
        if return_token_output:
            return result[0], result[1], token_output[:token_output_rows]
        return result

    def combine_balanced_moe_compact(
        self,
        expert_major_rows: torch.Tensor,
        expert_major_flat_positions: torch.Tensor | None,
        expert_major_to_rank_major_indices: torch.Tensor | None,
        recv_counts_rank_major: torch.Tensor | None,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
        num_output_rows: int,
        block_num: int = -1,
        warp_per_block: int = -1,
        return_flat_positions: bool = True,
        top_scores_flat: torch.Tensor | None = None,
        top_k: int = 0,
        flat_position_offset: int = 0,
        token_output_rows: int = 0,
        return_token_output: bool = False,
    ) -> BalancedMoeCompactCombineOutput:
        """Combine balanced-MoE normal/cold compact rows.

        This is the MORI-owned balanced-MoE return ABI matching
        :meth:`dispatch_balanced_moe_compact`.  It delegates to the existing
        native compact combine implementation and only wraps the returned tuple
        in a named object, so it does not add extra row movement.
        """

        output = self.combine_standard_ep_compact_native(
            expert_major_rows,
            expert_major_flat_positions,
            expert_major_to_rank_major_indices,
            recv_counts_rank_major,
            input_splits,
            output_splits,
            num_output_rows,
            block_num=block_num,
            warp_per_block=warp_per_block,
            return_flat_positions=return_flat_positions,
            top_scores_flat=top_scores_flat,
            top_k=top_k,
            flat_position_offset=flat_position_offset,
            token_output_rows=token_output_rows,
            return_token_output=return_token_output,
        )
        if return_token_output:
            source_rank_rows, source_rank_flat_positions, token_output = output
        else:
            source_rank_rows, source_rank_flat_positions = output
            token_output = None
        return BalancedMoeCompactCombineOutput(
            source_rank_rows=source_rank_rows,
            source_rank_flat_positions=source_rank_flat_positions,
            token_output=token_output,
        )

    def standard_ep_compact_weighted_output_backward_native(
        self,
        source_rank_rows: torch.Tensor,
        source_rank_flat_positions: torch.Tensor,
        top_scores_flat: torch.Tensor,
        grad_token_output: torch.Tensor,
        *,
        top_k: int,
        flat_position_offset: int = 0,
        block_num: int = -1,
        warp_per_block: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native inverse of compact combine's weighted token-output path."""

        launch_fn = _cpp_dispatch_combine_factory(
            "launch_standard_ep_compact_weighted_output_backward",
            allow_missing=True,
        )
        if launch_fn is None:
            raise RuntimeError(
                "standard_ep_compact_weighted_output_backward_native is not available. "
                "Rebuild MORI with the native compact weighted-output backward ABI."
            )
        _require_cuda_contiguous_tensor("source_rank_rows", source_rank_rows, ndim=2)
        _require_cuda_contiguous_tensor(
            "source_rank_flat_positions",
            source_rank_flat_positions,
            dtype=torch.int64,
            ndim=1,
        )
        _require_cuda_contiguous_tensor(
            "top_scores_flat",
            top_scores_flat,
            dtype=torch.float32,
            ndim=1,
        )
        _require_cuda_contiguous_tensor(
            "grad_token_output",
            grad_token_output,
            dtype=source_rank_rows.dtype,
            ndim=2,
        )
        if source_rank_rows.size(0) != source_rank_flat_positions.numel():
            raise ValueError(
                "source_rank_rows and source_rank_flat_positions must describe the same rows"
            )
        if source_rank_rows.size(1) != grad_token_output.size(1):
            raise ValueError(
                "source_rank_rows and grad_token_output hidden dimensions must match"
            )
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_scores_flat.numel() <= 0:
            raise ValueError("top_scores_flat must be non-empty")
        if grad_token_output.size(0) <= 0:
            raise ValueError("grad_token_output must have at least one token row")

        num_rows = int(source_rank_rows.size(0))
        hidden_dim = int(source_rank_rows.size(1))
        source_rank_rows_for_ptr = _non_null_2d_ptr_tensor(source_rank_rows)
        source_rank_flat_positions_for_ptr = _non_null_1d_ptr_tensor(
            source_rank_flat_positions
        )
        grad_source_rank_rows = torch.empty_like(source_rank_rows_for_ptr)
        grad_top_scores_flat = torch.empty_like(top_scores_flat)
        launch_fn(
            self._handle,
            dtype_to_int(source_rank_rows.dtype),
            source_rank_rows_for_ptr.data_ptr(),
            source_rank_flat_positions_for_ptr.data_ptr(),
            top_scores_flat.data_ptr(),
            int(top_scores_flat.numel()),
            top_k,
            int(flat_position_offset),
            grad_token_output.data_ptr(),
            int(grad_token_output.size(0)),
            num_rows,
            grad_source_rank_rows.data_ptr(),
            grad_top_scores_flat.data_ptr(),
            block_num,
            warp_per_block,
            _current_stream(),
            hidden_dim,
        )
        return grad_source_rank_rows[:num_rows], grad_top_scores_flat

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        route_mask: torch.Tensor | None = None,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "dispatch_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        hidden_dim = input.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        route_mask_ptr = 0
        if route_mask is not None:
            if self.config.kernel_type.value != EpDispatchCombineKernelType.IntraNode.value:
                raise ValueError(
                    "dispatch_standard_moe route_mask is currently supported only "
                    "for IntraNode MORI EP"
                )
            if route_mask.dtype != torch.uint8:
                raise TypeError("route_mask must be a torch.uint8 tensor")
            if route_mask.shape != indices.shape:
                raise ValueError(
                    f"route_mask shape {tuple(route_mask.shape)} must match indices "
                    f"shape {tuple(indices.shape)}"
                )
            if route_mask.device != indices.device:
                raise ValueError("route_mask and indices must be on the same device")
            if not route_mask.is_contiguous():
                raise ValueError("route_mask must be contiguous")
            route_mask_ptr = route_mask.data_ptr()
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=input.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=input.device
        )

        set_fn(self._handle, packed_recv_x.data_ptr(), packed_recv_src_info.data_ptr())

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=(weights.data_ptr() if weights is not None else 0),
            scale_ptr=(
                scales.data_ptr()
                if scales is not None and self.config.scale_dim > 0
                else 0
            ),
            indices_ptr=indices.data_ptr(),
            route_mask_ptr=route_mask_ptr,
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(
                f"EpDispatchCopyToStaging_{sfx}", (mp,), block, 0, stream, args_ptr
            )
            self._launch(
                f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpDispatchIntraNodeKernel_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "dispatch_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def dispatch_hotcold_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        normal_route_mask: torch.Tensor,
        hot_flat_positions: torch.Tensor,
        hot_owner_slots: torch.Tensor,
        num_hot_slots: int,
        max_hot_rows_per_slot: int,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        pack_fn = _cpp_dispatch_combine_factory(
            "launch_hot_helper_pack", allow_missing=True
        )
        if pack_fn is None:
            raise RuntimeError(
                "dispatch_hotcold_standard_moe is not available. "
                "Rebuild MORI with the hot-helper pack kernel."
            )
        if weights is None:
            raise ValueError("dispatch_hotcold_standard_moe requires top-k weights")
        if weights.dtype != torch.float32:
            raise TypeError("hot-helper packed scores require float32 top-k weights")
        if hot_flat_positions.dtype != torch.int64:
            raise TypeError("hot_flat_positions must be torch.int64")
        if hot_owner_slots.dtype != torch.int64:
            raise TypeError("hot_owner_slots must be torch.int64")
        if hot_flat_positions.device != input.device:
            raise ValueError("hot_flat_positions and input must be on the same device")
        if hot_owner_slots.device != input.device:
            raise ValueError("hot_owner_slots and input must be on the same device")
        if hot_flat_positions.numel() != hot_owner_slots.numel():
            raise ValueError(
                "hot_flat_positions and hot_owner_slots must have the same length"
            )
        if not hot_flat_positions.is_contiguous():
            raise ValueError("hot_flat_positions must be contiguous")
        if not hot_owner_slots.is_contiguous():
            raise ValueError("hot_owner_slots must be contiguous")
        if num_hot_slots <= 0:
            raise ValueError("num_hot_slots must be positive")
        if max_hot_rows_per_slot <= 0:
            raise ValueError("max_hot_rows_per_slot must be positive")

        normal_outputs = self.dispatch_standard_moe(
            input,
            weights,
            scales,
            indices,
            route_mask=normal_route_mask,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        hidden_dim = input.size(1)
        hot_packed_x = torch.empty(
            (num_hot_slots, max_hot_rows_per_slot, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        hot_packed_scores = torch.empty(
            (num_hot_slots, max_hot_rows_per_slot),
            dtype=torch.float32,
            device=input.device,
        )
        hot_packed_src_info = torch.empty(
            (num_hot_slots, max_hot_rows_per_slot),
            dtype=torch.int64,
            device=input.device,
        )
        hot_packed_count = torch.empty(
            (num_hot_slots,),
            dtype=torch.int32,
            device=input.device,
        )

        if hot_flat_positions.numel() > 0:
            pack_fn(
                self._cpp_config,
                dtype_to_int(input.dtype),
                input.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                hot_flat_positions.numel(),
                hot_packed_x.data_ptr(),
                hot_packed_scores.data_ptr(),
                hot_packed_src_info.data_ptr(),
                hot_packed_count.data_ptr(),
                num_hot_slots,
                max_hot_rows_per_slot,
                block_num,
                warp_per_block,
                _current_stream(),
                hidden_dim,
            )
        else:
            hot_packed_count.zero_()

        return (
            *normal_outputs,
            hot_packed_x,
            hot_packed_scores,
            hot_packed_src_info,
            hot_packed_count,
        )

    def dispatch_hotcold_standard_moe_compact(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        normal_route_mask: torch.Tensor,
        hot_flat_positions: torch.Tensor,
        hot_owner_slots: torch.Tensor,
        num_hot_slots: int,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        compute_hot_counts: bool = False,
    ):
        """Dispatch normal rows and materialize helper-hot rows compactly.

        This variant matches TorchTitan's fast standard-EP hot-helper layout:
        helper rows are already presorted by the caller's hot partition plan and
        stored as ``[num_hot_routes, hidden]``. Counts/group ends should be
        reused from that plan; setting ``compute_hot_counts`` is diagnostic only
        because per-route atomics are measurably slower.
        """

        pack_fn = _cpp_dispatch_combine_factory(
            "launch_hot_helper_compact_pack", allow_missing=True
        )
        if pack_fn is None:
            raise RuntimeError(
                "dispatch_hotcold_standard_moe_compact is not available. "
                "Rebuild MORI with the compact hot-helper pack kernel."
            )
        if weights is None:
            raise ValueError("dispatch_hotcold_standard_moe_compact requires top-k weights")
        if weights.dtype != torch.float32:
            raise TypeError("hot-helper packed scores require float32 top-k weights")
        if hot_flat_positions.dtype != torch.int64:
            raise TypeError("hot_flat_positions must be torch.int64")
        if hot_owner_slots.dtype != torch.int64:
            raise TypeError("hot_owner_slots must be torch.int64")
        if hot_flat_positions.device != input.device:
            raise ValueError("hot_flat_positions and input must be on the same device")
        if hot_owner_slots.device != input.device:
            raise ValueError("hot_owner_slots and input must be on the same device")
        if hot_flat_positions.numel() != hot_owner_slots.numel():
            raise ValueError(
                "hot_flat_positions and hot_owner_slots must have the same length"
            )
        if not hot_flat_positions.is_contiguous():
            raise ValueError("hot_flat_positions must be contiguous")
        if not hot_owner_slots.is_contiguous():
            raise ValueError("hot_owner_slots must be contiguous")
        if num_hot_slots <= 0:
            raise ValueError("num_hot_slots must be positive")

        normal_outputs = self.dispatch_standard_moe(
            input,
            weights,
            scales,
            indices,
            route_mask=normal_route_mask,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )

        hidden_dim = input.size(1)
        num_hot_routes = int(hot_flat_positions.numel())
        hot_packed_x = torch.empty(
            (num_hot_routes, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        hot_packed_scores = torch.empty(
            (num_hot_routes,),
            dtype=torch.float32,
            device=input.device,
        )
        hot_packed_src_info = torch.empty(
            (num_hot_routes,),
            dtype=torch.int64,
            device=input.device,
        )
        hot_packed_count = (
            torch.empty((num_hot_slots,), dtype=torch.int32, device=input.device)
            if compute_hot_counts
            else None
        )

        if num_hot_routes > 0:
            pack_fn(
                self._cpp_config,
                dtype_to_int(input.dtype),
                input.data_ptr(),
                weights.data_ptr(),
                hot_flat_positions.data_ptr(),
                hot_owner_slots.data_ptr(),
                num_hot_routes,
                hot_packed_x.data_ptr(),
                hot_packed_scores.data_ptr(),
                hot_packed_src_info.data_ptr(),
                hot_packed_count.data_ptr() if hot_packed_count is not None else 0,
                num_hot_slots,
                block_num,
                warp_per_block,
                _current_stream(),
                hidden_dim,
            )
        elif hot_packed_count is not None:
            hot_packed_count.zero_()

        return (
            *normal_outputs,
            hot_packed_x,
            hot_packed_scores,
            hot_packed_src_info,
            hot_packed_count,
        )

    def dispatch_balanced_moe_compact(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        source_partition,
        num_hot_slots: int | None = None,
        *,
        hot_slot_kind: str = "owner_compact",
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        compute_hot_counts: bool = False,
    ) -> BalancedMoeCompactDispatchOutput:
        """Dispatch balanced-MoE normal/cold rows and compact helper-hot rows.

        ``source_partition`` is produced by ``mori.ops.balanced_moe``.  This is
        the MORI-owned API boundary for callers that already have the
        hot/cold/helper route plan: the caller passes the partition object
        directly instead of unpacking masks, flat positions, and owner offsets
        into a lower-level hot/cold function.
        """

        hot_slot_kind = hot_slot_kind.strip().lower()
        if hot_slot_kind == "":
            hot_slot_kind = "owner_compact"
        if hot_slot_kind not in {"owner_compact", "owner_shard", "selected"}:
            raise ValueError(
                "hot_slot_kind must be 'owner_compact', 'owner_shard', or "
                f"'selected', got {hot_slot_kind!r}"
            )

        try:
            normal_route_mask = source_partition.normal_route_mask
            hot_flat_positions = source_partition.remote_flat_positions
            hot_group_ends = source_partition.remote_group_ends
            hot_offsets_presorted_by = source_partition.remote_offsets_presorted_by
        except AttributeError as exc:
            raise TypeError(
                "source_partition must be a BalancedMoeSourcePartition-like object"
            ) from exc

        if hot_slot_kind == "owner_compact":
            hot_owner_slots = source_partition.remote_owner_compact_offsets
        elif hot_slot_kind == "owner_shard":
            hot_owner_slots = source_partition.remote_owner_shard_offsets
        else:
            hot_owner_slots = source_partition.remote_hot_offsets

        if num_hot_slots is None:
            if hot_group_ends is not None:
                num_hot_slots = int(hot_group_ends.numel())
            elif hot_owner_slots.numel() > 0:
                num_hot_slots = int(hot_owner_slots.max().item()) + 1
            else:
                num_hot_slots = 1
        num_hot_slots = max(1, int(num_hot_slots))

        outputs = self.dispatch_hotcold_standard_moe_compact(
            input,
            weights,
            scales,
            indices,
            normal_route_mask=normal_route_mask,
            hot_flat_positions=hot_flat_positions,
            hot_owner_slots=hot_owner_slots,
            num_hot_slots=num_hot_slots,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
            compute_hot_counts=compute_hot_counts,
        )
        (
            normal_packed_recv_x,
            normal_packed_recv_count,
            normal_packed_recv_src_info,
            normal_packed_recv_layout_range,
            hot_packed_x,
            hot_packed_scores,
            hot_packed_src_info,
            hot_packed_count,
        ) = outputs
        return BalancedMoeCompactDispatchOutput(
            normal_packed_recv_x=normal_packed_recv_x,
            normal_packed_recv_count=normal_packed_recv_count,
            normal_packed_recv_src_info=normal_packed_recv_src_info,
            normal_packed_recv_layout_range=normal_packed_recv_layout_range,
            hot_packed_x=hot_packed_x,
            hot_packed_scores=hot_packed_scores,
            hot_packed_src_info=hot_packed_src_info,
            hot_packed_count=hot_packed_count,
            hot_group_ends=hot_group_ends,
            hot_offsets_presorted_by=str(hot_offsets_presorted_by),
        )

    def combine_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "combine_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        hidden_dim = input.size(2)
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=self._get_cur_rank_num_token(self._handle),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=False,
            quant_type=self._qt_str,
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        set_fn(self._handle, input.data_ptr(), 0)

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=self._get_cur_rank_num_token(self._handle),
            weight_ptr=(
                weights.data_ptr()
                if weights is not None and weights.size(0) != 0
                else 0
            ),
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._combine_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(f"EpCombineSync_{sfx}", (mp,), block, 0, stream, args_ptr)
            self._launch(
                f"EpCombineSyncBarrier_{sfx}", (1,), (WARP_SIZE,), 0, stream, args_ptr
            )
            self._launch(
                f"EpCombineInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
            self._launch(
                f"EpCombineAll_{sfx}", (mp,), block, shared_mem, stream, args_ptr
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpCombineIntraNodeKernel_{sfx}_p2p_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "combine_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        out_ptr = self._combine_out_ptrs[0]
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def convert_dispatch_output(
        self,
        dispatch_out_x: torch.Tensor,
        dispatch_out_topk_idx: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_dispatch_output_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_dispatch_output is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = dispatch_out_x.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=dispatch_out_x.dtype,
            device=dispatch_out_x.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=dispatch_out_x.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=dispatch_out_x.device
        )

        args_ptr = build_fn(
            self._handle,
            dispatch_out_x.data_ptr(),
            dispatch_out_topk_idx.data_ptr(),
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
        )
        try:
            grid = (actual_bn,)
            block = (WARP_SIZE * actual_wpb,)
            self._launch(
                "mori_ConvertDispatchOutputKernel", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def convert_combine_input(
        self,
        packed_recv_x: torch.Tensor,
        packed_recv_src_info: torch.Tensor,
        packed_recv_layout_range: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_combine_input_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_combine_input is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = packed_recv_x.size(2)
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[packed_recv_x.dtype]

        args_ptr = build_fn(
            self._handle,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
        )
        try:
            grid = (actual_bn,)
            block = (WARP_SIZE * actual_wpb,)
            self._launch(
                f"ConvertCombineInputKernel_{sfx}", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        max_recv = self._cpp_config.max_num_tokens_to_recv()
        combine_input_ptr = mori_cpp.get_combine_input_ptr(self._handle)
        return from_gpu_ptr(
            combine_input_ptr, (max_recv, hidden_dim), packed_recv_x.dtype
        )

    def reset(self):
        self._reset_func(self._handle, _current_stream())

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
            EpDispatchCombineKernelType.InterNodeV1LL.value,
            EpDispatchCombineKernelType.AsyncLL.value,
        ):
            ptr, size = self._get_dispatch_src_token_pos_func(self._handle)
            return from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_sender_token_idx_map_func(self._handle)
        dispatch_sender_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_receiver_token_idx_map_func(self._handle)
        dispatch_receiver_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for i in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            src_token_pos.append(src_pe * max_num_token_to_send_per_rank + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

    def get_debug_time_buf(self):
        """Get the debug time buffer as a torch.Tensor (int64)."""
        ptr, size = mori_cpp.get_debug_time_buf(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int64)

    def get_debug_time_offset(self):
        """Get the debug time offset buffer as a torch.Tensor (int32)."""
        ptr, size = mori_cpp.get_debug_time_offset(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int32)
