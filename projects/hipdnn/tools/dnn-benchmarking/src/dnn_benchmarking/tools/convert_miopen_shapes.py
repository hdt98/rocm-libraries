# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convert MIOpen driver shape files to hipDNN JSON graph files.

Usage:
    python convert_miopen_shapes.py graphs/shapes.txt graphs/shapes_3D.txt
    python convert_miopen_shapes.py shapes.txt --outdir graphs/generic_convolutions/
"""

import argparse
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stride helpers
# ---------------------------------------------------------------------------


def _nchw_strides(N: int, C: int, H: int, W: int) -> List[int]:
    return [C * H * W, H * W, W, 1]


def _nhwc_strides(N: int, C: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, H, W]; strides reflect NHWC memory order
    return [H * W * C, 1, W * C, C]


def _ncdhw_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    return [C * D * H * W, D * H * W, H * W, W, 1]


def _ndhwc_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, D, H, W]; strides reflect NDHWC memory order
    return [D * H * W * C, 1, H * W * C, W * C, C]


def _input_strides(
    layout: str, N: int, C: int, H: int, W: int, D: Optional[int] = None
) -> List[int]:
    """Return strides for an input tensor given its memory layout."""
    if D is not None:
        if layout == "NDHWC":
            return _ndhwc_strides(N, C, D, H, W)
        return _ncdhw_strides(N, C, D, H, W)
    if layout == "NHWC":
        return _nhwc_strides(N, C, H, W)
    return _nchw_strides(N, C, H, W)


def _weight_strides(
    K: int, Cg: int, R: int, S: int, D: Optional[int] = None
) -> List[int]:
    """Weight strides are always row-major KCRS (or KCDRS for 3D)."""
    if D is not None:
        return [Cg * D * R * S, D * R * S, R * S, S, 1]
    return [Cg * R * S, R * S, S, 1]


# ---------------------------------------------------------------------------
# Output dimension formula
# ---------------------------------------------------------------------------


def _conv_out_dim(
    dim_in: int, pad: int, dilation: int, kernel: int, stride: int
) -> int:
    return math.floor((dim_in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)


# ---------------------------------------------------------------------------
# MIOpen argument parsers
# ---------------------------------------------------------------------------


def _is_flag(token: str) -> bool:
    """Return True if token looks like a CLI flag (e.g. -n, --layout).

    Negative numbers like -1 or -0.5 are values, not flags.
    """
    if not token.startswith("-"):
        return False
    rest = token[1:]
    if rest and (rest[0].isdigit() or rest[0] == "."):
        return False
    return True


def _parse_args(tokens: List[str]) -> Dict[str, str]:
    """Parse a flat list of flag/value tokens into a dict."""
    result: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_flag(tok):
            # Check if next token is a value (not a flag)
            if i + 1 < len(tokens) and not _is_flag(tokens[i + 1]):
                result[tok] = tokens[i + 1]
                i += 2
            else:
                result[tok] = "1"  # boolean flag
                i += 1
        else:
            i += 1
    return result


def _int(args: Dict[str, str], key: str, default: int = 0) -> int:
    return int(args.get(key, default))


# ---------------------------------------------------------------------------
# Build tensor objects
# ---------------------------------------------------------------------------


def _make_tensor(
    uid: int,
    name: str,
    dims: List[int],
    strides: List[int],
    data_type: str = "bfloat16",
    virtual: bool = False,
) -> Dict[str, Any]:
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": strides,
        "data_type": data_type,
        "virtual": virtual,
    }


# ---------------------------------------------------------------------------
# Convolution parameter container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ConvParams:
    """Parsed convolution parameters extracted from MIOpen driver args."""

    N: int
    C: int
    H: int
    W: int
    K: int
    R: int
    S: int
    pad_h: int
    pad_w: int
    stride_h: int
    stride_w: int
    dil_h: int
    dil_w: int
    groups: int
    F: int
    spatial_dim: int
    in_layout: str
    out_layout: str
    D: Optional[int] = None
    D_f: Optional[int] = None
    pad_d: int = 0
    stride_d: int = 1
    dil_d: int = 1

    @classmethod
    def from_args(cls, args: Dict[str, str]) -> "_ConvParams":
        """Parse MIOpen convolution args into a _ConvParams instance."""
        spatial_dim = _int(args, "--spatial_dim", 2)
        is_3d = spatial_dim == 3
        D: Optional[int] = None
        D_f: Optional[int] = None
        pad_d = 0
        stride_d = 1
        dil_d = 1
        if is_3d:
            D = _int(args, "--in_d", 1)
            D_f = _int(args, "--fil_d", 1)
            pad_d = _int(args, "--pad_d", 0)
            stride_d = _int(args, "--conv_stride_d", 1)
            dil_d = _int(args, "--dilation_d", 1)
        C = _int(args, "-c", 1)
        K = _int(args, "-k", 1)
        groups = _int(args, "-g", 1)
        if C % groups != 0:
            raise ValueError(
                f"Invalid convolution: C={C} is not divisible by groups={groups}"
            )
        if K % groups != 0:
            raise ValueError(
                f"Invalid convolution: K={K} is not divisible by groups={groups}"
            )
        return cls(
            N=_int(args, "-n", 1),
            C=C,
            H=_int(args, "-H", 1),
            W=_int(args, "-W", 1),
            K=K,
            R=_int(args, "-y", 1),
            S=_int(args, "-x", 1),
            pad_h=_int(args, "-p", 0),
            pad_w=_int(args, "-q", 0),
            stride_h=_int(args, "-u", 1),
            stride_w=_int(args, "-v", 1),
            dil_h=_int(args, "-l", 1),
            dil_w=_int(args, "-j", 1),
            groups=groups,
            F=_int(args, "-F", 1),
            spatial_dim=spatial_dim,
            in_layout=args.get("--in_layout", "NCHW"),
            out_layout=args.get("--out_layout", "NCHW"),
            D=D,
            D_f=D_f,
            pad_d=pad_d,
            stride_d=stride_d,
            dil_d=dil_d,
        )


# ---------------------------------------------------------------------------
# Convolution conversion
# ---------------------------------------------------------------------------


def _conv_direction_label(F: int) -> str:
    return {1: "fwd", 2: "dgrad", 4: "wgrad"}.get(F, f"F{F}")


def _conv_node_type(F: int) -> str:
    return {
        1: "ConvolutionFwdAttributes",
        2: "ConvolutionBwdAttributes",
        4: "ConvolutionWrwAttributes",
    }.get(F, "ConvolutionFwdAttributes")


def _build_conv_json(p: _ConvParams) -> Dict[str, Any]:
    """Build a hipDNN JSON graph dict from a _ConvParams instance."""
    is_3d = p.spatial_dim == 3
    Cg = p.C // p.groups  # channels per group for weight tensor

    # Compute output spatial dims
    H_out = _conv_out_dim(p.H, p.pad_h, p.dil_h, p.R, p.stride_h)
    W_out = _conv_out_dim(p.W, p.pad_w, p.dil_w, p.S, p.stride_w)
    D_out: Optional[int] = None
    if is_3d and p.D is not None and p.D_f is not None:
        D_out = _conv_out_dim(p.D, p.pad_d, p.dil_d, p.D_f, p.stride_d)

    # Build dims in canonical NCHW / NCDHW order
    if is_3d and p.D is not None and p.D_f is not None and D_out is not None:
        x_dims = [p.N, p.C, p.D, p.H, p.W]
        w_dims = [p.K, Cg, p.D_f, p.R, p.S]
        y_dims = [p.N, p.K, D_out, H_out, W_out]
    else:
        x_dims = [p.N, p.C, p.H, p.W]
        w_dims = [p.K, Cg, p.R, p.S]
        y_dims = [p.N, p.K, H_out, W_out]

    x_strides = _input_strides(p.in_layout, p.N, p.C, p.H, p.W, p.D)
    w_strides = _weight_strides(p.K, Cg, p.R, p.S, p.D_f)
    y_strides = _input_strides(p.out_layout, p.N, p.K, H_out, W_out, D_out)

    node_type = _conv_node_type(p.F)

    if is_3d and p.D_f is not None:
        pre_pad = [p.pad_d, p.pad_h, p.pad_w]
        post_pad = [p.pad_d, p.pad_h, p.pad_w]
        stride_list = [p.stride_d, p.stride_h, p.stride_w]
        dil_list = [p.dil_d, p.dil_h, p.dil_w]
    else:
        pre_pad = [p.pad_h, p.pad_w]
        post_pad = [p.pad_h, p.pad_w]
        stride_list = [p.stride_h, p.stride_w]
        dil_list = [p.dil_h, p.dil_w]

    # Wire up inputs/outputs differently per direction
    if p.F == 1:  # forward: x, w → y
        tensors = [
            _make_tensor(0, "output_y", y_dims, y_strides),
            _make_tensor(1, "input_x", x_dims, x_strides),
            _make_tensor(2, "weight_w", w_dims, w_strides),
        ]
        node_inputs = {"x_tensor_uid": 1, "w_tensor_uid": 2}
        node_outputs = {"y_tensor_uid": 0}
    elif p.F == 2:  # dgrad: dy, w → dx
        tensors = [
            _make_tensor(0, "output_dx", x_dims, x_strides),
            _make_tensor(1, "input_dy", y_dims, y_strides),
            _make_tensor(2, "weight_w", w_dims, w_strides),
        ]
        node_inputs = {"dy_tensor_uid": 1, "w_tensor_uid": 2}
        node_outputs = {"dx_tensor_uid": 0}
    else:  # wgrad (F==4): dy, x → dw
        tensors = [
            _make_tensor(0, "output_dw", w_dims, w_strides),
            _make_tensor(1, "input_dy", y_dims, y_strides),
            _make_tensor(2, "input_x", x_dims, x_strides),
        ]
        node_inputs = {"dy_tensor_uid": 1, "x_tensor_uid": 2}
        node_outputs = {"dw_tensor_uid": 0}

    nodes = [
        {
            "name": "conv_node",
            "type": node_type,
            "compute_data_type": "unset",
            "inputs": node_inputs,
            "outputs": node_outputs,
            "parameters": {
                "conv_mode": "CROSS_CORRELATION",
                "pre_padding": pre_pad,
                "post_padding": post_pad,
                "stride": stride_list,
                "dilation": dil_list,
            },
        }
    ]

    return {
        "compute_data_type": "float",
        "io_data_type": "bfloat16",
        "intermediate_data_type": "float",
        "tensors": tensors,
        "nodes": nodes,
    }


def _join_prefix(prefix: str, rest: str) -> str:
    """Join prefix and rest with '_', omitting the separator when prefix is empty."""
    return f"{prefix}_{rest}" if prefix else rest


def _conv_filename(prefix: str, p: _ConvParams) -> str:
    direction = _conv_direction_label(p.F)

    if p.spatial_dim == 3 and p.D is not None and p.D_f is not None:
        name = _join_prefix(
            prefix,
            f"conv_{direction}"
            f"_n{p.N}c{p.C}D{p.D}H{p.H}W{p.W}"
            f"_k{p.K}Df{p.D_f}R{p.R}S{p.S}"
            f"_pd{p.pad_d}p{p.pad_h}q{p.pad_w}"
            f"_sd{p.stride_d}u{p.stride_h}v{p.stride_w}"
            f"_g{p.groups}"
        )
    else:
        name = _join_prefix(
            prefix,
            f"conv_{direction}"
            f"_n{p.N}c{p.C}H{p.H}W{p.W}"
            f"_k{p.K}R{p.R}S{p.S}"
            f"_p{p.pad_h}q{p.pad_w}"
            f"_u{p.stride_h}v{p.stride_w}"
            f"_l{p.dil_h}j{p.dil_w}"
            f"_g{p.groups}",
        )
    return name


# ---------------------------------------------------------------------------
# Batchnorm conversion
# ---------------------------------------------------------------------------


def _build_bnorm_json(args: Dict[str, str]) -> Dict[str, Any]:
    """Build a hipDNN JSON graph dict from parsed bnormbfp16 args."""
    N = _int(args, "-n", 1)
    C = _int(args, "-c", 1)
    H = _int(args, "-H", 1)
    W = _int(args, "-W", 1)
    layout = args.get("--layout", "NCHW")
    forw = _int(args, "--forw", 1)

    is_3d = "-D" in args
    D: Optional[int] = _int(args, "-D", 1) if is_3d else None

    if is_3d and D is not None:
        x_dims = [N, C, D, H, W]
        x_strides = _input_strides(layout, N, C, H, W, D)
        scale_dims = [1, C, 1, 1, 1]
        scale_strides = [C, 1, 1, 1, 1]
    else:
        x_dims = [N, C, H, W]
        x_strides = _input_strides(layout, N, C, H, W)
        scale_dims = [1, C, 1, 1]
        scale_strides = [C, 1, 1, 1]

    if forw == 1:
        # Inference: x, mean, inv_variance, scale, bias → y
        node_type = "BatchnormInferenceAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides),
            _make_tensor(2, "mean", scale_dims, scale_strides, data_type="float"),
            _make_tensor(
                3, "inv_variance", scale_dims, scale_strides, data_type="float"
            ),
            _make_tensor(4, "scale", scale_dims, scale_strides, data_type="float"),
            _make_tensor(5, "bias", scale_dims, scale_strides, data_type="float"),
            _make_tensor(6, "output_y", x_dims, x_strides),
        ]
        nodes = [
            {
                "name": "batchnorm_inference_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "x_tensor_uid": 1,
                    "mean_tensor_uid": 2,
                    "inv_variance_tensor_uid": 3,
                    "scale_tensor_uid": 4,
                    "bias_tensor_uid": 5,
                },
                "outputs": {"y_tensor_uid": 6},
            }
        ]
    elif forw == 0:
        # Training forward: x, scale, bias, epsilon → y
        node_type = "BatchnormAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides),
            _make_tensor(2, "scale", scale_dims, scale_strides, data_type="float"),
            _make_tensor(3, "bias", scale_dims, scale_strides, data_type="float"),
            _make_tensor(4, "epsilon", [1], [1], data_type="float"),
            _make_tensor(5, "output_y", x_dims, x_strides),
        ]
        nodes = [
            {
                "name": "batchnorm_fwd_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "x_tensor_uid": 1,
                    "scale_tensor_uid": 2,
                    "bias_tensor_uid": 3,
                    "epsilon_tensor_uid": 4,
                },
                "outputs": {"y_tensor_uid": 5},
            }
        ]
    else:
        # Backward (forw == 2): dy, x, scale → dx, dscale, dbias
        node_type = "BatchnormBackwardAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides),
            _make_tensor(2, "input_dy", x_dims, x_strides),
            _make_tensor(3, "mean", scale_dims, scale_strides, data_type="float"),
            _make_tensor(
                4, "inv_variance", scale_dims, scale_strides, data_type="float"
            ),
            _make_tensor(5, "scale", scale_dims, scale_strides, data_type="float"),
            _make_tensor(6, "output_dx", x_dims, x_strides),
            _make_tensor(
                7, "output_dscale", scale_dims, scale_strides, data_type="float"
            ),
            _make_tensor(
                8, "output_dbias", scale_dims, scale_strides, data_type="float"
            ),
        ]
        nodes = [
            {
                "name": "batchnorm_backward_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "x_tensor_uid": 1,
                    "dy_tensor_uid": 2,
                    "mean_tensor_uid": 3,
                    "inv_variance_tensor_uid": 4,
                    "scale_tensor_uid": 5,
                },
                "outputs": {
                    "dx_tensor_uid": 6,
                    "dscale_tensor_uid": 7,
                    "dbias_tensor_uid": 8,
                },
            }
        ]

    return {
        "compute_data_type": "float",
        "io_data_type": "bfloat16",
        "intermediate_data_type": "float",
        "tensors": tensors,
        "nodes": nodes,
    }


def _bnorm_filename(prefix: str, args: Dict[str, str]) -> str:
    N = _int(args, "-n", 1)
    C = _int(args, "-c", 1)
    H = _int(args, "-H", 1)
    W = _int(args, "-W", 1)
    forw = _int(args, "--forw", 1)
    direction = {1: "inference", 0: "fwd"}.get(forw, "backward")

    is_3d = "-D" in args
    if is_3d:
        D = _int(args, "-D", 1)
        return _join_prefix(prefix, f"bnorm_{direction}_n{N}c{C}D{D}H{H}W{W}")
    return _join_prefix(prefix, f"bnorm_{direction}_n{N}c{C}H{H}W{W}")


# ---------------------------------------------------------------------------
# Line parser dispatcher
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse one shape file line. Returns (operation, args_dict) or None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Strip leading repeat count (e.g. "     5  ./bin/MIOpenDriver ...")
    m = re.match(r"^\s*\d+\s+", line)
    if m:
        line = line[m.end() :]

    parts = line.split()
    # parts[0] is the executable path, parts[1] is the operation
    if len(parts) < 2:
        return None

    operation = parts[1]  # e.g. "convbfp16" or "bnormbfp16"
    flag_tokens = parts[2:]
    args = _parse_args(flag_tokens)
    return operation, args


def _convert_line(
    operation: str, args: Dict[str, str], prefix: str
) -> Tuple[str, Dict[str, Any]]:
    """Convert parsed MIOpen args to (filename_stem, json_dict)."""
    if operation in ("convbfp16", "conv"):
        p = _ConvParams.from_args(args)
        graph = _build_conv_json(p)
        name_stem = _conv_filename(prefix, p)
    elif operation in ("bnormbfp16", "bnorm"):
        graph = _build_bnorm_json(args)
        name_stem = _bnorm_filename(prefix, args)
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    graph["name"] = name_stem
    return name_stem, graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MIOpen driver shape files or inline args to hipDNN JSON graph files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="SHAPES_FILE",
        help="One or more MIOpen shape .txt files to convert.",
    )
    parser.add_argument(
        "-A",
        "--args",
        metavar="MIOPEN_ARGS",
        default=None,
        help=(
            "Inline MIOpen driver arguments (everything after the executable path), "
            "e.g. 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 ...'. "
            "Use with --output to write to a specific file."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON file path (used with --args; ignored for file inputs).",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        default=None,
        help=(
            "Output directory for JSON files. "
            "Defaults to the same directory as each input file."
        ),
    )
    return parser


def _process_inline_args(args_str: str, output: Optional[str]) -> int:
    """Convert a single inline MIOpen driver argument string to a JSON file."""
    parts = args_str.split()
    if not parts:
        print("ERROR: --args is empty.", file=sys.stderr)
        return 1

    operation = parts[0]
    flag_tokens = parts[1:]
    parsed_args = _parse_args(flag_tokens)

    try:
        name_stem, graph = _convert_line(operation, parsed_args, "")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(f"{name_stem}.json")

    out_path.write_text(json.dumps(graph, indent=4) + "\n")
    print(f"Written: {out_path}")
    return 0


def _process_file(input_path: Path, outdir: Path) -> Tuple[int, int, int]:
    """Process one shape file. Returns (written, skipped, warnings)."""
    prefix = input_path.stem.lower().replace("-", "_")
    lines = input_path.read_text().splitlines()

    written = 0
    skipped = 0
    warnings = 0
    seen_stems: Dict[str, int] = {}

    for lineno, raw_line in enumerate(lines, start=1):
        parsed = _parse_line(raw_line)
        if parsed is None:
            skipped += 1
            continue

        operation, args = parsed
        try:
            name_stem, graph = _convert_line(operation, args, prefix)
        except Exception as exc:
            print(f"  WARNING line {lineno}: {exc}", file=sys.stderr)
            warnings += 1
            continue

        # Deduplicate: if same stem appears more than once, append a counter
        count = seen_stems.get(name_stem, 0)
        seen_stems[name_stem] = count + 1
        if count > 0:
            unique_stem = f"{name_stem}_{count}"
        else:
            unique_stem = name_stem

        out_path = outdir / f"{unique_stem}.json"
        out_path.write_text(json.dumps(graph, indent=4) + "\n")
        written += 1

    return written, skipped, warnings


def main() -> int:
    parser = _build_arg_parser()
    ns = parser.parse_args()

    if ns.args:
        if ns.inputs:
            print(
                "ERROR: cannot combine --args with positional SHAPES_FILE arguments.",
                file=sys.stderr,
            )
            return 1
        return _process_inline_args(ns.args, ns.output)

    if not ns.inputs:
        parser.print_help(sys.stderr)
        return 1

    total_written = 0
    total_skipped = 0
    total_warnings = 0

    for input_str in ns.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            return 1

        outdir = Path(ns.outdir) if ns.outdir else input_path.parent
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {input_path} → {outdir}/")
        written, skipped, warns = _process_file(input_path, outdir)
        print(f"  Written: {written}  Skipped: {skipped}  Warnings: {warns}")
        total_written += written
        total_skipped += skipped
        total_warnings += warns

    print(f"\nTotal: {total_written} files written, {total_warnings} warnings.")
    return 0 if total_warnings == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
