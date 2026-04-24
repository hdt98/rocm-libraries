# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for convert_miopen_shapes.py."""

import pytest

from dnn_benchmarking.tools.convert_miopen_shapes import (
    _ConvParams,
    _build_bnorm_json,
    _build_conv_json,
    _conv_out_dim,
    _convert_line,
    _is_flag,
    _nchw_strides,
    _ncdhw_strides,
    _nhwc_strides,
    _ndhwc_strides,
    _parse_args,
    _parse_line,
)


class TestStrideHelpers:
    """Tests for stride computation helpers."""

    def test_nchw_strides_known_values(self) -> None:
        """NCHW strides: innermost dim is 1, outermost is product of inner dims."""
        # N=2, C=3, H=4, W=5 → [C*H*W, H*W, W, 1] = [60, 20, 5, 1]
        assert _nchw_strides(2, 3, 4, 5) == [60, 20, 5, 1]

    def test_nhwc_strides_known_values(self) -> None:
        """NHWC strides: C is innermost (stride 1), then W*C, H*W*C, etc."""
        # N=2, C=3, H=4, W=5 → [H*W*C, 1, W*C, C] = [60, 1, 15, 3]
        assert _nhwc_strides(2, 3, 4, 5) == [60, 1, 15, 3]

    def test_ncdhw_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [C*D*H*W, D*H*W, H*W, W, 1] = [120, 60, 20, 5, 1]
        assert _ncdhw_strides(1, 2, 3, 4, 5) == [120, 60, 20, 5, 1]

    def test_ndhwc_strides_known_values(self) -> None:
        # N=1, C=2, D=3, H=4, W=5 → [D*H*W*C, 1, H*W*C, W*C, C] = [120, 1, 40, 10, 2]
        assert _ndhwc_strides(1, 2, 3, 4, 5) == [120, 1, 40, 10, 2]


class TestConvOutDim:
    """Tests for convolution output dimension formula."""

    def test_no_pad_stride1_dil1(self) -> None:
        # floor((16 + 0 - 1*(3-1) - 1) / 1 + 1) = floor(14/1 + 1) = 15
        assert _conv_out_dim(16, 0, 1, 3, 1) == 14

    def test_with_pad(self) -> None:
        # floor((16 + 2 - 1*(3-1) - 1) / 1 + 1) = floor(15/1 + 1) = 16
        assert _conv_out_dim(16, 1, 1, 3, 1) == 16

    def test_with_stride(self) -> None:
        # floor((16 + 0 - 1*(1-1) - 1) / 2 + 1) = floor(15/2 + 1) = 8
        assert _conv_out_dim(16, 0, 1, 1, 2) == 8


class TestIsFlag:
    """Tests for _is_flag helper."""

    def test_normal_flag(self) -> None:
        assert _is_flag("-n") is True

    def test_long_flag(self) -> None:
        assert _is_flag("--layout") is True

    def test_negative_integer_is_not_flag(self) -> None:
        assert _is_flag("-1") is False

    def test_negative_float_is_not_flag(self) -> None:
        assert _is_flag("-0.5") is False

    def test_non_flag_token(self) -> None:
        assert _is_flag("value") is False


class TestParseArgs:
    """Tests for _parse_args."""

    def test_basic_flag_value_pairs(self) -> None:
        result = _parse_args(["-n", "16", "-c", "64"])
        assert result == {"-n": "16", "-c": "64"}

    def test_boolean_flag(self) -> None:
        result = _parse_args(["--verbose"])
        assert result == {"--verbose": "1"}

    def test_negative_value(self) -> None:
        # -p -1: -1 should be treated as a value, not a new flag
        result = _parse_args(["-p", "-1"])
        assert result == {"-p": "-1"}

    def test_mixed(self) -> None:
        result = _parse_args(["-n", "4", "--verbose", "-c", "32"])
        assert result["-n"] == "4"
        assert result["--verbose"] == "1"
        assert result["-c"] == "32"


class TestBuildConvJson:
    """Tests for _build_conv_json via _ConvParams."""

    def _make_params(self, **overrides) -> _ConvParams:
        defaults = dict(
            N=2, C=32, H=8, W=8, K=64, R=3, S=3,
            pad_h=1, pad_w=1, stride_h=1, stride_w=1,
            dil_h=1, dil_w=1, groups=1, F=1, spatial_dim=2,
            in_layout="NCHW", out_layout="NCHW",
        )
        defaults.update(overrides)
        return _ConvParams(**defaults)

    def test_conv_fwd_node_type(self) -> None:
        p = self._make_params(F=1)
        graph = _build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_conv_fwd_tensor_wiring(self) -> None:
        p = self._make_params(F=1)
        graph = _build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"x_tensor_uid": 1, "w_tensor_uid": 2}
        assert node["outputs"] == {"y_tensor_uid": 0}

    def test_conv_dgrad_node_type(self) -> None:
        p = self._make_params(F=2)
        graph = _build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionBwdAttributes"

    def test_conv_dgrad_tensor_wiring(self) -> None:
        p = self._make_params(F=2)
        graph = _build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"dy_tensor_uid": 1, "w_tensor_uid": 2}
        assert node["outputs"] == {"dx_tensor_uid": 0}

    def test_conv_wgrad_node_type(self) -> None:
        p = self._make_params(F=4)
        graph = _build_conv_json(p)
        assert graph["nodes"][0]["type"] == "ConvolutionWrwAttributes"

    def test_conv_wgrad_tensor_wiring(self) -> None:
        p = self._make_params(F=4)
        graph = _build_conv_json(p)
        node = graph["nodes"][0]
        assert node["inputs"] == {"dy_tensor_uid": 1, "x_tensor_uid": 2}
        assert node["outputs"] == {"dw_tensor_uid": 0}

    def test_nhwc_layout_produces_different_strides(self) -> None:
        p_nchw = self._make_params(in_layout="NCHW")
        p_nhwc = self._make_params(in_layout="NHWC")
        g_nchw = _build_conv_json(p_nchw)
        g_nhwc = _build_conv_json(p_nhwc)
        # Input tensor (uid=1) strides should differ
        strides_nchw = next(t["strides"] for t in g_nchw["tensors"] if t["uid"] == 1)
        strides_nhwc = next(t["strides"] for t in g_nhwc["tensors"] if t["uid"] == 1)
        assert strides_nchw != strides_nhwc

    def test_3d_conv_produces_5d_dims(self) -> None:
        p = _ConvParams(
            N=1, C=4, H=4, W=4, K=8, R=3, S=3,
            pad_h=1, pad_w=1, stride_h=1, stride_w=1,
            dil_h=1, dil_w=1, groups=1, F=1, spatial_dim=3,
            in_layout="NCDHW", out_layout="NCDHW",
            D=4, D_f=3, pad_d=1, stride_d=1, dil_d=1,
        )
        graph = _build_conv_json(p)
        x_tensor = next(t for t in graph["tensors"] if t["uid"] == 1)
        assert len(x_tensor["dims"]) == 5
        assert len(x_tensor["strides"]) == 5


class TestBuildBnormJson:
    """Tests for _build_bnorm_json."""

    def _base_args(self, forw: int) -> dict:
        return {"-n": "2", "-c": "64", "-H": "14", "-W": "14", "--forw": str(forw)}

    def test_inference_node_type(self) -> None:
        graph = _build_bnorm_json(self._base_args(1))
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"

    def test_inference_tensor_count(self) -> None:
        graph = _build_bnorm_json(self._base_args(1))
        assert len(graph["tensors"]) == 6

    def test_fwd_training_node_type(self) -> None:
        graph = _build_bnorm_json(self._base_args(0))
        assert graph["nodes"][0]["type"] == "BatchnormAttributes"

    def test_fwd_training_has_epsilon_tensor(self) -> None:
        graph = _build_bnorm_json(self._base_args(0))
        node = graph["nodes"][0]
        assert "epsilon_tensor_uid" in node["inputs"]
        epsilon_uid = node["inputs"]["epsilon_tensor_uid"]
        epsilon_tensor = next(t for t in graph["tensors"] if t["uid"] == epsilon_uid)
        assert epsilon_tensor["dims"] == [1]

    def test_backward_node_type(self) -> None:
        graph = _build_bnorm_json(self._base_args(2))
        assert graph["nodes"][0]["type"] == "BatchnormBackwardAttributes"

    def test_backward_tensor_count(self) -> None:
        graph = _build_bnorm_json(self._base_args(2))
        assert len(graph["tensors"]) == 8

    def test_backward_has_mean_and_inv_variance(self) -> None:
        graph = _build_bnorm_json(self._base_args(2))
        node = graph["nodes"][0]
        assert "mean_tensor_uid" in node["inputs"]
        assert "inv_variance_tensor_uid" in node["inputs"]


class TestParseLine:
    """Tests for _parse_line."""

    def test_blank_returns_none(self) -> None:
        assert _parse_line("") is None
        assert _parse_line("   ") is None

    def test_comment_returns_none(self) -> None:
        assert _parse_line("# this is a comment") is None

    def test_valid_conv_line(self) -> None:
        line = "./bin/MIOpenDriver convbfp16 -n 2 -c 64 -H 8 -W 8 -k 128 -y 3 -x 3 -F 1"
        result = _parse_line(line)
        assert result is not None
        operation, args = result
        assert operation == "convbfp16"
        assert args["-n"] == "2"
        assert args["-F"] == "1"

    def test_repeat_count_prefix_stripped(self) -> None:
        line = "     5  ./bin/MIOpenDriver convbfp16 -n 1 -c 32 -H 4 -W 4 -k 32 -y 1 -x 1"
        result = _parse_line(line)
        assert result is not None
        operation, _ = result
        assert operation == "convbfp16"

    def test_too_short_returns_none(self) -> None:
        assert _parse_line("./bin/MIOpenDriver") is None


class TestConvertLine:
    """Tests for _convert_line dispatcher."""

    def test_unsupported_operation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported operation"):
            _convert_line("matmul", {}, "prefix")

    def test_conv_operation_succeeds(self) -> None:
        args = {"-n": "1", "-c": "8", "-H": "4", "-W": "4",
                "-k": "16", "-y": "1", "-x": "1", "-F": "1"}
        name_stem, graph = _convert_line("convbfp16", args, "test")
        assert "conv" in name_stem
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_bnorm_operation_succeeds(self) -> None:
        args = {"-n": "1", "-c": "32", "-H": "8", "-W": "8", "--forw": "1"}
        name_stem, graph = _convert_line("bnormbfp16", args, "test")
        assert "bnorm" in name_stem
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"
