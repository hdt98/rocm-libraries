# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph validation for supported operations."""

from typing import Any, Dict, List, Set, Tuple

from ..common.exceptions import GraphLoadError

_REQUIRED_TOP_LEVEL_KEYS: List[str] = ["tensors", "nodes"]

# Supported operation types - accepts any hipDNN operation
SUPPORTED_NODE_TYPES: Set[str] = {
    "ConvolutionFwdAttributes",
    "ConvolutionBwdAttributes",
    "ConvolutionWrwAttributes",
    "MatmulAttributes",
    "PointwiseAttributes",
    "BatchnormAttributes",
    "BatchnormBackwardAttributes",
    "BatchnormInferenceAttributes",
}

# Required keys per node type, as (section, key) tuples.
# section is "inputs", "outputs", "parameters", or None (top-level node).
# Derived from the C++ from_json implementations in data_sdk/utilities/json/.
_REQUIRED_NODE_FIELDS: Dict[str, List[Tuple[str, str]]] = {
    "ConvolutionFwdAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "w_tensor_uid"),
        ("outputs", "y_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "ConvolutionBwdAttributes": [
        ("inputs", "dy_tensor_uid"),
        ("inputs", "w_tensor_uid"),
        ("outputs", "dx_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "ConvolutionWrwAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "dy_tensor_uid"),
        ("outputs", "dw_tensor_uid"),
        ("parameters", "pre_padding"),
        ("parameters", "post_padding"),
        ("parameters", "stride"),
        ("parameters", "dilation"),
        ("parameters", "conv_mode"),
    ],
    "MatmulAttributes": [
        ("inputs", "a_tensor_uid"),
        ("inputs", "b_tensor_uid"),
        ("outputs", "c_tensor_uid"),
    ],
    "BatchnormInferenceAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("inputs", "bias_tensor_uid"),
        ("outputs", "y_tensor_uid"),
    ],
    "BatchnormAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("inputs", "bias_tensor_uid"),
        ("inputs", "epsilon_tensor_uid"),
        ("inputs", "peer_stats_tensor_uid"),
        ("outputs", "y_tensor_uid"),
    ],
    "BatchnormBackwardAttributes": [
        ("inputs", "x_tensor_uid"),
        ("inputs", "dy_tensor_uid"),
        ("inputs", "mean_tensor_uid"),
        ("inputs", "inv_variance_tensor_uid"),
        ("inputs", "scale_tensor_uid"),
        ("inputs", "peer_stats_tensor_uid"),
        ("outputs", "dx_tensor_uid"),
        ("outputs", "dscale_tensor_uid"),
        ("outputs", "dbias_tensor_uid"),
    ],
}


class GraphValidator:
    """Validates that a graph contains valid operations.

    Accepts any hipDNN operation type that can be deserialized and executed.
    """

    def __init__(self, supported_types: Set[str] = SUPPORTED_NODE_TYPES) -> None:
        """Initialize validator with supported operation types.

        Args:
            supported_types: Set of node type names that are supported.
                           If None, accepts any operation type.
        """
        self._supported_types = supported_types

    def validate(self, graph_json: Dict[str, Any]) -> None:
        """Validate top-level graph structure and node required fields.

        Args:
            graph_json: The parsed graph JSON dictionary.

        Raises:
            GraphLoadError: If required top-level keys are missing, the graph
                            contains no operation nodes, or any node is missing
                            required fields.
        """
        missing_keys = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in graph_json]
        if missing_keys:
            raise GraphLoadError(
                f"Graph is missing required top-level keys: {', '.join(missing_keys)}"
            )

        nodes = graph_json["nodes"]
        if not nodes:
            raise GraphLoadError("Graph contains no operation nodes")

        for node in nodes:
            self._validate_node_fields(node)

    def _validate_node_fields(self, node: Dict[str, Any]) -> None:
        """Check that a node has all required fields for its type.

        Args:
            node: A single node dictionary from graph JSON.

        Raises:
            GraphLoadError: If a required field is missing.
        """
        node_type = node.get("type", "")
        required = _REQUIRED_NODE_FIELDS.get(node_type)
        if required is None:
            # Unknown type — let hipDNN backend report the error at build time.
            return

        node_name = node.get("name", node_type)
        missing = []
        for section, key in required:
            container = node.get(section, {})
            if key not in container:
                missing.append(f"{section}.{key}")

        if missing:
            raise GraphLoadError(
                f"Node '{node_name}' ({node_type}) is missing required fields: "
                + ", ".join(missing)
            )

    def get_supported_types(self) -> Set[str]:
        """Get the set of supported operation types.

        Returns:
            Set of supported node type names.
        """
        return self._supported_types.copy()
