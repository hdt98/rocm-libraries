# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Origami Python bindings.

This package provides Python bindings for the Origami library.
Core functionality is always available. PyTorch-specific operations
are available if built with PyTorch support and accessed via torch.ops.origami
"""

from collections import namedtuple
from enum import IntEnum

# Import and re-export core functionality
try:
    from ._core import *
    HAS_CORE = True
except ImportError as e:
    HAS_CORE = False
    import warnings
    warnings.warn(f"Failed to import origami core bindings: {e}")
    raise

# Try to load ATen ops (registers torch.ops.origami.*)
HAS_ATEN_OPS = False
try:
    from . import _aten  # This imports the module and registers torch ops
    HAS_ATEN_OPS = True
except ImportError:
    # PyTorch not available or ATen module not built
    pass

# Define the ConfigResult named tuple for at_select_config return values
# This provides semantic field names instead of positional tuple indexing
ConfigResult = namedtuple('ConfigResult', [
    'latency',                      # Predicted latency (float)
    'blk_m',                        # Block size M dimension
    'blk_n',                        # Block size N dimension  
    'blk_k',                        # Block size K dimension
    'mi_m',                         # Matrix instruction M dimension
    'mi_n',                         # Matrix instruction N dimension
    'mi_k',                         # Matrix instruction K dimension
    'occupancy',                    # Kernel occupancy
    'workgroup_mapping',            # Workgroup mapping strategy
    'cache_hints_a',                # Cache hints for matrix A
    'cache_hints_b',                # Cache hints for matrix B
    'workspace_size',               # Workspace size
    'workspace_size_per_elem_c',    # Workspace size per element of C
    'reduction_strategy'            # Reduction strategy (can be cast to enum)
])

def at_select_config(problem_dict, config_dicts=None):
    """
    Select optimal configuration for the given GEMM problem.
    
    This function wraps torch.ops.origami.select_config to provide a more
    user-friendly interface with named return values instead of positional
    tuples.  It remains fully compatible with torch.compile and handles the
    awful 14-value tuple return.
    
    Args:
        problem_dict: Dictionary with problem parameters. Required keys:
            - SIZE_M, SIZE_N, SIZE_K: Problem dimensions
            - BATCH: Batch size
            - A_TRANSPOSE, B_TRANSPOSE: Transpose flags (0 or 1)
            - A_DTYPE, B_DTYPE, C_DTYPE, D_DTYPE, MI_DTYPE: Data type enums
            Optional keys:
            - A_MX_BLOCK_SIZE, B_MX_BLOCK_SIZE: MX block sizes
            
        config_dicts: Optional list of configuration dictionaries to choose
        from.  Each config dict should contain:
            - BLK_M, BLK_N, BLK_K: Block tile dimensions (required)
            - OCCUPANCY: Kernel occupancy (required)
            - MI_DIM_M, MI_DIM_N, MI_DIM_K: Matrix instruction dims (optional)
            - WG_MAPPING: Workgroup mapping strategy (optional)
            - WORKSPACE_SIZE, WORKSPACE_SIZE_PER_C: Workspace parameters
              (optional)
            - REDUCTION_STRATEGY: Reduction strategy (optional)
            - CACHE_HINTS_A, CACHE_HINTS_B: Cache hints (optional, not yet
              used)
        
    Returns:
        ConfigResult: Named tuple with full configuration parameters including:
            - latency: Predicted latency (float)
            - blk_m, blk_n, blk_k: Block tile dimensions
            - mi_m, mi_n, mi_k: Matrix instruction dimensions
            - occupancy: Kernel occupancy
            - workgroup_mapping: Workgroup mapping strategy
            - cache_hints_a, cache_hints_b: Cache hints
            - workspace_size, workspace_size_per_elem_c: Workspace parameters
            - reduction_strategy: Reduction strategy (int)
            
    Example:
        >>> import origami
        >>> # Using the convenient enums instead of raw integers
        >>> problem = {
        ...     "SIZE_M": 1024, "SIZE_N": 2048, "SIZE_K": 512,
        ...     "BATCH": 1,
        ...     "A_TRANSPOSE": origami.Transpose.T,
        ...     "B_TRANSPOSE": origami.Transpose.N,
        ...     "A_DTYPE": origami.DataType.BFloat16,
        ...     "B_DTYPE": origami.DataType.BFloat16,
        ...     "C_DTYPE": origami.DataType.BFloat16,
        ...     "D_DTYPE": origami.DataType.BFloat16,
        ...     "MI_DTYPE": origami.DataType.BFloat16
        ... }
        >>> configs = [
        ...     {"BLK_M": 64, "BLK_N": 64, "BLK_K": 32, "OCCUPANCY": 1, "WG_MAPPING": 6},
        ...     {"BLK_M": 128, "BLK_N": 128, "BLK_K": 32, "OCCUPANCY": 1, "WG_MAPPING": 6},
        ... ]
        >>> config = origami.at_select_config(problem, configs)
        >>> print(f"Block size: {config.blk_m}x{config.blk_n}x{config.blk_k}")
        >>> print(f"Predicted latency: {config.latency} ms")
        >>> print(f"Matrix instruction: {config.mi_m}x{config.mi_n}x{config.mi_k}")
    
    Note:
        This function requires HAS_ATEN_OPS to be True (PyTorch support enabled).
        The wrapper is torch.compile compatible - the underlying torch.ops call
        will be properly traced and specialized.
    """
    if not HAS_ATEN_OPS:
        raise RuntimeError(
            "at_select_config requires PyTorch support. "
            "Make sure Origami was built with PyTorch and _aten module is available."
        )
    
    import torch
    
    # Call the underlying torch.ops function
    # This returns a tuple: (float, int, int, int, int, int, int, int, int,
    #                        int, int, int, int, int)
    result_tuple = torch.ops.origami.select_config(problem_dict, config_dicts)
    
    # Unpack the 14-element tuple into a named tuple for user-friendly access
    return ConfigResult(*result_tuple)


# Python enums matching C++ enums from include/origami/types.hpp
# These make it easier to create problem_dict and config_dicts

class DataType(IntEnum):
    """Data type enum matching origami::data_type_t"""
    Float = 0
    Double = 1
    ComplexFloat = 2
    ComplexDouble = 3
    Half = 4
    Int8x4 = 5
    Int32 = 6
    BFloat16 = 7
    Int8 = 8
    Int64 = 9
    XFloat32 = 10
    Float8_fnuz = 11
    BFloat8_fnuz = 12
    Float8BFloat8_fnuz = 13
    BFloat8Float8_fnuz = 14
    Float8 = 15
    BFloat8 = 16
    Float8BFloat8 = 17
    BFloat8Float8 = 18
    Float6 = 19
    BFloat6 = 20
    Float4 = 21


class Transpose(IntEnum):
    """Transpose enum matching origami::transpose_t"""
    T = 0  # Transposed
    N = 1  # Not transposed


class ReductionStrategy(IntEnum):
    """Reduction strategy enum matching origami::reduction_t"""
    Spinlock = 0
    Tree = 1
    Parallel = 2
    Atomic = 3


# Export public API
__all__ = [
    'HAS_CORE', 
    'HAS_ATEN_OPS', 
    'at_select_config', 
    'ConfigResult',
    'DataType',
    'Transpose', 
    'ReductionStrategy'
]
