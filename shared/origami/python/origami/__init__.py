# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Origami Python bindings.

This package provides Python bindings for the Origami library.
Core functionality is always available. PyTorch-specific operations
are available if built with PyTorch support and accessed via torch.ops.origami
"""

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

# Export flags for capability checking
__all__ = ['HAS_CORE', 'HAS_ATEN_OPS']
