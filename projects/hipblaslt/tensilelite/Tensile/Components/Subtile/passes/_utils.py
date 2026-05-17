# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared utilities for scheduling passes."""

from __future__ import annotations
from typing import Dict, Tuple

from .._types import SchedulerConfig


def partition_tile_range(config: SchedulerConfig, pi: int) -> Dict[str, Tuple[int, int]]:
    """Return {'A': (start, end), 'B': (start, end)} for partition pi.

    Uses COLUMN_MAJOR ordering: M (A) varies fastest, N (B) varies slowest.
    """
    piM = pi % config.numPartitionsM
    piN = pi // config.numPartitionsM
    a0 = piM * config.partitionSizeM
    b0 = piN * config.partitionSizeN
    return {'A': (a0, a0 + config.partitionSizeM),
            'B': (b0, b0 + config.partitionSizeN)}
