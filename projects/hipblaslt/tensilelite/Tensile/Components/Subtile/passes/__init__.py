# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Standalone scheduling passes for the LogicalScheduler pipeline."""

from .place_lrs import place_lrs
from .assign_vgpr_tiles import assign_vgpr_tiles, VgprAllocation
from .place_grs import place_grs
from ._utils import partition_tile_range
