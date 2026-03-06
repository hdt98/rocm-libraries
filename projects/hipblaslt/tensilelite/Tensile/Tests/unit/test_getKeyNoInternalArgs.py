# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for getKeyNoInternalArgs in Naming.py

Verifies that the string-based key computation produces consistent
deduplication: solutions that differ only in internal args (WorkGroupMapping,
StaggerU, etc.) get the same key, while solutions differing in structural
params get different keys.
"""

import pytest

pytestmark = pytest.mark.unit

from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs


def _make_state(**overrides):
    """Create a minimal kernel-like state dict for testing."""
    base = {
        "ProblemType": {
            "GroupedGemm": False,
            "DataType": 0,
            "TransposeA": False,
            "TransposeB": True,
            "OperationType": "GEMM",
            "UseBeta": True,
            "HighPrecisionAccumulate": False,
            "UseInitialStridesAB": False,
            "UseInitialStridesCD": False,
        },
        "GlobalSplitU": 1,
        "WorkGroupMapping": 8,
        "WorkGroupMappingXCC": 1,
        "WorkGroupMappingXCCGroup": 1,
        "StaggerU": 32,
        "StaggerUStride": 256,
        "StaggerUMapping": 0,
        "GlobalSplitUCoalesced": True,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "SFCWGM": 0,
        "CustomKernelName": "",
        "MacroTile0": 128,
        "MacroTile1": 128,
        "DepthU": 32,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MIWaveTile": [2, 2],
        "UseCustomMainLoopSchedule": False,
        "SpaceFillingAlgo": [],
    }
    base.update(overrides)
    return base


class TestGetKeyNoInternalArgs:

    def test_returns_string(self):
        state = _make_state()
        key = getKeyNoInternalArgs(state, splitGSU=False)
        assert isinstance(key, str)

    def test_same_key_for_different_internal_args(self):
        """Solutions differing only in internal args should get same key."""
        s1 = _make_state(WorkGroupMapping=8, StaggerU=32)
        s2 = _make_state(WorkGroupMapping=16, StaggerU=64)
        assert getKeyNoInternalArgs(s1, False) == getKeyNoInternalArgs(s2, False)

    def test_different_key_for_different_structural_params(self):
        """Solutions differing in structural params should get different keys."""
        s1 = _make_state(MacroTile0=128)
        s2 = _make_state(MacroTile0=256)
        assert getKeyNoInternalArgs(s1, False) != getKeyNoInternalArgs(s2, False)

    def test_does_not_mutate_state(self):
        """Calling getKeyNoInternalArgs should not modify the input state."""
        state = _make_state(GlobalSplitU=4, WorkGroupMapping=8)
        original_gsu = state["GlobalSplitU"]
        original_wgm = state["WorkGroupMapping"]
        original_gg = state["ProblemType"]["GroupedGemm"]

        getKeyNoInternalArgs(state, splitGSU=True)

        assert state["GlobalSplitU"] == original_gsu
        assert state["WorkGroupMapping"] == original_wgm
        assert state["ProblemType"]["GroupedGemm"] == original_gg

    def test_grouped_gemm_ignored(self):
        """GroupedGemm should not affect the key."""
        s1 = _make_state()
        s1["ProblemType"]["GroupedGemm"] = True
        s2 = _make_state()
        s2["ProblemType"]["GroupedGemm"] = False
        assert getKeyNoInternalArgs(s1, False) == getKeyNoInternalArgs(s2, False)

    def test_gsu_masked_when_splitgsu_true(self):
        """With splitGSU=True, GSU > 1 values should produce same key."""
        s1 = _make_state(GlobalSplitU=2)
        s2 = _make_state(GlobalSplitU=4)
        assert getKeyNoInternalArgs(s1, True) == getKeyNoInternalArgs(s2, True)

    def test_gsu_zero_distinct_from_nonzero(self):
        """GSU=0 should produce a different key than GSU > 0."""
        s1 = _make_state(GlobalSplitU=0)
        s2 = _make_state(GlobalSplitU=1)
        assert getKeyNoInternalArgs(s1, False) != getKeyNoInternalArgs(s2, False)

    def test_usable_in_set(self):
        """Keys should be usable in sets for deduplication."""
        s1 = _make_state(WorkGroupMapping=8)
        s2 = _make_state(WorkGroupMapping=16)
        s3 = _make_state(MacroTile0=256)

        keys = {getKeyNoInternalArgs(s, False) for s in [s1, s2, s3]}
        # s1 and s2 should deduplicate, s3 is different
        assert len(keys) == 2
