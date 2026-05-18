################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Solution-level gating tests for the subtile tail-loop work (Phase 1.3).

These tests pin the Phase 2 expectations on `Solution.py`:

  * Solution.py:577 currently sets `minASEMforMX = 32 if not state["UseSubtileImpl"]
    else 256`. Phase 2 drops the conditional so subtile MX kernels accept the
    same ASEM=32 the rest of the MX path uses.
  * The downstream consequence is that `state["NoTailLoop"]` (set at
    Solution.py:3603-3607 from `ASEM % DepthU == 0`) is `False` for a typical
    subtile MX problem with K_rem != 0, instead of being `True` because ASEM was
    bumped up to 256 = DepthU.

Tests use a hand-crafted state and call
`Solution.assignProblemIndependentDerivedParameters` directly (the same entry
point the full `assignDerivedParameters` invokes). Downstream errors past the
bump line are tolerated — the assertion targets state values set well before
those points.
"""
import pytest
import yaml

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Types import IsaVersion
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.SolutionStructs.Solution import Solution


# Module-level setup: validate toolchain once and build the isaInfoMap.
_cxxCompiler = validateToolchain("amdclang++")
_isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, _cxxCompiler)


# ── State factories ──────────────────────────────────────────────────────────

def _build_subtile_mx_state(*, asem=32, depthU=256, mt0=128, mt1=128):
    """Build a state for a subtile MXFP4 (TN) kernel.

    `assignProblemIndependentDerivedParameters` writes to
    `state["AssertSummationElementMultiple"]` at Solution.py:577-581 based on
    `state["UseSubtileImpl"]`. Everything below is just enough state to reach
    that line without raising; later in the function execution may not always
    succeed (downstream paths read additional fields), but the bump value is
    fully observable post-call.
    """
    config = yaml.safe_load(
        """
        ProblemType:
          OperationType: GEMM
          DataType: F4
          DestDataType: b
          ComputeDataType: s
          HighPrecisionAccumulate: True
          MXBlockA: 32
          MXBlockB: 32
          TransposeA: True
          TransposeB: False
          UseBeta: True
          Batched: True
          StridedBatched: True
          ActivationFuncCall: True
        """
    )
    isa = IsaVersion(9, 5, 0)
    mi = [16, 16, 128, 1, 1, mt0 // 64, mt1 // 64, 2, 2]
    mi_params = matrixInstructionToMIParameters(
        mi, isa, 64, config["ProblemType"], [16, 16, 1], _isaInfoMap
    )

    state = dict(defaultSolution)
    state.update({
        "ISA": isa,
        "WavefrontSize": 64,
        "ScheduleIterAlg": 3,
        "UseSubtileImpl": True,
        "StreamK": 3,
        "DepthU": depthU,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "DirectToLds": 1,
        "StaggerU": 0,
        "LocalSplitU": 1,
        "AssertSummationElementMultiple": asem,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "NoReject": False,
        "MatrixInstruction": mi,
        "EnableMatrixInstruction": True,
        "UseF32XEmulation": False,
    })
    state.update(mi_params)
    state["ProblemType"] = ProblemType(config["ProblemType"], False)
    return state


def _build_subtile_bf16_state(*, asem=32, depthU=64, mt0=128, mt1=128):
    """Build a state for a subtile BF16 (TN) kernel — no MX scales."""
    config = yaml.safe_load(
        """
        ProblemType:
          OperationType: GEMM
          DataType: b
          DestDataType: b
          ComputeDataType: s
          HighPrecisionAccumulate: True
          TransposeA: True
          TransposeB: False
          UseBeta: True
          Batched: True
          StridedBatched: True
          ActivationFuncCall: True
        """
    )
    isa = IsaVersion(9, 5, 0)
    mi = [16, 16, 32, 1, 1, mt0 // 64, mt1 // 64, 2, 2]
    mi_params = matrixInstructionToMIParameters(
        mi, isa, 64, config["ProblemType"], [16, 16, 1], _isaInfoMap
    )

    state = dict(defaultSolution)
    state.update({
        "ISA": isa,
        "WavefrontSize": 64,
        "ScheduleIterAlg": 3,
        "UseSubtileImpl": True,
        "StreamK": 3,
        "DepthU": depthU,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "DirectToLds": 1,
        "StaggerU": 0,
        "LocalSplitU": 1,
        "AssertSummationElementMultiple": asem,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "NoReject": False,
        "MatrixInstruction": mi,
        "EnableMatrixInstruction": True,
        "UseF32XEmulation": False,
    })
    state.update(mi_params)
    state["ProblemType"] = ProblemType(config["ProblemType"], False)
    return state


def _run_assign_problem_independent(state):
    """Invoke `assignProblemIndependentDerivedParameters`; tolerate downstream
    KeyErrors that don't affect the bump assertion (the bump executes at line
    577-581, well before the function reaches any state field that varies in
    test setups).
    """
    try:
        Solution.assignProblemIndependentDerivedParameters(state, False, _isaInfoMap)
    except KeyError:
        # The function may try to read additional state keys that aren't set up
        # in this test harness. The bump logic at lines 577-581 already executed
        # and the resulting state["AssertSummationElementMultiple"] is what we
        # assert against.
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

class TestSubtileMxAsem:
    """Phase 2: minASEMforMX must be 32 for subtile (was 256)."""

    def test_subtile_mx_accepts_asem_32(self):
        """A subtile MX kernel with ASEM=32 must not be bumped up to 256.

        Today this fails because Solution.py:577 forces minASEMforMX=256 when
        UseSubtileImpl=1, which causes the MX-bump branch (Solution.py:578-581)
        to round AssertSummationElementMultiple up to 256.

        After Phase 2 drops the conditional, ASEM=32 satisfies the MX-bump
        condition (32 % 32 == 0) and is left unchanged.
        """
        state = _build_subtile_mx_state(asem=32, depthU=256)
        _run_assign_problem_independent(state)

        actual = state["AssertSummationElementMultiple"]
        assert actual == 32, (
            f"Subtile MX kernel: AssertSummationElementMultiple was bumped "
            f"to {actual} (expected to remain 32). "
            f"Solution.py:577 minASEMforMX should be 32 for subtile too."
        )

    def test_subtile_mx_K_smaller_than_DU_implies_NoTailLoop_false(self):
        """The downstream NoTailLoop derivation (Solution.py:3603-3607) sets
        `NoTailLoop = (ASEM % DepthU == 0)`. With Phase 2's ASEM=32 and the
        usual subtile MX DepthU=256, 32 % 256 != 0, so NoTailLoop must be
        False (i.e. the kernel emits a tail loop).

        Today the bump forces ASEM = 256 == DepthU → NoTailLoop=True (no tail
        emitted), so this assertion fails.
        """
        state = _build_subtile_mx_state(asem=32, depthU=256)
        _run_assign_problem_independent(state)

        # Mirror Solution.py:3603-3607 derivation locally (the full
        # assignDerivedParameters chain is too brittle to reach in a unit test;
        # this just pins the consequence).
        no_tail_loop_derived = (
            state["AssertSummationElementMultiple"] % state["DepthU"] == 0
        )
        assert no_tail_loop_derived is False, (
            f"NoTailLoop would be True because ASEM "
            f"({state['AssertSummationElementMultiple']}) % DepthU "
            f"({state['DepthU']}) == 0. Phase 2 must keep ASEM=32 so the "
            f"tail-loop emit path is taken for K-tail problems."
        )


class TestSubtileBf16Asem:
    """BF16 subtile path is unaffected by the MX-only minASEMforMX bump.

    These tests pass today and serve as a regression guard: the Phase 2
    changes are scoped to the MX bump and must not perturb non-MX subtile
    kernels.
    """

    def test_bf16_subtile_asem_32_unchanged(self):
        """Non-MX subtile kernels must keep ASEM=32 even today."""
        state = _build_subtile_bf16_state(asem=32, depthU=64)
        _run_assign_problem_independent(state)

        assert state["AssertSummationElementMultiple"] == 32, (
            f"BF16 subtile ASEM was changed to "
            f"{state['AssertSummationElementMultiple']} (expected 32)."
        )

    def test_bf16_subtile_K_tail_implies_NoTailLoop_false(self):
        """BF16 subtile with DepthU=64, ASEM=32 should emit a tail loop
        (32 % 64 != 0). This passes today; Phase 2 must preserve it.
        """
        state = _build_subtile_bf16_state(asem=32, depthU=64)
        _run_assign_problem_independent(state)

        no_tail_loop_derived = (
            state["AssertSummationElementMultiple"] % state["DepthU"] == 0
        )
        assert no_tail_loop_derived is False
