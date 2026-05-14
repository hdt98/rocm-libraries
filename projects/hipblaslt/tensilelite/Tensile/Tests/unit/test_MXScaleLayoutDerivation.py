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
"""Unit tests for the MX-scale layout & transport derivation introduced by
PR #7386 ("Extend MX BufferLoad + NoSwizzle to all transposes on gfx1250").

These tests drive :func:`_deriveAndValidateMXScaleLayoutAndTransport`
directly, which lets us exercise every reject and defaulting branch
without standing up an entire solution-derivation pipeline. The
``sk_mxf*gemm_{quick,tdm,explicit}.yaml`` end-to-end suites only cover
the happy paths; this module fills the negative-coverage gap.
"""

import pytest

from Tensile.SolutionStructs.Solution import (
    _deriveAndValidateMXScaleLayoutAndTransport,
)


ISA_GFX950  = (9, 5, 0)
ISA_GFX1250 = (12, 5, 0)


def _make_state(*,
                isa=ISA_GFX1250,
                mxLoadInst="Auto",
                mxScaleFormat="Auto",
                tdmInst=0,
                streamK=3,
                mxBlockA=32,
                mxBlockB=32):
    """Build the minimal state dict consumed by the derivation helper."""
    return {
        "ISA": isa,
        "MXLoadInst": mxLoadInst,
        "MXScaleFormat": mxScaleFormat,
        "TDMInst": tdmInst,
        "StreamK": streamK,
        "ProblemType": {"MXBlockA": mxBlockA, "MXBlockB": mxBlockB},
        # `reject()` keys off NoReject -- we always want the standard
        # "mutate Valid in place" path.
        "NoReject": False,
    }


def _caps(*, hasTDM=True, hasMXScaleSwizzle=True):
    return {"HasTDM": hasTDM}, {"HasMXScaleSwizzle": hasMXScaleSwizzle}


def _run(state, asmCaps=None, archCaps=None):
    """Invoke the helper; defaults give a fully-capable arch."""
    asm, arch = (asmCaps, archCaps) if asmCaps is not None else _caps()
    return _deriveAndValidateMXScaleLayoutAndTransport(state, asm, arch, False)


# ---------------------------------------------------------------------------
# "Auto" defaulting
# ---------------------------------------------------------------------------

class TestAutoDefaulting:
    """``MXLoadInst="Auto"`` / ``MXScaleFormat="Auto"`` must resolve to the
    concrete values documented in ``ValidParameters.py``. The 5 cases
    below cover every branch of the defaulting logic."""

    @pytest.mark.parametrize("tdmInst,expected", [
        (3, "TDM"),         # TDMInst != 0  -> TDM
        (0, "BufferLoad"),  # TDMInst == 0  -> BufferLoad
    ], ids=["tdminst3_to_TDM", "tdminst0_to_BufferLoad"])
    def test_mxloadinst_auto(self, tdmInst, expected):
        state = _make_state(mxLoadInst="Auto", tdmInst=tdmInst, streamK=3)
        assert _run(state) is True
        assert state["MXLoadInst"] == expected

    @pytest.mark.parametrize("isa,mxLoadInst,tdmInst,streamK,hasMXBlock,expected", [
        (ISA_GFX1250, "TDM",        3, 3, True,  "InMemorySwizzle"),  # TDM -> IMS
        (ISA_GFX950,  "BufferLoad", 0, 0, True,  "HostPreSwizzle"),   # gfx950+MX+BL -> HPS
        (ISA_GFX1250, "BufferLoad", 0, 3, True,  "NoSwizzle"),        # otherwise -> NS
        (ISA_GFX1250, "BufferLoad", 0, 3, False, "NoSwizzle"),        # no MX -> NS
    ], ids=["TDM_to_IMS", "gfx950_BL_MX_to_HPS",
            "gfx1250_BL_to_NS", "no_mx_block_to_NS"])
    def test_mxscaleformat_auto(self, isa, mxLoadInst, tdmInst, streamK,
                                hasMXBlock, expected):
        mxA = 32 if hasMXBlock else 0
        mxB = 32 if hasMXBlock else 0
        state = _make_state(isa=isa, mxLoadInst=mxLoadInst,
                            mxScaleFormat="Auto", tdmInst=tdmInst,
                            streamK=streamK, mxBlockA=mxA, mxBlockB=mxB)
        assert _run(state) is True
        assert state["MXScaleFormat"] == expected

    def test_explicit_value_not_overwritten(self):
        # Regression guard: if the user pinned a concrete value, the
        # helper must not silently rewrite it back to the Auto default.
        state = _make_state(mxLoadInst="BufferLoad", mxScaleFormat="NoSwizzle",
                            tdmInst=0, streamK=3)
        assert _run(state) is True
        assert state["MXLoadInst"] == "BufferLoad"
        assert state["MXScaleFormat"] == "NoSwizzle"


# ---------------------------------------------------------------------------
# TDMInst promotion
# ---------------------------------------------------------------------------

class TestTDMInstPromotion:
    """When the user pins ``MXLoadInst="TDM"`` but leaves the default
    ``TDMInst=0``, the helper bumps ``TDMInst`` to ``3`` (A+B). This
    honors the "both-or-none" invariant on ``TDMInst``."""

    @pytest.mark.parametrize("mxLoadInst,tdmInstIn,tdmInstOut", [
        ("TDM",        0, 3),  # TDM + 0 promotes to 3
        ("TDM",        1, 1),  # TDM + nonzero is preserved verbatim
        ("BufferLoad", 0, 0),  # non-TDM does not promote
    ], ids=["TDM_zero_promotes", "TDM_nonzero_preserved",
            "BufferLoad_no_promote"])
    def test_promotion(self, mxLoadInst, tdmInstIn, tdmInstOut):
        state = _make_state(mxLoadInst=mxLoadInst, tdmInst=tdmInstIn,
                            streamK=3)
        assert _run(state) is True
        assert state["TDMInst"] == tdmInstOut


# ---------------------------------------------------------------------------
# Rejects -- every PR-added reject has at least one row here
# ---------------------------------------------------------------------------

# Each row: (id, state-kwargs, caps-kwargs). All rows must drive the helper
# to a `reject` call (returns False and sets state["Valid"] = False).
REJECT_CASES = [
    # 1. MXLoadInst=GlobalLoad is reserved/not implemented.
    pytest.param(
        dict(mxLoadInst="GlobalLoad"),
        dict(),
        id="globalload_reserved",
    ),
    # 2. MXLoadInst=TDM requires asmCaps.HasTDM.
    pytest.param(
        dict(mxLoadInst="TDM", tdmInst=3),
        dict(hasTDM=False),
        id="tdm_without_hastdm_cap",
    ),
    # 3. Explicit non-TDM transport with non-zero TDMInst is inconsistent.
    pytest.param(
        dict(mxLoadInst="BufferLoad", tdmInst=3, streamK=3),
        dict(),
        id="bufferload_with_nonzero_tdminst",
    ),
    # 4a/4b. Swizzled formats require archCaps.HasMXScaleSwizzle.
    pytest.param(
        dict(mxLoadInst="TDM", mxScaleFormat="InMemorySwizzle", tdmInst=3),
        dict(hasMXScaleSwizzle=False),
        id="in_memory_swizzle_without_swizzle_cap",
    ),
    pytest.param(
        dict(isa=ISA_GFX950, mxLoadInst="BufferLoad",
             mxScaleFormat="HostPreSwizzle", tdmInst=0, streamK=0),
        dict(hasMXScaleSwizzle=False),
        id="host_pre_swizzle_without_swizzle_cap",
    ),
    # 5. InMemorySwizzle requires MXLoadInst=TDM.
    pytest.param(
        dict(mxLoadInst="BufferLoad", mxScaleFormat="InMemorySwizzle",
             tdmInst=0, streamK=3),
        dict(),
        id="in_memory_swizzle_with_bufferload",
    ),
    # 6. HostPreSwizzle requires MXLoadInst=BufferLoad. Pinning TDM
    #    actually trips reject #8 ("TDM only produces IMS") first because
    #    it is sequenced earlier; we just assert that *a* reject fires.
    pytest.param(
        dict(isa=ISA_GFX950, mxLoadInst="TDM",
             mxScaleFormat="HostPreSwizzle", tdmInst=3),
        dict(),
        id="host_pre_swizzle_with_tdm",
    ),
    # 7. HostPreSwizzle is gfx950-only.
    pytest.param(
        dict(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
             mxScaleFormat="HostPreSwizzle", tdmInst=0, streamK=3),
        dict(),
        id="host_pre_swizzle_off_gfx950",
    ),
    # 8. MXLoadInst=TDM currently only produces InMemorySwizzle.
    pytest.param(
        dict(mxLoadInst="TDM", mxScaleFormat="NoSwizzle", tdmInst=3),
        dict(),
        id="tdm_with_no_swizzle",
    ),
    # 9. gfx1250 + MX + TDMInst=0 + StreamK=0 is the unvalidated GSU
    #    NoSwizzle path that PR #7386 explicitly excludes from tuning.
    pytest.param(
        dict(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
             mxScaleFormat="NoSwizzle", tdmInst=0, streamK=0),
        dict(),
        id="gfx1250_mx_without_tdminst_or_streamk",
    ),
]


@pytest.mark.parametrize("state_kw,caps_kw", REJECT_CASES)
def test_reject(state_kw, caps_kw):
    """Each PR-added reject must (a) return False and (b) flip
    state["Valid"] -- i.e., the reject() path actually executed rather
    than the function simply falling out the bottom."""
    state = _make_state(**state_kw)
    assert _run(state, *_caps(**caps_kw)) is False
    assert state["Valid"] is False


# ---------------------------------------------------------------------------
# Targeted positive guards that complement the parametrized rejects
# ---------------------------------------------------------------------------

class TestRejectComplements:
    """Each test here asserts that a near-miss of one of the rejects
    above is correctly *not* rejected -- guarding the precise boundary
    of each rule."""

    def test_auto_with_nonzero_tdminst_resolves_to_tdm(self):
        # Counterpoint to "bufferload_with_nonzero_tdminst": the
        # incompatibility reject must only fire on an *explicit* non-TDM
        # transport, never via the Auto resolution path.
        state = _make_state(mxLoadInst="Auto", tdmInst=3, streamK=3)
        assert _run(state) is True
        assert state["MXLoadInst"] == "TDM"

    def test_gfx1250_mx_with_streamk_escape_passes(self):
        # Counterpoint to "gfx1250_mx_without_tdminst_or_streamk": the
        # StreamK!=0 escape hatch is the path PR #7386 validated E2E.
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="NoSwizzle", tdmInst=0, streamK=3)
        assert _run(state) is True

    def test_gfx1250_without_mx_block_passes(self):
        # The gfx1250-needs-TDMInst guard is gated on hasMXBlock, so a
        # plain non-MX kernel threads through even with TDMInst=0 +
        # StreamK=0.
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="Auto", tdmInst=0, streamK=0,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True


# ---------------------------------------------------------------------------
# Valid (MXLoadInst, MXScaleFormat) pairs at the explicit level
# ---------------------------------------------------------------------------

# Beyond the Auto-defaulting tests above, also confirm that the three
# documented valid pairs survive when pinned explicitly. The yaml side
# (sk_mxf*gemm_explicit.yaml) exercises these end-to-end on FFM; here we
# just make sure derivation accepts them so a yaml regression doesn't
# fail with a confusing "no enumerated solutions" message.
@pytest.mark.parametrize("isa,mxLoadInst,mxScaleFormat,tdmInst,streamK", [
    (ISA_GFX1250, "TDM",        "InMemorySwizzle", 3, 3),
    (ISA_GFX1250, "BufferLoad", "NoSwizzle",       0, 3),
    (ISA_GFX950,  "BufferLoad", "HostPreSwizzle",  0, 0),
], ids=["TDM_IMS_gfx1250", "BL_NS_gfx1250_streamk",
        "BL_HPS_gfx950"])
def test_explicit_valid_pair(isa, mxLoadInst, mxScaleFormat, tdmInst, streamK):
    state = _make_state(isa=isa, mxLoadInst=mxLoadInst,
                        mxScaleFormat=mxScaleFormat, tdmInst=tdmInst,
                        streamK=streamK)
    assert _run(state) is True
    assert state["MXLoadInst"]    == mxLoadInst
    assert state["MXScaleFormat"] == mxScaleFormat


# ---------------------------------------------------------------------------
# Helper-contract guards
# ---------------------------------------------------------------------------

class TestHelperContract:
    """Guards that the helper only reads the documented inputs and
    handles their edge cases correctly."""

    def test_missing_HasMXScaleSwizzle_treated_as_false(self):
        # archCaps without the key (some older archs omit it) must not
        # raise -- archCaps.get(..., False) is the documented contract.
        state = _make_state(mxLoadInst="TDM",
                            mxScaleFormat="InMemorySwizzle", tdmInst=3)
        asm, arch = {"HasTDM": True}, {}  # no HasMXScaleSwizzle key
        assert _deriveAndValidateMXScaleLayoutAndTransport(
            state, asm, arch, False) is False
        assert state["Valid"] is False

    def test_no_mx_block_bypasses_format_vs_transport_checks(self):
        # Without MX scales, the format-vs-transport rules are bypassed
        # entirely. A pair that would otherwise be rejected (BufferLoad
        # + InMemorySwizzle) threads through cleanly when hasMXBlock is
        # False -- because the format then becomes a don't-care.
        state = _make_state(isa=ISA_GFX1250, mxLoadInst="BufferLoad",
                            mxScaleFormat="InMemorySwizzle",
                            tdmInst=0, streamK=3,
                            mxBlockA=0, mxBlockB=0)
        assert _run(state) is True
