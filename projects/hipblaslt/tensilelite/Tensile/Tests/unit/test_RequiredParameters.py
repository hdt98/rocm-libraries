################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""
Guard against the failure mode that produced the rare TN bf16 segfault in
PR #3679 ("codegen: handle unaligned K for TN").

Background
----------
PR #3679 added two new solution parameters to
``Tensile/Common/ValidParameters.py``:

  - ``BAddrInterleave`` (bool)
  - ``KRingShift``      (bool)

Both *materially* change the generated GPU assembly: new SGPR setup, new
``initBInterleaveG`` / ``initKRingShift`` runtime, new tail-loop branches,
new JIT offset macros, etc. However, neither was added to
``getRequiredParametersMin()`` in
``Tensile/Common/RequiredParameters.py``. That set is what
``SolutionStructs/Naming.py::getKernelNameMin`` iterates over to build the
on-disk kernel filename. As a result, two solutions differing only in those
two flags hashed to the same filename, the second compile silently
overwrote the first, and the runtime predicate dispatcher then handed the
wrong ``.co`` binary to a problem that needed the other one. The hipBLASLt
bench repro from PR #7349 ([AIHPBLAS-3494]) was a TN bf16 GEMM at
``-m 6144 -n 931 -k 5120`` which crashed with "Memory access fault by GPU
node-2".

The fix in PR #7349 added the two parameters to ``RequiredParametersMin``.
This test additionally locks in a *categorization invariant* so the same
class of bug cannot land again silently:

    Every entry in ``validParameters`` MUST be classified as either
    codegen-affecting (in ``getRequiredParametersMin()``) or non-codegen
    (in ``getParamsNotAffectingKernelName()``). A new parameter that is in
    neither bucket fails this test and forces the author to make an
    explicit decision at PR-review time.
"""

import pytest

from Tensile.Common.ValidParameters import validParameters
from Tensile.Common.RequiredParameters import (
    getRequiredParametersMin,
    getRequiredParametersFull,
    getParamsNotAffectingKernelName,
    validateParametersClassification,
)
from Tensile.SolutionStructs.Naming import _INTERNAL_ARGS


class TestEveryValidParameterIsClassified:
    """The PR #3679 regression guard: every solution parameter must declare
    whether it affects the kernel filename (and therefore the compiled
    .s/.co) or does not. Adding a new parameter to ``validParameters``
    without classifying it breaks this test.
    """

    def test_classification_is_total_and_disjoint(self):
        errors = validateParametersClassification()
        if errors:
            pytest.fail("\n\n".join(errors))

    def test_no_param_is_in_both_sets(self):
        inMin = set(getRequiredParametersMin())
        notInName = set(getParamsNotAffectingKernelName())
        overlap = inMin & notInName
        assert not overlap, (
            f"These parameters are claimed to BOTH affect and not affect "
            f"kernel codegen: {sorted(overlap)}. Each must be in exactly one."
        )

    def test_full_equals_all_valid_parameters(self):
        """Sanity belt: getRequiredParametersFull() should always be the
        complete set of validParameters keys."""
        assert set(getRequiredParametersFull()) == set(validParameters.keys())


class TestPR3679Regression:
    """Named, explicit assertions for the two parameters that triggered the
    original segfault. These remain even if the broader categorization
    check above is relaxed in the future, because the historical context
    is worth preserving in the test name."""

    @pytest.mark.parametrize("param", ["BAddrInterleave", "KRingShift"])
    def test_codegen_parameter_in_required_min(self, param):
        inMin = set(getRequiredParametersMin())
        assert param in inMin, (
            f"{param!r} is in validParameters but missing from "
            f"getRequiredParametersMin(). It changes the generated "
            f"assembly (new SGPRs, tail-loop control flow, etc.), so two "
            f"solutions differing only in {param!r} MUST get distinct "
            f"kernel filenames. Without that, the runtime predicate "
            f"dispatcher can load the wrong .co binary and segfault on a "
            f'TN GEMM (see PR #3679 / PR #7349 [AIHPBLAS-3494], "Memory '
            f'access fault by GPU node-2").'
        )

    @pytest.mark.parametrize(
        "param",
        ["AssertFree1DivByMT1LowbitGT1", "AssertKRingShiftTailWrapOnly"],
    )
    def test_derived_predicate_value_not_in_required_min(self, param):
        """The two new ``Assert*`` packed-int values added in PR #3679 are
        host-side runtime predicate inputs (consumed by predicate
        dispatch), not codegen inputs. They must NOT bloat the kernel
        filename."""
        notInName = set(getParamsNotAffectingKernelName())
        inMin = set(getRequiredParametersMin())
        assert param in notInName, (
            f"{param!r} is a derived host-side predicate value and must "
            f"be in getParamsNotAffectingKernelName()."
        )
        assert param not in inMin, (
            f"{param!r} is a derived host-side predicate value and must "
            f"NOT be in getRequiredParametersMin() (it does not affect "
            f"generated assembly)."
        )


class TestInternalArgsConsistency:
    """SolutionStructs/Naming.py owns the canonical list of "internal
    dispatch args" — solution parameters that are masked out of the kernel
    name. Those entries must also be in
    ``getParamsNotAffectingKernelName()`` (because they are valid params
    that legitimately do not affect kernel codegen). If someone adds a new
    internal arg to ``Naming.py::_INTERNAL_ARGS`` without also adding it
    here, this test catches the drift.
    """

    def test_internal_args_are_subset_of_not_in_name(self):
        notInName = set(getParamsNotAffectingKernelName())
        # Only enforce the rule for entries that are actually in
        # validParameters (Naming.py also masks ProblemType-side knobs we
        # don't list as solution params).
        validInternalArgs = {
            arg for arg in _INTERNAL_ARGS if arg in validParameters
        }
        missing = validInternalArgs - notInName
        assert not missing, (
            f"These entries are in Naming.py::_INTERNAL_ARGS and in "
            f"validParameters but missing from "
            f"getParamsNotAffectingKernelName(): {sorted(missing)}. "
            f"Internal dispatch args do not change generated assembly and "
            f"must be classified as not affecting the kernel name."
        )

    def test_no_internal_arg_in_required_min(self):
        inMin = set(getRequiredParametersMin())
        validInternalArgs = {
            arg for arg in _INTERNAL_ARGS if arg in validParameters
        }
        leaked = validInternalArgs & inMin
        assert not leaked, (
            f"These internal dispatch args are erroneously in "
            f"getRequiredParametersMin(): {sorted(leaked)}. Internal args "
            f"are masked out of the kernel name and must not affect it."
        )
