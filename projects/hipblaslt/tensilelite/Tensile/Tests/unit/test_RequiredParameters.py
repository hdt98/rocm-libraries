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
Lock in the invariant that every key in validParameters is classified as
either codegen-affecting (in getRequiredParametersMin) or non-codegen
(in getParamsNotAffectingKernelName). A new parameter that is in neither
bucket fails this test, forcing an explicit decision at PR-review time.
Without this guard, a codegen-affecting parameter missing from
RequiredParametersMin can cause two distinct solutions to share one .co
file, leading to a runtime memory access fault when the wrong binary is
dispatched.
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
    def test_classification_is_total_and_disjoint(self):
        errors = validateParametersClassification()
        if errors:
            pytest.fail("\n\n".join(errors))

    def test_no_param_is_in_both_sets(self):
        overlap = set(getRequiredParametersMin()) & set(
            getParamsNotAffectingKernelName()
        )
        assert not overlap, (
            f"Parameters in both buckets: {sorted(overlap)}. Each must be "
            f"in exactly one."
        )

    def test_full_equals_all_valid_parameters(self):
        assert set(getRequiredParametersFull()) == set(validParameters.keys())


class TestKnownClassifications:
    """Named assertions for parameters whose classification is load-bearing
    enough that explicit coverage is preferable to relying solely on the
    broader invariant above."""

    @pytest.mark.parametrize("param", ["BAddrInterleave", "KRingShift"])
    def test_codegen_parameter_in_required_min(self, param):
        assert param in set(getRequiredParametersMin()), (
            f"{param!r} changes generated assembly; it must be in "
            f"getRequiredParametersMin() so kernels with different values "
            f"get distinct filenames."
        )

    @pytest.mark.parametrize(
        "param",
        ["AssertFree1DivByMT1LowbitGT1", "AssertKRingShiftTailWrapOnly"],
    )
    def test_derived_predicate_value_not_in_required_min(self, param):
        notInName = set(getParamsNotAffectingKernelName())
        inMin = set(getRequiredParametersMin())
        assert param in notInName and param not in inMin, (
            f"{param!r} is a host-side predicate value; it must be in "
            f"getParamsNotAffectingKernelName() and not in "
            f"getRequiredParametersMin()."
        )


class TestInternalArgsConsistency:
    """Naming.py::_INTERNAL_ARGS owns the canonical set of runtime-dispatch
    knobs that are masked out of the kernel name; mirror them here so drift
    is caught."""

    def _validInternalArgs(self):
        return {arg for arg in _INTERNAL_ARGS if arg in validParameters}

    def test_internal_args_are_subset_of_not_in_name(self):
        missing = self._validInternalArgs() - set(getParamsNotAffectingKernelName())
        assert not missing, (
            f"In Naming.py::_INTERNAL_ARGS but missing from "
            f"getParamsNotAffectingKernelName(): {sorted(missing)}."
        )

    def test_no_internal_arg_in_required_min(self):
        leaked = self._validInternalArgs() & set(getRequiredParametersMin())
        assert not leaked, (
            f"Internal dispatch args leaked into getRequiredParametersMin(): "
            f"{sorted(leaked)}."
        )
