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
################################################################################

import pytest
from unittest.mock import MagicMock, patch, Mock

from Tensile.Common.DataType import DataType
from Tensile.Activation import ActivationType
from Tensile.Contractions import (
    FreeIndex,
    BatchIndex,
    BoundIndex,
    ProblemType,
    ProblemPredicate,
    TaskPredicate,
    SizeMapping,
    InternalArgsSupport,
    Solution,
    extractDimPredicate,
    MIN_K_FOR_GSU
)

pytestmark = pytest.mark.unit


class TestFreeIndex:
    """Tests for FreeIndex class."""

    def test_free_index_initialization_with_all_params(self):
        """Test FreeIndex initialization with all parameters."""
        fi = FreeIndex(isA=True, i=0, c=1, d=2)

        assert fi.isA is True
        assert fi.i == 0
        assert fi.c == 1
        assert fi.d == 2

    def test_free_index_initialization_minimal(self):
        """Test FreeIndex initialization with minimal parameters."""
        fi = FreeIndex(isA=False)

        assert fi.isA is False
        assert fi.i is None
        assert fi.c is None
        assert fi.d is None

    def test_free_index_is_a_true(self):
        """Test FreeIndex with isA=True (for matrix A)."""
        fi = FreeIndex(isA=True, i=2, c=0, d=0)

        assert fi.isA is True
        assert fi.i == 2

    def test_free_index_is_a_false(self):
        """Test FreeIndex with isA=False (for matrix B)."""
        fi = FreeIndex(isA=False, i=1, c=1, d=1)

        assert fi.isA is False
        assert fi.i == 1

    def test_free_index_state_keys(self):
        """Test FreeIndex has correct StateKeys."""
        assert FreeIndex.StateKeys == ['isA', 'i', 'c', 'd']


class TestBatchIndex:
    """Tests for BatchIndex class."""

    def test_batch_index_initialization_with_all_params(self):
        """Test BatchIndex initialization with all parameters."""
        bi = BatchIndex(a=0, b=1, c=2, d=3)

        assert bi.a == 0
        assert bi.b == 1
        assert bi.c == 2
        assert bi.d == 3

    def test_batch_index_initialization_minimal(self):
        """Test BatchIndex initialization with default None values."""
        bi = BatchIndex()

        assert bi.a is None
        assert bi.b is None
        assert bi.c is None
        assert bi.d is None

    def test_batch_index_partial_initialization(self):
        """Test BatchIndex initialization with partial parameters."""
        bi = BatchIndex(a=1, c=2)

        assert bi.a == 1
        assert bi.b is None
        assert bi.c == 2
        assert bi.d is None

    def test_batch_index_state_keys(self):
        """Test BatchIndex has correct StateKeys."""
        assert BatchIndex.StateKeys == ['a', 'b', 'c', 'd']


class TestBoundIndex:
    """Tests for BoundIndex class."""

    def test_bound_index_initialization_with_all_params(self):
        """Test BoundIndex initialization with all parameters."""
        bi = BoundIndex(a=0, b=1, aMirror=True, bMirror=False)

        assert bi.a == 0
        assert bi.b == 1
        assert bi.aMirror is True
        assert bi.bMirror is False

    def test_bound_index_initialization_minimal(self):
        """Test BoundIndex initialization with default values."""
        bi = BoundIndex()

        assert bi.a is None
        assert bi.b is None
        assert bi.aMirror is False
        assert bi.bMirror is False

    def test_bound_index_mirror_dims(self):
        """Test BoundIndex with mirror dimensions enabled."""
        bi = BoundIndex(a=2, b=3, aMirror=True, bMirror=True)

        assert bi.aMirror is True
        assert bi.bMirror is True

    def test_bound_index_state_keys(self):
        """Test BoundIndex has correct StateKeys."""
        assert BoundIndex.StateKeys == ['a', 'b', 'aMirror', 'bMirror']


class TestProblemType:
    """Tests for ProblemType class."""

    def test_problem_type_initialization(self):
        """Test ProblemType basic initialization."""
        free_indices = [FreeIndex(True, 0, 0, 0)]
        batch_indices = [BatchIndex(0, 0, 1, 1)]
        bound_indices = [BoundIndex(1, 1)]

        pt = ProblemType(
            freeIndices=free_indices,
            batchIndices=batch_indices,
            boundIndices=bound_indices,
            aDims=2,
            bDims=2,
            cDims=2,
            dDims=2
        )

        assert pt.freeIndices == free_indices
        assert pt.batchIndices == batch_indices
        assert pt.boundIndices == bound_indices
        assert pt.aDims == 2
        assert pt.bDims == 2
        assert pt.cDims == 2
        assert pt.dDims == 2

    def test_problem_type_from_original_state_basic_gemm(self):
        """Test ProblemType.FromOriginalState with basic GEMM configuration."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,  # Float
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert len(pt.freeIndices) == 2
        assert len(pt.boundIndices) == 1
        assert len(pt.batchIndices) == 0
        assert pt.aDims == 2
        assert pt.bDims == 2
        assert pt.cDims == 2

    def test_problem_type_from_original_state_with_batch(self):
        """Test ProblemType.FromOriginalState with batched GEMM."""
        d = {
            'TotalIndices': 4,
            'NumIndicesC': 3,
            'IndicesSummation': [3],
            'IndicesBatch': [2],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 3, 2],
            'IndexAssignmentsB': [3, 1, 2],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': True,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert len(pt.freeIndices) == 2
        assert len(pt.boundIndices) == 1
        assert len(pt.batchIndices) == 1
        assert pt.batched is True

    def test_problem_type_from_original_state_with_data_types(self):
        """Test ProblemType.FromOriginalState with different data types."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 1,  # Half
            'DestDataType': 0,  # Float
            'ComputeDataType': 0,  # Float
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.aType == DataType(1)
        assert pt.cType == DataType(0)
        assert pt.dType == DataType(0)
        assert pt.computeType == DataType(0)

    def test_problem_type_from_original_state_with_f8_b8_types(self):
        """Test ProblemType.FromOriginalState with F8/B8 mixed types."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 17,  # Float8BFloat8 (index in DataType.properties)
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.aType == DataType("F8")
        assert pt.bType == DataType("B8")

    def test_problem_type_from_original_state_with_mirror_dims(self):
        """Test ProblemType.FromOriginalState with mirror dimensions."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'MirrorDimsA': [2],
            'MirrorDimsB': [],
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.boundIndices[0].aMirror is True
        assert pt.boundIndices[0].bMirror is False
        assert pt.mirrorDimsA == [2]
        assert pt.mirrorDimsB == []

    def test_problem_type_from_original_state_with_high_precision_accumulate(self):
        """Test ProblemType.FromOriginalState with HighPrecisionAccumulate."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'HighPrecisionAccumulate': True,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.highPrecisionAccumulate is True

    def test_problem_type_from_original_state_strided_batched(self):
        """Test ProblemType.FromOriginalState with StridedBatched flag."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': True,
            'StridedBatched': True,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.stridedBatched is True

    def test_problem_type_from_original_state_grouped_gemm(self):
        """Test ProblemType.FromOriginalState with GroupedGemm flag."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'GroupedGemm': True,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.groupedGemm is True

    def test_problem_type_from_original_state_with_bias(self):
        """Test ProblemType.FromOriginalState with bias configuration."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'UseBias': 2,
            'BiasDataTypeList': [0, 1],
            'BiasSrc': 'D',
            'SetConstStrideBias': [0, 1],
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.useBias == 2
        assert pt.biasDataTypeWhiteList == [0, 1]
        assert pt.biasSrcWhiteList == [3]  # 'D' is index 3 in ['A', 'B', 'C', 'D']
        assert pt.setConstStrideBias == [0, 1]

    def test_problem_type_from_original_state_with_activation(self):
        """Test ProblemType.FromOriginalState with activation function."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationType': 'relu',
            'ActivationComputeDataType': 0,
            'ActivationNoGuard': True,
            'Gradient': True
        }

        pt = ProblemType.FromOriginalState(d)

        assert pt.activationType == ActivationType('relu')
        assert pt.activationNoGuard is True
        assert pt.useGradient is True

    def test_problem_type_index_names_basic(self):
        """Test ProblemType.indexNames property for basic GEMM."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)
        aNames, bNames, cNames, dNames, sumNames = pt.indexNames

        assert len(aNames) == 2
        assert len(bNames) == 2
        assert len(cNames) == 2
        assert len(dNames) == 2
        assert len(sumNames) == 1

    def test_problem_type_operation_identifier(self):
        """Test ProblemType.operationIdentifier property."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)
        op_id = pt.operationIdentifier

        assert op_id.startswith('Contraction_')
        assert 'A' in op_id
        assert 'B' in op_id
        assert 'C' in op_id
        assert 'D' in op_id

    def test_problem_type_placeholder_str_with_operation(self):
        """Test ProblemType.placeholderStr with operation flag."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)
        placeholder = pt.placeholderStr(includeOperation=True)

        assert 'Contraction_' in placeholder
        assert 'StridedBatched' in placeholder

    def test_problem_type_placeholder_str_with_type(self):
        """Test ProblemType.placeholderStr with type flag."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'HighPrecisionAccumulate': True,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)
        placeholder = pt.placeholderStr(includeType=True)

        assert 'Type_' in placeholder
        assert 'HPA' in placeholder

    def test_problem_type_predicates(self):
        """Test ProblemType.predicates method."""
        d = {
            'TotalIndices': 3,
            'NumIndicesC': 2,
            'IndicesSummation': [2],
            'IndicesBatch': [],
            'IndicesFree': [0, 1],
            'IndexAssignmentsA': [0, 2],
            'IndexAssignmentsB': [2, 1],
            'ComplexConjugateA': False,
            'ComplexConjugateB': False,
            'DataType': 0,
            'TransposeA': False,
            'TransposeB': False,
            'Batched': False,
            'ActivationComputeDataType': 0
        }

        pt = ProblemType.FromOriginalState(d)
        predicates = pt.predicates(includeOperation=True, includeType=True)

        assert isinstance(predicates, list)
        assert len(predicates) > 0


class TestExtractDimPredicate:
    """Tests for extractDimPredicate function."""

    def test_extract_dim_predicate_single_value(self):
        """Test extractDimPredicate with single dimension."""
        from Tensile.Contractions import ProblemPredicate

        value = {0: 64}
        result = extractDimPredicate(ProblemPredicate, 'key', value, 'TestPredicate')

        assert result is not None
        assert result.tag == 'TestPredicate'
        assert result.index == 0
        assert result.value == 64

    def test_extract_dim_predicate_multiple_values(self):
        """Test extractDimPredicate with multiple dimensions."""
        from Tensile.Contractions import ProblemPredicate

        value = {0: 64, 1: 128}
        result = extractDimPredicate(ProblemPredicate, 'key', value, 'TestPredicate')

        assert result is not None
        assert result.tag == 'And'

    def test_extract_dim_predicate_skip_minus_one(self):
        """Test extractDimPredicate skips -1 values."""
        from Tensile.Contractions import ProblemPredicate

        value = {0: -1, 1: 64}
        result = extractDimPredicate(ProblemPredicate, 'key', value, 'TestPredicate')

        assert result is not None
        assert result.index == 1
        assert result.value == 64

    def test_extract_dim_predicate_all_minus_one(self):
        """Test extractDimPredicate with all -1 values returns None."""
        from Tensile.Contractions import ProblemPredicate

        value = {0: -1, 1: -1}
        result = extractDimPredicate(ProblemPredicate, 'key', value, 'TestPredicate')

        assert result is None


class TestTaskPredicate:
    """Tests for TaskPredicate class."""

    def test_task_predicate_from_original_key_pair_workspace(self):
        """Test TaskPredicate.FromOriginalKeyPair with workspace size."""
        pair = ('_WorkspaceSizePerElemC', 10)

        result = TaskPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'WorkspaceCheck'

    def test_task_predicate_from_original_key_pair_workspace_zero(self):
        """Test TaskPredicate.FromOriginalKeyPair with zero workspace."""
        pair = ('_WorkspaceSizePerElemC', 0)

        result = TaskPredicate.FromOriginalKeyPair(pair)

        assert result is None

    def test_task_predicate_from_original_key_pair_other_key(self):
        """Test TaskPredicate.FromOriginalKeyPair with unrecognized key."""
        pair = ('SomeOtherKey', 42)

        result = TaskPredicate.FromOriginalKeyPair(pair)

        assert result is None

    def test_task_predicate_extra_predicates_with_streamk(self):
        """Test TaskPredicate.ExtraPredicates with StreamK enabled."""
        state = {'StreamK': 1}

        result = TaskPredicate.ExtraPredicates(state)

        assert isinstance(result, list)
        assert len(result) == 0  # No LaunchLimits for StreamK

    def test_task_predicate_extra_predicates_without_streamk(self):
        """Test TaskPredicate.ExtraPredicates without StreamK."""
        state = {}

        result = TaskPredicate.ExtraPredicates(state)

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].tag == 'LaunchLimits'


class TestProblemPredicate:
    """Tests for ProblemPredicate class."""

    def test_problem_predicate_from_original_key_pair_ai_greater(self):
        """Test ProblemPredicate.FromOriginalKeyPair with AI greater than."""
        pair = ('AssertAIGreaterThanEqual', 2.5)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'AIGreaterThanEqual'
        assert result.value == 2.5

    def test_problem_predicate_from_original_key_pair_ai_less(self):
        """Test ProblemPredicate.FromOriginalKeyPair with AI less than."""
        pair = ('AssertAILessThanEqual', 5.0)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'AILessThanEqual'
        assert result.value == 5.0

    def test_problem_predicate_from_original_key_pair_free0_multiple(self):
        """Test ProblemPredicate.FromOriginalKeyPair with Free0ElementMultiple."""
        pair = ('AssertFree0ElementMultiple', 16)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'Free0SizeMultiple'
        assert result.value == 16
        assert result.index == 0

    def test_problem_predicate_from_original_key_pair_free1_multiple(self):
        """Test ProblemPredicate.FromOriginalKeyPair with Free1ElementMultiple."""
        pair = ('AssertFree1ElementMultiple', 32)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'Free1SizeMultiple'
        assert result.value == 32
        assert result.index == 0

    def test_problem_predicate_from_original_key_pair_summation_multiple(self):
        """Test ProblemPredicate.FromOriginalKeyPair with SummationElementMultiple."""
        pair = ('AssertSummationElementMultiple', 8)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is not None
        assert result.tag == 'BoundSizeMultiple'
        assert result.value == 8
        assert result.index == -1

    def test_problem_predicate_from_original_key_pair_multiple_value_one(self):
        """Test ProblemPredicate.FromOriginalKeyPair returns None for multiple=1."""
        pair = ('AssertFree0ElementMultiple', 1)

        result = ProblemPredicate.FromOriginalKeyPair(pair)

        assert result is None

    def test_problem_predicate_from_original_key_pair_unknown_assert(self):
        """Test ProblemPredicate.FromOriginalKeyPair raises for unknown assertion."""
        pair = ('AssertUnknownKey', 42)

        with pytest.raises(RuntimeError, match="Unknown assertion key"):
            ProblemPredicate.FromOriginalKeyPair(pair)

    def test_problem_predicate_compound_predicates_batch_size_equal(self):
        """Test ProblemPredicate.CompoundPredicates with BatchSizeEqual."""
        state = {
            'BatchSizeEqual': 4,
            'MacroTile0': 128,
            'MacroTile1': 128,
            'GlobalSplitU': 1,
            'GlobalReadVectorWidthA': 4,
            'GlobalReadVectorWidthB': 4,
            'PackedC0IndicesX': [0],
            'StreamK': 0,
            'ProblemType': {'TLUA': True, 'TLUB': False, 'SwizzleTensorA': False, 'SwizzleTensorB': False},
            'InternalSupportParams': {
                'KernArgsVersion': 1,
                'SupportUserGSU': True,
                'SupportCustomWGM': True,
                'SupportCustomStaggerU': True,
                'UseUniversalArgs': False,
                'UseSFC': False
            }
        }
        problem_type = MagicMock()
        problem_type.aType = DataType(0)

        result = ProblemPredicate.CompoundPredicates(state, problem_type)

        assert any(p.tag == 'BatchSizeEqual' for p in result)

    def test_problem_predicate_compound_predicates_global_split_u_check(self):
        """Test ProblemPredicate.CompoundPredicates with GlobalSplitU check."""
        state = {
            '_GlobalAccumulation': 'SingleBuffer',
            'GlobalSplitU': 2,
            'MacroTile0': 128,
            'MacroTile1': 128,
            'GlobalReadVectorWidthA': 4,
            'GlobalReadVectorWidthB': 4,
            'PackedC0IndicesX': [0],
            'StreamK': 0,
            'ProblemType': {'TLUA': True, 'TLUB': False, 'SwizzleTensorA': False, 'SwizzleTensorB': False},
            'InternalSupportParams': {
                'KernArgsVersion': 1,
                'SupportUserGSU': True,
                'SupportCustomWGM': True,
                'SupportCustomStaggerU': True,
                'UseUniversalArgs': False,
                'UseSFC': False
            }
        }
        problem_type = MagicMock()
        problem_type.aType = DataType(0)

        result = ProblemPredicate.CompoundPredicates(state, problem_type)

        assert any(p.tag == 'GlobalSplitUCheckMinK' for p in result)


class TestSizeMapping:
    """Tests for SizeMapping class."""

    def test_size_mapping_from_original_state_basic(self):
        """Test SizeMapping.FromOriginalState with basic configuration."""
        d = {
            'NumThreads': 256,
            'WavefrontSize': 64,
            'WorkGroup': [16, 16, 1],
            'MacroTile0': 128,
            'MacroTile1': 128,
            'ThreadTile': [4, 4],
            'ThreadTile0': 4,
            'ThreadTile1': 4,
            'MatrixInstruction': [16, 16, 1, 4],
            'GlobalReadVectorWidthA': 4,
            'GlobalReadVectorWidthB': 4,
            'StoreVectorWidth': 4,
            'DepthU': 16,
            'GlobalSplitU': 1,
            '_WorkspaceSizePerElemC': 0,
            '_WorkspaceSizePerElemBias': 0,
            'ActivationFused': False,
            'CustomKernelName': '',
            'WorkGroupMappingXCC': 0,
            'WorkGroupMappingXCCGroup': 1,
            'GlobalSplitUCoalesced': False,
            'GlobalSplitUWorkGroupMappingRoundRobin': False,
            'CUOccupancy': 1,
            'PrefetchGlobalRead': 1,
            'MathClocksUnrolledLoop': 0,
            'NonTemporalA': 0,
            'NonTemporalB': 0,
            'NonTemporalD': 0,
            'UseCustomMainLoopSchedule': False,
            'WaveSeparateGlobalReadA': False,
            'WaveSeparateGlobalReadB': False,
            'UnrollLoopSwapGlobalReadOrder': False,
            'DirectToVgprA': False,
            'DirectToVgprB': False,
            'NumLoadsCoalescedA': 1,
            'NumLoadsCoalescedB': 1,
            'MIWaveGroup': [1, 1],
            'VectorWidthA': 1,
            'VectorWidthB': 1,
            'LocalSplitU': 1,
            'DirectToLdsA': False,
            'DirectToLdsB': False,
            '_GlobalAccumulation': None,
            'KernelLanguage': 'Assembly',
            'WorkGroupMapping': 1,
            'SpaceFillingAlgo': [],
            'EnableMatrixInstruction': False,
            'NumElementsPerThread': 16,
            'NumElementsPerBatchStore': 4
        }

        sm = SizeMapping.FromOriginalState(d)

        assert sm.waveNum == 4  # 256 / 64
        assert sm.workGroup == [16, 16, 1]
        assert sm.macroTile == [128, 128, 1]
        assert sm.grvwA == 4
        assert sm.grvwB == 4

    def test_size_mapping_from_original_state_with_global_accumulation(self):
        """Test SizeMapping.FromOriginalState with global accumulation."""
        d = {
            'NumThreads': 256,
            'WavefrontSize': 64,
            'WorkGroup': [16, 16, 1],
            'MacroTile0': 128,
            'MacroTile1': 128,
            'ThreadTile': [4, 4],
            'ThreadTile0': 4,
            'ThreadTile1': 4,
            'MatrixInstruction': [16, 16, 1, 4],
            'GlobalReadVectorWidthA': 4,
            'GlobalReadVectorWidthB': 4,
            'StoreVectorWidth': 4,
            'DepthU': 16,
            'GlobalSplitU': 2,
            '_WorkspaceSizePerElemC': 0,
            '_WorkspaceSizePerElemBias': 0,
            'ActivationFused': False,
            'CustomKernelName': '',
            'WorkGroupMappingXCC': 0,
            'WorkGroupMappingXCCGroup': 1,
            'GlobalSplitUCoalesced': False,
            'GlobalSplitUWorkGroupMappingRoundRobin': False,
            'CUOccupancy': 1,
            'PrefetchGlobalRead': 1,
            'MathClocksUnrolledLoop': 0,
            'NonTemporalA': 0,
            'NonTemporalB': 0,
            'NonTemporalD': 0,
            'UseCustomMainLoopSchedule': False,
            'WaveSeparateGlobalReadA': False,
            'WaveSeparateGlobalReadB': False,
            'UnrollLoopSwapGlobalReadOrder': False,
            'DirectToVgprA': False,
            'DirectToVgprB': False,
            'NumLoadsCoalescedA': 1,
            'NumLoadsCoalescedB': 1,
            'MIWaveGroup': [1, 1],
            'VectorWidthA': 1,
            'VectorWidthB': 1,
            'LocalSplitU': 1,
            'DirectToLdsA': False,
            'DirectToLdsB': False,
            '_GlobalAccumulation': 'SingleBuffer',
            'KernelLanguage': 'Assembly',
            'WorkGroupMapping': 1,
            'SpaceFillingAlgo': [],
            'EnableMatrixInstruction': False,
            'NumElementsPerThread': 16,
            'NumElementsPerBatchStore': 4
        }

        sm = SizeMapping.FromOriginalState(d)

        assert sm.globalAccumulation == 1  # SingleBuffer
        assert sm.globalSplitU == 2

    def test_size_mapping_read_original_macro_tile(self):
        """Test SizeMapping.ReadOriginalMacroTile."""
        d = {'MacroTile0': 64, 'MacroTile1': 128}

        result = SizeMapping.ReadOriginalMacroTile(d)

        assert result == [64, 128, 1]


class TestInternalArgsSupport:
    """Tests for InternalArgsSupport class."""

    def test_internal_args_support_from_original_state(self):
        """Test InternalArgsSupport.FromOriginalState."""
        d = {
            'InternalSupportParams': {
                'KernArgsVersion': 2,
                'SupportUserGSU': True,
                'SupportCustomWGM': True,
                'SupportCustomStaggerU': False,
                'UseUniversalArgs': True,
                'UseSFC': False
            },
            'SpaceFillingAlgo': []
        }

        ias = InternalArgsSupport.FromOriginalState(d)

        assert ias.version == 2
        assert ias.gsu is True
        assert ias.wgm is True
        assert ias.staggerU is False
        assert ias.useUniversalArgs is True

    def test_internal_args_support_from_original_state_with_sfc(self):
        """Test InternalArgsSupport.FromOriginalState with space filling curve."""
        d = {
            'InternalSupportParams': {
                'KernArgsVersion': 2,
                'SupportUserGSU': True,
                'SupportCustomWGM': True,
                'SupportCustomStaggerU': False,
                'UseUniversalArgs': True,
                'UseSFC': False
            },
            'SpaceFillingAlgo': [1, 2]  # Non-empty
        }

        ias = InternalArgsSupport.FromOriginalState(d)

        assert ias.useSFC is True  # Should be True due to non-empty SpaceFillingAlgo


class TestSolution:
    """Tests for Solution class."""

    def test_solution_initialization(self):
        """Test Solution basic initialization."""
        sol = Solution()

        assert sol.name is None
        assert sol.problemType is None
        assert sol.debugKernel is False
        assert sol.index is None

    @patch('Tensile.Contractions.ProblemType.FromOriginalState')
    @patch('Tensile.Contractions.ProblemPredicate.FromOriginalState')
    @patch('Tensile.Contractions.TaskPredicate.FromOriginalState')
    @patch('Tensile.Contractions.SizeMapping.FromOriginalState')
    @patch('Tensile.Contractions.InternalArgsSupport.FromOriginalState')
    @patch('Tensile.Contractions.Hardware.HardwarePredicate.FromHardware')
    @patch('Tensile.Contractions.OriginalSolution')
    def test_solution_from_original_state(
        self, mock_orig_sol, mock_hw_pred, mock_ias, mock_size_map,
        mock_task_pred, mock_prob_pred, mock_prob_type
    ):
        """Test Solution.FromOriginalState."""
        mock_prob_type.return_value = MagicMock()
        mock_prob_pred.return_value = MagicMock()
        mock_task_pred.return_value = MagicMock()
        mock_size_map.return_value = MagicMock()
        mock_ias.return_value = MagicMock()
        mock_hw_pred.return_value = MagicMock()
        mock_orig_sol.return_value = MagicMock()

        d = {
            'SolutionNameMin': 'TestSolution',
            'KernelNameMin': 'TestKernel',
            'ProblemType': {},
            'DebugKernel': False,
            'SolutionIndex': 42,
            'ISA': [9, 0, 0],
            'CUCount': 64,
            'KernelLanguage': 'Assembly'
        }

        assembler = MagicMock()
        isaInfoMap = {}

        sol = Solution.FromOriginalState(
            d, False, False, False, assembler, isaInfoMap
        )

        assert sol.name == 'TestSolution'
        assert sol.kernelName == 'TestKernel'
        assert sol.index == 42
        assert sol.debugKernel is False

    def test_solution_read_original_info(self):
        """Test Solution.ReadOriginalInfo."""
        d = {
            'SolutionIndex': 42,
            'MacroTile0': 128,
            'DepthU': 16,
            'ProblemType': {}  # Should be excluded
        }

        result = Solution.ReadOriginalInfo(d)

        assert 'SolutionIndex' in result
        assert 'MacroTile0' in result
        assert 'DepthU' in result
        assert 'ProblemType' not in result

    def test_solution_get_solution_keys(self):
        """Test Solution.getSolutionKeys."""
        with patch('Tensile.Contractions.OriginalSolution') as mock_orig_sol:
            mock_instance = MagicMock()
            mock_instance.keys.return_value = ['key1', 'key2']
            mock_orig_sol.return_value = mock_instance

            sol = Solution()
            sol.originalSolution = mock_instance

            result = sol.getSolutionKeys()

            assert result == ['key1', 'key2']


class TestMinKForGSU:
    """Test MIN_K_FOR_GSU constant."""

    def test_min_k_for_gsu_value(self):
        """Test MIN_K_FOR_GSU has expected value."""
        assert MIN_K_FOR_GSU == 32
