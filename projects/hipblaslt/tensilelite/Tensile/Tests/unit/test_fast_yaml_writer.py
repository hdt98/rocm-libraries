# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for the fast YAML writer in LibraryIO.py

Verifies that _fast_yaml_write_document produces output that loads
to identical Python objects as yaml.dump (CSafeDumper).
"""

import pytest
import yaml
import os
import tempfile

pytestmark = pytest.mark.unit

from Tensile.LibraryIO import writeYAML


def _round_trip_compare(data, tmp_path):
    """Write data via fast path and yaml.dump, load both, compare."""
    fast_file = str(tmp_path / "fast.yaml")
    slow_file = str(tmp_path / "slow.yaml")

    writeYAML(fast_file, data)  # no kwargs -> fast path
    writeYAML(slow_file, data, explicit_start=True)  # kwargs -> yaml.dump

    with open(fast_file) as f:
        fast_loaded = yaml.safe_load(f)
    with open(slow_file) as f:
        slow_loaded = yaml.safe_load(f)

    assert fast_loaded == slow_loaded, (
        f"Fast and slow YAML differ:\nFast: {fast_loaded}\nSlow: {slow_loaded}"
    )
    return fast_loaded


class TestFastYamlWriter:

    def test_library_state_structure(self, tmp_path):
        """Test with data mimicking MasterSolutionLibrary.state() output."""
        data = {
            'solutions': [
                {'name': 'sol0', 'index': 0, 'ISA': [9, 5, 0], 'enabled': True,
                 'speed': 1.5,
                 'ProblemType': {'DataType': 'Half', 'TransposeA': False,
                                 'TransposeB': True}},
                {'name': 'sol1', 'index': 1, 'ISA': [9, 5, 0], 'enabled': False,
                 'speed': 0.0,
                 'ProblemType': {'DataType': 'Half', 'TransposeA': True,
                                 'TransposeB': False}},
            ],
            'library': {
                'type': 'Matching',
                'properties': [
                    {'type': 'FreeSizeA', 'index': 0},
                    {'type': 'FreeSizeB', 'index': 0},
                ],
                'table': [
                    {'key': [128, 128, 1], 'index': 0, 'speed': 1.5},
                    {'key': [256, 256, 1], 'index': 1, 'speed': 0.8},
                ],
                'distance': 'Euclidean',
            },
            'version': '1.0.0',
        }
        _round_trip_compare(data, tmp_path)

    def test_empty_collections(self, tmp_path):
        data = {'empty_dict': {}, 'empty_list': []}
        _round_trip_compare(data, tmp_path)

    def test_null_and_booleans(self, tmp_path):
        data = {'null_val': None, 'bool_true': True, 'bool_false': False}
        _round_trip_compare(data, tmp_path)

    def test_nested_dicts(self, tmp_path):
        data = {'a': {'b': {'c': {'d': 42}}}}
        _round_trip_compare(data, tmp_path)

    def test_string_quoting(self, tmp_path):
        data = {
            'yaml_bool': 'true',
            'yaml_null': 'null',
            'colon_space': 'hello: world',
            'number_like': '3.14',
            'special_start': '#comment-like',
            'empty_string': '',
        }
        _round_trip_compare(data, tmp_path)

    def test_list_of_scalars(self, tmp_path):
        data = {'ints': [1, 2, 3], 'mixed': [1, 'two', 3.0, True, None]}
        _round_trip_compare(data, tmp_path)

    def test_cache_data_structure(self, tmp_path):
        """Test with data mimicking BenchmarkProblems cache write."""
        data = {
            'CodeObjectFiles': ['kernel_a.co', 'kernel_b.co'],
            'ConstantParams': [{'key': 'val'}],
            'ForkParams': [],
            'ParamGroups': [1, 2, 3],
            'CustomKernels': [],
            'InternalSupportParams': [{'ISP1': True}],
            'CustomKernelWildcard': False,
        }
        _round_trip_compare(data, tmp_path)

    def test_kwargs_use_yaml_dump(self, tmp_path):
        """Verify kwargs callers still produce valid YAML via yaml.dump."""
        data = {'key': 'value'}
        outfile = str(tmp_path / "kwargs.yaml")
        writeYAML(outfile, data, explicit_start=False, explicit_end=False)
        with open(outfile) as f:
            loaded = yaml.safe_load(f)
        assert loaded == data

    def test_deeply_nested_list_of_dicts(self, tmp_path):
        data = {
            'outer': [
                {'inner': [
                    {'a': 1, 'b': 2},
                    {'a': 3, 'b': 4},
                ]},
            ],
        }
        _round_trip_compare(data, tmp_path)

    def test_integer_and_float_values(self, tmp_path):
        data = {'zero': 0, 'negative': -42, 'big': 999999, 'pi': 3.14159}
        _round_trip_compare(data, tmp_path)

    def test_single_solution(self, tmp_path):
        data = {
            'solutions': [
                {'name': 'only', 'index': 0, 'WorkGroup': [16, 16, 1]},
            ],
            'library': {'type': 'FreeSize', 'table': [0, 1]},
        }
        _round_trip_compare(data, tmp_path)
