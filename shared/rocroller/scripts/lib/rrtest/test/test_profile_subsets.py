# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests to verify profile subset relationships.

These tests ensure that profile hierarchies are maintained:
- quick <= full (quick tests are a subset of full)
- codecov-mci <= precheckin-mci (codecov tests are a subset of precheckin-mci)
"""

from pathlib import Path

import pytest
from rrtest import list_tests


@pytest.fixture
def build_dir():
    """Fixture to provide build directory."""
    return Path("build")


def test_quick_subset_of_full(build_dir):
    """Test that quick profile tests are a subset of full."""
    quick = list_tests("quick", build_dir)
    full = list_tests("full", build_dir)

    assert len(quick) > 0, "quick profile should have tests"
    assert len(full) > 0, "full profile should have tests"

    assert quick.issubset(full), (
        f"quick tests should be a subset of full. "
        f"Found {len(quick - full)} tests in quick but not in full"
    )


def test_codecov_subset_of_precheckin_mci(build_dir):
    """Test that codecov-mci profile tests are a subset of precheckin-mci."""
    codecov = list_tests("codecov-mci", build_dir)
    precheckin = list_tests("precheckin-mci", build_dir)

    assert len(codecov) > 0, "codecov-mci profile should have tests"
    assert len(precheckin) > 0, "precheckin-mci profile should have tests"

    assert codecov.issubset(precheckin), (
        f"codecov-mci tests should be a subset of precheckin-mci. "
        f"Found {len(codecov - precheckin)} tests in codecov-mci but not in precheckin-mci"
    )
