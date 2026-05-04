################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
from pathlib import Path

import pytest

from Tensile import Tensile
from config_helpers import configMarks, findAvailableArchs

_CONFIG_ROOT = Path(__file__).with_name("ductile") 


def _find_configs():
    root_dir = str(_CONFIG_ROOT)
    tests_root = _CONFIG_ROOT.parents[1]
    available_archs = findAvailableArchs()
    params = []

    for path in sorted(_CONFIG_ROOT.rglob("*.yaml")):
        marks = configMarks(str(path), root_dir, available_archs)
        test_id = os.path.relpath(path, tests_root)
        params.append(pytest.param(str(path), marks=marks, id=test_id))

    return params


def _find_prebuilt_client():
    repo_root = Path(__file__).resolve().parents[3]
    direct = repo_root / "build_tmp" / "tensilelite" / "client" / "tensilelite-client"
    if direct.is_file():
        return str(direct)

    workspace_root = repo_root.parents[5]
    for candidate in workspace_root.glob("**/build_tmp/tensilelite/client/tensilelite-client"):
        if candidate.is_file():
            return str(candidate)

    return None


@pytest.mark.parametrize("config", _find_configs())
def test_ductile_config(tensile_args, config, tmpdir):
    args = list(tensile_args)
    if "--prebuilt-client" not in args:
        prebuilt_client = _find_prebuilt_client()
        if prebuilt_client is None:
            pytest.skip("No tensilelite-client binary available for Ductile smoke test")
        args = ["--prebuilt-client", prebuilt_client, *args]

    Tensile.Tensile([config, tmpdir.strpath, *args])