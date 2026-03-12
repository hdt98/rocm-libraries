################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import glob
import pytest

from Tensile import Tensile
from config_helpers import findConfigs

@pytest.mark.parametrize("config", findConfigs())
def test_config(tensile_args, config, tmpdir):
    args = [config, tmpdir.strpath, *tensile_args]
    is_benchmark_timer_config = "benchmark_timer" in os.path.basename(config)
    # Disable timer and hwmonitor for CI correctness-only runs.
    # benchmark_timer tests retain timing to verify it works.
    if not is_benchmark_timer_config:
        _disable_timer_and_hwmonitor(args)
    Tensile.Tensile(args)

    _verify_timer_and_hwmonitor(tmpdir.strpath, expect_disabled=not is_benchmark_timer_config)
