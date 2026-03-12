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

def _disable_timer_and_hwmonitor(args):
    """Append --global-parameters to disable BenchmarkTimer and HardwareMonitor.

    Inserts the overrides after the existing --global-parameters flag if present,
    otherwise appends it. Modifies args in place.
    """
    try:
        idx = args.index("--global-parameters")
    except ValueError:
        args.append("--global-parameters")
        idx = len(args) - 1
    args.insert(idx + 1, "BenchmarkTimer=0")
    args.insert(idx + 2, "HardwareMonitor=0")


def _parse_ini_value(ini_path, key):
    """Parse a key=value from a Tensile client .ini file. Returns None if not found."""
    with open(ini_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
    return None


def _verify_timer_and_hwmonitor(output_dir, expect_disabled):
    """Verify benchmark-timer and hardware-monitor values in generated .ini files.

    Args:
        output_dir: The Tensile output directory to search for .ini files.
        expect_disabled: If True, assert both parameters are disabled.
            If False, assert benchmark-timer is enabled (hardware-monitor
            is only checked for presence since some architectures like
            gfx950 force-disable it).
    """
    ini_files = glob.glob(os.path.join(output_dir, "**", "*.ini"), recursive=True)
    assert ini_files, "Expected at least one .ini file in the output directory"

    # Values that indicate the feature is disabled (int 0 from CLI override,
    # or bool False from code).
    disabled_values = {"0", "False"}

    for ini_file in ini_files:
        bt_value = _parse_ini_value(ini_file, "benchmark-timer")
        hm_value = _parse_ini_value(ini_file, "hardware-monitor")

        assert bt_value is not None, \
            f"benchmark-timer not found in {ini_file}"
        assert hm_value is not None, \
            f"hardware-monitor not found in {ini_file}"

        if expect_disabled:
            assert bt_value in disabled_values, \
                f"Expected benchmark-timer disabled in {ini_file}, got {bt_value}"
            assert hm_value in disabled_values, \
                f"Expected hardware-monitor disabled in {ini_file}, got {hm_value}"
        else:
            assert bt_value not in disabled_values, \
                f"Expected benchmark-timer enabled in {ini_file}, got {bt_value}"


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
