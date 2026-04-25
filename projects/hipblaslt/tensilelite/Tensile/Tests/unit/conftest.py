################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import shutil
import pytest
from Tensile.Common import IsaVersion


@pytest.fixture(scope="session")
def isa_infrastructure():
    """Initialize ISA capabilities once per test session.

    Probes the compiler for ISA info (~3.8s) and creates an Assembler.
    Only tests that need real kernel-writer-produced idMaps use this fixture;
    mock-based tests are unaffected.
    """
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Component import Assembler

    compiler = shutil.which('amdclang++') or shutil.which('clang++')
    assembler_bin = shutil.which('amdclang') or shutil.which('clang')
    assert compiler, "No C++ compiler found for ISA capability probing"
    assert assembler_bin, "No assembler binary found"

    # Probe both gfx950 and gfx1151 so all CMS test classes can use real idMaps.
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0), IsaVersion(11, 5, 1)], compiler)
    # Note: do NOT call assignGlobalParameters here — it mutates global state
    # and breaks test_validateParameterTypes. The Solution and KernelWriter
    # work without it as long as isaInfoMap is passed explicitly.
    asm = Assembler(assembler_bin, 'V5')

    return None, isaInfoMap, asm
