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

"""Auto-discovery package for user-defined rocasm mainloop modules.

Drop your rocasm mainloop ``.py`` files into this directory.  Each module
should use the ``@RegisterROCasmMainloop(...)`` decorator from
``Tensile.Components.ROCasmRegistry`` to register its mainloop function.

When this package is imported, all ``.py`` files in this directory (except
``__init__.py``) are automatically imported, triggering decorator registration.

Example module::

    from Tensile.Components.ROCasmRegistry import RegisterROCasmMainloop
    from rocasm.block import Block
    from rocasm.regs import VgprArray, AccArray, SgprArray

    @RegisterROCasmMainloop(
        macro_tile_0=192, macro_tile_1=256, depth_u=64,
        matrix_inst=[16, 16, 32, 1],
        transpose_a=True, transpose_b=False,
    )
    def my_bf16_192x256x64():
        block = Block(...)
        # ... rocasm mainloop code ...
        return block
"""

import importlib
import pkgutil


# Auto-import all modules in this package directory.
# Each module's top-level @RegisterROCasmMainloop decorators will fire,
# populating _ROCASM_REGISTRY in Tensile.Components.ROCasmRegistry.
for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_modname}")
