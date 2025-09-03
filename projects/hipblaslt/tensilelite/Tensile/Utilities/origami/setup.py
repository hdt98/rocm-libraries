################################################################################
#
#  Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import glob
import sys
import argparse
import os
from pathlib import Path

# Get ROCM_PATH from environment or use default
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
HIPCC_PATH = os.path.join(ROCM_PATH, "bin", "hipcc")


def parse_args():
    parser = argparse.ArgumentParser(description="Origami Python Bindings Setup Script")
    parser.add_argument("--source", "-s", type=str, default=Path(__file__).parent.parent.parent.resolve() / "Source", help="Path to TensileLite source directory.")
    args, unknown = parser.parse_known_args()
    return args, unknown


class HIPCCBuildExt(build_ext):
    def build_extensions(self):
        if hasattr(self.compiler, 'compiler_so'):
            self.compiler.set_executable("compiler_so", HIPCC_PATH)
        if hasattr(self.compiler, 'compiler_cxx'):
            self.compiler.set_executable("compiler_cxx", HIPCC_PATH)
        # self.compiler.set_executable("linker_so", HIPCC_PATH) # optional
        super().build_extensions()


if __name__ == "__main__":
    args, unknown = parse_args()
    sys.argv = [sys.argv[0]] + unknown  # Preserve unrecognized arguments for setuptools

    source_dir = Path(args.source)
    cpp_path = source_dir / "lib" / "source" / "analytical" / "*.cpp"
    cpp_files = sorted(glob.glob(str(cpp_path)))

    cpp_files = ["origami_module.cpp"] + cpp_files

    ext_modules = [
        Extension(
            "origami",
            cpp_files,
            include_dirs=[
                pybind11.get_include(),
                str(source_dir / "lib" / "include"),
                os.path.join(ROCM_PATH, "include"),
            ],
            language="c++",
            extra_compile_args=["-D__HIP_PLATFORM_AMD__", "-fPIC", "-std=c++17", "-O3", "-Wall"],
            extra_link_args=[f"-L{os.path.join(ROCM_PATH, 'lib')}"],
        ),
    ]

    setup(
        name="origami",
        ext_modules=ext_modules,
        cmdclass={"build_ext": HIPCCBuildExt},
        setup_requires=["pybind11>=2.6.0"],
        install_requires=["pybind11>=2.6.0"],
    )
