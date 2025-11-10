#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Setup script for installing pre-built Origami Python bindings.

This setup.py is designed to work with binaries already built by CMake.
The typical workflow is:
    1. Build with CMake: cd build && cmake .. && make
    2. Install with pip: cd build/python && pip install .
"""

import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

def get_version():
    """Get version from git tags, fallback to dev version."""
    try:
        # Try to get version from git describe
        result = subprocess.run(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent  # Go to repo root
        )
        version = result.stdout.strip()
        
        # Clean up git version for PEP 440 compliance
        # v1.2.3 -> 1.2.3
        # v1.2.3-5-gabcdef -> 1.2.3.dev5+gabcdef
        # therock-7-g309 -> 0.0.0+therock.7.g309
        # abcdef (no tags) -> 0.0.0+abcdef
        
        if version.startswith('v'):
            version = version[1:]
        
        # Check if version starts with a valid semantic version pattern (digit.digit)
        # This distinguishes real version tags from branch/tag names like "therock"
        has_valid_version = version and version[0].isdigit() and '.' in version.split('-')[0]
        
        if '-' in version and has_valid_version:
            # Handle commits after a valid version tag: 1.2.3-5-gabcdef
            parts = version.split('-')
            if len(parts) >= 3:
                # parts = ['1.2.3', '5', 'gabcdef'] or ['1.2.3', '5', 'gabcdef', 'dirty']
                base_version = parts[0]
                commit_count = parts[1]
                commit_hash = parts[2]
                dirty = '-dirty' in version
                version = f"{base_version}.dev{commit_count}+{commit_hash}"
                if dirty:
                    version += ".dirty"
        elif not has_valid_version:
            # No valid version tag, treat everything as local version identifier
            # Replace - with . for PEP 440 compliance
            version = f"0.0.0+{version.replace('-', '.')}"
        
        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not in a git repo
        return "0.0.0+dev"

def get_long_description():
    """Read long description from README if available."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return "Origami: Performance modeling and optimization library for AMD GPUs"

setup(
    name="origami",
    version=get_version(),
    author="AMD",
    description="Python bindings for Origami GPU performance modeling library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/ROCm/rocm-libraries",
    
    # Force binary distribution (platform-specific wheel)
    distclass=BinaryDistribution,
    
    # Package discovery - explicitly list packages
    packages=["origami"],
    
    # Include all .so files and other package data
    package_data={
        "origami": ["*.so", "*.pyd", "*.dylib"],  # Include all binary extensions
    },
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # No build-time dependencies (binaries are pre-built)
    # Runtime dependencies are optional
    install_requires=[],
    
    # Optional dependencies
    extras_require={
        "torch": ["torch>=2.0.0"],  # For PyTorch ATen ops
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    
    # This is important: we're installing pre-built binaries
    zip_safe=False,
)
