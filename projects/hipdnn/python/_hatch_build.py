# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import glob
import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Copy a pre-built nanobind extension into the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        ext_dir = os.environ.get("HIPDNN_EXT_DIR", "")
        if not ext_dir:
            return

        ext_path = Path(ext_dir)
        extensions = glob.glob(str(ext_path / "hipdnn_frontend_python*"))
        if not extensions:
            raise RuntimeError(
                f"No hipdnn_frontend_python extension found in {ext_path}"
            )

        for ext in extensions:
            name = Path(ext).name
            build_data["force_include"][ext] = f"hipdnn_frontend/{name}"
