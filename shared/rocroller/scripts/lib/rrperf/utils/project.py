# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import datetime
import os
from pathlib import Path

from rrperf.utils import git


def get_work_dir(
    rundir: Path | None = None,
    build_dir: Path | None = None,
) -> Path:
    """Return a new work directory path."""

    date = datetime.date.today().strftime("%Y-%m-%d")
    root = "."
    commit = git.get_commit(rundir, build_dir)

    if rundir is not None:
        root = Path(rundir)

    serial = len(list(Path(root).glob(f"{date}-{commit}-*")))
    return root / Path(f"{date}-{commit}-{serial:03d}")


def get_build_dir() -> Path:
    varname = "ROCROLLER_BUILD_DIR"
    if varname in os.environ:
        return Path(os.environ[varname])
    default = git.top() / "shared" / "rocroller" / "build"
    if default.is_dir():
        return default

    raise RuntimeError(f"Build directory not found.  Set {varname} to override.")
