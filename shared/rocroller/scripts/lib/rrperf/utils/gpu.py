# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
import shutil
import subprocess
from pathlib import Path
from time import sleep


@functools.cache
def rocm_gfx() -> str:
    """Return GPU architecture (gfxXXXX) for local GPU device."""
    output = ""
    try:
        output = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return ""

    for line in output.splitlines():
        if line.startswith("  Name:"):
            _, arch, *_ = list(map(lambda x: x.strip(), line.split()))
            if arch.startswith("gfx"):
                return arch

    return ""


def pin_clocks(rocm_smi_path: Path):
    print("Attempting to pin clocks...")
    rocm_smi_found = shutil.which(rocm_smi_path) is not None
    if rocm_smi_found:
        print("{} found, pinning clocks...".format(rocm_smi_path))
        pinresult = subprocess.run(
            [
                rocm_smi_path,
                "-d",
                "0",
                "--setfan",
                "255",
                "--setsclk",
                "7",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(
            "Pinning clocks finished...\n{}\n{}".format(
                pinresult.stdout.decode("ascii"),
                pinresult.stderr.decode("ascii"),
            )
        )
        sleep(1)
        checkresult = subprocess.run(
            [rocm_smi_path, "-d", "0", "-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Clocks status:\n{}".format(checkresult.stdout.decode("ascii")))
        print("Setting up clock restore...")
        setupRestoreClocks(rocm_smi_path)
        print("Pinning clocks finished.")
    else:
        print("{} not found, unable to pin clocks.".format(rocm_smi_path))


def setupRestoreClocks(rocm_smi_path: Path):
    import atexit

    def restoreClocks():
        print("Resetting clocks...")
        subprocess.call([rocm_smi_path, "-d", "0", "--resetclocks"])
        print("Resetting fans...")
        subprocess.call([rocm_smi_path, "-d", "0", "--resetfans"])

    atexit.register(restoreClocks)
