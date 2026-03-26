#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

DRIVER = Path(__file__).parent.parent / "build" / "bin" / "MIOpenDriver"

commands = [
    [str(DRIVER), "convfp16", "-n", "32", "-c", "128", "-H", "200", "-W", "200", "-k", "128", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "1", "-t", "0", "--verify", "0"],
    [str(DRIVER), "convfp16", "-n", "32", "-c", "128", "-H", "200", "-W", "200", "-k", "128", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "2", "-t", "0", "--verify", "0"],
    [str(DRIVER), "convfp16", "-n", "32", "-c", "512", "-H", "50",  "-W", "50",  "-k", "512", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "1", "-t", "0", "--verify", "0"],
    [str(DRIVER), "convfp16", "-n", "32", "-c", "512", "-H", "50",  "-W", "50",  "-k", "512", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "2", "-t", "0", "--verify", "0"],
    [str(DRIVER), "convfp16", "-n", "32", "-c", "256", "-H", "100", "-W", "100", "-k", "256", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "1", "-t", "0", "--verify", "0"],
    [str(DRIVER), "convfp16", "-n", "32", "-c", "256", "-H", "100", "-W", "100", "-k", "256", "-y", "3", "-x", "3", "-p", "1", "-q", "1", "-u", "1", "-v", "1", "-l", "1", "-j", "1", "--in_layout", "NHWC", "--fil_layout", "NHWC", "--out_layout", "NHWC", "-m", "conv", "-g", "32", "-F", "2", "-t", "0", "--verify", "0"],
]

for cmd in commands:
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
