# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Heuristic Plugins for the Engine Selection Framework

# Default heuristics plugin: bundles the well-known Config and StaticOrdering
# policies behind a single shared library. Demonstrates the 1:N (one plugin →
# multiple policies) shape of the heuristic plugin C ABI.
add_subdirectory(default)
