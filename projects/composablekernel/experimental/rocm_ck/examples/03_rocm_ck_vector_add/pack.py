#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack per-architecture .hsaco files into a self-describing kpack archive.

Reads .spec.json files (emitted by build-time spec extractors) to discover
variants and their metadata. The archive carries both device code and
structured spec metadata in the msgpack TOC.

Archive format:

    [0x00]  "KPAK"           4 bytes   Magic
    [0x04]  version          4 bytes   uint32 LE (currently 1)
    [0x08]  toc_offset       8 bytes   uint64 LE
    [0x10]  blob_0           variable  Raw .hsaco bytes
            blob_1           variable  ...
    [toc_offset]  TOC        variable  MessagePack table of contents
"""

import json
import struct
import sys
from pathlib import Path

try:
    import msgpack
except ImportError:
    print(
        "Error: msgpack package required. Install with: pip install msgpack",
        file=sys.stderr,
    )
    sys.exit(1)


KPACK_MAGIC = b"KPAK"
KPACK_VERSION = 1
HEADER_SIZE = 16  # 4 (magic) + 4 (version) + 8 (toc_offset)


def main() -> None:
    # Accept an optional build directory argument (used by CMake).
    build_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build"
    )
    output_path = build_dir / "kernels.kpack"

    # --- Discover variants from .spec.json files ---
    spec_files = sorted(build_dir.glob("*.spec.json"))
    if not spec_files:
        print(
            f"Error: no .spec.json files found in {build_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    variants: list[dict] = []
    for spec_file in spec_files:
        with spec_file.open() as f:
            variant = json.load(f)
        variants.append(variant)
        print(f"  Loaded spec: {variant['name']} ({variant['spec_type']})")

    # --- Read .hsaco blobs for each variant x architecture ---
    blobs: list[bytes] = []
    found_arches: set[str] = set()
    variant_map: dict[str, dict[str, int]] = {}

    for variant in variants:
        name = variant["name"]
        variant_entries: dict[str, int] = {}

        # Scan for .hsaco files matching this variant
        for hsaco_path in sorted(build_dir.glob(f"{name}_*.hsaco")):
            # Extract arch from filename: {name}_{arch}.hsaco
            arch = hsaco_path.stem[len(name) + 1 :]
            if not arch or "_" in arch:
                # Arch names (gfx90a, gfx1100, etc.) never contain underscores.
                # An underscore means the glob matched a different variant.
                continue
            blob = hsaco_path.read_bytes()
            ordinal = len(blobs)
            blobs.append(blob)
            found_arches.add(arch)
            variant_entries[arch] = ordinal
            print(f"  Read {hsaco_path.name} ({len(blob)} bytes)")

        if variant_entries:
            variant_map[name] = variant_entries

    if not blobs:
        print("Error: no .hsaco files found in build/", file=sys.stderr)
        sys.exit(1)

    sorted_arches = sorted(found_arches)

    # --- Write the archive ---
    with output_path.open("wb") as out:
        # Header (toc_offset patched after writing blobs)
        out.write(KPACK_MAGIC)
        out.write(struct.pack("<I", KPACK_VERSION))
        out.write(struct.pack("<Q", 0))  # toc_offset placeholder

        # Concatenate blobs, recording offsets
        blob_infos: list[dict] = []
        for blob in blobs:
            offset = out.tell()
            out.write(blob)
            blob_infos.append({"offset": offset, "size": len(blob)})

        # Build and write the MessagePack TOC
        toc_offset = out.tell()

        # Structured variant specs (the self-describing metadata)
        variant_specs = {}
        for variant in variants:
            if variant["name"] in variant_map:
                variant_specs[variant["name"]] = {
                    "spec_type": variant["spec_type"],
                    "targets": variant.get("targets", []),
                    "spec": variant["spec"],
                }

        toc = {
            "compression_scheme": "none",
            "gfx_arches": sorted_arches,
            "blobs": blob_infos,
            "variant_specs": variant_specs,
            "toc": {
                variant_name: {
                    arch: {
                        "ordinal": ordinal,
                        "original_size": len(blobs[ordinal]),
                        "type": "hsaco",
                    }
                    for arch, ordinal in arch_entries.items()
                }
                for variant_name, arch_entries in variant_map.items()
            },
        }
        out.write(msgpack.packb(toc, use_bin_type=True))

        # Patch header with actual toc_offset
        out.seek(8)
        out.write(struct.pack("<Q", toc_offset))

    total_size = sum(len(b) for b in blobs)
    print(f"\nCreated {output_path}")
    print(f"  Architectures: {', '.join(sorted_arches)}")
    print(f"  Variants: {', '.join(variant_map.keys())}")
    print(f"  Total kernel data: {total_size} bytes")


if __name__ == "__main__":
    main()
