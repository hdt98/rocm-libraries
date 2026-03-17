#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack per-architecture .hsaco files into a kpack archive.

Hard-coded for the vector_add hello world example. The archive format is:

    [0x00]  "KPAK"           4 bytes   Magic
    [0x04]  version          4 bytes   uint32 LE (currently 1)
    [0x08]  toc_offset       8 bytes   uint64 LE
    [0x10]  blob_0           variable  Raw .hsaco bytes
            blob_1           variable  ...
    [toc_offset]  TOC        variable  MessagePack table of contents
"""

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

BINARY_NAME = "vector_add_kernel"
ARCHITECTURES = ["gfx90a", "gfx942", "gfx950"]


def main() -> None:
    # Accept an optional build directory argument (used by CMake).
    # Defaults to ./build relative to this script.
    build_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build"
    )
    output_path = build_dir / "kernels.kpack"

    # Read .hsaco blobs for each architecture
    blobs: list[bytes] = []
    found_arches: list[str] = []
    for arch in ARCHITECTURES:
        hsaco_path = build_dir / f"vector_add_{arch}.hsaco"
        if not hsaco_path.exists():
            print(f"Skipping {arch}: {hsaco_path} not found")
            continue
        blob = hsaco_path.read_bytes()
        blobs.append(blob)
        found_arches.append(arch)
        print(f"  Read {hsaco_path.name} ({len(blob)} bytes) for {arch}")

    if not blobs:
        print("Error: no .hsaco files found in build/", file=sys.stderr)
        sys.exit(1)

    # Write the archive
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
        toc = {
            "compression_scheme": "none",
            "gfx_arches": found_arches,
            "blobs": blob_infos,
            "toc": {
                BINARY_NAME: {
                    arch: {
                        "ordinal": i,
                        "original_size": len(blobs[i]),
                        "type": "hsaco",
                    }
                    for i, arch in enumerate(found_arches)
                },
            },
        }
        out.write(msgpack.packb(toc, use_bin_type=True))

        # Patch header with actual toc_offset
        out.seek(8)
        out.write(struct.pack("<Q", toc_offset))

    total_size = sum(len(b) for b in blobs)
    print(f"\nCreated {output_path}")
    print(f"  Architectures: {', '.join(found_arches)}")
    print(f"  Binary: {BINARY_NAME}")
    print(f"  Total kernel data: {total_size} bytes")


if __name__ == "__main__":
    main()
