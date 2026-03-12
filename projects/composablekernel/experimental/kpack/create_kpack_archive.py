#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pack .hsaco files into a .kpack archive.

Creates a kpack archive with KPAK header, concatenated raw .hsaco blobs
(NoOp compression), and a MessagePack TOC at the end.

Usage:
    python3 create_kpack_archive.py \
        --output build/kernels.kpack \
        --binary-name vector_add_kernel \
        --hsaco build/vector_add_gfx90a.hsaco:gfx90a \
        --hsaco build/vector_add_gfx942.hsaco:gfx942 \
        --hsaco build/vector_add_gfx1101.hsaco:gfx1101
"""

import argparse
import struct
import sys

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


def parse_hsaco_arg(arg: str) -> tuple:
    """Parse 'path:arch' argument into (path, arch)."""
    parts = arg.rsplit(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --hsaco format: '{arg}'. Expected 'path:arch'")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Create a kpack archive from .hsaco files"
    )
    parser.add_argument("--output", required=True, help="Output .kpack file path")
    parser.add_argument(
        "--binary-name",
        required=True,
        help="Binary name key in the TOC (e.g., 'vector_add_kernel')",
    )
    parser.add_argument(
        "--hsaco",
        action="append",
        required=True,
        help="Path to .hsaco file with arch suffix, e.g., 'build/out.hsaco:gfx90a'",
    )
    args = parser.parse_args()

    # Parse hsaco arguments
    hsaco_entries = []
    for h in args.hsaco:
        path, arch = parse_hsaco_arg(h)
        hsaco_entries.append((path, arch))

    # Read all .hsaco files
    blobs = []
    arches = []
    for path, arch in hsaco_entries:
        with open(path, "rb") as f:
            data = f.read()
        blobs.append(data)
        arches.append(arch)
        print(f"  Read {path} ({len(data)} bytes) for {arch}")

    # Build the archive
    with open(args.output, "wb") as out:
        # Write placeholder header (will patch toc_offset later)
        out.write(KPACK_MAGIC)
        out.write(struct.pack("<I", KPACK_VERSION))
        out.write(struct.pack("<Q", 0))  # toc_offset placeholder

        # Write blobs sequentially, recording offsets
        blob_infos = []
        for data in blobs:
            offset = out.tell()
            out.write(data)
            blob_infos.append({"offset": offset, "size": len(data)})

        # Record TOC offset
        toc_offset = out.tell()

        # Build TOC
        toc_entries = {}
        for i, arch in enumerate(arches):
            toc_entries[arch] = {
                "ordinal": i,
                "original_size": len(blobs[i]),
                "type": "hsaco",
            }

        toc = {
            "compression_scheme": "none",
            "gfx_arches": arches,
            "blobs": blob_infos,
            "toc": {
                args.binary_name: toc_entries,
            },
        }

        # Write MessagePack TOC
        packed_toc = msgpack.packb(toc, use_bin_type=True)
        out.write(packed_toc)

        # Patch header with actual toc_offset
        out.seek(8)
        out.write(struct.pack("<Q", toc_offset))

    total_size = sum(len(b) for b in blobs)
    print(f"\nCreated {args.output}")
    print(f"  Architectures: {', '.join(arches)}")
    print(f"  Binary: {args.binary_name}")
    print(f"  Total kernel data: {total_size} bytes")
    print(f"  TOC offset: {toc_offset}")


if __name__ == "__main__":
    main()
