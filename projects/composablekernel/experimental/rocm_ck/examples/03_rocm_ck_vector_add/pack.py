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

import argparse
import json
import struct
import subprocess
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


def _zstd_compress(data: bytes) -> bytes:
    """Compress a single buffer using the zstd CLI."""
    result = subprocess.run(
        ["zstd", "-c", "--no-progress"],
        input=data,
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            f"Error: zstd compression failed: {result.stderr.decode()}", file=sys.stderr
        )
        sys.exit(1)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack .hsaco files into a kpack archive"
    )
    parser.add_argument("build_dir", nargs="?", default=None, help="Build directory")
    parser.add_argument(
        "--zstd", action="store_true", help="Use zstd-per-kernel compression"
    )
    args = parser.parse_args()

    build_dir = (
        Path(args.build_dir) if args.build_dir else Path(__file__).parent / "build"
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
    use_zstd = args.zstd
    total_size = sum(len(b) for b in blobs)

    with output_path.open("wb") as out:
        # Header (toc_offset patched after writing blobs)
        out.write(KPACK_MAGIC)
        out.write(struct.pack("<I", KPACK_VERSION))
        out.write(struct.pack("<Q", 0))  # toc_offset placeholder

        # Structured variant specs (the self-describing metadata)
        variant_specs = {}
        for variant in variants:
            if variant["name"] in variant_map:
                variant_specs[variant["name"]] = {
                    "spec_type": variant["spec_type"],
                    "targets": variant.get("targets", []),
                    "spec": variant["spec"],
                }

        # Kernel TOC entries (shared between both schemes)
        kernel_toc = {
            variant_name: {
                arch: {
                    "ordinal": ordinal,
                    "original_size": len(blobs[ordinal]),
                    "type": "hsaco",
                }
                for arch, ordinal in arch_entries.items()
            }
            for variant_name, arch_entries in variant_map.items()
        }

        if use_zstd:
            # Write zstd blob: [num_kernels:u32] [frame_size:u32 frame_data]...
            zstd_offset = out.tell()
            out.write(struct.pack("<I", len(blobs)))

            compressed_total = 0
            for i, blob in enumerate(blobs):
                compressed = _zstd_compress(blob)
                out.write(struct.pack("<I", len(compressed)))
                out.write(compressed)
                compressed_total += len(compressed)
                print(
                    f"  Compressed blob {i}: {len(blob)} -> {len(compressed)} "
                    f"({len(compressed) / len(blob) * 100:.1f}%)"
                )

            zstd_size = out.tell() - zstd_offset

            toc_offset = out.tell()
            toc = {
                "compression_scheme": "zstd-per-kernel",
                "gfx_arches": sorted_arches,
                "zstd_offset": zstd_offset,
                "zstd_size": zstd_size,
                "variant_specs": variant_specs,
                "toc": kernel_toc,
            }
        else:
            # Write raw blobs
            blob_infos: list[dict] = []
            for blob in blobs:
                offset = out.tell()
                out.write(blob)
                blob_infos.append({"offset": offset, "size": len(blob)})

            toc_offset = out.tell()
            toc = {
                "compression_scheme": "none",
                "gfx_arches": sorted_arches,
                "blobs": blob_infos,
                "variant_specs": variant_specs,
                "toc": kernel_toc,
            }

        out.write(msgpack.packb(toc, use_bin_type=True))

        # Patch header with actual toc_offset
        out.seek(8)
        out.write(struct.pack("<Q", toc_offset))

    print(f"\nCreated {output_path}")
    print(f"  Architectures: {', '.join(sorted_arches)}")
    print(f"  Variants: {', '.join(variant_map.keys())}")
    print(f"  Total kernel data: {total_size} bytes")
    if use_zstd:
        file_size = output_path.stat().st_size
        print(
            f"  Compressed size: {file_size} bytes ({file_size / total_size * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
