#!/usr/bin/env python3
"""
Generate CMakeLists.txt files for layout-specific subdirectories.
Creates proper build structure for split convolution libraries.
"""

import json
from pathlib import Path
from collections import defaultdict

def get_relative_sources(base_dir, layout_dir, files):
    """
    Get source files relative to the layout directory.
    Files are already in layout subdirectories from migration.
    """
    sources = []
    for filepath in files:
        path = Path(filepath)
        # Find layout dir in path
        parts = list(path.parts)
        try:
            # Find gpu directory index
            gpu_idx = parts.index('gpu')
            # Everything after gpu/operation/layout/ is our relative path
            rel_parts = parts[gpu_idx+3:]  # Skip gpu/operation/layout
            if rel_parts:
                sources.append('/'.join(rel_parts))
        except (ValueError, IndexError):
            pass
    return sorted(sources)


def generate_layout_cmake(operation, layout, files):
    """Generate CMakeLists.txt for a single layout directory."""

    # Get relative paths
    base = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu')
    layout_dir = base / operation / layout.lower()

    # Group files by subdirectory (xdl, wmma, dl, etc.)
    by_subdir = defaultdict(list)
    for filepath in files:
        path = Path(filepath)
        parts = list(path.parts)
        try:
            layout_idx = parts.index(layout.lower())
            # Get path relative to layout directory
            rel_parts = parts[layout_idx+1:]
            if rel_parts:
                subdir = rel_parts[0] if len(rel_parts) > 1 else '.'
                rel_path = '/'.join(rel_parts)
                by_subdir[subdir].append(rel_path)
        except ValueError:
            pass

    if not by_subdir:
        return None

    # Create CMakeLists.txt content
    var_name = f"{operation.upper().replace('-', '_')}_{layout.upper()}"

    content = []
    content.append("# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.")
    content.append("# SPDX-License-Identifier: MIT")
    content.append("")
    content.append(f"# {layout.upper()} layout instances for {operation}")
    content.append(f"set({var_name}")

    # Add files organized by subdirectory
    for subdir in sorted(by_subdir.keys()):
        if subdir != '.':
            content.append(f"   # {subdir}")
        for source in sorted(by_subdir[subdir]):
            content.append(f"   {source}")

    content.append(")")
    content.append("")

    # Create library target
    lib_name = f"device_{operation}_{layout.lower()}_instance"
    content.append(f"add_instance_library({lib_name} ${{{var_name}}})")
    content.append("")

    return '\n'.join(content)


def generate_operation_cmake(operation, layouts):
    """Generate top-level CMakeLists.txt for an operation that includes all layouts."""

    content = []
    content.append("# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.")
    content.append("# SPDX-License-Identifier: MIT")
    content.append("")
    content.append(f"# {operation} - split by layout")
    content.append("")

    for layout in sorted(layouts):
        content.append(f"add_subdirectory({layout.lower()})")

    content.append("")

    return '\n'.join(content)


def main():
    # Load mapping
    layout_dir = Path(__file__).parent
    mapping_file = layout_dir / 'layout_mapping.json'

    with open(mapping_file, 'r') as f:
        data = json.load(f)

    base_dir = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu')

    # Group by operation and layout
    op_layouts = defaultdict(lambda: defaultdict(list))

    for library, files in data['by_library'].items():
        # Skip special libraries
        if library in ['device_conv_old_operations', 'device_convnd_generic_operations',
                       'device_quantization_operations']:
            continue

        # Parse library name: device_conv{dim}_{layout}_operations
        # Extract dimension and layout
        parts = library.split('_')
        if len(parts) >= 4:
            # device_conv2d_nhwgc_operations -> conv2d, nhwgc
            dim_part = parts[1]  # conv2d, conv3d, etc.
            layout = parts[2]    # nhwgc, ngchw, etc.

            # Group files by operation
            for filepath in files:
                path = Path(filepath)
                path_parts = list(path.parts)
                # Find the operation directory
                for i, part in enumerate(path_parts):
                    if 'conv' in part and part.startswith(('conv', 'grouped_conv')):
                        operation = part
                        op_layouts[operation][layout].append(filepath)
                        break

    # Generate CMakeLists.txt files
    generated = []

    for operation in sorted(op_layouts.keys()):
        layouts = op_layouts[operation]

        # Generate CMakeLists.txt for each layout
        for layout in sorted(layouts.keys()):
            files = layouts[layout]

            cmake_content = generate_layout_cmake(operation, layout, files)
            if cmake_content:
                output_path = base_dir / operation / layout.lower() / 'CMakeLists.txt'
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    f.write(cmake_content)

                generated.append((operation, layout, len(files), output_path))
                print(f"✅ {output_path.relative_to(base_dir)} ({len(files)} files)")

        # Generate top-level CMakeLists.txt for operation
        op_cmake = generate_operation_cmake(operation, layouts.keys())
        op_cmake_path = base_dir / operation / 'CMakeLists.txt'

        with open(op_cmake_path, 'w') as f:
            f.write(op_cmake)

        print(f"✅ {op_cmake_path.relative_to(base_dir)} (includes {len(layouts)} layouts)")
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Generated {len(generated)} layout CMakeLists.txt files")
    print(f"Generated {len(op_layouts)} operation CMakeLists.txt files")
    print()

    # Group by operation
    print("By Operation:")
    op_counts = defaultdict(lambda: {'layouts': 0, 'files': 0})
    for op, layout, count, _ in generated:
        op_counts[op]['layouts'] += 1
        op_counts[op]['files'] += count

    for op in sorted(op_counts.keys()):
        info = op_counts[op]
        print(f"  {op}: {info['layouts']} layouts, {info['files']} files")

    print("="*60)


if __name__ == '__main__':
    main()
