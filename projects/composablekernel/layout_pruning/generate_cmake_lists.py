#!/usr/bin/env python3
"""
Generate CMakeLists.txt files for layout-specific subdirectories.
Files stay in their current locations (xdl/, wmma/, dl/) but are organized via CMakeLists.
"""

import json
from pathlib import Path
from collections import defaultdict

def extract_operation_and_layout(filepath):
    """
    Extract operation type and layout from filepath.
    Returns: (operation_dir, layout, relative_path)

    Example:
    grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance.cpp
    -> ('grouped_conv2d_fwd', 'NGCHW', 'xdl/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instance.cpp')
    """
    parts = Path(filepath).parts

    # Find the operation directory (contains 'conv')
    op_dir = None
    op_idx = -1
    for i, part in enumerate(parts):
        if 'conv' in part and part.startswith(('conv', 'grouped_conv')):
            op_dir = part
            op_idx = i
            break

    if not op_dir:
        return None, None, None

    # Get relative path from operation directory
    rel_path = '/'.join(parts[op_idx+1:])

    # Extract layout from filename
    filename = parts[-1].lower()

    layout_patterns_3d = ['gndhwc', 'ndhwgc', 'ngcdhw', 'ndhwc']
    layout_patterns_2d = ['gnhwc', 'nhwgc', 'ngchw', 'nhwc']
    layout_patterns_1d = ['gnwc', 'nwgc', 'nwc']

    layout = None
    for pattern in layout_patterns_3d + layout_patterns_2d + layout_patterns_1d:
        if f'_{pattern}_' in filename:
            layout = pattern.upper()
            break

    return op_dir, layout, rel_path


def generate_cmake_for_layout(operation_dir, layout, files, output_path):
    """Generate a CMakeLists.txt for a specific layout."""

    # Group files by subdirectory (xdl, wmma, dl, etc.)
    by_subdir = defaultdict(list)
    for filepath in files:
        _, _, rel_path = extract_operation_and_layout(filepath)
        if rel_path:
            by_subdir[Path(rel_path).parts[0] if '/' in rel_path else '.'].append(rel_path)

    # Create variable name
    var_name = f"{operation_dir.upper().replace('-', '_')}_{layout}"
    lib_name = f"device_{operation_dir}_{layout.lower()}_instance"

    content = []
    content.append("# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.")
    content.append("# SPDX-License-Identifier: MIT")
    content.append("")
    content.append(f"# {layout} layout files for {operation_dir}")
    content.append(f"set({var_name}")

    # Sort files by subdirectory
    for subdir in sorted(by_subdir.keys()):
        if subdir != '.':
            content.append(f"   # {subdir} instances")
        for file in sorted(by_subdir[subdir]):
            content.append(f"   ../{file}")

    content.append(")")
    content.append("")

    # Check for template files (.in) that need sharding
    has_templates = any('.in' in str(f) for f in files)

    if has_templates:
        content.append("# Add generated files for sharded instantiations")
        content.append("include(ShardInstantiation)")
        content.append("")
        content.append("set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)")
        content.append("# TODO: Add generate_sharded_instantiations() calls for .in files")
        content.append("")

    content.append(f"add_instance_library({lib_name} ${{{var_name}}})")
    content.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(content))

    return len(files)


def main():
    # Load categorization
    layout_dir = Path(__file__).parent
    mapping_file = layout_dir / 'layout_mapping.json'

    with open(mapping_file, 'r') as f:
        data = json.load(f)

    base_gpu_dir = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu')

    # Organize files by operation and layout
    by_op_layout = defaultdict(lambda: defaultdict(list))

    for library, files in data['by_library'].items():
        if library in ['device_conv_old_operations', 'device_convnd_generic_operations',
                       'device_quantization_operations']:
            # These don't get split by layout
            continue

        for filepath in files:
            op_dir, layout, rel_path = extract_operation_and_layout(filepath)
            if op_dir and layout:
                by_op_layout[op_dir][layout].append(filepath)

    # Generate CMakeLists.txt for each operation/layout combination
    stats = []

    for op_dir in sorted(by_op_layout.keys()):
        for layout in sorted(by_op_layout[op_dir].keys()):
            files = by_op_layout[op_dir][layout]

            # Output path
            output_path = base_gpu_dir / op_dir / layout.lower() / 'CMakeLists.txt'

            # Generate
            count = generate_cmake_for_layout(op_dir, layout, files, output_path)
            stats.append((op_dir, layout, count, output_path))

            print(f"✅ Generated {output_path.relative_to(base_gpu_dir)} ({count} files)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total CMakeLists.txt files generated: {len(stats)}")
    print(f"Total source files organized: {sum(s[2] for s in stats)}")
    print("="*60)

    # Group by operation
    print("\nBy Operation:")
    op_totals = defaultdict(int)
    for op_dir, layout, count, _ in stats:
        op_totals[op_dir] += count

    for op in sorted(op_totals.keys()):
        layouts = [s[1] for s in stats if s[0] == op]
        print(f"  {op}: {op_totals[op]} files across {len(layouts)} layouts ({', '.join(layouts)})")


if __name__ == '__main__':
    main()
