#!/usr/bin/env python3
"""
Generate CMakeLists.txt files for layout-specific subdirectories.
Scans the actual filesystem to find migrated files.
"""

from pathlib import Path
from collections import defaultdict

def scan_layout_directories(base_dir):
    """
    Scan the filesystem for layout directories and their contents.
    Returns: dict[operation][layout] = [list of cpp files]
    """
    result = defaultdict(lambda: defaultdict(list))

    # Layout names to look for
    layouts = ['gnhwc', 'nhwgc', 'ngchw', 'gndhwc', 'ndhwgc', 'ngcdhw', 'gnwc', 'nwgc']

    # Find all layout directories
    for layout in layouts:
        # Find all directories with this layout name
        layout_dirs = list(base_dir.rglob(f'*/{layout}'))

        for layout_dir in layout_dirs:
            if not layout_dir.is_dir():
                continue

            # Get operation name (parent directory)
            operation = layout_dir.parent.name

            # Find all .cpp files in this layout directory
            cpp_files = list(layout_dir.rglob('*.cpp'))

            if cpp_files:
                # Store relative paths from layout directory
                for cpp_file in cpp_files:
                    rel_path = cpp_file.relative_to(layout_dir)
                    result[operation][layout].append(str(rel_path))

    return result


def generate_layout_cmake(operation, layout, files):
    """Generate CMakeLists.txt for a single layout directory."""

    # Group files by subdirectory
    by_subdir = defaultdict(list)
    for filepath in files:
        path = Path(filepath)
        subdir = path.parts[0] if len(path.parts) > 1 else '.'
        by_subdir[subdir].append(filepath)

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
        if subdir != '.' and len(by_subdir) > 1:
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
    content.append(f"# {operation} - layout-specific instances")
    content.append("")

    for layout in sorted(layouts):
        content.append(f"add_subdirectory({layout.lower()})")

    content.append("")

    return '\n'.join(content)


def main():
    base_dir = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu')

    print("Scanning for layout directories...")
    op_layouts = scan_layout_directories(base_dir)

    if not op_layouts:
        print("❌ No layout directories found!")
        return

    print(f"Found {len(op_layouts)} operations with layout splits")
    print()

    # Generate CMakeLists.txt files
    layout_files_generated = 0
    operation_files_generated = 0

    for operation in sorted(op_layouts.keys()):
        layouts = op_layouts[operation]

        print(f"📁 {operation}:")

        # Generate CMakeLists.txt for each layout
        for layout in sorted(layouts.keys()):
            files = layouts[layout]

            cmake_content = generate_layout_cmake(operation, layout, files)
            output_path = base_dir / operation / layout.lower() / 'CMakeLists.txt'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(cmake_content)

            layout_files_generated += 1
            print(f"  ✅ {layout.lower()}/CMakeLists.txt ({len(files)} files)")

        # Generate top-level CMakeLists.txt for operation
        op_cmake = generate_operation_cmake(operation, layouts.keys())
        op_cmake_path = base_dir / operation / 'CMakeLists.txt'

        with open(op_cmake_path, 'w') as f:
            f.write(op_cmake)

        operation_files_generated += 1
        print(f"  ✅ CMakeLists.txt (includes {len(layouts)} layouts)")
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Generated {layout_files_generated} layout CMakeLists.txt files")
    print(f"Generated {operation_files_generated} operation CMakeLists.txt files")
    print()

    # Detailed breakdown
    print("Breakdown by Operation:")
    total_files = 0
    for op in sorted(op_layouts.keys()):
        layouts = op_layouts[op]
        file_count = sum(len(files) for files in layouts.values())
        total_files += file_count
        print(f"  {op}:")
        for layout in sorted(layouts.keys()):
            print(f"    {layout}: {len(layouts[layout])} files")

    print()
    print(f"Total source files: {total_files}")
    print("="*60)


if __name__ == '__main__':
    main()
