#!/usr/bin/env python3
"""
Migrate convolution files to layout-specific subdirectories using git mv.
Preserves git history and organizes files according to the layout pruning plan.
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict
import sys

def extract_file_info(filepath):
    """
    Extract operation, layout, and current subdirectory from filepath.

    Example:
    .../grouped_conv2d_fwd/xdl/comp/device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
    -> operation: grouped_conv2d_fwd
    -> layout: ngchw
    -> subdir_path: xdl/comp
    -> filename: device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_comp_instance.cpp
    """
    parts = Path(filepath).parts

    # Find operation directory
    op_dir = None
    op_idx = -1
    for i, part in enumerate(parts):
        if 'conv' in part and part.startswith(('conv', 'grouped_conv')):
            op_dir = part
            op_idx = i
            break

    if not op_dir:
        return None

    # Get components after operation directory
    remaining_parts = parts[op_idx+1:]
    if not remaining_parts:
        return None

    filename = remaining_parts[-1]
    subdir_parts = remaining_parts[:-1]

    # Extract layout from filename
    filename_lower = filename.lower()
    layout_patterns = ['gndhwc', 'ndhwgc', 'ngcdhw', 'ndhwc',  # 3D
                       'gnhwc', 'nhwgc', 'ngchw', 'nhwc',       # 2D
                       'gnwc', 'nwgc', 'nwc']                    # 1D

    layout = None
    for pattern in layout_patterns:
        if f'_{pattern}_' in filename_lower:
            layout = pattern
            break

    if not layout:
        return None

    return {
        'operation': op_dir,
        'layout': layout,
        'subdir_path': '/'.join(subdir_parts) if subdir_parts else '',
        'filename': filename,
        'current_path': filepath
    }


def plan_migration(mapping_data, base_dir):
    """
    Plan file migrations based on layout mapping.
    Returns list of (source, dest) tuples.
    """
    migrations = []

    for library, files in mapping_data['by_library'].items():
        # Skip special libraries that don't get layout split
        if library in ['device_conv_old_operations', 'device_convnd_generic_operations',
                       'device_quantization_operations']:
            continue

        for filepath in files:
            info = extract_file_info(filepath)
            if not info:
                continue

            # Build target path
            # grouped_conv2d_fwd/xdl/comp/file.cpp -> grouped_conv2d_fwd/ngchw/xdl/comp/file.cpp
            target_parts = [info['operation'], info['layout']]
            if info['subdir_path']:
                target_parts.append(info['subdir_path'])
            target_parts.append(info['filename'])

            target_path = base_dir / '/'.join(target_parts)
            source_path = Path(filepath)

            # Only migrate if not already in layout subdir
            if info['layout'] not in source_path.parts:
                migrations.append((source_path, target_path, info))

    return migrations


def create_directories(migrations):
    """Create target directories for migrations."""
    dirs_created = set()

    for source, target, info in migrations:
        target_dir = target.parent
        if target_dir not in dirs_created:
            target_dir.mkdir(parents=True, exist_ok=True)
            dirs_created.add(target_dir)
            print(f"📁 Created: {target_dir}")

    print(f"\nCreated {len(dirs_created)} directories")
    return dirs_created


def execute_migration(migrations, dry_run=True):
    """Execute git mv for each migration."""
    success = 0
    failed = []
    skipped = 0

    for source, target, info in migrations:
        # Check if source exists
        if not source.exists():
            print(f"⚠️  Source not found: {source}")
            skipped += 1
            continue

        # Check if already migrated
        if target.exists():
            print(f"⏭️  Already exists: {target}")
            skipped += 1
            continue

        if dry_run:
            print(f"🔍 Would move: {source.name}")
            print(f"   From: {source.parent}")
            print(f"   To:   {target.parent}")
            success += 1
        else:
            try:
                # Execute git mv
                cmd = ['git', 'mv', str(source), str(target)]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ Moved: {source.name} -> {info['layout']}/{info['subdir_path']}")
                success += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {source}")
                print(f"   Error: {e.stderr}")
                failed.append((source, target, e.stderr))

    return success, skipped, failed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Migrate convolution files by layout')
    parser.add_argument('--execute', action='store_true',
                        help='Actually perform the migration (default is dry-run)')
    parser.add_argument('--operation', type=str,
                        help='Migrate only specific operation (e.g., grouped_conv2d_fwd)')
    parser.add_argument('--layout', type=str,
                        help='Migrate only specific layout (e.g., ngchw)')
    args = parser.parse_args()

    # Load mapping
    layout_dir = Path(__file__).parent
    mapping_file = layout_dir / 'layout_mapping.json'

    print("Loading file mapping...")
    with open(mapping_file, 'r') as f:
        data = json.load(f)

    base_dir = Path('/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu')

    # Plan migrations
    print("Planning migrations...")
    all_migrations = plan_migration(data, base_dir)

    # Filter if requested
    migrations = all_migrations
    if args.operation:
        migrations = [(s, t, i) for s, t, i in migrations if i['operation'] == args.operation]
        print(f"Filtered to operation: {args.operation}")

    if args.layout:
        migrations = [(s, t, i) for s, t, i in migrations if i['layout'] == args.layout]
        print(f"Filtered to layout: {args.layout}")

    print(f"\n{'='*60}")
    print(f"Migration Plan: {len(migrations)} files")
    print(f"{'='*60}\n")

    if not migrations:
        print("No files to migrate!")
        return

    # Group by operation and layout for summary
    by_op_layout = defaultdict(lambda: defaultdict(int))
    for source, target, info in migrations:
        by_op_layout[info['operation']][info['layout']] += 1

    print("Files to migrate by operation:")
    for op in sorted(by_op_layout.keys()):
        total = sum(by_op_layout[op].values())
        layouts = ', '.join(f"{l}:{c}" for l, c in sorted(by_op_layout[op].items()))
        print(f"  {op}: {total} files ({layouts})")

    print(f"\n{'='*60}\n")

    # Create directories
    if not args.execute:
        print("DRY RUN MODE - No actual changes will be made")
        print("Add --execute flag to perform the migration\n")

    dirs_created = create_directories(migrations)

    print(f"\n{'='*60}\n")

    # Execute migration
    success, skipped, failed = execute_migration(migrations, dry_run=not args.execute)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✅ {'Would migrate' if not args.execute else 'Migrated'}: {success} files")
    print(f"⏭️  Skipped: {skipped} files")
    print(f"❌ Failed: {len(failed)} files")
    print(f"{'='*60}\n")

    if failed:
        print("Failed migrations:")
        for source, target, error in failed[:10]:
            print(f"  {source} -> {target}")
            print(f"    {error}")

    if not args.execute:
        print("\n💡 This was a DRY RUN. Use --execute to perform the actual migration.")
        print("💡 You can test with specific operations: --operation grouped_conv2d_fwd")
        print("💡 Or specific layouts: --layout ngchw")


if __name__ == '__main__':
    main()
