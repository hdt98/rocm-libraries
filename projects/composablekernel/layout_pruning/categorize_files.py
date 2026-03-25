#!/usr/bin/env python3
"""
Categorize all convolution files by layout for library splitting.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def extract_layout(filepath):
    """
    Extract layout from filepath.
    Returns tuple: (primary_layout, category, is_grouped)
    """
    filepath_str = str(filepath)

    # Determine if grouped
    is_grouped = 'grouped_conv' in filepath_str

    # Determine dimension (1d, 2d, 3d)
    if 'conv1d' in filepath_str:
        dim = '1d'
    elif 'conv3d' in filepath_str:
        dim = '3d'
    elif 'conv2d' in filepath_str:
        dim = '2d'
    else:
        dim = 'unknown'

    # Extract layout pattern from filename
    layout_patterns_3d = ['gndhwc', 'ndhwgc', 'ngcdhw', 'ndhwc']
    layout_patterns_2d = ['gnhwc', 'nhwgc', 'ngchw', 'nhwc']
    layout_patterns_1d = ['gnwc', 'nwgc', 'nwc']

    filename = Path(filepath).name.lower()

    # Try to find layout in filename
    layout = None
    for pattern in layout_patterns_3d + layout_patterns_2d + layout_patterns_1d:
        if f'_{pattern}_' in filename:
            layout = pattern.upper()
            break

    # If not in filename, check parent directory
    if not layout:
        parts = Path(filepath).parts
        for part in reversed(parts):
            part_lower = part.lower()
            for pattern in layout_patterns_3d + layout_patterns_2d + layout_patterns_1d:
                if pattern == part_lower:
                    layout = pattern.upper()
                    break
            if layout:
                break

    # Determine operation type
    if 'bwd_weight' in filepath_str:
        op_type = 'bwd_weight'
    elif 'bwd_data' in filepath_str:
        op_type = 'bwd_data'
    elif 'fwd' in filepath_str:
        op_type = 'fwd'
    else:
        op_type = 'unknown'

    # Special categories
    if 'quantization' in filepath_str:
        category = 'quantization'
        layout = 'QUANTIZATION'  # Mark as special
    elif 'grouped_convnd_bwd_weight' in filepath_str:
        category = 'convnd_generic'
        layout = 'ND_GENERIC'  # Layout-agnostic N-dimensional
    elif not is_grouped:
        category = 'old'  # Non-grouped go to "old" library
        if not layout:
            layout = 'OLD'  # Mark as old/legacy
    else:
        category = f'{dim}_{op_type}'

    return layout, category, is_grouped, dim, op_type


def categorize_files(input_file):
    """Read file list and categorize by layout."""

    with open(input_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    # Storage structures
    by_layout = defaultdict(list)
    by_library = defaultdict(list)
    uncategorized = []
    stats = defaultdict(int)

    for filepath in files:
        layout, category, is_grouped, dim, op_type = extract_layout(filepath)

        if layout:
            by_layout[layout].append(filepath)

            # Determine target library
            if category == 'quantization':
                library = 'device_quantization_operations'
            elif category == 'convnd_generic':
                library = 'device_convnd_generic_operations'
            elif category == 'old' or not is_grouped:
                library = 'device_conv_old_operations'
            else:
                # Grouped convolutions get layout-specific libraries
                library = f'device_conv{dim}_{layout.lower()}_operations'

            by_library[library].append(filepath)
            stats[library] += 1
        else:
            uncategorized.append(filepath)
            stats['UNCATEGORIZED'] += 1

    return {
        'by_layout': dict(by_layout),
        'by_library': dict(by_library),
        'uncategorized': uncategorized,
        'stats': dict(stats),
        'total_files': len(files)
    }


def main():
    input_file = Path(__file__).parent / 'all_convolution_files.txt'
    output_json = Path(__file__).parent / 'layout_mapping.json'
    output_md = Path(__file__).parent / 'LAYOUT_FILE_MANIFEST.md'

    print(f"Reading files from: {input_file}")
    results = categorize_files(input_file)

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON mapping to: {output_json}")

    # Generate Markdown report
    with open(output_md, 'w') as f:
        f.write("# Layout-Based File Manifest\n\n")
        f.write(f"**Total Files**: {results['total_files']}\n\n")

        f.write("## Summary by Target Library\n\n")
        f.write("| Library Target | File Count |\n")
        f.write("|----------------|------------|\n")

        for lib in sorted(results['stats'].keys()):
            count = results['stats'][lib]
            f.write(f"| `{lib}` | {count} |\n")

        f.write(f"\n**Total**: {results['total_files']} files\n\n")

        # Detailed breakdown
        f.write("## Detailed File Listings by Library\n\n")

        for library in sorted(results['by_library'].keys()):
            files = results['by_library'][library]
            f.write(f"### {library} ({len(files)} files)\n\n")

            for filepath in sorted(files):
                # Make path relative for readability
                rel_path = filepath.replace(
                    '/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu/',
                    ''
                )
                f.write(f"- `{rel_path}`\n")

            f.write("\n")

        # Uncategorized
        if results['uncategorized']:
            f.write(f"## ⚠️ Uncategorized Files ({len(results['uncategorized'])})\n\n")
            for filepath in sorted(results['uncategorized']):
                rel_path = filepath.replace(
                    '/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/library/src/tensor_operation_instance/gpu/',
                    ''
                )
                f.write(f"- `{rel_path}`\n")

    print(f"Saved Markdown report to: {output_md}")

    # Print summary
    print("\n" + "="*60)
    print("CATEGORIZATION SUMMARY")
    print("="*60)
    print(f"Total files: {results['total_files']}")
    print(f"\nBy Library:")
    for lib in sorted(results['stats'].keys()):
        print(f"  {lib}: {results['stats'][lib]}")

    if results['uncategorized']:
        print(f"\n⚠️  WARNING: {len(results['uncategorized'])} files could not be categorized!")
    else:
        print(f"\n✅ All {results['total_files']} files successfully categorized!")

    print("="*60)


if __name__ == '__main__':
    main()
