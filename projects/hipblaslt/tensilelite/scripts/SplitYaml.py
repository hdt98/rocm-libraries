#!/usr/bin/env python3
"""
SplitYaml.py - Split a YAML test file into multiple files based on BenchmarkProblems sections.

Each section in BenchmarkProblems is identified by a comment block like:
  ########################################
  # F32 TN
  ########################################

The script will split each section into its own file, preserving the header metadata
(TestParameters, GlobalParameters, etc.) in each output file.

Usage:
    SplitYaml.py <input.yaml>

Example:
    SplitYaml.py dtl.yaml -> generates dtl_f32_tn.yaml, dtl_f32_nt.yaml, etc.
"""

import argparse
import os
import re
import sys


def find_benchmark_problems_line(lines):
    """Find the line index where 'BenchmarkProblems:' starts."""
    for i, line in enumerate(lines):
        if line.strip() == 'BenchmarkProblems:':
            return i
    return None


def parse_section_name(comment_line):
    """
    Extract section name from a comment line like '  # F32 TN'.
    Returns a cleaned name suitable for a filename (lowercase, underscores).
    """
    # Remove leading/trailing whitespace and the # prefix
    match = re.match(r'\s*#\s*(.+)\s*$', comment_line)
    if match:
        name = match.group(1).strip()
        # Convert to lowercase and replace spaces/special chars with underscores
        name = re.sub(r'[^a-zA-Z0-9]+', '_', name).lower()
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    return None


def find_sections(lines, benchmark_start):
    """
    Find all sections within BenchmarkProblems.

    Each section starts with a comment block like:
      ########################################
      # F32 TN
      ########################################

    Returns a list of tuples: (section_name, start_line, end_line)
    where start_line is the line with the first '#' of the section header
    and end_line is the last line of the section (exclusive).
    """
    sections = []
    current_section_name = None
    current_section_start = None

    # Pattern for the separator line (lots of # characters)
    separator_pattern = re.compile(r'^\s*#{5,}\s*$')

    i = benchmark_start + 1
    while i < len(lines):
        line = lines[i]

        # Check if this is a 3-line header block:
        # Line i:   ########################################
        # Line i+1: # Section Name
        # Line i+2: ########################################
        if separator_pattern.match(line):
            # Check if this is a full header block (3 lines)
            if (i + 2 < len(lines) and
                separator_pattern.match(lines[i + 2])):
                # This looks like a proper section header
                name = parse_section_name(lines[i + 1])
                if name:
                    # If we were tracking a section, close it
                    if current_section_name is not None:
                        sections.append((current_section_name, current_section_start, i))

                    # Start tracking new section
                    current_section_start = i
                    current_section_name = name

                    # Skip past the header block
                    i += 2

        i += 1

    # Don't forget the last section
    if current_section_name is not None:
        sections.append((current_section_name, current_section_start, len(lines)))

    return sections


def get_header_content(lines, benchmark_start):
    """Get the header content (everything before BenchmarkProblems content)."""
    return lines[:benchmark_start + 1]  # Include 'BenchmarkProblems:'


def get_section_content(lines, start, end):
    """Get the content of a section."""
    return lines[start:end]


def generate_output_filename(input_path, section_name):
    """Generate output filename based on input filename and section name."""
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)

    output_name = f"{name}_{section_name}{ext}"
    return os.path.join(directory, output_name)


def get_snippet(lines, start, max_lines=10):
    """Get a snippet of lines for display."""
    end = min(start + max_lines, len(lines))
    snippet_lines = lines[start:end]
    if end < len(lines) and len(lines) - start > max_lines:
        snippet_lines.append('    ...')
    return ''.join(snippet_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Split a YAML test file into multiple files based on BenchmarkProblems sections.'
    )
    parser.add_argument('input_file', help='Input YAML file to split')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt and proceed with splitting')
    args = parser.parse_args()

    input_path = args.input_file

    # Verify input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Read the input file
    with open(input_path, 'r') as f:
        content = f.read()

    lines = content.splitlines(keepends=True)

    # Ensure last line has newline
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'

    # Find BenchmarkProblems section
    benchmark_start = find_benchmark_problems_line(lines)
    if benchmark_start is None:
        print("Error: Could not find 'BenchmarkProblems:' in the input file.", file=sys.stderr)
        sys.exit(1)

    # Find all sections
    sections = find_sections(lines, benchmark_start)

    if not sections:
        print("Error: No sections found within BenchmarkProblems.", file=sys.stderr)
        sys.exit(1)

    # Check for duplicate section names and make them unique
    name_counts = {}
    unique_sections = []
    for name, start, end in sections:
        if name in name_counts:
            name_counts[name] += 1
            unique_name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 0
            unique_name = name
        unique_sections.append((unique_name, start, end))

    sections = unique_sections

    # Get header content
    header_lines = get_header_content(lines, benchmark_start)

    # Generate output information
    output_files = []
    for section_name, start, end in sections:
        output_path = generate_output_filename(input_path, section_name)
        output_files.append((output_path, section_name, start, end))

    # Print summary
    print(f"\nInput file: {input_path}")
    print(f"Found {len(sections)} section(s) to split\n")
    print("=" * 70)

    for output_path, section_name, start, end in output_files:
        # Line numbers are 1-indexed for display
        start_line = start + 1
        end_line = end

        print(f"\nFound section \"{section_name}\" at lines {start_line} through {end_line}:")
        print("-" * 50)
        snippet = get_snippet(lines, start, max_lines=8)
        # Indent snippet for readability
        for line in snippet.splitlines(keepends=True):
            print(f"  {line}", end='')
        if not snippet.endswith('\n'):
            print()
        print("-" * 50)
        print(f"  -> Will create: {os.path.basename(output_path)}")

    print("\n" + "=" * 70)
    print(f"\nSummary:")
    print(f"  - {len(sections)} new file(s) will be created")
    print(f"  - Files will be written to: {os.path.dirname(os.path.abspath(input_path)) or '.'}")
    print(f"\nNew files:")
    for output_path, _, _, _ in output_files:
        print(f"  - {os.path.basename(output_path)}")

    # Ask for confirmation
    if not args.yes:
        print()
        response = input("Proceed with creating these files? [yes/no]: ").strip().lower()
        if response != 'yes':
            print("Aborted.")
            sys.exit(0)

    # Create output files
    print("\nCreating files...")
    for output_path, section_name, start, end in output_files:
        section_lines = get_section_content(lines, start, end)

        # Combine header with section content
        output_content = ''.join(header_lines) + ''.join(section_lines)

        with open(output_path, 'w') as f:
            f.write(output_content)

        print(f"  Created: {output_path}")

    print(f"\nDone! Created {len(output_files)} file(s).")


if __name__ == '__main__':
    main()
