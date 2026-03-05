#!/usr/bin/env python3
"""Merge benchmarking result files.

Takes a baseline results file and an update file. For each entry in the update
file that contains actual results (not "Skipping ..."), the baseline entry is
replaced. Skipped entries in the update file leave the baseline unchanged.
Entries that exist only in the update file are appended. The merged output is
written to a new file; the originals are never modified.
"""

import argparse
import re
import sys
from collections import OrderedDict

SEPARATOR = "=" * 80


def parse_entries(text: str) -> OrderedDict:
    """Parse a results file into an ordered dict keyed by the input command.

    Each value is the full text block for that entry (separator + command +
    separator + result body).
    """
    entries = OrderedDict()

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Use \Z (end of string) instead of $ to avoid MULTILINE matching at
    # end-of-line positions, which would cause the non-greedy body to stop
    # prematurely.
    pattern = re.compile(
        r"(={80}\n"                  # first separator
        r"Input command:\s*(.+?)\n"  # command line (captured)
        r"={80}\n"                   # second separator
        r"(.*?))"                    # body (captured, non-greedy)
        r"(?=\n={80}\n|\Z)",         # lookahead: next separator or end of string
        re.DOTALL,
    )

    for m in pattern.finditer(text):
        full_block = m.group(1).rstrip("\n")
        command = m.group(2).strip()
        entries[command] = full_block

    return entries


def is_skipped(block: str) -> bool:
    """Return True if this entry's body indicates a skipped computation."""
    for line in block.split("\n"):
        if re.match(r"^\s*Skipping\b", line):
            return True
    return False


def merge(baseline_text: str, update_text: str) -> str:
    """Merge update results into baseline, returning the merged text."""
    baseline = parse_entries(baseline_text)
    updates = parse_entries(update_text)

    merged = OrderedDict(baseline)  # start with a copy of baseline

    updated_count = 0
    skipped_count = 0
    added_count = 0

    for cmd, block in updates.items():
        if is_skipped(block):
            skipped_count += 1
            continue  # keep baseline entry (or ignore if not in baseline)
        if cmd in merged:
            updated_count += 1
        else:
            added_count += 1
        merged[cmd] = block

    print(f"Baseline entries : {len(baseline)}", file=sys.stderr)
    print(f"Update entries   : {len(updates)}", file=sys.stderr)
    print(f"  Updated        : {updated_count}", file=sys.stderr)
    print(f"  Skipped (kept) : {skipped_count}", file=sys.stderr)
    print(f"  New (appended) : {added_count}", file=sys.stderr)
    print(f"Output entries   : {len(merged)}", file=sys.stderr)

    return "\n\n".join(merged.values()) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Merge ckProfiler benchmark result files. "
        "Updates the baseline with new results from the update file. "
        "Entries marked as 'Skipping' in the update file are ignored, "
        "keeping the baseline values intact.",
    )
    parser.add_argument(
        "baseline",
        help="Path to the baseline results file",
    )
    parser.add_argument(
        "update",
        help="Path to the update results file (may contain skipped entries)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the merged output file",
    )
    args = parser.parse_args()

    with open(args.baseline, "r") as f:
        baseline_text = f.read()

    with open(args.update, "r") as f:
        update_text = f.read()

    result = merge(baseline_text, update_text)

    with open(args.output, "w") as f:
        f.write(result)

    print(f"Merged results written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
