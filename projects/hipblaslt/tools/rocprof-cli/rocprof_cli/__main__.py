"""Entry point for ``python -m rocprof_cli``.

Usage::

    python -m rocprof_cli profile.txt                        # view profile
    python -m rocprof_cli profile.txt +42                    # open at line 42
    python -m rocprof_cli profile.txt --map mainloop.map.json  # with assembly
    python -m rocprof_cli profile.txt --watch                # live reload
"""

import argparse
import sys

from rocprof_cli.viewer import ProfileViewer


def main():
    parser = argparse.ArgumentParser(
        prog="rocprof-cli",
        description="Terminal UI for viewing rocasm code with ATT profiling",
    )
    parser.add_argument("profile", help="Path to profile dump file")
    parser.add_argument("line", nargs="?", default=None,
                        help="Line number to open at (e.g., +42)")
    parser.add_argument("--map", default=None,
                        help="Path to _mainloop.map.json for assembly view")
    parser.add_argument("--watch", action="store_true",
                        help="Watch for file changes and reload automatically")

    args = parser.parse_args()

    # Parse +N line number
    start_line = 0
    if args.line:
        if args.line.startswith("+"):
            try:
                start_line = int(args.line[1:])
            except ValueError:
                print(f"Invalid line number: {args.line}", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                start_line = int(args.line)
            except ValueError:
                print(f"Invalid line number: {args.line}", file=sys.stderr)
                sys.exit(1)

    app = ProfileViewer(
        profile_path=args.profile,
        map_path=args.map,
        start_line=start_line,
        watch=args.watch,
    )
    app.run()


if __name__ == "__main__":
    main()
