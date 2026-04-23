"""Load and parse profile dump files and rocasm source files.

Reads the pipe-delimited profile dump format produced by
``python -m rocasm.profile_dump`` and pairs it with source map data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProfiledSourceLine:
    """A single line of rocasm Python code with associated profiling data."""
    index: int              # 0-based line index in the mainloop body
    inst_type: str          # "mfma", "ds_read", "buffer_load", etc.
    avg_lat: float          # average latency per hit
    stall: int              # total stall cycles across all hits
    stall_per_hit: float    # stall / hitcount (computed from avg_lat context)
    python_text: str        # the Python rocasm instruction text
    asm_text: str           # assembly text (from source map, if available)


@dataclass
class ProfileData:
    """Parsed profile dump + optional source map data."""
    lines: list[ProfiledSourceLine] = field(default_factory=list)
    file_mtime: float = 0.0    # mtime of the profile file (for watch mode)

    def has_changed(self, path: Path) -> bool:
        """Check if the file at path has been modified since last load."""
        try:
            return os.path.getmtime(path) > self.file_mtime
        except OSError:
            return False


def parse_profile_dump(path: str | Path) -> ProfileData:
    """Parse a profile dump flat file into ProfileData.

    Expected format (pipe-delimited)::

        # TYPE         | AVG_LAT | STALL | PYTHON
        mfma           |     4.0 |     0 | Acc[0:4] = vmfma_f32_16x16x16bf16_1k(...)
    """
    path = Path(path)
    lines: list[ProfiledSourceLine] = []
    mtime = os.path.getmtime(path)

    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.rstrip("\n")
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            parts = raw_line.split("|")
            if len(parts) != 4:
                continue
            inst_type = parts[0].strip()
            try:
                avg_lat = float(parts[1].strip())
                stall = int(parts[2].strip())
            except ValueError:
                continue
            python_text = parts[3].strip()

            # Estimate stall_per_hit from avg_lat:
            # avg_lat = latency / hitcount, and latency = stall + issue_time
            # For simplicity, use stall / (latency / avg_lat) if we had hitcount.
            # Since we don't have hitcount directly, approximate:
            # if avg_lat > 0 and stall > 0, stall_per_hit ≈ stall * avg_lat / latency
            # But we can derive: hitcount = latency / avg_lat, stall_per_hit = stall / hitcount
            # latency = hitcount * avg_lat, so hitcount = latency / avg_lat
            # We don't have total latency... but stall_per_hit = avg_lat - min_issue
            # Simpler: just use (avg_lat - 4.0) as stall_per_hit for colored types,
            # since 4 is the minimum issue latency on CDNA3.
            # Actually, even simpler: stall_per_hit = stall / hitcount.
            # We can back-derive hitcount if we know that for any line with stall=0,
            # avg_lat should be ~4.0 (min), so hitcount ≈ total_lat / avg_lat.
            # But we only have avg_lat and stall.
            # Best approach: estimate hitcount from a reference line.
            stall_per_hit = avg_lat - 4.0 if avg_lat >= 4.0 else 0.0

            lines.append(ProfiledSourceLine(
                index=len(lines),
                inst_type=inst_type,
                avg_lat=avg_lat,
                stall=stall,
                stall_per_hit=stall_per_hit,
                python_text=python_text,
                asm_text="",  # populated from source map if available
            ))

    return ProfileData(lines=lines, file_mtime=mtime)


def load_source_map_asm(map_path: str | Path,
                        profile: ProfileData) -> ProfileData:
    """Enrich ProfileData with assembly text from a source map JSON file."""
    import json
    map_path = Path(map_path)
    if not map_path.exists():
        return profile

    with open(map_path) as f:
        entries = json.load(f)

    for i, entry in enumerate(entries):
        if i < len(profile.lines):
            profile.lines[i].asm_text = entry.get("asm_text", "")

    return profile
