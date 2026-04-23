"""Color scheme and threshold logic for profiled rocasm lines.

Maps instruction types to colors (matching RocProf Compute Viewer) and
determines intensity levels based on absolute stall thresholds.

Colors:
    Green  — mfma / VALU instructions
    Orange — ds_read (LDS load) instructions
    Yellow — buffer_load (global read) instructions

Intensity levels:
    neutral — no meaningful stall (background matches normal editor)
    medium  — stall above baseline, worth noting
    bright  — severe stall, performance problem
"""

from __future__ import annotations

from dataclasses import dataclass


# Absolute thresholds for per-hit average stall above class baseline.
# Anchored to MFMA issue latency (~8 cycles on CDNA3) as a reference unit.
MEDIUM_THRESHOLD = 8    # 1× MFMA latency above baseline
BRIGHT_THRESHOLD = 32   # 4× MFMA latency above baseline


# Instruction type → (neutral_bg, medium_bg, bright_bg) as hex RGB strings.
# neutral is a subtle tint so the instruction class is visible even without stalls.
# Colors are (neutral, medium, bright).
# Neutral is always empty (no background) — color only appears when there's a stall.
# Medium/bright use saturated colors that are clearly distinguishable even in
# 256-color terminals (tmux with TERM=screen).
_COLOR_MAP = {
    "mfma":        ("", "#005f00", "#00af00"),    # green (256-color: 22, 34)
    "ds_read":     ("", "#af5f00", "#ff8700"),    # orange (256-color: 130, 208)
    "ds_write":    ("", "#af5f00", "#ff8700"),    # orange (same as ds_read)
    "buffer_load": ("", "#afaf00", "#ffff00"),    # yellow (256-color: 142, 226)
    "s_waitcnt":   ("", "#005faf", "#0087ff"),    # blue (256-color: 25, 33)
    "s_barrier":   ("", "#005faf", "#0087ff"),    # blue (sync)
}

# Default for instruction types that don't get colored (scalar, branch, label, other)
_DEFAULT_COLORS = ("", "", "")


@dataclass
class LineStyle:
    """Styling information for a single profiled line."""
    bg_color: str       # hex RGB for background, or "" for default
    intensity: str      # "neutral", "medium", or "bright"
    inst_type: str      # instruction classification


def compute_baselines(profile_lines: list[dict]) -> dict[str, float]:
    """Compute per-instruction-class baseline (minimum avg stall per hit).

    Args:
        profile_lines: list of dicts with keys "type", "stall", "avg_lat"
            (parsed from the profile dump flat file).

    Returns:
        dict mapping instruction type to minimum avg_stall_per_hit for that class.
    """
    from collections import defaultdict
    class_stalls: dict[str, list[float]] = defaultdict(list)
    for line in profile_lines:
        itype = line["type"]
        if itype in _COLOR_MAP:
            class_stalls[itype].append(line["stall_per_hit"])
    return {itype: min(vals) if vals else 0.0
            for itype, vals in class_stalls.items()}


def style_line(inst_type: str, stall_per_hit: float,
               baselines: dict[str, float]) -> LineStyle:
    """Determine the visual style for a profiled line.

    Args:
        inst_type: instruction classification (e.g., "mfma", "buffer_load")
        stall_per_hit: average stall cycles per execution for this instruction
        baselines: per-class minimum stall from compute_baselines()

    Returns:
        LineStyle with background color and intensity level.
    """
    colors = _COLOR_MAP.get(inst_type, _DEFAULT_COLORS)
    baseline = baselines.get(inst_type, 0.0)
    excess = stall_per_hit - baseline

    if excess >= BRIGHT_THRESHOLD:
        return LineStyle(bg_color=colors[2], intensity="bright", inst_type=inst_type)
    elif excess >= MEDIUM_THRESHOLD:
        return LineStyle(bg_color=colors[1], intensity="medium", inst_type=inst_type)
    else:
        return LineStyle(bg_color=colors[0], intensity="neutral", inst_type=inst_type)
