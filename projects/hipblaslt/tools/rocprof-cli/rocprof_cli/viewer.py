"""Textual TUI for viewing rocasm Python code with ATT profiling annotations.

Displays each line of rocasm with background color indicating instruction type
and stall severity.  Navigable with vim-like keybindings.  Bottom panel shows
the assembly instruction for the currently selected line.

Supports watch mode: polls for file changes and reloads automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static, ListView, ListItem, Label

from rocprof_cli.colors import compute_baselines, style_line, LineStyle
from rocprof_cli.loader import (
    ProfileData,
    ProfiledSourceLine,
    parse_profile_dump,
    load_source_map_asm,
)


class WarningBar(Static):
    """Warning bar shown at the bottom when files are out of sync."""

    DEFAULT_CSS = """
    WarningBar {
        height: 1;
        dock: bottom;
        background: #af5f00;
        color: #ffffff;
        text-style: bold;
        padding: 0 1;
        display: none;
    }
    """

    def show_warning(self, message: str):
        self.update(f" ⚠ {message}")
        self.display = True

    def hide_warning(self):
        self.display = False


class AsmPanel(Static):
    """Bottom panel showing assembly text for the selected line."""

    DEFAULT_CSS = """
    AsmPanel {
        height: 3;
        dock: bottom;
        background: $surface;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    def update_line(self, line: Optional[ProfiledSourceLine]):
        if line is None:
            self.update("[dim]No line selected[/dim]")
            return
        asm = line.asm_text or "[dim]no assembly mapping[/dim]"
        stats = (f"[bold]{line.inst_type}[/bold]  "
                 f"avg_lat={line.avg_lat:.1f}  stall={line.stall}  "
                 f"stall/hit={line.stall_per_hit:.1f}")
        self.update(f"{stats}\n{asm}")


class CodeLine(Static):
    """A single styled line of profiled code, rendered as a Static widget."""

    def __init__(self, rich_text: Text, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rich_text = rich_text

    def render(self):
        return self._rich_text


def _build_line_text(line: ProfiledSourceLine, line_style: LineStyle,
                     line_num: int, width: int = 200) -> Text:
    """Build a Rich Text object for a single profiled line with background color."""
    # Build the text content
    stall_str = f"{line.stall:>6d}" if line.stall > 0 else "     ."
    content = f" {line_num:>3d} {stall_str} │ {line.python_text}"
    # Pad to full width so background color fills the line
    content = content.ljust(width)

    # Determine styles
    bg = line_style.bg_color if line_style.bg_color else None
    if bg:
        line_rich_style = Style(bgcolor=bg)
    else:
        line_rich_style = Style()

    text = Text(content, style=line_rich_style)

    # Highlight the gutter (line number + stall) portion
    gutter_end = 13  # " NNN SSSSSS "
    text.stylize(Style(dim=True), 0, 5)  # line number dim
    if line.stall > 100:
        text.stylize(Style(bold=True), 5, 12)  # stall bold if significant

    return text


class ProfileViewer(App):
    """Main TUI application for viewing profiled rocasm code."""

    TITLE = "rocprof-cli"
    SUB_TITLE = "rocasm profiling viewer"

    CSS = """
    Screen {
        background: #1a1a1a;
    }
    ListView {
        height: 1fr;
        background: #1a1a1a;
    }
    ListItem {
        height: 1;
        padding: 0;
    }
    ListItem > CodeLine {
        width: 1fr;
        height: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "go_top", "Top", show=False),
        Binding("G", "go_bottom", "Bottom", show=False),
        Binding("ctrl+d", "page_down", "Page Down", show=False),
        Binding("ctrl+u", "page_up", "Page Up", show=False),
        Binding("A", "toggle_asm", "Toggle asm"),
    ]

    def __init__(self, profile_path: str, map_path: str | None = None,
                 start_line: int = 0, watch: bool = False):
        super().__init__()
        self.profile_path = Path(profile_path)
        self.map_path = Path(map_path) if map_path else None
        self.start_line = start_line
        self.watch_mode = watch
        self.profile_data: ProfileData | None = None
        self.baselines: dict[str, float] = {}
        self._line_items: list[tuple[ListItem, ProfiledSourceLine]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield ListView(id="code-list")
        yield AsmPanel(id="asm-panel")
        yield WarningBar(id="warning-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_profile()
        self._render_lines()
        self._check_staleness()

        # Jump to start line
        if self.start_line > 0 and self._line_items:
            list_view = self.query_one("#code-list", ListView)
            idx = min(self.start_line, len(self._line_items) - 1)
            list_view.index = idx

        # Always poll in watch mode; also poll when profile doesn't exist
        # yet so we pick it up when it appears
        if self.watch_mode or self.profile_data is None:
            self.set_interval(1.5, self._check_reload)

    def _load_profile(self) -> None:
        """Load profile data from the flat file.

        If the file doesn't exist yet, sets profile_data to None and
        shows a warning. The watch-mode poller will pick it up when
        it appears.
        """
        if not self.profile_path.exists():
            self.profile_data = None
            return

        self.profile_data = parse_profile_dump(self.profile_path)
        if self.map_path and self.map_path.exists():
            self.profile_data = load_source_map_asm(
                self.map_path, self.profile_data)
        # Compute baselines for color thresholds
        line_dicts = [{"type": l.inst_type, "stall_per_hit": l.stall_per_hit}
                      for l in self.profile_data.lines]
        self.baselines = compute_baselines(line_dicts)

    def _render_lines(self) -> None:
        """Populate the list view with styled lines."""
        list_view = self.query_one("#code-list", ListView)
        list_view.clear()
        self._line_items.clear()
        if not self.profile_data:
            # No data yet — show a waiting message
            from rich.text import Text as RichText
            msg = RichText("  Waiting for profile data...", style="dim italic")
            list_view.append(ListItem(CodeLine(msg)))
            return
        for i, line in enumerate(self.profile_data.lines):
            ls = style_line(line.inst_type, line.stall_per_hit, self.baselines)
            rich_text = _build_line_text(line, ls, i + 1)
            code_widget = CodeLine(rich_text)
            item = ListItem(code_widget)
            self._line_items.append((item, line))
            list_view.append(item)

    def _check_staleness(self) -> None:
        """Check if the source map is newer than the profile data.

        If .map.json is newer than profile.txt, the kernel was rebuilt
        but the profiler wasn't re-run — the profile data is stale.
        If profile.txt doesn't exist yet, show a waiting warning.
        """
        warning = self.query_one("#warning-bar", WarningBar)
        if self.profile_data is None:
            warning.show_warning(
                f"Waiting for profile data: {self.profile_path.name}")
            return
        if self.map_path and self.map_path.exists() and self.profile_path.exists():
            map_mtime = self.map_path.stat().st_mtime
            profile_mtime = self.profile_path.stat().st_mtime
            if map_mtime > profile_mtime:
                warning.show_warning(
                    "Profile data is stale: source map is newer than profile.txt. "
                    "Re-run profiler and profile_dump.")
                return
        warning.hide_warning()

    def _check_reload(self) -> None:
        """Poll for file changes in watch mode.

        Also handles the case where profile.txt didn't exist at startup
        but has now appeared.
        """
        if self.profile_data is None:
            # Profile didn't exist at startup — check if it appeared
            if self.profile_path.exists():
                self._load_profile()
                self._render_lines()
                self._check_staleness()
                self.sub_title = f"loaded — {self.profile_path.name}"
                self.set_timer(3.0, self._reset_subtitle)
            return

        if self.profile_data.has_changed(self.profile_path):
            old_index = self.query_one("#code-list", ListView).index
            self._load_profile()
            self._render_lines()
            self._check_staleness()
            # Restore scroll position
            list_view = self.query_one("#code-list", ListView)
            if old_index is not None and old_index < len(self._line_items):
                list_view.index = old_index
            self.sub_title = f"reloaded — {self.profile_path.name}"
            self.set_timer(3.0, self._reset_subtitle)

    def _reset_subtitle(self) -> None:
        self.sub_title = "rocasm profiling viewer"

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Update the assembly panel when cursor moves."""
        asm_panel = self.query_one("#asm-panel", AsmPanel)
        if event.item:
            # Find the profiled line for this item
            for item, profiled_line in self._line_items:
                if item is event.item:
                    asm_panel.update_line(profiled_line)
                    return
        asm_panel.update_line(None)

    # --- Vim-like navigation actions ---

    def action_cursor_down(self) -> None:
        self.query_one("#code-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#code-list", ListView).action_cursor_up()

    def action_go_top(self) -> None:
        self.query_one("#code-list", ListView).index = 0

    def action_go_bottom(self) -> None:
        lv = self.query_one("#code-list", ListView)
        lv.index = len(lv.children) - 1

    def action_page_down(self) -> None:
        lv = self.query_one("#code-list", ListView)
        page = lv.size.height // 2
        lv.index = min((lv.index or 0) + page, len(lv.children) - 1)

    def action_page_up(self) -> None:
        lv = self.query_one("#code-list", ListView)
        page = lv.size.height // 2
        lv.index = max((lv.index or 0) - page, 0)

    def action_toggle_asm(self) -> None:
        panel = self.query_one("#asm-panel", AsmPanel)
        panel.display = not panel.display
