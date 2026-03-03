#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert a Markdown file to PDF.

Dependencies:
    pip install markdown fpdf2

Usage:
    python3 md_to_pdf.py [INPUT.md] [OUTPUT.pdf]

If no arguments are given, converts operation_support_matrix.md in the same
directory as this script to operation_support_matrix.pdf.
"""

import os
import sys
from html.parser import HTMLParser

import markdown
from fpdf import FPDF, XPos, YPos


# Map emoji to plain-text equivalents that render in any font.
_EMOJI_MAP = {
    "\u2705": "Y",  # ✅ → Y (supported)
    "\u274c": "N",  # ❌ → N (not supported)
}


def _replace_emoji(text: str) -> str:
    for emoji, replacement in _EMOJI_MAP.items():
        text = text.replace(emoji, replacement)
    return text


# ---------------------------------------------------------------------------
# Lightweight HTML-table parser that feeds rows/cells into the PDF renderer.
# ---------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Extract tables and surrounding prose from simple HTML."""

    def __init__(self):
        super().__init__()
        self.blocks = []  # ("text",str) | ("heading",level,str) | ("table",rows)
        self._in_table = False
        self._in_cell = False
        self._rows = []
        self._row = []
        self._cell = ""
        self._is_header = False
        self._row_is_header = False
        self._prose = ""
        self._in_code = False
        self._in_li = False
        self._li_depth = 0
        self._heading_tag = None
        self._skip_tags = {"thead", "tbody", "tfoot"}

    def _flush_prose(self):
        t = self._prose.strip()
        if t:
            self.blocks.append(("text", t))
        self._prose = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._flush_prose()
            self._in_table = True
            self._rows = []
            return
        if tag == "tr":
            self._row = []
            self._row_is_header = False
            return
        if tag in ("th", "td"):
            self._in_cell = True
            self._is_header = tag == "th"
            if self._is_header:
                self._row_is_header = True
            self._cell = ""
            return
        if tag == "br" and self._in_cell:
            self._cell += "\n"
            return
        if tag == "code":
            self._in_code = True
            return
        if tag in ("ul", "ol"):
            self._li_depth += 1
            return
        if tag == "li":
            self._in_li = True
            return
        if tag in ("h1", "h2", "h3"):
            self._flush_prose()
            self._heading_tag = tag
            return
        if tag in self._skip_tags:
            return

    def handle_endtag(self, tag):
        if tag == "table":
            self._in_table = False
            self.blocks.append(("table", self._rows))
            self._rows = []
            return
        if tag == "tr":
            self._rows.append((self._row_is_header, self._row))
            return
        if tag in ("th", "td"):
            self._in_cell = False
            self._row.append(self._cell.strip())
            return
        if tag == "code":
            self._in_code = False
            return
        if tag in ("ul", "ol"):
            self._li_depth -= 1
            return
        if tag == "li":
            self._in_li = False
            self._prose += "\n"
            return
        if tag in ("h1", "h2", "h3"):
            return
        if tag == "p":
            self._prose += "\n"
            return
        if tag in self._skip_tags:
            return

    def handle_data(self, data):
        if self._in_cell:
            self._cell += data
            return
        if self._heading_tag:
            self._flush_prose()
            level = int(self._heading_tag[1])
            self.blocks.append(("heading", level, data.strip()))
            self._heading_tag = None
            return
        if self._in_li:
            indent = "  " * max(0, self._li_depth - 1)
            self._prose += f"{indent}* {data.strip()} "
            return
        self._prose += data

    def close(self):
        super().close()
        self._flush_prose()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------
# Cell background colours used in the matrix table.
_CLR_YES = (200, 240, 200)  # light green  → supported (Y)
_CLR_NO = (250, 210, 210)  # light red    → not yet (N)
_CLR_BLANK = (245, 245, 245)  # light grey   → n/a


class MatrixPDF(FPDF):
    """Landscape PDF with auto-sized tables."""

    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=12)

        self._font_name = "Helvetica"
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        ):
            if os.path.isfile(candidate):
                bold = candidate.replace("Sans.", "Sans-Bold.")
                self.add_font("DejaVu", "", candidate)
                if os.path.isfile(bold):
                    self.add_font("DejaVu", "B", bold)
                else:
                    self.add_font("DejaVu", "B", candidate)
                self._font_name = "DejaVu"
                break

    # --- public API --------------------------------------------------------
    def render_blocks(self, blocks):
        for block in blocks:
            kind = block[0]
            if kind == "text":
                self._render_text(block[1])
            elif kind == "heading":
                self._render_heading(block[1], block[2])
            elif kind == "table":
                self._render_table(block[1])

    # --- internals ---------------------------------------------------------
    def _render_heading(self, level: int, text: str):
        sizes = {1: 14, 2: 11, 3: 9}
        size = sizes.get(level, 9)
        self.ln(3 if level == 1 else 2)
        self.set_font(self._font_name, "B", size)
        self.cell(
            0, size * 0.6, _replace_emoji(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        if level == 1:
            # underline
            y = self.get_y() + 1
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, y, self.w - self.r_margin, y)
            self.set_draw_color(0, 0, 0)
        self.ln(3)

    def _render_text(self, text: str):
        self.set_font(self._font_name, size=8)
        for paragraph in text.split("\n"):
            p = _replace_emoji(paragraph.strip())
            if not p:
                continue
            is_heading = p.startswith("**") and p.endswith("**")
            if is_heading:
                p = p.strip("*").strip()
                self.set_font(self._font_name, "B", 9)
            else:
                self.set_font(self._font_name, size=8)
            self.multi_cell(0, 4, p)
            self.ln(1)

    def _cell_bg(self, text: str):
        """Return (r,g,b) background for an indicator cell."""
        t = text.strip()
        if t == "Y":
            return _CLR_YES
        if t == "N":
            return _CLR_NO
        return _CLR_BLANK

    def _render_table(self, rows):
        if not rows:
            return
        ncols = max(len(r) for _, r in rows)
        if ncols == 0:
            return

        page_w = self.w - self.l_margin - self.r_margin

        # Column widths: col0 = Op, col1 = Kernel name, rest = indicators
        if ncols >= 3:
            col0_w = 18
            col1_w = 62
            remaining = page_w - col0_w - col1_w
            rest_w = max(remaining / (ncols - 2), 8)
            col_widths = [col0_w, col1_w] + [rest_w] * (ncols - 2)
        else:
            col_widths = [page_w / ncols] * ncols

        row_h = 5

        for is_header, cells in rows:
            cells = [_replace_emoji(c) for c in cells]
            cells += [""] * (ncols - len(cells))

            # Compute multi-line height
            cell_lines = []
            max_lines = 1
            for i, cell in enumerate(cells):
                w = col_widths[i] - 2
                lines = self._wrap_text(cell, w, 6 if i == 1 else 5.5)
                cell_lines.append(lines)
                max_lines = max(max_lines, len(lines))
            this_row_h = max(row_h, max_lines * row_h)

            if self.get_y() + this_row_h > self.h - self.b_margin:
                self.add_page()

            y0 = self.get_y()
            x0 = self.l_margin

            for i, cell in enumerate(cells):
                x = x0 + sum(col_widths[:i])
                w = col_widths[i]

                # --- background ---
                if is_header:
                    self.set_fill_color(220, 228, 240)
                    self.set_draw_color(180, 190, 200)
                elif i >= 2:
                    r, g, b = self._cell_bg(cell)
                    self.set_fill_color(r, g, b)
                    self.set_draw_color(200, 207, 214)
                else:
                    self.set_fill_color(255, 255, 255)
                    self.set_draw_color(200, 207, 214)

                self.rect(x, y0, w, this_row_h, "FD")

                # --- text ---
                if is_header:
                    self.set_font(self._font_name, "B", 5.5)
                    self.set_text_color(30, 40, 60)
                elif i >= 2:
                    t = cell.strip()
                    self.set_font(self._font_name, "B", 6)
                    if t == "Y":
                        self.set_text_color(30, 120, 30)
                    elif t == "N":
                        self.set_text_color(180, 40, 40)
                    else:
                        self.set_text_color(160, 160, 160)
                else:
                    self.set_font(self._font_name, size=6 if i == 1 else 5.5)
                    self.set_text_color(30, 30, 30)

                lines = cell_lines[i]
                text_block_h = len(lines) * row_h
                y_off = (this_row_h - text_block_h) / 2
                for li, line in enumerate(lines):
                    self.set_xy(x, y0 + y_off + li * row_h)
                    align = "L" if i == 1 else "C"
                    self.cell(w, row_h, line, align=align)

            # Reset colours
            self.set_text_color(0, 0, 0)
            self.set_draw_color(0, 0, 0)
            self.set_y(y0 + this_row_h)

    def _wrap_text(self, text: str, max_w_mm: float, font_size: float):
        self.set_font(self._font_name, size=font_size)
        lines = []
        for raw_line in text.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                lines.append("")
                continue
            if self.get_string_width(raw_line) <= max_w_mm:
                lines.append(raw_line)
            else:
                words = raw_line.split()
                current = ""
                for w in words:
                    test = f"{current} {w}".strip()
                    if self.get_string_width(test) <= max_w_mm:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = w
                if current:
                    lines.append(current)
        return lines or [""]


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=["tables"])

    parser = _TableParser()
    parser.feed(html_body)
    parser.close()

    pdf = MatrixPDF()
    pdf.add_page()

    pdf.render_blocks(parser.blocks)
    pdf.output(output_path)
    print(f"PDF written to {output_path}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) >= 3:
        input_path, output_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        output_path = os.path.splitext(input_path)[0] + ".pdf"
    else:
        input_path = os.path.join(script_dir, "operation_support_matrix.md")
        output_path = os.path.join(script_dir, "operation_support_matrix.pdf")

    if not os.path.isfile(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
