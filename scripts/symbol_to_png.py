#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo-root imports work when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets

from fonts import register_font_from_bytes


def _build_symbol(args: argparse.Namespace) -> str:
    if args.codepoint:
        cp = args.codepoint.strip().lower().replace("0x", "")
        return chr(int(cp, 16))
    return args.symbol


def _resolve_leland_family() -> str:
    requested = register_font_from_bytes("LelandText") or "LelandText"
    try:
        families = set(QtGui.QFontDatabase.families())
    except Exception:
        return requested
    for candidate in (requested, "LelandText", "Leland Text"):
        if candidate in families:
            return candidate
    return requested


def _qcolor_or_exit(value: str, label: str) -> QtGui.QColor:
    color = QtGui.QColor(value)
    if not color.isValid():
        raise SystemExit(f"Invalid {label} color: {value}")
    return color


def render_symbol(args: argparse.Namespace) -> Path:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app  # Keep a reference.

    symbol = _build_symbol(args)
    if not symbol:
        raise SystemExit("No symbol provided.")

    family = _resolve_leland_family()

    font = QtGui.QFont()
    font.setFamily(family)
    font.setPointSizeF(float(args.point_size))
    font.setStyleStrategy(
        QtGui.QFont.StyleStrategy.PreferMatch
        | QtGui.QFont.StyleStrategy.NoFontMerging
    )

    size = int(args.size)
    padding = int(args.padding)
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)

    bg = _qcolor_or_exit(args.background, "background")
    if args.background.lower() == "transparent":
        image.fill(QtCore.Qt.GlobalColor.transparent)
    else:
        image.fill(bg)

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
    painter.setFont(font)

    fg = _qcolor_or_exit(args.color, "foreground")
    painter.setPen(QtGui.QPen(fg))

    draw_rect = QtCore.QRectF(padding, padding, size - (2 * padding), size - (2 * padding))
    painter.drawText(draw_rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), symbol)
    painter.end()

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = image.save(str(out_path), "PNG")
    if not ok:
        raise SystemExit(f"Failed to write PNG: {out_path}")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a single LelandText symbol into a PNG file."
    )
    p.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "scripts" / "mf_symbol.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--symbol",
        default="\ue52d",
        help="Direct glyph text to render (default: mf in LelandText).",
    )
    p.add_argument(
        "--codepoint",
        default="",
        help="Hex codepoint like e52d or 0xe52d (overrides --symbol).",
    )
    p.add_argument(
        "--size",
        type=int,
        default=128,
        help="Square image size in px.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=92.0,
        help="Font point size.",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding around glyph in px.",
    )
    p.add_argument(
        "--color",
        default="#000000",
        help="Foreground color (#RRGGBB or named Qt color).",
    )
    p.add_argument(
        "--background",
        default="transparent",
        help="Background color or 'transparent'.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = render_symbol(args)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
