from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ui.widgets.draw_util import DrawUtil
from symbol_design.symbol_util import SymbolUtil

if TYPE_CHECKING:
    from symbol_design.noteheads.notehead import Notehead


def _bullet_height_mm(symbol: SymbolUtil) -> float:
    base_h = float(symbol.semitone_space_mm) * 2.0
    # Make the bullet 50% longer than the base notehead height.
    return base_h * 1.5


def _down_bullet_points(x_mm: float, top_y: float, half_w: float, bullet_h: float, curve_points: int) -> list[tuple[float, float]]:
    """Build points for a down-facing bullet: flat top, long pointy rounded bottom."""
    body_h = float(bullet_h) / 3.0
    tail_h = float(bullet_h) - body_h
    shoulder_y = float(top_y) + body_h
    cx = float(x_mm)

    points: list[tuple[float, float]] = [
        (cx - half_w, float(top_y)),
        (cx + half_w, float(top_y)),
        (cx + half_w, shoulder_y),
    ]

    # Taper to tip with a power profile for a pointier bullet shape.
    n = max(4, int(curve_points))
    half_n = max(2, n // 2)
    taper_power = 1.8

    # Right shoulder -> tip
    for i in range(1, half_n + 1):
        t = float(i) / float(half_n)
        px = cx + (half_w * ((1.0 - t) ** taper_power))
        py = shoulder_y + (tail_h * t)
        points.append((px, py))

    # Tip -> left shoulder
    for i in range(1, half_n + 1):
        s = float(i) / float(half_n)
        px = cx - (half_w * (s ** taper_power))
        py = shoulder_y + (tail_h * (1.0 - s))
        points.append((px, py))

    return points


def _flip_vertical(points: list[tuple[float, float]], top_y: float, bullet_h: float) -> list[tuple[float, float]]:
    """Mirror points vertically inside the [top_y, top_y + bullet_h] bounds."""
    y0 = float(top_y)
    y1 = float(top_y) + float(bullet_h)
    return [(float(x), y0 + y1 - float(y)) for x, y in points]


def draw_bullet_notehead(
    du: DrawUtil,
    symbol: SymbolUtil,
    *,
    x_mm: float,
    y_mm: float,
    direction: str,
    filled: bool,
    item_id: int,
    tags: list[str],
    stroke_color_override: tuple[float, float, float, float] | None = None,
    fill_color_override: tuple[float, float, float, float] | None = None,
) -> None:
    half_w = float(symbol.semitone_space_mm) * float(symbol.note_width_scaling)
    bullet_h = _bullet_height_mm(symbol)
    top_y = float(y_mm) - bullet_h if str(direction) == "up" else float(y_mm)

    stroke_color = stroke_color_override if stroke_color_override is not None else symbol.notation_color
    fill_color = fill_color_override if fill_color_override is not None else symbol.notation_color

    points = _down_bullet_points(float(x_mm), top_y, half_w, bullet_h, curve_points=32)
    # Inverted orientation by design, matching triangle behavior:
    # - "down" noteheads point upward
    # - "up" noteheads point downward
    if str(direction) == "down":
        points = _flip_vertical(points, top_y, bullet_h)

    if filled:
        du.add_polygon(
            points,
            stroke_color=stroke_color,
            stroke_width_mm=max(0.05, float(symbol.notehead_outline_width_mm) * 0.5),
            fill_color=fill_color,
            id=int(item_id),
            tags=list(tags),
        )
        return

    du.add_polygon(
        points,
        stroke_color=stroke_color,
        stroke_width_mm=symbol.notehead_outline_width_mm,
        fill_color=symbol.paper_color,
        id=int(item_id),
        tags=list(tags),
    )


def draw_bullet_left_dot(
    du: DrawUtil,
    symbol: "Notehead",
) -> None:
    hand = str(getattr(symbol, "hand", "") or "")
    if hand != "l":
        return
    if not bool(symbol._layout_value("note_leftdot_visible", False)):
        return

    bullet_h = _bullet_height_mm(symbol)
    dot_d = bullet_h * 0.3
    top_y = float(symbol.y_mm) - bullet_h if str(symbol.direction) == "up" else float(symbol.y_mm)
    cy = top_y + (bullet_h / 2.0)
    fill = symbol.paper_color if bool(symbol.filled) else symbol.notation_color
    du.add_oval(
        float(symbol.x_mm) - (dot_d / 3.0),
        cy - (dot_d / 3.0),
        float(symbol.x_mm) + (dot_d / 3.0),
        cy + (dot_d / 3.0),
        stroke_color=None,
        fill_color=fill,
        id=0,
        tags=["left_dot"],
    )
