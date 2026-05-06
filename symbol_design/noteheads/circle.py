from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ui.widgets.draw_util import DrawUtil
from symbol_design.symbol_util import SymbolUtil
from utils.CONSTANT import BLACK_KEYS

if TYPE_CHECKING:
    from symbol_design.noteheads.notehead import Notehead
    from file_model.events.note import Note


def _sheared_oval_points(
    cx: float, cy: float, rx: float, ry: float, shear: float, n: int = 64
) -> list[tuple[float, float]]:
    """Return polygon points for a vertically sheared oval.

    A vertical shear shifts y proportional to x: y' = y + shear * x.
    Applied to an ellipse this produces the traditional tilted notehead shape.
    """
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t) + shear * rx * math.cos(t)
        pts.append((x, y))
    return pts


def draw_circle_notehead(
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
    full_h = float(symbol.semitone_space_mm) * 2.0 * float(symbol.notehead_height_scaling)
    top_y = float(y_mm) - full_h if str(direction) == "up" else float(y_mm)

    stroke_color = stroke_color_override if stroke_color_override is not None else symbol.notation_color
    fill_color = fill_color_override if fill_color_override is not None else symbol.notation_color
    tilt = float(symbol.notehead_tilt)
    note_obj: Note = getattr(symbol, "note", None)
    if note_obj is not None:
        # set tilt
        if isinstance(note_obj, dict):
            hand = str(note_obj.get("hand", "") or "").lower()
        else:
            hand = str(getattr(note_obj, "hand", "") or "").lower()
        if hand == "r":
            tilt = -tilt
        else:
            tilt = tilt

    if abs(tilt) <= 1e-9:
        if filled:
            du.add_oval(
                float(x_mm) - half_w,
                top_y,
                float(x_mm) + half_w,
                top_y + full_h,
                stroke_color=stroke_color,
                stroke_width_mm=max(0.05, float(symbol.notehead_outline_width_mm) * 0.5),
                fill_color=fill_color,
                id=int(item_id),
                tags=list(tags),
            )
        else:
            du.add_oval(
                float(x_mm) - half_w,
                top_y,
                float(x_mm) + half_w,
                top_y + full_h,
                stroke_color=stroke_color,
                stroke_width_mm=symbol.notehead_outline_width_mm,
                fill_color=symbol.paper_color,
                id=int(item_id),
                tags=list(tags),
            )
    else:
        # draw tilted notehead with polygon points
        cx = float(x_mm)
        cy = top_y + full_h / 2.0
        pts = _sheared_oval_points(cx, cy, half_w, full_h / 2.0, tilt)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        hit_x = min(xs)
        hit_y = min(ys)
        hit_rect = (hit_x, hit_y, max(xs) - hit_x, max(ys) - hit_y)
        if filled:
            du.add_polygon(
                pts,
                stroke_color=stroke_color,
                stroke_width_mm=max(0.05, float(symbol.notehead_outline_width_mm) * 0.5),
                fill_color=fill_color,
                id=int(item_id),
                tags=list(tags),
                hit_rect_mm=hit_rect,
            )
        else:
            du.add_polygon(
                pts,
                stroke_color=stroke_color,
                stroke_width_mm=symbol.notehead_outline_width_mm,
                fill_color=symbol.paper_color,
                id=int(item_id),
                tags=list(tags),
                hit_rect_mm=hit_rect,
            )


def draw_circle_left_dot(
    du: DrawUtil,
    symbol: "Notehead",
) -> None:
    hand = str(getattr(symbol, "hand", "") or "")
    if hand != "l":
        return
    if not bool(symbol._layout_value("note_leftdot_visible", False)):
        return

    full_h = float(symbol.semitone_space_mm) * 2.0 * float(symbol.notehead_height_scaling)
    dot_d = full_h * 0.3
    top_y = float(symbol.y_mm) - full_h if str(symbol.direction) == "up" else float(symbol.y_mm)
    cy = top_y + (full_h / 2.0)
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
