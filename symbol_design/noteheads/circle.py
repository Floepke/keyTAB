from __future__ import annotations

from typing import TYPE_CHECKING

from ui.widgets.draw_util import DrawUtil
from symbol_design.noteheads.geometry import sheared_notehead_outline_points
from symbol_design.symbol_util import SymbolUtil
from utils.CONSTANT import BLACK_KEYS

if TYPE_CHECKING:
    from symbol_design.noteheads.notehead import Notehead
    from file_model.events.note import Note


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
    apply_tilt: bool = True,
) -> None:
    half_w = float(symbol.semitone_space_mm) * float(symbol.note_width_scaling)
    full_h = float(symbol.semitone_space_mm) * 2.0 * float(symbol.notehead_height_scaling)
    top_y = float(y_mm) - full_h if str(direction) == "up" else float(y_mm)

    stroke_color = stroke_color_override if stroke_color_override is not None else symbol.notation_color
    fill_color = fill_color_override if fill_color_override is not None else symbol.notation_color
    tilt = float(symbol.notehead_tilt) if apply_tilt else 0.0
    note_obj: Note = getattr(symbol, "note", None)
    hand = "l"
    if note_obj is not None:
        if isinstance(note_obj, dict):
            hand = str(note_obj.get("hand", "") or "").lower()
        else:
            hand = str(getattr(note_obj, "hand", "") or "").lower()

    if abs(tilt) <= 1e-9:
        if filled:
            du.add_oval(
                float(x_mm) - half_w,
                top_y,
                float(x_mm) + half_w,
                top_y + full_h,
                stroke_color=stroke_color,
                stroke_width_mm=max(0.05, float(symbol.notehead_outline_width_mm)),
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
        # Draw tilted notehead with cached local outline points, then translate.
        local_pts = sheared_notehead_outline_points(
            hand=hand,
            is_up=(str(direction) == "up"),
            semitone_space_mm=float(symbol.semitone_space_mm),
            width_scale=float(symbol.note_width_scaling),
            height_scale=float(symbol.notehead_height_scaling),
            # Hand-direction tilt sign is applied in geometry.py.
            base_tilt=float(symbol.notehead_tilt) if apply_tilt else 0.0,
            sample_count=64,
        )
        pts = [(float(x_mm) + float(px), float(y_mm) + float(py)) for (px, py) in local_pts]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        hit_x = min(xs)
        hit_y = min(ys)
        hit_rect = (hit_x, hit_y, max(xs) - hit_x, max(ys) - hit_y)
        if filled:
            du.add_polygon(
                pts,
                stroke_color=stroke_color,
                stroke_width_mm=max(0.05, float(symbol.notehead_outline_width_mm)),
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
