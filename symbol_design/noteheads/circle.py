from __future__ import annotations

from typing import TYPE_CHECKING

from ui.widgets.draw_util import DrawUtil
from symbol_design.symbol_util import SymbolUtil

if TYPE_CHECKING:
    from symbol_design.noteheads.notehead import Notehead


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
    full_h = float(symbol.semitone_space_mm) * 2.0
    top_y = float(y_mm) - full_h if str(direction) == "up" else float(y_mm)
    
    stroke_color = stroke_color_override if stroke_color_override is not None else symbol.notation_color
    fill_color = fill_color_override if fill_color_override is not None else symbol.notation_color

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
        return

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


def draw_circle_left_dot(
    du: DrawUtil,
    symbol: "Notehead",
) -> None:
    if str(getattr(symbol, "hand", "l") or "l") != "l":
        return
    if not bool(symbol._layout_value("note_leftdot_visible", False)):
        return

    full_h = float(symbol.semitone_space_mm) * 2.0
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
