from __future__ import annotations

from ui.widgets.draw_util import Color, DrawUtil


def draw_up_symbol(
    du: DrawUtil,
    *,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    thickness_mm: float,
    color: Color,
    paper_color: Color,
    item_id: int = 0,
    tags: list[str] | None = None,
) -> None:
    """Draw an upward pedal triangle anchored by its tip and bottom time edge."""
    half_width = max(0.05, float(width_mm)) * 0.5
    height = max(0.05, float(height_mm))
    tip = (float(x_mm), float(y_mm) - height)
    left_base = (float(x_mm) - half_width, float(y_mm))
    right_base = (float(x_mm) + half_width, float(y_mm))
    du.add_polygon(
        [left_base, tip, right_base, left_base],
        stroke_color=color,
        stroke_width_mm=max(0.05, float(thickness_mm)),
        fill_color=paper_color,
        id=int(item_id),
        tags=list(tags or []),
    )