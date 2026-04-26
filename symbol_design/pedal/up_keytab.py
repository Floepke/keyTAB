from __future__ import annotations

from typing import Callable, Any

from file_model.events.pedal import Pedal
from ui.widgets.draw_util import DrawUtil


def _read_pedal_field(pedal: Pedal | dict, name: str, default):
    if isinstance(pedal, dict):
        return pedal.get(name, default)
    return getattr(pedal, name, default)


def draw_up_keytab(
    du: DrawUtil,
    pedal: Pedal | dict,
    *,
    time_to_y_mm: Callable[[float], float],
    rpitch_to_x_mm: Callable[[int], float],
    color: tuple[float, float, float, float] = (0, 0, 0, 1),
    background_color: tuple[float, float, float, float] = (1, 1, 1, 1),
    width_mm: float = 0.3,
    semitone_space_mm: float = 0.5,
    layout: Any | None = None,
    id: int | None = None,
    tags: list[str] | None = None,
) -> tuple[float, float]:
    """keyTAB pedal up symbol.

    Vertical mirror of the down symbol: triangle pointing upward.
    Apex at y1_mm - dy*2, horizontal bar at y1_mm.

    Returns the anchor (x_mm, y_mm) resolved from Pedal.time/Pedal.rpitch.
    """
    rpitch = int(_read_pedal_field(pedal, "rpitch", 0) or 0)
    p_time = float(_read_pedal_field(pedal, "time", 0.0) or 0.0)

    x1_mm = float(rpitch_to_x_mm(rpitch))
    y1_mm = float(time_to_y_mm(p_time))

    use_id = int(_read_pedal_field(pedal, "_id", 0) if id is None else id)
    use_tags = list(tags) if tags is not None else ["pedal_symbol", "pedal_up"]

    # Get pedal appearance settings from layout
    pedal_thickness_mm = max(0.05, float(width_mm))
    background_padding_mm = 0.25
    if layout is not None:
        try:
            if isinstance(layout, dict):
                background_padding_mm = float(layout.get('pedal_background_padding_mm', 0.25) or 0.0)
            else:
                background_padding_mm = float(getattr(layout, 'pedal_background_padding_mm', 0.25) or 0.0)
        except (TypeError, ValueError, AttributeError):
            background_padding_mm = 0.25
    background_padding_mm = max(0.0, float(background_padding_mm))

    dx = float(semitone_space_mm)
    dy = float(semitone_space_mm)
    if dx <= 0.0:
        dx = 0.5
    if dy <= 0.0:
        dy = 0.5

    # Symbol extents: apex at y1_mm - dy*2, base (horizontal bar) at y1_mm.
    x_min = float(x1_mm - dx*2)
    x_max = float(x1_mm + dx*2)
    y_min = float(y1_mm - dy*2)
    y_max = float(y1_mm)

    # Rounded background behind the symbol to improve legibility over grid/staves.
    pad = float(background_padding_mm)
    if pad > 0.0:
        du.add_rectangle(
            x_min - pad,
            y_min - pad,
            x_max + pad,
            y_max + pad,
            stroke_color=None,
            fill_color=background_color,
            corner_radius=pad,
            id=use_id,
            tags=use_tags + ["pedal_symbol_bg"],
        )

    # Triangle pointing up: bar at y1_mm, apex at y1_mm - dy*2
    du.add_polygon(
        [
            (x1_mm - dx*2, y1_mm),
            (x1_mm,        y1_mm - dy*2),
            (x1_mm + dx*2, y1_mm),
        ],
        stroke_color=color,
        stroke_width_mm=pedal_thickness_mm,
        fill_color=None,
        id=use_id,
        tags=use_tags + ["pedal_symbol"],
    )

    return x1_mm, y1_mm
