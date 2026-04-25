from __future__ import annotations

from typing import Callable, Literal, Any

from file_model.events.pedal import Pedal
from ui.widgets.draw_util import DrawUtil

from .down import draw_down
from .heel import draw_heel
from .toe import draw_toe
from .up import draw_up

PedalSymbol = Literal["down", "up", "toe", "heel"]


def draw_pedal_symbol(
    du: DrawUtil,
    pedal: Pedal,
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
    """Dispatch to the pedal-symbol template selected by pedal.symbol."""
    if isinstance(pedal, dict):
        raw_symbol = pedal.get("symbol", None)
        if raw_symbol is None:
            legacy_type = str(pedal.get("type", "") or "").strip().lower()
            raw_symbol = "up" if legacy_type in ("^", "up") else "down"
    else:
        raw_symbol = getattr(pedal, "symbol", "down")
    symbol = str(raw_symbol or "down").strip().lower()

    kwargs = {
        "time_to_y_mm": time_to_y_mm,
        "rpitch_to_x_mm": rpitch_to_x_mm,
        "color": color,
        "background_color": background_color,
        "width_mm": width_mm,
        "semitone_space_mm": semitone_space_mm,
        "layout": layout,
        "id": id,
        "tags": tags,
    }

    if symbol == "up":
        return draw_up(du, pedal, **kwargs)
    if symbol == "toe":
        return draw_toe(du, pedal, **kwargs)
    if symbol == "heel":
        return draw_heel(du, pedal, **kwargs)
    return draw_down(du, pedal, **kwargs)


__all__ = [
    "PedalSymbol",
    "draw_down",
    "draw_up",
    "draw_toe",
    "draw_heel",
    "draw_pedal_symbol",
]
