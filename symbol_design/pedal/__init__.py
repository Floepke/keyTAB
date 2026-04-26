from __future__ import annotations

from typing import Callable, Literal, Any

from file_model.events.pedal import Pedal
from ui.widgets.draw_util import DrawUtil

from .down_keytab import draw_down_keytab
from .down_klavarskribo import draw_down_klavarskribo
from .heel import draw_heel
from .toe import draw_toe
from .up_keytab import draw_up_keytab
from .up_klavarskribo import draw_up_klavarskribo

PedalSymbol = Literal["down_keytab", "up_keytab", "down_klavarskribo", "up_klavarskribo", "toe", "heel"]


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
        rpitch = int(pedal.get("rpitch", 0) or 0)
        p_time = float(pedal.get("time", 0.0) or 0.0)
    else:
        raw_symbol = getattr(pedal, "symbol", None)
        rpitch = int(getattr(pedal, "rpitch", 0) or 0)
        p_time = float(getattr(pedal, "time", 0.0) or 0.0)
    symbol = str(raw_symbol or "").strip().lower()

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

    if symbol == "up_keytab":
        return draw_up_keytab(du, pedal, **kwargs)
    if symbol == "down_keytab":
        return draw_down_keytab(du, pedal, **kwargs)
    if symbol == "up_klavarskribo":
        return draw_up_klavarskribo(du, pedal, **kwargs)
    if symbol == "down_klavarskribo":
        return draw_down_klavarskribo(du, pedal, **kwargs)
    if symbol == "toe":
        return draw_toe(du, pedal, **kwargs)
    if symbol == "heel":
        return draw_heel(du, pedal, **kwargs)
    # Unknown/legacy symbols are intentionally ignored.
    return (float(rpitch_to_x_mm(rpitch)), float(time_to_y_mm(p_time)))


__all__ = [
    "PedalSymbol",
    "draw_down_keytab",
    "draw_up_keytab",
    "draw_down_klavarskribo",
    "draw_up_klavarskribo",
    "draw_toe",
    "draw_heel",
    "draw_pedal_symbol",
]
