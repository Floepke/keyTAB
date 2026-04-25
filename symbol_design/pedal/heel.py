from __future__ import annotations

from typing import Callable, Any

from file_model.events.pedal import Pedal
from ui.widgets.draw_util import DrawUtil


def _read_pedal_field(pedal: Pedal | dict, name: str, default):
    if isinstance(pedal, dict):
        return pedal.get(name, default)
    return getattr(pedal, name, default)


def draw_heel(
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
    """Template entry-point for the 'heel' pedal symbol.

    Returns the anchor (x_mm, y_mm) resolved from Pedal.time/Pedal.rpitch.
    Replace the example du.add_line call with your final symbol geometry.
    """
    x0_mm = float(rpitch_to_x_mm(int(_read_pedal_field(pedal, "rpitch", 0) or 0)))
    y0_mm = float(time_to_y_mm(float(_read_pedal_field(pedal, "time", 0.0) or 0.0)))

    use_id = int(_read_pedal_field(pedal, "_id", 0) if id is None else id)
    use_tags = list(tags) if tags is not None else ["pedal_symbol", "pedal_heel"]

    # Example stroke from the anchor; edit this to design your symbol.
    du.add_line(
        x0_mm,
        y0_mm,
        x0_mm,
        y0_mm + 1.4,
        color=color,
        width_mm=float(width_mm),
        id=use_id,
        tags=use_tags,
    )

    return x0_mm, y0_mm
