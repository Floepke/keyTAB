from __future__ import annotations

from typing import Callable

from file_model.events.pedal import Pedal
from ui.widgets.draw_util import DrawUtil


def draw_toe(
    du: DrawUtil,
    pedal: Pedal,
    *,
    time_to_y_mm: Callable[[float], float],
    rpitch_to_x_mm: Callable[[int], float],
    color: tuple[float, float, float, float] = (0, 0, 0, 1),
    width_mm: float = 0.3,
    id: int | None = None,
    tags: list[str] | None = None,
) -> tuple[float, float]:
    """Template entry-point for the 'toe' pedal symbol.

    Returns the anchor (x_mm, y_mm) resolved from Pedal.time/Pedal.rpitch.
    Replace the example du.add_line call with your final symbol geometry.
    """
    x0_mm = float(rpitch_to_x_mm(int(pedal.rpitch)))
    y0_mm = float(time_to_y_mm(float(pedal.time)))

    use_id = int(pedal._id if id is None else id)
    use_tags = list(tags) if tags is not None else ["pedal_symbol", "pedal_toe"]

    # Example stroke from the anchor; edit this to design your symbol.
    du.add_line(
        x0_mm,
        y0_mm,
        x0_mm + 1.4,
        y0_mm,
        color=color,
        width_mm=float(width_mm),
        id=use_id,
        tags=use_tags,
    )

    return x0_mm, y0_mm
