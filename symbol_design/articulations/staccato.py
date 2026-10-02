from __future__ import annotations

from dataclasses import dataclass

from ui.widgets.draw_util import DrawUtil


@dataclass(frozen=True)
class StaccatoSym:
    """Filled circular staccato articulation."""

    x_mm: float
    y_mm: float
    radius_mm: float
    color: tuple[float, float, float, float]

    def draw(self, du: DrawUtil, *, item_id: int = 0, tags: list[str] | None = None) -> None:
        radius = max(0.1, float(self.radius_mm))
        du.add_oval(
            float(self.x_mm) - radius,
            float(self.y_mm) - radius,
            float(self.x_mm) + radius,
            float(self.y_mm) + radius,
            stroke_color=None,
            fill_color=self.color,
            id=int(item_id),
            tags=list(tags or []),
        )