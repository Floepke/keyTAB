from __future__ import annotations

from dataclasses import dataclass

from ui.widgets.draw_util import DrawUtil


@dataclass(frozen=True)
class TenutoSym:
    """Sharp-cornered vertical tenuto line centered on its note time."""

    x_mm: float
    y_mm: float
    color: tuple[float, float, float, float]

    def draw(
        self,
        du: DrawUtil,
        *,
        length_mm: float,
        thickness_mm: float,
        item_id: int = 0,
        tags: list[str] | None = None,
    ) -> None:
        length = max(0.05, float(length_mm))
        thickness = max(0.05, float(thickness_mm))
        du.add_rectangle(
            float(self.x_mm) - (thickness * 0.5),
            float(self.y_mm) - (length * 0.5),
            float(self.x_mm) + (thickness * 0.5),
            float(self.y_mm) + (length * 0.5),
            stroke_color=None,
            fill_color=self.color,
            id=int(item_id),
            tags=list(tags or []),
        )