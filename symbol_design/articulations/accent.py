from __future__ import annotations

from dataclasses import dataclass

from ui.widgets.draw_util import DrawUtil


@dataclass(frozen=True)
class AccentSym:
    """Sharp-bottomed accent symbol with square upper ends."""

    x_mm: float
    y_mm: float
    color: tuple[float, float, float, float]

    @staticmethod
    def half_width_mm(height_span_mm: float) -> float:
        return max(0.05, float(height_span_mm)) * 0.5

    def draw(
        self,
        du: DrawUtil,
        *,
        height_span_mm: float,
        thickness_mm: float,
        item_id: int = 0,
        tags: list[str] | None = None,
    ) -> None:
        height_span = max(0.05, float(height_span_mm))
        thickness = max(0.05, float(thickness_mm))
        half_span = height_span * 0.5
        inset = min(thickness, half_span * 0.9)
        x_center = float(self.x_mm)
        y_center = float(self.y_mm)
        y_top = y_center - half_span
        y_bottom = y_center + half_span
        points = [
            (x_center - half_span, y_top),
            (x_center, y_bottom),
            (x_center + half_span, y_top),
            (x_center + half_span - inset, y_top),
            (x_center, y_bottom - inset),
            (x_center - half_span + inset, y_top),
        ]
        du.add_polygon(
            points,
            stroke_color=None,
            fill_color=self.color,
            id=int(item_id),
            tags=list(tags or []),
        )