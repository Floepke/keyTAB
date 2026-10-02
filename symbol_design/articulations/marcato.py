from __future__ import annotations

from dataclasses import dataclass

from ui.widgets.draw_util import DrawUtil


@dataclass(frozen=True)
class MarcatoSym:
    """Side-facing marcato chevron."""

    x_mm: float
    y_mm: float
    color: tuple[float, float, float, float]

    @staticmethod
    def half_width_mm(width_mm: float, thickness_mm: float) -> float:
        """Return the painted horizontal extent, including round stroke ends."""
        return (max(0.05, float(width_mm)) + max(0.05, float(thickness_mm))) * 0.5

    def draw(
        self,
        du: DrawUtil,
        *,
        hand: str,
        width_mm: float,
        height_mm: float,
        thickness_mm: float,
        item_id: int = 0,
        tags: list[str] | None = None,
    ) -> None:
        half_width = max(0.05, float(width_mm)) * 0.5
        half_height = max(0.05, float(height_mm)) * 0.5
        x_center = float(self.x_mm)
        y_center = float(self.y_mm)
        if str(hand or 'l') == 'l':
            points = [
                (x_center + half_width, y_center - half_height),
                (x_center - half_width, y_center),
                (x_center + half_width, y_center + half_height),
            ]
        else:
            points = [
                (x_center - half_width, y_center - half_height),
                (x_center + half_width, y_center),
                (x_center - half_width, y_center + half_height),
            ]
        du.add_polyline(
            points,
            stroke_color=self.color,
            stroke_width_mm=max(0.05, float(thickness_mm)),
            id=int(item_id),
            tags=list(tags or []),
        )