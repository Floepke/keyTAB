from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SymbolUtil:
    """Shared layout/style context for symbol drawing helpers."""

    layout: Any
    semitone_space_mm: float
    notation_color: tuple[float, float, float, float]
    paper_color: tuple[float, float, float, float]

    def _layout_value(self, name: str, default):
        src = self.layout
        if isinstance(src, dict):
            return src.get(name, default)
        return getattr(src, name, default)

    @property
    def note_width_scaling(self) -> float:
        try:
            val = float(self._layout_value("note_width_scaling", 1.0) or 1.0)
        except Exception:
            val = 1.0
        return max(0.05, val)

    @property
    def note_stem_thickness_mm(self) -> float:
        try:
            stem = float(self._layout_value("note_stem_thickness_mm", 0.5) or 0.5)
        except Exception:
            stem = 0.5
        try:
            scale = float(self._layout_value("scale", 1.0) or 1.0)
        except Exception:
            scale = 1.0
        return max(0.05, stem * scale)

    @property
    def notehead_outline_width_mm(self) -> float:
        return self.note_stem_thickness_mm
