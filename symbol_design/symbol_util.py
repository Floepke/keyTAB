from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from file_model.layout import Layout


@dataclass
class SymbolUtil:
    """Shared layout/style context for symbol drawing helpers."""

    layout: Layout
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
        if getattr(self, 'black_note_narrow', False):
            try:
                narrow_scale = float(getattr(self, 'black_note_narrow_width_percent', 0.75) or 0.75)
            except Exception:
                narrow_scale = 0.75
            val *= max(0.05, narrow_scale)
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
        try:
            override = getattr(self, 'outline_width_mm_override', None)
            if override is not None:
                return max(0.05, float(override))
        except Exception:
            pass
        return self.note_stem_thickness_mm

    @property
    def notehead_height_scaling(self) -> float:
        try:
            val = float(self._layout_value("notehead_height_scaling", 1.0) or 1.0)
        except Exception:
            val = 1.0
        return max(0.1, val)

    @property
    def notehead_tilt(self) -> float:
        try:
            val = float(self._layout_value("notehead_tilt", 0.0) or 0.0)
        except Exception:
            val = 0.0
        return max(-1.0, min(1.0, val))
