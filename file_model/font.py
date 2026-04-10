from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Font:
    family: str = "Edwin"
    size_pt: float = 12.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    x_offset: float = 0.0
    y_offset: float = 0.0

    def resolve_family(self) -> str:
        try:
            from fonts import resolve_font_family
            return resolve_font_family(self.family)
        except Exception:
            return self.family
