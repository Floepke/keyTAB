from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Decrescendo:
    """
    Decrescendo hairpin: starts open at `time`, closes toward `time + duration`.
    x_rpitch is the horizontal position as semitone offset from C4 (key 40).
    Visual appearance is controlled by layout.hairpin_line_width_mm and layout.hairpin_spread_mm.
    """
    time: float = 0.0
    duration: float = 256.0
    x_rpitch: int = 0
    start_text: str = ""
    end_text: str = ""
    _id: int = 0
