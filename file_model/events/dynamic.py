from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DynamicSymbol:
    """
    Standalone dynamic marking anchored by time and horizontal pitch-relative position.
    `x_rpitch` is the semitone offset from C4 (key 40).
    `symbol` stores a LelandText glyph.
    """
    time: float = 0.0
    x_rpitch: int = 0
    symbol: str = ""
    rotation: float = 0.0
    _id: int = 0
