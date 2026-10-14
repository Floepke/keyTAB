from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Volta:
    time: float = 0.0
    duration: float = 256.0
    number: int = 1
    x_offset: float = 0.0
    _id: int = 0