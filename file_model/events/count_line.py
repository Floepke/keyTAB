from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CountLine:
    time: float = 0.0
    rpitch1: int = 0
    rpitch2: int = 4
    _id: int = 0
