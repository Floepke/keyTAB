from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class Beam:
    time: float = 0.0
    duration: float = 256.0
    hand: Literal['l', 'r'] = 'l'
    _id: int = 0
