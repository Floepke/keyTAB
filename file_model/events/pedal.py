from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class Pedal:
    time: float = 0.0
    rpitch: int = 0
    symbol: Literal['down', 'up', 'toe', 'heel'] = 'down'
    _id: int = 0
