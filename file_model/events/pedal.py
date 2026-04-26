from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class Pedal:
    time: float = 0.0
    rpitch: int = 0
    symbol: Literal['down_keytab', 'up_keytab', 'down_klavarskribo', 'up_klavarskribo', 'toe', 'heel'] = 'down_keytab'
    invisible: bool = False
    _id: int = 0
