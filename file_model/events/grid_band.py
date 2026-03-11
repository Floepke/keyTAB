from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class GridBand:
    """Represents a grid band marker with time and duration."""
    time: float = 0.0
    duration: float = 256.0
    _id: int = 0
