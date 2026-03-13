

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ArpeggioType = Literal["up/ending", "down/ending", "up/starting", "down/starting"]


@dataclass
class Arpeggio:
    """Arpeggio marker attached to a chord.

    `notes` stores the chord's pitches (int) at the given `time`.
    """

    time: float = 0.0
    duration: float = 32.0
    notes: list[int] = field(default_factory=list)
    type: ArpeggioType = "up/starting"
    _id: int = 0