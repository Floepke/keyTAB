

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Arpeggio:
    """Arpeggio marker attached to a chord.

    `note_pitches` stores the pitches of the chord member notes at `time`.
    Pitches are stable across save/load; session `_id` values are not.
    """

    time: float = 0.0
    rtime1: float = 0.0
    rtime2: float = 32.0
    note_pitches: list[int] = field(default_factory=list)
    _id: int = 0