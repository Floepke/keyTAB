from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Tempo:
    '''
        Tempo change event. Tempo is in units per minute.

        Unit is defined in CONSTANT.py as QUARTER_NOTE_UNIT (256.0 by default).
        So a tempo of 120 with a duration of 256.0 ticks means 120 quarter notes per minute.
    '''
    time: float = 0.0        # start time in ticks
    duration: float = 256.0  # duration in ticks
    tempo: int = 120         # units per minute
    _id: int = 0
