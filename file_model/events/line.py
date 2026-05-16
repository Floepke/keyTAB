from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Line:
    '''
        Line futures:
        - dashed lines with configurable dash pattern and phase
        - color
        - arrowheads at either end (configurable to none, left only, right only, or both)
        - zigzag lines with configurable zigzag amplitude and frequency
        # time is where on the timeline the line starts.
        # handles are the points that the line connects, in (time, rpitch) format.
        # the time field is always equal to the lowest handle time value.
    '''
    # POSITION
    time: float = 0.0 # lowest time value among the handles
    time1: float = 0.0
    time2: float = 0.0
    rpitch1: float = 0.0
    rpitch2: float = 0.0

    # VISUAL
    width_mm: float = 0.5
    dash_pattern_mm: list[float] = field(default_factory=lambda: [3.0])
    dash_phase_mm: float = 0.0 # 0..1. The phase is applied as an offset to the dash pattern, where 1.0 means a full cycle of the dash pattern.
    # color: 'auto' means the color uses the default notation color.
    # otherwise it tries to resolve hex color, falls back to notation color if invalid.
    color: Literal['auto'] | str = "auto"
    zigzag_type: Literal['triangle', 'sine'] | None = None
    zigzag_amp_semitone: float = 0.0 # in semitones (0.5 means half a semitone distance up and down, 1.0 means a full semitone up and down, etc.)
    zigzag_freq: float = 64.0 # only for zigzag lines, in time (256.0 == quarter note length)
    arrow: Literal['left', 'right', 'both'] | None = None
    arrow_form: list[float] = field(default_factory=lambda: [0.5, 0.5]) # [length_mm, width_mm]

    # INTERNAL
    _id: int = 0