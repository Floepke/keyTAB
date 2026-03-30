from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


GraceNoteheadLiteral = Literal[
    'auto',
    'circle_white_up',
    'circle_white_down',
    'circle_black_up',
    'circle_black_down',
    'bullet_white_up',
    'bullet_white_down',
    'bullet_black_up',
    'bullet_black_down',
    'triangle_white_up',
    'triangle_white_down',
    'triangle_black_up',
    'triangle_black_down',
    'cross_up',
    'cross_down',
]

@dataclass
class GraceNote:
    pitch: int = 41
    time: float = 50.0
    notehead: GraceNoteheadLiteral = 'auto'
    _id: int = 0
