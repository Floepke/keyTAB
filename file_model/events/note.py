from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

NoteColor = Literal['auto'] | str

@dataclass
class Note:
    pitch: int = 40
    time: float = 0.0
    duration: float = 100.0
    velocity: int = 64
    hand: str = '<'
    '''
        Notehead types:
        in default mode:
            - white notes (abcdefg) use white noteheads
            - black notes (sharps/flats) use black noteheads 80% the width of white noteheads
                        - the notehead follows the black_note_rule ('below_stem', 'above_stem',
                            or 'above_stem_if_collision')
                in the layout section.
    '''
    notehead: Literal['auto',
                      # these are the available noteheads:
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
                      'cross_down'] = 'auto'
    color: NoteColor = 'auto'
    _id: int = 0
