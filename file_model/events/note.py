from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from utils.CONSTANT import BLACK_KEYS, PIANO_KEY_AMOUNT

NoteColor = Literal['auto'] | str

@dataclass
class Note:
    pitch: int = 40
    time: float = 0.0
    duration: float = 100.0
    velocity: int = 64
    hand: str = 'l'
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
    # Compact accidental offset in semitones; valid range: -2..2.
    # 0 means no accidental marker.
    acc: int = 0
    _id: int = 0

    @staticmethod
    def is_valid_accidental(note: "Note | dict") -> bool:
        """Validate accidental notation for a note-like object.

        Rules:
        - `acc` must be one of -2, -1, 0, 1, 2.
        - derived pitch (`pitch + acc`) cannot be a black key.
        - derived pitch must remain within playable key range.
        """
        if isinstance(note, dict):
            acc = int(note.get('acc', 0) or 0)
        else:
            acc = int(getattr(note, 'acc', 0) or 0)
        
        if acc not in (-2, -1, 0, 1, 2):
            return False
        
        if isinstance(note, dict):
            pitch = int(note.get('pitch', 0) or 0)
        else:
            pitch = int(getattr(note, 'pitch', 0) or 0)
        
        derived_from = int(pitch + acc)
        if derived_from < 1 or derived_from > int(PIANO_KEY_AMOUNT):
            return False
        
        return int(derived_from) not in BLACK_KEYS
