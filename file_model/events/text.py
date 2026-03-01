from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from file_model.layout import LayoutFont

@dataclass
class Text:
    '''
        Represents a time bounded text element to be rendered on the score.
    '''
    text: str = 'myText'

    # position and rotation
    time: float = 0.0 # y coordinate uses time units (e.g., quarter note = 256.0)
    x_rpitch: float = 0 # x coordinate uses the relative distance from c4 position in semitone distances
    rotation: float = 0.0 # 0..360 degrees, clockwise
    x_offset_mm: float = 0.0
    y_offset_mm: float = 0.0
    
    # font settings
    font: LayoutFont = field(default_factory=lambda: LayoutFont(
        family="Latin Modern Roman Caps",
        size_pt=12.0,
        bold=False,
        italic=True,
    ))
    use_custom_font: bool = False
    _id: int = 0
