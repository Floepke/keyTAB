from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class LineBreak:
    time: float = 0.0
    margin_mm: List[float] = field(default_factory=lambda: [5.0, 5.0]) # [left, right]
    # [lowest_key, highest_key] or 'auto' for automatic detection
    stave_range: List[int] | Literal['auto'] = 'auto'
    # Whether this line break indicates a page break or a line break
    page_break: bool = False
    _id: int = 0
