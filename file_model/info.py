from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Info:
    title: str = "Title"
    composer: str = "Composer"
    copyright: str = f"© keyTAB {datetime.now().year}"
    arranger: str = ""
    lyricist: str = ""
    comment: str = ""
