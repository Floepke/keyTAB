from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AppState:
    """Per-project UI/app state persisted in .piano files."""
    zoom_mm_per_quarter: float = 25.0
    x_zoom_factor: float = 1.0
    print_view_page_index: int = 0
    editor_scroll_pos: int = 0
    snap_base: int = 8
    snap_divide: int = 1
    selected_tool: str = "note"
    selected_stave_index: int = 0
    style_dialog_tab_index: int = 0
