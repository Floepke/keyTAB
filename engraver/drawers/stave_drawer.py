"""StaveDrawer: renders vertical stave lines and ledger lines for the keyboard."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.tiny_tool import key_class_filter
from utils.CONSTANT import PIANO_KEY_AMOUNT

if TYPE_CHECKING:
    from engraver.engraver2 import EngravingContext

class StaveDrawer:
    """Draw vertical stave lines (one per piano key) and ledger lines for out-of-range notes."""
    
    def __init__(self, context: EngravingContext):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = self.layout_data.get('scale', 1.0)
        
        # Self-cache for per-line stave drawing
        self._cache = context.drawer_caches.setdefault("stave", {})
    
    def draw(self) -> None:
        """Draw all staves and ledger lines for the current page and line."""
        page_lines = self.layout_data.get('page_lines', [])
        if not page_lines:
            return
        
        for line_index, line in enumerate(page_lines):
            self._draw_stave_line(line_index, line)
    
    def _draw_stave_line(self, line_index: int, line: dict) -> None:
        """Draw stave lines (vertical key columns) and ledger lines for a single system line."""
        # Check if staves are visible for this line
        if not bool(line.get('stave_visible', True)):
            return
        
        y1 = float(line.get('y_top', 0.0))
        y2_draw = float(line.get('y_bottom', 0.0))
        
        # Get layout parameters
        layout = self.layout_data.get('layout', {})
        scale = self.scale
        
        # Stave line thicknesses (in mm, scaled)
        stave_two_w = float(layout.get('stave_two_line_thickness_mm', 0.5) or 0.5) * scale
        stave_three_w = float(layout.get('stave_three_line_thickness_mm', 0.5) or 0.5) * scale
        stave_clef_w = float(layout.get('stave_clef_line_thickness_mm', 0.5) or 0.5) * scale
        stave_ledger_len = float(layout.get('stave_ledger_line_length_mm', 7.0) or 7.0) * scale
        
        # Clef line dash pattern (for F and A lines at 41, 43)
        from engraver.helpers import scaled_dash_pattern_with_default as _scaled_dash_pattern_with_default
        default_clef_dash_mm = list(getattr(__import__('file_model.layout', fromlist=['Layout']).Layout(), 
                                             'stave_clef_line_dash_pattern_mm', [3.0]) or [3.0])
        clef_dash = _scaled_dash_pattern_with_default(
            layout.get('stave_clef_line_dash_pattern_mm', default_clef_dash_mm),
            default_clef_dash_mm,
            scale,
        )
        
        # Get helper functions from line data
        _key_to_x = self.layout_data.get('key_to_x_for_line', {})[line_index] if isinstance(
            self.layout_data.get('key_to_x_for_line', {}), dict) else None
        semitone_mm = self.layout_data.get('semitone_mm', 1.0)
        
        if _key_to_x is None:
            # If helper not pre-computed, build from line geometry
            origin = float(line.get('origin', 0.0))
            line_x_start = float(line.get('line_x_start', 0.0))
            key_positions = self.layout_data.get('key_positions', {})
            
            def _key_to_x(key: int) -> float:
                return line_x_start + (float(key_positions.get(key, 0.0)) - origin)
        
        # Visible keys for this line (main stave range)
        visible_keys = list(line.get('visible_keys', []))
        natural_bound_left = int(line.get('natural_bound_left', line.get('range', [1, 88])[0]))
        natural_bound_right = int(line.get('natural_bound_right', line.get('range', [1, 88])[1]))
        
        if not visible_keys:
            visible_keys = [k for k in range(int(line.get('range', [1, 88])[0]), 
                                             int(line.get('range', [1, 88])[1]) + 1) if k in self.layout_data.get('line_keys', [])]
        
        # Track which ledger lines have been drawn (to avoid duplicates)
        ledger_drawn: set[tuple[int, int]] = set()
        
        # Draw special low register stave line (A#0, key 2)
        low_key_present = bool(line.get('low_key_left', False))
        a0_ledger_mode = bool(line.get('a0_ledger_mode', False))
        if low_key_present and not a0_ledger_mode:
            x_pos = _key_to_x(2)
            self.du.add_line(
                x_pos, y1,
                x_pos, y2_draw,
                color=self.notation_color,
                width_mm=stave_three_w,
                dash_pattern=None,
                id=0,
                tags=['stave'],
            )
        
        # Draw main stave lines (one vertical line per visible key)
        for key in visible_keys:
            if low_key_present and int(key) == 2:
                continue
            
            x_pos = _key_to_x(key)
            
            # Determine line thickness and dash pattern based on key type
            is_clef_line = key in (41, 43)  # F4 and A4
            is_three_line = key in key_class_filter('FGA')  # F, G, A lines
            
            if is_clef_line:
                width_mm = stave_clef_w
                dash = clef_dash
            elif is_three_line:
                width_mm = stave_three_w
                dash = None
            else:
                width_mm = stave_two_w
                dash = None
            
            self.du.add_line(
                x_pos, y1,
                x_pos, y2_draw,
                color=self.notation_color,
                width_mm=width_mm,
                dash_pattern=dash,
                id=0,
                tags=['stave'],
            )
        
        # Draw ledger lines for notes outside the visible stave range
        # (short vertical segments, drawn per-note in actual noteheads drawing,
        # but we can pre-draw full-span ledger groups here if manual range mode)
        manual_range = isinstance(line.get('stave_range'), list) and len(line.get('stave_range')) >= 2
        if manual_range:
            self._draw_manual_ledger_groups(line, natural_bound_left, natural_bound_right, 
                                           y1, y2_draw, _key_to_x, stave_two_w, stave_three_w, 
                                           stave_clef_w, clef_dash, stave_ledger_len, semitone_mm)
    
    def _draw_manual_ledger_groups(self, line: dict, nat_left: int, nat_right: int, y1: float, y2_draw: float,
                                   _key_to_x, stave_two_w: float, stave_three_w: float, stave_clef_w: float,
                                   clef_dash, stave_ledger_len: float, semitone_mm: float) -> None:
        """Draw complete ledger line groups (vertical staves) outside the natural stave range."""
        # This is called when manual stave range is active to draw ledger groups as full vertical lines
        # (short segments, not full system height)
        # Actual per-note ledger drawing happens in the notehead rendering phase
        pass  # Placeholder for now; per-note ledgers are drawn during notehead phase
