"""StaveDrawer: renders vertical stave lines and ledger lines for the keyboard."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.tiny_tool import key_class_filter
from utils.CONSTANT import PIANO_KEY_AMOUNT, SHORTEST_DURATION, BLACK_KEYS, hex_to_rgba
from utils.operator import Operator
from symbol_design.noteheads import resolve_notehead_spec
from engraver.helpers import black_note_above_stem, normalize_hex_color

if TYPE_CHECKING:
    from engraver.engraver2 import EngravingContext

class StaveDrawer:
    op = Operator(SHORTEST_DURATION)
    """Draw vertical black key stave lines based on current line range and layout settings, as well as ledger lines for notes outside the line stave range."""
    
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

    def _pitch_to_x(self, line_index: int, line: dict, pitch: int):
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
        if callable(key_to_x):
            return float(key_to_x(int(pitch)))

        key_to_x_map = self.layout_data.get('key_to_x_for_line', {})
        key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
        if callable(key_to_x):
            return float(key_to_x(int(pitch)))

        origin = float(line.get('origin', 0.0))
        line_x_start = float(line.get('line_x_start', 0.0))
        key_positions = self.layout_data.get('key_positions', {})
        return line_x_start + (float(key_positions.get(int(pitch), 0.0)) - origin)

    def _rpitch_to_x(self, line_index: int, line: dict, rpitch: float):
        rpitch_to_x_map = self.layout_data.get('rpitch_to_x_for_line', {})
        rpitch_to_x = rpitch_to_x_map.get(line_index) if isinstance(rpitch_to_x_map, dict) else None
        if callable(rpitch_to_x):
            return float(rpitch_to_x(float(rpitch)))

        semitone_mm = float(self.layout_data.get('semitone_mm', 1.0) or 1.0)
        return self._pitch_to_x(line_index, line, 40) + (float(rpitch) * semitone_mm)

    def _time_to_y(self, line_index: int, line: dict, ticks: float):
        time_to_y_map = self.layout_data.get('time_to_y_for_line', {})
        time_to_y = time_to_y_map.get(line_index) if isinstance(time_to_y_map, dict) else None
        if callable(time_to_y):
            return float(time_to_y(float(ticks)))

        t0 = float(line.get('time_start', 0.0) or 0.0)
        t1 = float(line.get('time_end', t0) or t0)
        y0 = float(line.get('y_top', 0.0) or 0.0)
        y1 = float(line.get('y_bottom', y0) or y0)
        denom = max(1e-6, t1 - t0)
        rel = max(0.0, min(1.0, (float(ticks) - t0) / denom))
        return y0 + ((y1 - y0) * rel)
    
    def _draw_stave_line(self, line_index: int, line: dict) -> None:
        """Draw stave lines (vertical key columns) and ledger lines for a single system line."""
        # Match legacy behavior: global layout setting controls stave visibility.
        if not bool(self.layout_data.get('layout', {}).get('stave_visible', True)):
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
        if not visible_keys:
            line_keys = list(self.layout_data.get('line_keys', []))
            line_range = list(line.get('range', [1, PIANO_KEY_AMOUNT]) or [1, PIANO_KEY_AMOUNT])
            range_lo = int(line_range[0]) if len(line_range) >= 1 else 1
            range_hi = int(line_range[1]) if len(line_range) >= 2 else PIANO_KEY_AMOUNT
            visible_keys = [k for k in range(range_lo, range_hi + 1) if k in line_keys]

        # natural bounds for ledger line drawing (notes outside these bounds get ledger lines)
        natural_bound_left = int(line.get('natural_bound_left', line.get('range', [1, 88])[0]))
        natural_bound_right = int(line.get('natural_bound_right', line.get('range', [1, 88])[1]))
        
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
        fga_keys = key_class_filter('FGA')
        for key in visible_keys:
            if low_key_present and int(key) == 2:
                continue
            
            x_pos = _key_to_x(key)
            
            # Determine line thickness and dash pattern based on key type
            is_clef_line = key in (41, 43)  # F4 and A4
            is_three_line = key in fga_keys  # F, G, A lines
            
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
        
        # Draw per-note ledger groups whenever notes fall outside natural bounds.
        self._draw_ledger_line_groups(line_index, line, natural_bound_left, natural_bound_right,
                                      y1, y2_draw, _key_to_x, stave_two_w, stave_three_w,
                                      stave_clef_w, clef_dash, stave_ledger_len, semitone_mm)



    
    def _draw_ledger_line_groups(self, line_index: int, line: dict, nat_left: int, nat_right: int, y1: float, y2_draw: float,
                                 _key_to_x, stave_two_w: float, stave_three_w: float, stave_clef_w: float,
                                 clef_dash, stave_ledger_len: float, semitone_mm: float) -> None:
        """Draw complete ledger line groups (vertical staves) outside the natural stave range."""
        if not bool(self.layout_data.get('layout', {}).get('stave_visible', True)):
            return

        stave_line_groups = list(self.layout_data.get('stave_line_groups', []) or [])
        if not stave_line_groups:
            return

        def _group_index_for_key(key: int) -> int:
            for idx, grp in enumerate(stave_line_groups):
                lo = int(grp.get('range_low', 1) or 1)
                hi = int(grp.get('range_high', PIANO_KEY_AMOUNT) or PIANO_KEY_AMOUNT)
                if lo <= int(key) <= hi:
                    return idx
            first_lo = int(stave_line_groups[0].get('range_low', 1) or 1)
            return 0 if int(key) <= first_lo else len(stave_line_groups) - 1

        bound_group_low = _group_index_for_key(int(nat_left))
        bound_group_high = _group_index_for_key(int(nat_right))
        fga_keys = set(key_class_filter('FGA'))
        layout = self.layout_data.get('layout', {})
        black_rule = str(layout.get('black_note_rule', 'below_stem') or 'below_stem')
        note_height_scale = max(0.1, float(layout.get('notehead_height_scaling', 1.0) or 1.0))
        notehead_h = float(semitone_mm) * 2.0 * note_height_scale

        line_notes = list(line.get('notes', []) or [])
        if not line_notes:
            return

        notes_sorted = sorted(line_notes, key=lambda n: float(n.get('time', 0.0) or 0.0))
        time_groups: list[dict] = []
        note_group_idx: dict[int, int] = {}
        for note in notes_sorted:
            note_time = float(note.get('time', 0.0) or 0.0)
            pitch_value = int(note.get('pitch', 0) or 0)
            if (not time_groups) or (not self.op.eq(note_time, float(time_groups[-1]['time']))):
                time_groups.append({'time': note_time, 'has_black': False, 'has_white': False})
            grp = time_groups[-1]
            if pitch_value in BLACK_KEYS:
                grp['has_black'] = True
            else:
                grp['has_white'] = True
            note_group_idx[id(note)] = len(time_groups) - 1

        mixed_black_white = [bool(grp['has_black']) and bool(grp['has_white']) for grp in time_groups]

        segments: dict[tuple[int, int], dict] = {}

        def _accumulate_segment(group_idx: int, key_i: int, seg_start: float, seg_end: float, width_mm: float, dash_pattern):
            seg_key = (int(group_idx), int(key_i))
            cur = segments.get(seg_key)
            if cur is None:
                segments[seg_key] = {
                    'start': float(seg_start),
                    'end': float(seg_end),
                    'width_mm': float(width_mm),
                    'dash': dash_pattern,
                }
                return
            if float(seg_start) < float(cur['start']):
                cur['start'] = float(seg_start)
            if float(seg_end) > float(cur['end']):
                cur['end'] = float(seg_end)

        for note in line_notes:
            pitch_value = int(note.get('pitch', 0) or 0)
            note_time = float(note.get('time', 0.0) or 0.0)
            if pitch_value < 1 or pitch_value > PIANO_KEY_AMOUNT:
                continue

            group_idx = int(note_group_idx.get(id(note), 0))

            note_y = self._time_to_y(line_index, line, note_time)
            default_black_above = bool(
                pitch_value in BLACK_KEYS and black_note_above_stem(note, black_rule, line_notes)
            )
            spec = resolve_notehead_spec(note.get('raw', {}) or {}, default_black_above=default_black_above)
            is_up = bool(getattr(spec, 'is_up', False))

            note_top = float(note_y - notehead_h) if is_up else float(note_y)
            note_bottom = float(note_y) if is_up else float(note_y + notehead_h)

            y_seg_start = float(note_top - float(semitone_mm))
            y_seg_end = float(note_bottom + float(semitone_mm))

            if 0 <= group_idx < len(mixed_black_white) and mixed_black_white[group_idx]:
                # Mixed white/black chords need one extra head-height of ledger span.
                half_expand = float(notehead_h) * 0.5
                y_seg_start -= half_expand
                y_seg_end += half_expand

            if int(pitch_value) in (1, 2, 3) and bool(line.get('a0_ledger_mode', False)):
                _accumulate_segment(group_idx, 2, y_seg_start, y_seg_end, stave_three_w, None)

            ledger_groups: list[dict] = []
            if self.op.less(pitch_value, nat_left):
                g_start = _group_index_for_key(pitch_value)
                g_end = int(bound_group_low) - 1
                if g_start <= g_end:
                    ledger_groups = stave_line_groups[g_start:g_end + 1]
            elif self.op.greater(pitch_value, nat_right):
                g_start = int(bound_group_high) + 1
                g_end = _group_index_for_key(pitch_value)
                if g_start <= g_end:
                    ledger_groups = stave_line_groups[g_start:g_end + 1]

            if not ledger_groups:
                continue

            for grp in ledger_groups:
                for key in grp.get('keys', []):
                    key_i = int(key)
                    is_clef_line = key_i in (41, 43)
                    is_three_line = key_i in fga_keys
                    if is_clef_line:
                        width_mm = stave_clef_w
                        dash = clef_dash
                    elif is_three_line:
                        width_mm = stave_three_w
                        dash = None
                    else:
                        width_mm = stave_two_w
                        dash = None

                    _accumulate_segment(group_idx, key_i, y_seg_start, y_seg_end, width_mm, dash)

        for (_group_idx, key_i), seg in segments.items():
            x_pos = _key_to_x(int(key_i))
            self.du.add_line(
                x_pos,
                float(seg['start']),
                x_pos,
                float(seg['end']),
                color=self.notation_color,
                width_mm=float(seg['width_mm']),
                dash_pattern=seg['dash'],
                id=0,
                tags=['stave'],
            )
