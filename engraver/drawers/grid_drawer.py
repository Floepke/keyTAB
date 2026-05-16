# grid_drawer.py
# Ported grid and bar line logic from engraver.py for the new drawer pipeline.
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator
from file_model.base_grid import resolve_grid_layer_offsets

from engraver.helpers import time_to_y


class GridDrawer:
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.layout = self.layout_data.get('layout', {})
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        self.op_time = Operator(SHORTEST_DURATION)
        self.notation_color = self.layout_data.get('notation_color')
        self.grid_left = 0.0
        self.grid_right = 0.0
        self.line_index = 0
        self.line = None
        self.norm_notes = []

        self.bar_width_mm = float(self.layout.get('grid_barline_thickness_mm', 0.25) or 0.25) * self.scale
        self.grid_width_mm = float(self.layout.get('grid_gridline_thickness_mm', 0.15) or 0.15) * self.scale
        self.barline_visible = bool(self.layout.get('barline_visible', True))
        self.grid_line_visible = bool(self.layout.get('grid_line_visible', True))
        self.dash_pattern = self.layout.get('grid_gridline_dash_pattern_mm', [2.5, 4.0])
        self.dash_pattern = [float(x) * self.scale for x in self.dash_pattern]
        self.note_head_half_w = self.semitone_mm * float(self.layout.get('note_width_scaling', 0.75) or 0.75)
        self.stem_len_mm = float(self.layout.get('note_stem_length_semitone', 3) or 3) * self.semitone_mm
        self.stem_collision_pad = max(0.15, float(self.layout.get('note_stem_thickness_mm', 0.5) or 0.5) * self.scale)
        self.head_collision_pad = max(0.15, self.semitone_mm * 0.15)
        self.barline_symbol_gap_mm = max(0.0, self.semitone_mm)

    def draw(self):
        page_lines = list(self.layout_data.get('page_lines', []) or [])
        if not page_lines:
            return

        score = self.context.score or {}
        base_grid = list(score.get('base_grid', []) or [])
        if not base_grid:
            return

        for line_index, line in enumerate(page_lines):
            self.line_index = int(line_index)
            self.line = line
            self.norm_notes = list(line.get('notes', []) or [])

            self.grid_left = self._key_to_x(line_index, int(line.get('natural_bound_left', line.get('bound_left', 1))))
            self.grid_right = self._key_to_x(line_index, int(line.get('natural_bound_right', line.get('bound_right', 88))))
            if self.grid_right < self.grid_left:
                self.grid_left, self.grid_right = self.grid_right, self.grid_left

            line_start_ticks = float(line.get('time_start', 0.0) or 0.0)
            line_end_ticks = float(line.get('time_end', line_start_ticks) or line_start_ticks)
            self.draw_lines_from_base_grid(base_grid, line_start_ticks, line_end_ticks)

    def _key_to_x(self, line_index, key):
        pitch_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        pitch_to_x = pitch_to_x_map.get(line_index) if isinstance(pitch_to_x_map, dict) else None
        if callable(pitch_to_x):
            return float(pitch_to_x(int(key)))

        key_to_x_map = self.layout_data.get('key_to_x_for_line', {})
        key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
        if callable(key_to_x):
            return float(key_to_x(int(key)))

        return 0.0


    def _time_to_y(self, ticks):
        return time_to_y(self.line, ticks)

    def _merge_intervals(self, intervals):
        if not intervals:
            return []
        clipped = []
        for a, b in intervals:
            x0 = max(float(self.grid_left), min(float(self.grid_right), float(min(a, b))))
            x1 = max(float(self.grid_left), min(float(self.grid_right), float(max(a, b))))
            if x1 <= x0:
                continue
            clipped.append((x0, x1))
        if not clipped:
            return []
        clipped.sort(key=lambda it: it[0])
        merged = [clipped[0]]
        for a, b in clipped[1:]:
            la, lb = merged[-1]
            if a <= lb:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    def _barline_cut_intervals(self, ticks):
        intervals = []
        for item in self.norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            if not self.op_time.eq(n_t, float(ticks)):
                continue
            p = int(item.get('pitch', 0) or 0)
            x_note = self._key_to_x(self.line_index, p)
            intervals.append((
                x_note - self.note_head_half_w - self.head_collision_pad - self.barline_symbol_gap_mm,
                x_note + self.note_head_half_w + self.head_collision_pad + self.barline_symbol_gap_mm,
            ))
            if bool(self.layout.get('note_stem_visible', True)):
                hand_key = str(item.get('hand', 'l') or 'l')
                x_stem_tip = x_note - self.stem_len_mm if hand_key == 'l' else x_note + self.stem_len_mm
                intervals.append((
                    min(x_note, x_stem_tip) - self.stem_collision_pad - self.barline_symbol_gap_mm,
                    max(x_note, x_stem_tip) + self.stem_collision_pad + self.barline_symbol_gap_mm,
                ))
        return self._merge_intervals(intervals)

    def _draw_barline_segments(self, yb, cuts, width_mm, tags, item_id=0, dash_pattern=None):
        if not cuts:
            self.du.add_line(
                self.grid_left, yb, self.grid_right, yb,
                color=self.notation_color, width_mm=width_mm, id=item_id, tags=tags, dash_pattern=dash_pattern,
            )
            return
        x_cursor_seg = float(self.grid_left)
        min_seg = max(0.05, width_mm * 0.5)
        for c0, c1 in cuts:
            if c0 - x_cursor_seg > min_seg:
                self.du.add_line(
                    x_cursor_seg, yb, c0, yb,
                    color=self.notation_color, width_mm=width_mm, id=item_id, tags=tags, dash_pattern=dash_pattern,
                )
            x_cursor_seg = max(x_cursor_seg, c1)
        if float(self.grid_right) - x_cursor_seg > min_seg:
            self.du.add_line(
                x_cursor_seg, yb, self.grid_right, yb,
                color=self.notation_color, width_mm=width_mm, id=item_id, tags=tags, dash_pattern=dash_pattern,
            )

    def draw_barline(self, ticks):
        yb = self._time_to_y(float(ticks))
        cuts = self._barline_cut_intervals(float(ticks))
        self._draw_barline_segments(yb, cuts, self.bar_width_mm, ['barline'], 0)

    def draw_gridline(self, ticks):
        yb = self._time_to_y(float(ticks))
        cuts = self._barline_cut_intervals(float(ticks))
        self._draw_barline_segments(yb, cuts, self.grid_width_mm, ['grid_line'], 0, dash_pattern=self.dash_pattern)

    def draw_lines_from_base_grid(self, base_grid, line_start_ticks, line_end_ticks):
        op_time = self.op_time
        time_cursor = 0.0
        for bg in base_grid:
            numerator = int(bg.get('numerator', 4) or 4)
            denominator = int(bg.get('denominator', 4) or 4)
            measure_amount = int(bg.get('measure_amount', 1) or 1)
            beat_grouping = list(bg.get('beat_grouping', []) or [])
            bar_offsets, grid_offsets = resolve_grid_layer_offsets(beat_grouping, numerator, denominator)
            if measure_amount <= 0:
                continue
            measure_len = float(numerator) * (4.0 / float(max(1, denominator))) * float(QUARTER_NOTE_UNIT)
            for _ in range(measure_amount):
                if op_time.gt(time_cursor, float(line_end_ticks)):
                    break
                for off in bar_offsets:
                    t = float(time_cursor + float(off))
                    if op_time.lt(t, float(line_start_ticks)) or op_time.gt(t, float(line_end_ticks)):
                        continue
                    if not self.barline_visible:
                        continue
                    self.draw_barline(t)
                for off in grid_offsets:
                    t = float(time_cursor + float(off))
                    if op_time.lt(t, float(line_start_ticks)) or op_time.gt(t, float(line_end_ticks)):
                        continue
                    if not self.grid_line_visible:
                        continue
                    self.draw_gridline(t)
                time_cursor += measure_len
            if op_time.gt(time_cursor, float(line_end_ticks)):
                break
