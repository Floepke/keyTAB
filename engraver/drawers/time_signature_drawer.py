"""TimeSignatureDrawer: renders time signature indicators."""

from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator
from file_model.base_grid import resolve_grid_layer_offsets
from engraver.helpers import time_to_y


class TimeSignatureDrawer:
    """Draw time signature indicators."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all time signature indicators for the current page and line."""
        score = self.context.score or {}
        base_grid = list(score.get('base_grid', []) or [])
        if not base_grid:
            return

        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('time_signature_visible', True)):
            return

        indicator_type = str(layout.get('time_signature_indicator_type', 'classical') or 'classical')
        lane_w = float(layout.get('time_signature_indicator_lane_width_mm', 35.0) or 35.0) * self.scale
        guide_thickness = float(layout.get('time_signature_indicator_guide_thickness_mm', 0.5) or 0.5) * self.scale
        divider_thickness = float(layout.get('time_signature_indicator_divide_guide_thickness_mm', 1.0) or 1.0) * self.scale
        classic_font = dict(layout.get('time_signature_indicator_classic_font', {}) or {})
        klav_font = dict(layout.get('time_signature_indicator_klavarskribo_font', {}) or {})
        classic_family = str(classic_font.get('family', 'Edwin') or 'Edwin')
        klav_family = str(klav_font.get('family', 'Edwin') or 'Edwin')
        classic_size = float(classic_font.get('size_pt', 35.0) or 35.0) * self.scale
        klav_size = float(klav_font.get('size_pt', 25.0) or 25.0) * self.scale
        classic_bold = bool(classic_font.get('bold', True))
        classic_italic = bool(classic_font.get('italic', False))
        klav_bold = bool(klav_font.get('bold', True))
        klav_italic = bool(klav_font.get('italic', False))
        half_span = 3.0 * self.scale

        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue

            right_edge = float(key_to_x(int(line.get('natural_bound_right', line.get('bound_right', 88)))))
            ts_x_right = right_edge + (lane_w * (2.0 / 3.0))
            ts_x_mid = right_edge + (lane_w * (1.0 / 3.0))
            ts_x_left = right_edge

            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)
            tick_per_mm = max(1e-6, float(line.get('tick_per_mm', 1.0) or 1.0))
            mm_per_quarter = float(QUARTER_NOTE_UNIT) / tick_per_mm

            cursor = 0.0
            for bg in base_grid:
                numer = int(bg.get('numerator', 4) or 4)
                denom = int(bg.get('denominator', 4) or 4)
                measures = int(bg.get('measure_amount', 1) or 1)
                beat_grouping = list(bg.get('beat_grouping', []) or [])
                enabled = bool(bg.get('indicator_enabled', True))
                if measures <= 0:
                    continue
                measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                for _ in range(measures):
                    t = float(cursor)
                    cursor += measure_len_ticks
                    if self.op.lt(t, lt0) or self.op.ge(t, lt1):
                        continue
                    if not enabled:
                        continue
                    y = float(time_to_y(line, t))

                    if indicator_type in ('classical', 'classical & klavarskribo'):
                        self.du.add_text(
                            ts_x_right,
                            y - half_span,
                            str(numer),
                            family=classic_family,
                            size_pt=classic_size,
                            bold=classic_bold,
                            italic=classic_italic,
                            color=self.notation_color,
                            anchor='center',
                            id=0,
                            tags=['ts_classic'],
                        )
                        self.du.add_line(
                            ts_x_right - half_span,
                            y,
                            ts_x_right + half_span,
                            y,
                            color=self.notation_color,
                            width_mm=divider_thickness,
                            id=0,
                            tags=['ts_classic'],
                        )
                        self.du.add_text(
                            ts_x_right,
                            y + half_span,
                            str(denom),
                            family=classic_family,
                            size_pt=classic_size,
                            bold=classic_bold,
                            italic=classic_italic,
                            color=self.notation_color,
                            anchor='center',
                            id=0,
                            tags=['ts_classic'],
                        )

                    if indicator_type in ('klavarskribo', 'classical & klavarskribo'):
                        beat_len_mm = (float(numer) * (4.0 / float(max(1, denom))) * mm_per_quarter) / max(1, numer)
                        bar_offsets, grid_offsets = resolve_grid_layer_offsets(beat_grouping, numer, denom)
                        boundaries = sorted(list(dict.fromkeys([0.0] + [float(v) for v in (bar_offsets + grid_offsets)])))
                        for k in range(1, numer + 1):
                            yk = y + ((k - 1) * beat_len_mm)
                            self.du.add_line(
                                ts_x_right - half_span,
                                yk,
                                ts_x_right + half_span,
                                yk,
                                color=self.notation_color,
                                width_mm=guide_thickness,
                                id=0,
                                tags=['ts_klavarskribo'],
                            )
                            self.du.add_text(
                                ts_x_mid,
                                yk,
                                str(k),
                                family=klav_family,
                                size_pt=klav_size,
                                bold=klav_bold,
                                italic=klav_italic,
                                color=self.notation_color,
                                anchor='center',
                                id=0,
                                tags=['ts_klavarskribo'],
                            )
                        self.du.add_text(
                            ts_x_left,
                            y,
                            str(len(boundaries) if boundaries else 1),
                            family=klav_family,
                            size_pt=klav_size,
                            bold=klav_bold,
                            italic=klav_italic,
                            color=self.notation_color,
                            anchor='center',
                            id=0,
                            tags=['ts_klavarskribo'],
                        )
