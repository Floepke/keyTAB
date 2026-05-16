"""GridBandDrawer: renders alternating grid band backgrounds."""

from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION, hex_to_rgba
from utils.operator import Operator
from engraver.helpers import build_grid_band_dark_intervals, normalize_hex_color, time_to_y


class GridBandDrawer:
    """Draw grid band (alternating background shading)."""

    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.op = Operator(SHORTEST_DURATION)

    def draw(self) -> None:
        """Draw grid band bands for the current page and line."""
        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('grid_band_visible', True)):
            return

        score = self.context.score or {}
        base_grid = list(score.get('base_grid', []) or [])
        grid_bands = list(layout.get('grid_band_track', []) or [])
        if not base_grid or not grid_bands:
            return

        band_hex = normalize_hex_color(str(layout.get('grid_band_color', '#ccc') or '#ccc')) or '#ccc'
        r, g, b, _ = hex_to_rgba(band_hex, 1.0)
        band_fill = (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0)

        barlines: list[float] = [0.0]
        cur = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            for _ in range(int(max(0, measures))):
                barlines.append(float(cur))
                cur += measure_len
        barlines.append(float(cur))
        barlines = sorted(list(dict.fromkeys(round(float(t), 6) for t in barlines)))

        starts_dark = str(layout.get('grid_band_start_phase', 'dark') or 'dark').strip().lower() != 'light'
        dark_intervals = build_grid_band_dark_intervals(grid_bands, barlines, float(cur), starts_dark=starts_dark)
        if not dark_intervals:
            return

        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            left_pitch = int(line.get('natural_bound_left', line.get('bound_left', 1)) or 1)
            right_pitch = int(line.get('natural_bound_right', line.get('bound_right', 88)) or 88)
            x1 = float(key_to_x(left_pitch))
            x2 = float(key_to_x(right_pitch))
            if x2 < x1:
                x1, x2 = x2, x1

            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)
            for a, b_int in dark_intervals:
                t0 = max(float(a), float(lt0))
                t1 = min(float(b_int), float(lt1))
                if self.op.le(t1, t0):
                    continue
                y1 = time_to_y(line, t0)
                y2 = time_to_y(line, t1)
                if y2 < y1:
                    y1, y2 = y2, y1
                self.du.add_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    stroke_color=None,
                    fill_color=band_fill,
                    id=0,
                    tags=['grid_band'],
                )
