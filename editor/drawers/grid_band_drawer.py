from __future__ import annotations
from typing import TYPE_CHECKING, cast

from PySide6 import QtGui

from ui.widgets.draw_util import DrawUtil
from utils.operator import Operator
from utils.CONSTANT import EDITOR_SIDE_BAND_INSET_SEMITONES, SHORTEST_DURATION

from ui.style import Style

if TYPE_CHECKING:
    from editor.editor import Editor


class GridBandDrawerMixin:
    def _marker_field(self, marker, name: str, default):
        if isinstance(marker, dict):
            return marker.get(name, default)
        return getattr(marker, name, default)

    def _all_barlines(self, score) -> list[float]:
        bars: list[float] = []
        cur = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            numer = int(getattr(bg, 'numerator', 4) or 4)
            denom = int(getattr(bg, 'denominator', 4) or 4)
            measure_len = float(numer) * (4.0 / float(denom)) * 256.0
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            for _ in range(max(1, mcount)):
                bars.append(float(cur))
                cur += measure_len
        bars.append(float(cur))
        out = sorted(list(dict.fromkeys(round(float(v), 6) for v in bars)))
        return out if out else [0.0, 256.0]

    def _build_repeating_dark_intervals(self, markers: list, barlines: list[float], score_end: float, starts_dark: bool = True) -> list[tuple[float, float]]:
        """Build dark-band intervals using marker duration, resetting at each barline.

        Rules:
        - Marker defines step size (duration) from marker.time until next marker (or score end).
        - At every barline, color resets to dark.
        - At marker starts, phase (step boundaries) resets to marker.time.
        - Marker starts preserve the current dark/light color at that time.
        - Marker start truncates the previous marker range.
        - Bands are clipped at barlines, then restart dark after each barline.
        """
        op = Operator()
        if op.le(float(score_end), 0.0):
            return []
        if not markers:
            return []

        bars = [
            float(b)
            for b in (barlines or [])
            if op.ge(float(b), 0.0) and op.le(float(b), float(score_end))
        ]
        bars = sorted(list(dict.fromkeys(round(float(b), 6) for b in bars)))
        if not bars or op.not_equal(float(bars[0]), 0.0):
            bars = [0.0] + bars
        if op.not_equal(float(bars[-1]), float(score_end)):
            bars.append(float(score_end))

        track: list[tuple[float, float, int]] = []
        for mk in markers:
            try:
                mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                dur = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
                mid = int(self._marker_field(mk, '_id', 0) or 0)
            except Exception:
                continue
            if op.lt(dur, 0.0):
                continue
            if op.ge(mt, float(score_end)):
                continue
            track.append((max(0.0, mt), dur, mid))

        if not track:
            return []
        track.sort(key=lambda x: (float(x[0]), int(x[2])))

        segments: list[tuple[float, float, float]] = []
        for i, (start, step, _mid) in enumerate(track):
            end = float(track[i + 1][0]) if (i + 1) < len(track) else float(score_end)
            if op.le(end, start):
                continue
            segments.append((float(start), float(end), float(step)))

        if not segments:
            return []

        out: list[tuple[float, float]] = []
        for bi in range(len(bars) - 1):
            bar_start = float(bars[bi])
            bar_end = float(bars[bi + 1])
            if op.le(bar_end, bar_start):
                continue

            # Color resets at each barline to the configured phase.
            is_dark = bool(starts_dark)

            for seg_start_raw, seg_end_raw, step in segments:
                if op.le(seg_end_raw, bar_start):
                    continue
                if op.ge(seg_start_raw, bar_end):
                    break

                seg_start = max(float(seg_start_raw), float(bar_start))
                seg_end = min(float(seg_end_raw), float(bar_end))
                if op.le(seg_end, seg_start):
                    continue

                # duration == 0 means stop engraving bands for this segment.
                if op.le(step, 0.0):
                    # OFF marker: next resumed segment should start with configured phase.
                    is_dark = bool(starts_dark)
                    continue

                # Marker start resets phase boundaries to seg_start,
                # but keeps the current color state.
                t0 = float(seg_start)
                color = bool(is_dark)
                while op.lt(t0, seg_end):
                    t1 = min(float(seg_end), float(t0 + step))
                    if color and op.gt(t1, t0):
                        out.append((float(t0), float(t1)))
                    t0 = float(t1)
                    color = not color

                is_dark = bool(color)

        if not out:
            return []

        out.sort(key=lambda x: (float(x[0]), float(x[1])))
        merged: list[list[float]] = []
        for s, e in out:
            fs = float(s)
            fe = float(e)
            if (not merged) or op.gt(fs, float(merged[-1][1])):
                merged.append([fs, fe])
            elif op.gt(fe, float(merged[-1][1])):
                merged[-1][1] = fe

        return [(a, b) for a, b in merged if op.gt(float(b), float(a))]

    def _grid_band_fill_rgba(self, layout, field_name: str, side: str) -> tuple[float, float, float, float]:
        """Get the fill color for grid bands."""
        custom = str(getattr(layout, field_name, '') or '').strip()
        if custom and not custom.startswith('#'):
            custom = f"#{custom}"
        qcustom = QtGui.QColor(custom)
        if qcustom.isValid():
            return (qcustom.red() / 255.0, qcustom.green() / 255.0, qcustom.blue() / 255.0, 1.0)
        key = 'midi_left' if str(side).lower().startswith('l') else 'midi_right'
        r, g, b = Style.get_named_rgb(key, (200, 200, 200))
        return (
            max(0, min(255, int(r))) / 255.0,
            max(0, min(255, int(g))) / 255.0,
            max(0, min(255, int(b))) / 255.0,
            0.3,
        )

    def _grid_band_off_rgba(self) -> tuple[float, float, float, float]:
        r, g, b = Style.get_named_rgb('accent', (200, 80, 80))
        return (
            max(0, min(255, int(r))) / 255.0,
            max(0, min(255, int(g))) / 255.0,
            max(0, min(255, int(b))) / 255.0,
            1.0,
        )

    def _zero_marker_times(self, markers: list, score_end: float) -> list[float]:
        op = Operator()
        out: list[float] = []
        for mk in markers:
            try:
                mt = float(self._marker_field(mk, 'time', 0.0) or 0.0)
                dur = float(self._marker_field(mk, 'duration', 0.0) or 0.0)
            except Exception:
                continue
            if not op.eq(dur, 0.0):
                continue
            if op.lt(mt, 0.0) or op.gt(mt, float(score_end)):
                continue
            out.append(float(mt))
        return sorted(list(dict.fromkeys(round(float(v), 6) for v in out)))

    def _draw_zero_off_indicators(
        self,
        du: DrawUtil,
        times: list[float],
        _x_mm: float,
        color: tuple[float, float, float, float],
        _angle_deg: float,
    ) -> None:
        self = cast("Editor", self)
        x_min = float(min(self.pitch_to_x(2), self.pitch_to_x(86)))
        x_max = float(max(self.pitch_to_x(2), self.pitch_to_x(86)))
        for t in times:
            y = float(self.time_to_mm(float(t)))
            du.add_line(
                x_min,
                y,
                x_max,
                y,
                color=color,
                width_mm=1.0,
                dash_pattern=None,
                id=0,
                tags=["grid_band_stop_line"],
            )

    def _draw_grid_band_side(
        self,
        du: DrawUtil,
        intervals: list[tuple[float, float]],
        xa: float,
        xb: float,
        fill: tuple[float, float, float, float],
    ) -> None:
        """Draw grid band markers on one side (left or right)."""
        op = Operator()
        if not intervals:
            return
        
        for t0, t1 in intervals:
            y0 = float(self.time_to_mm(float(t0)))
            y1 = float(self.time_to_mm(float(t1)))
            
            if op.le(y1, y0):
                continue
            
            # Draw the band rectangle
            du.add_rectangle(
                xa,
                y0,
                xb,
                y1,
                stroke_color=None,
                fill_color=fill,
                id=0,
                tags=["grid_band"],
            )

    def draw_grid_band(self, du: DrawUtil) -> None:
        """Draw all grid band markers (Grid Band tool visualization)."""
        self = cast("Editor", self)
        active_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', ''))
        if active_tool != 'grid_band':
            return

        score = self.current_score()
        if score is None:
            return
        layout = score.layout

        # Fixed horizontal span: draw a single band area from key 10 to key 77.
        x10 = float(self.pitch_to_x(10))
        x77 = float(self.pitch_to_x(77))
        band_x_a = min(x10, x77)
        band_x_b = max(x10, x77)

        # Single grid band track with legacy fallback.
        markers = list(getattr(layout, 'grid_band_track', []) or [])
        if not markers:
            markers = list(getattr(layout, 'grid_band_left_track', []) or []) + list(getattr(layout, 'grid_band_right_track', []) or [])
        bars = self._all_barlines(score)
        score_end = float(bars[-1]) if bars else 0.0
        phase = str(getattr(layout, 'grid_band_start_phase', 'dark') or 'dark').strip().lower()
        starts_dark = phase != 'light'

        intervals = self._build_repeating_dark_intervals(markers, bars, score_end, starts_dark=starts_dark)

        # Draw band
        fill = self._grid_band_fill_rgba(layout, 'grid_band_color', 'left')
        self._draw_grid_band_side(
            du,
            intervals,
            band_x_a,
            band_x_b,
            fill,
        )

        if active_tool == 'grid_band':
            off_color = self._grid_band_off_rgba()
            zero_markers = self._zero_marker_times(markers, score_end)
            self._draw_zero_off_indicators(
                du,
                zero_markers,
                band_x_a,
                off_color,
                0.0,
            )
