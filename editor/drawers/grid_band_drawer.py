from __future__ import annotations
import math
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

    def _build_repeating_dark_intervals(self, markers: list, barlines: list[float], score_end: float) -> list[tuple[float, float]]:
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

            # Color resets at each barline.
            is_dark = True

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
                    # OFF marker: next resumed segment should start dark.
                    is_dark = True
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

    def _grid_band_fill_rgba(self, layout, field_name: str) -> tuple[float, float, float, float]:
        """Get the fill color for grid bands."""
        custom = str(getattr(layout, field_name, '') or '').strip()
        if custom and not custom.startswith('#'):
            custom = f"#{custom}"
        qcustom = QtGui.QColor(custom)
        if qcustom.isValid():
            return (qcustom.red() / 255.0, qcustom.green() / 255.0, qcustom.blue() / 255.0, 1.0)
        r, g, b = Style.get_named_rgb('accent', (200, 200, 200))
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
        x_mm: float,
        color: tuple[float, float, float, float],
        angle_deg: float,
    ) -> None:
        self = cast("Editor", self)
        if not times:
            return

        def _rounded_badge_poly(cx: float, cy: float, text: str, angle: float, pad: float) -> list[tuple[float, float]]:
            # Reuse text_drawer geometry idea: rounded rect then rotate around center.
            _, _, w_mm, h_mm = du._get_text_extents_mm(text, 'Edwin', 16.0, False, False)
            w_mm += 2.0 * pad
            h_mm += 2.0 * pad
            hw = 0.5 * w_mm
            hh = 0.5 * h_mm
            r = min(pad, hw, hh)

            pts: list[tuple[float, float]] = []
            if r <= 1e-6:
                pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            else:
                corners = [
                    (-hw + r, -hh + r, 180.0, 270.0),
                    (hw - r, -hh + r, 270.0, 360.0),
                    (hw - r, hh - r, 0.0, 90.0),
                    (-hw + r, hh - r, 90.0, 180.0),
                ]
                step = 15.0
                for ox, oy, a0, a1 in corners:
                    deg = a0
                    while deg <= a1 + 0.01:
                        rad = math.radians(deg)
                        pts.append((ox + r * math.cos(rad), oy + r * math.sin(rad)))
                        deg += step

            ang = math.radians(angle)
            ca = math.cos(ang)
            sa = math.sin(ang)
            out: list[tuple[float, float]] = []
            for dx, dy in pts:
                rx = dx * ca - dy * sa
                ry = dx * sa + dy * ca
                out.append((cx + rx, cy + ry))
            return out

        for t in times:
            y = float(self.time_to_mm(float(t + self.snap_size_units * .5)))
            text_x = float(x_mm) + self.semitone_dist * 24.0 if Operator(SHORTEST_DURATION).le(float(x_mm), 50.0) else float(x_mm) - self.semitone_dist * 24.0
            text_anchor = 'center'

            # White rounded mask behind OFF label
            bg_poly = _rounded_badge_poly(text_x, y, 'Band OFF', float(angle_deg), pad=1.1)
            du.add_polygon(
                bg_poly,
                stroke_color=None,
                fill_color=(0.0, 0.0, 0.0, 1.0),
                id=0,
                tags=["grid_band_stop_bg"],
            )

            du.add_text(
                text_x,
                y,
                "Band OFF",
                size_pt=16,
                family='Edwin',
                color=color,
                anchor=text_anchor,
                angle_deg=float(angle_deg),
                id=0,
                tags=["grid_band_stop"],
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
        score = self.current_score()
        if score is None:
            return
        layout = score.layout
        if not bool(getattr(layout, 'sub_band_visible', True)):
            # Reuse sub_band_visible setting for grid_band visibility
            return

        # Fixed horizontal spans requested:
        # left: key 2 -> 44, right: key 44 -> 77
        xx2 = float(self.pitch_to_x(2))
        xx86 = float(self.pitch_to_x(86))
        x10 = float(self.pitch_to_x(10))
        x44 = float(self.pitch_to_x(44))
        x77 = float(self.pitch_to_x(77))
        left_band_x_a = min(x10, x44)
        left_band_x_b = max(x10, x44)
        right_band_x_a = min(x44, x77)
        right_band_x_b = max(x44, x77)

        # Get left and right band tracks
        left_markers = list(getattr(layout, 'grid_band_left_track', []) or [])
        right_markers = list(getattr(layout, 'grid_band_right_track', []) or [])
        bars = self._all_barlines(score)
        score_end = float(bars[-1]) if bars else 0.0

        left_intervals = self._build_repeating_dark_intervals(left_markers, bars, score_end)
        right_intervals = self._build_repeating_dark_intervals(right_markers, bars, score_end)

        # Draw left band
        left_fill = self._grid_band_fill_rgba(layout, 'grid_band_left_color')
        self._draw_grid_band_side(
            du,
            left_intervals,
            left_band_x_a,
            left_band_x_b,
            left_fill,
        )

        # Draw right band
        right_fill = self._grid_band_fill_rgba(layout, 'grid_band_right_color')
        self._draw_grid_band_side(
            du,
            right_intervals,
            right_band_x_a,
            right_band_x_b,
            right_fill,
        )

        active_tool = str(getattr(getattr(self, '_tool', None), 'TOOL_NAME', ''))
        if active_tool == 'grid_band':
            off_color = self._grid_band_off_rgba()
            left_zero = self._zero_marker_times(left_markers, score_end)
            right_zero = self._zero_marker_times(right_markers, score_end)
            self._draw_zero_off_indicators(
                du,
                left_zero,
                float(self.pitch_to_x(2)),
                off_color,
                0.0,
            )
            self._draw_zero_off_indicators(
                du,
                right_zero,
                float(self.pitch_to_x(86)),
                off_color,
                0.0,
            )
