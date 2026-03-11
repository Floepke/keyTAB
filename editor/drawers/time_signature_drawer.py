from __future__ import annotations
from typing import TYPE_CHECKING, cast
from PySide6 import QtGui
from ui.widgets.draw_util import DrawUtil
from fonts import register_font_from_bytes
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION, EDITOR_SIDE_BAND_INSET_SEMITONES
from file_model.base_grid import resolve_grid_layer_offsets
from utils.operator import Operator
try:
    from ui.style import Style  # type: ignore
except Exception:
    Style = None  # type: ignore

if TYPE_CHECKING:
    from editor.editor import Editor


class TimeSignatureDrawerMixin:
    def draw_time_signature(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return
        tool = getattr(self, "_tool", None)
        tool_name = getattr(tool, "TOOL_NAME", "")
        ts_mode = str(getattr(tool, "_edit_mode", "Gr") or "Gr")
        # Read global indicator type from Layout
        indicator_type = getattr(score.layout, 'time_signature_indicator_type', 'classical')
        layout = score.layout

        def _resolve_font_family(font) -> str:
            family = str(getattr(font, 'family', 'Latin Modern Roman') or 'Latin Modern Roman')
            if family != 'Latin Modern Roman':
                return family
            reg = register_font_from_bytes('Latin Modern Roman') if register_font_from_bytes else 'Latin Modern Roman'
            return reg or 'Latin Modern Roman'

        classic_font = getattr(layout, 'time_signature_indicator_classic_font', None)
        klav_font = getattr(layout, 'time_signature_indicator_klavarskribo_font', None)
        classic_requested = str(getattr(classic_font, 'family', 'Latin Modern Roman') or 'Latin Modern Roman')
        klav_requested = str(getattr(klav_font, 'family', 'Latin Modern Roman') or 'Latin Modern Roman')

        if getattr(self, '_ts_cached_classic_requested', None) != classic_requested:
            self._ts_cached_classic_requested = classic_requested
            self._ts_cached_classic_family = _resolve_font_family(classic_font)
        if getattr(self, '_ts_cached_klav_requested', None) != klav_requested:
            self._ts_cached_klav_requested = klav_requested
            self._ts_cached_klav_family = _resolve_font_family(klav_font)

        classic_family = str(getattr(self, '_ts_cached_classic_family', None) or _resolve_font_family(classic_font))
        klav_family = str(getattr(self, '_ts_cached_klav_family', None) or _resolve_font_family(klav_font))
        classic_size = 25.0
        klav_size = 15.0
        guide_width_mm = float(getattr(layout, 'time_signature_indicator_guide_thickness_mm', 0.5) or 0.5)
        divider_width_mm = float(getattr(layout, 'time_signature_indicator_divide_guide_thickness_mm', 1.0) or 1.0)

        def _subband_fill_rgba(field_name: str) -> tuple[float, float, float, float]:
            custom = str(getattr(layout, field_name, '') or '').strip()
            if custom and not custom.startswith('#'):
                custom = f"#{custom}"
            qcustom = QtGui.QColor(custom)
            if qcustom.isValid():
                return (qcustom.red() / 255.0, qcustom.green() / 255.0, qcustom.blue() / 255.0, 1.0)
            if not Style:
                return (0.5, 0.5, 0.5, 1.0)
            r, g, b = Style.get_named_rgb('accent', (130, 130, 130))
            return (
                max(0, min(255, int(r))) / 255.0,
                max(0, min(255, int(g))) / 255.0,
                max(0, min(255, int(b))) / 255.0,
                1.0,
            )

        def draw_sub_band(bg, y_mm: float) -> None:
            if not bool(getattr(layout, 'sub_band_visible', True)):
                return
            sub_band_left = getattr(bg, 'sub_band_left', None)
            sub_band_right = getattr(bg, 'sub_band_right', None)
            if (sub_band_left is None) and (sub_band_right is None):
                return
            numer = int(getattr(bg, 'numerator', 4) or 4)
            denom = int(getattr(bg, 'denominator', 4) or 4)
            zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
            measure_len_mm = float(numer) * (4.0 / max(1.0, float(denom))) * zpq
            measure_len_ticks = float(numer) * (4.0 / max(1.0, float(denom))) * float(QUARTER_NOTE_UNIT)
            left_raw = [float(v) for v in (sub_band_left or []) if isinstance(v, (int, float))]
            right_raw = [float(v) for v in (sub_band_right or []) if isinstance(v, (int, float))]

            margin = float(self.margin or 0.0)
            width_mm, _ = du.current_page_size_mm()
            semitone = float(self.semitone_dist or 0.0)
            stave_left = margin + semitone
            stave_right = max(stave_left, width_mm - margin - semitone * 2.0)
            inset_w = max(0.0, float(EDITOR_SIDE_BAND_INSET_SEMITONES) * semitone)
            x1 = min(stave_right, stave_left + inset_w)
            x2 = max(x1, stave_right - inset_w)
            x_center = float(self.pitch_to_x(45))
            x_center = max(x1, min(x2, x_center))
            left_fill = _subband_fill_rgba('sub_band_left_color')
            right_fill = _subband_fill_rgba('sub_band_right_color')

            def _draw_side(raw_positions: list[float], xa: float, xb: float, fill: tuple[float, float, float, float]) -> None:
                valid = [p for p in raw_positions if 0.0 <= p < measure_len_ticks]
                positions = sorted(set(valid))
                if not positions:
                    return
                if abs(float(positions[0])) > 1e-6:
                    positions = [0.0] + positions
                boundaries = list(positions) + [measure_len_ticks]
                for i in range(len(boundaries) - 1):
                    if (i % 2) != 0:
                        continue
                    t0 = float(boundaries[i])
                    t1 = float(boundaries[i + 1])
                    if t1 <= t0:
                        continue
                    y0 = y_mm + (t0 / max(1e-9, measure_len_ticks)) * measure_len_mm
                    y1 = y_mm + (t1 / max(1e-9, measure_len_ticks)) * measure_len_mm
                    if y1 <= y0:
                        continue
                    du.add_rectangle(
                        xa,
                        y0,
                        xb,
                        y1,
                        stroke_color=None,
                        fill_color=fill,
                        id=0,
                        tags=["sub_band"],
                    )

            _draw_side(left_raw, x1, x_center, left_fill)
            _draw_side(right_raw, x_center, x2, right_fill)

        # Shared layout metrics
        margin = float(self.margin or 0.0)
        stave_left_position = margin + float(self.semitone_dist or 0.0)
        # Render at segment starts along time axis
        time_cursor = margin

        # Helper: draw classical numerator/denominator at segment boundary
        def draw_classical(numerator: int, denominator: int, enabled: bool, y_mm: float) -> None:
            color = (0.6, 0.6, 0.6, 1.0) if not enabled else self.notation_color
            x = stave_left_position - 7.5
            # Numerator
            du.add_text(
                x,
                y_mm - 3.0,
                f"{int(numerator)}",
                size_pt=classic_size,
                color=color,
                id=0,
                tags=["time_signature"],
                anchor='s',
                family=classic_family,
            )
            # Divider line
            du.add_line(
                x - 3.0,
                y_mm,
                x + 3.0,
                y_mm,
                color=color,
                width_mm=divider_width_mm,
                id=0,
                tags=["time_signature"],
                dash_pattern=None,
            )
            # Denominator
            du.add_text(
                x,
                y_mm + 3.0,
                f"{int(denominator)}",
                size_pt=classic_size,
                color=color,
                id=0,
                tags=["time_signature"],
                anchor='n',
                family=classic_family,
            )

        # Helper: draw Klavarskribo-style three-column indicator at segment boundary
        def draw_klavarskribo(numerator: int, denominator: int, enabled: bool, y_mm: float, grid_positions: list[float]) -> None:
            color = (0.6, 0.6, 0.6, 1.0) if not enabled else self.notation_color
            zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
            quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
            measure_len_mm = quarters_per_measure * zpq
            beat_len_mm = measure_len_mm / max(1, int(numerator))
            measure_len_ticks = float(numerator) * (4.0 / max(1.0, float(denominator))) * float(QUARTER_NOTE_UNIT)
            beat_len_ticks = measure_len_ticks / max(1.0, float(numerator))
            op = Operator(float(SHORTEST_DURATION))

            # Column positions: left, middle, right (closest to stave)
            base_x = stave_left_position - margin + 7.5
            col_gap = 5.0
            x_right = base_x + 10.0         # right column (guides)
            x_mid = base_x                  # middle column (beat numbers)
            x_left = base_x - col_gap       # left column (group numbers)

            # Derive grid-1 reset positions (ticks) from beat_grouping.
            grid_bar_off, grid_off = resolve_grid_layer_offsets(
                [float(v) for v in (grid_positions or []) if isinstance(v, (int, float))],
                int(numerator),
                int(denominator),
            )
            grid1_positions = sorted(
                list(
                    dict.fromkeys(
                        [float(v) for v in (grid_bar_off + grid_off) if 0.0 <= float(v) < float(measure_len_ticks)]
                    )
                )
            )

            beat_positions = [float(k) * float(beat_len_ticks) for k in range(0, int(numerator))]
            beat_has_reset = [any(op.eq(float(bp), float(gp)) for gp in grid1_positions) for bp in beat_positions]
            full_group_mode = bool(beat_has_reset) and all(bool(v) for v in beat_has_reset)

            # Build middle-column and group-column values by following grid-1 tick resets per beat.
            mid_values: list[int] = []
            group_values: list[int] = []
            group_starts: list[int] = []
            cur_mid = 1
            cur_group = 1
            for k in range(1, int(numerator) + 1):
                if k == 1:
                    reset_here = True
                elif full_group_mode:
                    # All beats marked as Grid-1 lines means one single group.
                    # Keep middle column counting normally (1,2,3,...) and left column at one group.
                    reset_here = False
                else:
                    reset_here = bool(beat_has_reset[k - 1])
                if reset_here:
                    cur_mid = 1
                    if k > 1:
                        cur_group += 1
                    group_starts.append(k)
                    group_values.append(cur_group)
                else:
                    cur_mid += 1
                mid_values.append(cur_mid)

            # Right column: draw short thick horizontal guide lines at group starts (value 1),
            # but draw all beats when grouping is a single full group (1..numer)
            guide_half_len = 3.0
            #full_group = [int(v) for v in seq] == list(range(1, int(numerator) + 1))
            for k in range(1, int(numerator) + 1):
                # if not full_group and val != 1:
                #     continue
                y = y_mm + (k - 1) * beat_len_mm
                du.add_line(x_right - guide_half_len, y, x_right + guide_half_len, y,
                            color=color, width_mm=guide_width_mm, id=0, tags=["ts_klavars_guide"], dash_pattern=None)
            
            # Final guide at start of next measure
            du.add_line(x_right - guide_half_len, y_mm + measure_len_mm, x_right + guide_half_len, y_mm + measure_len_mm,
                        color=color, width_mm=guide_width_mm, id=0, tags=["ts_klavars_guide"], dash_pattern=None)

            # Middle column: reset to 1 where Grid-1 hits the beat position.
            for k, val in enumerate(mid_values, start=1):
                y = y_mm + (k - 1) * beat_len_mm
                du.add_text(x_mid, y, str(val), size_pt=klav_size, color=color, id=0, tags=["ts_klavars_mid"], anchor='w', family=klav_family)
            # Final 1 at next measure barline (start of next measure)
            du.add_text(x_mid, y_mm + measure_len_mm, "1", size_pt=klav_size, color=color, id=0, tags=["ts_klavars_mid"], anchor='w', family=klav_family)
            # Left column: count groups up each time the middle column resets to 1.
            for gi, s in zip(group_values, group_starts):
                y = y_mm + (s - 1) * beat_len_mm
                du.add_text(x_left - 2.0, y, str(gi), size_pt=klav_size, color=color, id=0, tags=["ts_klavars_left"], anchor='w', family=klav_family)

        # Iterate BaseGrid segments and draw based on indicator_type
        # Classical is always shown; Klavarskribo only when the time-signature tool is active.
        show_classic = True
        show_klavars = (indicator_type in ('klavarskribo', 'both')) and (tool_name == 'time_signature')
        for bg in list(getattr(score, 'base_grid', []) or []):
            numerator = int(getattr(bg, 'numerator', 4) or 4)
            denominator = int(getattr(bg, 'denominator', 4) or 4)
            measure_amount = int(getattr(bg, 'measure_amount', 1) or 1)
            enabled = bool(getattr(bg, 'indicator_enabled', True))
            grid_positions = list(getattr(bg, 'beat_grouping', []) or [])
            quarters_per_measure = float(numerator) * (4.0 / max(1.0, float(denominator)))
            measure_len_mm = quarters_per_measure * float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
            if show_classic:
                draw_classical(numerator, denominator, enabled, time_cursor)
            if show_klavars:
                draw_klavarskribo(numerator, denominator, enabled, time_cursor, grid_positions)
            for m in range(max(1, int(measure_amount))):
                draw_sub_band(bg, time_cursor + (float(m) * float(measure_len_mm)))
            # Advance time cursor by the segment length (mm) to next segment start
            time_cursor += measure_len_mm * float(measure_amount)
