from __future__ import annotations
import math
from typing import Optional, List
from PySide6 import QtCore, QtWidgets

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE, BaseGrid
from utils.CONSTANT import QUARTER_NOTE_UNIT


class TimeSignatureTool(BaseTool):
    TOOL_NAME = 'time_signature'
    def __init__(self) -> None:
        super().__init__()
        # Prevent re-entrant dialog openings on a single click
        self._dialog_open: bool = False

    def toolbar_spec(self) -> list[dict]:
        return []

    def _get_base_grid_start_positions(self) -> List[float]:
        """Return a list of segment-start positions (barline times in ticks) for each BaseGrid.

        Positions include the global start (0) and each subsequent segment start.
        """
        starts: List[float] = []
        score: SCORE | None = self._editor.current_score() if self._editor else None
        if score is None:
            return starts
        cur_t = 0.0
        starts.append(cur_t)
        for bg in getattr(score, 'base_grid', []) or []:
            num = float(getattr(bg, 'numerator', 4) or 4)
            den = float(getattr(bg, 'denominator', 4) or 4)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
            # Next segment starts after this segment's measures
            cur_t += measure_len * mcount
            starts.append(cur_t)
        # De-duplicate and sort
        try:
            starts = sorted(list(dict.fromkeys(starts)))
        except Exception:
            starts = sorted(starts)
        return starts

    def _compute_all_barline_positions(self) -> List[float]:
        """Return all barline positions (start of every measure across segments) in ticks."""
        bars: List[float] = []
        score: SCORE | None = self._editor.current_score() if self._editor else None
        if score is None:
            return bars
        cur_t = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            num = float(getattr(bg, 'numerator', 4) or 4)
            den = float(getattr(bg, 'denominator', 4) or 4)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
            for _ in range(mcount):
                bars.append(cur_t)
                cur_t += measure_len
        # Include terminal end barline for completeness
        bars.append(cur_t)
        # De-duplicate and sort
        try:
            bars = sorted(list(dict.fromkeys(bars)))
        except Exception:
            bars = sorted(bars)
        return bars

    def on_left_click(self, x: float, y: float) -> None:
        """Detect the closest barline and edit/insert a time signature change at that barline.

        - If the barline is exactly a BaseGrid segment start: edit that segment's signature.
        - Otherwise (barline inside a segment): split the segment at the barline and
          insert a new BaseGrid with measure_amount=1.
        """
        super().on_left_click(x, y)
        score: SCORE | None = self._editor.current_score()

        # Map click to time (ticks)
        click_t = float(self._editor.y_to_time(y))
        
        # Find nearest barline
        bars = self._compute_all_barline_positions()
        nearest = bars[0]
        min_abs = abs(click_t - float(nearest))
        for t in bars:
            d = abs(click_t - float(t))
            if d < min_abs:
                min_abs = d
                nearest = float(t)
        
        # Determine segment and offset within segment for this barline
        tol = 1e-6
        base_list = list(getattr(score, 'base_grid', []) or [])
        
        # Build segment boundaries (start times)
        seg_starts = []
        cur_t = 0.0
        for bg in base_list:
            seg_starts.append(cur_t)
            num = float(getattr(bg, 'numerator', 4) or 4)
            den = float(getattr(bg, 'denominator', 4) or 4)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
            cur_t += measure_len * mcount
        
        # Find segment index such that barline lies within [start, next_start]
        seg_i = 0
        for i in range(len(base_list)):
            start_t = seg_starts[i]
            end_t = seg_starts[i + 1] if i + 1 < len(seg_starts) else cur_t
            if start_t - tol <= nearest <= end_t + tol:
                seg_i = i
                break

        # If click is on an existing change (segment start), edit that segment
        edit_i = None
        for i, s in enumerate(seg_starts):
            if abs(float(s) - float(nearest)) <= tol:
                edit_i = i
                break
        if edit_i is None:
            seg_bg = base_list[seg_i]
        else:
            seg_i = edit_i
            seg_bg = base_list[seg_i]

        # Find the base_grid active at the click time (for dialog prefill)
        active_i = 0
        for i in range(len(base_list)):
            start_t = seg_starts[i]
            end_t = seg_starts[i + 1] if i + 1 < len(seg_starts) else cur_t
            if start_t - tol <= click_t < end_t + tol:
                active_i = i
                break
        active_bg = base_list[active_i]
        
        # Compute measure index offset within segment
        num = float(getattr(seg_bg, 'numerator', 4) or 4)
        den = float(getattr(seg_bg, 'denominator', 4) or 4)
        measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
        
        # Offset in measures from segment start
        offset_measures = int(math.ceil((nearest - seg_starts[seg_i]) / max(1e-9, measure_len))) # not sure yet for floor/ceil...
        
        # Clamp offset into valid range [0..measure_amount]
        m_total = int(getattr(seg_bg, 'measure_amount', 1) or 1)
        offset_measures = max(0, min(offset_measures, m_total))
        
        # Open dialog and build a new BaseGrid from its values
        from ui.dialogs.time_signature_dialog import TimeSignatureDialog
        
        # Prefill dialog from the current segment for sensible defaults
        initial_numer = 4
        initial_denom = 4
        initial_grid_positions: list[int] = [1, 2, 3, 4]
        initial_indicator_enabled = True
        cur_bg = active_bg
        initial_numer = int(getattr(cur_bg, 'numerator', initial_numer) or initial_numer)
        initial_denom = int(getattr(cur_bg, 'denominator', initial_denom) or initial_denom)
        gp_attr = getattr(cur_bg, 'beat_grouping', None)
        initial_grid_positions = list(gp_attr if gp_attr is not None else (getattr(cur_bg, 'grid_positions', []) or initial_grid_positions))
        initial_indicator_enabled = bool(getattr(cur_bg, 'indicator_enabled', initial_indicator_enabled))
        
        # Parent: try active window
        parent_w = QtWidgets.QApplication.activeWindow()
        
        # Guard against re-entrancy
        if getattr(self, '_dialog_open', False):
            return
        self._dialog_open = True
        editor_widget = getattr(self._editor, 'widget', None) if self._editor else None
        dlg = TimeSignatureDialog(parent=parent_w,
                                  initial_numer=initial_numer,
                                  initial_denom=initial_denom,
                                  initial_grid_positions=initial_grid_positions,
                                  initial_indicator_enabled=initial_indicator_enabled,
                                  editor_widget=editor_widget)

        original_seg_state = {
            'numer': int(getattr(seg_bg, 'numerator', initial_numer) or initial_numer),
            'denom': int(getattr(seg_bg, 'denominator', initial_denom) or initial_denom),
            'beat_grouping': list(getattr(seg_bg, 'beat_grouping', None) or list(initial_grid_positions)),
            'indicator_enabled': bool(getattr(seg_bg, 'indicator_enabled', initial_indicator_enabled)),
            'measure_amount': int(getattr(seg_bg, 'measure_amount', 1) or 1),
        }
        preview_state: dict = {'new_bg': None}

        def _refresh_editor_view() -> None:
            if self._editor is None:
                return
            self._editor.update_score_length()
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()

        def _apply_preview(numer: int, denom: int, grid_positions: list[int], indicator_enabled: bool) -> None:
            try:
                if score is None:
                    return
                if edit_i is not None:
                    seg_bg.numerator = int(numer)
                    seg_bg.denominator = int(denom)
                    seg_bg.beat_grouping = list(grid_positions)
                    seg_bg.indicator_enabled = bool(indicator_enabled)
                else:
                    if preview_state['new_bg'] is None:
                        seg_bg.measure_amount = int(offset_measures)
                        new_bg = BaseGrid(numerator=int(numer), denominator=int(denom),
                                          beat_grouping=list(grid_positions), measure_amount=1,
                                          indicator_enabled=bool(indicator_enabled))
                        score.base_grid.insert(seg_i + 1, new_bg)
                        preview_state['new_bg'] = new_bg
                    else:
                        new_bg = preview_state['new_bg']
                        new_bg.numerator = int(numer)
                        new_bg.denominator = int(denom)
                        new_bg.beat_grouping = list(grid_positions)
                        new_bg.indicator_enabled = bool(indicator_enabled)
                _refresh_editor_view()
            except Exception:
                pass

        def _revert_preview() -> None:
            try:
                if edit_i is not None:
                    seg_bg.numerator = original_seg_state['numer']
                    seg_bg.denominator = original_seg_state['denom']
                    seg_bg.beat_grouping = list(original_seg_state['beat_grouping'])
                    seg_bg.indicator_enabled = bool(original_seg_state['indicator_enabled'])
                else:
                    if preview_state['new_bg'] is not None:
                        try:
                            score.base_grid.remove(preview_state['new_bg'])
                        except Exception:
                            pass
                        preview_state['new_bg'] = None
                    seg_bg.measure_amount = original_seg_state['measure_amount']
                _refresh_editor_view()
            except Exception:
                pass

        dlg.previewChanged.connect(_apply_preview)
        dlg.raise_()
        dlg.activateWindow()

        def _finalize_dialog(result: int) -> None:
            try:
                if result != QtWidgets.QDialog.Accepted:
                    _revert_preview()
                    return
                numer, denom, grid_positions, indicator_enabled = dlg.get_values()

                # Apply change: edit if click is on a change, otherwise insert at barline
                if edit_i is not None:
                    seg_bg.numerator = int(numer)
                    seg_bg.denominator = int(denom)
                    seg_bg.beat_grouping = list(grid_positions)
                    seg_bg.indicator_enabled = bool(indicator_enabled)
                else:
                    if preview_state['new_bg'] is None:
                        seg_bg.measure_amount = int(offset_measures)
                        preview_state['new_bg'] = BaseGrid(numerator=int(numer), denominator=int(denom),
                                                            beat_grouping=list(grid_positions),
                                                            measure_amount=1, indicator_enabled=bool(indicator_enabled))
                        score.base_grid.insert(seg_i + 1, preview_state['new_bg'])
                    else:
                        preview_state['new_bg'].numerator = int(numer)
                        preview_state['new_bg'].denominator = int(denom)
                        preview_state['new_bg'].beat_grouping = list(grid_positions)
                        preview_state['new_bg'].indicator_enabled = bool(indicator_enabled)

                # Snapshot and update after dialog closes (next event loop tick)
                self._editor._snapshot_if_changed(coalesce=False, label='time_signature_append')
                _refresh_editor_view()
            finally:
                self._dialog_open = False
                preview_state['new_bg'] = None

        dlg.finished.connect(_finalize_dialog)
        dlg.show()
        # # Nudge scroll to force visual refresh (some views only update on scroll)
        # w = getattr(self._editor, 'widget', None)
        # if w is not None and hasattr(w, 'set_scroll_logical_px'):
        #     cur = int(getattr(w, '_scroll_logical_px', 0) or 0)
        #     w.set_scroll_logical_px(cur + 1)
        #     w.set_scroll_logical_px(cur)

    def on_right_click(self, x: float, y: float) -> None:
        """Delete the BaseGrid at the clicked change (segment start), except for the first."""
        super().on_right_click(x, y)
        if self._editor is None:
            return
        score: SCORE | None = self._editor.current_score()
        if score is None:
            return
        base_list = list(getattr(score, 'base_grid', []) or [])
        if not base_list:
            return
        click_t = float(self._editor.y_to_time(y))
        bars = self._compute_all_barline_positions()
        if not bars:
            return
        nearest = min(bars, key=lambda t: abs(click_t - float(t)))

        # Find segment start positions
        seg_starts = []
        cur_t = 0.0
        for bg in base_list:
            seg_starts.append(cur_t)
            num = float(getattr(bg, 'numerator', 4) or 4)
            den = float(getattr(bg, 'denominator', 4) or 4)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
            cur_t += measure_len * mcount
        tol = 1e-6

        # Identify segment whose start matches the nearest barline
        seg_i = None
        for i, s in enumerate(seg_starts):
            if abs(float(s) - float(nearest)) <= tol:
                seg_i = i
                break

        # cannot delete first or non-change barline
        if seg_i is None or seg_i <= 0:
            return

        # Delete the BaseGrid at this segment start
        del base_list[seg_i]
        score.base_grid = base_list
        self._editor._snapshot_if_changed(coalesce=False, label='time_signature_delete')
        self._editor.update_score_length()

    # Block other mouse handlers for this tool; we use click only for now
    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

    def on_toolbar_button(self, name: str) -> None:
        # no-op
        return
