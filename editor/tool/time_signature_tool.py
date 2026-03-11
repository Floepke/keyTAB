from __future__ import annotations

import copy
import math
import time
from typing import List

from PySide6 import QtCore, QtWidgets

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE, BaseGrid
from file_model.base_grid import resolve_grid_layer_offsets
from ui.dialogs.time_signature_dialog import TimeSignatureDialog
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


class TimeSignatureTool(BaseTool):
    TOOL_NAME = 'time_signature'

    def __init__(self) -> None:
        super().__init__()
        self._dialog_open: bool = False
        self._dialog_cooldown_until: float = 0.0
        self._op = Operator(float(SHORTEST_DURATION))

    def toolbar_spec(self) -> list[dict]:
        return []

    def on_toolbar_button(self, name: str) -> None:
        pass

    def _show_mode_status(self) -> None:
        pass

    def _measure_len_ticks(self, bg: BaseGrid) -> float:
        numer = float(getattr(bg, 'numerator', 4) or 4)
        denom = float(getattr(bg, 'denominator', 4) or 4)
        return numer * (4.0 / max(1.0, denom)) * float(QUARTER_NOTE_UNIT)

    def _segments(self, score: SCORE) -> list[tuple[int, BaseGrid, float, float, float]]:
        out: list[tuple[int, BaseGrid, float, float, float]] = []
        cur = 0.0
        for idx, bg in enumerate(list(getattr(score, 'base_grid', []) or [])):
            mlen = self._measure_len_ticks(bg)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            end = cur + (mlen * float(max(1, mcount)))
            out.append((idx, bg, cur, end, mlen))
            cur = end
        return out

    def _find_segment_for_time(self, score: SCORE, ticks: float) -> tuple[int, BaseGrid, float, float, float] | None:
        t = float(max(0.0, ticks))
        segs = self._segments(score)
        if not segs:
            return None
        for seg in segs:
            _idx, _bg, start, end, _mlen = seg
            if start <= t < end:
                return seg
        return segs[-1]

    def _all_barlines(self, score: SCORE) -> list[float]:
        bars: list[float] = []
        cur = 0.0
        for bg in list(getattr(score, 'base_grid', []) or []):
            mlen = self._measure_len_ticks(bg)
            mcount = int(getattr(bg, 'measure_amount', 1) or 1)
            for _ in range(max(1, mcount)):
                bars.append(float(cur))
                cur += mlen
        bars.append(float(cur))
        return sorted(list(dict.fromkeys(float(v) for v in bars)))

    def _barline_hit_tolerance(self) -> float:
        snap = float(getattr(self._editor, 'snap_size_units', 0.0) or 0.0)
        return max(4.0, snap * 0.20)

    def _nearest_barline(self, score: SCORE, ticks: float) -> tuple[float, bool]:
        bars = self._all_barlines(score)
        if not bars:
            return float(ticks), False
        t = float(ticks)
        nearest = min(bars, key=lambda v: abs(float(v) - t))
        return float(nearest), abs(float(nearest) - t) <= self._barline_hit_tolerance()

    def _measure_start_for_time(self, score: SCORE, seg_start: float, seg_end: float, ticks: float) -> float:
        bars = self._all_barlines(score)
        t = float(ticks)
        in_segment = [b for b in bars if float(seg_start) - 1e-6 <= float(b) <= float(seg_end) + 1e-6 and float(b) <= t + 1e-6]
        if in_segment:
            return float(max(in_segment))
        return float(seg_start)

    def _to_unique_sorted(self, values: list[float]) -> list[float]:
        out: list[float] = []
        for v in sorted(float(x) for x in values):
            if not out or not self._op.eq(float(v), float(out[-1])):
                out.append(v)
        return out

    def _value_delete_tolerance(self) -> float:
        return max(4.0, self._barline_hit_tolerance())

    def _grid_positions_as_times(self, bg: BaseGrid) -> list[float]:
        seq = [float(v) for v in (getattr(bg, 'beat_grouping', []) or []) if isinstance(v, (int, float))]
        bar, grid = resolve_grid_layer_offsets(
            seq,
            int(getattr(bg, 'numerator', 4) or 4),
            int(getattr(bg, 'denominator', 4) or 4),
        )
        return self._to_unique_sorted([float(v) for v in (bar + grid)])

    def _edit_grid_value(self, score: SCORE, click_t: float, x: float, delete: bool) -> None:
        # Edit beat_grouping grid lines only
        click_t_f = float(click_t)
        
        seg = self._find_segment_for_time(score, click_t)
        if seg is None:
            return
        _seg_i, bg, seg_start, seg_end, measure_len = seg
        measure_start = self._measure_start_for_time(score, seg_start, seg_end, click_t_f)

        # Grid lines are editable only in the time-signature change measure.
        if not self._op.eq(float(measure_start), float(seg_start)):
            return

        local = max(0.0, min(float(measure_len) - 1e-6, float(click_t_f) - float(measure_start)))
        current = self._grid_positions_as_times(bg)
        current = self._to_unique_sorted([float(v) for v in current if 0.0 <= float(v) < float(measure_len)])

        changed = False

        if delete:
            if not current:
                return
            nearest = min(current, key=lambda v: abs(float(v) - float(local)))
            if abs(float(nearest) - float(local)) <= self._value_delete_tolerance():
                current = [v for v in current if not self._op.eq(float(v), float(nearest))]
                changed = True
        else:
            if not any(self._op.eq(float(v), float(local)) for v in current):
                current.append(float(local))
                current = self._to_unique_sorted(current)
                changed = True

        if not changed:
            return

        bg.beat_grouping = list(current)

        self._editor._snapshot_if_changed(coalesce=False, label='time_signature_grid_edit')
        self._editor.update_score_length()
        self._editor.force_redraw_from_model()

    def _open_time_signature_dialog_at_barline(self, score: SCORE, barline_t: float) -> None:
        if self._dialog_open:
            return
        self._dialog_open = True
        segs = self._segments(score)
        if not segs:
            self._dialog_open = False
            return

        try:
            target_seg = None
            for seg in segs:
                seg_i, _bg, seg_start, _seg_end, _mlen = seg
                if abs(float(seg_start) - float(barline_t)) <= 1e-6:
                    target_seg = seg
                    break

            inserted_seg_index: int | None = None
            if target_seg is None:
                parent_seg = None
                for seg in segs:
                    _seg_i, _bg, seg_start, seg_end, _mlen = seg
                    if seg_start < float(barline_t) < seg_end:
                        parent_seg = seg
                        break
                if parent_seg is None:
                    # Allow insertion at terminal end barline by appending a new segment.
                    bars = self._all_barlines(score)
                    terminal = float(bars[-1]) if bars else 0.0
                    if abs(float(barline_t) - terminal) > 1e-6:
                        return
                    last_i, last_bg, _last_start, _last_end, _last_mlen = segs[-1]
                    new_bg = copy.deepcopy(last_bg)
                    new_bg.measure_amount = 1
                    score.base_grid.append(new_bg)
                    inserted_seg_index = len(score.base_grid) - 1
                    target_seg = (
                        inserted_seg_index,
                        new_bg,
                        float(terminal),
                        float(terminal) + self._measure_len_ticks(new_bg) * float(new_bg.measure_amount),
                        self._measure_len_ticks(new_bg),
                    )
                    seg_i = last_i
                else:
                    seg_i, bg, seg_start, _seg_end, mlen = parent_seg
                    mcount = int(getattr(bg, 'measure_amount', 1) or 1)
                    split_measures = int(round((float(barline_t) - float(seg_start)) / max(1e-9, float(mlen))))
                    split_measures = max(1, min(mcount - 1, split_measures))
                    if split_measures <= 0 or split_measures >= mcount:
                        return

                    old_bg = bg
                    new_bg = copy.deepcopy(old_bg)
                    old_bg.measure_amount = int(split_measures)
                    new_bg.measure_amount = int(mcount - split_measures)
                    score.base_grid.insert(seg_i + 1, new_bg)
                    inserted_seg_index = seg_i + 1
                    target_seg = (seg_i + 1, new_bg, float(barline_t), float(barline_t) + self._measure_len_ticks(new_bg) * new_bg.measure_amount, self._measure_len_ticks(new_bg))

            seg_i, seg_bg, _seg_start, _seg_end, _mlen = target_seg

            initial_grid = self._grid_positions_as_times(seg_bg)
            dlg = TimeSignatureDialog(
                parent=QtWidgets.QApplication.activeWindow(),
                initial_numer=int(getattr(seg_bg, 'numerator', 4) or 4),
                initial_denom=int(getattr(seg_bg, 'denominator', 4) or 4),
                initial_grid_positions=list(initial_grid),
                initial_indicator_enabled=bool(getattr(seg_bg, 'indicator_enabled', True)),
                editor_widget=getattr(self._editor, 'widget', None),
            )

            if dlg.exec() != QtWidgets.QDialog.Accepted:
                if inserted_seg_index is not None:
                    try:
                        del score.base_grid[inserted_seg_index]
                    except Exception:
                        pass
                    try:
                        if inserted_seg_index - 1 >= 0:
                            prev_bg = score.base_grid[inserted_seg_index - 1]
                            prev_bg.measure_amount = int(prev_bg.measure_amount) + int(seg_bg.measure_amount)
                    except Exception:
                        pass
                self._editor.force_redraw_from_model()
                return

            numer, denom, grid_positions, indicator_enabled = dlg.get_values()
            seg_bg.numerator = int(numer)
            seg_bg.denominator = int(denom)
            seg_bg.beat_grouping = [float(v) for v in (grid_positions or [])]
            seg_bg.indicator_enabled = bool(indicator_enabled)

            self._editor._snapshot_if_changed(coalesce=False, label='time_signature_change')
            self._editor.update_score_length()
            self._editor.force_redraw_from_model()
        finally:
            self._dialog_open = False
            self._dialog_cooldown_until = time.monotonic() + 0.20

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        if self._editor is None:
            return
        if time.monotonic() < float(getattr(self, '_dialog_cooldown_until', 0.0) or 0.0):
            return
        score: SCORE | None = self._editor.current_score()
        if score is None:
            return

        click_t = float(max(0.0, self._editor.snap_time(self._editor.y_to_time(y))))
        nearest_barline, on_barline = self._nearest_barline(score, click_t)

        if on_barline:
            # Defer dialog opening until mouse-release processing fully completes.
            try:
                QtCore.QTimer.singleShot(
                    0,
                    lambda sc=score, t=nearest_barline: self._open_time_signature_dialog_at_barline(sc, t),
                )
            except Exception:
                self._open_time_signature_dialog_at_barline(score, nearest_barline)
            return

        self._edit_grid_value(score, click_t, x, delete=False)

    def on_right_unpress(self, x: float, y: float) -> None:
        super().on_right_unpress(x, y)
        if self._editor is None:
            return
        score: SCORE | None = self._editor.current_score()
        if score is None:
            return

        click_t = float(max(0.0, self._editor.snap_time(self._editor.y_to_time(y))))
        nearest_barline, on_barline = self._nearest_barline(score, click_t)
        if on_barline:
            # Delete time-signature change if this barline is a segment start (except first segment).
            segs = self._segments(score)
            delete_idx: int | None = None
            for seg_i, _bg, seg_start, _seg_end, _mlen in segs:
                if abs(float(seg_start) - float(nearest_barline)) <= 1e-6:
                    delete_idx = int(seg_i)
                    break
            if delete_idx is None or delete_idx <= 0:
                return

            try:
                base_list = list(getattr(score, 'base_grid', []) or [])
                if delete_idx >= len(base_list):
                    return
                removed = base_list.pop(delete_idx)
                prev = base_list[delete_idx - 1]
                prev.measure_amount = int(getattr(prev, 'measure_amount', 1) or 1) + int(getattr(removed, 'measure_amount', 1) or 1)
                score.base_grid = base_list
                self._editor.update_score_length()
                self._editor.force_redraw_from_model()
            except Exception:
                pass
            return

        self._edit_grid_value(score, click_t, x, delete=True)
