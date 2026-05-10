from __future__ import annotations

import copy
import dataclasses
import math

from PySide6 import QtCore

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator
from utils.tiny_tool import key_class_filter


class SelectionMixin:
    def _init_selection_state(self) -> None:
        # Selection window state (time-based, tool-agnostic)
        self._selection_active = False
        self._sel_start_units = 0.0
        self._sel_end_units = 0.0
        # Anchor time for selection (absolute ticks, unaffected by scroll)
        self._sel_anchor_units = 0.0
        # Pitch-constrained selection (1..88)
        self._sel_min_pitch = 1
        self._sel_max_pitch = 88
        self._sel_anchor_pitch = 1
        # Clipboard for cut/copy/paste of detected events
        self.clipboard = None
        self._clipboard_start_units = None
        # Debounced snapshot for fast transpose bursts
        self._transpose_timer = QtCore.QTimer(self)
        self._transpose_timer.setSingleShot(True)
        self._transpose_timer.timeout.connect(self._finalize_transpose_snapshot)
        self._pending_snapshot_label = 'transpose_notes'

    def _queue_transpose_snapshot(self, delay_ms: int = 200, label: str = 'transpose_notes') -> None:
        """Debounce transpose snapshots to avoid heavy work on every keypress."""
        if self._transpose_timer.isActive():
            self._transpose_timer.stop()
        self._pending_snapshot_label = str(label or 'transpose_notes')
        self._transpose_timer.start(int(max(1, delay_ms)))

    def _finalize_transpose_snapshot(self) -> None:
        label = getattr(self, '_pending_snapshot_label', 'transpose_notes')
        self._snapshot_if_changed(coalesce=True, label=str(label or 'transpose_notes'))

    def _begin_selection_drag(self, x: float, y: float) -> None:
        anchor_t = self.snap_time(self.widget_px_to_time(x, y))
        self._sel_anchor_units = float(anchor_t)
        self._sel_start_units = float(anchor_t)
        snap_units = max(1e-6, float(self.snap_size_units))
        self._sel_end_units = float(anchor_t + snap_units)
        anchor_p = int(self.widget_px_to_pitch(x, y))
        anchor_p = max(1, min(88, anchor_p))
        self._sel_anchor_pitch = anchor_p
        self._sel_min_pitch = anchor_p
        self._sel_max_pitch = anchor_p
        self._selection_active = True

    def _update_selection_drag(self, x: float, y: float) -> None:
        cur_t = self.snap_time(self.widget_px_to_time(x, y))
        anchor_t = float(self._sel_anchor_units)
        snap_units = max(1e-6, float(self.snap_size_units))
        if cur_t >= anchor_t:
            self._sel_start_units = float(anchor_t)
            self._sel_end_units = float(cur_t + snap_units)
        else:
            self._sel_start_units = float(cur_t)
            self._sel_end_units = float(anchor_t)
        cur_p = int(self.widget_px_to_pitch(x, y))
        cur_p = max(1, min(88, cur_p))
        anchor_p = int(self._sel_anchor_pitch)
        self._sel_min_pitch = int(min(anchor_p, cur_p))
        self._sel_max_pitch = int(max(anchor_p, cur_p))
        self._selection_active = True

    def _draw_selection_overlay(self, du) -> None:
        if not self._selection_active:
            return
        y1_mm = float(self.time_to_mm(float(self._sel_start_units)))
        y2_mm = float(self.time_to_mm(float(self._sel_end_units)))
        sel_top_mm = min(y1_mm, y2_mm)
        sel_bottom_mm = max(y1_mm, y2_mm)
        vp_top = float(self._view_y_mm_offset or 0.0)
        vp_bottom = vp_top + float(self._viewport_h_mm or 0.0)
        draw_top = max(sel_top_mm, vp_top)
        draw_bottom = min(sel_bottom_mm, vp_bottom)
        if draw_bottom <= draw_top:
            return
        min_p = max(1, min(88, int(self._sel_min_pitch)))
        max_p = max(1, min(88, int(self._sel_max_pitch)))
        x_left = float(self.pitch_to_x(min_p))
        x_right = float(self.pitch_to_x(max_p))
        x2 = min(x_left, x_right)
        x1 = max(x_left, x_right)
        du.add_rectangle(
            x2,
            draw_top,
            x1,
            draw_bottom,
            stroke_color=None,
            fill_color=self.selection_color,
            id=0,
            tags=['selection_rect'],
        )

    def set_selection_window(self, start_units: float, end_units: float, active: bool = True) -> None:
        """Programmatically set the selection window in ticks and toggle its visibility."""
        self._sel_start_units = float(start_units)
        self._sel_end_units = float(end_units)
        self._selection_active = bool(active)

    def clear_selection(self) -> None:
        """Clear selection window and clipboard."""
        self._selection_active = False
        self._sel_start_units = 0.0
        self._sel_end_units = 0.0
        self._sel_anchor_units = 0.0
        self._sel_min_pitch = 1
        self._sel_max_pitch = 88
        self._sel_anchor_pitch = 1
        # Persistent clipboard is not cleared here

    def select_all(self) -> None:
        """Select the full score range and all pitches."""
        total_len = float(self._calc_base_grid_list_total_length())
        snap_units = max(1e-6, float(getattr(self, 'snap_size_units', 1.0) or 1.0))
        end_units = float(total_len) if total_len > snap_units else float(snap_units)
        self._sel_anchor_units = 0.0
        self._sel_start_units = 0.0
        self._sel_end_units = end_units
        self._sel_anchor_pitch = 1
        self._sel_min_pitch = 1
        self._sel_max_pitch = 88
        self._selection_active = True

    def _selected_notes_from_model(self, include_grace: bool = False) -> list:
        """Return selected pitch events resolved from the live SCORE model."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return []
        a = float(min(self._sel_start_units, self._sel_end_units))
        b = float(max(self._sel_start_units, self._sel_end_units)) - 0.1
        min_p = max(1, min(88, int(getattr(self, '_sel_min_pitch', 1))))
        max_p = max(1, min(88, int(getattr(self, '_sel_max_pitch', 88))))
        op = Operator(SHORTEST_DURATION)
        out = []
        event_lists = [list(getattr(score.events, 'note', []) or [])]
        if bool(include_grace):
            event_lists.append(list(getattr(score.events, 'grace_note', []) or []))

        for lst in event_lists:
            for note in lst:
                try:
                    t0 = float(getattr(note, 'time', 0.0) or 0.0)
                    p0 = int(getattr(note, 'pitch', 0) or 0)
                except Exception:
                    continue
                if op.ge(t0, a) and t0 <= b and min_p <= p0 <= max_p:
                    out.append(note)
        return out

    def _transpose_step_units_for_anchor_key(self, anchor_key: int, direction: int) -> int:
        """Return editor X-step units for one semitone at `anchor_key`."""
        key_now = max(1, min(88, int(anchor_key)))
        if direction == 0:
            return 0
        be_gap_keys = set(int(k) for k in key_class_filter('be'))
        if direction > 0:
            extra = 1 if key_now in be_gap_keys else 0
            return 1 + extra
        extra = 1 if (key_now - 1) in be_gap_keys else 0
        return -(1 + extra)

    def _shift_relative_pitch_with_gaps(self, rpitch: float, delta_semitones: int) -> float:
        """Shift a C4-relative pitch offset while honoring stave gap crossings."""
        delta = int(delta_semitones)
        if delta == 0:
            return float(rpitch)

        rp_base = int(round(float(rpitch)))
        rp_frac = float(rpitch) - float(rp_base)
        anchor_key = max(1, min(88, 40 + rp_base))

        units = 0
        step_dir = 1 if delta > 0 else -1
        steps = abs(delta)
        key_cursor = int(anchor_key)
        for _ in range(steps):
            units += self._transpose_step_units_for_anchor_key(key_cursor, step_dir)
            key_cursor = max(1, min(88, key_cursor + step_dir))

        return float(rp_base + units) + rp_frac

    def transpose_selected_notes(self, delta_semitones: int) -> bool:
        """Move selected notes and selected slur/text X by semitone steps."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        delta = int(delta_semitones)
        if delta == 0:
            return False
        sel = self.detect_events_from_time_window(
            float(self._sel_start_units),
            float(self._sel_end_units) - 0.1,
        )
        notes = self._selected_notes_from_model(include_grace=True)
        updated = False

        for note in notes:
            pitch = int(getattr(note, 'pitch', 0) or 0)
            if pitch <= 0:
                continue
            new_pitch = max(1, min(88, pitch + delta))
            if new_pitch != pitch:
                setattr(note, 'pitch', int(new_pitch))
                updated = True

        for slur in list(sel.get('slur', []) or []):
            for attr in ('x1_rpitch', 'x2_rpitch', 'x3_rpitch', 'x4_rpitch'):
                old_rp = float(getattr(slur, attr, 0) or 0)
                new_rp = int(round(self._shift_relative_pitch_with_gaps(old_rp, delta)))
                if new_rp != int(round(old_rp)):
                    setattr(slur, attr, int(new_rp))
                    updated = True

        for text in list(sel.get('text', []) or []):
            old_rp = float(getattr(text, 'x_rpitch', 0) or 0)
            new_rp = self._shift_relative_pitch_with_gaps(old_rp, delta)
            if not math.isclose(new_rp, old_rp, abs_tol=1e-9):
                setattr(text, 'x_rpitch', float(new_rp))
                updated = True

        if not updated:
            return False
        self._sel_min_pitch = max(1, min(88, int(self._sel_min_pitch) + delta))
        self._sel_max_pitch = max(1, min(88, int(self._sel_max_pitch) + delta))
        self._sel_anchor_pitch = max(1, min(88, int(self._sel_anchor_pitch) + delta))
        widget = getattr(self, 'widget', None)
        if widget is not None and hasattr(widget, 'force_full_redraw'):
            widget.force_full_redraw()
        self._queue_transpose_snapshot()
        return True

    def shift_selected_notes_time(self, delta_units: float) -> bool:
        """Shift selected content in time by one shared delta."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        dt = float(delta_units)
        if abs(dt) < 1e-9:
            return False

        sel = self.detect_events_from_time_window(
            float(self._sel_start_units),
            float(self._sel_end_units) - 0.1,
        )
        notes = self._selected_notes_from_model(include_grace=True)
        texts = list(sel.get('text', []) or [])
        slurs = list(sel.get('slur', []) or [])

        min_time = min(
            [float(self._sel_start_units), float(self._sel_end_units), float(self._sel_anchor_units)]
            + [float(getattr(note, 'time', 0.0) or 0.0) for note in notes]
            + [float(getattr(text, 'time', 0.0) or 0.0) for text in texts]
            + [
                float(getattr(slur, attr, 0.0) or 0.0)
                for slur in slurs
                for attr in ('y1_time', 'y2_time', 'y3_time', 'y4_time')
            ]
        )
        if dt < 0.0:
            dt = max(float(dt), -float(min_time))
            if abs(dt) < 1e-9:
                return False

        model_updated = False

        for note in notes:
            old_time = float(getattr(note, 'time', 0.0) or 0.0)
            new_time = max(0.0, old_time + dt)
            if not math.isclose(new_time, old_time, abs_tol=1e-9):
                setattr(note, 'time', new_time)
                model_updated = True

        for text in texts:
            old_time = float(getattr(text, 'time', 0.0) or 0.0)
            new_time = max(0.0, old_time + dt)
            if not math.isclose(new_time, old_time, abs_tol=1e-9):
                setattr(text, 'time', float(new_time))
                model_updated = True

        for slur in slurs:
            changed = False
            for attr in ('y1_time', 'y2_time', 'y3_time', 'y4_time'):
                old_time = float(getattr(slur, attr, 0.0) or 0.0)
                new_time = max(0.0, old_time + dt)
                if not math.isclose(new_time, old_time, abs_tol=1e-9):
                    setattr(slur, attr, float(new_time))
                    changed = True
            if changed:
                model_updated = True

        self._sel_start_units = max(0.0, float(self._sel_start_units) + dt)
        self._sel_end_units = max(0.0, float(self._sel_end_units) + dt)
        self._sel_anchor_units = max(0.0, float(self._sel_anchor_units) + dt)

        units = float(max(1e-6, getattr(self, 'snap_size_units', 0.0) or 0.0))
        if self._sel_end_units <= self._sel_start_units:
            self._sel_end_units = float(self._sel_start_units) + units

        if model_updated:
            self.update_score_length()
            self._reuse_draw_cache_once = True
        widget = getattr(self, 'widget', None)
        if widget is not None and hasattr(widget, 'force_full_redraw'):
            widget.force_full_redraw()
        if model_updated:
            self._queue_transpose_snapshot(label='shift_selected_notes_time')
        return True

    def quantize_selected_notes(self, qtype: str = 'start/end') -> bool:
        """Quantize selected notes to current snap size."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        mode = str(qtype or 'start/end').strip().lower()
        if mode not in ('start/end', 'start', 'end'):
            mode = 'start/end'
        units = float(max(1e-6, getattr(self, 'snap_size_units', 0.0) or 0.0))
        notes = self._selected_notes_from_model()
        if not notes:
            return False

        def _q(value: float) -> float:
            return float(round(float(value) / units) * units)

        updated = False
        op = Operator(1.0)
        for note in notes:
            t0 = float(getattr(note, 'time', 0.0) or 0.0)
            dur = float(getattr(note, 'duration', 0.0) or 0.0)
            t1 = t0 + max(0.0, dur)

            qt0 = max(0.0, _q(t0)) if mode in ('start/end', 'start') else t0
            qt1 = max(0.0, _q(t1)) if mode in ('start/end', 'end') else t1
            if qt1 <= qt0:
                qt1 = qt0 + units
            qdur = max(units, qt1 - qt0)

            if (not op.eq(qt0, t0)) or (not op.eq(qdur, dur)):
                setattr(note, 'time', float(qt0))
                setattr(note, 'duration', float(qdur))
                updated = True

        if not updated:
            return False

        self.update_score_length()
        self._snapshot_if_changed(coalesce=True, label='quantize_selected_notes')
        return True

    def set_selected_notes_hand(self, hand: str) -> bool:
        """Assign selected notes to a hand and snapshot the change."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        hand_value = str(hand)
        if hand_value not in ('l', 'r'):
            return False
        notes = self._selected_notes_from_model()
        if not notes:
            return False
        updated = False
        for note in notes:
            note_hand = str(getattr(note, 'hand', '') or '')
            note_color = str(getattr(note, 'color', '') or '')
            if note_hand != hand_value or note_color != 'auto':
                setattr(note, 'hand', hand_value)
                setattr(note, 'color', 'auto')
                updated = True
        if updated:
            self._snapshot_if_changed(coalesce=True, label='set_note_hand')
        return updated

    @property
    def selection_window_start(self) -> float:
        return float(self._sel_start_units)

    @selection_window_start.setter
    def selection_window_start(self, value: float) -> None:
        self._sel_start_units = float(value)
        self._sel_anchor_units = float(value)

    @property
    def selection_window_end(self) -> float:
        return float(self._sel_end_units)

    @selection_window_end.setter
    def selection_window_end(self, value: float) -> None:
        self._sel_end_units = float(value)

    def get_selected_note_ids_cached(self, score: SCORE | None = None) -> set[int]:
        """Return selected note IDs using draw-cache data when possible."""
        if not bool(getattr(self, '_selection_active', False)):
            return set()
        if score is None:
            score = self.current_score()
        if score is None:
            return set()

        a = float(min(self._sel_start_units, self._sel_end_units))
        b = float(max(self._sel_start_units, self._sel_end_units)) - 0.1
        min_p = max(1, min(88, int(getattr(self, '_sel_min_pitch', 1))))
        max_p = max(1, min(88, int(getattr(self, '_sel_max_pitch', 88))))
        op = Operator(SHORTEST_DURATION)

        cache = getattr(self, '_draw_cache', None) or {}
        sel_sig = (round(a, 6), round(b, 6), int(min_p), int(max_p))
        cached_sig = cache.get('selected_note_ids_sig') if isinstance(cache, dict) else None
        cached_ids = cache.get('selected_note_ids') if isinstance(cache, dict) else None
        if cached_sig == sel_sig and isinstance(cached_ids, set):
            return cached_ids

        t_begin = float(cache.get('time_begin', float('inf'))) if isinstance(cache, dict) else float('inf')
        t_end = float(cache.get('time_end', float('-inf'))) if isinstance(cache, dict) else float('-inf')
        if a >= t_begin and b <= t_end and isinstance(cache, dict):
            candidates = list(cache.get('notes_view') or [])
        else:
            candidates = list(getattr(score.events, 'note', []) or [])

        ids = set()
        for note in candidates:
            try:
                st = float(getattr(note, 'time', 0.0) or 0.0)
                sp = int(getattr(note, 'pitch', 0) or 0)
                sid = int(getattr(note, '_id', 0) or 0)
            except Exception:
                continue
            if sid <= 0:
                continue
            if op.ge(st, a) and st <= b and min_p <= sp <= max_p:
                ids.add(sid)

        if isinstance(cache, dict):
            cache['selected_note_ids_sig'] = sel_sig
            cache['selected_note_ids'] = ids
        return ids

    def detect_events_from_time_window(self, start_units: float, end_units: float) -> dict:
        """Return SCORE events whose start time falls within the current selection window."""
        score: SCORE | None = self.current_score()
        if score is None:
            return {}
        a = float(min(start_units, end_units))
        b = float(max(start_units, end_units))
        op = Operator(SHORTEST_DURATION)
        min_p = max(1, min(88, int(getattr(self, '_sel_min_pitch', 1))))
        max_p = max(1, min(88, int(getattr(self, '_sel_max_pitch', 88))))

        event_fields = [f.name for f in dataclasses.fields(type(score.events))]
        event_fields = [name for name in event_fields if name not in ('tempo', 'line_break')]

        out = {name: [] for name in event_fields}

        def start_time(ev) -> float:
            if hasattr(ev, 'time'):
                return float(getattr(ev, 'time', 0.0) or 0.0)
            data = dataclasses.asdict(ev)
            times = [float(v or 0.0) for k, v in data.items() if k.endswith('_time')]
            if times:
                return float(min(times))
            return 0.0

        def pitch_ok(ev) -> bool:
            if hasattr(ev, 'pitch'):
                pitch = int(getattr(ev, 'pitch', 0) or 0)
                return pitch and (min_p <= pitch <= max_p)
            return True

        cached_notes_view = None
        cache = getattr(self, '_draw_cache', None) or {}
        t_begin = float(cache.get('time_begin', float('inf')))
        t_end = float(cache.get('time_end', float('-inf')))
        if a >= t_begin and b <= t_end:
            cached_notes_view = cache.get('notes_view') or []

        for name in event_fields:
            if name == 'note' and cached_notes_view is not None:
                event_list = list(cached_notes_view)
            else:
                event_list = getattr(score.events, name, []) or []
            if name == 'slur':
                for event in event_list:
                    rpitches = [
                        int(getattr(event, 'x1_rpitch', 0) or 0),
                        int(getattr(event, 'x2_rpitch', 0) or 0),
                        int(getattr(event, 'x3_rpitch', 0) or 0),
                        int(getattr(event, 'x4_rpitch', 0) or 0),
                    ]
                    times_h = [
                        float(getattr(event, 'y1_time', 0.0) or 0.0),
                        float(getattr(event, 'y2_time', 0.0) or 0.0),
                        float(getattr(event, 'y3_time', 0.0) or 0.0),
                        float(getattr(event, 'y4_time', 0.0) or 0.0),
                    ]
                    endpoint_indices = (0, 3)
                    anchor_idx = min(endpoint_indices, key=lambda i: (times_h[i], i))
                    anchor_time = float(times_h[anchor_idx])
                    anchor_key = max(1, min(88, 40 + int(rpitches[anchor_idx])))
                    if (min_p <= anchor_key <= max_p) and (op.ge(anchor_time, a) and anchor_time <= b):
                        out[name].append(event)
            else:
                for event in event_list:
                    t0 = start_time(event)
                    if op.ge(t0, a) and t0 <= b and pitch_ok(event):
                        out[name].append(event)
        return out

    def copy_selection(self) -> dict | None:
        """Copy current selection window events into the editor clipboard and return it."""
        if not self._selection_active:
            return None
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1)
        safe_copy = copy.deepcopy(sel)
        self.clipboard = safe_copy
        self._clipboard_start_units = float(min(self._sel_start_units, self._sel_end_units))
        return safe_copy

    def cut_selection(self) -> dict | None:
        """Cut current selection window events: copy to clipboard, then remove from SCORE."""
        score: SCORE | None = self.current_score()
        if score is None:
            return None
        sel = self.copy_selection()
        if not sel:
            return None
        self.clipboard = sel
        for key in sel:
            event_list = getattr(score.events, key, None)
            if isinstance(event_list, list):
                remain = [event for event in event_list if event not in sel[key]]
                setattr(score.events, key, remain)
        self.update_score_length()
        self._snapshot_if_changed(coalesce=True, label='cut_selection')
        self.score_changed.emit()
        return sel

    def delete_selection(self) -> bool:
        """Delete current selection window events without copying to clipboard."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1)
        deleted_any = False
        for key in sel:
            event_list = getattr(score.events, key, None)
            if isinstance(event_list, list) and sel[key]:
                remain = [event for event in event_list if event not in sel[key]]
                if len(remain) != len(event_list):
                    deleted_any = True
                setattr(score.events, key, remain)
        if deleted_any:
            self.update_score_length()
            self._snapshot_if_changed(coalesce=True, label='delete_selection')
            self.score_changed.emit()
        self.clear_selection()
        return deleted_any

    def paste_selection_at_cursor(self) -> None:
        """Paste events from clipboard so copied selection start aligns to `self.time_cursor`."""
        score: SCORE | None = self.current_score()
        if score is None or self.clipboard is None:
            return
        if self.time_cursor is None:
            return

        source_start = self._clipboard_start_units
        if source_start is None:
            source_start = float(min(self._sel_start_units, self._sel_end_units))
        target = float(self.time_cursor)
        delta = float(target - source_start)

        furthest_end = float(self._calc_base_grid_list_total_length())

        for ev_type, items in (self.clipboard.items() if isinstance(self.clipboard, dict) else []):
            if not items:
                continue
            if ev_type == 'arpeggio':
                continue
            ctor = getattr(score, f'new_{ev_type}', None)
            if ctor is None:
                continue
            for event in items:
                data = dataclasses.asdict(event)
                data.pop('_id', None)
                for key in list(data.keys()):
                    if key == 'time' or key.endswith('_time'):
                        data[key] = float(data.get(key, 0.0)) + delta
                ctor(**data)
                time_fields = [float(v or 0.0) for name, v in data.items() if name == 'time' or name.endswith('_time')]
                t_end = max(time_fields) if time_fields else float(data.get('time', 0.0) or 0.0)
                dur = float(data.get('duration', 0.0) or 0.0)
                if dur > 0.0 and 'time' in data:
                    t_end = float(data.get('time', 0.0) or 0.0) + dur
                furthest_end = max(furthest_end, float(t_end))
        cur_end = float(self._calc_base_grid_list_total_length())
        if furthest_end > cur_end:
            bg_list = list(getattr(score, 'base_grid', []) or [])
            if bg_list:
                last_bg = bg_list[-1]
                num = float(getattr(last_bg, 'numerator', 4) or 4)
                den = float(getattr(last_bg, 'denominator', 4) or 4)
                measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
                extra_measures = int(max(1, math.ceil((furthest_end - cur_end) / max(1e-6, measure_len))))
                last_bg.measure_amount = int(getattr(last_bg, 'measure_amount', 1) or 1) + extra_measures
        self.update_score_length()
        self._snapshot_if_changed(coalesce=True, label='paste_selection')
        self.score_changed.emit()
        self._selection_active = False