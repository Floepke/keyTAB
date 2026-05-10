from typing import Optional
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from settings_manager import get_preferences
from utils.operator import Operator
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from file_model.events.note import Note
from file_model.events.arpeggio import Arpeggio
from symbol_design.noteheads import resolve_notehead_spec
from ui.dialogs.notehead_dialog import NoteheadDialog
from PySide6 import QtCore, QtGui, QtWidgets


class NoteTool(BaseTool):
    TOOL_NAME = 'note'
    _ARP_TYPES = ["up/ending", "down/ending", "up/starting", "down/starting"]
    _arp_op: Operator = Operator(float(SHORTEST_DURATION))
    def __init__(self):
        super().__init__()
        self._acc_cycle: tuple[int, ...] = (0, 1, -1, 2, -2)
        self._acc_toggle: int = 0
        # Currently edited/created note during a press/drag session
        self.edit_note = None
        self._hand: str = 'l'
        self.expanded_score_flag: bool = False
        # Drag-session context
        self._editing_existing: bool = False
        self._orig_duration: float = 0.0
        self._press_start_time: float = 0.0
        self._duration_edit_armed: bool = False
        self._last_audition_pitch: int | None = None
        self._move_pitch_time_mode: bool = False
        self._velocity_mode: bool = False
        self._velocity_dragging: bool = False
        self._velocity_target: Note | None = None
        self._velocity_targets: list[Note] = []
        self._velocity_display_value: int | None = None
        self._velocity_display_x_mm: float | None = None
        self._velocity_display_y_mm: float | None = None
        self._ignore_next_left_release: bool = False
        self._suppress_note_edit_until_release: bool = False
        self._notehead_dialog_active: bool = False
        self._pending_notehead_dialog_note_id: int | None = None
        # Arpeggio edit session
        self._arpeggio_dragging: bool = False
        self._arpeggio_target: Arpeggio | None = None
        self._arpeggio_drag_anchor_time: float = 0.0

    def _play_note_on_edit_enabled(self) -> bool:
        try:
            from settings_manager import get_preferences_manager
            pm = get_preferences_manager()
            return bool(pm.get("play_note_on_edit", True))
        except Exception:
            return True

    def _audition_pitch(self, pitch: int) -> None:
        if not self._play_note_on_edit_enabled():
            return
        if hasattr(self._editor, 'player') and self._editor.player is not None:
            self._editor.player.audition_note(pitch=int(pitch))
            self._last_audition_pitch = int(pitch)

    def _arpeggio_button_text(self) -> tuple[str, str]:
        current = self._current_arpeggio_type()
        if current:
            return ("Arp", QtCore.QCoreApplication.translate('NoteTool', 'Cycle arpeggio (current: {0})').format(current))
        return ("Arp", QtCore.QCoreApplication.translate('NoteTool', 'Cycle arpeggio on selected chord (currently off)'))

    def _current_arpeggio_type(self) -> str | None:
        chord_notes, base_time = self._find_chord_notes()
        if not chord_notes:
            return None
        score = self._editor.current_score()
        if score is None:
            return None
        arp = self._find_arpeggio_for_notes(score, chord_notes, base_time)
        if arp is None:
            return None
        kind = str(getattr(arp, 'type', '') or '')
        return kind if kind in self._ARP_TYPES else None

    def _find_chord_notes(self) -> tuple[list[Note], float | None]:
        if self._editor is None:
            return ([], None)
        score: SCORE | None = self._editor.current_score()
        if score is None:
            return ([], None)

        op = self._arp_op

        def _cluster(notes: list[Note]) -> tuple[list[Note], float | None]:
            clusters: list[tuple[float, list[Note]]] = []
            for n in notes:
                t = float(getattr(n, 'time', 0.0) or 0.0)
                placed = False
                for idx, (ct, lst) in enumerate(clusters):
                    if op.eq(ct, t):
                        lst.append(n)
                        placed = True
                        break
                if not placed:
                    clusters.append((t, [n]))
            if not clusters:
                return ([], None)
            clusters.sort(key=lambda c: len(c[1]), reverse=True)
            t0, members = clusters[0]
            if len(members) < 2:
                return ([], None)
            hands = {str(getattr(m, 'hand', 'l') or 'l') for m in members}
            if len(hands) > 1:
                return ([], None)
            return (members, float(t0))

        # Prefer an active selection window if present
        try:
            sel_active = bool(getattr(self._editor, '_selection_active', False))
        except Exception:
            sel_active = False
        if sel_active:
            start = float(getattr(self._editor, '_sel_start_units', 0.0) or 0.0)
            end = float(getattr(self._editor, '_sel_end_units', 0.0) or 0.0)
            sel = self._editor.detect_events_from_time_window(start, end - 0.1)
            notes = sel.get('note', []) if isinstance(sel, dict) else []
            found, t0 = _cluster(list(notes))
            if found:
                return (found, t0)

        # Fallback: use time cursor if available
        cursor_t = getattr(self._editor, 'time_cursor', None)
        if cursor_t is None:
            return ([], None)
        grouped: dict[float, list[Note]] = {}
        for n in getattr(score.events, 'note', []) or []:
            t = float(getattr(n, 'time', 0.0) or 0.0)
            if op.eq(t, float(cursor_t)):
                grouped.setdefault(t, []).append(n)
        if not grouped:
            return ([], None)
        grouped_items = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
        t_sel, lst = grouped_items[0]
        if len(lst) < 2:
            return ([], None)
        hands = {str(getattr(m, 'hand', 'l') or 'l') for m in lst}
        if len(hands) > 1:
            return ([], None)
        return (lst, float(t_sel))

    def _find_arpeggio_for_notes(self, score: SCORE, notes: list[Note], base_time: float | None) -> Arpeggio | None:
        if base_time is None:
            return None
        op = self._arp_op
        target_pitches = sorted(int(getattr(n, 'pitch', 0) or 0) for n in notes)
        if len(target_pitches) < 2:
            return None
        for arp in getattr(score.events, 'arpeggio', []) or []:
            t = float(getattr(arp, 'time', 0.0) or 0.0)
            if not op.eq(t, float(base_time)):
                continue
            arp_pitches = sorted(int(i) for i in (getattr(arp, 'notes', []) or []) if int(i) != 0 or i == 0)
            if arp_pitches == target_pitches:
                return arp
        return None

    def _find_arpeggio_by_id(self, score: SCORE, arp_id: int) -> Arpeggio | None:
        for arp in getattr(score.events, 'arpeggio', []) or []:
            if int(getattr(arp, '_id', -1) or -1) == int(arp_id):
                return arp
        return None

    def _cleanup_arpeggios(self, score: SCORE) -> None:
        notes_all = list(getattr(score.events, 'note', []) or [])
        cleaned: list[Arpeggio] = []
        op = self._arp_op
        for arp in getattr(score.events, 'arpeggio', []) or []:
            base_time = float(getattr(arp, 'time', 0.0) or 0.0)
            target_pitches = list(getattr(arp, 'notes', []) or [])
            if len(target_pitches) < 2:
                continue
            chord_notes = [n for n in notes_all if op.eq(float(getattr(n, 'time', 0.0) or 0.0), base_time)]
            # Match pitches with multiplicity
            remaining = list(int(p) for p in target_pitches)
            matched: list[int] = []
            for n in sorted(chord_notes, key=lambda m: int(getattr(m, 'pitch', 0) or 0)):
                p = int(getattr(n, 'pitch', 0) or 0)
                if p in remaining:
                    matched.append(p)
                    remaining.remove(p)
            if len(matched) < 2:
                continue
            arp.notes = sorted(matched)
            cleaned.append(arp)
        score.events.arpeggio = cleaned

    def _toggle_arpeggio(self) -> bool:
        score: SCORE | None = self._editor.current_score()
        if score is None:
            return False
        chord_notes, base_time = self._find_chord_notes()
        if not chord_notes or base_time is None:
            return False
        existing = self._find_arpeggio_for_notes(score, chord_notes, base_time)
        order = self._ARP_TYPES
        next_type: str | None
        if existing is None:
            next_type = order[0]
        else:
            try:
                idx = order.index(str(getattr(existing, 'type', order[0]) or order[0]))
            except ValueError:
                idx = -1
            next_type = order[idx + 1] if 0 <= idx < len(order) - 1 else None

        note_pitches = sorted(int(getattr(n, 'pitch', 0) or 0) for n in chord_notes)
        if len(note_pitches) < 2:
            return False
        if next_type is None and existing is not None:
            try:
                score.events.arpeggio.remove(existing)
            except ValueError:
                score.events.arpeggio = [a for a in getattr(score.events, 'arpeggio', []) or [] if a is not existing]
        else:
            if existing is None:
                default_duration = 32.0
                existing = score.new_arpeggio(time=float(base_time), duration=default_duration, notes=note_pitches, type=next_type or order[0])
            else:
                existing.type = next_type or order[0]
            existing.time = float(base_time)
            existing.notes = note_pitches
            if float(getattr(existing, 'duration', 0.0) or 0.0) <= 0.0:
                existing.duration = 32.0
        self._cleanup_arpeggios(score)
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
        if hasattr(self._editor, '_snapshot_if_changed'):
            self._editor._snapshot_if_changed(coalesce=True, label='arpeggio_toggle')
        return True

    def _update_arpeggio_duration_from_cursor(self, x: float, y: float) -> None:
        if self._editor is None or self._arpeggio_target is None:
            return
        base_time = float(getattr(self._arpeggio_target, 'time', self._arpeggio_drag_anchor_time) or self._arpeggio_drag_anchor_time)
        kind = str(getattr(self._arpeggio_target, 'type', 'up/starting') or 'up/starting')
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        step = float(max(1e-6, getattr(self._editor, 'snap_size_units', SHORTEST_DURATION)))
        if kind.endswith('ending'):
            new_duration = max(step, float(base_time) - t_snap)
        else:
            new_duration = max(step, t_snap - float(base_time))
        self._arpeggio_target.duration = float(new_duration)
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def toolbar_spec(self) -> list[dict]:
        # Two explicit hand selectors for quick switching
        _text, _tip = self._arpeggio_button_text()
        hand = str(getattr(self._editor, 'hand_cursor', self._hand) or self._hand)
        
        # read editor orientation
        preferences = get_preferences()
        if preferences.get("editor_orientation", 'vertical') == 'horizontal':
            editor_orientation = 'horizontal'
        else:
            editor_orientation = 'vertical'

        return [
            {
                'name': 'hand_right',
                'icon': 'note_right',
                'active': hand == 'r',
                'tooltip': QtCore.QCoreApplication.translate('NoteTool', 'Click to write right hand notes (shortcut: . ).'),
                'rotation': 270.0 if editor_orientation == 'horizontal' else 0.0,
            },
            {
                'name': 'hand_left',
                'icon': 'note_left',
                'active': hand == 'l',
                'tooltip': QtCore.QCoreApplication.translate('NoteTool', 'Click to write left hand notes (shortcut: , ).'),
                'rotation': 270.0 if editor_orientation == 'horizontal' else 0.0,
            },
            {
                'name': 'selection_right',
                'icon': 'selection_right',
                'tooltip': QtCore.QCoreApplication.translate('NoteTool', 'Set selected notes to right hand (shortcut: ] )'),
                'rotation': 270.0 if editor_orientation == 'horizontal' else 0.0,
            },
            {
                'name': 'selection_left',
                'icon': 'selection_left',
                'tooltip': QtCore.QCoreApplication.translate('NoteTool', 'Set selected notes to left hand (shortcut: [ )'),
                'rotation': 270.0 if editor_orientation == 'horizontal' else 0.0,
            },
            {
                'name': 'velocity_toggle',
                'icon': 'velocity',
                'text': 'Vel',
                'active': bool(self._velocity_mode),
                'tooltip': (QtCore.QCoreApplication.translate('NoteTool', 'Velocity editing is on. Toggle on/off to edit the note velocities using the sliders on the sides of the editor.') if self._velocity_mode else QtCore.QCoreApplication.translate('NoteTool', 'Velocity editing is off. Toggle on/off to edit the note velocities using the sliders on the sides of the editor.')),
            },
            {
                'name': 'arpeggio_toggle',
                'icon': 'arpeggio',
                'text': _text,
                'tooltip': _tip,
            },
        ]

    @property
    def velocity_mode(self) -> bool:
        return bool(self._velocity_mode)

    def _cursor_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _hit_note_and_rect(self, score: SCORE, x_px: float, y_px: float):
        x_mm, y_mm = self._cursor_mm(x_px, y_px)
        hit_rect = self._editor.hit_test_hit_rect(x_mm, y_mm, 'note')
        if hit_rect is None:
            return None, None, y_mm
        hit_id = int(hit_rect.get('_id', -1) or -1)
        for n in getattr(score.events, 'note', []) or []:
            if int(getattr(n, '_id', -1) or -1) == hit_id:
                return n, hit_rect, y_mm
        return None, hit_rect, y_mm

    def _hit_velocity_handle(self, score: SCORE, x_px: float, y_px: float):
        x_mm, y_mm = self._cursor_mm(x_px, y_px)
        hit_rect = self._editor.hit_test_hit_rect(x_mm, y_mm, 'velocity')
        if hit_rect is None:
            return None, None, y_mm
        hit_id = int(hit_rect.get('_id', -1) or -1)
        for n in getattr(score.events, 'note', []) or []:
            if int(getattr(n, '_id', -1) or -1) == hit_id:
                return n, hit_rect, y_mm
        return None, hit_rect, y_mm

    def _resolve_velocity_targets(self, score: SCORE, primary: Note | None) -> list[Note]:
        if primary is None:
            return []
        primary_id = int(getattr(primary, '_id', 0) or 0)
        selected_ids: set[int] = set()
        if hasattr(self._editor, 'get_selected_note_ids_cached'):
            try:
                selected_ids = set(self._editor.get_selected_note_ids_cached(score) or set())
            except Exception:
                selected_ids = set()
        if primary_id > 0 and primary_id in selected_ids:
            out = [n for n in (getattr(score.events, 'note', []) or []) if int(getattr(n, '_id', 0) or 0) in selected_ids]
            if out:
                return out
        return [primary]

    def _apply_velocity_value(self, value: int) -> None:
        if self._editor is None or self._velocity_target is None:
            return
        new_vel = int(max(0, min(127, int(value))))
        targets = list(self._velocity_targets or [])
        if not targets:
            targets = [self._velocity_target]
        for n in targets:
            try:
                n.velocity = new_vel
            except Exception:
                continue
        t = float(getattr(self._velocity_target, 'time', 0.0) or 0.0)
        self._velocity_display_y_mm = float(self._editor.time_to_mm(t))
        self._velocity_display_value = new_vel
        if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
            w = getattr(self._editor, 'widget')
            if hasattr(w, 'request_overlay_refresh'):
                w.request_overlay_refresh()

    def _apply_velocity_from_cursor(self, x_px: float, y_px: float) -> None:
        if self._editor is None or self._velocity_target is None:
            return
        x_mm, _ = self._cursor_mm(x_px, y_px)
        margin = float(getattr(self._editor, 'margin', 12.0) or 12.0)
        stave_width = float(getattr(self._editor, 'stave_width', 120.0) or 120.0)
        max_len = max(2.0, margin * 0.85)
        hand = str(getattr(self._velocity_target, 'hand', 'l') or 'l')
        if hand == 'l':
            dist = max(0.0, float(margin) - float(x_mm))
        else:
            dist = max(0.0, float(x_mm) - float(margin + stave_width))
        ratio = max(0.0, min(1.0, dist / max_len))
        new_vel = int(round((1.0 - ratio) * 127.0))
        self._velocity_display_x_mm = x_mm
        self._apply_velocity_value(new_vel)

    def on_activate(self) -> None:
        super().on_activate()
        self._acc_toggle = 0
        self._velocity_mode = False
        self._velocity_dragging = False
        self._velocity_target = None
        self._velocity_targets = []
        self._velocity_display_value = None
        self._velocity_display_x_mm = None
        self._velocity_display_y_mm = None
        # Refresh overlay to hide velocity sliders on activation
        if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
            w = getattr(self._editor, 'widget')
            if hasattr(w, 'request_overlay_refresh'):
                w.request_overlay_refresh()

    def _can_apply_duration(self, note: Note, candidate_duration: float) -> bool:
        score: SCORE = self._editor.current_score()
        if score is None:
            return True
        start_t = float(getattr(note, 'time', 0.0) or 0.0)
        end_t = float(start_t + max(0.0, float(candidate_duration)))
        pitch = int(getattr(note, 'pitch', 0) or 0)
        note_id = int(getattr(note, '_id', -1) or -1)
        op = self._arp_op

        for other in getattr(score.events, 'note', []) or []:
            other_id = int(getattr(other, '_id', -2) or -2)
            if other_id == note_id:
                continue
            other_pitch = int(getattr(other, 'pitch', 0) or 0)
            if other_pitch != pitch:
                continue
            other_start = float(getattr(other, 'time', 0.0) or 0.0)
            if op.less(start_t, other_start) and op.less(other_start, end_t):
                return False
        return True

    def _can_apply_time_pitch_move(self, note: Note, candidate_time: float, candidate_pitch: int) -> bool:
        score: SCORE = self._editor.current_score()
        if score is None:
            return True
        note_id = int(getattr(note, '_id', -1) or -1)
        start_t = float(max(0.0, candidate_time))
        duration = float(max(0.0, self._orig_duration))
        end_t = float(start_t + duration)
        op = self._arp_op

        for other in getattr(score.events, 'note', []) or []:
            other_id = int(getattr(other, '_id', -2) or -2)
            if other_id == note_id:
                continue
            if int(getattr(other, 'pitch', 0) or 0) != int(candidate_pitch):
                continue
            other_start = float(getattr(other, 'time', 0.0) or 0.0)
            other_duration = float(getattr(other, 'duration', 0.0) or 0.0)
            other_end = float(other_start + max(0.0, other_duration))
            if op.less(start_t, other_end) and op.less(other_start, end_t):
                return False
        return True

    def on_left_press(self, x: float, y: float) -> None:
        '''Detect existing note under cursor or create a new one, then enter edit mode'''
        super().on_left_press(x, y)
        if self._notehead_dialog_active or self._suppress_note_edit_until_release:
            return
        self._velocity_dragging = False
        self._velocity_target = None
        self._velocity_targets = []
        self._velocity_display_value = None
        self._velocity_display_x_mm = None
        self._velocity_display_y_mm = None
        
        score: SCORE = self._editor.current_score()

        # Velocity handle hit test first when velocity editing is enabled
        if self._velocity_mode and score is not None:
            v_note, _v_rect, _y_mm = self._hit_velocity_handle(score, x, y)
            if v_note is not None:
                self._velocity_target = v_note
                self._velocity_targets = self._resolve_velocity_targets(score, v_note)
                self._velocity_dragging = True
                self._editing_existing = True
                self.edit_note = v_note
                # Do not arm duration edits while dragging velocity
                self._duration_edit_armed = False
                self._move_pitch_time_mode = False
                self._apply_velocity_from_cursor(x, y)
                return

        # Arpeggio handle hit test (resize)
        if score is not None and hasattr(self._editor, 'hit_test_arpeggio_handle'):
            arp_id = self._editor.hit_test_arpeggio_handle(x, y)
            if arp_id is not None:
                target = self._find_arpeggio_by_id(score, arp_id)
                if target is not None:
                    self._arpeggio_dragging = True
                    self._arpeggio_target = target
                    self._arpeggio_drag_anchor_time = float(getattr(target, 'time', 0.0) or 0.0)
                    return

        # Compute raw (non-snapped) time for detection and snapped for creation
        t_press_raw = float(self._editor.widget_px_to_time(x, y))
        t_press_snap = float(self._editor.snap_time(t_press_raw))
        pitch_press = int(self._editor.widget_px_to_pitch(x, y))
        self._hand = str(getattr(self._editor, 'hand_cursor', 'l') or 'l')

        # Rectangle-based hit detection for precise clickable area
        found, hit_rect, y_mm_abs = self._hit_note_and_rect(score, x, y)
        mode = "existing" if found else "new"
        if mode == "existing" and found is not None and found.pitch == self._editor.pitch_cursor:
            # Edit existing note
            self.edit_note = found
            self._editing_existing = True
            self._apply_active_accidental_to_note(found)
            self._move_pitch_time_mode = False
            try:
                self._last_audition_pitch = int(getattr(found, 'pitch', pitch_press) or pitch_press)
            except Exception:
                self._last_audition_pitch = int(pitch_press)
            self._audition_pitch(int(getattr(found, 'pitch', pitch_press) or pitch_press))
            try:
                self._orig_duration = float(getattr(found, 'duration', 0.0) or 0.0)
                self._press_start_time = float(getattr(found, 'time', 0.0) or 0.0)
            except Exception:
                self._orig_duration = 0.0
                self._press_start_time = float(t_press_snap)
            self._duration_edit_armed = False
            if hit_rect is not None:
                note_start_mm = float(self._editor.time_to_mm(float(getattr(found, 'time', 0.0) or 0.0)))
                notehead_len_mm = float(getattr(self._editor, 'semitone_dist', 0.0) or 0.0) * 2.0
                notehead_end_mm = note_start_mm + notehead_len_mm
                self._move_pitch_time_mode = bool(notehead_len_mm > 0.0 and y_mm_abs <= notehead_end_mm)
        else:
            # Create a new note at the snapped press time with minimum duration = snap size
            units = float(max(1e-6, getattr(self._editor, 'snap_size_units', 8.0)))
            acc_preview = int(self.preview_accidental_for_pitch(int(pitch_press)))
            self.edit_note = score.new_note(pitch=pitch_press, time=t_press_snap, duration=units, hand=self._hand, acc=acc_preview)
            self._editing_existing = False
            self._orig_duration = float(units)
            self._press_start_time = float(t_press_snap)
            self._duration_edit_armed = False
            self._last_audition_pitch = None
            self._move_pitch_time_mode = False
            self._audition_pitch(pitch_press)

        # switch guides off during note editing
        self._editor.guides_active = False

        # Ensure score length covers latest note end
        self._editor.update_score_length()

        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        if hasattr(self._editor, 'clear_single_note_timing_dirty'):
            self._editor.clear_single_note_timing_dirty()
        if self._suppress_note_edit_until_release:
            self._suppress_note_edit_until_release = False
            self._notehead_dialog_active = False
            self._ignore_next_left_release = False
            return
        if self._ignore_next_left_release:
            self._ignore_next_left_release = False
            return
        # Keep last edit and clear the session handle
        if self._arpeggio_dragging and hasattr(self._editor, '_snapshot_if_changed'):
            self._editor._snapshot_if_changed(coalesce=True, label='arpeggio_resize')
        self._arpeggio_dragging = False
        self._arpeggio_target = None
        self._arpeggio_drag_anchor_time = 0.0
        self._velocity_dragging = False
        self._velocity_target = None
        self._velocity_targets = []
        self._velocity_display_value = None
        self._velocity_display_x_mm = None
        self._velocity_display_y_mm = None
        self.edit_note = None
        self._editing_existing = False
        self._duration_edit_armed = False
        self._last_audition_pitch = None
        self._move_pitch_time_mode = False
        
        # switch guides back on after note editing
        self._editor.guides_active = True

        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_note_edit_until_release or self._notehead_dialog_active:
            return
        if self._ignore_next_left_release:
            return
        # Click handled on press; avoid duplicate creation on release-click path
        return

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        score: SCORE = self._editor.current_score()
        if score is None:
            self._pending_notehead_dialog_note_id = None
            return
        found, _hit_rect, y_mm_abs = self._hit_note_and_rect(score, x, y)
        if found is None:
            self._pending_notehead_dialog_note_id = None
            return
        note_top_mm, note_bottom_mm = self._notehead_vertical_bounds_mm(score, found)
        if not (float(note_top_mm) <= float(y_mm_abs) <= float(note_bottom_mm)):
            self._pending_notehead_dialog_note_id = None
            return

        # Queue dialog handling for release so mouse state is fully settled.
        self._pending_notehead_dialog_note_id = int(getattr(found, '_id', -1) or -1)

    def on_left_double_unpress(self, x: float, y: float) -> None:
        super().on_left_double_unpress(x, y)
        note_id = self._pending_notehead_dialog_note_id
        self._pending_notehead_dialog_note_id = None
        if note_id is None:
            return

        score: SCORE = self._editor.current_score()
        if score is None:
            return

        found = None
        for n in getattr(score.events, 'note', []) or []:
            if int(getattr(n, '_id', -1) or -1) == int(note_id):
                found = n
                break
        if found is None:
            return

        # Check if this notehead is custom (not auto)
        current_notehead = str(getattr(found, 'notehead', 'auto') or 'auto').strip()
        is_custom = current_notehead != "auto"
        
        # If custom, reset to auto instead of opening dialog
        if is_custom:
            found.notehead = "auto"
            self._cancel_active_note_edit(redraw=False)
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
            if hasattr(self._editor, '_snapshot_if_changed'):
                self._editor._snapshot_if_changed(coalesce=True, label='notehead_reset')
            return

        # Otherwise, open the notehead dialog
        self._notehead_dialog_active = True
        self._cancel_active_note_edit(redraw=False)

        parent = QtWidgets.QApplication.activeWindow()
        layout = getattr(score, 'layout', None)
        default_black_above = self._black_note_above_stem(score, found)
        selected, accepted = NoteheadDialog.get_notehead(
            note=found,
            layout=layout,
            semitone_space_mm=float(getattr(self._editor, 'semitone_dist', 0.5) or 0.5),
            notation_color=self._editor.notation_color,
            paper_color=self._editor.paper_color,
            default_black_above=default_black_above,
            parent=parent,
        )
        self._notehead_dialog_active = False
        if not accepted:
            self._refresh_cursor_overlay_from_pointer()
            return

        found.notehead = selected
        self._cancel_active_note_edit(redraw=False)
        self._refresh_cursor_overlay_from_pointer()
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
        if hasattr(self._editor, '_snapshot_if_changed'):
            self._editor._snapshot_if_changed(coalesce=True, label='notehead_override')

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        if self._notehead_dialog_active or self._suppress_note_edit_until_release:
            return
        # Nothing to do; edit_note is established on press
        return

    def _cancel_active_note_edit(self, redraw: bool) -> None:
        if hasattr(self._editor, 'clear_single_note_timing_dirty'):
            self._editor.clear_single_note_timing_dirty()
        self._velocity_dragging = False
        self._velocity_target = None
        self._velocity_targets = []
        self._velocity_display_value = None
        self._velocity_display_x_mm = None
        self._velocity_display_y_mm = None
        self.edit_note = None
        self._editing_existing = False
        self._duration_edit_armed = False
        self._last_audition_pitch = None
        self._move_pitch_time_mode = False
        self._editor.guides_active = True
        if redraw:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()

    def _refresh_cursor_overlay_from_pointer(self) -> None:
        """Recompute cursor guides from current pointer position after modal dialogs."""
        try:
            widget = getattr(self._editor, 'widget', None)
            if widget is None:
                return
            pos = widget.mapFromGlobal(QtGui.QCursor.pos())
            x = float(pos.x())
            y = float(pos.y())
            self._editor.mouse_move(x, y, 0.0, 0.0)
            if hasattr(widget, 'request_overlay_refresh'):
                widget.request_overlay_refresh()
            else:
                widget.update()
        except Exception:
            return

    def _editor_left_button_down(self) -> bool:
        try:
            widget = getattr(self._editor, 'widget', None)
            if widget is not None:
                return bool(getattr(widget, '_left_down', False))
        except Exception:
            pass
        return bool(getattr(self._editor, '_left_pressed', False))

    def _finalize_notehead_dialog_transition(self) -> None:
        # Fully clear tool/editor/widget mouse states touched by double-click dialog path.
        self._notehead_dialog_active = False
        self._suppress_note_edit_until_release = False
        self._ignore_next_left_release = False
        self._cancel_active_note_edit(redraw=False)
        try:
            setattr(self._editor, '_ignore_next_left_release', False)
            setattr(self._editor, '_left_pressed', False)
            setattr(self._editor, '_dragging_left', False)
            setattr(self._editor, '_left_selection_mode', False)
            widget = getattr(self._editor, 'widget', None)
            if widget is not None:
                setattr(widget, '_left_down', False)
                try:
                    widget.releaseMouse()
                except Exception:
                    pass
        except Exception:
            pass

    def _black_note_above_stem(self, score: SCORE, note: Note) -> bool:
        layout = getattr(score, 'layout', None)
        rule = str(getattr(layout, 'black_note_rule', 'below_stem') or 'below_stem')
        if rule == 'above_stem':
            return True
        t0 = float(getattr(note, 'time', 0.0) or 0.0)
        p0 = int(getattr(note, 'pitch', 0) or 0)
        note_id = int(getattr(note, '_id', -1) or -1)
        op = Operator(7)
        if rule in ('above_stem_if_collision', 'only_above_stem_if_collision'):
            for other in getattr(score.events, 'note', []) or []:
                if int(getattr(other, '_id', -2) or -2) == note_id:
                    continue
                if not op.eq(float(getattr(other, 'time', 0.0) or 0.0), t0):
                    continue
                if abs(int(getattr(other, 'pitch', 0) or 0) - p0) == 1:
                    return True
            return False
        if rule == 'above_stem_if_chord_and_white_note':
            for other in getattr(score.events, 'note', []) or []:
                if int(getattr(other, '_id', -2) or -2) == note_id:
                    continue
                if not op.eq(float(getattr(other, 'time', 0.0) or 0.0), t0):
                    continue
                other_pitch = int(getattr(other, 'pitch', 0) or 0)
                if other_pitch not in BLACK_KEYS and other_pitch != p0:
                    return True
            return False
        if rule != 'above_stem_if_chord_and_white_note_same_hand':
            return False
        hand0 = str(getattr(note, 'hand', 'l') or 'l')
        for other in getattr(score.events, 'note', []) or []:
            if int(getattr(other, '_id', -2) or -2) == note_id:
                continue
            if not op.eq(float(getattr(other, 'time', 0.0) or 0.0), t0):
                continue
            if str(getattr(other, 'hand', 'l') or 'l') != hand0:
                continue
            other_pitch = int(getattr(other, 'pitch', 0) or 0)
            if other_pitch not in BLACK_KEYS and other_pitch != p0:
                return True
        return False

    def _notehead_vertical_bounds_mm(self, score: SCORE, note: Note) -> tuple[float, float]:
        note_start_mm = float(self._editor.time_to_mm(float(getattr(note, 'time', 0.0) or 0.0)))
        notehead_h_mm = float(getattr(self._editor, 'semitone_dist', 0.5) or 0.5) * 2.0
        spec = resolve_notehead_spec(note, default_black_above=self._black_note_above_stem(score, note))
        if bool(getattr(spec, 'is_up', False)):
            return (float(note_start_mm - notehead_h_mm), float(note_start_mm))
        return (float(note_start_mm), float(note_start_mm + notehead_h_mm))

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if not self._editor_left_button_down():
            self._cancel_active_note_edit(redraw=False)
            return
        if self._notehead_dialog_active or self._suppress_note_edit_until_release:
            return
        if self._arpeggio_dragging:
            self._update_arpeggio_duration_from_cursor(x, y)
            return
        if self._velocity_dragging:
            self._apply_velocity_from_cursor(x, y)
            return
        if self.edit_note is None:
            return
        
        # Update the in-progress note based on current mouse
        note = self.edit_note
        prev_time = float(getattr(note, 'time', 0.0) or 0.0)
        prev_duration = float(getattr(note, 'duration', 0.0) or 0.0)
        cur_t_raw = float(self._editor.widget_px_to_time(x, y))
        cur_t_snap = float(self._editor.snap_time(cur_t_raw))
        cur_pitch = int(self._editor.widget_px_to_pitch(x, y))

        # Update rules:
        # - New note: pitch-only before start; else duration adjust with min snap
        # - Existing note: do NOT shorten to snap while within one snap from start; allow pitch-only there.
        start_t = float(getattr(note, 'time', 0.0) or 0.0)
        units = float(max(1e-6, getattr(self._editor, 'snap_size_units', 8.0)))
        snapped_end = float(max(cur_t_snap, start_t + units))
        # Thresholded comparator to avoid floating-point jitter around band boundaries
        op = Operator(SHORTEST_DURATION)

        if not self._editing_existing:
            # Creating a new note: original behavior
            if op.le(cur_t_raw, start_t):
                prev_pitch = int(getattr(note, 'pitch', cur_pitch) or cur_pitch)
                note.pitch = cur_pitch
                if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
                    self._audition_pitch(cur_pitch)
            else:
                # Always snap the end to the grid based on the current snap size
                candidate = float(max(units, snapped_end - start_t))
                if self._can_apply_duration(note, candidate):
                    note.duration = candidate
        else:
            # Editing existing note:
            if self._move_pitch_time_mode:
                candidate_time = float(max(0.0, cur_t_snap))
                candidate_pitch = int(cur_pitch)
                if not self._can_apply_time_pitch_move(note, candidate_time, candidate_pitch):
                    return
                prev_pitch = int(getattr(note, 'pitch', cur_pitch) or cur_pitch)
                note.pitch = candidate_pitch
                self._apply_active_accidental_to_note(note)
                if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
                    self._audition_pitch(cur_pitch)
                note.time = candidate_time
                note.duration = float(max(0.0, self._orig_duration))
                if (not Operator(SHORTEST_DURATION).eq(prev_time, float(getattr(note, 'time', 0.0) or 0.0))) or (not Operator(SHORTEST_DURATION).eq(prev_duration, float(getattr(note, 'duration', 0.0) or 0.0))):
                    if hasattr(self._editor, 'mark_single_note_timing_dirty'):
                        self._editor.mark_single_note_timing_dirty(note, prev_time, prev_duration)
                    self._editor.update_score_length(note)
                return

            # Editing existing note:
            # - Mouse before start time: pitch-only
            # - Until we cross one snap unit past start, do pitch-only and do not alter duration
            # - Once we cross into the second snap band, arm duration editing. From then on:
            #   * If back inside first band: set duration to exactly one snap unit
            #   * Else: adjust duration snapped as usual
            if op.le(cur_t_raw, start_t):
                prev_pitch = int(getattr(note, 'pitch', cur_pitch) or cur_pitch)
                note.pitch = cur_pitch
                self._apply_active_accidental_to_note(note)
                if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
                    self._audition_pitch(cur_pitch)
            else:
                if not self._duration_edit_armed:
                    if op.ge(cur_t_raw, start_t + units):
                        self._duration_edit_armed = True
                if self._duration_edit_armed:
                    # Armed: allow duration edits, including 1 snap when back inside first band
                    candidate = float(max(units, snapped_end - start_t))
                    if self._can_apply_duration(note, candidate):
                        note.duration = candidate

        if (not op.eq(prev_time, float(getattr(note, 'time', 0.0) or 0.0))) or (not op.eq(prev_duration, float(getattr(note, 'duration', 0.0) or 0.0))):
            # Fast path: only the actively edited note changed.
            if hasattr(self._editor, 'mark_single_note_timing_dirty'):
                self._editor.mark_single_note_timing_dirty(note, prev_time, prev_duration)
            self._editor.update_score_length(note)

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._notehead_dialog_active or self._suppress_note_edit_until_release:
            return
        # Finalize edit session
        if self._arpeggio_dragging and hasattr(self._editor, '_snapshot_if_changed'):
            self._editor._snapshot_if_changed(coalesce=True, label='arpeggio_resize')
        self._arpeggio_dragging = False
        self._arpeggio_target = None
        self._arpeggio_drag_anchor_time = 0.0
        self._velocity_dragging = False
        self._velocity_target = None
        self._velocity_targets = []
        self._velocity_display_value = None
        self._velocity_display_x_mm = None
        self._velocity_display_y_mm = None
        self.edit_note = None
        self._editing_existing = False
        self._duration_edit_armed = False
        self._last_audition_pitch = None
        self._move_pitch_time_mode = False

    def on_right_press(self, x: float, y: float) -> None:
        super().on_right_press(x, y)

    def on_right_unpress(self, x: float, y: float) -> None:
        super().on_right_unpress(x, y)

        # Ensure the music/base_grid covers latest note end
        self._editor.update_score_length()

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        # Detect a note at click position; if found, delete and redraw
        score: SCORE = self._editor.current_score()

        # Use rectangle hit detection for delete
        target, _hit_rect, _y_mm = self._hit_note_and_rect(score, x, y)

        deleted_any = False
        if target is not None:
            notes_list = getattr(score.events, 'note', None)
            if isinstance(notes_list, list):
                if target in notes_list:
                    notes_list.remove(target)
                    deleted_any = True
                else:
                    tid = int(getattr(target, '_id', -1) or -1)
                    new_list = [m for m in notes_list if int(getattr(m, '_id', -2) or -2) != tid]
                    if len(new_list) != len(notes_list):
                        score.events.note = new_list
                        deleted_any = True
        if deleted_any:
            self._cleanup_arpeggios(score)
            # Keep base_grid in sync and trigger engrave via snapshot.
            self._editor.update_score_length()

    def _latest_measure_has_notes(self, score: SCORE) -> bool:
        """Return True if there is at least one note in the score's latest measure window.

        The latest measure window is computed from score.base_grid by walking all segments
        and measures to find the final measure's start/end times in ticks.
        """
        # Compute last measure start/end in ticks
        start_t, end_t = self._last_measure_window_ticks(score)
        if start_t is None or end_t is None:
            return False
        notes = list(getattr(score.events, 'note', []) or [])
        for n in notes:
            t = float(getattr(n, 'time', 0.0) or 0.0)
            if start_t <= t < end_t:
                return True
        return False

    def _last_measure_window_ticks(self, score: SCORE) -> tuple[Optional[float], Optional[float]]:
        """Compute the start and end times (ticks) of the latest measure in the score.

        Returns (start_t, end_t) or (None, None) if base_grid is missing.
        """
        base_grid = list(getattr(score, 'base_grid', []) or [])
        if not base_grid:
            return (None, None)
        cur_t = 0.0
        last_start = 0.0
        last_end = 0.0
        for bg in base_grid:
            num = float(getattr(bg, 'numerator', 4) or 4)
            den = float(getattr(bg, 'denominator', 4) or 4)
            m_count = int(getattr(bg, 'measure_amount', 1) or 1)
            measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
            for _ in range(max(0, m_count)):
                last_start = cur_t
                last_end = cur_t + measure_len
                cur_t = last_end
        return (last_start, last_end)

    def on_right_double_click(self, x: float, y: float) -> None:
        super().on_right_double_click(x, y)

    def on_right_drag_start(self, x: float, y: float) -> None:
        super().on_right_drag_start(x, y)

    def on_right_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_right_drag(x, y, dx, dy)

    def on_right_drag_end(self, x: float, y: float) -> None:
        super().on_right_drag_end(x, y)

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

    def on_key_press(self, key: int, modifiers) -> bool:
        if self._editor is None:
            return False
        if key == QtCore.Qt.Key.Key_A and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            self._cycle_accidental_toggle()
            if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
                w = getattr(self._editor, 'widget')
                if hasattr(w, 'request_overlay_refresh'):
                    w.request_overlay_refresh()
            return True
        return False

    def on_toolbar_button(self, name: str) -> None:
        if self._editor is None:
            return
        if name == 'hand_left':
            self._editor.hand_cursor = 'l'
        elif name == 'hand_right':
            self._editor.hand_cursor = 'r'
        elif name == 'selection_left':
            try:
                self._editor.set_selected_notes_hand('l')
            except Exception:
                pass
        elif name == 'selection_right':
            try:
                self._editor.set_selected_notes_hand('r')
            except Exception:
                pass
        elif name == 'velocity_toggle':
            self._velocity_mode = not self._velocity_mode
            self._velocity_dragging = False
            self._velocity_target = None
            self._velocity_targets = []
            self._velocity_display_value = None
            self._velocity_display_x_mm = None
            self._velocity_display_y_mm = None
            # Refresh overlay to show or hide velocity sliders immediately
            if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
                w = getattr(self._editor, 'widget')
                if hasattr(w, 'request_overlay_refresh'):
                    w.request_overlay_refresh()
        elif name == 'arpeggio_toggle':
            self._toggle_arpeggio()
        # Refresh overlay guides to reflect the change immediately
        if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
            w = getattr(self._editor, 'widget')
            if hasattr(w, 'request_overlay_refresh'):
                w.request_overlay_refresh()

    def accidental_toggle_value(self) -> int:
        return int(self._acc_toggle)

    def preview_accidental_for_pitch(self, pitch: int) -> int:
        try:
            p = int(pitch)
        except Exception:
            return 0
        if int(self._acc_toggle) == 0:
            return 0
        probe = Note(pitch=int(p), acc=int(self._acc_toggle))
        return int(self._acc_toggle) if Note.is_valid_accidental(probe) else 0

    def _cycle_accidental_toggle(self) -> None:
        cycle = list(self._acc_cycle)
        try:
            idx = cycle.index(int(self._acc_toggle))
        except ValueError:
            idx = 0
        self._acc_toggle = int(cycle[(idx + 1) % len(cycle)])

    def _apply_active_accidental_to_note(self, note: Note) -> None:
        """Overwrite note accidental from current accidental toggle during editing."""
        pitch = int(getattr(note, 'pitch', 0) or 0)
        acc = int(self.preview_accidental_for_pitch(pitch))
        setattr(note, 'acc', int(acc))
