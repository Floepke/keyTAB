import math
from typing import Optional
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from utils.operator import Operator
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT
from file_model.events.note import Note
from utils.CONSTANT import QUARTER_NOTE_UNIT


class NoteTool(BaseTool):
    TOOL_NAME = 'note'
    def __init__(self):
        super().__init__()
        # Currently edited/created note during a press/drag session
        self.edit_note = None
        self._hand: str = '<'
        self.expanded_score_flag: bool = False
        # Drag-session context
        self._editing_existing: bool = False
        self._orig_duration: float = 0.0
        self._press_start_time: float = 0.0
        self._duration_edit_armed: bool = False
        self._last_audition_pitch: int | None = None

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
        try:
            if hasattr(self._editor, 'player') and self._editor.player is not None:
                self._editor.player.audition_note(pitch=int(pitch))
                self._last_audition_pitch = int(pitch)
        except Exception:
            pass

    def toolbar_spec(self) -> list[dict]:
        # Two explicit hand selectors for quick switching
        return [
            {'name': 'hand_left', 'icon': 'note_left', 'tooltip': 'Switch To Left Hand'},
            {'name': 'hand_right', 'icon': 'note_right', 'tooltip': 'Switch To Right Hand'},
        ]

    def on_left_press(self, x: float, y: float) -> None:
        '''Detect existing note under cursor or create a new one, then enter edit mode'''
        super().on_left_press(x, y)
        
        score: SCORE = self._editor.current_score()

        # Compute raw (non-snapped) time for detection and snapped for creation
        t_press_raw = float(self._editor.y_to_time(y))
        t_press_snap = float(self._editor.snap_time(t_press_raw))
        pitch_press = int(self._editor.x_to_pitch(x))
        self._hand = str(getattr(self._editor, 'hand_cursor', '<') or '<')

        # Prefer rectangle-based hit detection for precise clickable area
        found = None
        hit_id = None
        hit_test = getattr(self._editor, 'hit_test_note_id', None)
        if callable(hit_test):
            hit_id = hit_test(x, y)
        if hit_id is not None:
            for n in getattr(score.events, 'note', []) or []:
                if int(getattr(n, '_id', -1) or -1) == int(hit_id):
                    found = n
                    break

        if found:
            # Edit existing note
            self.edit_note = found
            self._editing_existing = True
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
        else:
            # Create a new note at the snapped press time with minimum duration = snap size
            units = float(max(1e-6, getattr(self._editor, 'snap_size_units', 8.0)))
            self.edit_note = score.new_note(pitch=pitch_press, time=t_press_snap, duration=units, hand=self._hand)
            self._editing_existing = False
            self._orig_duration = float(units)
            self._press_start_time = float(t_press_snap)
            self._duration_edit_armed = False
            self._last_audition_pitch = None

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
        # Keep last edit and clear the session handle
        self.edit_note = None
        self._editing_existing = False
        self._duration_edit_armed = False
        self._last_audition_pitch = None
        
        # switch guides back on after note editing
        self._editor.guides_active = True

        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        # Click handled on press; avoid duplicate creation on release-click path
        return

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        # Nothing to do; edit_note is established on press
        return

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        # Update the in-progress note based on current mouse
        if self.edit_note is None:
            return
        
        # Get note being edited and current raw/snap time and pitch
        note = self.edit_note
        cur_t_raw = float(self._editor.y_to_time(y))
        cur_t_snap = float(self._editor.snap_time(cur_t_raw))
        cur_pitch = int(self._editor.x_to_pitch(x))

        # Update rules:
        # - New note: pitch-only before start; else duration adjust with min snap
        # - Existing note: do NOT shorten to snap while within one snap from start; allow pitch-only there.
        start_t = float(getattr(note, 'time', 0.0) or 0.0)
        units = float(max(1e-6, getattr(self._editor, 'snap_size_units', 8.0)))
        # Thresholded comparator to avoid floating-point jitter around band boundaries
        op = Operator(7)

        if not self._editing_existing:
            # Creating a new note: original behavior
            if op.le(cur_t_raw, start_t):
                note.pitch = cur_pitch
            else:
                if op.lt(cur_t_raw, start_t + units):
                    note.duration = units
                else:
                    # Compute bands beyond the first using raw time to reduce snapping jitter
                    ratio = max(0.0, (float(cur_t_raw) - float(start_t)) / max(1e-6, units))
                    bands_beyond_first = int(math.floor(ratio + 1e-9))
                    note.duration = float(bands_beyond_first + 1) * float(units)
        else:
            # Editing existing note:
            # - Before start: pitch-only
            # - Until we cross one snap unit past start, do pitch-only and do not alter duration
            # - Once we cross into the second snap band, arm duration editing. From then on:
            #   * If back inside first band: set duration to exactly one snap unit
            #   * Else: adjust duration snapped as usual
            if op.le(cur_t_raw, start_t):
                prev_pitch = int(getattr(note, 'pitch', cur_pitch) or cur_pitch)
                note.pitch = cur_pitch
                if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
                    self._audition_pitch(cur_pitch)
            else:
                if not self._duration_edit_armed:
                    if op.ge(cur_t_raw, start_t + units):
                        self._duration_edit_armed = True
                else:
                    # Armed: allow duration edits, including 1 snap when back inside first band
                    if op.lt(cur_t_raw, start_t + units):
                        note.duration = units
                    else:
                        ratio = max(0.0, (float(cur_t_raw) - float(start_t)) / max(1e-6, units))
                        bands_beyond_first = int(math.floor(ratio + 1e-9))
                        note.duration = float(bands_beyond_first + 1) * float(units)

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        # Finalize edit session
        self.edit_note = None
        self._editing_existing = False
        self._duration_edit_armed = False
        self._last_audition_pitch = None
        
        # Ensure the music/base_grid covers latest note end
        self._editor.update_score_length()

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
        target = None
        hit_id = None
        hit_test = getattr(self._editor, 'hit_test_note_id', None)
        if callable(hit_test):
            hit_id = hit_test(x, y)
        if hit_id is not None:
            for n in getattr(score.events, 'note', []) or []:
                if int(getattr(n, '_id', -1) or -1) == int(hit_id):
                    target = n
                    break

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

    def on_toolbar_button(self, name: str) -> None:
        if self._editor is None:
            return
        if name == 'hand_left':
            self._editor.hand_cursor = '<'
        elif name == 'hand_right':
            self._editor.hand_cursor = '>'
        # Refresh overlay guides to reflect the change immediately
        if hasattr(self._editor, 'widget') and getattr(self._editor, 'widget', None) is not None:
            w = getattr(self._editor, 'widget')
            if hasattr(w, 'request_overlay_refresh'):
                w.request_overlay_refresh()
