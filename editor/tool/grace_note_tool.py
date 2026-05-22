from typing import Optional
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from ui.dialogs.notehead_dialog import NoteheadDialog
from PySide6 import QtWidgets
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator


class GraceNoteTool(BaseTool):
    TOOL_NAME = 'grace_note'

    def __init__(self):
        super().__init__()
        self._drag_grace = None
        self._drag_started = False
        self._suppress_click = False
        self._last_audition_pitch: int | None = None
        self._pending_grace_notehead_id: int | None = None
        self._time_op = Operator(float(SHORTEST_DURATION))

    _GRACE_NOTEHEAD_CHOICES: list[tuple[str, str]] = [
        ("auto", "Auto"),
        ("circle_white_down", "White"),
        ("circle_black_down", "Black"),
    ]


    def toolbar_spec(self) -> list[dict]:
        return []

    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

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

    def _add_grace_note(self, x: float, y: float) -> None:
        score = self._score()
        if score is None:
            return
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        pitch = int(self._editor.widget_px_to_pitch(x, y))
        self._audition_pitch(pitch)
        grace = score.new_grace_note(pitch=pitch, time=t_snap)
        self._editor.update_score_length(grace)
        self._editor._snapshot_if_changed(coalesce=True, label='grace_note_add')
        try:
            self._editor.force_redraw_from_model()
        except Exception:
            self._editor.draw_frame()

    def _cursor_pitch_and_time(self, x: float, y: float) -> tuple[int, float]:
        pitch_cursor = getattr(self._editor, 'pitch_cursor', None)
        if pitch_cursor is None:
            pitch_cursor = self._editor.widget_px_to_pitch(x, y)
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))
        return int(pitch_cursor), float(t_snap)

    def _grace_matches_cursor(self, grace, x: float, y: float) -> bool:
        cursor_pitch, cursor_time = self._cursor_pitch_and_time(x, y)
        grace_pitch = int(getattr(grace, 'pitch', cursor_pitch) or cursor_pitch)
        grace_time = float(getattr(grace, 'time', cursor_time) or cursor_time)
        return bool(grace_pitch == cursor_pitch and self._time_op.eq(grace_time, cursor_time))

    def _find_grace_at_cursor(self, score: SCORE, x: float, y: float):
        events = self._editor.current_events(score)
        if events is None:
            return None
        hit_gid = None
        if hasattr(self._editor, 'hit_test_grace_note_id'):
            hit_gid = self._editor.hit_test_grace_note_id(x, y)
        if hit_gid is not None:
            for g in (getattr(events, 'grace_note', []) or []):
                if int(getattr(g, '_id', -1) or -1) == int(hit_gid):
                    return g
        return None

    def _delete_grace_note(self, x: float, y: float) -> None:
        score = self._score()
        if score is None:
            return
        target = self._find_grace_at_cursor(score, x, y)
        if target is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return
        lst = getattr(events, 'grace_note', None)
        if isinstance(lst, list):
            try:
                lst.remove(target)
            except ValueError:
                tid = int(getattr(target, '_id', -2) or -2)
                events.grace_note = [m for m in lst if int(getattr(m, '_id', -2) or -2) != tid]
        self._editor.update_score_length()

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        # Detect an existing grace note under the cursor for drag edits
        self._drag_grace = None
        self._drag_started = False
        self._suppress_click = False
        # Hide guides/overlay cursor during grace edit for clarity
        if self._editor is not None:
            self._editor.guides_active = False
        score = self._score()
        if score is None:
            return
        self._drag_grace = self._find_grace_at_cursor(score, x, y)
        if self._drag_grace is not None:
            try:
                gp = int(getattr(self._drag_grace, 'pitch', self._editor.widget_px_to_pitch(x, y)) or self._editor.widget_px_to_pitch(x, y))
            except Exception:
                gp = int(self._editor.widget_px_to_pitch(x, y))
            self._audition_pitch(gp)
            self._suppress_click = True

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        if self._drag_grace is not None:
            self._drag_started = True

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._drag_grace is None:
            return
        # Update pitch and time live while dragging
        cur_t = float(self._editor.snap_time(self._editor.widget_px_to_time(x, y)))
        cur_pitch = int(self._editor.widget_px_to_pitch(x, y))
        prev_pitch = int(getattr(self._drag_grace, 'pitch', cur_pitch) or cur_pitch)
        self._drag_grace.time = cur_t
        self._drag_grace.pitch = cur_pitch
        if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
            self._audition_pitch(cur_pitch)

        self._editor.draw_frame()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._drag_grace is not None:
            self._editor.update_score_length(self._drag_grace)
            self._editor._snapshot_if_changed(coalesce=True, label='grace_note_move')
            try:
                self._editor.force_redraw_from_model()
            except Exception:
                self._editor.draw_frame()
        self._drag_grace = None
        self._drag_started = False
        self._last_audition_pitch = None
        # If we dragged an existing note, suppress creation on click path
        self._suppress_click = True
        # Restore guides after editing session ends
        if self._editor is not None:
            self._editor.guides_active = True

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_click:
            return
        self._add_grace_note(x, y)
        # Re-enable guides after add to keep overlay consistent
        if self._editor is not None:
            self._editor.guides_active = True

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        self._delete_grace_note(x, y)

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        self._pending_grace_notehead_id = None
        score = self._score()
        if score is None:
            return
        target = self._find_grace_at_cursor(score, x, y)
        if target is not None:
            self._pending_grace_notehead_id = int(getattr(target, '_id', -1) or -1)

    def on_left_double_unpress(self, x: float, y: float) -> None:
        super().on_left_double_unpress(x, y)
        gid = self._pending_grace_notehead_id
        self._pending_grace_notehead_id = None
        if gid is None:
            return
        score = self._score()
        if score is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return
        target = None
        for g in getattr(events, 'grace_note', []) or []:
            if int(getattr(g, '_id', -1) or -1) == int(gid):
                target = g
                break
        if target is None:
            return

        current_notehead = str(getattr(target, 'notehead', 'auto') or 'auto').strip()
        if current_notehead != 'auto':
            target.notehead = 'auto'
            self._editor._snapshot_if_changed(coalesce=True, label='grace_notehead_reset')
            try:
                self._editor.force_redraw_from_model()
            except Exception:
                self._editor.draw_frame()
            return

        layout = getattr(score, 'layout', None)
        grace_scale = float(getattr(layout, 'grace_note_scale', 0.8) or 0.8)
        style_scale = float(getattr(layout, 'scale', 1.0) or 1.0)
        grace_outline = float(getattr(layout, 'grace_note_outline_width_mm', getattr(layout, 'grace_note_outline_width', 0.3)) or 0.3) * style_scale
        parent = QtWidgets.QApplication.activeWindow()
        selected, accepted = NoteheadDialog.get_notehead(
            note=target,
            layout=layout,
            semitone_space_mm=float(getattr(self._editor, 'semitone_dist', 0.5) or 0.5) * max(0.05, grace_scale),
            notation_color=self._editor.notation_color,
            paper_color=self._editor.paper_color,
            default_black_above=False,
            choices=self._GRACE_NOTEHEAD_CHOICES,
            show_stem=False,
            outline_width_mm_override=grace_outline,
            parent=parent,
        )
        if not accepted:
            return
        target.notehead = str(selected or 'auto')
        self._editor._snapshot_if_changed(coalesce=True, label='grace_notehead_override')
        try:
            self._editor.force_redraw_from_model()
        except Exception:
            self._editor.draw_frame()
