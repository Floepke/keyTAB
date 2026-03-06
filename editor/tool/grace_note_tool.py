from typing import Optional
from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE


class GraceNoteTool(BaseTool):
    TOOL_NAME = 'grace_note'

    def __init__(self):
        super().__init__()
        self._drag_grace = None
        self._drag_started = False
        self._suppress_click = False
        self._last_audition_pitch: int | None = None


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
        try:
            if hasattr(self._editor, 'player') and self._editor.player is not None:
                self._editor.player.audition_note(pitch=int(pitch))
                self._last_audition_pitch = int(pitch)
        except Exception:
            pass

    def _add_grace_note(self, x: float, y: float) -> None:
        score = self._score()
        if score is None:
            return
        t_raw = float(self._editor.y_to_time(y))
        t_snap = float(self._editor.snap_time(t_raw))
        pitch = int(self._editor.x_to_pitch(x))
        self._audition_pitch(pitch)
        score.new_grace_note(pitch=pitch, time=t_snap)
        try:
            self._editor.update_score_length()
        except Exception:
            pass
        try:
            self._editor._snapshot_if_changed(coalesce=True, label='grace_note_add')
        except Exception:
            pass
        try:
            self._editor.force_redraw_from_model()
        except Exception:
            self._editor.draw_frame()

    def _delete_grace_note(self, x: float, y: float) -> None:
        score = self._score()
        if score is None:
            return
        target = None
        hit_id = None
        hit_test = getattr(self._editor, 'hit_test_note_id', None)
        if callable(hit_test):
            hit_id = hit_test(x, y)
        if hit_id is not None:
            for g in getattr(score.events, 'grace_note', []) or []:
                if int(getattr(g, '_id', -1) or -1) == int(hit_id):
                    target = g
                    break
        if target is None:
            return
        lst = getattr(score.events, 'grace_note', None)
        if isinstance(lst, list):
            try:
                lst.remove(target)
            except ValueError:
                tid = int(getattr(target, '_id', -2) or -2)
                score.events.grace_note = [m for m in lst if int(getattr(m, '_id', -2) or -2) != tid]
        try:
            self._editor.update_score_length()
        except Exception:
            pass

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        # Detect an existing grace note under the cursor for drag edits
        self._drag_grace = None
        self._drag_started = False
        self._suppress_click = False
        # Hide guides/overlay cursor during grace edit for clarity
        try:
            if self._editor is not None:
                self._editor.guides_active = False
        except Exception:
            pass
        score = self._score()
        if score is None:
            return
        hit_id = None
        hit_test = getattr(self._editor, 'hit_test_note_id', None)
        if callable(hit_test):
            hit_id = hit_test(x, y)
        if hit_id is not None:
            for g in getattr(score.events, 'grace_note', []) or []:
                if int(getattr(g, '_id', -1) or -1) == int(hit_id):
                    self._drag_grace = g
                    try:
                        gp = int(getattr(g, 'pitch', self._editor.x_to_pitch(x)) or self._editor.x_to_pitch(x))
                    except Exception:
                        gp = int(self._editor.x_to_pitch(x))
                    self._audition_pitch(gp)
                    self._suppress_click = True
                    break

    def on_left_drag_start(self, x: float, y: float) -> None:
        super().on_left_drag_start(x, y)
        if self._drag_grace is not None:
            self._drag_started = True

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._drag_grace is None:
            return
        # Update pitch and time live while dragging
        cur_t = float(self._editor.snap_time(self._editor.y_to_time(y)))
        cur_pitch = int(self._editor.x_to_pitch(x))
        try:
            prev_pitch = int(getattr(self._drag_grace, 'pitch', cur_pitch) or cur_pitch)
            self._drag_grace.time = cur_t
            self._drag_grace.pitch = cur_pitch
            if cur_pitch != prev_pitch and cur_pitch != self._last_audition_pitch:
                self._audition_pitch(cur_pitch)
        except Exception:
            pass
        try:
            self._editor.draw_frame()
        except Exception:
            pass

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._drag_grace is not None:
            try:
                self._editor.update_score_length()
            except Exception:
                pass
            try:
                self._editor._snapshot_if_changed(coalesce=True, label='grace_note_move')
            except Exception:
                pass
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
        try:
            if self._editor is not None:
                self._editor.guides_active = True
        except Exception:
            pass

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_click:
            return
        self._add_grace_note(x, y)
        # Re-enable guides after add to keep overlay consistent
        try:
            if self._editor is not None:
                self._editor.guides_active = True
        except Exception:
            pass

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        self._delete_grace_note(x, y)
