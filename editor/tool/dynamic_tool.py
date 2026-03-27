from __future__ import annotations
from typing import Optional, Tuple
from PySide6 import QtWidgets

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from ui.dialogs.dynamic_dialog import DynamicDialog
from utils.CONSTANT import QUARTER_NOTE_UNIT


class DynamicTool(BaseTool):
    """
    Tool for placing and editing crescendo (<) and decrescendo (>) hairpins.
    The active hairpin type is selected via toolbar buttons. Left click on empty
    space creates a new hairpin at the snapped time position with a snap-size
    initial duration. Drag handles to adjust the start time, end time, and
    horizontal position (x_rpitch). Right click on a handle removes the hairpin.
    """

    TOOL_NAME = 'dynamic'

    _MODE_CRESCENDO = 'crescendo'
    _MODE_DECRESCENDO = 'decrescendo'

    def __init__(self) -> None:
        super().__init__()
        self._mode: str = self._MODE_CRESCENDO
        self._active_hairpin = None          # current event object being dragged
        self._active_type: Optional[str] = None   # 'crescendo' or 'decrescendo'
        self._active_handle: Optional[str] = None # 'start' or 'end'
        self._created_on_press: bool = False
        self._suppress_next_left_interaction: bool = False

    def _clear_active_interaction(self) -> None:
        self._active_hairpin = None
        self._active_handle = None
        self._active_type = None

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': self._MODE_CRESCENDO,
                'icon': 'crescendo',
                'text': '<',
                'tooltip': 'Crescendo hairpin',
                'active': self._mode == self._MODE_CRESCENDO,
            },
            {
                'name': self._MODE_DECRESCENDO,
                'icon': 'decrescendo',
                'text': '>',
                'tooltip': 'Decrescendo hairpin',
                'active': self._mode == self._MODE_DECRESCENDO,
            },
        ]

    def on_toolbar_button(self, name: str) -> None:
        if name in (self._MODE_CRESCENDO, self._MODE_DECRESCENDO):
            self._mode = str(name)

    # ---- Helpers ----

    def _score(self) -> Optional[SCORE]:
        try:
            return self._editor.current_score()
        except Exception:
            return None

    def _cursor_mm(self, x_px: float, y_px: float) -> Tuple[float, float]:
        px_per_mm = float(getattr(self._editor, '_widget_px_per_mm', 1.0) or 1.0)
        view_offset = float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
        x_mm = float(x_px) / max(1e-6, px_per_mm)
        y_mm = float(y_px) / max(1e-6, px_per_mm) + view_offset
        return x_mm, y_mm

    def _view_width_mm(self) -> float:
        try:
            du = self.draw_util()
            w, _ = du.current_page_size_mm()
            return float(w or 0.0)
        except Exception:
            return 0.0

    def _x_mm_to_rpitch(self, x_mm: float) -> int:
        base_x = float(self._editor.pitch_to_x(40))
        dist = float(self._editor.semitone_dist or 0.0)
        vw = self._view_width_mm()
        x_clamped = max(0.0, min(float(x_mm), vw if vw > 0 else float(x_mm)))
        offset = round((x_clamped - base_x) / dist)
        return int(max(-68, min(73, offset)))

    def _snap_time(self, y_px: float) -> float:
        t_raw = float(self._editor.y_to_time(y_px))
        return float(self._editor.snap_time(t_raw))

    def _snap_units(self) -> float:
        return float(getattr(self._editor, 'snap_size_units', QUARTER_NOTE_UNIT) or QUARTER_NOTE_UNIT)

    def _redraw(self) -> None:
        if self._editor is None:
            return
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()

    def _set_hairpin_text_via_dialog(self, hp, handle: str) -> None:
        parent_w = QtWidgets.QApplication.activeWindow()
        field = 'start_text' if str(handle) == 'start' else 'end_text'
        current = str(getattr(hp, field, '') or '')
        self._clear_active_interaction()
        # Ignore the immediate post-dialog click/release sequence.
        self._suppress_next_left_interaction = True
        selected_glyph, ok = DynamicDialog.get_dynamic_glyph(parent=parent_w, current_value=current)
        if not ok:
            return
        setattr(hp, field, str(selected_glyph or ''))
        self._mirror_symbol_to_connected_side(hp, str(handle))
        if self._editor is not None and hasattr(self._editor, '_snapshot_if_changed'):
            self._editor._snapshot_if_changed(coalesce=False, label='hairpin_set_text')
        self._redraw()

    def _all_hairpins(self) -> list[object]:
        score = self._score()
        if score is None:
            return []
        return list(getattr(score.events, 'crescendo', []) or []) + list(getattr(score.events, 'decrescendo', []) or [])

    def _mirror_symbol_to_connected_side(self, hp, handle: str) -> None:
        all_hairpins = self._all_hairpins()
        if not all_hairpins:
            return
        x_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)
        start_t = float(getattr(hp, 'time', 0.0) or 0.0)
        dur = float(getattr(hp, 'duration', 0.0) or 0.0)
        end_t = start_t + dur
        eps = 1e-6

        if handle == 'start':
            winner = str(getattr(hp, 'start_text', '') or '')
            for peer in all_hairpins:
                if peer is hp:
                    continue
                if int(getattr(peer, 'x_rpitch', 0) or 0) != x_rpitch:
                    continue
                peer_end_t = float(getattr(peer, 'time', 0.0) or 0.0) + float(getattr(peer, 'duration', 0.0) or 0.0)
                if abs(peer_end_t - start_t) <= eps:
                    setattr(peer, 'end_text', winner)
        else:
            winner = str(getattr(hp, 'end_text', '') or '')
            for peer in all_hairpins:
                if peer is hp:
                    continue
                if int(getattr(peer, 'x_rpitch', 0) or 0) != x_rpitch:
                    continue
                peer_start_t = float(getattr(peer, 'time', 0.0) or 0.0)
                if abs(peer_start_t - end_t) <= eps:
                    setattr(peer, 'start_text', winner)

    def _sync_symbols_on_new_connections(self, hp) -> None:
        """When drag creates a connection, copy top-side symbol to bottom-side symbol."""
        all_hairpins = self._all_hairpins()
        if not all_hairpins:
            return

        x_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)
        start_t = float(getattr(hp, 'time', 0.0) or 0.0)
        dur = float(getattr(hp, 'duration', 0.0) or 0.0)
        end_t = start_t + dur
        eps = 1e-6

        # Join at current start: peer end (top) -> hp start (bottom)
        for peer in all_hairpins:
            if peer is hp:
                continue
            if int(getattr(peer, 'x_rpitch', 0) or 0) != x_rpitch:
                continue
            peer_end_t = float(getattr(peer, 'time', 0.0) or 0.0) + float(getattr(peer, 'duration', 0.0) or 0.0)
            if abs(peer_end_t - start_t) <= eps:
                top_symbol = str(getattr(peer, 'end_text', '') or '')
                setattr(hp, 'start_text', top_symbol)

        # Join at current end: hp end (top) -> peer start (bottom)
        top_symbol = str(getattr(hp, 'end_text', '') or '')
        for peer in all_hairpins:
            if peer is hp:
                continue
            if int(getattr(peer, 'x_rpitch', 0) or 0) != x_rpitch:
                continue
            peer_start_t = float(getattr(peer, 'time', 0.0) or 0.0)
            if abs(peer_start_t - end_t) <= eps:
                setattr(peer, 'start_text', top_symbol)

    def _create_hairpin(self, x: float, y: float):
        score = self._score()
        if score is None:
            return None
        t_snap = self._snap_time(y)
        x_mm, _ = self._cursor_mm(x, y)
        rpitch = self._x_mm_to_rpitch(x_mm)
        dur = self._snap_units()
        if self._mode == self._MODE_CRESCENDO:
            return score.new_crescendo(time=float(t_snap), duration=float(dur), x_rpitch=int(rpitch))
        else:
            return score.new_decrescendo(time=float(t_snap), duration=float(dur), x_rpitch=int(rpitch))

    def _apply_handle_drag(self, hp, handle: str, t_snap: float, rpitch: int) -> None:
        snap_units = self._snap_units()
        if handle == 'start':
            # Move start: keep end time fixed, shrink/grow duration
            old_end = float(getattr(hp, 'time', 0.0) or 0.0) + float(getattr(hp, 'duration', snap_units) or snap_units)
            new_time = max(0.0, min(t_snap, old_end - snap_units))
            new_dur = old_end - new_time
            hp.time = float(new_time)
            hp.duration = float(max(snap_units, new_dur))
        else:  # 'end'
            # Move end: change duration, keep start fixed
            start = float(getattr(hp, 'time', 0.0) or 0.0)
            new_end = max(start + snap_units, t_snap)
            hp.duration = float(new_end - start)
        hp.x_rpitch = int(rpitch)

    # ---- Events ----

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        if self._suppress_next_left_interaction:
            self._clear_active_interaction()
            return
        if self._editor is None:
            return
        # Ensure draw cache is up to date for hit testing
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        hp, hp_type, handle = self._editor.hit_test_hairpin_mm(x_mm, y_mm)
        self._active_hairpin = hp
        self._active_type = hp_type
        self._active_handle = handle
        self._created_on_press = False
        if hp is None:
            # No handle hit — create a new hairpin
            new_hp = self._create_hairpin(x, y)
            if new_hp is not None:
                self._active_hairpin = new_hp
                self._active_type = self._mode
                self._active_handle = 'end'
                self._created_on_press = True
            self._redraw()

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if self._suppress_next_left_interaction:
            return
        if self._active_hairpin is None or self._active_handle is None:
            return
        t_snap = self._snap_time(y)
        try:
            x_mm, _ = self._cursor_mm(x, y)
            rpitch = self._x_mm_to_rpitch(x_mm)
        except Exception:
            rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
        self._apply_handle_drag(self._active_hairpin, self._active_handle, t_snap, rpitch)
        self._redraw()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._active_hairpin is not None:
            self._sync_symbols_on_new_connections(self._active_hairpin)
        self._clear_active_interaction()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_next_left_interaction:
            return
        # Snapshot is handled by the editor after this call; just clear state.
        self._created_on_press = False
        self._clear_active_interaction()

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        if self._suppress_next_left_interaction:
            return
        if self._editor is None:
            return
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        hp, _hp_type, handle = self._editor.hit_test_hairpin_mm(x_mm, y_mm)
        if hp is None or handle not in ('start', 'end'):
            return
        self._set_hairpin_text_via_dialog(hp, str(handle))

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        self._clear_active_interaction()
        if self._suppress_next_left_interaction:
            self._suppress_next_left_interaction = False

    def on_right_click(self, x: float, y: float) -> None:
        super().on_right_click(x, y)
        if self._editor is None:
            return
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        hp, hp_type, _handle = self._editor.hit_test_hairpin_mm(x_mm, y_mm)
        if hp is None:
            return
        score = self._score()
        if score is None:
            return
        hp_id = int(getattr(hp, '_id', -1) or -1)
        if hp_type == 'crescendo':
            score.events.crescendo = [
                e for e in (score.events.crescendo or [])
                if int(getattr(e, '_id', -2) or -2) != hp_id
            ]
        else:
            score.events.decrescendo = [
                e for e in (score.events.decrescendo or [])
                if int(getattr(e, '_id', -2) or -2) != hp_id
            ]
        self._editor._snapshot_if_changed(coalesce=False, label='hairpin_delete')
        self._redraw()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

