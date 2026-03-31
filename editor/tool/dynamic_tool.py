from __future__ import annotations
from typing import Optional, Tuple
from PySide6 import QtWidgets, QtGui

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from ui.dialogs.dynamic_menu import DynamicSymbolMenu
from utils.CONSTANT import QUARTER_NOTE_UNIT


class DynamicTool(BaseTool):
    """
    Tool for placing and editing crescendo/decrescendo hairpins and standalone
    dynamic symbols.

    Modes:
    - crescendo: place/edit crescendo hairpins
    - decrescendo: place/edit decrescendo hairpins
    - dynamic: insert/edit/move/delete standalone dynamic symbols
    """

    TOOL_NAME = 'dynamic'

    _MODE_CRESCENDO = 'crescendo'
    _MODE_DECRESCENDO = 'decrescendo'
    _MODE_DYNAMIC_SYMBOL = 'dynamic'

    def __init__(self) -> None:
        super().__init__()
        self._mode: str = self._MODE_CRESCENDO
        self._active_hairpin = None          # current event object being dragged
        self._active_type: Optional[str] = None   # 'crescendo' or 'decrescendo'
        self._active_handle: Optional[str] = None # 'start' or 'end'
        self._active_symbol = None
        self._created_on_press: bool = False
        self._dragged_symbol: bool = False
        self._suppress_next_left_interaction: bool = False
        self._dialog_open: bool = False  # Gate to prevent multiple dialogs opening

    def _clear_active_interaction(self) -> None:
        self._active_hairpin = None
        self._active_handle = None
        self._active_type = None
        self._active_symbol = None

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': self._MODE_CRESCENDO,
                'icon': 'crescendo',
                'text': '<',
                'tooltip': 'Click to insert a crescendo hairpin. Drag the red handle to adjust position and length. Right-click to delete.',
                'active': self._mode == self._MODE_CRESCENDO,
            },
            {
                'name': self._MODE_DECRESCENDO,
                'icon': 'decrescendo',
                'text': '>',
                'tooltip': 'Click to insert a decrescendo hairpin. Drag the red handle to adjust position and length. Right-click to delete.',
                'active': self._mode == self._MODE_DECRESCENDO,
            },
            {
                'name': self._MODE_DYNAMIC_SYMBOL,
                'icon': 'dynamics',
                'tooltip': 'Click to insert or edit an existing dynamic symbol. Drag to adjust position. Right-click to delete.',
                'active': self._mode == self._MODE_DYNAMIC_SYMBOL,
            },
        ]

    def on_toolbar_button(self, name: str) -> None:
        if name in (self._MODE_CRESCENDO, self._MODE_DECRESCENDO, self._MODE_DYNAMIC_SYMBOL):
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

    def _select_dynamic_symbol(self, current: str = '') -> tuple[str, bool]:
        # Prevent multiple dialogs from opening simultaneously
        if self._dialog_open:
            return '', False
        
        parent_w = QtWidgets.QApplication.activeWindow()
        self._dialog_open = True
        self._suppress_next_left_interaction = True
        try:
            selected_glyph, ok = DynamicSymbolMenu.get_dynamic_glyph(
                parent=parent_w, 
                current_value=current,
                pos=QtGui.QCursor.pos()
            )
        finally:
            self._dialog_open = False
            if parent_w is not None:
                parent_w.activateWindow()
                parent_w.raise_()
        return str(selected_glyph or ''), bool(ok)

    def _create_dynamic_symbol(self, x: float, y: float, symbol: str):
        score = self._score()
        if score is None:
            return None
        t_snap = self._snap_time(y)
        x_mm, _ = self._cursor_mm(x, y)
        rpitch = self._x_mm_to_rpitch(x_mm)
        return score.new_dynamic_symbol(time=float(t_snap), x_rpitch=int(rpitch), symbol=str(symbol or ''))

    def _delete_dynamic_symbol(self, ev) -> None:
        score = self._score()
        if score is None:
            return
        ev_id = int(getattr(ev, '_id', -1) or -1)
        score.events.dynamic_symbol = [
            d for d in (score.events.dynamic_symbol or [])
            if int(getattr(d, '_id', -2) or -2) != ev_id
        ]

    def _all_hairpins(self) -> list[object]:
        score = self._score()
        if score is None:
            return []
        return list(getattr(score.events, 'crescendo', []) or []) + list(getattr(score.events, 'decrescendo', []) or [])

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

    def _hairpin_endpoint_time(self, hp, handle: str) -> float:
        start = float(getattr(hp, 'time', 0.0) or 0.0)
        if handle == 'start':
            return start
        return start + float(getattr(hp, 'duration', self._snap_units()) or self._snap_units())

    def _find_connected_symbol_for_handle(self, hp, handle: str):
        score = self._score()
        if score is None:
            return None
        endpoint_time = self._hairpin_endpoint_time(hp, handle)
        endpoint_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)
        for symbol in (getattr(score.events, 'dynamic_symbol', []) or []):
            symbol_time = float(getattr(symbol, 'time', 0.0) or 0.0)
            symbol_rpitch = int(getattr(symbol, 'x_rpitch', 0) or 0)
            if abs(symbol_time - endpoint_time) < 0.1 and symbol_rpitch == endpoint_rpitch:
                return symbol
        return None

    def _move_connected_symbol_with_handle(self, hp, handle: str, symbol) -> None:
        if symbol is None:
            return
        symbol.time = float(max(0.0, self._hairpin_endpoint_time(hp, handle)))
        symbol.x_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)

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
        if self._suppress_next_left_interaction or self._dialog_open:
            self._clear_active_interaction()
            return
        if self._editor is None:
            return
        # Ensure draw cache is up to date for hit testing
        self._editor.draw_frame()
        x_mm, y_mm = self._cursor_mm(x, y)
        self._dragged_symbol = False

        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            sym, _sym_type, _ = self._editor.hit_test_dynamic_symbol_mm(x_mm, y_mm)
            self._active_symbol = sym
            self._active_hairpin = None
            self._active_type = None
            self._active_handle = None
            self._created_on_press = False
            return

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
        if self._suppress_next_left_interaction or self._dialog_open:
            return
        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            if self._active_symbol is None:
                return
            t_snap = self._snap_time(y)
            try:
                x_mm, _ = self._cursor_mm(x, y)
                rpitch = self._x_mm_to_rpitch(x_mm)
            except Exception:
                rpitch = int(getattr(self._active_symbol, 'x_rpitch', 0) or 0)
            self._active_symbol.time = float(max(0.0, t_snap))
            self._active_symbol.x_rpitch = int(rpitch)
            self._dragged_symbol = True
            self._redraw()
            return
        if self._active_hairpin is None or self._active_handle is None:
            return
        connected_symbol = self._find_connected_symbol_for_handle(self._active_hairpin, self._active_handle)
        t_snap = self._snap_time(y)
        try:
            x_mm, _ = self._cursor_mm(x, y)
            rpitch = self._x_mm_to_rpitch(x_mm)
        except Exception:
            rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
        self._apply_handle_drag(self._active_hairpin, self._active_handle, t_snap, rpitch)
        self._move_connected_symbol_with_handle(self._active_hairpin, self._active_handle, connected_symbol)
        self._redraw()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        self._clear_active_interaction()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_next_left_interaction or self._dialog_open:
            return
        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            if self._editor is None:
                return
            x_mm, y_mm = self._cursor_mm(x, y)
            symbol_ev, _sym_type, _ = self._editor.hit_test_dynamic_symbol_mm(x_mm, y_mm)
            if symbol_ev is not None and not self._dragged_symbol:
                glyph, ok = self._select_dynamic_symbol(str(getattr(symbol_ev, 'symbol', '') or ''))
                if ok:
                    symbol_ev.symbol = str(glyph or '')
                    self._redraw()
                return

            if symbol_ev is None:
                glyph, ok = self._select_dynamic_symbol('')
                if not ok:
                    return
                created = self._create_dynamic_symbol(x, y, glyph)
                if created is not None:
                    self._redraw()
            return

        # Hairpin modes
        self._created_on_press = False
        self._clear_active_interaction()

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        # Dynamic symbol dialog is opened from left-click only.
        # Ignoring canvas double-click avoids duplicate dialogs.
        return

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

        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            sym, _sym_type, _ = self._editor.hit_test_dynamic_symbol_mm(x_mm, y_mm)
            if sym is None:
                return
            self._delete_dynamic_symbol(sym)
            self._redraw()
            return

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
        self._redraw()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)

