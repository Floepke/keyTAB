from __future__ import annotations
from typing import Optional, Tuple
from PySide6 import QtCore, QtWidgets, QtGui

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
        self._dragged_hairpin: bool = False
        self._suppress_next_left_interaction: bool = False
        self._dialog_open: bool = False  # Gate to prevent multiple dialogs opening

    def _clear_active_interaction(self) -> None:
        self._active_hairpin = None
        self._active_handle = None
        self._active_type = None
        self._active_symbol = None
        self._dragged_hairpin = False

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': self._MODE_CRESCENDO,
                'icon': 'crescendo',
                'text': '<',
                'tooltip': QtCore.QCoreApplication.translate('DynamicTool', 'Click to insert a crescendo hairpin. Drag the red handle to adjust position and length. Right-click to delete.'),
                'active': self._mode == self._MODE_CRESCENDO,
            },
            {
                'name': self._MODE_DECRESCENDO,
                'icon': 'decrescendo',
                'text': '>',
                'tooltip': QtCore.QCoreApplication.translate('DynamicTool', 'Click to insert a decrescendo hairpin. Drag the red handle to adjust position and length. Right-click to delete.'),
                'active': self._mode == self._MODE_DECRESCENDO,
            },
            {
                'name': self._MODE_DYNAMIC_SYMBOL,
                'icon': 'dynamics',
                'tooltip': QtCore.QCoreApplication.translate('DynamicTool', 'Click to insert or edit an existing dynamic symbol. Drag to adjust position. Right-click to delete.'),
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
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

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

    def _snap_time(self, x_px: float, y_px: float) -> float:
        t_raw = float(self._editor.widget_px_to_time(x_px, y_px))
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
        t_snap = self._snap_time(x, y)
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
        t_snap = self._snap_time(x, y)
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

    def _symbols_at_point(self, point_time: float, point_rpitch: int) -> list[object]:
        score = self._score()
        if score is None:
            return []
        out: list[object] = []
        for symbol in (getattr(score.events, 'dynamic_symbol', []) or []):
            symbol_time = float(getattr(symbol, 'time', 0.0) or 0.0)
            symbol_rpitch = int(getattr(symbol, 'x_rpitch', 0) or 0)
            if abs(symbol_time - float(point_time)) < 0.1 and symbol_rpitch == int(point_rpitch):
                out.append(symbol)
        return out

    def _hairpins_at_point(self, point_time: float, point_rpitch: int) -> list[tuple[object, str]]:
        out: list[tuple[object, str]] = []
        for hp in self._all_hairpins():
            endpoint_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)
            for handle in ('start', 'end'):
                endpoint_time = self._hairpin_endpoint_time(hp, handle)
                if abs(float(endpoint_time) - float(point_time)) < 0.1 and endpoint_rpitch == int(point_rpitch):
                    out.append((hp, handle))
        return out

    def _other_handle(self, handle: str) -> str:
        return 'end' if str(handle) == 'start' else 'start'

    def _handle_point(self, hp, handle: str) -> tuple[float, int]:
        return (
            float(self._hairpin_endpoint_time(hp, handle)),
            int(getattr(hp, 'x_rpitch', 0) or 0),
        )

    def _translate_symbol(self, symbol, delta_time: float, delta_rpitch: int) -> None:
        symbol.time = float(max(0.0, float(getattr(symbol, 'time', 0.0) or 0.0) + float(delta_time)))
        symbol.x_rpitch = int(getattr(symbol, 'x_rpitch', 0) or 0) + int(delta_rpitch)

    def _translate_hairpin(self, hp, delta_time: float, delta_rpitch: int) -> None:
        hp.time = float(max(0.0, float(getattr(hp, 'time', 0.0) or 0.0) + float(delta_time)))
        hp.x_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0) + int(delta_rpitch)

    def _set_hairpin_handle(self, hp, handle: str, t_snap: float, rpitch: int) -> None:
        self._apply_handle_drag(hp, handle, float(t_snap), int(rpitch))

    def _propagate_chain_pitch_only(
        self,
        point_time: float,
        point_rpitch: int,
        delta_rpitch: int,
        *,
        visited_hairpins: set[int] | None = None,
        visited_symbols: set[int] | None = None,
    ) -> None:
        visited_hairpins = visited_hairpins if visited_hairpins is not None else set()
        visited_symbols = visited_symbols if visited_symbols is not None else set()

        for symbol in self._symbols_at_point(point_time, point_rpitch):
            sym_id = int(getattr(symbol, '_id', -1) or -1)
            if sym_id in visited_symbols:
                continue
            visited_symbols.add(sym_id)
            self._translate_symbol(symbol, 0.0, delta_rpitch)

        for hp, matched_handle in self._hairpins_at_point(point_time, point_rpitch):
            hp_id = int(getattr(hp, '_id', -1) or -1)
            if hp_id in visited_hairpins:
                continue
            visited_hairpins.add(hp_id)
            far_handle = self._other_handle(matched_handle)
            far_time, far_rpitch = self._handle_point(hp, far_handle)
            self._translate_hairpin(hp, 0.0, delta_rpitch)
            self._propagate_chain_pitch_only(
                far_time,
                far_rpitch,
                delta_rpitch,
                visited_hairpins=visited_hairpins,
                visited_symbols=visited_symbols,
            )

    def _set_symbol_for_handle(self, hp, handle: str, glyph: str) -> None:
        score = self._score()
        if score is None:
            return
        connected = self._find_connected_symbol_for_handle(hp, handle)
        clean = str(glyph or '')
        if connected is None:
            if clean == '':
                return
            score.new_dynamic_symbol(
                time=float(max(0.0, self._hairpin_endpoint_time(hp, handle))),
                x_rpitch=int(getattr(hp, 'x_rpitch', 0) or 0),
                symbol=clean,
            )
            return
        if clean == '':
            self._delete_dynamic_symbol(connected)
            return
        connected.symbol = clean
        connected.time = float(max(0.0, self._hairpin_endpoint_time(hp, handle)))
        connected.x_rpitch = int(getattr(hp, 'x_rpitch', 0) or 0)

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
        self._dragged_hairpin = False

        # Handle editing is available in all three modes.
        hp, hp_type, handle = self._editor.hit_test_hairpin_mm(x_mm, y_mm)
        if hp is not None and handle is not None:
            self._active_hairpin = hp
            self._active_type = hp_type
            self._active_handle = handle
            self._active_symbol = None
            self._created_on_press = False
            return

        # Dynamic symbols are editable in all three modes.
        sym, _sym_type, _ = self._editor.hit_test_dynamic_symbol_mm(x_mm, y_mm)
        if sym is not None:
            self._active_symbol = sym
            self._active_hairpin = None
            self._active_type = None
            self._active_handle = None
            self._created_on_press = False
            return

        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            self._active_symbol = None
            self._active_hairpin = None
            self._active_type = None
            self._active_handle = None
            self._created_on_press = False
            return

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
        if self._active_hairpin is not None and self._active_handle is not None:
            old_joint_time = self._hairpin_endpoint_time(self._active_hairpin, self._active_handle)
            old_joint_rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
            far_handle = self._other_handle(self._active_handle)
            old_far_time = self._hairpin_endpoint_time(self._active_hairpin, far_handle)
            old_far_rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
            connected_symbols = self._symbols_at_point(old_joint_time, old_joint_rpitch)
            connected_hairpins = [
                (hp, handle)
                for hp, handle in self._hairpins_at_point(old_joint_time, old_joint_rpitch)
                if hp is not self._active_hairpin
            ]
            t_snap = self._snap_time(x, y)
            try:
                x_mm, _ = self._cursor_mm(x, y)
                rpitch = self._x_mm_to_rpitch(x_mm)
            except Exception:
                rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
            self._apply_handle_drag(self._active_hairpin, self._active_handle, t_snap, rpitch)
            new_joint_time = self._hairpin_endpoint_time(self._active_hairpin, self._active_handle)
            new_joint_rpitch = int(getattr(self._active_hairpin, 'x_rpitch', 0) or 0)
            delta_time = float(new_joint_time - old_joint_time)
            delta_rpitch = int(new_joint_rpitch - old_joint_rpitch)

            for symbol in connected_symbols:
                self._translate_symbol(symbol, delta_time, delta_rpitch)

            visited_hairpins: set[int] = {int(getattr(self._active_hairpin, '_id', -1) or -1)}
            visited_symbols: set[int] = {int(getattr(sym, '_id', -1) or -1) for sym in connected_symbols}

            for hp, matched_handle in connected_hairpins:
                hp_id = int(getattr(hp, '_id', -1) or -1)
                visited_hairpins.add(hp_id)
                neighbor_far_handle = self._other_handle(matched_handle)
                neighbor_far_time, neighbor_far_rpitch = self._handle_point(hp, neighbor_far_handle)
                self._set_hairpin_handle(hp, matched_handle, new_joint_time, new_joint_rpitch)
                self._propagate_chain_pitch_only(
                    neighbor_far_time,
                    neighbor_far_rpitch,
                    delta_rpitch,
                    visited_hairpins=visited_hairpins,
                    visited_symbols=visited_symbols,
                )

            self._propagate_chain_pitch_only(
                old_far_time,
                old_far_rpitch,
                delta_rpitch,
                visited_hairpins=visited_hairpins,
                visited_symbols=visited_symbols,
            )
            self._dragged_hairpin = True
            self._redraw()
            return
        if self._active_symbol is not None:
            if self._active_symbol is None:
                return
            t_snap = self._snap_time(x, y)
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
        return

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        self._clear_active_interaction()

    def on_left_click(self, x: float, y: float) -> None:
        super().on_left_click(x, y)
        if self._suppress_next_left_interaction or self._dialog_open:
            return
        if self._active_symbol is not None and not self._dragged_symbol:
            glyph, ok = self._select_dynamic_symbol(str(getattr(self._active_symbol, 'symbol', '') or ''))
            if ok:
                self._active_symbol.symbol = str(glyph or '')
                self._redraw()
            return
        if self._mode == self._MODE_DYNAMIC_SYMBOL:
            # In dynamic-symbol mode: click on a handle sets/edits a symbol at that handle endpoint.
            if self._active_hairpin is not None and self._active_handle is not None and not self._dragged_hairpin:
                connected = self._find_connected_symbol_for_handle(self._active_hairpin, self._active_handle)
                current = str(getattr(connected, 'symbol', '') or '') if connected is not None else ''
                glyph, ok = self._select_dynamic_symbol(current)
                if ok:
                    self._set_symbol_for_handle(self._active_hairpin, self._active_handle, glyph)
                    self._redraw()
                return
            if self._editor is None:
                return
            if self._active_symbol is None:
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

        sym, _sym_type, _ = self._editor.hit_test_dynamic_symbol_mm(x_mm, y_mm)
        if sym is not None:
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

