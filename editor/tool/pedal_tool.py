from __future__ import annotations

from typing import Literal, Optional

from PySide6 import QtCore

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from file_model.events.pedal import Pedal


PedalSymbol = Literal['down', 'up', 'toe', 'heel']


class PedalTool(BaseTool):
    TOOL_NAME = 'pedal'

    _SYMBOLS: tuple[PedalSymbol, ...] = ('up', 'down', 'toe', 'heel')
    _HIT_THRESHOLD_MM = 4.0

    def __init__(self) -> None:
        super().__init__()
        self._symbol: PedalSymbol = 'down'
        self._active_pedal: Optional[Pedal] = None
        self._linked_pedals: list[Pedal] = []
        self._pressed_existing = False

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': f'pedal_{symbol}',
                'text': symbol.capitalize(),
                'tooltip': QtCore.QCoreApplication.translate(
                    'PedalTool',
                    f'Insert pedal {symbol} symbol.',
                ),
                'active': self._symbol == symbol,
            }
            for symbol in self._SYMBOLS
        ]

    def on_toolbar_button(self, name: str) -> None:
        symbol = name.removeprefix('pedal_')
        if symbol in self._SYMBOLS:
            self._symbol = symbol  # type: ignore[assignment]

    def _score(self) -> Optional[SCORE]:
        if self._editor is None:
            return None
        return self._editor.current_score()

    def _cursor_position(self, x_px: float, y_px: float) -> tuple[float, int]:
        if self._editor is None:
            return (0.0, 0)
        time = float(self._editor.widget_px_to_time(x_px, y_px))
        time = float(self._editor.snap_time(time))
        time = float(self._editor.clamp_time_to_visible_range(time))
        x_mm, _ = self._editor.widget_px_to_page_mm(float(x_px), float(y_px))
        return (time, self.x_mm_to_rpitch_clamped(float(x_mm)))

    def _find_pedal(self, x_px: float, y_px: float) -> Optional[Pedal]:
        score = self._score()
        if score is None or self._editor is None:
            return None
        events = self._editor.current_events(score)
        if events is None:
            return None
        x_mm, y_mm = self._editor.widget_px_to_page_mm(float(x_px), float(y_px))
        hit = self._editor.hit_test_hit_rect(x_mm, y_mm, 'pedal')
        if hit is not None:
            pedal_id = int(hit.get('_id', -1) or -1)
            for pedal in getattr(events, 'pedal', []):
                if int(getattr(pedal, '_id', -2) or -2) == pedal_id:
                    return pedal

        nearest: Optional[Pedal] = None
        nearest_distance = self._HIT_THRESHOLD_MM
        for pedal in getattr(events, 'pedal', []):
            pedal_x = float(self._editor.relative_c4pitch_to_x(int(pedal.rpitch)))
            pedal_y = float(self._editor.time_to_mm(float(pedal.time)))
            distance = ((pedal_x - x_mm) ** 2 + (pedal_y - y_mm) ** 2) ** 0.5
            if distance <= nearest_distance:
                nearest = pedal
                nearest_distance = distance
        return nearest

    def _redraw(self) -> None:
        if self._editor is None:
            return
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def _coincident_opposite_pedals(self, pedal: Pedal) -> list[Pedal]:
        symbol = str(getattr(pedal, 'symbol', '') or '')
        if symbol not in ('up', 'down'):
            return []
        score = self._score()
        if score is None or self._editor is None:
            return []
        events = self._editor.current_events(score)
        if events is None:
            return []
        opposite_symbol = 'down' if symbol == 'up' else 'up'
        pedal_time = float(getattr(pedal, 'time', 0.0) or 0.0)
        pedal_rpitch = int(getattr(pedal, 'rpitch', 0) or 0)
        return [
            candidate
            for candidate in getattr(events, 'pedal', [])
            if candidate is not pedal
            and str(getattr(candidate, 'symbol', '') or '') == opposite_symbol
            and int(getattr(candidate, 'rpitch', 0) or 0) == pedal_rpitch
            and abs(float(getattr(candidate, 'time', 0.0) or 0.0) - pedal_time) <= 1e-9
        ]

    def _commit(self, label: str) -> None:
        if self._editor is None:
            return
        self._editor._snapshot_if_changed(coalesce=False, label=label)
        self._redraw()

    def _move_active_pedal(self, x: float, y: float) -> None:
        if self._active_pedal is None:
            return
        time, rpitch = self._cursor_position(x, y)
        for pedal in [self._active_pedal, *self._linked_pedals]:
            pedal.time = time
            pedal.rpitch = rpitch
        self._linked_pedals = self._coincident_opposite_pedals(self._active_pedal)
        self._redraw()

    def on_left_press(self, x: float, y: float) -> None:
        self._active_pedal = self._find_pedal(x, y)
        self._linked_pedals = (
            self._coincident_opposite_pedals(self._active_pedal)
            if self._active_pedal is not None
            else []
        )
        self._pressed_existing = self._active_pedal is not None

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        self._move_active_pedal(x, y)

    def on_left_drag_end(self, x: float, y: float) -> None:
        if self._active_pedal is not None:
            self._move_active_pedal(x, y)
            self._commit('pedal_move')
        self._active_pedal = None
        self._linked_pedals = []
        self._pressed_existing = False

    def on_left_click(self, x: float, y: float) -> None:
        if self._pressed_existing:
            self._active_pedal = None
            self._pressed_existing = False
            return
        score = self._score()
        if score is None:
            return
        time, rpitch = self._cursor_position(x, y)
        score.new_pedal(time=time, rpitch=rpitch, symbol=self._symbol)
        self._commit('pedal_create')

    def on_left_unpress(self, x: float, y: float) -> None:
        self._active_pedal = None
        self._linked_pedals = []

    def on_right_click(self, x: float, y: float) -> None:
        score = self._score()
        if score is None or self._editor is None:
            return
        pedal = self._find_pedal(x, y)
        if pedal is None:
            return
        events = self._editor.current_events(score)
        if events is None:
            return
        events.pedal.remove(pedal)
        self._commit('pedal_delete')
