from __future__ import annotations

from PySide6 import QtCore
from editor.tool.base_tool import BaseTool
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator


class PedalTool(BaseTool):
    TOOL_NAME = 'pedal'

    _SYMBOL_DOWN_KEYTAB = 'down_keytab'
    _SYMBOL_UP_KEYTAB = 'up_keytab'
    _SYMBOL_DOWN_KLAVARSKRIBO = 'down_klavarskribo'
    _SYMBOL_UP_KLAVARSKRIBO = 'up_klavarskribo'
    _SYMBOL_TOE = 'toe'
    _SYMBOL_HEEL = 'heel'


    def __init__(self) -> None:
        super().__init__()
        self._symbol: str = self._SYMBOL_DOWN_KEYTAB
        self._active_pedal = None
        self._active_bound_partner = None

    @staticmethod
    def _ev_id(ev) -> int:
        return int(getattr(ev, '_id', 0) or 0)

    @staticmethod
    def _ev_symbol(ev) -> str:
        return str(getattr(ev, 'symbol', '') or '').strip().lower()

    @staticmethod
    def _opposite_symbol(symbol: str) -> str:
        if symbol == 'down_keytab':
            return 'up_keytab'
        if symbol == 'up_keytab':
            return 'down_keytab'
        if symbol == 'down_klavarskribo':
            return 'up_klavarskribo'
        if symbol == 'up_klavarskribo':
            return 'down_klavarskribo'
        return ''

    def _find_matching_opposite(self, score, active_ev, rpitch: int, time_val: float):
        if score is None or active_ev is None:
            return None
        active_symbol = self._ev_symbol(active_ev)
        opposite = self._opposite_symbol(active_symbol)
        if not opposite:
            return None
        op = Operator(float(SHORTEST_DURATION))

        active_id = self._ev_id(active_ev)
        for ev in list(getattr(score.events, 'pedal', []) or []):
            if self._ev_id(ev) == active_id:
                continue
            if self._ev_symbol(ev) != opposite:
                continue
            if int(getattr(ev, 'rpitch', 0) or 0) != int(rpitch):
                continue
            if not op.eq(float(getattr(ev, 'time', 0.0) or 0.0), float(time_val)):
                continue
            return ev
        return None

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                'name': self._SYMBOL_UP_KEYTAB,
                'icon': 'up',
                'text': 'U-K',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal up symbol (keyTAB)'),
                'active': self._symbol == self._SYMBOL_UP_KEYTAB,
            },
            {
                'name': self._SYMBOL_DOWN_KEYTAB,
                'icon': 'down',
                'text': 'D-K',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal down symbol (keyTAB)'),
                'active': self._symbol == self._SYMBOL_DOWN_KEYTAB,
            },
            {
                'name': self._SYMBOL_UP_KLAVARSKRIBO,
                'icon': 'up',
                'text': 'U-L',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal up symbol (Klavarskribo)'),
                'active': self._symbol == self._SYMBOL_UP_KLAVARSKRIBO,
            },
            {
                'name': self._SYMBOL_DOWN_KLAVARSKRIBO,
                'icon': 'down',
                'text': 'D-L',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal down symbol (Klavarskribo)'),
                'active': self._symbol == self._SYMBOL_DOWN_KLAVARSKRIBO,
            },
            {
                'name': self._SYMBOL_TOE,
                'icon': 'toe',
                'text': 'T',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal toe symbol'),
                'active': self._symbol == self._SYMBOL_TOE,
            },
            {
                'name': self._SYMBOL_HEEL,
                'icon': 'heel',
                'text': 'H',
                'tooltip': QtCore.QCoreApplication.translate('PedalTool', 'Insert pedal heel symbol'),
                'active': self._symbol == self._SYMBOL_HEEL,
            },
        ]

    def _x_mm_to_rpitch(self, x_mm: float) -> int:
        return self.x_mm_to_rpitch_clamped(float(x_mm))

    def _cursor_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        if self._editor is None:
            return (0.0, 0.0)
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _hit_pedal(self, x_px: float, y_px: float):
        if self._editor is None:
            return None
        x_mm, y_mm = self._cursor_mm(x_px, y_px)
        return self._editor.hit_test_hit_rect(float(x_mm), float(y_mm), 'pedal')

    def _request_light_repaint(self) -> None:
        if self._editor is None:
            return
        w = getattr(self._editor, 'widget', None)
        if w is not None and hasattr(w, 'update'):
            try:
                w.update()
                return
            except Exception:
                pass
        try:
            self._editor.draw_frame()
        except Exception:
            pass

    def on_toolbar_button(self, name: str) -> None:
        if name in (
            self._SYMBOL_DOWN_KEYTAB,
            self._SYMBOL_UP_KEYTAB,
            self._SYMBOL_DOWN_KLAVARSKRIBO,
            self._SYMBOL_UP_KLAVARSKRIBO,
            self._SYMBOL_TOE,
            self._SYMBOL_HEEL,
        ):
            self._symbol = str(name)

    def on_left_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return

        # Add only when clicking in empty space (outside existing pedal symbol).
        if self._hit_pedal(x, y) is not None:
            return

        x_mm, _y_mm = self._cursor_mm(x, y)
        t_click = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_click))
        rpitch = int(self._x_mm_to_rpitch(x_mm))

        score.new_pedal(
            time=float(t_snap),
            rpitch=int(rpitch),
            symbol=str(self._symbol),
        )
        self._editor._snapshot_if_changed(coalesce=False, label='pedal_symbol_create')
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_press(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return
        hit = self._hit_pedal(x, y)
        if hit is None:
            self._active_pedal = None
            self._active_bound_partner = None
            return
        hit_id = int(hit.get('_id', 0) or 0)
        self._active_pedal = None
        self._active_bound_partner = None
        for ev in list(getattr(score.events, 'pedal', []) or []):
            if int(getattr(ev, '_id', 0) or 0) == hit_id:
                self._active_pedal = ev
                # Start locked immediately when an opposite symbol already shares position.
                rp = int(getattr(ev, 'rpitch', 0) or 0)
                tm = float(getattr(ev, 'time', 0.0) or 0.0)
                self._active_bound_partner = self._find_matching_opposite(score, ev, rp, tm)
                break

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        if self._editor is None or self._active_pedal is None:
            return
        score = self._editor.current_score()
        if score is None:
            return

        x_mm, _y_mm = self._cursor_mm(x, y)
        rpitch = int(self._x_mm_to_rpitch(x_mm))
        t_raw = float(self._editor.widget_px_to_time(x, y))
        t_snap = float(self._editor.snap_time(t_raw))

        try:
            self._active_pedal.rpitch = int(self.clamp_rpitch(rpitch))
            self._active_pedal.time = float(t_snap)
        except Exception:
            return

        # Bind by position for up/down: once matched, keep partner aligned while dragging.
        active_symbol = self._ev_symbol(self._active_pedal)
        if active_symbol in ('up_keytab', 'down_keytab', 'up_klavarskribo', 'down_klavarskribo'):
            partner = self._active_bound_partner
            if partner is None:
                partner = self._find_matching_opposite(score, self._active_pedal, int(rpitch), float(t_snap))
                if partner is not None:
                    self._active_bound_partner = partner

            if partner is not None:
                try:
                    partner.rpitch = int(self.clamp_rpitch(rpitch))
                    partner.time = float(t_snap)
                except Exception:
                    self._active_bound_partner = None

        self._request_light_repaint()

    def on_left_drag_end(self, x: float, y: float) -> None:
        if self._editor is None:
            self._active_pedal = None
            self._active_bound_partner = None
            return
        if self._active_pedal is not None:
            self._editor._snapshot_if_changed(coalesce=True, label='pedal_symbol_move')
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            else:
                self._editor.draw_frame()
        self._active_pedal = None
        self._active_bound_partner = None

    def on_left_unpress(self, x: float, y: float) -> None:
        # Keep state tidy for non-drag presses.
        self._active_pedal = None
        self._active_bound_partner = None

    def on_right_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return

        hit = self._hit_pedal(x, y)
        if hit is None:
            return

        hit_id = int(hit.get('_id', 0) or 0)
        removed = False
        out = []
        for ev in list(getattr(score.events, 'pedal', []) or []):
            if int(getattr(ev, '_id', 0) or 0) == hit_id:
                removed = True
                continue
            out.append(ev)
        if not removed:
            return

        score.events.pedal = out
        self._editor._snapshot_if_changed(coalesce=False, label='pedal_symbol_delete')
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_left_double_click(self, x: float, y: float) -> None:
        super().on_left_double_click(x, y)
        if self._editor is None:
            return
        score = self._editor.current_score()
        if score is None:
            return

        hit = self._hit_pedal(x, y)
        if hit is None:
            return

        hit_id = int(hit.get('_id', 0) or 0)
        target = None
        for ev in list(getattr(score.events, 'pedal', []) or []):
            if int(getattr(ev, '_id', 0) or 0) == hit_id:
                target = ev
                break
        if target is None:
            return

        current = bool(getattr(target, 'invisible', False))
        try:
            target.invisible = not current
        except Exception:
            return

        self._editor._snapshot_if_changed(coalesce=False, label='pedal_symbol_toggle_visibility')
        if hasattr(self._editor, 'force_redraw_from_model'):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()
