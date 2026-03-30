from __future__ import annotations

from typing import Optional

from editor.tool.base_tool import BaseTool
from file_model.base_grid import resolve_grid_layer_offsets
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


class BarlineTool(BaseTool):
    TOOL_NAME = "barline"

    _MODE_START = "start_repeat"
    _MODE_END = "end_repeat"
    _MODE_DOUBLE = "double_bar"

    def __init__(self) -> None:
        super().__init__()
        self._mode: str = self._MODE_START

    def toolbar_spec(self) -> list[dict]:
        return [
            {
                "name": self._MODE_START,
                "icon": "start_repeat",
                "tooltip": "Insert start repeat symbol",
                "active": self._mode == self._MODE_START,
            },
            {
                "name": self._MODE_END,
                "icon": "end_repeat",
                "tooltip": "Insert end repeat symbol",
                "active": self._mode == self._MODE_END,
            },
            {
                "name": self._MODE_DOUBLE,
                "icon": "double_bar",
                "text": "d",
                "tooltip": "Insert double barline symbol",
                "active": self._mode == self._MODE_DOUBLE,
            },
        ]

    def on_toolbar_button(self, name: str) -> None:
        if name in (self._MODE_START, self._MODE_END, self._MODE_DOUBLE):
            self._mode = str(name)

    def _barline_positions(self) -> list[float]:
        if self._editor is None:
            return []
        bars = list(self._editor._get_barline_positions() or [])
        if bars:
            score = self._editor.current_score()
            total = 0.0
            for bg in list(getattr(score, "base_grid", []) or []):
                numer = int(getattr(bg, "numerator", 4) or 4)
                denom = int(getattr(bg, "denominator", 4) or 4)
                measures = int(getattr(bg, "measure_amount", 1) or 1)
                beat_grouping = list(getattr(bg, "beat_grouping", []) or [])
                bar_offsets, _ = resolve_grid_layer_offsets(beat_grouping, numer, denom)
                measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                for _ in range(max(0, measures)):
                    for off in bar_offsets:
                        bars.append(float(total + float(off)))
                    total += measure_len
            bars.append(float(total))
        if not bars:
            return []
        return sorted(list(dict.fromkeys(round(float(v), 6) for v in bars)))

    def _nearest_barline_time(self, t: float) -> Optional[float]:
        bars = self._barline_positions()
        if not bars:
            return None
        best = bars[0]
        best_dt = abs(float(t) - float(best))
        for b in bars[1:]:
            dt = abs(float(t) - float(b))
            if dt < best_dt:
                best = b
                best_dt = dt
        return float(best)

    def _event_list(self, score):
        if self._mode == self._MODE_START:
            return list(getattr(score.events, "start_repeat", []) or [])
        if self._mode == self._MODE_END:
            return list(getattr(score.events, "end_repeat", []) or [])
        return list(getattr(score.events, "double_bar", []) or [])

    def _set_event_list(self, score, lst: list) -> None:
        if self._mode == self._MODE_START:
            score.events.start_repeat = lst
        elif self._mode == self._MODE_END:
            score.events.end_repeat = lst
        else:
            score.events.double_bar = lst

    def _create_event(self, score, t: float) -> None:
        if self._mode == self._MODE_START:
            score.new_start_repeat(time=float(t))
        elif self._mode == self._MODE_END:
            score.new_end_repeat(time=float(t))
        else:
            score.new_double_bar(time=float(t))

    def on_left_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        t_click = float(self._editor.y_to_time(y))
        t_bar = self._nearest_barline_time(t_click)
        if t_bar is None:
            return
        op = Operator(SHORTEST_DURATION)
        for ev in self._event_list(score):
            if op.eq(float(getattr(ev, "time", 0.0) or 0.0), float(t_bar)):
                return
        self._create_event(score, float(t_bar))
        self._editor._snapshot_if_changed(coalesce=False, label="barline_symbol_create")
        if hasattr(self._editor, "force_redraw_from_model"):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_right_click(self, x: float, y: float) -> None:
        if self._editor is None:
            return
        score = self._editor.current_score()
        t_click = float(self._editor.y_to_time(y))
        t_bar = self._nearest_barline_time(t_click)
        if t_bar is None:
            return
        op = Operator(SHORTEST_DURATION)
        removed = False

        # Right-click is global delete for any barline symbol type at this position.
        def _filter_events(events: list) -> list:
            nonlocal removed
            out: list = []
            for ev in list(events or []):
                ev_t = float(getattr(ev, "time", 0.0) or 0.0)
                if op.eq(ev_t, float(t_bar)):
                    removed = True
                    continue
                out.append(ev)
            return out

        score.events.start_repeat = _filter_events(getattr(score.events, "start_repeat", []) or [])
        score.events.end_repeat = _filter_events(getattr(score.events, "end_repeat", []) or [])
        score.events.double_bar = _filter_events(getattr(score.events, "double_bar", []) or [])

        if not removed:
            return

        # ctlz snapshot with coalescing to merge multiple deletes in the same area into one undo step
        self._editor._snapshot_if_changed(coalesce=False, label="barline_symbol_delete")
        if hasattr(self._editor, "force_redraw_from_model"):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def on_mouse_move(self, x: float, y: float) -> None:
        super().on_mouse_move(x, y)
