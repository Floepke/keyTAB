from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from editor.editor_defaults import SCALE
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor


class BeamDrawerMixin:
    def draw_beam(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        cache = getattr(self, "_draw_cache", None)
        if not cache:
            return

        if getattr(self, "is_tiny_mode", None) and self.is_tiny_mode():
            self._beam_groups_by_hand = {}
            return

        op: Operator = cache.get("op") or Operator(float(SHORTEST_DURATION))
        score = self.current_score()
        layout = score.layout if score else None
        if layout is None:
            return

        groups_all: dict[str, list[list]] = dict(cache.get("beam_groups_by_hand") or {})
        windows_all: dict[str, list[tuple[float, float]]] = dict(cache.get("beam_windows_by_hand") or {})
        marker_windows_all: dict[str, list[tuple[float, float]]] = dict(cache.get("beam_marker_windows_by_hand") or {})
        if not groups_all:
            self._beam_groups_by_hand = {}
            return
        self._beam_groups_by_hand = groups_all

        stem_metrics_by_id: dict[int, dict] = dict(cache.get("note_stem_metrics_by_id") or {})

        tool_name = getattr(getattr(self, "_tool", None), "TOOL_NAME", "")
        if tool_name == "beam":
            margin = float(self.margin or 0.0)
            stave_w = float(getattr(self, "stave_width", 0.0) or 0.0)
            gutter_w = 1.5
            dx = float(self.semitone_dist or 0.5)
            left_center = margin * 0.5
            right_center = margin + stave_w + (margin * 0.5)
            stroke_color = getattr(self, "accent_color", self.notation_color)
            outer_left = margin
            outer_right = margin + stave_w - (self.semitone_dist * 2)
            dash = (1.0, 1.0)
            for hand_key, marker_windows in marker_windows_all.items():
                if hand_key == "r":
                    x1 = right_center
                    x2 = x1 + dx
                    x_outer = outer_right
                else:
                    x1 = left_center
                    x2 = x1 - dx
                    x_outer = outer_left
                for (w0, w1) in marker_windows:
                    y0 = float(self.time_to_mm(w0))
                    y1 = float(self.time_to_mm(w1))
                    du.add_line(x1, y0, x2, y1, color=stroke_color, width_mm=max(0.15, gutter_w), id=0, tags=["beam_marker", f"beam_marker_{hand_key}"])
                    du.add_line(x1, y0, x_outer, y0, color=stroke_color, width_mm=0.5, dash_pattern=None, id=0, tags=["beam_marker", f"beam_marker_{hand_key}"])
                    du.add_line(x2, y1, x_outer, y1, color=stroke_color, width_mm=0.5, dash_pattern=dash, id=0, tags=["beam_marker", f"beam_marker_{hand_key}"])

        stem_len = float(getattr(layout, "note_stem_length_semitone", 1.0) or 1.0)
        semitone_mm = float(self.semitone_dist or 0.5)
        beam_w = float(getattr(layout, "beam_thickness_mm", 1.0) or 1.0) * SCALE
        stem_w = float(getattr(layout, "note_stem_thickness_mm", 0.5) or 0.5) * SCALE

        right_groups = groups_all.get("r") or []
        right_windows = windows_all.get("r") or []
        for idx, grp in enumerate(right_groups):
            if not grp:
                continue
            t0, t1 = right_windows[idx] if idx < len(right_windows) else (float(min(grp, key=lambda n: float(n.time)).time), float(max(grp, key=lambda n: float(n.time)).time))
            s_min = None
            s_max = None
            for n in grp:
                nt = float(n.time)
                if not (op.ge(nt, float(t0)) and op.lt(nt, float(t1))):
                    continue
                if s_min is None or nt < s_min:
                    s_min = nt
                if s_max is None or nt > s_max:
                    s_max = nt
            if s_min is None or s_max is None or op.eq(float(s_min), float(s_max)):
                continue
            t_first = float(s_min)
            t_last = float(s_max)

            highest = max(grp, key=lambda n: int(getattr(n, "pitch", 0)))
            high_id = int(getattr(highest, "_id", 0) or 0)
            high_metric = stem_metrics_by_id.get(high_id) if high_id > 0 else None
            x1 = float(high_metric.get("x_tip")) if isinstance(high_metric, dict) and "x_tip" in high_metric else float(self.pitch_to_x(int(getattr(highest, "pitch", 0)))) + float(stem_len * semitone_mm)
            x2 = x1 + float(semitone_mm)
            y1 = float(self.time_to_mm(t_first))
            y2 = float(self.time_to_mm(t_last))
            self._draw_beam(du, x1, y1, x2, y2, beam_w, tags=["beam_line_right"])

            for m in grp:
                mt = float(getattr(m, "time", t_first))
                if not (op.ge(mt, float(t0)) and op.lt(mt, float(t1))):
                    continue
                mid = int(getattr(m, "_id", 0) or 0)
                metric = stem_metrics_by_id.get(mid) if mid > 0 else None
                y_note = float(metric.get("y")) if isinstance(metric, dict) and "y" in metric else float(self.time_to_mm(mt))
                x_tip = float(metric.get("x_tip")) if isinstance(metric, dict) and "x_tip" in metric else float(self.pitch_to_x(int(getattr(m, "pitch", 0)))) + float(stem_len)
                if abs(y2 - y1) > 1e-6:
                    t = (y_note - y1) / (y2 - y1)
                    x_on_beam = x1 + t * (x2 - x1)
                else:
                    x_on_beam = x1
                du.add_line(x_tip, y_note, float(x_on_beam), y_note, color=self.notation_color, width_mm=max(0.15, stem_w), id=0, tags=["beam_connect_right"])

        left_groups = groups_all.get("l") or []
        left_windows = windows_all.get("l") or []
        for idx, grp in enumerate(left_groups):
            if not grp:
                continue
            t0, t1 = left_windows[idx] if idx < len(left_windows) else (float(min(grp, key=lambda n: float(n.time)).time), float(max(grp, key=lambda n: float(n.time)).time))
            s_min = None
            s_max = None
            for n in grp:
                nt = float(n.time)
                if not (op.ge(nt, float(t0)) and op.lt(nt, float(t1))):
                    continue
                if s_min is None or nt < s_min:
                    s_min = nt
                if s_max is None or nt > s_max:
                    s_max = nt
            if s_min is None or s_max is None or op.eq(float(s_min), float(s_max)):
                continue
            t_first = float(s_min)
            t_last = float(s_max)

            lowest = min(grp, key=lambda n: int(getattr(n, "pitch", 0)))
            low_id = int(getattr(lowest, "_id", 0) or 0)
            low_metric = stem_metrics_by_id.get(low_id) if low_id > 0 else None
            x1 = float(low_metric.get("x_tip")) if isinstance(low_metric, dict) and "x_tip" in low_metric else float(self.pitch_to_x(int(getattr(lowest, "pitch", 0)))) - float(stem_len * semitone_mm)
            x2 = x1 - float(semitone_mm)
            y1 = float(self.time_to_mm(t_first))
            y2 = float(self.time_to_mm(t_last))
            self._draw_beam(du, x1, y1, x2, y2, beam_w, tags=["beam_line_left"])

            for m in grp:
                mt = float(getattr(m, "time", t_first))
                if not (op.ge(mt, float(t0)) and op.lt(mt, float(t1))):
                    continue
                mid = int(getattr(m, "_id", 0) or 0)
                metric = stem_metrics_by_id.get(mid) if mid > 0 else None
                y_note = float(metric.get("y")) if isinstance(metric, dict) and "y" in metric else float(self.time_to_mm(mt))
                x_tip = float(metric.get("x_tip")) if isinstance(metric, dict) and "x_tip" in metric else float(self.pitch_to_x(int(getattr(m, "pitch", 0)))) - float(stem_len)
                if abs(y2 - y1) > 1e-6:
                    t = (y_note - y1) / (y2 - y1)
                    x_on_beam = x1 + t * (x2 - x1)
                else:
                    x_on_beam = x1
                du.add_line(x_tip, y_note, float(x_on_beam), y_note, color=self.notation_color, width_mm=max(0.15, stem_w), id=0, tags=["beam_connect_left"])

    def _draw_beam(self, du: DrawUtil, x1: float, y1: float, x2: float, y2: float, beam_w: float, tags: list[str]) -> None:
        width = max(0.1, float(beam_w))
        cap_ext = width * 0.5

        x1f = float(x1)
        y1f = float(y1)
        x2f = float(x2)
        y2f = float(y2)

        dx = x2f - x1f
        dy = y2f - y1f
        seg_len = float(math.hypot(dx, dy))
        if seg_len > (2.0 * cap_ext + 1e-6):
            ux = dx / seg_len
            uy = dy / seg_len
            x1f += ux * cap_ext
            y1f += uy * cap_ext
            x2f -= ux * cap_ext
            y2f -= uy * cap_ext

        du.add_line(
            x1f,
            y1f,
            x2f,
            y2f,
            color=self.notation_color,
            width_mm=width,
            line_cap="round",
            id=0,
            tags=tags,
        )
