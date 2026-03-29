from __future__ import annotations
from __future__ import annotations
import math
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SLUR_SEGMENT_COUNT

if TYPE_CHECKING:
    from editor.editor import Editor
    from editor.tool.slur_tool import SlurTool


class SlurDrawerMixin:
    def draw_slur(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        score = getattr(self, 'current_score', lambda: None)()
        if score is None:
            return

        slurs = getattr(score.events, 'slur', []) or []
        if not slurs:
            return

        lay = getattr(score, 'layout', None)
        side_w = float(getattr(lay, 'slur_width_sides_mm', 0.1) or 0.1)
        mid_w = float(getattr(lay, 'slur_width_middle_mm', 1.5) or 1.5)
        n_seg = max(2, int(SLUR_SEGMENT_COUNT))

        is_slur_tool = False
        try:
            from editor.tool.slur_tool import SlurTool
            is_slur_tool = isinstance(getattr(self, '_tool', None), SlurTool)
        except Exception:
            is_slur_tool = False

        def tri_interp(t: float) -> float:
            # Triangle profile peaking at t=0.5; 0 at t=0 and t=1
            return max(0.0, 1.0 - abs(2.0 * t - 1.0))

        def width_at(t: float) -> float:
            return side_w + (mid_w - side_w) * tri_interp(t) / 2

        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        def bezier_point(t: float, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> tuple[float, float]:
            q0x = lerp(p0[0], p1[0], t)
            q0y = lerp(p0[1], p1[1], t)
            q1x = lerp(p1[0], p2[0], t)
            q1y = lerp(p1[1], p2[1], t)
            q2x = lerp(p2[0], p3[0], t)
            q2y = lerp(p2[1], p3[1], t)

            r0x = lerp(q0x, q1x, t)
            r0y = lerp(q0y, q1y, t)
            r1x = lerp(q1x, q2x, t)
            r1y = lerp(q1y, q2y, t)

            return lerp(r0x, r1x, t), lerp(r0y, r1y, t)

        page_w, _ = du.current_page_size_mm()

        def clamp_x(val: float) -> float:
            if page_w <= 0:
                return val
            return max(0.0, min(float(val), float(page_w)))

        for sl in slurs:
            try:
                x1_raw = float(self.relative_c4pitch_to_x(int(getattr(sl, 'x1_rpitch', 0) or 0)))
                x2_raw = float(self.relative_c4pitch_to_x(int(getattr(sl, 'x2_rpitch', 0) or 0)))
                x3_raw = float(self.relative_c4pitch_to_x(int(getattr(sl, 'x3_rpitch', 0) or 0)))
                x4_raw = float(self.relative_c4pitch_to_x(int(getattr(sl, 'x4_rpitch', 0) or 0)))
                x1 = clamp_x(x1_raw)
                x2 = clamp_x(x2_raw)
                x3 = clamp_x(x3_raw)
                x4 = clamp_x(x4_raw)
                y1 = float(self.time_to_mm(float(getattr(sl, 'y1_time', 0.0) or 0.0)))
                y2 = float(self.time_to_mm(float(getattr(sl, 'y2_time', 0.0) or 0.0)))
                y3 = float(self.time_to_mm(float(getattr(sl, 'y3_time', 0.0) or 0.0)))
                y4 = float(self.time_to_mm(float(getattr(sl, 'y4_time', 0.0) or 0.0)))
            except Exception:
                continue

            pts: list[tuple[float, float]] = []
            p0 = (x1, y1)
            p1 = (x2, y2)
            p2 = (x3, y3)
            p3 = (x4, y4)
            for i in range(n_seg):
                t = i / float(n_seg - 1)
                bx, by = bezier_point(t, p0, p1, p2, p3)
                pts.append((bx, by))

            if len(pts) >= 2:
                left_edge: list[tuple[float, float]] = []
                right_edge: list[tuple[float, float]] = []
                last_nx, last_ny = 0.0, 1.0

                for i, (cx, cy) in enumerate(pts):
                    t_cur = i / float(n_seg - 1)
                    w = max(0.0, float(width_at(t_cur)))
                    half_w = 0.5 * w

                    if i == 0:
                        px, py = pts[i]
                        nxp, nyp = pts[i + 1]
                    elif i == len(pts) - 1:
                        px, py = pts[i - 1]
                        nxp, nyp = pts[i]
                    else:
                        px, py = pts[i - 1]
                        nxp, nyp = pts[i + 1]

                    dx = float(nxp) - float(px)
                    dy = float(nyp) - float(py)
                    dlen = math.hypot(dx, dy)
                    if dlen <= 1e-9:
                        nx, ny = last_nx, last_ny
                    else:
                        nx = -dy / dlen
                        ny = dx / dlen
                        last_nx, last_ny = nx, ny

                    left_edge.append((float(cx) + nx * half_w, float(cy) + ny * half_w))
                    right_edge.append((float(cx) - nx * half_w, float(cy) - ny * half_w))

                slur_poly = left_edge + list(reversed(right_edge))
                if len(slur_poly) >= 3:
                    du.add_polygon(
                        slur_poly,
                        stroke_color=None,
                        fill_color=self.notation_color,
                        id=int(getattr(sl, '_id', 0) or 0),
                        tags=["slur"],
                    )

            if is_slur_tool:
                # Match count-line handle geometry: side = 1.4 * max(2.0, semitone_dist)
                handle_size = max(2.0, float(self.semitone_dist or 2.5)) * 1.4
                handle_half = handle_size * 0.5
                handles = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                for idx, (hx, hy) in enumerate(handles, start=1):
                    is_anchor = idx in (1, 4)
                    fill_col = (.5, 0.0, 0.0, 1.0) if is_anchor else self.accent_color
                    du.add_rectangle(
                        hx - handle_half,
                        hy - handle_half,
                        hx + handle_half,
                        hy + handle_half,
                        stroke_color=None,
                        stroke_width_mm=0.0,
                        fill_color=fill_col,
                        id=int(getattr(sl, '_id', 0) or 0),
                        tags=["slur-handle"],
                    )
