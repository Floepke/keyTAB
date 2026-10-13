from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, cast

from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class LineDrawerMixin:
    def draw_line(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, "is_tiny_mode", None) and self.is_tiny_mode():
            return
        score = self.current_score()
        if score is None:
            return
        events = self.current_events(score)
        lines = list(getattr(events, "line", []) or [])
        if not lines:
            return

        def color(value: object) -> tuple[float, float, float, float]:
            if not isinstance(value, str) or value.strip().lower() == "auto":
                return tuple(self.notation_color)
            match = re.fullmatch(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8})", value.strip())
            if match is None:
                return tuple(self.notation_color)
            raw = match.group(1)
            components = tuple(int(raw[index:index + 2], 16) / 255.0 for index in range(0, 6, 2))
            alpha = int(raw[6:8], 16) / 255.0 if len(raw) == 8 else 1.0
            return (*components, alpha)

        active = str(getattr(getattr(self, "_tool", None), "TOOL_NAME", "")) == "line"
        line_scale = max(0.01, float(getattr(getattr(score, "layout", None), "scale", 1.0) or 1.0))
        for line in lines:
            try:
                time1 = float(getattr(line, "time1", 0.0) or 0.0)
                time2 = float(getattr(line, "time2", 0.0) or 0.0)
                x1 = float(self.relative_c4pitch_to_x(int(getattr(line, "rpitch1", 0) or 0)))
                x2 = float(self.relative_c4pitch_to_x(int(getattr(line, "rpitch2", 0) or 0)))
                y1 = float(self.time_to_mm(time1))
                y2 = float(self.time_to_mm(time2))
            except Exception:
                continue
            line.time = min(time1, time2)
            thickness = max(0.05, float(getattr(line, "thickness_mm", 1.0) or 1.0) * line_scale)
            try:
                dash = [max(0.01, float(value)) * line_scale for value in getattr(line, "dash_pattern_mm", []) or []]
            except (TypeError, ValueError):
                dash = []
            dash = dash if any(dash) else None
            offset = float(getattr(line, "dash_offset_mm", 0.0) or 0.0) * line_scale
            draw_color = color(getattr(line, "color", "auto"))
            points = [(x1, y1), (x2, y2)]

            zigzag_type = getattr(line, "zigzag_type", None)
            if zigzag_type in ("triangle", "sine"):
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                freq_time = max(1e-6, float(getattr(line, "zigzag_freq", 64.0) or 64.0))
                wavelength = abs(float(self.time_to_mm(freq_time)) - float(self.time_to_mm(0.0)))
                if length > 1e-6 and wavelength > 1e-6:
                    cycles = length / wavelength
                    steps = max(8, min(512, int(math.ceil(cycles * 16.0))))
                    nx, ny = -dy / length, dx / length
                    amplitude = abs(float(getattr(line, "zigzag_amp_semitone", 2.0) or 0.0)) * float(self.semitone_dist or 0.0)
                    points = []
                    for index in range(steps + 1):
                        progress = index / steps
                        phase = 2.0 * math.pi * cycles * progress
                        wave = math.sin(phase) if zigzag_type == "sine" else (2.0 / math.pi) * math.asin(math.sin(phase))
                        points.append((x1 + dx * progress + nx * amplitude * wave, y1 + dy * progress + ny * amplitude * wave))

            line_id = int(getattr(line, "_id", 0) or 0)
            if len(points) == 2:
                du.add_line(*points[0], *points[1], color=draw_color, width_mm=thickness, dash_pattern=dash, dash_offset_mm=offset, id=line_id, tags=["line"])
            else:
                du.add_polyline(points, stroke_color=draw_color, stroke_width_mm=thickness, dash_pattern=dash, dash_offset_mm=offset, id=line_id, tags=["line"])

            if getattr(line, "arrow", None) in ("left", "right", "both"):
                arrow_form = list(getattr(line, "arrow_form", [0.5, 0.5]) or [])
                arrow_length = arrow_form[0] if arrow_form else 0.5
                arrow_width = arrow_form[1] if len(arrow_form) > 1 else 0.5
                arrow_length = max(0.1, float(arrow_length) * line_scale)
                arrow_width = max(0.1, float(arrow_width) * line_scale)
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length > 1e-6:
                    ux, uy = dx / length, dy / length
                    for endpoint, direction, name in (
                        ((x1, y1), (-ux, -uy), "left"),
                        ((x2, y2), (ux, uy), "right"),
                    ):
                        if getattr(line, "arrow", None) not in (name, "both"):
                            continue
                        perpendicular = (-direction[1] * arrow_width * 0.5, direction[0] * arrow_width * 0.5)
                        tip = (
                            endpoint[0] + direction[0] * arrow_length,
                            endpoint[1] + direction[1] * arrow_length,
                        )
                        du.add_polygon([tip, (endpoint[0] + perpendicular[0], endpoint[1] + perpendicular[1]), (endpoint[0] - perpendicular[0], endpoint[1] - perpendicular[1])], stroke_color=None, fill_color=draw_color, id=line_id, tags=["line"])

            if active:
                handle = max(2.0, float(self.semitone_dist or 2.5)) * 0.7
                for px, py in ((x1, y1), (x2, y2)):
                    du.add_rectangle(px - handle, py - handle, px + handle, py + handle, stroke_color=None, fill_color=(0.5, 0.0, 0.0, 1.0), id=line_id, tags=["line"])
